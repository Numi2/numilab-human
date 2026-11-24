"""Separate stored source overlaps from those introduced by the current muscle registration."""
import itertools,json,mmap,runpy,struct
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_material_interface import solid
from compile_connected_joint import digest


def parts(path,sid):
    raw=path.read_bytes()
    magic,abi,nr,nb,nv,ni,_=struct.unpack_from('<8s6I',raw)
    if magic.rstrip(bytes([0]))!=b'NHTISS4' or abi!=5:raise ValueError('Unknown tissue ABI')
    rows=np.frombuffer(raw,'<u4',nr*8,64).reshape(-1,8)
    fb,count,first,n,fi,nfaces,_,_=map(int,rows[rows[:,6]==sid][0])
    binding=64+nr*32;vertex=binding+36*nb
    rawv=np.frombuffer(raw,'<f4',14*nv,vertex).reshape(-1,14)[first:first+n]
    slots=rawv[:,6:10].copy().view('<u4');vw=rawv[:,10:14]
    weights=np.column_stack([np.where(slots==i,vw,0).sum(1) for i in range(count)])
    if np.max(abs(weights.sum(1)-1))>2e-6:raise ValueError('Invalid source weights')
    tri=np.frombuffer(raw,'<u4',nfaces,vertex+nv*56+4*fi).reshape(-1,3).astype(int)-first
    donors=[];aff=[]
    for k in range(count):
        b,*params=struct.unpack_from('<I8f',raw,binding+36*(fb+k))
        donors.append(b)
        aff.append((Rotation.from_quat(params[3:7]).as_matrix()*params[7],np.array(params[:3])))
    return rawv[:,:3].astype(float),weights,donors,tri,aff


def run(case,registry,neighbors,out):
    capture=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    receipt=json.loads(capture.read_bytes());cfg=json.loads((case/'launch.json').read_bytes())
    a=cfg['argv'];source=Path(a[a.index('--soft-tissue-payload')+1])
    evidence=json.loads(neighbors.read_bytes());assert evidence['capture']==digest(capture)
    selected=sorted({83}|{r['bank'] for r in evidence['surrounding'] if r['kind']==51005 and max(r.get('before',0.),r.get('after',0.))>.0001})
    initial={r['body_index']:dict(position_m=r['current']['position_world_m'],quaternion_xyzw=r['current']['orientation_world_xyzw'])
             for r in json.loads((case/'native-initial-bodies.json').read_bytes())['all_bodies']}
    accepted={r['body_index']:r for r in receipt['accepted_body_poses']}
    neutral={r['body_index']:dict(position_m=r['rest']['position_world_m'],quaternion_xyzw=r['rest']['orientation_world_xyzw'])
             for r in json.loads((case/'native-initial-bodies.json').read_bytes())['all_bodies']}
    stages=['stored','skeletal_reference','supported','accepted']
    shapes={k:{} for k in stages};replay={}
    with Path(receipt['accepted_pack_path']).open('rb') as s,mmap.mmap(s.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for key,_,v,t in runpy.run_path(str(registry))['primitives'](buf,accepted):
            if key[0]==51005 and key[1] in selected:replay[key[1]]=(v.copy(),t.copy())
    err={};cache={}
    for sid in selected:
        v,w,b,f,aff=parts(source,sid)
        vs=[v]
        for target in [neutral,initial,accepted]:
            x=np.zeros_like(v)
            for i,owner in enumerate(b):
                R=Rotation.from_quat(target[owner]['quaternion_xyzw']).as_matrix()
                x+=w[:,i,None]*(v@(R@aff[i][0]).T+target[owner]['position_m']+R@aff[i][1])
            vs.append(x)
        used=np.unique(f);fv=np.searchsorted(used,f);verts,faces=replay[sid]
        if not np.array_equal(fv,faces):raise ValueError(f'Changed face identity for {sid}')
        err[sid]=float(np.linalg.norm(vs[-1][used]-verts,axis=1).max())
        if not np.isfinite(err[sid]):raise ValueError('Nonfinite reconstruction')
        cache[sid]=dict(ref=v[used],f=fv,supported=vs[2][used],accepted=vs[3][used])
        for k,x in zip(stages,vs):
            native=verts if k=='accepted' else x[used]
            shape,_,meta=solid(native*1000,fv,str(sid))
            if str(shape.status())!='Error.NoError' or meta['mesh']['nonmanifold_or_open']:raise ValueError((k,sid,meta))
            shapes[k][sid]=shape
    rows=[]
    for i,j in itertools.combinations(selected,2):
        row=dict(pair=[i,j])
        for k in stages:
            val=shapes[k][i]^shapes[k][j];row[k]=float(abs(val.volume()))
        rows.append(row)
    result=dict(payload=digest(source),accepted_receipt=digest(capture),neighbor_source=digest(neighbors),native_replay_max_m=err,
                units='mm3',pairwise=rows,volume_by_stage={k:{i:s.volume() for i,s in sh.items()}for k,sh in shapes.items()})
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in rows:
        if 83 in r['pair']:print(r,flush=True)
    return cache

if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))

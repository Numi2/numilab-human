"""Compare registered child positions against their exact source-parent stencil over the complete tissue inventory."""
from pathlib import Path
import hashlib,struct,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest

def records(path,poses,requested):
    raw=path.read_bytes();magic,abi,n,b,v,i,_,_=struct.unpack_from('<8s6I32s',raw)
    if magic!=b'NHTISS4\0' or abi!=5 or len(raw)!=64+32*n+36*b+56*v+4*i:raise ValueError('Invalid tissue source')
    rows=np.frombuffer(raw,'<u4',n*8,64).reshape(-1,8).astype(int)
    bo=64+32*n;vo=bo+36*b;body=np.ndarray((b,),'<u4',buffer=raw,offset=bo,strides=(36,));tf=np.ndarray((b,8),'<f4',buffer=raw,offset=bo+4,strides=(36,4)).astype(float)
    bind=Rotation.from_quat(tf[:,3:7]).as_matrix()*tf[:,7,None,None];xyz=np.ndarray((v,3),'<f4',buffer=raw,offset=vo,strides=(56,4));idx=np.ndarray((v,4),'<u4',buffer=raw,offset=vo+24,strides=(56,4));w=np.ndarray((v,4),'<f4',buffer=raw,offset=vo+40,strides=(56,4))
    R=Rotation.from_quat([a['orientation_world_xyzw'] for a in poses]).as_matrix();T=np.array([a['position_world_m'] for a in poses]);A=R[body]@bind;B=T[body]+np.einsum('nij,nj->ni',R[body],tf[:,:3])
    result={}
    for fb,nb,first,count,_,_,sid,_ in rows:
        if sid not in requested:continue
        sl=slice(first,first+count);which=idx[sl].astype(int);weight=w[sl].astype(float);which[weight==0]=0
        if np.any(which<0) or np.any(which>=nb):raise ValueError('Joint binding out of range')
        ids=which+fb;M=np.einsum('nk,nkij->nij',weight,A[ids]);offset=np.einsum('nk,nki->ni',weight,B[ids]);world=np.einsum('nij,nj->ni',M,xyz[sl])+offset
        result[int(sid)]=(world,M,first,vo)
    return result

def run(scene,out):
    cfg=json.loads((scene/'launch.json').read_bytes());a=cfg['argv'];r=json.loads(Path(a[a.index('--resting-anatomy-receipt')+1]).read_bytes())['passive_tissue_parent_bindings']
    payload=Path(a[a.index('--soft-tissue-payload')+1]);rawPose=json.loads((scene/'registered-initial-bodies.json').read_bytes());poses=[row['rest'] for row in rawPose['all_bodies']]
    ids={s['stable_id'] for s in r['surfaces']};current=records(payload,poses,ids);parents=[];requested=set()
    if digest(payload)!=r['child_payload_sha256']:raise ValueError('Mismatched child')
    for s in r['surfaces']:
        for row in s['vertices']:
            for p,b,v,c in row:requested.add(b)
    for spec in r['source_payloads']:
        path=Path(spec['path'])
        if digest(path)!=spec['sha256']:raise ValueError('Mismatched carrier')
        parents.append(records(path,poses,requested))
    result={};out.mkdir()
    for s in r['surfaces']:
        child=s['stable_id'];world,M,first,vo=current[child];source=np.zeros_like(world)
        lengths=np.array([len(z) for z in s['vertices']]);q=np.repeat(np.arange(len(world)),lengths)
        arr=np.array([term for row in s['vertices'] for term in row]);p,b,v=arr[:,:3].astype(int).T;c=arr[:,3]
        for ip,ib in np.unique(np.column_stack([p,b]),axis=0):
            wh=(p==ip)&(b==ib);np.add.at(source,q[wh],c[wh,None]*parents[ip][ib][0][v[wh]])
        err=np.linalg.norm(world-source,axis=1);bad=err>2e-6;row=int(np.argmax(err))
        result[child]=dict(vertices=len(world),failing=int(bad.sum()),worst_vertex=row,worst_um=float(err[row]*1e6),position=world[row].tolist(),parent=source[row].tolist())
        if np.any(bad):np.savez_compressed(out/f'{child}.npz',nodes=np.flatnonzero(bad),target=source[bad],current=world[bad],jacobian=M[bad],first=first,offset=vo)
    (out/'result.json').write_text(json.dumps(dict(input=digest(payload),captured_frame=digest(scene/'registered-initial-bodies.json'),method='Float64 source interpolation in retained native rest frames, each original child and source member retained',surfaces=result),indent=2)+'\n')
    print([(k,v['failing'],round(v['worst_um'],3)) for k,v in result.items()])
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

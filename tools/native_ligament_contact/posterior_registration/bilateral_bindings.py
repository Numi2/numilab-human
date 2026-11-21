"""Replay both donor transforms before testing any change to a gluteal tissue binding."""
from pathlib import Path
import json,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from material_context import applied,sha
from prepare_posterior_muscles import load_registered,interface_audit,closed,mesh_volume
def decode(payload,ids,poses):
    raw=payload.read_bytes()
    magic,abi,nr,nb,nv,ni,_,_=struct.unpack_from('<8s6I32s',raw)
    if (magic,abi)!=(b'NHTISS4\0',5):raise ValueError('Wrong native roster format')
    dtype=np.dtype([('p','<f4',3),('n','<f4',3),('slots','<u4',4),('weights','<f4',4)])
    records=np.frombuffer(raw,'<u4',nr*8,64).reshape(-1,8)
    binding_offset=64+nr*32;vertex_offset=binding_offset+nb*36
    vertices=np.frombuffer(raw,dtype,nv,vertex_offset)
    faces=np.frombuffer(raw,'<u4',ni,vertex_offset+nv*56)
    assert len(raw)==vertex_offset+nv*56+4*ni
    result={}
    for fb,b,first,n,start,count,sid,layer in records.astype(int):
        if sid not in ids:continue
        if b!=2:raise ValueError('Wrong donors')
        v=vertices[first:first+n];rr=[];bodies=[];weight=np.zeros((n,b))
        for j in range(b):
            body,*t=struct.unpack_from('<I8f',raw,binding_offset+36*(fb+j))
            bodies.append(body)
            point=t[:3]+Rotation.from_quat(t[3:7]).apply(v['p'])*t[7]
            rr.append(applied(poses[body],point))
            weight[:,j]=np.where(v['slots']==j,v['weights'],0).sum(1)
        assert abs(weight.sum(1)-1).max()<2e-6
        f=faces[start:start+count].reshape(-1,3)
        if f.min()>=first and f.max()<first+n:f=f-first
        assert f.max()<n
        result[sid]=(v['p'],np.stack(rr,axis=1),weight,bodies,f)
    return result
def run(w,cap,package,audit,exe,out):
    meshes,names,identity=load_registered(w,cap,package,audit)
    args=json.loads(package.read_text())['argv'];payload=Path(args[args.index('--soft-tissue-payload')+1])
    tendons=Path(args[args.index('--tendon-payload')+1]);end=json.loads(tendons.with_suffix('.manifest.json').read_text())
    assert sha(tendons)==end['payload']['sha256']
    poses=np.array(next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))['bodyPoses'])
    raw=decode(payload,(29,30),poses);rows={}
    for sid,(source,endpos,weight,donors,faces) in raw.items():
        key=(51005,sid);v,f=meshes[key]
        if not np.array_equal(f,faces):raise ValueError('Triangle ancestry differs')
        x=(endpos*weight[:,:,None]).sum(1)
        delta=x-v
        row=dict(min_diff_mm=(delta.min(0)*1000).tolist(),max_diff_mm=(delta.max(0)*1000).tolist(),max_replay_error_m=float(np.linalg.norm(delta,axis=1).max()))
        centres={}
        for e in end['endpoints']:
            if e['muscle'] not in {a['name'] for a in names[sid]['matched_muscles']}:continue
            site=applied(poses[e['body_index']],np.array(e['resolved_local_point_m']))
            centres[e['muscle']+'/'+e['endpoint']]=site.tolist()
        row['actuator_sites_world_m']=centres
        cross=json.loads((out/str(sid)/'intersections.json').read_text())
        ids=np.unique(f[np.unique(np.array(cross['51005/%d'%(59-sid)])[:,0])])
        row['medial_crossing_vertices']=len(ids)
        row['femur_weight_at_crossing_q00_q50_q100']=np.quantile(weight[ids,1],[0,.5,1]).tolist()
        row['reference_bounds_m']=[source.min(0).tolist(),source.max(0).tolist()]
        rows[int(sid)]=row
    (out/'binding-parity.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows,flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

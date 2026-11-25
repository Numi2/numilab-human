"""Separate source-surface clearance from a pose-dependent material registration defect."""
import json,mmap,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from rebind_ligament_material import banks
from bindings import audit
from rebind_human_skin_reference import decode
from fast_skin_intersection_audit import install_for_float32
from compile_connected_joint import digest
def exact(w,f,B,T):
    rec=audit.clearance._exact_surface_records(np.ascontiguousarray(w,dtype='f'),f)
    aabb=audit.clearance._float32_skin_aabb_index(rec);tree=audit.intersections._prepare_surface_aabb(rec)
    chosen,n=audit.clearance._candidate_target_rows_for_skin_aabb(T,B,aabb)
    used=np.unique(T[chosen]);z=audit.clearance._exact_surface_records(B[used],np.searchsorted(used,T[chosen]))
    z=[(a,b,c,int(chosen[i]),e)for a,b,c,i,e in z]
    r=audit.intersections._audit_pair_prepared_first(tree,z)
    if r['aabb_candidate_pairs']!=n:raise ValueError('Incomplete exact test')
    return r
def run(material,case,prior,out):
    install_for_float32()
    reg=json.loads((material/'material-boundary.json').read_bytes())
    frame=Path(reg['initial_bodies_path']);ref={b['body_index']:b['current']for b in json.loads(frame.read_bytes())['bodies']}
    skin=material/'registered-boundary.nhskin';s=decode(skin.read_bytes());v=s['verts'][:,:3].astype(float);f=s['inds']
    if digest(skin)!=reg['nhskin_sha256'] or digest(frame)!=reg['initial_frame_sha256']:raise ValueError('Unmatched material')
    meta=json.loads(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')).read_bytes());p=Path(meta['accepted_pack_path'])
    previous=json.loads(prior.read_bytes())
    if digest(p)!=previous['pack'] or digest(p)!=meta['pack_file_sha256']:raise ValueError('Unmatched native capture')
    launch=json.loads((case/'launch.json').read_bytes());bones=Path(launch['argv'][3]);source=banks(bones)
    if digest(bones)!=launch['pins'][str(bones)]:raise ValueError('Unmatched cortical material')
    pose={b['body_index']:b for b in meta['accepted_body_poses']};rows={}
    with p.open('rb')as st,mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)as raw:
        for (sem,sid),_,b,t in audit.primitives(raw,pose):
            if sem!=51004 or sid not in (16,18,19):continue
            owners=[owner for owner,idx in source if idx==sid]
            if len(owners)!=1:raise ValueError('Ambiguous cortical owner')
            owner,=owners;a=ref[owner];current=pose[owner]
            sf,X,_=source[owner,sid]
            P=Rotation.from_quat(current['quaternion_xyzw']).apply(X)+current['position_m']
            va=np.asarray(b,dtype=float);err=max(cKDTree(va).query(P)[0].max(),cKDTree(P).query(va)[0].max())
            if len(t)!=len(sf) or err>1.e-6:raise ValueError(f'Cortex {sid} does not match native geometry: {err} m')
            R=Rotation.from_quat(current['quaternion_xyzw'])*Rotation.from_quat(a['orientation_world_xyzw']).inv()
            transported=R.apply(v-a['position_world_m'])+current['position_m']
            r=exact(transported,f,b,t);pairs=r['triangle_pairs']
            rows[sid]=dict(owner=owner,source_to_native_max_m=float(err),source_pairs=len(pairs),native_pairs=previous['rows'][str(sid)]['native']['pairs'],source_skin_ids=np.unique(f[sorted(set(i for i,j in pairs))]).tolist() if pairs else [])
    out.write_text(json.dumps(dict(bones=digest(bones),loaded_skin=digest(skin),source_frame=digest(frame),native_pack=digest(p),rows=rows),indent=2)+'\n')
    print({k:(v['source_pairs'],v['native_pairs'])for k,v in rows.items()})
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))

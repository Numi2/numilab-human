"""Locate the pelvic crossing in material coordinates, separately from bed displacement."""
from pathlib import Path
import json,mmap,sys
import igl,numpy as np
from scipy.spatial.transform import Rotation
from bindings import audit,read_skin
from fast_skin_intersection_audit import install_for_float32
from rebind_human_skin_reference import decode
from compile_connected_joint import digest
def run(material,case,out):
    install_for_float32()
    s=decode((material/'registered-boundary.nhskin').read_bytes());reg=json.loads((material/'material-boundary.json').read_bytes());r0=json.loads(Path(reg['initial_bodies_path']).read_bytes())['bodies']
    rc,=(case/'native-traces/accepted-geometry').glob('*.receipt.json');meta=json.loads(rc.read_bytes());p=Path(meta['accepted_pack_path'])
    if digest(p)!=meta['pack_file_sha256'] or reg['nhskin_sha256']!=digest(material/'registered-boundary.nhskin'):raise ValueError('Unmatched source')
    pose={b['body_index']:b for b in meta['accepted_body_poses']}
    v,f=read_skin(p,meta);X=s['verts'][:,:3].astype(float)
    if not np.array_equal(f,s['inds']): raise ValueError('Different material surface connectivity')
    bind=s['binds']['body'];weights=s['weights']
    posed=np.zeros_like(X)
    for k,(b,body) in enumerate(zip(bind,r0)):
        if b!=body['body_index']:raise ValueError('Mismatched material coordinates')
        a=body['current'];q=pose[b];R=Rotation.from_quat(q['quaternion_xyzw'])*Rotation.from_quat(a['orientation_world_xyzw']).inv()
        posed+=weights[:,k,None]*(R.apply(X-a['position_world_m'])+q['position_m'])
    prepared={}
    for key,points in [('native',v),('kinematic',posed.astype('f'))]:
        w=np.ascontiguousarray(points,dtype='f8')*1000;faces=np.ascontiguousarray(f,dtype='i8');tree=igl.AABB();tree.init(w,faces)
        records=audit.clearance._exact_surface_records(points.astype('f'),f)
        prepared[key]=(w,faces,tree,audit.intersections._prepare_surface_aabb(records),audit.clearance._float32_skin_aabb_index(records))
    rows={}
    with p.open('rb')as st,mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)as raw:
        for (sem,sid),_,verts,indices in audit.primitives(raw,pose):
            if sem!=51004 or sid not in (16,18,19):continue
            rows[sid]={};qc=verts[indices].astype(float).mean(axis=1);Q=np.ascontiguousarray(np.vstack([np.unique(verts,axis=0),qc])*1000)
            for k,(w,faces,tree,solid,box) in prepared.items():
                chosen,n=audit.clearance._candidate_target_rows_for_skin_aabb(indices,verts,box);used=np.unique(indices[chosen])
                z=audit.clearance._exact_surface_records(verts[used],np.searchsorted(used,indices[chosen]))
                z=[(a,b,c,int(chosen[i]),e)for a,b,c,i,e in z]
                h=audit.intersections._audit_pair_prepared_first(solid,z)
                if h['aabb_candidate_pairs']!=n:raise ValueError('Invalid broad phase')
                d2,fi,_=tree.squared_distance(w,faces,Q)
                wf=np.abs(igl.fast_winding_number(w,faces,Q));maybe=(wf>.25)&(wf<.75)
                if maybe.any():wf[maybe]=np.abs(igl.winding_number(w,faces,Q[maybe]))
                outside=np.flatnonzero(wf<.5);cross=sorted(set(int(i)for i,j in h['triangle_pairs']));ids=np.unique(f[cross]) if cross else np.array([],int)
                m=np.max(np.linalg.norm(v[ids]-posed[ids],axis=1))*1000 if len(ids) else 0
                own=np.argsort(weights[ids].mean(0))[::-1][:4] if len(ids) else []
                rows[sid][k]=dict(pairs=h['count'],exterior_samples=len(outside),max_exterior_mm=float(np.sqrt(d2[outside].max())) if len(outside) else 0,
                    skin_vertex_ids=ids.tolist(),max_bed_or_numerical_displacement_mm=float(m),donors={str(bind[i]):float(weights[ids,i].mean())for i in own})
    out.write_text(json.dumps(dict(pack=digest(p),seconds=meta['accepted_time_s'],rows=rows),indent=2)+'\n')
    print({k:{s:(v['pairs'],round(v['max_exterior_mm'],4),round(v['max_bed_or_numerical_displacement_mm'],4))for s,v in r.items()}for k,r in rows.items()})
if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))

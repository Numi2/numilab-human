"""Check actual Float32 APL surfaces against the native skin and all nearby anatomical banks."""
import json,mmap,sys
from pathlib import Path
import igl
import numpy as np
from igl.copyleft import cgal
from bindings import audit
from compile_connected_joint import digest
from fast_skin_intersection_audit import install_for_float32

def records(v,f):
    return audit.clearance._exact_surface_records(v,f)

def check(case,out):
    install_for_float32()
    rc,=(case/'native-traces/accepted-geometry').glob('*.receipt.json')
    meta=json.loads(rc.read_bytes());pack=Path(meta['accepted_pack_path'])
    if digest(pack)!=meta['pack_file_sha256'] or meta['physical_endpoint']!='accepted':raise ValueError('Unmatched native geometry')
    poses={b['body_index']:b for b in meta['accepted_body_poses']}
    with pack.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as raw:
        targets={};result={};available=0
        for key,_,v,f in audit.primitives(raw,poses):
            if key[0]==51005 and key[1] in (147,148):
                vert,indices=np.unique(v,axis=0,return_inverse=True)
                faces=np.ascontiguousarray(indices[f],dtype='i4')
                s=records(v,f)
                result[key[1]]=dict(self_pairs=len(cgal.remesh_self_intersections(np.ascontiguousarray(vert,dtype='f8'),faces,detect_only=True)[2]),pairs_by_semantic={},witnesses=[],bone_containment=[])
                targets[key]=(audit.intersections._prepare_surface_aabb(s),audit.clearance._float32_skin_aabb_index(s),np.min(v,axis=0),np.max(v,axis=0),vert,faces)
        if len(targets)!=2:raise ValueError('Bilateral surfaces required')
        for key,primitive,v,f in audit.primitives(raw,poses):
            available+=1
            for t,(tree,boxes,lo,hi,verts,tfaces) in targets.items():
                if key==t or np.any(v.max(0)<lo) or np.any(v.min(0)>hi):continue
                r=result[t[1]];chosen,n=audit.clearance._candidate_target_rows_for_skin_aabb(f,v,boxes)
                if len(chosen):
                    used=np.unique(f[chosen]);z=records(v[used],np.searchsorted(used,f[chosen]))
                    q=[(a,b,c,int(chosen[i]),e)for a,b,c,i,e in z]
                    h=audit.intersections._audit_pair_prepared_first(tree,q)
                    if h['aabb_candidate_pairs']!=n:raise ValueError('Incomplete enumeration')
                    if h['count']:
                        k=key[0];r['pairs_by_semantic'][k]=r['pairs_by_semantic'].get(k,0)+h['count']
                        r['witnesses'].append(dict(semantic=k,stable_id=key[1],primitive=primitive,count=h['count'],first_pair=h['triangle_pairs'][0]))
                if key[0]==51004:
                    xyz,iv=np.unique(v,axis=0,return_inverse=True)
                    vf=np.ascontiguousarray(iv[f],dtype='i8');xyz=np.ascontiguousarray(xyz,dtype='f8')
                    a=np.ascontiguousarray(verts,dtype='f8')
                    inside=abs(igl.winding_number(xyz,vf,a))>.5
                    if inside.any():
                        tr=igl.AABB();tr.init(xyz,vf)
                        d=np.sqrt(tr.squared_distance(xyz,vf,a[inside])[0])
                        r['bone_containment'].append(dict(cortex=key[1],muscle_vertices_inside=int(inside.sum()),maximum_depth_mm=float(1000*d.max())))
    out.write_text(json.dumps(dict(pack=digest(pack),accepted_time_s=meta['accepted_time_s'],tested_banks=available,rows=result),indent=2)+'\n')
    print([(sid,s['self_pairs'],s['pairs_by_semantic'])for sid,s in result.items()],flush=True)

if __name__=='__main__':check(*(Path(p).resolve() for p in sys.argv[1:]))

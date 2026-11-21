"""Recover the exact source-to-body registrations and check the canonical reference pose."""
from pathlib import Path
import json
import sys
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from compare_atlas import vertices_and_faces,closest_distance
from shoulder_inputs import load,transforms,digest
from audit_solids import winding

def prepare(launch,registration,motion,archive):
    cfg,c,m,an,poses=load(launch,registration,motion)
    manifest=Path(cfg['argv'][3]).with_suffix('.manifest.json')
    doc=json.loads(manifest.read_text())
    ref=doc['source']['runtime_reference']['manifest']
    refpath=Path(ref['file']);assert digest(refpath)==ref['sha256']
    records=json.loads(refpath.read_text())['core_tree']['source_body_records']
    default={r['core_body_index']:np.array(r['default_com_position_world_m']+r['default_inertial_quaternion_world_xyzw']) for r in records}
    assert digest(archive)==doc['source']['bodyparts']['archives'][0]['sha256']
    with zipfile.ZipFile(archive) as z:
        data={}
        members={a['member_id']:a for a in doc['source']['anchors']}
        for mid in ('FJ3384','FJ3368','FJ3279','FJ3262','FJ3362','FJ3237','FJ3391','FJ3286'):
            ref=members[mid];body=ref['core_body_index']
            index=next(s for s,r in c.items() if r['body']==body and r['vertexCount']==ref['vertex_count'] and r['indexCount']==3*ref['triangle_count'])
            raw=z.read(next(p for p in z.namelist() if p.endswith('/'+mid+'.obj')))
            import hashlib
            assert hashlib.sha256(raw).hexdigest()==ref['member_sha256']
            v,f=vertices_and_faces(raw); local,ff,*_=m[index]
            assert np.array_equal(f,ff)
            b=default[body]
            world=b[:3]+Rotation.from_quat(b[3:]).apply(local)
            lhs=np.c_[v,np.ones(len(v))]
            affine=np.linalg.lstsq(lhs,world,rcond=None)[0]; linear=affine[:3].T
            u,s,vt=np.linalg.svd(linear); rot=Rotation.from_matrix(u@vt); scale=s.mean(); center=world.mean(axis=0)-rot.apply(v.mean(axis=0))*scale
            error=np.linalg.norm(rot.apply(v)*scale+center-world,axis=1)
            assert error.max()<2e-6 and np.linalg.det(linear)>0
            data[index]=(mid,body,v,f,local,world,rot,scale,center)
    return cfg,c,m,an,poses,default,data

if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:])
    _,c,m,an,poses,default,data=prepare(*args)
    summary=[]
    for a in an['candidates']:
        s,h=a['stable_ids']
        rs,rh=data[s],data[h]
        def point(r,j):
            return np.asarray(a['material_barycentric'][j])@r[5][r[3][a['triangle_rows'][j]]]
        p=point(rs,0);w=winding(rh[5][rh[3]],p);d,_=closest_distance(rh[5][rh[3]],p)
        # Disagreement between the two placements evaluated at the same source scapular site.
        atlas_site=np.array(a['material_barycentric'][0])@rs[2][rs[3][a['triangle_rows'][0]]]
        transfer=rh[6].apply(atlas_site)*rh[7]+rh[8]-p
        summary.append(dict(side=a['side'],native_default_origin_winding=w,default_separation_m=d,
           inferred_registration_displacement_at_origin_m=transfer.tolist(),magnitude_m=float(np.linalg.norm(transfer)),
           group_rotation_delta_deg=float((rh[6]*rs[6].inv()).magnitude()*180/np.pi),scapula_scale=rs[7],humerus_scale=rh[7]))
    out.write_text(json.dumps(summary,indent=2)+'\n')

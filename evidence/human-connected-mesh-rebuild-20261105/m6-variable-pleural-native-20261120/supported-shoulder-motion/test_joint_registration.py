"""Test a single source-derived joint-base correction per side on the retained motion."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compare_registration import prepare
from compare_atlas import closest_distance
from audit_solids import winding
from shoulder_inputs import transforms, unit, old

def evaluate(launch,reg,motion,archive):
    _,c,m,an,poses,default,data=prepare(launch,reg,motion,archive)
    result=[]
    for row in an['candidates']:
        sid,hid=row['stable_ids']; s,h=data[sid],data[hid]
        fs,fh=row['triangle_rows']; ws,wh=map(np.asarray,row['material_barycentric'])
        scap_body,hum_body=s[1],h[1]
        native_s=Rotation.from_quat(default[scap_body][3:])
        origin=ws@s[4][s[3][fs]]; end=wh@h[4][h[3][fh]]
        atlas_end=wh@h[2][h[3][fh]]
        atlas_origin=ws@s[2][s[3][fs]]
        # Rigid placement, preserving all humeral proper lengths. Constrain translation at the
        # source humeral site, using the native scapula's atlas mapping at the reference pose.
        rot_world=s[6]*h[6].inv()
        pivot=h[6].apply(atlas_end)*h[7]+h[8]
        wanted=s[6].apply(atlas_end)*s[7]+s[8]
        delta_local=native_s.inv()*rot_world*native_s
        centroid=default[scap_body][:3]
        disp_local=native_s.inv().apply(rot_world.apply(centroid-pivot)+wanted-centroid)
        n0=unit(ws@m[sid][2][m[sid][1][fs]]);n1=unit(wh@m[hid][2][m[hid][1][fh]])
        reports=[]
        for key,pose in [(0,default),*poses.items()]:
            tr,rot=transforms(pose,scap_body,hum_body)
            newrot=delta_local*rot
            newtr=delta_local.apply(tr)+disp_local
            x=newrot.inv().apply(origin-newtr);tri=h[4][h[3]]
            d,f=closest_distance(tri,x)
            point=newtr+newrot.apply(end)
            di=unit(point-origin)
            hit0=old.audit.lp.vector_hits(origin,point,(s[4],s[3]))
            hit1=old.audit.lp.vector_hits(x,end,(h[4],h[3]))
            reports.append(dict(step=key,scapular_origin_in_humerus=winding(tri,x),
                       separation_m=d,length_m=float(np.linalg.norm(point-origin)),
                       cosine=[float(n0@di),float(-newrot.apply(n1)@di)],
                       shoulder_bone_interior_hits=[len(hit0),len(hit1)]))
        result.append(dict(side=row['side'],atlas_length_m=float(np.linalg.norm(atlas_origin-atlas_end)),scapular_body=scap_body,humeral_body=hum_body,
                           delta_parent_quaternion_xyzw=delta_local.as_quat().tolist(),delta_parent_translation_m=disp_local.tolist(),poses=reports))
    return result
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(evaluate(*args),indent=2,allow_nan=False)+'\n')

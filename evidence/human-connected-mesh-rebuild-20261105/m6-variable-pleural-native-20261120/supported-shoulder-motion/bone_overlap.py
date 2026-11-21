"""Exact triangle pairs on quantized source and registered shoulder surfaces."""
from pathlib import Path
import json,sys
from scipy.spatial.transform import Rotation
import numpy as np
from compare_registration import prepare
from fast_skin_intersection_audit import install_for_float32,owner


def intersections(v,f,p,g):
    # Preserve face row identity for witnesses. Whole-bank AABB screening
    # discards only triangles that cannot meet the other bank.
    v,p=(np.asarray(vertices,dtype=np.float32) for vertices in (v,p))
    selection=[]
    for points,faces,other in ((v,f,p),(p,g,v)):
        tris=points[faces]
        selected=np.flatnonzero(np.all(tris.max(axis=1)>=other.min(axis=0),axis=1)&
                                np.all(tris.min(axis=1)<=other.max(axis=0),axis=1))
        records,_=owner.float32_surface_records(points,faces[selected])
        selection.append((records,selected))
    (a,ix),(b,jx)=selection
    if not a or not b:return []
    result=owner._audit_pair_prepared_first(owner._prepare_surface_aabb(a),b)
    return [[int(ix[i]),int(jx[j])] for i,j in result['triangle_pairs']]


def audit(launch,registration,motion,archive,correction):
    _,_,_,anchors,poses,default,data=prepare(launch,registration,motion,archive)
    corrections=json.loads(correction.read_text());report=[]
    install_for_float32()
    for sites,change in zip(anchors['candidates'],corrections):
        s,h=(data[k] for k in sites['stable_ids']); delta=Rotation.from_quat(change['delta_parent_quaternion_xyzw'])
        shift=change['delta_parent_translation_m']
        for key in ('atlas',0,32,256,8000):
            if key=='atlas':
                runs={'source':(s[2],h[2])}
            else:
                pose=default if key==0 else poses[key]
                ps,ph=pose[s[1]],pose[h[1]]
                r=Rotation.from_quat(ps[3:]); x=ps[:3]+r.apply(s[4]); y=ph[:3]+Rotation.from_quat(ph[3:]).apply(h[4])
                yc=ps[:3]+r.apply(delta.apply(r.inv().apply(y-ps[:3]))+shift)
                runs={'registered':(x,y),'restored_joint_candidate':(x,yc)}
            for name,(x,y) in runs.items():
                pairs=intersections(x,s[3],y,h[3])
                report.append(dict(side=sites['side'],step=key,placement=name,count=len(pairs),face_pairs=pairs))
                print(sites['side'],key,name,len(pairs),flush=True)
    return report
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(audit(*args),separators=(',',':'))+'\n')

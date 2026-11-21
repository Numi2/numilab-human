"""Test the complementary bone registration while retaining the original articulated source."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from shoulder_inputs import load
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32

def audit(launch,anchors,motion,frames):
    _,catalog,meshes,sites,poses=load(launch,anchors,motion)
    install_for_float32()
    corrections=json.loads(frames.read_text()); moving={r['stable_ids'][0]:i for i,r in enumerate(sites['candidates'])}
    assert len(moving)==2
    calibrated={}
    for sid,k in moving.items():
        r=corrections[k]; rot=Rotation.from_quat(r['parent_rotation_xyzw'])
        v=meshes[sid][0]
        calibrated[sid]=rot.inv().apply(v-np.asarray(r['parent_translation_base_m']))
    result=[]
    for key in (-3,32,256,8000):
        pose=poses[key];geometry={}
        for sid,(v,f,*_) in meshes.items():
            p=pose[catalog[sid]['body']];r=Rotation.from_quat(p[3:]);vs=[v,calibrated[sid]] if sid in moving else [v,v]
            geometry[sid]=[p[:3]+r.apply(x) for x in vs]
        rows=[]
        for sid in moving:
            for other in sorted(geometry):
                if other==sid or other in moving and other<sid:continue
                counts=[];ex=[]
                for k in (0,1):
                    x,y=geometry[sid][k],geometry[other][k]
                    if np.any(x.min(axis=0)>y.max(axis=0)+1e-7) or np.any(y.min(axis=0)>x.max(axis=0)+1e-7):hits=[]
                    else:hits=intersections(x,meshes[sid][1],y,meshes[other][1])
                    counts.append(len(hits));ex.append(hits[:1])
                if any(counts):rows.append(dict(stable_ids=[sid,other],before=counts[0],after=counts[1],witness=ex))
        print(key,sum(r['before'] for r in rows),sum(r['after'] for r in rows),
              [(r['stable_ids'],r['after']) for r in rows if r['after']>0 and r['before']==0],flush=True)
        result.append(dict(step=key,rows=rows))
    return result
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(audit(*args),indent=2)+'\n')

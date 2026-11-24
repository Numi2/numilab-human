"""Check the registered shoulder muscle against the actual 185 rigid bone banks."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from audit_registered_skeleton import closed_solid
from rebind_ligament_material import banks


def assess(case,volume,out):
    packet=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    d=json.loads(packet.read_bytes());cfg=json.loads((case/'launch.json').read_bytes())
    asset=Path(cfg['argv'][3]);z=np.load(volume);surfs=banks(asset)
    if digest(asset)!=cfg['pins'][str(asset)] or len(surfs)!=185:raise ValueError('Incomplete native bone domain')
    results={}
    for name,p in [('opening',d['initial_anatomical_registration']['body_poses']),('accepted',d['accepted_body_poses'])]:
        poses={r['body_index']:r for r in p}
        x=z[name];o=x.mean(0);_,_,muscle=closed_solid((x-o)*1000,z['boundary'])
        entries=[]
        for (body,sid),(f,v,_) in surfs.items():
            pose=poses[body]
            bone=(Rotation.from_quat(pose['quaternion_xyzw']).apply(v)+pose['position_m']).astype('f4').astype(float)
            if np.any(x.max(0)<bone.min(0)-.01) or np.any(bone.max(0)<x.min(0)-.01):continue
            _,_,solid=closed_solid((bone-o)*1000,f)
            overlap=muscle^solid
            if str(overlap.status())!='Error.NoError':raise ValueError(str(overlap.status()))
            vol=float(overlap.volume()); gap=float(muscle.min_gap(solid,10.0))
            entries.append(dict(body=int(body),bank=int(sid),overlap_mm3=vol,gap_with_10mm_search=gap))
        results[name]=dict(muscle_mm3=float(muscle.volume()),interactions=entries,checked=len(surfs))
    out.write_text(json.dumps(dict(volume_registration=digest(volume),bone_payload=digest(asset),
        native_pose=digest(packet),results=results),indent=2,allow_nan=False)+'\n')
    print(results['accepted'],flush=True)

if __name__=='__main__':assess(*(Path(s).resolve()for s in sys.argv[1:]))

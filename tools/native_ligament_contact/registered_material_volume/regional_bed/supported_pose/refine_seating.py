"""Re-seat an unchanged joint configuration using the recomputed native material witnesses."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_registered_bed_release import evaluate as measure
from solve_material_seating import evaluate as solve
from compile_connected_joint import digest
def apply(base,out):
    c=json.loads((base/'launch.json').read_bytes());rel=measure(base);fit=solve(base,rel)
    if rel['launch']!=digest(base/'launch.json') or fit['initial_witness_sha256']!=rel['witness'] or fit['max_stock_fraction']>=1:raise ValueError('Unmatched accepted support state')
    root=json.loads((base/'native-initial-bodies.json').read_bytes())['all_bodies'][0]['current']
    com=np.array(fit['center_world_m']);r=Rotation.from_rotvec(fit['rotation_world_vector_rad']);n=np.array(rel['normal'])
    xyz=com+r.apply(np.array(root['position_world_m'])-com)-fit['translation_toward_plane_m']*n
    rot=r*Rotation.from_quat(root['orientation_world_xyzw']);i=c['argv'].index('--root-pose')+1
    c['argv'][i:i+7]=list(map(str,np.r_[xyz,rot.as_quat()]))
    out.mkdir();doc=out/'seating.json';doc.write_text(json.dumps(dict(release=rel,fit=fit),indent=2)+'\n')
    c['pins'][str(doc.resolve())]=digest(doc);(out/'launch.json').write_text(json.dumps(c,indent=2)+'\n')
    print(fit['translation_toward_plane_m'],fit['rotation_deg'],fit['load_normal_n'],fit['max_stock_fraction'])
if __name__=='__main__':apply(*(Path(p).resolve() for p in sys.argv[1:]))

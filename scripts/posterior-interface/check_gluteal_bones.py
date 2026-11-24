"""Reuse the current native body-pose and material-solid intersection check."""
import json
from pathlib import Path
import sys
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run


def check(case,owner):
    m=json.loads((case/'material-check.json').read_bytes())
    receipt=case/f"native-traces/accepted-geometry/step-{m['accepted_step']}.receipt.json"
    assert m['receipt_sha256']==digest(receipt)
    bodies=json.loads(receipt.read_bytes())['accepted_body_poses']
    packet=dict(receipt_sha256=digest(receipt),all_bodies=[dict(body_index=r['body_index'],
        current=dict(position_world_m=r['position_m'],orientation_world_xyzw=r['quaternion_xyzw']))for r in bodies])
    frame=case/'accepted-registered-bone-pose.json'
    frame.write_text(json.dumps(packet)+'\n')
    src=case/'bone-audit-input'
    src.mkdir(exist_ok=True)
    for name,f in [('mesh.node',case/'physical-accepted.node'),('mesh.ele',case/'audit/witness-material/volume/mesh.ele')]:
        if not (src/name).exists():(src/name).symlink_to(f)
    run(owner,case/'bone-solids',frame,None,src)
    r=json.loads((case/'bone-solids/result.json').read_bytes())
    if len(r['results'])!=185 or r['capture']!=digest(frame) or r['mesh']!=digest(src/'mesh.node'):
        raise ValueError('Source, capture or bank coverage mismatch')
    print(sum(d['skin']+d['fat'] for d in r['results'].values()),flush=True)

if __name__=='__main__':check(*(Path(x).resolve() for x in sys.argv[1:]))

"""Bind the MyoSim visual default pose to the pinned source equalities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
from myo_sim.build.compose import build_model

from numilab_human.myosim_visual import _visual_qpos


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'Sources/myosim/myo_sim-33c89c2b.tar.gz'
model = build_model('myofullbody')
raw_q, raw_info = _visual_qpos(model, mujoco, True)
projected_q, projected_info = _visual_qpos(model, mujoco, False)
assert raw_info['pose_state'] == 'literal_unprojected_source_qpos0'
assert projected_info['pose_state'] == 'source_equality_projected_neutral'
assert projected_info['source_joint_equalities_projected'] == 51
assert projected_info['maximum_equality_coordinate_correction'] > .05

positions = {}
for name, qpos in [('raw_source_rest', raw_q), ('projected_neutral', projected_q)]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    positions[name] = {
        side: [float(value) for value in data.xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, 'patella_'+side)]]
        for side in ('r', 'l')
    }
shift = {
    side: positions['raw_source_rest'][side][1] - positions['projected_neutral'][side][1]
    for side in ('r', 'l')
}
assert all(value > .05 for value in shift.values())
result = {
    'schema': 'numi.human.myosim-source-visual-pose-selection.v1',
    'source_archive_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'source_revision': '33c89c2bde282553dde3f526768eb3bdcfaa7649',
    'mujoco_version': mujoco.__version__,
    'raw': raw_info, 'projected': projected_info,
    'patella_body_center_world_m': positions,
    'patella_body_center_anterior_shift_m': shift,
    'source_anterior_axis': [0., -1., 0.],
    'boundary': ('CPU source-kinematics and visual pose-selection check only. '
                 'The mesh full-support check is separately recorded in the compiled-bone '
                 'patellar audit. This receipt is not a new native rendered image, '
                 'cartilage orientation, loaded contact, or clinical anatomy.'),
}
(HERE/'source-visual-pose-selection.json').write_text(
    json.dumps(result, indent=2, sort_keys=True)+'\n'
)
print(json.dumps({'active_equalities': projected_info['source_joint_equalities_projected'],
                  'anterior_shift_m': shift}))

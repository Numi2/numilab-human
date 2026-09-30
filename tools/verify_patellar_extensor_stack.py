"""Independently remeasure the pinned Open Knee extensor ordering and source pose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from myo_sim.build.compose import build_model
from numilab_human.open_knee import parse_source
from numilab_human.myosim_visual import _visual_qpos


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'Build/patellar-extensor-stack-20260930'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    names = ('PTB', 'PTC', 'QAT', 'PTL')
    centers_mm = {name: np.asarray(source.regions[name].nodes_mm).mean(axis=0)
                  for name in names}
    anterior = np.asarray(source.landmarks['Yf_axis'])
    proximal = np.asarray(source.landmarks['Zf_axis'])
    source_metrics_mm = {
        'patellar_cartilage_posterior_to_bone_m':
            float((centers_mm['PTB'] - centers_mm['PTC']) @ anterior),
        'quadriceps_tendon_proximal_to_patella_m':
            float((centers_mm['QAT'] - centers_mm['PTB']) @ proximal),
        'patellar_tendon_distal_to_patella_m':
            float((centers_mm['PTB'] - centers_mm['PTL']) @ proximal),
    }
    compiled = {}
    for side, stem in [('left', 'open-knee-oks003-left'),
                       ('right', 'open-knee-oks003-right-mirrored')]:
        manifest_path = BUILD / side / (stem + '.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        payload = BUILD / side / (stem + '.nhknee')
        assert manifest['payload']['sha256'] == sha(payload)
        assert manifest['registration']['patella_anterior_offset_m'] >= .025
        assert manifest['registration']['fibula_lateral_offset_m'] >= .020
        scale = manifest['registration']['uniform_scale'] * .001
        offsets = manifest['registration']['patellar_extensor_stack']
        for key, raw in source_metrics_mm.items():
            # The source Yf axis and admitted world anterior differ by only
            # 0.001 rad; the 5 um bound includes that exact registration angle.
            assert abs(offsets[key] - scale * raw) < 5e-6, (side, key)
        compiled[side] = {
            'manifest_sha256': sha(manifest_path),
            'payload_sha256': sha(payload),
            'patella_anterior_to_knee_anchor_m': manifest['registration']['patella_anterior_offset_m'],
            'fibula_lateral_to_tibia_m': manifest['registration']['fibula_lateral_offset_m'],
            'extensor_stack_m': offsets,
            'uniform_scale': manifest['registration']['uniform_scale'],
        }
    assert all(compiled['left']['extensor_stack_m'][name] ==
               compiled['right']['extensor_stack_m'][name] for name in source_metrics_mm)

    model = build_model('myofullbody')
    data = mujoco.MjData(model)
    visual = {}
    for raw in (True, False):
        qpos, pose = _visual_qpos(model, mujoco, raw)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        values = {}
        for side in ('r', 'l'):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'patella_' + side)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_angle_' + side)
            assert body >= 0 and joint >= 0
            values[side] = float(data.xanchor[joint][1] - data.xpos[body][1])
        visual[pose['pose_state']] = values
    assert all(value < 0 for value in visual['literal_unprojected_source_qpos0'].values())
    assert all(value >= .025 for value in visual['source_equality_projected_neutral'].values())

    patella_path = ROOT / 'Docs/media/patellar-anteriority-20260929/summary.json'
    patella = json.loads(patella_path.read_text())
    posed = [row for pose in patella['projected_poses'] for row in pose['patellar_anteriority']]
    assert len(posed) == 16 and all(row['passed'] for row in posed)
    result = {
        'schema': 'numi.human.patellar-extensor-stack-check.v1',
        'status': 'passed_gross_source_ordering_and_projected_presentation',
        'open_knee_source_sha256': {name: sha(ROOT / 'Sources/open-knee-oks003' / name)
                                      for name in ('Geometry.feb', 'ModelProperties.xml',
                                                   'FeBio_custom.feb', 'license.txt')},
        'registration_sha256': sha(ROOT / 'Build/knee-parity-registration-20260929/candidate.v6.registration.json'),
        'compiled': compiled,
        'source_axis_offsets_mm': source_metrics_mm,
        'myosim_visual_body_center_anterior_offset_m': visual,
        'compiled_bone_full_support_summary_sha256': sha(patella_path),
        'compiled_bone_full_support_side_pose_checks_passed': len(posed),
        'source_code_sha256': {
            name: sha(ROOT / name) for name in (
                'src/numilab_human/myosim_visual.py', 'src/numilab_human/open_knee.py',
                'tools/verify_patellar_extensor_stack.py')},
        'cartilage_facing_surface_qualified': False,
        'loaded_patellofemoral_contact_qualified': False,
        'clinical_anatomy_qualified': False,
        'boundary': ('Bone/cartilage/tendon centroid ordering is a source-specific regression '
                     'check. The selected compiled-bone vertex check covers static projected '
                     'poses. Neither check proves patellar cartilage-facing normals, contact '
                     'pressure, loaded extensor force transfer, or clinical anatomy.'),
    }
    output = ROOT / 'Docs/media/patellar-extensor-stack-20260930/receipt.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'source_visual': visual,
                      'extensor_stack_left_m': compiled['left']['extensor_stack_m']}))


if __name__ == '__main__':
    main()

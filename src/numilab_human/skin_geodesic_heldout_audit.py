"""Exact native skin audit for four source-pose cases unused in candidate fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import model as human
from .compiled_quotient_embeddedness import coordinate_quotient
from .skin_embeddedness_gate import _native_mesh, _source_mesh
from .surface_topology_audit import exact_embedding
from .whole_body_embeddedness import atomic_json


SCHEMA = 'numi.human.derived-geodesic-skin-heldout-native-audit.v1'
CODE_FILES = ('skin_geodesic_heldout_audit.py',
              'skin_embeddedness_gate.py',
              'compiled_quotient_embeddedness.py',
              'surface_topology_audit.py',
              'cardiac_cavity_intersections.py')
POSES = {
    'knee-bilateral-045': ('knee-flexion', [[106, .45], [120, .45]]),
    'knee-bilateral-120': ('knee-flexion', [[106, 1.2], [120, 1.2]]),
    'knee-asymmetric-060-015': ('asymmetric-knee', [[106, .6], [120, .15], [101, .2], [115, .05]]),
    'reach-bilateral-040-050': ('coupled-reach', [[36, .4], [39, .5], [74, .4], [77, .5]]),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('geodesic skin heldout audit: ' + message)


def _read_run(root: Path, label: str, kind: str, prior: dict,
              payload: Path, fingerprint: int, nb: int, nv: int, ni: int,
              faces: np.ndarray):
    folder = root/label/kind
    command = human.read_json(folder/'command.json')
    argv = command['argv']
    source_case, coords = POSES[label]
    original = prior[source_case]['command']
    expected = original[:original.index('--pose-q')]
    expected[4] = str(folder)
    expected[6] = str(payload)
    for index, value in coords:
        expected.extend(['--pose-q', str(index), str(value)])
    require(command['case'] == label and command['kind'] == kind
            and command['source_command_case'] == source_case
            and command['pose_q'] == coords and argv == expected
            and command['skin_payload_sha256'] == human.sha256(payload)
            and command['probe_sha256'] == human.sha256(Path(argv[0]))
            and command['rigid_sha256'] == human.sha256(Path(argv[1]))
            and command['muscle_sha256'] == human.sha256(Path(argv[2]))
            and command['bones_sha256'] == human.sha256(Path(argv[3]))
            and (folder/'exit.code').read_text().strip() == '0'
            and (folder/'stderr').read_bytes() == b'',
            f'{label}/{kind} native command identity')
    stdout = (folder/'stdout').read_text()
    require('myosim_articulated_bodyparts_bone_visual=ok' in stdout
            and 'rendering_performed=true' in stdout
            and 'metal_pose_device="Apple M4"' in stdout
            and 'renderer_device="Apple M4"' in stdout,
            f'{label}/{kind} Apple M4 native output')
    packs, poses = list(folder.glob('*.mrvpack')), list(folder.glob('*.skin-poses.json'))
    require(len(packs) == len(poses) == 1, f'{label}/{kind} native files')
    vertices = _native_mesh(packs[0], poses[0], fingerprint, nb, nv, ni, faces)
    return vertices, human.read_json(poses[0]), {
        'command_sha256': human.sha256(folder/'command.json'),
        'pack_sha256': human.sha256(packs[0]),
        'pose_sha256': human.sha256(poses[0]),
        'stdout_sha256': human.sha256(folder/'stdout'),
    }


def audit(base_payload: Path, base_manifest_path: Path, candidate_dir: Path,
          prior_receipt_path: Path, heldout_root: Path, output_dir: Path) -> dict:
    paths = [Path(p).resolve() for p in (base_payload, base_manifest_path,
             candidate_dir, prior_receipt_path, heldout_root, output_dir)]
    base_payload, base_manifest_path, candidate_dir, prior_receipt_path, heldout_root, output_dir = paths
    nb, nv, ni, fingerprint, source, faces = _source_mesh(
        base_payload, human.read_json(base_manifest_path))
    manifest = human.read_json(candidate_dir/'candidate.manifest.json')
    candidate_payload = candidate_dir/manifest['payload']['file']
    code = {name: human.sha256(Path(__file__).with_name(name))
            for name in CODE_FILES}
    require(human.sha256(candidate_payload) == manifest['payload']['sha256']
            and manifest['base_payload_sha256'] == human.sha256(base_payload),
            'candidate source identity')
    prior = {case['case']: case for case in human.read_json(prior_receipt_path)['native_source_audits']}
    raw = candidate_payload.read_bytes()
    body_ids = np.frombuffer(raw, '<u4', count=nb*9, offset=60).reshape(nb, 9)[:, 0]
    weights = np.frombuffer(raw, '<f4', count=nv*nb,
                            offset=60+36*nb+56*nv+4*ni).reshape(nv, nb)
    rest_case = prior['raw-source-rest']
    rest = _native_mesh(Path(rest_case['inputs']['native_pack']['path']),
                        Path(rest_case['inputs']['native_poses']['path']),
                        fingerprint, nb, nv, ni, faces)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in POSES:
        baseline, base_pose, base_info = _read_run(heldout_root, label, 'baseline', prior,
                                                  base_payload, fingerprint, nb, nv, ni, faces)
        candidate, pose, candidate_info = _read_run(heldout_root, label, 'candidate', prior,
                                                    candidate_payload, fingerprint, nb, nv, ni, faces)
        require(base_info['pose_sha256'] == candidate_info['pose_sha256'],
                f'{label} native pose drift')
        indexed = {b['body_index']: b for b in pose['bodies']}
        predicted = np.zeros_like(rest)
        for binding, body_id in enumerate(body_ids):
            body = indexed[int(body_id)]
            a, c = body['rest'], body['current']
            pr = np.asarray(a['position_world_m'])
            pc = np.asarray(c['position_world_m'])
            rr = Rotation.from_quat(a['orientation_world_xyzw']).as_matrix()
            rc = Rotation.from_quat(c['orientation_world_xyzw']).as_matrix()
            predicted += weights[:, binding, None]*((rest-pr)@(rc@rr.T).T+pc)
        error = float(np.linalg.norm(predicted-candidate, axis=1).max())
        require(error <= 2e-6, f'{label} candidate native position oracle')
        baseline_qv, quotient_faces, _ = coordinate_quotient(baseline.tolist(), faces.tolist())
        candidate_qv, candidate_faces, _ = coordinate_quotient(candidate.tolist(), faces.tolist())
        require(candidate_faces == quotient_faces, f'{label} quotient face drift')
        base_exact = exact_embedding(baseline_qv, quotient_faces)
        candidate_exact = exact_embedding(candidate_qv, candidate_faces)
        base_pairs = {tuple(pair) for pair in base_exact['triangle_pairs']}
        candidate_pairs = {tuple(pair) for pair in candidate_exact['triangle_pairs']}
        new_pairs = sorted(candidate_pairs-base_pairs)
        removed_pairs = sorted(base_pairs-candidate_pairs)
        change = np.linalg.norm(candidate-baseline, axis=1)
        row = {
            'case': label, 'source_case': POSES[label][0], 'pose_q': POSES[label][1],
            'baseline': base_info, 'candidate': candidate_info,
            'baseline_exact_intersection_pairs': base_exact['count'],
            'baseline_intersecting_face_pairs': base_exact['triangle_pairs'],
            'candidate_exact_intersection_pairs': candidate_exact['count'],
            'candidate_intersecting_face_pairs': candidate_exact['triangle_pairs'],
            'new_intersecting_face_pairs': [list(pair) for pair in new_pairs],
            'removed_intersecting_face_pairs': [list(pair) for pair in removed_pairs],
            'candidate_topology': candidate_exact['topology'],
            'maximum_candidate_native_oracle_error_m': error,
            'maximum_position_change_m': float(change.max()),
            'native_device': 'Apple M4',
        }
        atomic_json(output_dir/(label+'.json'), row)
        rows.append(row)
        print(json.dumps({'case': label,
                          'baseline_intersections': row['baseline_exact_intersection_pairs'],
                          'candidate_intersections': row['candidate_exact_intersection_pairs'],
                          'maximum_position_change_mm': round(1000*change.max(), 3)}), flush=True)
    no_new = all(not row['new_intersecting_face_pairs'] for row in rows)
    require({name: human.sha256(Path(__file__).with_name(name))
             for name in CODE_FILES} == code, 'heldout predicate source drift')
    result = {
        'schema': SCHEMA,
        'status': 'passed_heldout_sampled_visual_nonregression' if no_new else 'failed_heldout_sampled_visual_nonregression',
        'base_payload_sha256': human.sha256(base_payload),
        'candidate_payload_sha256': human.sha256(candidate_payload),
        'prior_receipt_sha256': human.sha256(prior_receipt_path),
        'predicate_source_sha256': code,
        'heldout_case_count': len(rows),
        'baseline_intersection_pairs': sum(row['baseline_exact_intersection_pairs'] for row in rows),
        'candidate_intersection_pairs': sum(row['candidate_exact_intersection_pairs'] for row in rows),
        'new_intersection_pair_count': sum(len(row['new_intersecting_face_pairs']) for row in rows),
        'removed_intersection_pair_count': sum(len(row['removed_intersecting_face_pairs']) for row in rows),
        'rows_sha256': {label+'.json': human.sha256(output_dir/(label+'.json')) for label in POSES},
        'native_device': 'Apple M4', 'clinical_anatomy': False,
        'continuous_motion_qualified': False, 'physical_skin_mechanics': False,
        'boundary': 'Four retained held-out static source poses. No trajectory, loaded contact, material or clinical qualification.',
    }
    atomic_json(output_dir/'summary.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = human.REPOSITORY_ROOT
    parser.add_argument('--base-payload', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.nhskin')
    parser.add_argument('--base-manifest', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.manifest.json')
    parser.add_argument('--candidate-dir', type=Path, default=root/'Build/skin-embeddedness-20260930/native-candidate')
    parser.add_argument('--prior-receipt', type=Path, default=root/'Docs/media/skin-exact-seam-continuity-20260929/receipt.json')
    parser.add_argument('--heldout-root', type=Path, default=root/'Build/skin-embeddedness-20260930/heldout-native')
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    result = audit(a.base_payload, a.base_manifest, a.candidate_dir,
                   a.prior_receipt, a.heldout_root, a.output_dir)
    print(json.dumps({'status': result['status'],
                      'baseline_pairs': result['baseline_intersection_pairs'],
                      'candidate_pairs': result['candidate_intersection_pairs']}))
    return 0 if result['status'] == 'passed_heldout_sampled_visual_nonregression' else 2


if __name__ == '__main__':
    raise SystemExit(main())

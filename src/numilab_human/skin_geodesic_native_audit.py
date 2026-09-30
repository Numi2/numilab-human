"""Independently check derived ABI-5 skin weights against native M4 captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from . import model as human
from .compiled_quotient_embeddedness import classify_quotient
from .skin_embeddedness_gate import _native_mesh, _source_mesh
from .skin_geometry_diagnostics import source_seam_diagnostics, surface_face_diagnostics
from .skin_surface_audit import _edge_deformation_diagnostics
from .torso_anatomy_audit import _pack_sections
from .whole_body_embeddedness import atomic_json


SCHEMA = 'numi.human.derived-geodesic-skin-native-audit.v1'
CASES = ('raw-source-rest', 'neutral', 'coupled-torso', 'coupled-reach',
         'knee-flexion', 'shoulder-elevation', 'hip-flexion',
         'unilateral-reach', 'asymmetric-knee')
CODE_FILES = ('skin_geodesic_native_audit.py',
              'skin_geodesic_visual_candidate.py',
              'skin_embeddedness_gate.py',
              'compiled_quotient_embeddedness.py',
              'surface_topology_audit.py',
              'cardiac_cavity_intersections.py')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('geodesic skin native audit: ' + message)


def _native_normals(pack: Path, nv: int, ni: int):
    sections = _pack_sections(pack)
    vertices = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
    indices = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
    selected = primitives[primitives[:, 4] == 51007]
    require(len(selected) == 1 and int(selected[0, 1]) == ni,
            'native skin normal primitive')
    first = int(selected[0, 0])
    base = int(indices[first:first+ni].min())
    require(base+nv <= len(vertices), 'native skin normal vertices')
    normals = vertices[base:base+nv, 4:7].astype(float)
    require(bool(np.isfinite(normals).all()), 'native skin normals')
    return normals


def _case_files(directory: Path, case: dict, candidate_sha: str):
    name = case['case']
    folder = directory/f'native-{name}'
    command_file = folder/'command.json'
    require(command_file.is_file() and not command_file.is_symlink(),
            f'{name} native command')
    command = human.read_json(command_file)
    argv = command['argv']
    require(command['case'] == name and command['prior_command'] == case['command']
            and command['derived_skin_payload_sha256'] == candidate_sha
            and argv[:4] == case['command'][:4]
            and argv[4] == str(folder)
            and argv[5] == '--skin-payload'
            and argv[7:] == case['command'][7:]
            and human.sha256(Path(argv[0])) == command['native_probe_sha256']
            and human.sha256(Path(argv[1])) == command['rigid_sha256']
            and human.sha256(Path(argv[2])) == command['muscle_sha256']
            and human.sha256(Path(argv[3])) == command['bones_sha256'],
            f'{name} native input/command identity')
    require((folder/'exit.code').read_text().strip() == '0'
            and (folder/'stderr').read_bytes() == b'',
            f'{name} native command failed')
    stdout = (folder/'stdout').read_text()
    require('myosim_articulated_bodyparts_bone_visual=ok' in stdout
            and 'rendering_performed=true' in stdout
            and 'visual_coverage_qualified=true' in stdout
            and 'metal_pose_device="Apple M4"' in stdout
            and 'renderer_device="Apple M4"' in stdout
            and 'bodyparts_skin_shells=1' in stdout,
            f'{name} native M4 visual result')
    packs = list(folder.glob('*.mrvpack'))
    poses = list(folder.glob('*.skin-poses.json'))
    require(len(packs) == len(poses) == 1
            and human.sha256(poses[0]) == case['inputs']['native_poses']['sha256'],
            f'{name} native pose identity')
    images = sorted(folder.glob('*.png'))
    require(len(images) == 4 and all(p.stat().st_size > 0 for p in images),
            f'{name} four native views')
    return packs[0], poses[0], folder, images


def _replay_candidate_weights(base_vertices: np.ndarray, faces: np.ndarray,
                              body_ids: np.ndarray, baseline: bytes,
                              candidate: bytes, manifest: dict, base_manifest: dict,
                              solution_path: Path, attribution_path: Path,
                              nb: int, nv: int, ni: int) -> None:
    require(human.sha256(solution_path) == manifest['binding_solution_sha256']
            and human.sha256(attribution_path) == manifest['crossing_attribution_sha256']
            and manifest['blend_fraction'] == .05
            and manifest['source_geodesic_radius_m'] == .12,
            'candidate source derivation identity')
    with np.load(solution_path) as archive:
        source_weights = archive['full_weights']
        targets = archive['seed_targets']
    full_start = 60+36*nb+56*nv+4*ni
    baseline_weights = np.frombuffer(baseline, '<f4', count=nv*nb,
                                     offset=full_start).reshape(nv, nb)
    candidate_weights = np.frombuffer(candidate, '<f4', count=nv*nb,
                                      offset=full_start).reshape(nv, nb)
    require(np.array_equal(source_weights.astype('<f4'), baseline_weights),
            'source harmonic full-field parity')
    points, inverse = np.unique(base_vertices, axis=0, return_inverse=True)
    quotient_faces = inverse[faces]
    edges = np.unique(np.sort(np.concatenate((quotient_faces[:, [0, 1]],
                                               quotient_faces[:, [1, 2]],
                                               quotient_faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    require(len(points) == 54663 and len(edges) == 163860
            and bool(np.isfinite(lengths).all() and (lengths > 0).all()),
            'candidate source graph')
    graph = coo_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(points), len(points))).tocsr()
    distance = np.empty((len(points), nb), dtype='<f4')
    for binding in range(nb):
        seeds = np.unique(inverse[targets[(targets[:, 3] == binding)
                                           & (targets[:, 4] == 1), 2]])
        require(len(seeds) > 0, f'candidate source seed {binding}')
        distance[:, binding] = dijkstra(graph, directed=False,
                                       indices=seeds, min_only=True)
    envelopes = {b['core_body_index']: b
                 for b in base_manifest['source']['registered_bone_envelopes']}
    diameter = np.asarray([
        np.linalg.norm(np.asarray(envelopes[int(i)]['maximum_world_m'])
                       -np.asarray(envelopes[int(i)]['minimum_world_m']))
        for i in body_ids])
    screened = source_weights*np.exp(-distance[inverse]/diameter)
    screened /= screened.sum(axis=1, keepdims=True)
    attribution = human.read_json(attribution_path)
    require(attribution['schema'] == 'numi.human.native-skin-crossing-attribution.v1'
            and attribution['total_exact_intersection_pairs'] == 42,
            'candidate exact support identity')
    support_faces = sorted({face for case in attribution['cases']
                            for pair in case['exact_intersection_pairs'] for face in pair})
    require(support_faces == manifest['support_face_ids'],
            'candidate source support faces')
    support = np.unique(quotient_faces[support_faces])
    support_distance = dijkstra(graph, directed=False,
                                indices=support, min_only=True)[inverse]
    u = np.clip(support_distance/.12, 0, 1)
    locality = 1-(3*u*u-2*u*u*u)
    expected = source_weights+.05*locality[:, None]*(screened-source_weights)
    require(np.array_equal(expected.astype('<f4'), candidate_weights),
            'candidate full-field weight law')
    vertex_start = 60+36*nb
    for vertex in range(nv):
        quartet = np.argsort(-expected[vertex], kind='stable')[:4]
        selected = expected[vertex, quartet]
        selected /= selected.sum()
        stored = struct.unpack_from('<4I4f', candidate,
                                    vertex_start+56*vertex+24)
        require(list(quartet) == list(stored[:4])
                and float(np.max(np.abs(selected-np.asarray(stored[4:])))) < 1e-7,
                f'candidate diagnostic quartet {vertex}')


def audit(base_payload: Path, base_manifest_path: Path, candidate_dir: Path,
          prior_receipt_path: Path, solution_path: Path,
          attribution_path: Path, output_dir: Path) -> dict:
    base_payload, base_manifest_path, candidate_dir, prior_receipt_path, solution_path, attribution_path, output_dir = (
        Path(p).resolve() for p in (base_payload, base_manifest_path,
                                    candidate_dir, prior_receipt_path,
                                    solution_path, attribution_path, output_dir))
    base_manifest = human.read_json(base_manifest_path)
    nb, nv, ni, fingerprint, base_vertices, faces = _source_mesh(base_payload, base_manifest)
    manifest_path = candidate_dir/'candidate.manifest.json'
    manifest = human.read_json(manifest_path)
    candidate_path = candidate_dir/manifest['payload']['file']
    require(manifest['schema'] == 'numi.human.derived-geodesic-skin-visual-candidate.v1'
            and manifest['base_payload_sha256'] == human.sha256(base_payload)
            and manifest['base_manifest_sha256'] == human.sha256(base_manifest_path)
            and human.sha256(candidate_path) == manifest['payload']['sha256'],
            'derived payload identity')
    cnb, cnv, cni, cfp, candidate_vertices, candidate_faces = _source_mesh(candidate_path, manifest)
    require((cnb, cnv, cni, cfp) == (nb, nv, ni, fingerprint)
            and np.array_equal(candidate_vertices, base_vertices)
            and np.array_equal(candidate_faces, faces),
            'derived source skin changed')
    baseline, candidate = base_payload.read_bytes(), candidate_path.read_bytes()
    vertex_start = 60+36*nb
    full_start = vertex_start+56*nv+4*ni
    require(len(baseline) == len(candidate)
            and baseline[:vertex_start] == candidate[:vertex_start]
            and baseline[vertex_start+56*nv:full_start] == candidate[vertex_start+56*nv:full_start]
            and all(baseline[vertex_start+56*i:vertex_start+56*i+24]
                    == candidate[vertex_start+56*i:vertex_start+56*i+24]
                    for i in range(nv)),
            'derived payload moved source, normals, bindings or faces')
    candidate_weights = np.frombuffer(candidate, '<f4', count=nv*nb,
                                      offset=full_start).reshape(nv, nb)
    source_normals = np.ndarray((nv, 3), dtype='<f4', buffer=candidate,
                                offset=vertex_start+12, strides=(56, 4)).astype(float)
    body_ids = np.frombuffer(candidate, '<u4', count=nb*9, offset=60).reshape(nb, 9)[:, 0]
    require(bool(np.isfinite(candidate_weights).all())
            and candidate_weights.min() >= 0
            and float(np.max(np.abs(candidate_weights.sum(axis=1)-1))) < 1e-6,
            'native candidate full-field weights')
    _replay_candidate_weights(base_vertices, faces, body_ids, baseline, candidate,
                              manifest, base_manifest, solution_path,
                              attribution_path, nb, nv, ni)
    prior = human.read_json(prior_receipt_path)
    cases = {row['case']: row for row in prior['native_source_audits']}
    require(set(cases) == set(CASES), 'prior native case scope')
    code = {name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rest_case = cases['raw-source-rest']
    base_rest = _native_mesh(Path(base_rest_case['inputs']['native_pack']['path']),
                             Path(base_rest_case['inputs']['native_poses']['path']),
                             fingerprint, nb, nv, ni, faces)
    rows = []
    for name in CASES:
        case = cases[name]
        pack, pose_path, folder, images = _case_files(candidate_dir, case,
                                                       manifest['payload']['sha256'])
        actual = _native_mesh(pack, pose_path, fingerprint, nb, nv, ni, faces)
        pose = human.read_json(pose_path)
        by_body = {body['body_index']: body for body in pose['bodies']}
        require(set(map(int, body_ids)) == set(by_body),
                f'{name} native body coverage')
        predicted = np.zeros_like(base_rest)
        normal_prediction = np.zeros_like(base_rest)
        for binding, body_id in enumerate(body_ids):
            body = by_body[int(body_id)]
            a, c = body['rest'], body['current']
            pr = np.asarray(a['position_world_m'])
            pc = np.asarray(c['position_world_m'])
            rr = Rotation.from_quat(a['orientation_world_xyzw']).as_matrix()
            rc = Rotation.from_quat(c['orientation_world_xyzw']).as_matrix()
            rotation = rc @ rr.T
            predicted += candidate_weights[:, binding, None] * ((base_rest-pr)@rotation.T+pc)
            normal_prediction += candidate_weights[:, binding, None]*(source_normals@rotation.T)
        normal_prediction /= np.linalg.norm(normal_prediction, axis=1)[:, None]
        native_normals = _native_normals(pack, nv, ni)
        position_error = float(np.linalg.norm(predicted-actual, axis=1).max())
        normal_error = float(np.linalg.norm(normal_prediction-native_normals, axis=1).max())
        require(position_error <= 2e-6 and normal_error <= 2e-5,
                f'{name} native visual candidate oracle')
        exact = classify_quotient(actual.tolist(), faces.tolist())
        baseline_pack = Path(case['inputs']['native_pack']['path'])
        require(human.sha256(baseline_pack) == case['inputs']['native_pack']['sha256'],
                f'{name} baseline native pack identity')
        original = _native_mesh(baseline_pack,
                                Path(case['inputs']['native_poses']['path']),
                                fingerprint, nb, nv, ni, faces)
        position_change = np.linalg.norm(actual-original, axis=1)
        row = {
            'case': name, 'native_pack_sha256': human.sha256(pack),
            'native_pose_sha256': human.sha256(pose_path),
            'native_command_sha256': human.sha256(folder/'command.json'),
            'native_stdout_sha256': human.sha256(folder/'stdout'),
            'native_stderr_sha256': human.sha256(folder/'stderr'),
            'native_images_sha256': {p.name: human.sha256(p) for p in images},
            'metal_pose_device': 'Apple M4', 'renderer_device': 'Apple M4',
            'maximum_native_pose_oracle_position_error_m': position_error,
            'maximum_native_pose_oracle_normal_error': normal_error,
            'maximum_position_change_from_baseline_m': float(position_change.max()),
            'position_change_p99_m': float(np.quantile(position_change, .99)),
            'seams': source_seam_diagnostics(base_rest, actual),
            'edges': _edge_deformation_diagnostics(base_rest, faces, actual),
            'faces': surface_face_diagnostics(base_rest, faces, actual, native_normals),
            'exact': exact,
        }
        atomic_json(output_dir/(name+'.json'), row)
        rows.append(row)
        print(json.dumps({'case': name,
                          'native_exact_intersection_pairs': exact['exact_intersection_pairs'],
                          'maximum_position_change_mm': round(1000*position_change.max(), 3),
                          'oracle_error_um': round(position_error*1e6, 3)}), flush=True)
    require({name: human.sha256(Path(__file__).with_name(name))
             for name in CODE_FILES} == code, 'audit source drift')
    clear = sum(row['exact']['exact_intersection_pairs'] == 0 for row in rows)
    result = {
        'schema': SCHEMA, 'status': ('passed_sampled_native_visual_embeddedness'
                                   if clear == len(CASES) else 'failed_sampled_native_visual_embeddedness'),
        'candidate_manifest_sha256': human.sha256(manifest_path),
        'candidate_payload_sha256': human.sha256(candidate_path),
        'base_payload_sha256': human.sha256(base_payload),
        'prior_native_receipt_sha256': human.sha256(prior_receipt_path),
        'binding_solution_sha256': human.sha256(solution_path),
        'crossing_attribution_sha256': human.sha256(attribution_path),
        'source_code_sha256': code,
        'native_pose_count': len(CASES),
        'exact_self_intersection_free_native_pose_count': clear,
        'source_boundary_edge_count': rows[0]['exact']['topology']['boundary_edge_count'],
        'maximum_position_change_from_baseline_m': max(row['maximum_position_change_from_baseline_m'] for row in rows),
        'maximum_native_pose_oracle_position_error_m': max(row['maximum_native_pose_oracle_position_error_m'] for row in rows),
        'rows_sha256': {name+'.json': human.sha256(output_dir/(name+'.json')) for name in CASES},
        'source_positions_normals_bindings_and_faces_byte_identical': True,
        'visual_candidate_only': True, 'closed_skin_volume': False,
        'physical_skin_mechanics': False, 'clinical_anatomy': False,
        'motion_trajectory_qualification': False,
        'boundary': ('Native Apple M4 visual pack and exact sampled pose skin intersections. '
                     'Open source boundaries remain; no continuous motion, contact, material, '
                     'fat, cartilage, loaded knee or clinical anatomy admission.'),
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
    parser.add_argument('--solution', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-skin-binding-solution.npz')
    parser.add_argument('--attribution', type=Path, default=root/'Build/skin-embeddedness-20260930/crossing-attribution.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    result = audit(a.base_payload, a.base_manifest, a.candidate_dir,
                   a.prior_receipt, a.solution, a.attribution, a.output_dir)
    print(json.dumps({'status': result['status'],
                      'clear_native_poses': result['exact_self_intersection_free_native_pose_count']}))
    return 0 if result['status'] == 'passed_sampled_native_visual_embeddedness' else 2


if __name__ == '__main__':
    raise SystemExit(main())

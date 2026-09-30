"""Source-bound attribution of exact crossings in sampled native skin poses.

This is a diagnostic of visual linear-blend weights and body motion. It does
not alter skin geometry or infer physical skin mechanics or clinical anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from . import model as human
from .compiled_quotient_embeddedness import coordinate_quotient
from .skin_embeddedness_gate import _native_mesh, _source_mesh
from .surface_topology_audit import exact_embedding
from .whole_body_embeddedness import atomic_json


SCHEMA = 'numi.human.native-skin-crossing-attribution.v1'
CODE_FILES = ('skin_crossing_attribution.py', 'skin_embeddedness_gate.py',
              'compiled_quotient_embeddedness.py', 'surface_topology_audit.py',
              'cardiac_cavity_intersections.py')
EXPECTED_FAILURES = {'coupled-reach': 3, 'knee-flexion': 18,
                     'asymmetric-knee': 21}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('skin crossing attribution: ' + message)


def _source_graph_distances(vertices: np.ndarray, faces: np.ndarray,
                            targets: np.ndarray, binding_count: int):
    points, inverse = np.unique(vertices, axis=0, return_inverse=True)
    quotient_faces = inverse[faces]
    edges = np.unique(np.sort(np.concatenate((quotient_faces[:, [0, 1]],
                                               quotient_faces[:, [1, 2]],
                                               quotient_faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    require(len(points) == 54663 and len(edges) == 163860
            and bool(np.isfinite(lengths).all() and (lengths > 0).all()),
            'source quotient graph')
    graph = coo_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(points), len(points))).tocsr()
    distances = np.empty((len(points), binding_count), dtype='<f4')
    for owner in range(binding_count):
        seeds = np.unique(inverse[targets[(targets[:, 3] == owner)
                                           & (targets[:, 4] == 1), 2]])
        require(len(seeds) > 0, f'no admitted source skin seeds for binding {owner}')
        distances[:, owner] = dijkstra(graph, directed=False,
                                       indices=seeds, min_only=True)
    require(bool(np.isfinite(distances).all()), 'disconnected source seed graph')
    return inverse, distances


def _native_case(case: dict, fingerprint: int, nb: int, nv: int, ni: int,
                 faces: np.ndarray):
    pack, pose = (Path(case['inputs'][key]['path']) for key in
                  ('native_pack', 'native_poses'))
    for key, path in (('native_pack', pack), ('native_poses', pose)):
        require(path.is_file() and not path.is_symlink()
                and human.sha256(path) == case['inputs'][key]['sha256'],
                f"{case['case']} {key} source hash")
    return (_native_mesh(pack, pose, fingerprint, nb, nv, ni, faces),
            human.read_json(pose))


def audit(payload: Path, manifest_path: Path, solution_path: Path,
          source_audit_receipt: Path, exact_gate_dir: Path,
          core_manifest_path: Path, output: Path) -> dict:
    paths = [Path(p).resolve() for p in (payload, manifest_path, solution_path,
             source_audit_receipt, core_manifest_path)]
    payload, manifest_path, solution_path, source_audit_receipt, core_manifest_path = paths
    require(all(p.is_file() and not p.is_symlink() for p in paths),
            'retained source input missing')
    manifest = human.read_json(manifest_path)
    nb, nv, ni, fingerprint, source_vertices, faces = _source_mesh(payload, manifest)
    require(manifest['coverage']['binding_solution']['sha256'] == human.sha256(solution_path),
            'full-field source solution hash')
    with np.load(solution_path) as archive:
        full = archive['full_weights']
        targets = archive['seed_targets']
    require(full.shape == (nv, nb) and targets.shape[1] == 5
            and bool(np.isfinite(full).all()), 'source binding solution shape')
    raw = payload.read_bytes()
    binding_body_ids = np.frombuffer(raw, '<u4', count=nb*9, offset=60).reshape(nb, 9)[:, 0]
    payload_weights = np.frombuffer(raw, '<f4', count=nv*nb,
                                    offset=60+36*nb+56*nv+4*ni).reshape(nv, nb)
    require(np.array_equal(full.astype('<f4'), payload_weights)
            and float(np.max(np.abs(payload_weights.sum(axis=1)-1))) < 1e-6,
            'native full-field weight parity')
    core = human.read_json(core_manifest_path)
    names = core['core_tree']['body_order']
    require(len(names) == 157 and len(set(map(int, binding_body_ids))) == nb
            and max(binding_body_ids) < len(names), 'source core body names')
    source_receipt = human.read_json(source_audit_receipt)
    require(source_receipt['schema'] == 'numi.human.skin-exact-seam-evidence.v1',
            'source/native receipt schema')
    cases = {case['case']: case for case in source_receipt['native_source_audits']}
    require(set(EXPECTED_FAILURES).issubset(cases)
            and all(case['inputs']['payload']['sha256'] == human.sha256(payload)
                    and case['inputs']['manifest']['sha256'] == human.sha256(manifest_path)
                    for case in cases.values()), 'prior native source identity')
    exact_gate_dir = Path(exact_gate_dir).resolve()
    exact_summary = human.read_json(exact_gate_dir/'summary.json')
    require(exact_summary['schema'] == 'numi.human.compiled-native-skin-embeddedness.v1'
            and exact_summary['sampled_state_count'] == 10
            and exact_summary['source_boundary_edge_count'] == 171,
            'exact gate identity')
    for name, digest in exact_summary['row_files_sha256'].items():
        require(human.sha256(exact_gate_dir/'rows'/name) == digest,
                f'exact gate row hash {name}')
    require({name: human.read_json(exact_gate_dir/'rows'/(name+'.json'))['exact_intersection_pairs']
             for name in EXPECTED_FAILURES} == EXPECTED_FAILURES,
            'expected crossing scope')

    inverse, distance = _source_graph_distances(source_vertices, faces, targets, nb)
    rest, _ = _native_case(cases['raw-source-rest'], fingerprint, nb, nv, ni, faces)
    source_hashes = {name: human.sha256(Path(__file__).with_name(name))
                     for name in CODE_FILES}
    records = []
    for name, expected_count in EXPECTED_FAILURES.items():
        native, pose = _native_case(cases[name], fingerprint, nb, nv, ni, faces)
        quotient_vertices, quotient_faces, _ = coordinate_quotient(native.tolist(), faces.tolist())
        exact = exact_embedding(quotient_vertices, quotient_faces)
        require(exact['count'] == expected_count, f'{name} exact crossing replay')
        row = human.read_json(exact_gate_dir/'rows'/(name+'.json'))
        require(exact['triangle_pairs'][:16] == row['first_intersecting_face_pairs'],
                f'{name} first exact witnesses')
        pair_faces = sorted({face for pair in exact['triangle_pairs'] for face in pair})
        witness_ids = np.unique(faces[pair_faces])
        pose_by_body = {b['body_index']: b for b in pose['bodies']}
        require(set(map(int, binding_body_ids)) == set(pose_by_body),
                f'{name} native body coverage')
        predicted = np.zeros_like(rest)
        body_data = []
        for binding, core_id in enumerate(binding_body_ids):
            body = pose_by_body[int(core_id)]
            a, c = body['rest'], body['current']
            rest_position = np.asarray(a['position_world_m'])
            current_position = np.asarray(c['position_world_m'])
            rest_rotation = Rotation.from_quat(a['orientation_world_xyzw']).as_matrix()
            current_rotation = Rotation.from_quat(c['orientation_world_xyzw']).as_matrix()
            transformed = (rest-rest_position) @ (current_rotation @ rest_rotation.T).T + current_position
            predicted += payload_weights[:, binding, None]*transformed
            motion = transformed[witness_ids]-rest[witness_ids]
            weights = payload_weights[witness_ids, binding]
            body_data.append({
                'binding_index': binding, 'core_body_index': int(core_id),
                'source_body_name': names[int(core_id)],
                'mean_skin_weight': float(weights.mean()),
                'maximum_skin_weight': float(weights.max()),
                'mean_geodesic_distance_to_admitted_seed_m':
                    float(distance[inverse[witness_ids], binding].mean()),
                'mean_body_motion_m': float(np.linalg.norm(motion, axis=1).mean()),
                'mean_weighted_body_motion_m':
                    float(np.mean(weights*np.linalg.norm(motion, axis=1))),
                'mean_weighted_motion_vector_m':
                    np.mean(weights[:, None]*motion, axis=0).tolist(),
            })
        oracle_error = float(np.linalg.norm(predicted-native, axis=1).max())
        require(oracle_error <= 2e-6, f'{name} native motion oracle {oracle_error}')
        body_data.sort(key=lambda b: -b['mean_weighted_body_motion_m'])
        native_points = native[witness_ids]
        rest_points = rest[witness_ids]
        records.append({
            'case': name, 'native_pack_sha256': cases[name]['inputs']['native_pack']['sha256'],
            'native_pose_sha256': cases[name]['inputs']['native_poses']['sha256'],
            'exact_intersection_pairs': exact['triangle_pairs'],
            'exact_intersection_pair_count': exact['count'],
            'exact_aabb_candidate_pairs': exact['aabb_candidate_pairs'],
            'crossing_face_ids': pair_faces,
            'crossing_source_vertex_ids': witness_ids.tolist(),
            'crossing_native_world_bbox_m': [native_points.min(axis=0).tolist(),
                                             native_points.max(axis=0).tolist()],
            'crossing_rest_world_bbox_m': [rest_points.min(axis=0).tolist(),
                                           rest_points.max(axis=0).tolist()],
            'native_pose_oracle_max_position_error_m': oracle_error,
            'binding_motion_attribution': body_data,
        })
        print(json.dumps({'case': name, 'exact_pairs': exact['count'],
                          'top_motion_bodies': [(b['source_body_name'], round(1000*b['mean_weighted_body_motion_m'], 3))
                                                for b in body_data[:5]]}), flush=True)
    require({name: human.sha256(Path(__file__).with_name(name))
             for name in CODE_FILES} == source_hashes, 'source code drift during audit')
    result = {
        'schema': SCHEMA, 'status': 'exact_crossings_attributed_not_repaired',
        'source': {
            'payload_sha256': human.sha256(payload),
            'manifest_sha256': human.sha256(manifest_path),
            'binding_solution_sha256': human.sha256(solution_path),
            'source_audit_receipt_sha256': human.sha256(source_audit_receipt),
            'exact_gate_summary_sha256': human.sha256(exact_gate_dir/'summary.json'),
            'core_manifest_sha256': human.sha256(core_manifest_path),
            'predicate_source_sha256': source_hashes,
            'source_skin_member_id': manifest['source']['skin']['member_id'],
            'source_skin_member_sha256': manifest['source']['skin']['member_sha256'],
        },
        'source_vertex_count': nv, 'source_triangle_count': ni//3,
        'source_quotient_vertex_count': int(len(distance)),
        'binding_count': nb,
        'binding_method': manifest['coverage']['source_surface_binding']['method'],
        'crossing_case_count': len(records),
        'total_exact_intersection_pairs': sum(r['exact_intersection_pair_count'] for r in records),
        'cases': records,
        'skin_volume_qualified': False, 'native_candidate_repair': False,
        'clinical_anatomy': False, 'physical_skin_mechanics': False,
        'boundary': ('Exact crossing face witnesses, full native weight parity, source graph '
                     'seed distance and native pose motion attribution. Correlation with distant '
                     'body weights does not itself prove a clinically correct replacement field.'),
    }
    atomic_json(Path(output).resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = human.REPOSITORY_ROOT
    parser.add_argument('--payload', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.nhskin')
    parser.add_argument('--manifest', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.manifest.json')
    parser.add_argument('--solution', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-skin-binding-solution.npz')
    parser.add_argument('--source-audit-receipt', type=Path, default=root/'Docs/media/skin-exact-seam-continuity-20260929/receipt.json')
    parser.add_argument('--exact-gate-dir', type=Path, default=root/'Docs/media/skin-native-embeddedness-20260929')
    parser.add_argument('--core-manifest', type=Path, default=root/'Build/myosim-fullbody/myosim-fullbody-reference.manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    result = audit(a.payload, a.manifest, a.solution, a.source_audit_receipt,
                   a.exact_gate_dir, a.core_manifest, a.output)
    print(json.dumps({'status': result['status'],
                      'total_exact_intersection_pairs': result['total_exact_intersection_pairs']}))


if __name__ == '__main__':
    main()

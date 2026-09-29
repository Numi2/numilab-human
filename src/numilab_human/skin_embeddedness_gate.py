"""Exact compiled/native skin self-intersection gate for retained inspection poses.

The source skin is open. A zero-intersection result is useful geometric evidence
but does not make it a closed skin volume, physical tissue, or motion-qualified
exterior. Each native row is bound to the prior source/native skin audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from . import model as human
from .compiled_quotient_embeddedness import classify_quotient
from .torso_anatomy_audit import _pack_sections
from .whole_body_embeddedness import atomic_json


ROOT = human.REPOSITORY_ROOT
SCHEMA = 'numi.human.compiled-native-skin-embeddedness.v1'
ROW_SCHEMA = 'numi.human.compiled-native-skin-embeddedness-row.v1'
CASES = ('raw-source-rest', 'neutral', 'coupled-torso', 'coupled-reach',
         'knee-flexion', 'shoulder-elevation', 'hip-flexion',
         'unilateral-reach', 'asymmetric-knee')
SOURCE_CODE = ('skin_embeddedness_gate.py', 'compiled_quotient_embeddedness.py',
               'whole_body_embeddedness.py', 'surface_topology_audit.py',
               'cardiac_cavity_intersections.py', 'cardiac_cavity_geometry.py')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('skin embeddedness gate: '+message)


def _path(value: str, root: Path, label: str) -> Path:
    path = Path(value).resolve()
    require(path.is_file() and not path.is_symlink() and path.is_relative_to(root),
            f'{label} not a retained local file')
    return path


def _source_mesh(payload: Path, manifest: dict):
    raw = payload.read_bytes()
    magic, abi, nb, nv, ni, fingerprint, source_sha = struct.unpack_from('<8s5I32s', raw)
    require(magic == b'NHSKIN1\0' and abi == 5 and nv == 54949 and ni == 109183*3
            and nb == 86 and len(raw) == 60+36*nb+56*nv+4*ni+4*nv*nb
            and manifest['payload']['sha256'] == human.sha256(payload)
            and manifest['payload']['payload_abi'] == abi
            and manifest['source']['skin']['member_id'] == 'FJ2810',
            'compiled skin payload identity')
    offset = 60+36*nb
    vertices = np.ndarray((nv, 3), dtype='<f4', buffer=raw,
                          offset=offset, strides=(56, 4)).astype(np.float64)
    faces = np.frombuffer(raw, dtype='<u4', count=ni,
                          offset=offset+56*nv).reshape(-1, 3)
    require(bool(np.isfinite(vertices).all() and (faces < nv).all()),
            'compiled source geometry')
    return nb, nv, ni, fingerprint, vertices, faces


def _native_mesh(pack: Path, pose: Path, fingerprint: int,
                 nb: int, nv: int, ni: int, source_faces: np.ndarray):
    snapshot = human.read_json(pose)
    require(snapshot['schema'] == 'numi.human.native-skin-pose-snapshot.v1'
            and snapshot['registration_fingerprint32'] == fingerprint
            and snapshot['payload_abi'] == 5
            and snapshot['binding_count'] == nb
            and snapshot['vertex_count'] == nv
            and len(snapshot['bodies']) == nb,
            'native pose/payload identity')
    sections = _pack_sections(pack)
    for kind, stride in ((2, 80), (3, 4), (4, 64), (5, 80)):
        require(kind in sections and sections[kind][2] == stride
                and len(sections[kind][0]) == sections[kind][1]*stride,
                'native visual pack layout')
    vertices = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
    indices = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
    selected = primitives[primitives[:, 4] == 51007]
    require(len(selected) == 1, 'native skin primitive identity')
    first, count, _, instance = map(int, selected[0, :4])
    require(count == ni and first+count <= len(indices)
            and int(selected[0, 5]) == 1
            and int(selected[0, 6]) == 0xffffffff,
            'native skin primitive range/owner')
    emitted_indices = indices[first:first+count]
    base = int(emitted_indices.min())
    require(base+nv <= len(vertices)
            and np.array_equal(emitted_indices-base, source_faces.ravel()),
            'native skin face topology drift')
    instances = np.frombuffer(sections[5][0], '<u4').reshape(-1, 20)
    require(instance < len(instances) and instances[instance, 9] == 0xffffffff
            and instances[instance, 10] == 0
            and np.array_equal(instances[instance, 12:16], selected[0, 4:8]),
            'native skin instance identity')
    positions = vertices[base:base+nv, :3].astype(np.float64)
    require(bool(np.isfinite(positions).all()), 'native skin positions')
    return positions


def _row(name: str, positions: np.ndarray, faces: np.ndarray,
         source_identity: dict) -> dict:
    checked = classify_quotient(positions.tolist(), faces.tolist())
    return {
        'schema': ROW_SCHEMA, 'case': name,
        'geometry_sha256': _geometry_sha256(positions, faces),
        'source_identity': source_identity,
        'vertex_count': len(positions), 'triangle_count': len(faces),
        'compiled_coordinate_quotient_vertex_count':
            checked['compiled_coordinate_quotient_vertex_count'],
        'exact_duplicate_vertex_count': checked['compiled_exact_duplicate_vertex_count'],
        'status': checked['status'], 'self_intersection': checked['self_intersection'],
        'aabb_candidate_pairs': checked['aabb_candidate_pairs'],
        'exact_intersection_pairs': checked['exact_intersection_pairs'],
        'first_intersecting_face_pairs': checked['first_intersecting_face_pairs'],
        'topology': checked['topology'],
        'closed_embedded_surface_candidate': checked['closed_embedded_surface_candidate'],
        'clinical_anatomy': False, 'physical_skin_volume': False,
    }


def _geometry_sha256(positions: np.ndarray, faces: np.ndarray) -> str:
    return hashlib.sha256(positions.astype('<f4').tobytes()
                          + faces.astype('<u4').tobytes()).hexdigest()


def audit(payload: Path, manifest: Path, prior_receipt: Path,
          output_dir: Path) -> dict:
    payload, manifest, prior_receipt, output_dir = (
        Path(p).resolve() for p in (payload, manifest, prior_receipt, output_dir))
    require(payload.is_file() and manifest.is_file() and prior_receipt.is_file(),
            'retained input missing')
    declared = human.read_json(manifest)
    prior = human.read_json(prior_receipt)
    require(prior['schema'] == 'numi.human.skin-exact-seam-evidence.v1'
            and len(prior['native_source_audits']) == len(CASES)
            and {case['case'] for case in prior['native_source_audits']} == set(CASES),
            'prior native audit scope')
    nb, nv, ni, fingerprint, source_positions, faces = _source_mesh(payload, declared)
    require(all(case['passed'] is True
                and case['inputs']['payload']['sha256'] == human.sha256(payload)
                and case['inputs']['manifest']['sha256'] == human.sha256(manifest)
                for case in prior['native_source_audits']),
            'prior source/native admission')
    code = {name: human.sha256(Path(__file__).with_name(name)) for name in SOURCE_CODE}
    identity = {
        'schema': SCHEMA, 'payload_sha256': human.sha256(payload),
        'manifest_sha256': human.sha256(manifest),
        'prior_native_audit_receipt_sha256': human.sha256(prior_receipt),
        'predicate_source_sha256': code,
        'source_member_id': 'FJ2810',
        'source_member_sha256': declared['source']['skin']['member_sha256'],
        'expected_cases': list(CASES),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir/'identity.json'
    if identity_path.exists():
        require(human.read_json(identity_path) == identity, 'resume identity changed')
    else:
        atomic_json(identity_path, identity)
    rows_dir = output_dir/'rows'
    rows_dir.mkdir(exist_ok=True)

    source_row = rows_dir/'source-compiled.json'
    source_identity = {'payload_sha256': identity['payload_sha256'],
                       'predicate_source_sha256': code}
    if source_row.exists():
        row = human.read_json(source_row)
        require(row['schema'] == ROW_SCHEMA and row['case'] == 'source-compiled'
                and row['source_identity'] == source_identity
                and row['geometry_sha256'] == _geometry_sha256(source_positions, faces),
                'source resume identity')
    else:
        atomic_json(source_row, _row('source-compiled', source_positions, faces,
                                     source_identity))
    prior_cases = {case['case']: case for case in prior['native_source_audits']}
    for name in CASES:
        case = prior_cases[name]
        row_path = rows_dir/(name+'.json')
        audit_path = _path(str(prior_receipt.parent/case['after_audit_file']),
                           prior_receipt.parent, 'native source audit')
        require(human.sha256(audit_path) == case['after_audit_sha256'],
                f'native source audit hash for {name}')
        upstream = human.read_json(audit_path)
        require(upstream['passed'] is True and upstream['vertex_count'] == nv
                and upstream['triangle_count'] == ni//3
                and upstream['payload_abi'] == 5
                and upstream['source_member_sha256'] == identity['source_member_sha256']
                and upstream['maximum_native_world_vertex_error_m'] <= 2e-5,
                f'native source audit failed for {name}')
        for item in ('payload', 'manifest', 'native_pack', 'native_poses'):
            require(upstream['inputs'][item] == case['inputs'][item],
                    f'prior audit input drift for {name}: {item}')
        pack_info, pose_info = case['inputs']['native_pack'], case['inputs']['native_poses']
        pack = _path(pack_info['path'], ROOT/'Build/skin-seam-continuity-20260929',
                     f'native pack {name}')
        pose = _path(pose_info['path'], ROOT/'Build/skin-seam-continuity-20260929',
                     f'native pose {name}')
        require(human.sha256(pack) == pack_info['sha256']
                and human.sha256(pose) == pose_info['sha256'],
                f'native pack/pose hash for {name}')
        source_identity = {'native_pack_sha256': pack_info['sha256'],
                           'native_pose_sha256': pose_info['sha256'],
                           'source_audit_sha256': case['after_audit_sha256'],
                           'predicate_source_sha256': code}
        positions = _native_mesh(pack, pose, fingerprint, nb, nv, ni, faces)
        if row_path.exists():
            row = human.read_json(row_path)
            require(row['schema'] == ROW_SCHEMA and row['case'] == name
                    and row['source_identity'] == source_identity
                    and row['geometry_sha256'] == _geometry_sha256(positions, faces),
                    f'resume identity changed for {name}')
            continue
        row = _row(name, positions, faces, source_identity)
        atomic_json(row_path, row)
        print(json.dumps({'case': name, 'status': row['status'],
                          'exact_intersection_pairs': row['exact_intersection_pairs']}),
              flush=True)
    files = [source_row] + [rows_dir/(name+'.json') for name in CASES]
    rows = [human.read_json(path) for path in files]
    require(all(row['schema'] == ROW_SCHEMA and row['triangle_count'] == ni//3
                and row['self_intersection'] == 'exact_checked' for row in rows),
            'incomplete exact skin rows')
    require({name: human.sha256(Path(__file__).with_name(name))
             for name in SOURCE_CODE} == code, 'predicate source drift during scan')
    exact_clear = sum(row['exact_intersection_pairs'] == 0 for row in rows)
    closed = all(row['closed_embedded_surface_candidate'] for row in rows)
    result = {
        'schema': SCHEMA, 'identity_sha256': human.sha256(identity_path),
        'row_files_sha256': {path.name: human.sha256(path) for path in files},
        'sampled_state_count': len(rows), 'native_pose_count': len(CASES),
        'triangle_count_per_state': ni//3,
        'exact_self_intersection_free_state_count': exact_clear,
        'all_sampled_states_self_intersection_free': exact_clear == len(rows),
        'all_sampled_states_closed_embedded': closed,
        'source_boundary_edge_count': rows[0]['topology']['boundary_edge_count'],
        'source_vertex_link_defect_count': rows[0]['topology']['vertex_manifold_defect_ids'],
        'status': ('passed_sampled_closed_skin' if closed else
                   'failed_closed_skin_volume_gate'),
        'clinical_anatomy': False, 'physical_skin_volume': False,
        'skin_material_or_contact': False,
        'motion_trajectory_qualification': False,
        'boundary': ('Exact Float32 source and nine retained native pose snapshots only. '
                     'A zero self-intersection count does not close the open source '
                     'skin, certify intervening motion, clinical shape, skin material, '
                     'contact, mass or whole-body mechanics.'),
    }
    atomic_json(output_dir/'summary.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('payload', 'manifest', 'prior-receipt', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.payload, args.manifest, args.prior_receipt,
                   args.output_dir)
    print(json.dumps({key: result[key] for key in
                      ('status', 'sampled_state_count',
                       'exact_self_intersection_free_state_count',
                       'source_boundary_edge_count')}), flush=True)
    return 0 if result['status'] == 'passed_sampled_closed_skin' else 2


if __name__ == '__main__':
    raise SystemExit(main())

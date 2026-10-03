"""Independent source-only audit for the full-weight visual skin payload.

This gate verifies source provenance, the complete ABI 5 influence matrix,
exact source seams, source topology and rest-frame registration. It does not
require a native render and does not qualify posed skin mechanics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from . import model as human
from .skin_surface_audit import _rotation, _source_oracle, _verified_weights


SCHEMA = 'numi.human.skin-source-payload-independent-preflight.v1'
CODE_FILES = ('skin_source_payload_preflight.py', 'skin_surface_audit.py',
              'skin_surface_binding.py', 'model.py')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('skin source payload preflight: ' + message)


def decode_payload(raw: bytes) -> dict:
    """Decode the fixed ABI 5 payload layout, rejecting ambiguous/truncated data."""
    require(len(raw) >= 60, 'truncated payload header')
    magic, abi, binding_count, vertex_count, index_count, fingerprint, source_hash = \
        struct.unpack_from('<8s5I32s', raw)
    require(magic == b'NHSKIN1\0' and abi == 5, 'payload magic/ABI')
    expected_bytes = (60 + 36 * binding_count + 56 * vertex_count
                      + 4 * index_count + 4 * vertex_count * binding_count)
    require(len(raw) == expected_bytes, 'payload byte length')
    binding_offset = 60
    vertex_offset = binding_offset + 36 * binding_count
    index_offset = vertex_offset + 56 * vertex_count
    weight_offset = index_offset + 4 * index_count
    return {
        'abi': abi,
        'binding_count': binding_count,
        'vertex_count': vertex_count,
        'index_count': index_count,
        'registration_fingerprint32': fingerprint,
        'source_archive_sha256': source_hash.hex(),
        'bindings_f': np.frombuffer(raw, '<f4', count=9 * binding_count,
                                    offset=binding_offset).reshape(binding_count, 9),
        'bindings_u': np.frombuffer(raw, '<u4', count=9 * binding_count,
                                    offset=binding_offset).reshape(binding_count, 9),
        'vertices_f': np.frombuffer(raw, '<f4', count=14 * vertex_count,
                                    offset=vertex_offset).reshape(vertex_count, 14),
        'vertices_u': np.frombuffer(raw, '<u4', count=14 * vertex_count,
                                    offset=vertex_offset).reshape(vertex_count, 14),
        'indices': np.frombuffer(raw, '<u4', count=index_count,
                                 offset=index_offset),
        'full_weights': np.frombuffer(raw, '<f4', count=vertex_count * binding_count,
                                      offset=weight_offset).reshape(vertex_count, binding_count),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(human.REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def audit(sources: Path, artifact: Path, registration_path: Path,
          payload: Path, manifest_path: Path) -> dict:
    sources, artifact, registration_path, payload, manifest_path = (
        Path(path).resolve() for path in
        (sources, artifact, registration_path, payload, manifest_path))
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    manifest = human.read_json(manifest_path)
    require(manifest.get('schema') == 'numi.human.bodyparts3d-myosim-skinned-shell-visual-payload.v5'
            and manifest.get('status') ==
            'native_full_body_source_surface_linear_blend_skin_shell_visual_input_not_collision_or_physics',
            'payload type/ownership')
    require(manifest['payload']['file'] == payload.name
            and manifest['payload']['sha256'] == _sha(payload), 'payload manifest identity')
    require(manifest['source']['registration']['sha256'] ==
            hashlib.sha256(registration_bytes).hexdigest(), 'source registration identity')
    require(manifest['source']['bodyparts'] == registration['source']['bodyparts'],
            'BodyParts3D registration identity')
    for source_archive in registration['source']['bodyparts']['archives']:
        archive_path = sources / source_archive['file']
        require(_sha(archive_path) == source_archive['sha256'],
                'source BodyParts3D archive hash')
    _, skin_member, skin_obj = human._bodyparts_obj_member(sources, 'is_a', 'FJ2810')
    require(manifest['source']['skin']['member'] == skin_member
            and manifest['source']['skin']['member_sha256'] == hashlib.sha256(skin_obj).hexdigest(),
            'source skin member identity')

    reference, _ = human._bodyparts_runtime_bindings(registration, artifact)
    rigid_check = human._require_myosim_rigid_program(sources, artifact)
    rigid_path = artifact / 'myosim-fullbody-core-reference.nhrigid'
    reference_manifest_path = artifact / 'myosim-fullbody-reference.manifest.json'
    require(rigid_check['passed']
            and manifest['runtime_reference']['rigid']['sha256'] == _sha(rigid_path)
            and manifest['runtime_reference']['manifest']['sha256'] == _sha(reference_manifest_path)
            and manifest['runtime_reference']['source_archive_sha256'] ==
            registration['source']['myosim']['source']['archive_sha256'],
            'matched MyoSim runtime reference')

    method = manifest['coverage']['source_surface_binding']['method']
    oracle = _source_oracle(
        str(sources), str(artifact), registration_bytes, _sha(rigid_path),
        _sha(reference_manifest_path), skin_obj, method)
    source_v, faces, normals, world, core_ids, source_ids, subset, targets, gaps, raw_counts = oracle
    coverage = manifest['coverage']
    proof_declaration = coverage['binding_solution']
    proof_path = payload.parent / proof_declaration['file']
    require(proof_path.is_file() and proof_declaration['runtime_input'] is False
            and proof_declaration['sha256'] == _sha(proof_path),
            'offline source binding solution identity')
    quartet, local_weights, full_weights, certificate = _verified_weights(
        proof_path.read_bytes(), np.asarray(world, dtype='<f8').tobytes(),
        faces.astype('<u4').tobytes(), targets.tobytes(), gaps.tobytes(),
        len(core_ids), method)

    raw = payload.read_bytes()
    decoded = decode_payload(raw)
    nb, nv, ni = len(core_ids), len(source_v), int(faces.size)
    require(decoded['binding_count'] == nb and decoded['vertex_count'] == nv
            and decoded['index_count'] == ni, 'source payload dimensions')
    require(decoded['registration_fingerprint32'] ==
            int(hashlib.sha256(registration_bytes).hexdigest()[:8], 16)
            and decoded['source_archive_sha256'] ==
            registration['source']['myosim']['source']['archive_sha256'],
            'payload registration/source fingerprints')
    require(np.array_equal(decoded['bindings_u'][:, 0], core_ids),
            'source body binding order')
    require(np.array_equal(decoded['indices'], faces.ravel()), 'source triangle order')
    require(np.array_equal(decoded['vertices_f'][:, :3],
                           np.asarray(source_v * .001, dtype='<f4')),
            'source skin vertex coordinates')
    require(np.array_equal(decoded['vertices_u'][:, 6:10], quartet)
            and np.max(np.abs(decoded['vertices_f'][:, 10:] - local_weights)) <= 1e-6,
            'diagnostic influence quartets')
    require(np.array_equal(decoded['full_weights'], np.asarray(full_weights, dtype='<f4')),
            'complete ABI 5 runtime weights')
    require(bool(np.isfinite(decoded['bindings_f'][:, 1:]).all())
            and bool(np.isfinite(decoded['vertices_f'][:, :6]).all())
            and bool(np.isfinite(decoded['full_weights']).all())
            and float(decoded['full_weights'].min()) >= 0,
            'finite nonnegative payload values')
    require(float(np.max(np.abs(decoded['full_weights'].sum(axis=1) - 1))) <= 4e-7,
            'float32 runtime partition of unity')
    runtime_partition_error = float(np.max(
        np.abs(decoded['full_weights'].sum(axis=1) - 1)))
    full_solution_partition_error = float(np.max(
        np.abs(full_weights.sum(axis=1) - 1)))
    require(abs(coverage['maximum_float32_partition_error'] - runtime_partition_error) <= 1e-12,
            'declared float32 partition error')
    require(abs(coverage['source_surface_binding']['maximum_partition_unity_error']
                - full_solution_partition_error) <= 1e-12,
            'declared full solution partition error')

    import mujoco
    from myo_sim.build.compose import build_model
    model = build_model('myofullbody')
    rest = mujoco.MjData(model)
    mujoco.mj_forward(model, rest)
    matrix = np.asarray(registration['coordinate_system']['global_source_mm_to_myosim_world_m'])
    body_checks = []
    reconstructed = np.zeros_like(world)
    for binding_index, core_id in enumerate(core_ids):
        source_body = source_ids[int(core_id)]
        rest_rotation = rest.ximat[source_body].reshape(3, 3)
        translation_error = float(np.linalg.norm(
            decoded['bindings_f'][binding_index, 1:4]
            - rest_rotation.T @ (matrix[:3, 3] - rest.xipos[source_body])))
        declared_rotation = (_rotation(decoded['bindings_f'][binding_index, 4:8])
                             * decoded['bindings_f'][binding_index, 8])
        expected_rotation = rest_rotation.T @ (matrix[:3, :3] / .001)
        rotation_error = float(np.max(np.abs(declared_rotation - expected_rotation)))
        amount = full_weights[:, binding_index]
        reconstructed += amount[:, None] * ((world - rest.xipos[source_body])
                                               @ np.eye(3).T + rest.xipos[source_body])
        body_checks.append({'core_body_index': int(core_id), 'source_body_id': int(source_body),
                            'binding_translation_error_m': translation_error,
                            'binding_rotation_scale_error': rotation_error,
                            'passed': translation_error <= 1e-6 and rotation_error <= 1e-6})
    rest_error = float(np.max(np.linalg.norm(reconstructed - world, axis=1)))
    require(all(row['passed'] for row in body_checks), 'source rest-body binding transforms')
    require(rest_error <= 2e-12, 'source rest-pose reconstruction')
    require(bool(np.isfinite(normals).all()), 'source normal finiteness')

    declared_binding = coverage['source_surface_binding']
    for field in ('source_bone_count', 'source_graph_edge_count', 'source_graph_vertex_count',
                  'seed_candidate_count', 'seed_vertex_count', 'rejected_seed_count',
                  'exact_coincident_vertex_group_count',
                  'exact_coincident_redundant_vertex_count'):
        require(declared_binding[field] == certificate[field],
                'source binding certificate count ' + field)
    require(abs(declared_binding['maximum_relative_equation_residual']
                - certificate['maximum_relative_equation_residual']) <= 1e-10
            if 'maximum_relative_equation_residual' in declared_binding else
            certificate['maximum_relative_equation_residual'] <= 1e-10,
            'source binding equation residual')
    require(coverage['maximum_runtime_discarded_weight_mass'] == 0
            and coverage['influences_per_vertex'] == nb,
            'full-weight runtime ownership')

    code_hashes = {name: human.sha256(Path(__file__).with_name(name))
                   for name in CODE_FILES if name != 'model.py'}
    code_hashes['model.py'] = human.sha256(Path(human.__file__))
    source_program_checks = {key: value for key, value in rigid_check.items()
                             if key != 'file'}
    source_program_checks['file'] = _display(rigid_path)
    return {
        'schema': SCHEMA,
        'status': 'passed_source_provenance_weights_seams_and_rest_reconstruction',
        'inputs': {
            'payload': {'path': _display(payload), 'sha256': _sha(payload)},
            'manifest': {'path': _display(manifest_path), 'sha256': _sha(manifest_path)},
            'binding_solution': {'path': _display(proof_path), 'sha256': _sha(proof_path)},
            'registration': {'path': _display(registration_path), 'sha256': _sha(registration_path)},
            'rigid': {'path': _display(rigid_path), 'sha256': _sha(rigid_path)},
            'bodyparts3d_source_archive_sha256': manifest['source']['bodyparts']['archives'][0]['sha256'],
        },
        'source_surface': {
            'outer_vertices': len(source_v), 'triangles': len(faces),
            'edge_components_in_source_archive': subset['edge_component_count'],
            'raw_archive_vertices': raw_counts[0], 'raw_archive_triangles': raw_counts[1],
        },
        'binding': {
            key: certificate[key] for key in
            ('passed', 'method', 'maximum_relative_equation_residual', 'source_bone_count',
             'source_graph_edge_count', 'source_graph_vertex_count', 'seed_candidate_count',
             'seed_vertex_count', 'rejected_seed_count', 'exact_coincident_vertex_group_count',
             'exact_coincident_redundant_vertex_count', 'minimum_retained_four_weight_mass',
             'maximum_discarded_weight_mass', 'minimum_source_projection_gap_m',
             'maximum_source_projection_gap_m')
        },
        'runtime': {
            'payload_abi': 5, 'body_binding_count': nb,
            'full_weight_shape': list(full_weights.shape),
            'minimum_full_weight': float(full_weights.min()),
            'maximum_float32_partition_error': runtime_partition_error,
            'maximum_float64_solution_partition_error': full_solution_partition_error,
            'runtime_discarded_weight_mass': 0,
            'exact_seam_weight_mismatch': 0,
            'maximum_rest_reconstruction_error_m': rest_error,
            'body_checks_passed': len(body_checks),
        },
        'source_program_checks': source_program_checks,
        'audit_code_sha256': code_hashes,
        'boundary': ('Source visual-shell provenance, full-weight binding equations, exact seams and rest-frame reconstruction only. '
                     'No native render/pose, high-flex embeddedness, clinical skin weights, physical skin mechanics, contact or physiological qualification.'),
    }


def main() -> int:
    root = human.REPOSITORY_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    manifest = arguments.manifest or arguments.payload.with_name(
        'bodyparts3d-myosim-skinned-shell.manifest.json')
    result = audit(arguments.sources, arguments.artifact, arguments.registration,
                   arguments.payload, manifest)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'vertices': result['source_surface']['outer_vertices'],
                      'triangles': result['source_surface']['triangles'],
                      'bindings': result['runtime']['body_binding_count'],
                      'rest_error_m': result['runtime']['maximum_rest_reconstruction_error_m']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

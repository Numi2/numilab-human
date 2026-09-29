"""Verify the bounded EHL visual-projection change against the prior NHTISS4 pack.

This is an emitted-payload comparison, not a clinical enthesis or tendon-force
certificate. All records, bindings, indices, and skinning weights must agree;
only one named visual surface may change Float32 positions and normals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from . import model as human
from .compiled_quotient_embeddedness import classify_quotient

SCHEMA = 'numi.human.hallux-fp32-projection-delta.v1'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('hallux Float32 projection delta: ' + message)


def scan(baseline_payload: Path, baseline_manifest: Path,
         candidate_payload: Path, candidate_manifest: Path,
         output: Path) -> dict:
    import numpy as np

    paths = [Path(p).resolve() for p in (
        baseline_payload, baseline_manifest, candidate_payload, candidate_manifest, output
    )]
    baseline_payload, baseline_manifest, candidate_payload, candidate_manifest, output = paths
    old_doc, new_doc = (human.read_json(p) for p in (baseline_manifest, candidate_manifest))
    for payload, doc in ((baseline_payload, old_doc), (candidate_payload, new_doc)):
        descriptor = doc['payload']
        require(descriptor['file'] == payload.name
                and descriptor['sha256'] == human.sha256(payload)
                and descriptor['bytes'] == payload.stat().st_size,
                'payload/manifest identity')
    old_members = [(row['stable_id'], row['member_id'], row['member_sha256'])
                   for row in old_doc['source']['surfaces']]
    new_members = [(row['stable_id'], row['member_id'], row['member_sha256'])
                   for row in new_doc['source']['surfaces']]
    require(old_members == new_members and len(old_members) == 150
            and old_members[22][1] == 'FJ1408', 'named source-member identity')
    old_raw, new_raw = baseline_payload.read_bytes(), candidate_payload.read_bytes()
    require(old_raw[:64] == new_raw[:64], 'NHTISS4 header/frame changed')
    magic, abi, nr, nb, nv, ni, _, _ = struct.unpack_from('<8s6I32s', old_raw)
    require(magic == b'NHTISS4\0' and abi == 5 and nr == 150
            and len(old_raw) == len(new_raw)
            == 64 + 32*nr + 36*nb + 56*nv + 4*ni,
            'NHTISS4 ranges')
    vertex_offset = 64 + 32*nr + 36*nb
    index_offset = vertex_offset + 56*nv
    require(old_raw[64:vertex_offset] == new_raw[64:vertex_offset],
            'surface records or body bindings changed')
    require(old_raw[index_offset:] == new_raw[index_offset:],
            'compiled face indices changed')
    records = np.frombuffer(old_raw, dtype='<u4', count=nr*8, offset=64).reshape(-1, 8)
    old_vertices = np.frombuffer(old_raw, dtype='<u4', count=nv*14,
                                 offset=vertex_offset).reshape(-1, 14)
    new_vertices = np.frombuffer(new_raw, dtype='<u4', count=nv*14,
                                 offset=vertex_offset).reshape(-1, 14)
    changed = []
    for record in records:
        first, count, stable = int(record[2]), int(record[3]), int(record[6])
        if not np.array_equal(old_vertices[first:first+count],
                              new_vertices[first:first+count]):
            changed.append(stable)
    require(changed == [23], 'unexpected compiled visual surface changed')
    row = records[22]
    first, count, first_index, index_count = (int(row[i]) for i in (2, 3, 4, 5))
    old = old_vertices[first:first+count]
    new = new_vertices[first:first+count]
    require(np.array_equal(old[:, 6:], new[:, 6:]),
            'EHL skinning indices or weights changed')
    old_position = old[:, :3].view('<f4').astype(float)
    new_position = new[:, :3].view('<f4').astype(float)
    require(bool(np.isfinite(new_position).all()), 'nonfinite EHL position')
    deltas = np.linalg.norm(new_position-old_position, axis=1)
    max_delta = float(deltas.max())
    require(0.0 < max_delta <= 1.0e-5, 'EHL position retreat exceeded 10 micrometres')
    quantization = new_doc['source']['surfaces'][22][
        'toe_enthesis_weight_lock']['visual_enthesis_registration'][
            'compiled_fp32_face_area']
    require(quantization['status'] == 'bounded_source_segment_backoff'
            and quantization['degenerate_face_ids_before'] == [1475, 1477]
            and quantization['degenerate_face_ids_after'] == []
            and 0.0 < quantization['backoff_fraction'] <= 2.0**-12
            and 0.0 < quantization['max_retreat_m'] <= 1.0e-5,
            'EHL projection witness')
    require(all(
        surface.get('toe_enthesis_weight_lock', {}).get(
            'visual_enthesis_registration', {}).get(
                'compiled_fp32_face_area', {}).get('status')
        in (None, 'no_backoff_needed')
        for surface in new_doc['source']['surfaces'] if surface['stable_id'] != 23
    ), 'another visual source received a projection backoff')
    indices = np.frombuffer(new_raw, dtype='<u4', count=ni, offset=index_offset)
    faces = (indices[first_index:first_index+index_count].reshape(-1, 3)-first).tolist()
    old_status = classify_quotient(old_position.tolist(), faces)
    new_status = classify_quotient(new_position.tolist(), faces)
    require(old_status['status'] == 'exact_degenerate_face'
            and new_status['self_intersection'] == 'exact_checked'
            and new_status['topology']['degenerate_face_ids'] == 0,
            'compiled EHL face-area result')
    result = {
        'schema': SCHEMA,
        'baseline_payload_sha256': human.sha256(baseline_payload),
        'baseline_manifest_sha256': human.sha256(baseline_manifest),
        'candidate_payload_sha256': human.sha256(candidate_payload),
        'candidate_manifest_sha256': human.sha256(candidate_manifest),
        'compiler_source_sha256': human.sha256(Path(__file__).with_name('model.py')),
        'predicate_sha256': {
            name: human.sha256(Path(__file__).with_name(name))
            for name in ('hallux_fp32_projection_delta.py',
                         'compiled_quotient_embeddedness.py',
                         'whole_body_embeddedness.py',
                         'surface_topology_audit.py',
                         'cardiac_cavity_intersections.py',
                         'cardiac_cavity_geometry.py')
        },
        'stable_id': 23, 'member_id': 'FJ1408',
        'source_member_sha256': old_members[22][2],
        'changed_surface_ids': changed,
        'changed_position_vertex_count': int((old[:, :3] != new[:, :3]).any(axis=1).sum()),
        'changed_normal_vertex_count': int((old[:, 3:6] != new[:, 3:6]).any(axis=1).sum()),
        'max_compiled_position_delta_m': max_delta,
        'body_bindings_byte_identical': True,
        'face_indices_byte_identical': True,
        'skinning_indices_weights_byte_identical': True,
        'projection_quantization': quantization,
        'before': {'status': old_status['status'],
                   'self_intersection': old_status['self_intersection']},
        'after': {'status': new_status['status'],
                  'self_intersection': new_status['self_intersection'],
                  'boundary_edge_count': new_status['topology']['boundary_edge_count'],
                  'exact_intersection_pairs': new_status['exact_intersection_pairs']},
        'clinical_anatomy': False, 'mechanics': False, 'physical_volume': False,
        'boundary': 'This repairs only an emitted Float32 zero-area visual face. '
                    'The EHL source remains open/intersecting; bone enthesis, force transfer, '
                    'clinical placement, and standing are not qualified.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_result = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    if output.exists():
        require(output.read_bytes() == raw_result, 'existing receipt differs')
    else:
        output.write_bytes(raw_result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-payload', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--candidate-payload', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.baseline_payload, args.baseline_manifest,
                  args.candidate_payload, args.candidate_manifest, args.output)
    print(json.dumps({'changed_surface_ids': result['changed_surface_ids'],
                      'max_compiled_position_delta_m': result['max_compiled_position_delta_m'],
                      'before': result['before'], 'after': result['after']}, sort_keys=True))


if __name__ == '__main__':
    main()

"""Audit the source-bound calcaneal tendon visual delta against the prior build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'Build/ehl-fp32-backoff-20260929/full'
NEW = ROOT / 'Build/tendon-harmonic-boundary-20260929/full'
SCAN = ROOT / 'Docs/media/muscle-surface-embeddedness-20260929'
HERE = Path(__file__).resolve().parent
STEM = 'bodyparts3d-myosim-fullbody-muscle-surfaces'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pack(directory: Path):
    payload = directory / (STEM + '.nhtissue')
    manifest = directory / (STEM + '.manifest.json')
    raw = payload.read_bytes()
    declared = json.loads(manifest.read_text())
    header = struct.unpack_from('<8s6I32s', raw)
    magic, abi, count, bindings, vertices, indices, fingerprint, source_sha = header
    assert magic == b'NHTISS4\0' and abi == 5 and count == 150
    assert len(raw) == 64 + 32*count + 36*bindings + 56*vertices + 4*indices
    assert declared['payload']['sha256'] == sha(payload)
    assert declared['payload']['vertex_count'] == vertices
    assert declared['payload']['index_count'] == indices
    binding_start = 64 + 32*count
    vertex_start = binding_start + 36*bindings
    index_start = vertex_start + 56*vertices
    rows = []
    for index in range(count):
        record = struct.unpack_from('<8I', raw, 64 + 32*index)
        first_binding, binding_count, first_vertex, vertex_count, first_index, \
            index_count, stable_id, layer_code = record
        assert index_count % 3 == 0
        assert first_binding + binding_count <= bindings
        assert first_vertex + vertex_count <= vertices
        assert first_index + index_count <= indices
        vertex_bytes = raw[vertex_start+56*first_vertex:
                           vertex_start+56*(first_vertex+vertex_count)]
        absolute = struct.unpack_from('<' + 'I'*index_count, raw,
                                      index_start+4*first_index)
        assert all(first_vertex <= item < first_vertex+vertex_count for item in absolute)
        rows.append({
            'stable_id': stable_id, 'layer_code': layer_code,
            'binding_count': binding_count, 'vertex_count': vertex_count,
            'triangle_count': index_count//3, 'vertex_bytes': vertex_bytes,
            'local_indices': tuple(item-first_vertex for item in absolute),
        })
    return raw[binding_start:vertex_start], rows, declared, header, payload, manifest


def main() -> None:
    old_bindings, old_rows, old_manifest, old_header, old_payload, old_path = read_pack(OLD)
    new_bindings, new_rows, new_manifest, new_header, new_payload, new_path = read_pack(NEW)
    assert old_bindings == new_bindings
    assert old_header[:4] == new_header[:4]
    assert old_header[6:] == new_header[6:]
    old_members = [(row['stable_id'], row['member_id'], row['member_sha256'])
                   for row in old_manifest['source']['surfaces']]
    new_members = [(row['stable_id'], row['member_id'], row['member_sha256'])
                   for row in new_manifest['source']['surfaces']]
    assert old_members == new_members
    old_scan = json.loads((SCAN/'receipt-v2.json').read_text())
    new_scan = json.loads((SCAN/'receipt-v3.json').read_text())
    assert old_scan['source']['payload_sha256'] == sha(old_payload)
    assert new_scan['source']['payload_sha256'] == sha(new_payload)
    assert old_scan['source']['manifest_sha256'] == sha(old_path)
    assert new_scan['source']['manifest_sha256'] == sha(new_path)
    assert old_scan['source']['predicate_sha256'] == new_scan['source']['predicate_sha256']
    old_class = {row['stable_id']: row for row in old_scan['surfaces']}
    new_class = {row['stable_id']: row for row in new_scan['surfaces']}
    assert set(old_class) == set(new_class) == {row['stable_id'] for row in new_rows}

    changed = []
    tendons = []
    for old, new, provenance in zip(old_rows, new_rows,
                                    new_manifest['source']['surfaces'], strict=True):
        stable_id = old['stable_id']
        assert stable_id == new['stable_id'] == provenance['stable_id']
        assert old['layer_code'] == new['layer_code']
        assert old['binding_count'] == new['binding_count']
        if old['vertex_bytes'] != new['vertex_bytes'] or \
                old['local_indices'] != new['local_indices']:
            changed.append(stable_id)
        if stable_id not in (7, 8):
            assert old['vertex_bytes'] == new['vertex_bytes']
            assert old['local_indices'] == new['local_indices']
            assert old_class[stable_id]['exact_intersection_pairs'] \
                == new_class[stable_id]['exact_intersection_pairs']
            assert old_class[stable_id]['status'] == new_class[stable_id]['status']
            continue
        assert provenance['layer'] == 'tendon' and new['layer_code'] == 2
        assert new['vertex_count'] < old['vertex_count']
        assert new['triangle_count'] < old['triangle_count']
        assert new['local_indices'] == old['local_indices'][:len(new['local_indices'])]
        before, after = old_class[stable_id], new_class[stable_id]
        assert before['exact_intersection_pairs'] > 0
        assert after['exact_intersection_pairs'] == 0
        assert after['self_intersection_status'] == 'exact_checked'
        assert after['status'] == 'open_or_nonmanifold'
        presentation = provenance['secondary_attachment_weight_lock']['distal_boundary_presentation']
        assert presentation['generated_triangle_count'] == 0
        assert presentation['compiled_fp32_self_intersection_pairs'] == 0
        assert presentation['compiled_fp32_boundary_edge_count'] > 0
        tendons.append({
            'stable_id': stable_id, 'member_id': provenance['member_id'],
            'member_sha256': provenance['member_sha256'],
            'bone_member_id': provenance['secondary_attachment_weight_lock']['secondary_bone_member_id'],
            'old_exact_intersection_pairs': before['exact_intersection_pairs'],
            'new_exact_intersection_pairs': after['exact_intersection_pairs'],
            'old_triangle_count': old['triangle_count'],
            'new_triangle_count': new['triangle_count'],
            'removed_generated_strip_triangles': old['triangle_count']-new['triangle_count'],
            'old_vertex_count': old['vertex_count'],
            'new_vertex_count': new['vertex_count'],
            'open_boundary_edge_count': presentation['compiled_fp32_boundary_edge_count'],
            'source_face_normal_reversal_ids': presentation['source_face_normal_reversal_ids'],
            'max_source_edge_stretch_ratio': presentation['max_source_edge_stretch_ratio'],
            'p99_source_edge_stretch_ratio': presentation['p99_source_edge_stretch_ratio'],
            'source_boundary_to_named_bone_distance_upper_bound_m':
                presentation['source_boundary_to_named_bone_distance_upper_bound_m'],
        })
    assert changed == [7, 8]
    assert [row['old_exact_intersection_pairs'] for row in tendons] == [145, 163]
    assert sum(row['removed_generated_strip_triangles'] for row in tendons) == 130
    result = {
        'schema': 'numi.human.calcaneal-tendon-visual-delta.v1',
        'old_payload_sha256': sha(old_payload), 'new_payload_sha256': sha(new_payload),
        'old_manifest_sha256': sha(old_path), 'new_manifest_sha256': sha(new_path),
        'old_exact_scan_sha256': sha(SCAN/'receipt-v2.json'),
        'new_exact_scan_sha256': sha(SCAN/'receipt-v3.json'),
        'binding_bytes_identical': True,
        'source_member_id_and_sha256_identical': True,
        'unchanged_surface_count': 148,
        'changed_stable_ids': changed,
        'tendons': tendons,
        'old_status_counts': old_scan['coverage']['status_counts'],
        'new_status_counts': new_scan['coverage']['status_counts'],
        'boundary': (
            'The two visual sheets have zero detected exact compiled self-intersections, '
            'but remain open with local source-face normal reversals. This is not a '
            'closed tendon volume, clinically validated enthesis, load path, or standing result.'
        ),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/'receipt-v1.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'changed_stable_ids': changed,
                      'new_tendon_intersection_pairs': [row['new_exact_intersection_pairs']
                                                        for row in tendons],
                      'unchanged_surface_count': 148}))


if __name__ == '__main__':
    main()

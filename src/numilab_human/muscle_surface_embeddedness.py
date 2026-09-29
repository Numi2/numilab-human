"""Source-bound exact embeddedness gate for compiled NHTISS4 tissue surfaces.

An indexed visual mesh, even with a closed source OBJ, is not automatically a
single embedded tissue volume. This audit checks the executed Float32 position
quotient and retains the separate component and self-intersection decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

from . import model as human
from .compiled_quotient_embeddedness import classify_quotient

ROOT = human.REPOSITORY_ROOT
SOURCE_AUDIT = ROOT / 'Docs/media/muscle-surface-geometry-audit-20260914/receipt-v1.json'
SCHEMA = 'numi.human.compiled-muscle-surface-embeddedness.v1'
PREDICATES = ('muscle_surface_embeddedness.py', 'compiled_quotient_embeddedness.py',
              'whole_body_embeddedness.py', 'surface_topology_audit.py',
              'cardiac_cavity_intersections.py', 'cardiac_cavity_geometry.py')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('muscle surface embeddedness: ' + message)


def scan(payload: Path, manifest: Path, output: Path,
         source_audit: Path = SOURCE_AUDIT,
         baseline_payload: Path | None = None,
         baseline_manifest: Path | None = None) -> dict:
    import numpy as np

    payload, manifest, output, source_audit = (
        Path(path).resolve() for path in (payload, manifest, output, source_audit)
    )
    declared = human.read_json(manifest)
    audit = human.read_json(source_audit)
    require(declared.get('schema') ==
            'numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1',
            'manifest schema')
    require(audit.get('schema') == 'HumanPack.muscle-surface-geometry-audit.v1',
            'source audit schema')
    expected = declared['payload']
    require(expected['file'] == payload.name and expected['sha256'] == human.sha256(payload)
            and expected['bytes'] == payload.stat().st_size, 'payload manifest identity')
    source_rows = audit['surfaces']
    manifest_rows = declared['source']['surfaces']
    require(len(source_rows) == len(manifest_rows) == 150, 'complete source inventory')
    raw = payload.read_bytes()
    require(len(raw) >= 64, 'truncated NHTISS4 header')
    magic, abi, nrecords, nbindings, nvertices, nindices, fingerprint, source_sha = \
        struct.unpack_from('<8s6I32s', raw)
    require(magic == b'NHTISS4\0' and abi == 5 and nrecords == 150,
            'NHTISS4 header identity')
    require(len(raw) == 64 + 32*nrecords + 36*nbindings + 56*nvertices + 4*nindices,
            'NHTISS4 byte ranges')
    require(nrecords == expected['surface_count']
            and nbindings == expected['binding_count']
            and nvertices == expected['vertex_count']
            and nindices == expected['index_count']
            and f'{fingerprint:08x}' == expected['registration_fingerprint32']
            and source_sha.hex() == declared['source']['myosim_source_archive_sha256'],
            'NHTISS4 semantic header')
    records = np.frombuffer(raw, dtype='<u4', count=nrecords*8, offset=64).reshape(-1, 8)
    vertex_offset = 64 + 32*nrecords + 36*nbindings
    positions = np.ndarray((nvertices, 3), dtype='<f4', buffer=raw,
                           offset=vertex_offset, strides=(56, 4))
    index_offset = vertex_offset + 56*nvertices
    indices = np.frombuffer(raw, dtype='<u4', count=nindices, offset=index_offset)
    require(bool(np.isfinite(positions).all()), 'nonfinite compiled positions')
    require((baseline_payload is None) == (baseline_manifest is None),
            'baseline payload and manifest must be paired')
    baseline = None
    if baseline_payload is not None and baseline_manifest is not None:
        baseline_payload = Path(baseline_payload).resolve()
        baseline_manifest = Path(baseline_manifest).resolve()
        old_declared = human.read_json(baseline_manifest)
        old_expected = old_declared['payload']
        require(old_expected['file'] == baseline_payload.name
                and old_expected['sha256'] == human.sha256(baseline_payload)
                and old_declared['source']['surfaces']
                and [(row['stable_id'], row['member_id'], row['member_sha256'])
                     for row in old_declared['source']['surfaces']]
                == [(row['stable_id'], row['member_id'], row['member_sha256'])
                    for row in manifest_rows], 'baseline source identity')
        old_raw = baseline_payload.read_bytes()
        old_header = struct.unpack_from('<8s6I32s', old_raw)
        _, old_abi, old_nr, old_nb, old_nv, old_ni, old_fp, old_source_sha = old_header
        require(old_header[0] == magic and old_abi == abi and old_nr == nrecords
                and old_nb == nbindings and old_fp == fingerprint
                and old_source_sha == source_sha
                and len(old_raw) == 64 + 32*old_nr + 36*old_nb + 56*old_nv + 4*old_ni,
                'baseline ABI and frame identity')
        old_records = np.frombuffer(old_raw, dtype='<u4', count=old_nr*8,
                                    offset=64).reshape(-1, 8)
        old_vertex_offset = 64 + 32*old_nr + 36*old_nb
        old_positions = np.ndarray((old_nv, 3), dtype='<f4', buffer=old_raw,
                                   offset=old_vertex_offset, strides=(56, 4))
        old_indices = np.frombuffer(old_raw, dtype='<u4', count=old_ni,
                                    offset=old_vertex_offset + 56*old_nv)
        require(old_raw[64+32*old_nr:old_vertex_offset]
                == raw[64+32*nrecords:vertex_offset], 'baseline body bindings changed')
        require(old_nv == nvertices and
                old_raw[old_vertex_offset:old_vertex_offset+56*old_nv]
                == raw[vertex_offset:index_offset],
                'baseline vertex positions, normals or skinning weights changed')
        baseline = (old_records, old_positions, old_indices,
                    baseline_payload, baseline_manifest)

    rows = []
    for record_index, (source, mapped, record) in enumerate(
        zip(source_rows, manifest_rows, records, strict=True)
    ):
        first_binding, binding_count, first_vertex, vertex_count, first_index, \
            index_count, stable_id, layer_code = (int(x) for x in record)
        require(stable_id == source['stable_id'] == mapped['stable_id']
                and mapped['member_id'] == source['member_id']
                and mapped['member_sha256'] == source['member_sha256'],
                'source member identity')
        require((layer_code == 1 and mapped['layer'] == 'muscle')
                or (layer_code == 2 and mapped['layer'] == 'tendon'),
                'source layer identity')
        require(first_binding + binding_count <= nbindings
                and first_vertex + vertex_count <= nvertices
                and first_index + index_count <= nindices
                and index_count > 0 and index_count % 3 == 0
                and mapped['vertex_count'] == vertex_count
                and mapped['triangle_count'] == index_count//3,
                'compiled surface ranges')
        local_indices = indices[first_index:first_index+index_count]
        require(bool(((local_indices >= first_vertex)
                      & (local_indices < first_vertex + vertex_count)).all()),
                'surface indices escape record')
        cancellation = mapped.get('source_topology_cancellation')
        if cancellation is not None:
            pairs = cancellation['cancelled_opposite_face_pairs']
            require(cancellation['source_triangle_count'] - 2*len(pairs)
                    == cancellation['retained_triangle_count'] == index_count//3
                    and cancellation['retained_vertex_count'] <= vertex_count
                    and cancellation['emitted_vertex_count'] == vertex_count
                    and cancellation['emitted_triangle_count'] == index_count//3
                    and cancellation['emitted_vertices_compacted'] is False
                    and all(len(pair) == 2 and pair[0] != pair[1]
                            and all(type(i) is int and 0 <= i < cancellation['source_triangle_count']
                                    for i in pair) for pair in pairs)
                    and len({i for pair in pairs for i in pair}) == 2*len(pairs)
                    and cancellation['oriented_source_chain_preserved'] is True
                    and type(cancellation['removed_unique_visual_support_triangle_count']) is int
                    and 0 <= cancellation['removed_unique_visual_support_triangle_count'] <= len(pairs)
                    and math.isfinite(cancellation['removed_visual_support_area_mm2'])
                    and cancellation['removed_visual_support_area_mm2'] >= 0
                    and cancellation['source_coordinate_support_preserved']
                    == (cancellation['removed_unique_visual_support_triangle_count'] == 0)
                    and cancellation['new_faces_added'] is False
                    and cancellation['vertices_moved'] is False,
                    'source face cancellation provenance')
        local_positions = positions[first_vertex:first_vertex+vertex_count]
        local_faces = (local_indices-first_vertex).reshape(-1, 3)
        if baseline is not None:
            old_records, old_positions, old_indices, _, _ = baseline
            old_record = [int(x) for x in old_records[record_index]]
            old_first_vertex, old_first_index = old_record[2], old_record[4]
            old_index_count = old_record[5]
            require(old_record[6] == stable_id and old_record[7] == layer_code
                    and old_record[:2] == [first_binding, binding_count]
                    and old_index_count//3 - 2*(len(cancellation['cancelled_opposite_face_pairs'])
                                                  if cancellation is not None else 0)
                    == index_count//3, 'baseline record projection')
            old_face_ids = old_indices[old_first_index:old_first_index+old_index_count]
            require(bool(((old_face_ids >= old_first_vertex)
                          & (old_face_ids < old_first_vertex + old_record[3])).all()),
                    'baseline surface indices escape record')
            old_faces = old_face_ids.reshape(-1, 3)
            retained = np.ones(len(old_faces), dtype=bool)
            if cancellation is not None:
                retained[[i for pair in cancellation['cancelled_opposite_face_pairs']
                          for i in pair]] = False
            require(np.array_equal(old_positions[old_faces[retained]],
                                   positions[local_indices.reshape(-1, 3)]),
                    'retained compiled triangle positions changed')
        checked = classify_quotient(local_positions.astype(float).tolist(),
                                    local_faces.tolist())
        one_component = checked['topology']['face_component_count'] == 1
        candidate = checked['closed_embedded_surface_candidate'] and one_component
        vertex_bytes = raw[vertex_offset + 56*first_vertex:
                           vertex_offset + 56*(first_vertex+vertex_count)]
        index_bytes = raw[index_offset + 4*first_index:
                          index_offset + 4*(first_index+index_count)]
        rows.append({
            'stable_id': stable_id, 'member_id': source['member_id'],
            'member_sha256': source['member_sha256'], 'layer': mapped['layer'],
            'source_topology_status': source['status'],
            'source_opposite_face_pairs_cancelled': len(cancellation['cancelled_opposite_face_pairs'])
            if cancellation is not None else 0,
            'removed_unique_visual_support_triangle_count':
                cancellation['removed_unique_visual_support_triangle_count']
                if cancellation is not None else 0,
            'removed_visual_support_area_mm2':
                cancellation['removed_visual_support_area_mm2']
                if cancellation is not None else 0.0,
            'compiled_geometry_sha256': hashlib.sha256(vertex_bytes + index_bytes).hexdigest(),
            'compiled_vertex_count': vertex_count, 'compiled_triangle_count': index_count//3,
            'compiled_quotient_vertex_count': checked['compiled_coordinate_quotient_vertex_count'],
            'compiled_exact_duplicate_vertex_count': checked['compiled_exact_duplicate_vertex_count'],
            'topology_closed_oriented': checked['topology']['closed_oriented_manifold_candidate'],
            'face_component_count': checked['topology']['face_component_count'],
            'self_intersection_status': checked['self_intersection'],
            'exact_intersection_pairs': checked['exact_intersection_pairs'],
            'first_intersecting_face_pairs': checked['first_intersecting_face_pairs'],
            'status': checked['status'],
            'single_embedded_surface_candidate': candidate,
            'physical_volume_owner': False,
        })
    counts = Counter(row['status'] for row in rows)
    result = {
        'schema': SCHEMA,
        'source': {'payload': str(payload.relative_to(ROOT)) if payload.is_relative_to(ROOT) else str(payload),
                   'payload_sha256': expected['sha256'],
                   'manifest': str(manifest.relative_to(ROOT)) if manifest.is_relative_to(ROOT) else str(manifest),
                   'manifest_sha256': human.sha256(manifest),
                   'source_audit_sha256': human.sha256(source_audit),
                   'baseline_payload_sha256': human.sha256(baseline_payload)
                   if baseline_payload is not None else None,
                   'baseline_manifest_sha256': human.sha256(baseline_manifest)
                   if baseline_manifest is not None else None,
                   'predicate_sha256': {name: human.sha256(Path(__file__).with_name(name))
                                        for name in PREDICATES}},
        'coverage': {'surface_count': len(rows),
                     'muscle_surface_count': sum(row['layer'] == 'muscle' for row in rows),
                     'tendon_surface_count': sum(row['layer'] == 'tendon' for row in rows),
                     'cancelled_opposite_face_pair_count': sum(row['source_opposite_face_pairs_cancelled'] for row in rows),
                     'removed_unique_visual_support_triangle_count': sum(
                         row['removed_unique_visual_support_triangle_count'] for row in rows),
                     'removed_visual_support_area_mm2': sum(
                         row['removed_visual_support_area_mm2'] for row in rows),
                     'exact_predicate_surface_count': sum(row['self_intersection_status'] == 'exact_checked' for row in rows),
                     'status_counts': dict(sorted(counts.items())),
                     'single_embedded_muscle_surface_candidate_count': sum(
                         row['layer'] == 'muscle' and row['single_embedded_surface_candidate'] for row in rows),
                     'single_embedded_tendon_surface_candidate_count': sum(
                         row['layer'] == 'tendon' and row['single_embedded_surface_candidate'] for row in rows)},
        'surfaces': rows,
        'physical_volume_owner': False,
        'compiled_vertex_records_match_baseline':
            True if baseline is not None else None,
        'retained_compiled_triangle_positions_match_baseline':
            True if baseline is not None else None,
        'clinical_anatomy': False,
        'mechanics': False,
        'boundary': 'Exact compiled Float32 per-surface topology and self-intersection only. '
                    'This does not establish tissue volume, anatomical placement, cross-surface '
                    'disjointness, material, force transmission, or standing qualification.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    if output.exists():
        require(output.read_bytes() == serialized, 'existing receipt differs')
    else:
        output.write_bytes(serialized)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-payload', type=Path)
    parser.add_argument('--baseline-manifest', type=Path)
    args = parser.parse_args()
    result = scan(args.payload, args.manifest, args.output,
                  baseline_payload=args.baseline_payload,
                  baseline_manifest=args.baseline_manifest)
    print(json.dumps(result['coverage'], sort_keys=True))


if __name__ == '__main__':
    main()

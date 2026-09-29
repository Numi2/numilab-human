"""Verify the bilateral short-head visual repair against its exact prior packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from . import model as human


SCHEMA = 'numi.human.bilateral-biceps-short-head-visual-delta.v1'
TARGETS = {19: 'FJ1444', 20: 'FJ1444M'}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('biceps visual delta: '+message)


def audit(old_payload: Path, old_manifest: Path, old_census: Path,
          new_payload: Path, new_manifest: Path, new_census: Path,
          source_root: Path) -> dict:
    paths = tuple(Path(p).resolve() for p in (old_payload, old_manifest, old_census,
                                               new_payload, new_manifest, new_census,
                                               source_root))
    (old_payload, old_manifest, old_census, new_payload, new_manifest,
     new_census, source_root) = paths
    old_m, new_m = human.read_json(old_manifest), human.read_json(new_manifest)
    old_a, new_a = human.read_json(old_census), human.read_json(new_census)
    old, new = old_payload.read_bytes(), new_payload.read_bytes()
    require(old_m['payload']['sha256'] == human.sha256(old_payload)
            and new_m['payload']['sha256'] == human.sha256(new_payload)
            and old_a['source']['payload_sha256'] == old_m['payload']['sha256']
            and new_a['source']['payload_sha256'] == new_m['payload']['sha256']
            and old_a['coverage']['surface_count'] == new_a['coverage']['surface_count'] == 150,
            'payload and full-census identities')
    require(old_m['source']['registration'] == new_m['source']['registration']
            and old_m['source']['myosim_manifest'] == new_m['source']['myosim_manifest']
            and old_m['source']['surface_map'] == new_m['source']['surface_map']
            and old_m['source']['bodyparts'] == new_m['source']['bodyparts']
            and len(old_m['source']['surfaces']) == len(new_m['source']['surfaces']) == 150,
            'source and registration drift')
    require(len(old) == len(new) and old[:64] == new[:64], 'NHTISS4 header drift')
    magic, abi, nr, nb, nv, ni, _, _ = struct.unpack_from('<8s6I32s', old)
    require(magic == b'NHTISS4\0' and abi == 5 and nr == 150
            and len(old) == 64+32*nr+36*nb+56*nv+4*ni,
            'NHTISS4 byte ranges')
    vertex_start = 64+32*nr+36*nb
    index_start = vertex_start+56*nv
    require(old[:vertex_start] == new[:vertex_start]
            and old[index_start:] == new[index_start:],
            'record, binding or face-index drift')
    records = np.frombuffer(old, dtype='<u4', count=nr*8, offset=64).reshape(nr, 8)
    old_vertices = np.frombuffer(old, dtype=np.uint8, count=nv*56,
                                 offset=vertex_start).reshape(nv, 56)
    new_vertices = np.frombuffer(new, dtype=np.uint8, count=nv*56,
                                 offset=vertex_start).reshape(nv, 56)
    changed = np.flatnonzero(np.any(old_vertices != new_vertices, axis=1))
    changed_rows = []
    for sid, (a, b) in enumerate(zip(old_m['source']['surfaces'],
                                     new_m['source']['surfaces'], strict=True), 1):
        require(a['stable_id'] == b['stable_id'] == sid, 'ordered surface records')
        before, after = old_a['surfaces'][sid-1], new_a['surfaces'][sid-1]
        first, count = map(int, records[sid-1, 2:4])
        local = changed[(changed >= first) & (changed < first+count)]-first
        if sid not in TARGETS:
            require(a == b and before == after and len(local) == 0,
                    f'unrelated surface {sid} changed')
            continue
        require(a == {k: v for k, v in b.items() if k != 'source_visual_untangle'}
                and b['member_id'] == TARGETS[sid], 'target source provenance')
        proof = b['source_visual_untangle']
        require(proof['source_member_sha256'] == b['member_sha256']
                and proof['moved_source_vertex_ids']
                and proof['source_visual_geometry_modified'] is True
                and proof['compiled_fp32_self_intersection_pairs_before'] == 3
                and proof['compiled_fp32_self_intersection_pairs_after'] == 0,
                'target derived-geometry provenance')
        _, member, obj = human._bodyparts_obj_member(source_root, 'is_a', TARGETS[sid])
        archive = next(item for item in old_m['source']['bodyparts']['archives']
                       if item['hierarchy'] == 'is_a')
        require(human.sha256(source_root/archive['file']) == archive['sha256'],
                'source archive identity')
        require(hashlib.sha256(obj).hexdigest() == b['member_sha256'],
                'source member hash')
        original_mm, triangles = human._bodyparts_obj_triangles(obj, member)
        from .muscle_tip_visual_untangle import untangle
        expected_mm, rerun = untangle(original_mm, triangles,
                                      TARGETS[sid], b['member_sha256'])
        require(rerun == proof, 'derived candidate replay')
        expected_positions = np.asarray(expected_mm, dtype='<f8')*.001
        expected_positions = expected_positions.astype('<f4')
        position_start = vertex_start+56*first
        emitted = np.ndarray((count, 3), dtype='<f4', buffer=new,
                             offset=position_start, strides=(56, 4))
        require(np.array_equal(expected_positions, emitted),
                'emitted positions differ from pinned candidate')
        expected_normals = np.asarray(
            human._bodyparts_vertex_normals(expected_mm, triangles, member),
            dtype='<f4',
        )
        emitted_normals = np.ndarray((count, 3), dtype='<f4', buffer=new,
                                     offset=position_start+12, strides=(56, 4))
        require(np.array_equal(expected_normals, emitted_normals),
                'emitted normals differ from derived geometry')
        old_position = np.ndarray((count, 3), dtype='<f4', buffer=old,
                                  offset=position_start, strides=(56, 4))
        moved = np.flatnonzero(np.any(old_position != emitted, axis=1)).tolist()
        require(moved == proof['moved_source_vertex_ids']
                and np.array_equal(old_vertices[first:first+count, 24:],
                                   new_vertices[first:first+count, 24:])
                and len(local) == 16,
                'positions, normals or weights outside intended delta')
        require(before['status'] == 'closed_self_intersecting'
                and before['exact_intersection_pairs'] == 3
                and after['status'] == 'closed_embedded_source_candidate'
                and after['exact_intersection_pairs'] == 0
                and after['single_embedded_surface_candidate'] is True,
                'exact compiled candidate did not improve')
        changed_rows.append({'stable_id': sid, 'member_id': TARGETS[sid],
                             'moved_source_vertex_ids': moved,
                             'changed_vertex_record_count': len(local),
                             'maximum_source_displacement_mm':
                                 proof['maximum_vertex_displacement_mm'],
                             'old_exact_intersection_pairs': 3,
                             'new_exact_intersection_pairs': 0})
    require(len(changed_rows) == 2 and len(changed) == 32
            and old_a['coverage']['single_embedded_muscle_surface_candidate_count'] == 52
            and new_a['coverage']['single_embedded_muscle_surface_candidate_count'] == 54,
            'unexpected full-census delta')
    return {
        'schema': SCHEMA,
        'old_payload_sha256': human.sha256(old_payload),
        'new_payload_sha256': human.sha256(new_payload),
        'old_manifest_sha256': human.sha256(old_manifest),
        'new_manifest_sha256': human.sha256(new_manifest),
        'old_census_sha256': human.sha256(old_census),
        'new_census_sha256': human.sha256(new_census),
        'predicate_source_sha256': {name: human.sha256(Path(__file__).with_name(name))
                                    for name in ('muscle_tip_visual_delta.py',
                                                 'muscle_tip_visual_untangle.py',
                                                 'muscle_surface_embeddedness.py',
                                                 'model.py')},
        'all_header_record_binding_index_bytes_unchanged': True,
        'all_non_target_vertex_records_unchanged': True,
        'all_skinning_weights_unchanged': True,
        'changed_vertex_record_count': len(changed),
        'changed_surfaces': changed_rows,
        'single_embedded_muscle_surfaces_before': 52,
        'single_embedded_muscle_surfaces_after': 54,
        'clinical_anatomy': False, 'physical_volume_owner': False,
        'muscle_force_path_changed': False,
        'boundary': ('Only a pinned derived bilateral visual tip changes. '
                     'Source OBJ members, route mechanics, compiled bindings, '
                     'weights and face indices remain unchanged. Closed embedded '
                     'visual surfaces do not establish clinical muscle anatomy, '
                     'material, contact, or force transfer.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-payload', 'old-manifest', 'old-census', 'new-payload',
                 'new-manifest', 'new-census', 'source-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.old_payload, args.old_manifest, args.old_census,
                   args.new_payload, args.new_manifest, args.new_census,
                   args.source_root)
    human.write_json(args.output, result)
    print(json.dumps({'changed_vertex_record_count': result['changed_vertex_record_count'],
                      'single_embedded_muscle_surfaces_after':
                          result['single_embedded_muscle_surfaces_after']}))


if __name__ == '__main__':
    main()

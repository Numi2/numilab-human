"""Source-bound exact separation audit for eight independent abdominal organs.

This extends the retained five-organ failure receipt without changing it. The
three added source-named organs share its native Abdomen owner, so exact
compiled Float32 local coordinates are directly comparable.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import abdominal_organ_separation as original
from . import anatomical_laterality as laterality
from . import compiled_quotient_embeddedness as quotient
from . import lung_envelope as lung
from . import model as human
from . import surface_topology_repair as repair
from . import whole_body_embeddedness as indexed
from . import whole_body_surface_gate as selected_gate


SCHEMA = 'numi.human.compiled-abdominal-organ-separation.v2'
ADDED_ORGANS = (
    (561, 'gallbladder', 'FJ2817'),
    (562, 'left adrenal gland', 'FJ3129'),
    (563, 'right adrenal gland', 'FJ3130'),
)
ORGANS = original.ORGANS + ADDED_ORGANS


def audit(baseline_payload: Path, selected_payload: Path, selected_manifest: Path,
          selected_report: Path, census_archive: Path, native_audit: Path,
          native_pack: Path, pose_path: Path, prior_receipt: Path) -> dict:
    inputs = [Path(p).resolve() for p in (
        baseline_payload, selected_payload, selected_manifest, selected_report,
        census_archive, native_audit, native_pack, pose_path, prior_receipt,
    )]
    (baseline_payload, selected_payload, selected_manifest, selected_report,
     census_archive, native_audit, native_pack, pose_path, prior_receipt) = inputs
    prior = original.audit(baseline_payload, selected_payload, selected_manifest,
                           selected_report, census_archive, native_audit,
                           native_pack, pose_path)
    original.require(prior == human.read_json(prior_receipt)
                     and prior['pair_count'] == 10
                     and prior['status'] == 'failed_cross_surface_separation',
                     'published five-organ receipt changed')

    semantic, provenance = laterality.source_metadata(baseline_payload)
    specs, _ = repair.source_specs(baseline_payload)
    _, base_header, base_records, base_vertices, base_indices = lung.decode(baseline_payload)
    _, header, records, vertices, indices = lung.decode(selected_payload)
    original.require(base_header[:2] == (5, 579) and header[:2] == (5, 602),
                     'compiled source/selected ABI changed')
    gate = human.read_json(selected_report)
    original.require(gate['schema'] == selected_gate.SCHEMA
                     and len(gate['rows']) == 579, 'selected gate changed')
    _, _, census, census_sha = selected_gate.census_rows(census_archive, repair.BASE_SHA)
    original.require(census_sha == prior['source_census_summary_sha256']
                     and provenance == prior['source_semantic_manifest_sha256'],
                     'source provenance differs from five-organ receipt')

    meshes = {}
    organ_rows = []
    for sid, label, member in ORGANS:
        source = semantic[sid-1]
        choice = gate['rows'][sid-1]
        record = list(map(int, records[sid-1]))
        baseline_record = list(map(int, base_records[sid-1]))
        original.require(source['stable_id'] == sid
                         and (source.get('label') or source.get('source_name')) == label
                         and source['member_id'] == member
                         and (source.get('member_sha256') or source.get('source_member_sha256'))
                             == specs[sid-1]['source_member_sha256']
                         and source['source_structure_kind'] == 'organ'
                         and source['source_named_organ_type_matches'] is True
                         and source['core_body_index'] == 7
                         and choice['source_stable_id'] == sid
                         and choice['visible_stable_id'] == sid
                         and choice['selected_closed_embedded_surface_candidate'] is True,
                         f'whole-organ semantic/selection identity for {label}')
        original.require(record == baseline_record and record[0] == 7
                         and record[5] == sid, f'compiled record drift for {label}')
        _, fv, nv, fi, ni, _, _, _ = record
        original.require(np.array_equal(base_vertices[fv:fv+nv], vertices[fv:fv+nv])
                         and np.array_equal(base_indices[fi:fi+ni], indices[fi:fi+ni]),
                         f'compiled geometry drift for {label}')
        faces = (indices[fi:fi+ni].reshape(-1, 3)-fv).astype(np.int64)
        positions = vertices[fv:fv+nv, :3].astype(np.float64)
        original.require(census[sid-1]['geometry_sha256'] == indexed.geometry_hash(
                             base_vertices[fv:fv+nv], base_indices[fi:fi+ni])
                         and census[sid-1]['topology']['face_component_count'] == 1
                         and census[sid-1]['closed_embedded_surface_candidate'] is True,
                         f'published source embeddedness for {label}')
        current = quotient.classify_quotient(positions.tolist(), faces.tolist())
        original.require(current['topology']['face_component_count'] == 1
                         and current['topology']['closed_oriented_manifold_candidate']
                         and current['exact_intersection_pairs'] == 0
                         and current['self_intersection'] == 'exact_checked',
                         f'current embeddedness for {label}')
        meshes[label] = (positions, faces)
        organ_rows.append({'source_stable_id': sid, 'label': label,
                           'member_id': member,
                           'member_sha256': specs[sid-1]['source_member_sha256'],
                           'native_body_index': 7, 'compiled_vertex_count': nv,
                           'compiled_triangle_count': ni//3,
                           'single_embedded_surface_candidate': True})

    denominator, exact = original.exact_integer_meshes(meshes)
    pairs = []
    for first, second in itertools.combinations((row[1] for row in ORGANS), 2):
        pairs.append({'first': first, 'second': second,
                      **original.audit_pair(exact[first], exact[second], denominator)})
    prior_pairs = {(row['first'], row['second']): row for row in prior['pairs']}
    original.require(all(row == prior_pairs[(row['first'], row['second'])]
                         for row in pairs if (row['first'], row['second']) in prior_pairs),
                     'original five-organ pair results changed')
    failed = [f"{row['first']}--{row['second']}" for row in pairs
              if not row['disjoint_closed_domains']]
    return {
        'schema': SCHEMA,
        'status': ('failed_cross_surface_separation' if failed
                   else 'passed_selected_organ_separation'),
        'prior_five_organ_receipt_sha256': human.sha256(prior_receipt),
        'selected_payload_sha256': prior['selected_payload_sha256'],
        'native_pack_sha256': prior['native_pack_sha256'],
        'native_source_rest_pose_sha256': prior['native_source_rest_pose_sha256'],
        'source_census_summary_sha256': census_sha,
        'source_semantic_manifest_sha256': provenance,
        'executed_source_sha256': {
            **{
                name: human.sha256(Path(__file__).with_name(name))
                for name in ('expanded_abdominal_organ_separation.py',
                             'anatomical_laterality.py', 'surface_topology_repair.py',
                             'lung_envelope.py', 'whole_body_surface_gate.py')
            },
            **prior['executed_source_sha256'],
        },
        'coordinate_semantics': prior['coordinate_semantics'],
        'all_organs_share_native_abdomen_body_index': 7,
        'organs': organ_rows, 'pairs': pairs, 'pair_count': len(pairs),
        'clear_pair_count': len(pairs)-len(failed), 'failed_pair_names': failed,
        'clinical_anatomy': False, 'physical_volume_owner': False,
        'organ_mechanics': False,
        'boundary': ('Eight source-named whole-organ atlas representations are individually '
                     'closed and exactly embedded. Their 28 pairwise domain relationships are '
                     'tested in one native owner frame. Any surface crossing blocks disjoint '
                     'organ-domain admission. No source geometry was moved, clipped, or capped; '
                     'this does not establish clinical registration, tissue boundaries, '
                     'material, or mechanics.'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-payload', 'selected-payload', 'selected-manifest',
                 'selected-report', 'census-archive', 'native-audit',
                 'native-pack', 'pose', 'prior-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.baseline_payload, args.selected_payload,
                   args.selected_manifest, args.selected_report,
                   args.census_archive, args.native_audit, args.native_pack,
                   args.pose, args.prior_receipt)
    human.write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'pair_count': result['pair_count'],
                      'clear_pair_count': result['clear_pair_count'],
                      'failed_pair_names': result['failed_pair_names']}), flush=True)
    return 0 if not result['failed_pair_names'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

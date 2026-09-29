"""Exact inter-organ separation gate for five named abdominal atlas organs.

All five compiled surfaces share one native Abdomen owner. Their local Float32
coordinates can therefore be compared exactly without rounding a posed world
transform. A shared rigid transform cannot create or remove an intersection.
Per-surface embeddedness is rechecked before making any pairwise domain claim.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

from . import anatomical_laterality as laterality
from . import compiled_quotient_embeddedness as quotient
from . import lung_envelope as lung
from . import model as human
from . import surface_topology_repair as repair
from . import whole_body_embeddedness as indexed
from . import whole_body_surface_gate as selected_gate
from .cardiac_cavity_intersections import (
    _records, point_location, triangle_intersection_points,
)


SCHEMA = 'numi.human.compiled-abdominal-organ-separation.v1'
ORGANS = (
    (2, 'stomach', 'FJ2564'),
    (3, 'pancreas', 'FJ1895'),
    (4, 'right kidney', 'FJ3147'),
    (5, 'left kidney', 'FJ3145'),
    (13, 'spleen', 'FJ2561'),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('abdominal organ separation: ' + message)


def exact_integer_meshes(meshes: dict[str, tuple[object, object]]):
    """Represent exact binary32 metre coordinates on one integer lattice."""
    import numpy as np

    denominator = 1
    prepared = {}
    for name, (vertices, faces) in meshes.items():
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4
                and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) >= 4
                and np.isfinite(vertices).all() and (faces >= 0).all()
                and (faces < len(vertices)).all(), f'invalid mesh {name}')
        rational = [tuple(Fraction.from_float(float(x)) for x in point)
                    for point in vertices]
        for point in rational:
            for coordinate in point:
                denominator = math.lcm(denominator, coordinate.denominator)
        prepared[name] = (vertices, faces, rational)
    converted = {}
    for name, (vertices, faces, rational) in prepared.items():
        integer = [tuple(x.numerator*(denominator//x.denominator) for x in point)
                   for point in rational]
        triangle = vertices[faces]
        converted[name] = {
            'points': integer,
            'faces': faces,
            'records': _records(integer, faces.tolist()),
            'triangle_min': triangle.min(axis=1),
            'triangle_max': triangle.max(axis=1),
        }
    return denominator, converted


def audit_pair(first: dict, second: dict, denominator: int) -> dict:
    """Use Float32-exact AABBs, then exact rational triangle predicates."""
    import numpy as np

    a_lo, a_hi = first['triangle_min'], first['triangle_max']
    b_lo, b_hi = second['triangle_min'], second['triangle_max']
    candidates = point_contacts = crossing_pairs = 0
    first_pairs = []
    maximum_segment_m = 0.0
    if not (np.any(a_hi.max(axis=0) < b_lo.min(axis=0)) or
            np.any(b_hi.max(axis=0) < a_lo.min(axis=0))):
        for i, (lo, hi) in enumerate(zip(a_lo, a_hi, strict=True)):
            for j in np.flatnonzero(np.all((b_lo <= hi) & (b_hi >= lo), axis=1)):
                candidates += 1
                points = set(triangle_intersection_points(
                    first['records'][i][0], second['records'][int(j)][0],
                ))
                if not points:
                    continue
                if len(points) == 1:
                    point_contacts += 1
                else:
                    crossing_pairs += 1
                    points = list(points)
                    maximum_segment_m = max(maximum_segment_m, max(
                        math.dist(tuple(float(x)/denominator for x in left),
                                  tuple(float(x)/denominator for x in right))
                        for left, right in itertools.combinations(points, 2)
                    ))
                if len(first_pairs) < 12:
                    first_pairs.append([i, int(j)])
    intersections = point_contacts + crossing_pairs
    if intersections:
        containment = {'status': 'not_checked_intersecting_surfaces'}
        status = 'surface_crossing' if crossing_pairs else 'point_contact'
    else:
        a = point_location(first['points'][int(first['faces'][0, 0])], second['records'])
        b = point_location(second['points'][int(second['faces'][0, 0])], first['records'])
        containment = {'status': 'checked', 'first_in_second': a,
                       'second_in_first': b}
        status = ('separate_closed_domains' if a['location'] == b['location'] == 'outside'
                  else 'nested_or_indeterminate')
    return {
        'status': status, 'disjoint_closed_domains': status == 'separate_closed_domains',
        'aabb_candidate_triangle_pairs': candidates,
        'exact_intersecting_triangle_pairs': intersections,
        'exact_segment_or_polygon_crossing_pairs': crossing_pairs,
        'exact_point_contact_pairs': point_contacts,
        'maximum_intersection_segment_m': maximum_segment_m,
        'first_intersecting_triangle_pairs': first_pairs,
        'containment': containment,
    }


def audit(baseline_payload: Path, selected_payload: Path, selected_manifest: Path,
          selected_report: Path, census_archive: Path, native_audit: Path,
          native_pack: Path, pose_path: Path) -> dict:
    import numpy as np

    paths = tuple(Path(p).resolve() for p in (
        baseline_payload, selected_payload, selected_manifest, selected_report,
        census_archive, native_audit, native_pack, pose_path,
    ))
    baseline_payload, selected_payload, selected_manifest, selected_report, \
        census_archive, native_audit, native_pack, pose_path = paths
    require(human.sha256(baseline_payload) == repair.BASE_SHA,
            'pinned 579-surface baseline payload')
    manifest = human.read_json(selected_manifest)
    proof = human.read_json(native_audit)
    native = proof['native_audit']
    selected_sha = human.sha256(selected_payload)
    require(manifest['payload']['sha256'] == selected_sha
            and manifest['payload']['surfaces'] == 602
            and native['passed'] is True and native['surface_count'] == 602
            and native['payload_sha256'] == selected_sha
            and human.sha256(native_pack) == native['native_pack_sha256']
            and human.sha256(pose_path) == native['snapshot_sha256']
            and proof['source_rest_body_pose_unchanged'] is True,
            'selected payload/native source-rest identity')
    pose = human.read_json(pose_path)
    require(pose['schema'] == 'numi.human.native-torso-anatomy-pose-snapshot.v1'
            and pose['surface_count'] == 602 and pose['visible_layer_mask'] == 32767
            and len([body for body in pose['bodies'] if body['body_index'] == 7]) == 1,
            'native source-rest Abdomen owner')
    gate = human.read_json(selected_report)
    require(gate['schema'] == selected_gate.SCHEMA
            and gate['baseline_payload_sha256'] == repair.BASE_SHA
            and len(gate['rows']) == 579,
            'selected per-surface gate identity')
    _, census_identity, rows, summary_sha = selected_gate.census_rows(
        census_archive, repair.BASE_SHA,
    )
    for filename, digest in census_identity['predicate_source_sha256'].items():
        require(human.sha256(Path(__file__).with_name(filename)) == digest,
                'compiled quotient predicate source drift')
    semantic, provenance = laterality.source_metadata(baseline_payload)
    source_specs, _ = repair.source_specs(baseline_payload)
    _, base_header, base_records, base_vertices, base_indices = lung.decode(baseline_payload)
    _, header, records, vertices, indices = lung.decode(selected_payload)
    require(base_header[:2] == (5, 579) and header[:2] == (5, 602),
            'compiled anatomy ABI/source scope')

    meshes = {}
    organs = []
    for sid, label, member in ORGANS:
        source = semantic[sid-1]
        choice = gate['rows'][sid-1]
        require(source['stable_id'] == sid and source['label'] == label
                and source['member_id'] == member
                and source['member_sha256'] == source_specs[sid-1]['source_member_sha256']
                and source['source_structure_kind'] == 'organ'
                and source['source_named_organ_type_matches'] is True
                and source['core_body_index'] == 7
                and choice['source_stable_id'] == choice['visible_stable_id'] == sid
                and choice['selected_closed_embedded_surface_candidate'] is True,
                f'source semantic/selected owner for {label}')
        old = list(map(int, base_records[sid-1]))
        new = list(map(int, records[sid-1]))
        require(old == new and old[0] == 7 and old[5] == sid,
                f'compiled record drift for {label}')
        fv, nv, fi, ni = old[1:5]
        require(np.array_equal(base_vertices[fv:fv+nv], vertices[fv:fv+nv])
                and np.array_equal(base_indices[fi:fi+ni], indices[fi:fi+ni]),
                f'compiled source geometry drift for {label}')
        require(rows[sid-1]['geometry_sha256'] == indexed.geometry_hash(
                    base_vertices[fv:fv+nv], base_indices[fi:fi+ni])
                and rows[sid-1]['topology']['face_component_count'] == 1
                and rows[sid-1]['closed_embedded_surface_candidate'] is True,
                f'published source embeddedness for {label}')
        faces = (indices[fi:fi+ni].reshape(-1, 3)-fv).astype(np.int64)
        positions = vertices[fv:fv+nv, :3].astype(np.float64)
        current = quotient.classify_quotient(positions.tolist(), faces.tolist())
        require(current['topology']['face_component_count'] == 1
                and current['topology']['closed_oriented_manifold_candidate']
                and current['exact_intersection_pairs'] == 0
                and current['self_intersection'] == 'exact_checked',
                f'current embeddedness for {label}')
        meshes[label] = (positions, faces)
        organs.append({'source_stable_id': sid, 'label': label, 'member_id': member,
                       'member_sha256': source['member_sha256'],
                       'native_body_index': 7, 'compiled_vertex_count': nv,
                       'compiled_triangle_count': ni//3,
                       'single_embedded_surface_candidate': True})
    denominator, exact = exact_integer_meshes(meshes)
    pairs = []
    for first, second in itertools.combinations((row[1] for row in ORGANS), 2):
        pairs.append({'first': first, 'second': second,
                      **audit_pair(exact[first], exact[second], denominator)})
    failed = [f"{row['first']}--{row['second']}" for row in pairs
              if not row['disjoint_closed_domains']]
    return {
        'schema': SCHEMA, 'status': ('failed_cross_surface_separation' if failed
                                     else 'passed_selected_organ_separation'),
        'source_baseline_payload_sha256': repair.BASE_SHA,
        'selected_payload_sha256': selected_sha,
        'selected_manifest_sha256': human.sha256(selected_manifest),
        'native_audit_sha256': human.sha256(native_audit),
        'native_pack_sha256': human.sha256(native_pack),
        'native_source_rest_pose_sha256': human.sha256(pose_path),
        'selected_surface_gate_sha256': human.sha256(selected_report),
        'source_census_summary_sha256': summary_sha,
        'source_semantic_manifest_sha256': provenance,
        'executed_source_sha256': {
            name: human.sha256(Path(__file__).with_name(name))
            for name in ('abdominal_organ_separation.py',
                         'cardiac_cavity_intersections.py',
                         'compiled_quotient_embeddedness.py',
                         'whole_body_embeddedness.py')
        },
        'coordinate_semantics': 'exact_rational_value_of_compiled_Float32_owner_local_metres',
        'all_organs_share_native_abdomen_body_index': 7,
        'organs': organs, 'pairs': pairs,
        'pair_count': len(pairs),
        'clear_pair_count': len(pairs)-len(failed),
        'failed_pair_names': failed,
        'clinical_anatomy': False, 'physical_volume_owner': False,
        'organ_mechanics': False,
        'boundary': ('Five named whole-organ atlas representations are individually closed, '
                     'connected and exactly self-intersection-free in the selected compiled '
                     'packet. Pairwise exact surface crossings and containment are a separate '
                     'anatomical placement gate. Crossing surfaces block a disjoint organ-domain '
                     'claim; no source geometry is moved or clipped, and this does not establish '
                     'clinical registration, tissue boundaries, material, or mechanics.'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-payload', 'selected-payload', 'selected-manifest',
                 'selected-report', 'census-archive', 'native-audit',
                 'native-pack', 'pose', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.baseline_payload, args.selected_payload,
                   args.selected_manifest, args.selected_report,
                   args.census_archive, args.native_audit,
                   args.native_pack, args.pose)
    human.write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'pair_count': result['pair_count'],
                      'clear_pair_count': result['clear_pair_count'],
                      'failed_pair_names': result['failed_pair_names']}), flush=True)
    return 0 if not result['failed_pair_names'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evidence-scoped validation of 925 lung-seam events against exact NHA patch topology."""
from __future__ import annotations
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

E = Path('/Users/n/numi-human-resting-evidence-20261005')
OUT = E / 'native-lung-seam-patch-linkage-933'
NHA = E / 'lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy'
AUDIT = E / 'native-lung-seam-structural-audit-927'
SIGNED = E / 'native-lung-seam-signed-range-review-931/report.json'
CAPTURE_DIR = E / 'native-lung-seam-cycle-925/accepted-geometry'
OWNER_SRC = Path('/Users/n/numi-human-lung-seam-correction-001/src/numilab_human')
PARSER_PATH = OWNER_SRC / 'resting_anatomy_interface_patch.py'
RECIPROCAL_HELPER_PATH = OWNER_SRC / 'resting_lung_edge_repair.py'
EXPECTED = {
    'nha': '3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e',
    'targeted_native': '6767f41cd089ed4d652a030f15659840e3032d2a6002da822eb781462d1a5549',
    'source_reciprocal': 'dae6f8cb44d3c618fae1706b5004de6a8375c6a0386b520be1d3f0db65e744df',
    'source_native_pairs': '89377065ddc456d015c3253730a2f3e953a8f89c36c1c0ea6e2a6d9c14cd8832',
    'signed_range_report': '2d6820f8fb73611dbc2ee959d6b7217e1a1db2204b4871e1cda03339a14650b5',
    'payload_parser': 'e7692e4239c6a5ca4c12667b9bce94c06efa9bc3bec86cbb74e4625eeb416187',
    'reciprocal_owner': 'a6e6153547a0c16d540153a1aa5ff8c705b0ba031825c7ec8e319732633d43e1',
}
PATCH_PAIRS = {(305, 308): 8884, (306, 307): 8727, (307, 309): 9605}
EXPECTED_BOUNDARY_EDGES = {(305, 308): (1342, 1342), (306, 307): (923, 923), (307, 309): (601, 601)}
TARGET_PAIR_STEPS = {0, 4991, 5375, 5759, 6111, 6495, 7743, 9999}
SIDE_EPSILON_M = 1.0e-14
SIDE_EPSILON_SENSITIVITY_M = (0.0, 1.0e-15, 1.0e-14, 1.0e-13, 1.0e-12)
BARYCENTRIC_NUMERIC_TOLERANCE = 1.0e-8  # dimensionless reconstruction check, not a geometric clearance allowance


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f'missing {label}: {path}')
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{label} SHA mismatch: expected {expected}, got {actual}')
    return actual


def classify_source_side(source_range: dict, epsilon_m: float = SIDE_EPSILON_M) -> int:
    """Return +1/-1 for a source-established side; 0 means ambiguous and must fail closed."""
    lo = float(source_range['min_m'])
    hi = float(source_range['max_m'])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError('invalid source signed range')
    if lo >= -epsilon_m and hi > epsilon_m:
        return 1
    if hi <= epsilon_m and lo < -epsilon_m:
        return -1
    return 0


def opposite_source_side_intrusion(side: int, native_range: dict) -> float:
    lo = float(native_range['min_m'])
    hi = float(native_range['max_m'])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError('invalid native signed range')
    if side == 1:
        return max(0.0, -lo)
    if side == -1:
        return max(0.0, hi)
    raise ValueError('cannot measure intrusion from an ambiguous source side')


def assert_within_quantization_bound(intrusion_m: float, bound_m: float) -> None:
    if not math.isfinite(intrusion_m) or not math.isfinite(bound_m) or bound_m <= 0.0:
        raise ValueError('invalid intrusion or half-ULP bound')
    if intrusion_m > bound_m:
        raise ValueError(f'opposite-side excursion {intrusion_m:.17g} exceeds bound {bound_m:.17g}')


def assert_source_pair_allowed(pair: dict) -> None:
    if pair.get('candidate_source_intersection') and not pair.get('candidate_source_intersection_allowed_vertex_edge'):
        raise ValueError('source overlap is not an allowed vertex/edge contact')


def is_patch_linked(side: dict) -> bool:
    return bool(side['face_is_reciprocal_patch'] or side['patch_boundary_edge_count'] > 0 or side['patch_boundary_vertex_count'] > 0)


def assert_event_patch_linked(side_relations: list[dict]) -> None:
    if not any(is_patch_linked(side) for side in side_relations):
        raise ValueError('event lacks an exact reciprocal-patch face/edge/vertex topology link')


def pkey(point) -> bytes:
    return np.ascontiguousarray(point, dtype='<f4').tobytes()


def edgekey(a: bytes, b: bytes) -> tuple[bytes, bytes]:
    return (a, b) if a <= b else (b, a)


def build_patch_map(rows: dict, owners: tuple[int, int]):
    """Reproduce the exact coordinate-key/opposite-winding map used by the owner audit."""
    grouped = collections.defaultdict(list)
    for owner in owners:
        row = rows[owner]
        xyz, faces = row['vertices6'][:, :3], row['faces']
        for face_index, face in enumerate(faces):
            oriented = tuple(pkey(xyz[int(vertex)]) for vertex in face)
            grouped[tuple(sorted(oriented))].append((owner, int(face_index), oriented))

    patch_face_ids = {owner: set() for owner in owners}
    patch_faces = {owner: [] for owner in owners}
    for key, entries in grouped.items():
        if len(entries) > 2 or len({entry[0] for entry in entries}) != len(entries):
            raise ValueError(f'duplicate/multiply-owned source face key in patch {owners}')
        if len(entries) != 2:
            continue
        a, b = entries
        if {a[0], b[0]} != set(owners):
            raise ValueError(f'unexpected owner pair on reciprocal face key in patch {owners}')
        reverse = any(tuple(a[2][(shift - index) % 3] for index in range(3)) == b[2] for shift in range(3))
        if not reverse:
            raise ValueError(f'reciprocal patch face has same/invalid winding in patch {owners}')
        patch_face_ids[a[0]].add(a[1])
        patch_face_ids[b[0]].add(b[1])
        patch_faces[a[0]].append(a[2])
        patch_faces[b[0]].append(b[2])

    patch_boundary_edges = {owner: set() for owner in owners}
    patch_boundary_vertices = {owner: set() for owner in owners}
    for owner in owners:
        counts = collections.Counter()
        for tri in patch_faces[owner]:
            for i in range(3):
                counts[edgekey(tri[i], tri[(i + 1) % 3])] += 1
        patch_boundary_edges[owner] = {edge for edge, count in counts.items() if count == 1}
        patch_boundary_vertices[owner] = {vertex for edge in patch_boundary_edges[owner] for vertex in edge}
    matched = len(patch_faces[owners[0]])
    return {
        'matched_faces': matched,
        'face_ids': patch_face_ids,
        'boundary_edges': patch_boundary_edges,
        'boundary_vertices': patch_boundary_vertices,
    }


def barycentric(point, tri):
    a, b, c = np.asarray(tri, dtype=np.float64)
    v0, v1, v2 = b - a, c - a, np.asarray(point, dtype=np.float64) - a
    d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
    d20, d21 = float(v2 @ v0), float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if not math.isfinite(denom) or abs(denom) < 1.0e-28:
        raise ValueError('degenerate captured face in event footprint')
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w], dtype=np.float64)


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def distance_to_boundary(point: np.ndarray, edge_keys: set[tuple[bytes, bytes]]) -> float:
    if not edge_keys:
        raise ValueError('reciprocal patch has no boundary edges')
    return min(
        point_segment_distance(
            point,
            np.frombuffer(a, dtype='<f4').astype(np.float64),
            np.frombuffer(b, dtype='<f4').astype(np.float64),
        )
        for a, b in edge_keys
    )


def verify_and_load_inputs():
    verify_hash(NHA, EXPECTED['nha'], 'source NHA')
    verify_hash(AUDIT / 'targeted-native-report-v2.json', EXPECTED['targeted_native'], '927 native report')
    verify_hash(AUDIT / 'source-reciprocal-patches.json', EXPECTED['source_reciprocal'], '927 reciprocal patch report')
    verify_hash(AUDIT / 'source-native-hit-pair-comparison.json', EXPECTED['source_native_pairs'], '927 source/native pair report')
    verify_hash(SIGNED, EXPECTED['signed_range_report'], '931 signed-range report')
    verify_hash(PARSER_PATH, EXPECTED['payload_parser'], 'owner payload parser')
    verify_hash(RECIPROCAL_HELPER_PATH, EXPECTED['reciprocal_owner'], 'owner reciprocal map implementation')

    signed_report = json.loads(SIGNED.read_text())
    target_report = json.loads((AUDIT / 'targeted-native-report-v2.json').read_text())
    source_patch_report = json.loads((AUDIT / 'source-reciprocal-patches.json').read_text())
    source_pair_report = json.loads((AUDIT / 'source-native-hit-pair-comparison.json').read_text())
    capture_pins = signed_report['inputs']['native_capture_hashes']
    if set(map(int, capture_pins)) != TARGET_PAIR_STEPS:
        raise ValueError('931 capture pin list does not match the eight expected steps')
    for step, pin in capture_pins.items():
        verify_hash(CAPTURE_DIR / f'step-{step}.mrvpack', pin['pack_sha256'], f'925 step {step} pack')
        verify_hash(CAPTURE_DIR / f'step-{step}.receipt.json', pin['receipt_sha256'], f'925 step {step} receipt')

    sys.path.insert(0, str(OWNER_SRC.parent))
    from numilab_human.resting_anatomy_interface_patch import parse_payload
    _, rows = parse_payload(NHA)
    return signed_report, target_report, source_patch_report, source_pair_report, rows


ALL_LUNG_ROWS = tuple(range(305, 312))
DECLARED_CROSS_OWNER_PAIRS = ((305, 308), (305, 311), (306, 307), (306, 309), (306, 311), (307, 309), (307, 311), (308, 311))


def validate_full_scan_coverage(coverage: dict, source_nha_sha256: str, row_face_counts: dict[str, int]) -> None:
    """Fail closed unless a new capture scanned every face in each row and declared pair."""
    if coverage.get('source_nha_sha256') != source_nha_sha256:
        raise ValueError('new scan coverage is not bound to the expected source NHA')
    if coverage.get('all_faces_considered') is not True:
        raise ValueError('new scan does not attest full-face coverage')
    if tuple(sorted(map(int, coverage.get('self_rows', [])))) != ALL_LUNG_ROWS:
        raise ValueError('new scan omitted or duplicated a required lung/pleura/diaphragm self row')
    expected_self_counts = {str(row): int(row_face_counts[str(row)]) for row in ALL_LUNG_ROWS}
    actual_self_counts = {str(k): int(v) for k, v in coverage.get('self_face_counts', {}).items()}
    if actual_self_counts != expected_self_counts:
        raise ValueError('new scan self-face counts do not match pinned source topology')
    raw_pairs = [tuple(map(int, pair)) for pair in coverage.get('cross_pairs', [])]
    if len(raw_pairs) != len(set(raw_pairs)) or set(raw_pairs) != set(DECLARED_CROSS_OWNER_PAIRS):
        raise ValueError('new scan omitted, duplicated, or added a declared reciprocal cross-owner pair')
    expected_cross_counts = {
        f'{a}-{b}': [int(row_face_counts[str(a)]), int(row_face_counts[str(b)])]
        for a, b in DECLARED_CROSS_OWNER_PAIRS
    }
    actual_cross_counts = {str(k): list(map(int, v)) for k, v in coverage.get('cross_face_counts', {}).items()}
    if actual_cross_counts != expected_cross_counts:
        raise ValueError('new scan cross-pair face counts do not match pinned source topology')
    if 'step' not in coverage:
        raise ValueError('new scan coverage is missing accepted-capture identity field step')
    try:
        if int(coverage['step']) < 0:
            raise ValueError('negative accepted step')
    except (TypeError, ValueError) as error:
        raise ValueError('new scan coverage has an invalid accepted step') from error
    for key in ('pack_sha256', 'receipt_sha256'):
        value = str(coverage.get(key, ''))
        if re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
            raise ValueError(f'new scan coverage is missing a valid SHA-256 field {key}')


def reject_native_self_witnesses(self_witnesses: list[dict]) -> None:
    """A full native self scan must have no unallowed same-row face intersections."""
    if self_witnesses:
        first = self_witnesses[0]
        raise ValueError(f'native self-intersection witness requires review: {first}')


def reject_unreviewed_witnesses(events: list[dict], known_source_face_pairs: set[tuple[tuple[int, int], tuple[int, int]]]) -> None:
    """Reject any new unallowed face pair until its source/topology relation is reviewed."""
    for event in events:
        owners = tuple(map(int, event['owners']))
        face_ids = tuple(map(int, event['face_ids']))
        if (owners, face_ids) not in known_source_face_pairs:
            raise ValueError(f'unreviewed/new native witness {owners}/{face_ids}')


def analyze():
    signed_report, target_report, source_patch_report, source_pair_report, rows = verify_and_load_inputs()
    by_patch_summary = {tuple(row['owners']): row for row in source_patch_report['pairs']}
    patch_maps = {}
    for owners, expected_faces in PATCH_PAIRS.items():
        summary = by_patch_summary.get(owners)
        if summary is None or summary['final_matched_face_pairs'] != expected_faces:
            raise ValueError(f'patch summary mismatch for {owners}')
        patch = build_patch_map(rows, owners)
        if patch['matched_faces'] != expected_faces:
            raise ValueError(f'parsed source patch map mismatch for {owners}')
        edge_counts = tuple(len(patch['boundary_edges'][owner]) for owner in owners)
        if edge_counts != EXPECTED_BOUNDARY_EDGES[owners]:
            raise ValueError(f'patch boundary count mismatch for {owners}: {edge_counts}')
        patch_maps[owners] = patch

    source_pairs = {(tuple(row['owners']), tuple(row['face_ids'])): row for row in source_pair_report['face_pairs']}
    if len(source_pairs) != 33:
        raise ValueError(f'expected 33 unique source face pairs, got {len(source_pairs)}')
    for pair in source_pairs.values():
        assert_source_pair_allowed(pair)
    source_intersections = [pair for pair in source_pairs.values() if pair['candidate_source_intersection']]
    source_disjoint = [pair for pair in source_pairs.values() if not pair['candidate_source_intersection']]
    if len(source_intersections) != 11 or len(source_disjoint) != 22:
        raise ValueError('source pair intersection census differs from pinned 927 report')

    target_events = {}
    for pose in target_report['native_pose_unique_unallowed_face_pairs']:
        for event in pose['face_pairs']:
            key = (int(pose['step']), tuple(event['owners']), tuple(event['face_ids']))
            if key in target_events:
                raise ValueError(f'duplicate 927 event key {key}')
            target_events[key] = event
    signed_events = {}
    for event in signed_report['all_events']:
        key = (int(event['step']), tuple(event['owners']), tuple(event['face_ids']))
        if key in signed_events:
            raise ValueError(f'duplicate 931 event key {key}')
        signed_events[key] = event
    if len(target_events) != 93 or set(target_events) != set(signed_events):
        raise ValueError('canonical 927/931 event identities do not match 93 events')

    event_rows = []
    link_counts = collections.Counter()
    relation_by_pair = collections.defaultdict(collections.Counter)
    side_counts = collections.Counter()
    max_ratio = 0.0
    max_intrusion = 0.0
    max_quantization_bound = 0.0
    min_quantization_bound = math.inf
    max_footprint_barycentric_error = 0.0
    min_boundary_distance = math.inf
    max_boundary_distance = 0.0
    for key in sorted(signed_events):
        step, owners, face_ids = key
        signed_event = signed_events[key]
        target_event = target_events[key]
        patch = patch_maps.get(owners)
        source_pair = source_pairs.get((owners, face_ids))
        if patch is None or source_pair is None:
            raise ValueError(f'event is outside declared reciprocal patches/source pair map: {key}')
        assert_source_pair_allowed(source_pair)
        side = classify_source_side(signed_event['source_signed_gap_m'])
        if side == 0:
            raise ValueError(f'ambiguous source side for event {key}')
        side_counts[side] += 1
        intrusion = opposite_source_side_intrusion(side, signed_event['native_signed_gap_m'])
        bound = float(signed_event['half_ulp_pair_normal_bound_m'])
        assert_within_quantization_bound(intrusion, bound)
        ratio = intrusion / bound
        max_ratio = max(max_ratio, ratio)
        max_intrusion = max(max_intrusion, intrusion)
        max_quantization_bound = max(max_quantization_bound, bound)
        min_quantization_bound = min(min_quantization_bound, bound)

        owner_sides = []
        for index, owner in enumerate(owners):
            row = rows[owner]
            face_index = int(face_ids[index])
            face = row['faces'][face_index]
            source_tri = np.asarray(row['vertices6'][face, :3], dtype=np.float64)
            source_tri_f32 = np.asarray(row['vertices6'][face, :3], dtype='<f4')
            reported_source_tri_f32 = np.asarray(target_event['source_vertex_xyz_m'][index], dtype='<f4')
            if source_tri_f32.tobytes() != reported_source_tri_f32.tobytes():
                raise ValueError(f'927 source triangle does not match pinned NHA face {owner}/{face_index}')

            keys = [pkey(x) for x in row['vertices6'][face, :3]]
            event_edges = {edgekey(keys[i], keys[(i + 1) % 3]) for i in range(3)}
            boundary_edge_count = len(event_edges & patch['boundary_edges'][owner])
            boundary_vertex_count = len(set(keys) & patch['boundary_vertices'][owner])
            is_patch_face = face_index in patch['face_ids'][owner]
            side_record = {
                'owner': owner,
                'face_id': face_index,
                'face_is_reciprocal_patch': bool(is_patch_face),
                'patch_boundary_edge_count': boundary_edge_count,
                'patch_boundary_vertex_count': boundary_vertex_count,
                'linked_by_exact_source_topology': bool(is_patch_face or boundary_edge_count or boundary_vertex_count),
            }

            captured_tri = np.asarray(target_event['captured_world_vertex_xyz_m'][index], dtype=np.float64)
            min_bary = math.inf
            max_bary = -math.inf
            boundary_distances = []
            for point in target_event['intersection_points_m']:
                weights = barycentric(point, captured_tri)
                min_bary = min(min_bary, float(weights.min()))
                max_bary = max(max_bary, float(weights.max()))
                if min_bary < -BARYCENTRIC_NUMERIC_TOLERANCE or max_bary > 1.0 + BARYCENTRIC_NUMERIC_TOLERANCE:
                    raise ValueError(f'intersection point lies outside captured face beyond numeric reconstruction tolerance: {key}')
                mapped_source_point = weights @ source_tri
                boundary_distances.append(distance_to_boundary(mapped_source_point, patch['boundary_edges'][owner]))
            side_record['captured_footprint_barycentric_min'] = min_bary
            side_record['captured_footprint_barycentric_max'] = max_bary
            side_record['mapped_source_footprint_min_distance_to_patch_boundary_m'] = min(boundary_distances)
            side_record['mapped_source_footprint_max_distance_to_patch_boundary_m'] = max(boundary_distances)
            max_footprint_barycentric_error = max(max_footprint_barycentric_error, max(0.0, -min_bary, max_bary - 1.0))
            min_boundary_distance = min(min_boundary_distance, min(boundary_distances))
            max_boundary_distance = max(max_boundary_distance, max(boundary_distances))
            owner_sides.append(side_record)

        assert_event_patch_linked(owner_sides)
        if any(side_record['linked_by_exact_source_topology'] for side_record in owner_sides):
            if any(side_record['face_is_reciprocal_patch'] for side_record in owner_sides):
                relation = 'event_face_is_reciprocal_patch_face'
            elif any(side_record['patch_boundary_edge_count'] for side_record in owner_sides):
                relation = 'event_face_shares_exact_patch_boundary_edge'
            else:
                relation = 'event_face_shares_exact_patch_boundary_vertex'
        else:
            raise ValueError(f'no topological patch relation for event {key}')
        link_counts[relation] += 1
        relation_by_pair[f'{owners[0]}-{owners[1]}'][relation] += 1
        event_rows.append({
            'step': step,
            'owners': list(owners),
            'face_ids': list(face_ids),
            'raw_native_intersection_point_count': int(signed_event['point_count']),
            'raw_native_intersection_span_m': float(signed_event['native_intersection_span_m']),
            'captured_common_vertex_edge_count': int(signed_event['captured_common_vertex_edge_count']),
            'shared_source_coordinate_count': int(signed_event['shared_source_coordinate_count']),
            'source_closest_distance_m': float(signed_event['source_closest_distance_m']),
            'source_orientation_method': signed_event['source_orientation_method'],
            'source_side': 'positive' if side > 0 else 'negative',
            'source_signed_range_m': signed_event['source_signed_gap_m'],
            'native_signed_range_m': signed_event['native_signed_gap_m'],
            'opposite_source_side_intrusion_m': intrusion,
            'pair_half_ulp_normal_bound_m': bound,
            'intrusion_over_pair_half_ulp_bound': ratio,
            'source_pair_intersection': bool(source_pair['candidate_source_intersection']),
            'source_pair_intersection_allowed_vertex_edge': bool(source_pair['candidate_source_intersection_allowed_vertex_edge']),
            'patch_relation': relation,
            'owner_face_relations': owner_sides,
        })

    if sum(link_counts.values()) != 93 or link_counts['event_face_is_reciprocal_patch_face'] != 6:
        raise ValueError('event patch-linkage census differs from expected complete linkage')
    expected_relations = {
        '305-308': {'event_face_shares_exact_patch_boundary_edge': 43, 'event_face_shares_exact_patch_boundary_vertex': 38},
        '306-307': {'event_face_shares_exact_patch_boundary_edge': 4, 'event_face_shares_exact_patch_boundary_vertex': 2},
        '307-309': {'event_face_is_reciprocal_patch_face': 6},
    }
    for pair, expected in expected_relations.items():
        if dict(relation_by_pair[pair]) != expected:
            raise ValueError(f'event-level topology linkage changed for {pair}: {dict(relation_by_pair[pair])}')

    epsilon_sensitivity = {}
    for epsilon in SIDE_EPSILON_SENSITIVITY_M:
        counts = collections.Counter(classify_source_side(event['source_signed_gap_m'], epsilon) for event in signed_events.values())
        epsilon_sensitivity[f'{epsilon:.1e}'] = {'positive': counts[1], 'negative': counts[-1], 'ambiguous': counts[0]}
    canonical_counts = {'positive': side_counts[1], 'negative': side_counts[-1], 'ambiguous': side_counts[0]}
    if canonical_counts != {'positive': 90, 'negative': 3, 'ambiguous': 0}:
        raise ValueError(f'source-side census changed: {canonical_counts}')
    if len(set(tuple(v.values()) for v in epsilon_sensitivity.values())) != 1:
        raise ValueError('source-side classifications are sensitive across the diagnostic epsilon grid')

    patch_summaries = {}
    for owners, patch in patch_maps.items():
        patch_summaries[f'{owners[0]}-{owners[1]}'] = {
            'matched_reciprocal_face_pairs': patch['matched_faces'],
            'boundary_edges_by_owner': {str(owner): len(patch['boundary_edges'][owner]) for owner in owners},
            'boundary_vertex_counts_by_owner': {str(owner): len(patch['boundary_vertices'][owner]) for owner in owners},
        }
    max_coordinate = max(float(np.max(np.abs(rows[owner]['vertices6'][:, :3]))) for pair in PATCH_PAIRS for owner in pair)
    report = {
        'schema': 'numi.human.native925.lung-patch-linkage-and-sidedness.v1',
        'status': 'PASS_SCOPED_GEOMETRIC_CLASSIFICATION',
        'scope': '93 canonical exact native pose/owner/face events for the three declared reciprocal fissure patches only; this does not certify all lung interfaces or global anatomy',
        'inputs': {
            'source_nha': {'path': str(NHA), 'sha256': EXPECTED['nha']},
            'targeted_native_report': {'path': str(AUDIT / 'targeted-native-report-v2.json'), 'sha256': EXPECTED['targeted_native']},
            'source_reciprocal_patches': {'path': str(AUDIT / 'source-reciprocal-patches.json'), 'sha256': EXPECTED['source_reciprocal']},
            'source_native_hit_pair_comparison': {'path': str(AUDIT / 'source-native-hit-pair-comparison.json'), 'sha256': EXPECTED['source_native_pairs']},
            'signed_range_report': {'path': str(SIGNED), 'sha256': EXPECTED['signed_range_report']},
            'owner_payload_parser': {'path': str(PARSER_PATH), 'sha256': EXPECTED['payload_parser']},
            'owner_reciprocal_map_source': {'path': str(RECIPROCAL_HELPER_PATH), 'sha256': EXPECTED['reciprocal_owner'], 'reference': 'shared_face_maps exact coordinate-key/opposite-winding implementation'},
            'native_925_capture_hashes': signed_report['inputs']['native_capture_hashes'],
        },
        'method': {
            'patch_linkage': 'Each exact native intersection point is barycentrically verified inside its captured triangle and mapped to that owner NHA source face. The source face must either be a face of the exact reciprocal patch, or share an exact float32-coordinate boundary edge or vertex with that patch. This is topological face/edge/vertex incidence; no metric-distance threshold is used. Minimum source-footprint-to-patch-boundary distances are descriptive only.',
            'reciprocal_patch': 'Exact float32 position-byte triangle keys, exactly two owners, reverse winding, and boundary edges counted once over the matched reciprocal faces; implementation matches the pinned existing owner helper.',
            'source_side': 'For each event, classify source signed range as positive when min >= -epsilon and max > epsilon; negative when max <= epsilon and min < -epsilon; otherwise ambiguous and fail closed. epsilon is 1e-14 m, a diagnostic numerical deadband only, not an anatomy or clearance tolerance.',
            'epsilon_basis': 'All 93 classifications are unchanged for epsilon = 0, 1e-15, 1e-14, 1e-13, and 1e-12 m. The selected epsilon is 100x below the 1e-12 sensitivity point and several orders below the measured pair half-ULP bounds; this checks sign stability but is not a formal floating-point forward-error proof.',
            'opposite_side_intrusion': 'positive source side: max(0, -native.min); negative source side: max(0, native.max). Ambiguous source ranges reject. Each value is compared against the existing pair half-ULP normal-coordinate bound without changing that bound or native exact-intersection gate.',
            'barycentric_check': f'intersection footprint weights must lie within [0,1] up to {BARYCENTRIC_NUMERIC_TOLERANCE:g} dimensionless reconstruction tolerance; this is only a triangle-membership numerical check.',
        },
        'code': {
            'analysis_script_path': str(Path(__file__).resolve()),
            'analysis_script_sha256': sha256(Path(__file__).resolve()),
            'test_script_path': str(OUT / 'test_patch_linkage.py'),
            'test_script_sha256': sha256(OUT / 'test_patch_linkage.py'),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'analyze_command': 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python analyze.py',
            'test_command': 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_patch_linkage.py',
        },
        'patches': patch_summaries,
        'canonical_events': {
            'count': len(event_rows),
            'by_pose': {str(step): sum(row['step'] == step for row in event_rows) for step in sorted(TARGET_PAIR_STEPS)},
            'by_owner_pair': {pair: sum(tuple(row['owners']) == tuple(map(int, pair.split('-'))) for row in event_rows) for pair in ('305-308', '306-307', '307-309')},
            'source_side_counts': canonical_counts,
            'source_side_epsilon_sensitivity': epsilon_sensitivity,
            'patch_topology_relation_counts': dict(link_counts),
            'patch_topology_relation_counts_by_pair': {pair: dict(counts) for pair, counts in relation_by_pair.items()},
            'unique_source_face_pair_count': len(source_pairs),
            'unique_source_exact_vertex_or_edge_pairs': len(source_intersections),
            'unique_source_disjoint_pairs': len(source_disjoint),
            'source_overlap_beyond_allowed_vertex_edge_contacts': 0,
            'maximum_opposite_source_side_intrusion_m': max_intrusion,
            'maximum_intrusion_over_pair_half_ulp_bound': max_ratio,
            'minimum_pair_half_ulp_bound_m': min_quantization_bound,
            'maximum_pair_half_ulp_bound_m': max_quantization_bound,
            'maximum_mapped_footprint_to_patch_boundary_distance_m': max_boundary_distance,
            'minimum_mapped_footprint_to_patch_boundary_distance_m': min_boundary_distance,
            'maximum_barycentric_numeric_excess': max_footprint_barycentric_error,
            'maximum_source_coordinate_magnitude_m': max_coordinate,
            'epsilon_to_float64_unit_roundoff_at_max_coordinate_ratio': SIDE_EPSILON_M / (np.finfo(np.float64).eps * max_coordinate),
        },
        'events': event_rows,
        'reusable_new_capture_contract': {
            'source_row_face_counts': {str(owner): int(len(rows[owner]['faces'])) for owner in ALL_LUNG_ROWS},
            'known_unallowed_source_face_pair_catalog': [
                {'owners': list(owners), 'face_ids': list(face_ids)}
                for owners, face_ids in sorted(source_pairs)
            ],
            'required_self_rows_all_faces': list(ALL_LUNG_ROWS),
            'required_declared_cross_owner_pairs_all_faces': [list(pair) for pair in DECLARED_CROSS_OWNER_PAIRS],
            'unknown_or_new_unallowed_witness_policy': 'reject unless the exact owner/face key is present in the pinned 927 source-native face-pair catalog; additions require explicit source/topology review',
            'coverage_fields_required': ['source_nha_sha256', 'all_faces_considered', 'self_rows', 'self_face_counts', 'cross_pairs', 'cross_face_counts', 'step', 'pack_sha256', 'receipt_sha256'],
            'validator_functions': ['validate_full_scan_coverage', 'reject_native_self_witnesses', 'reject_unreviewed_witnesses'],
        },
        'interpretation_and_limits': [
            'All 93 exact native intersections remain in this record; no event is dropped and no native geometry or exact gate is changed.',
            'All 93 are topologically linked to a declared reciprocal fissure patch: 47 event faces share an exact patch-boundary edge, 40 share an exact patch-boundary vertex, and 6 events lie on reciprocal patch faces themselves. The last six are inside the 307/309 patch rather than on its outer boundary.',
            'The source-side excursion is bounded by the existing pair half-ULP coordinate-quantization estimate for this event set. This supports only a scoped source-linked, quantization-bounded apposition classification; it is not proof of anatomical intention, zero contact, global clearance, or tissue penetration depth.',
            'The half-ULP calculation is not a bound on respiratory deformation, GPU skinning arithmetic, registration uncertainty, or deformation-map interpolation. No claim is made that the 93 events are caused only by float rounding.',
            'The reciprocal patches themselves retain multiple components/open boundaries in 927; this report validates the event-to-patch topology relation, not patch closure or global lung qualification.',
        ],
    }
    report_path = OUT / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    result = analyze()
    print(json.dumps({
        'report': str(OUT / 'report.json'),
        'events': result['canonical_events']['count'],
        'source_side_counts': result['canonical_events']['source_side_counts'],
        'patch_relations': result['canonical_events']['patch_topology_relation_counts'],
        'max_intrusion_m': result['canonical_events']['maximum_opposite_source_side_intrusion_m'],
        'max_intrusion_over_bound': result['canonical_events']['maximum_intrusion_over_pair_half_ulp_bound'],
    }, sort_keys=True))

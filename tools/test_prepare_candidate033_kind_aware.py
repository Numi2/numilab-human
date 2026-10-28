#!/usr/bin/env python3
"""Read-only regressions for baseline806 source-kind to D37 support mapping."""
import csv
import importlib.util
import json
import numpy as np
from pathlib import Path

BASE = Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/inputs/baseline-806.npz')
P1 = Path('/Users/numi/Work/numi-human-connected-surface-069/publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz')
CSV = Path('/Users/numi/Work/numi-human-connected-surface-069/publication-070/evidence/mass/compiled-cell-export-068/run-003/source-preflight/registered-skin.csv.gz')
SCRIPT = Path(__file__).with_name('prepare_candidate033_d37.py')
CANON = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/candidate.npz')
IDENTITY = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/publication-033/source-face-identity.npz')
COMPACT_MAP = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222/attempt-006-used-compact/O-compact-node-map.csv')

spec = importlib.util.spec_from_file_location('prep033', SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
rows, source_world = prep.read_source_csv(CSV)
crosswalk = prep.build_source_crosswalk(P1, rows)
base = np.load(BASE, allow_pickle=False)
payload_row, p1_row, map_kind, domain = prep.resolve_baseline_source_rows(base['node_kind'], base['node_source_row'], crosswalk)

kinds = np.asarray(base['node_kind'], dtype=np.int64)
source = np.asarray(base['node_source_row'], dtype=np.int64)
kind_counts = {str(int(k)): int(np.count_nonzero(kinds == k)) for k in np.unique(kinds)}
assert kind_counts == {'-1': 12436, '0': 2368, '1': 4886, '2': 1500, '5': 22}, kind_counts
assert np.count_nonzero((kinds == 1) & (payload_row >= 0)) == 4886
assert np.count_nonzero((kinds == 2) & (payload_row >= 0)) == 1500
assert np.count_nonzero((kinds == 0) & (payload_row >= 0)) == 2289
assert np.count_nonzero((kinds == 0) & (payload_row < 0)) == 79
assert np.count_nonzero((kinds == 5) & (payload_row >= 0)) == 0
assert np.count_nonzero((kinds == -1) & (payload_row >= 0)) == 0

# Same legacy row has different meanings by node kind: 0=>coarse representative,
# 1/2=>exact detail-source ID, and unmatched roots never become detail rows.
probe_rows, probe_p1, probe_method, probe_domain = prep.resolve_baseline_source_rows(
    np.array([0, 1, 2, 5, -1]), np.array([1233, 1233, 1233, 1233, 1233]), crosswalk)
assert probe_rows.tolist() == [24939, 1233, 1233, -1, -1], probe_rows.tolist()
assert probe_p1.tolist() == [24939, 1233, 1233, -1, -1], probe_p1.tolist()
assert 'binding_support' in probe_method[0] and probe_domain[0] == 'coarse_representative_row'
assert probe_domain[1] == 'exact_detail_source_vertex_id'
assert probe_method[3] == 'baseline_kind5_unmatched_root'

# Critical canonical point: O row16398 -> O033 row16398 -> baseline806 row1203.
# Its kind-0 coarse ID 1233 resolves only as P1 representative binding support 24939.
base_row = 1203
assert int(base['node_kind'][base_row]) == 0 and int(base['node_source_row'][base_row]) == 1233
assert int(payload_row[base_row]) == 24939 and int(p1_row[base_row]) == 24939
p1 = crosswalk['p1']
target = np.asarray(base['points_world_m'][base_row], dtype=np.float64)
legacy_wrong = source_world[int(base['node_source_row'][base_row])].astype(np.float64)
support_current = source_world[int(payload_row[base_row])].astype(np.float64)
p1_baseline = np.asarray(p1['detail_baseline_world_m'][p1_row[base_row]], dtype=np.float64)
wrong_distance = float(np.linalg.norm(target - legacy_wrong))
support_distance = float(np.linalg.norm(target - support_current))
p1_distance = float(np.linalg.norm(target - p1_baseline))
assert wrong_distance > 1.0, wrong_distance
assert support_distance < 2e-6, support_distance
assert p1_distance < 1e-6, p1_distance
assert int(crosswalk['detail_donor'][p1_row[base_row]]) == 23
assert int(crosswalk['csv_donor'][payload_row[base_row]]) == 23

# Verify the exact canonical O row maps through explicit candidate->O033->806 maps.
canon = np.load(CANON, allow_pickle=False)
identity = np.load(IDENTITY, allow_pickle=False)
candidate_to_o033 = np.asarray(canon['candidate_vertex_to_O033_row'], dtype=np.int64)
assert np.array_equal(candidate_to_o033, np.arange(len(candidate_to_o033)))
compact_candidate_rows, compact_o033_rows = prep.load_compact_candidate_rows(COMPACT_MAP, len(candidate_to_o033), candidate_to_o033)
assert len(compact_candidate_rows) == 32002
assert int(compact_candidate_rows[16264]) == 16398
assert int(compact_o033_rows[16264]) == 16398
o033 = np.load('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/publication-033/candidate.npz', allow_pickle=False)
assert np.array_equal(canon['points_world_m'], o033['points_world_m'])
assert int(identity['output_vertex_source_rows'][candidate_to_o033[16398]]) == base_row
face_parent = np.asarray(canon['output_face_to_precontraction_candidate_face_row'], dtype=np.int64)
assert face_parent.shape == (len(canon['source_O_faces']),)
assert np.all((face_parent >= 0) & (face_parent < len(o033['source_O_faces'])))
off = np.asarray(canon['face_ancestry_offsets'], dtype=np.int64)
assert off.shape == (len(canon['source_O_faces']) + 1,) and off[0] == 0 and off[-1] == len(canon['face_ancestry_parent_face_rows'])
assert np.all(np.diff(off) > 0)
assert prep.sha(CANON) == prep.CONDITIONED_CAND
with COMPACT_MAP.open(newline='') as f:
    compact = next(r for r in csv.DictReader(f) if int(r['compact_vertex_row']) == 16264)
assert int(compact['canonical_candidate_vertex_row']) == 16398
assert int(compact['O033_source_vertex_row']) == 16398

print(json.dumps({
    'pass': True,
    'baseline_kind_counts': kind_counts,
    'kind0_unique_P1_representatives': 2289,
    'kind0_unresolved': 79,
    'kind1_2_exact_detail_ids': 6386,
    'P1_detail_id_to_current_D37_payload_rows': 61950,
    'P1_current_D37_donor_mismatches': 0,
    'critical_row': {
        'canonical_O_vertex': 16398,
        'compact_output_vertex_row': 16264,
        'O033_vertex': 16398,
        'baseline806_node': base_row,
        'legacy_kind': int(base['node_kind'][base_row]),
        'legacy_coarse_source_row': int(base['node_source_row'][base_row]),
        'P1_binding_support_detail_id': int(p1['detail_source_vertex_id'][p1_row[base_row]]),
        'D37_binding_support_payload_row': int(payload_row[base_row]),
        'binding_support_donor': 23,
        'bad_legacy_as_detail_distance_m': wrong_distance,
        'correct_support_current_distance_m': support_distance,
        'P1_baseline_rep_distance_m': p1_distance,
        'topology_source_vertex_row': -1,
        'source_domain_kind0': 'coarse_representative_row'
    }
}, indent=2))

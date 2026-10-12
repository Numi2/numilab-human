#!/usr/bin/env python3
"""Compact CPU-only review of back-row zero-area counts and row157 preflight false positive."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

A = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
OUT = A / 'back-parent-default-pose-preflight-008-back408' / 'review-001'
PRED_DIR = A / 'back-parent-default-pose-preflight-008-back408'
NATIVE_DIR = A / 'back-native-014-009-60s'
NATIVE_RUN = NATIVE_DIR / 'baseline/native-run'
NATIVE_AUDIT = NATIVE_DIR / 'native-back-audit-001'
REJECT_RUN = A / 'back-forearm-native-014-015-back455-right177' / 'baseline/native-run'
REJECT_PACK = REJECT_RUN / 'rejected-surface-audit/step-0.mrvpack'
FAILURE = REJECT_RUN / 'surface-audit-failure-step-0.json'
OWNER = Path('/Users/n/numi-human-left-pq-publication-1344/src')
HEADER = Path('/Users/n/numi-lab-parent-bindings-1353/apps/NumiHumanRestingVisual.hpp')
RUNNER = Path('/Users/n/numi-human-complete-reference-inventory-1335/src/numilab_human/resting_run.py')
PREP = A / 'prepare_back_parent_composition_014_008.py'
PRED = PRED_DIR / 'parent-default-pose-row-157.npz'
APPENDED = PRED_DIR / 'append/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
APPENDED_MANIFEST = APPENDED.with_suffix('.manifest.json')
NATIVE_PACK = NATIVE_RUN / 'accepted-geometry/step-0.mrvpack'
NATIVE_RECEIPT = NATIVE_RUN / 'accepted-geometry/step-0.receipt.json'
NATIVE_INVOCATION = NATIVE_RUN / 'invocation.json'
NATIVE_METADATA = NATIVE_RUN / 'run-metadata.json'
NATIVE_ROW = NATIVE_AUDIT / 'step-0-row-157.npz'
NATIVE_ROW_JSON = NATIVE_AUDIT / 'step-0-row-157.json'
BACK_ROWS = (153, 154, 155, 156, 157, 158)
sys.path.insert(0, str(OWNER))
from numilab_human import common_atlas_skin_clearance as clearance
from numilab_human import cardiac_cavity_intersections as intersections

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def lattice_zero_face_ids(points, faces):
    face_array = np.asarray(faces, dtype=np.int64)
    referenced = sorted(set(int(v) for v in face_array.reshape(-1)))
    keys = {
        vertex: intersections.float32_point_lattice_key(tuple(float(x) for x in points[vertex]))
        for vertex in referenced
    }
    zero = []
    for face_id, face in enumerate(face_array):
        a, b, c = (keys[int(v)] for v in face)
        if not any(intersections._cross(intersections._sub(b, a), intersections._sub(c, a))):
            zero.append(int(face_id))
    return zero

def area_stats(points, faces):
    tri = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    twice = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(twice, axis=1)
    return {
        'min_area_m2': float(np.min(area)),
        'min_nonzero_area_m2': float(np.min(area[area > 0])),
        'zero_area_count_in_float64_diagnostic': int(np.count_nonzero(area == 0)),
    }

def box(points):
    points = np.asarray(points, dtype=np.float64)
    return {
        'min_m': points.min(axis=0).tolist(),
        'max_m': points.max(axis=0).tolist(),
        'center_m': (0.5 * (points.min(axis=0) + points.max(axis=0))).tolist(),
    }

def main():
    paths = [
        Path(__file__), PREP, PRED, APPENDED, APPENDED_MANIFEST,
        NATIVE_PACK, NATIVE_RECEIPT, NATIVE_INVOCATION, NATIVE_METADATA,
        NATIVE_ROW, NATIVE_ROW_JSON, REJECT_PACK, FAILURE,
        HEADER, RUNNER, OWNER / 'numilab_human/common_atlas_skin_clearance.py',
        OWNER / 'numilab_human/cardiac_cavity_intersections.py',
    ]
    for path in paths:
        require(path.is_file(), 'missing pinned input: ' + str(path))
    pins_start = {str(path): sha256(path) for path in paths}
    require(pins_start[str(PRED)] == 'b4081410389ae3d660e88ecb52e3a53e44837bd31d85d38f957d50eb05317e56',
            'row157 default-pose predictor hash changed')
    require(pins_start[str(NATIVE_ROW)] == 'b55dc45ba17ead16ab870651c1d47b199b2a88d9a328be9bd7c78ffe868a262d',
            'row157 native step-0 extraction hash changed')
    require(pins_start[str(REJECT_PACK)] == '3d59952cd9e82eed01ce4961ff67c5a691bcfa280020441f78217b6f23ab372c',
            'rejected pack hash changed')

    target_keys = {(51005, sid) for sid in BACK_ROWS}
    reject_positions, reject_surfaces, reject_header = clearance._pack_surfaces(REJECT_PACK, target_keys)
    counts = {}
    zero_ids_by_row = {}
    for sid in BACK_ROWS:
        key = (51005, sid)
        require(key in reject_surfaces, 'rejected pack omits expected back row ' + str(sid))
        faces = reject_surfaces[key]['faces']
        zero_ids = lattice_zero_face_ids(reject_positions, faces)
        zero_ids_by_row[sid] = zero_ids
        counts[str(sid)] = {
            'semantic': 51005,
            'stable_id': sid,
            'triangle_count': int(len(faces)),
            'exact_zero_area_face_count': len(zero_ids),
            'exact_zero_area_face_ids_sample_first_8': zero_ids[:8],
        }
    total_zero = sum(item['exact_zero_area_face_count'] for item in counts.values())
    require(total_zero == 1545, 'all-row zero-area count changed')
    require([counts[str(sid)]['exact_zero_area_face_count'] for sid in BACK_ROWS] ==
            [732, 719, 30, 64, 0, 0], 'all-row count vector changed')
    failure = json.loads(FAILURE.read_text())
    require(failure['surface_semantic'] == 51005 and failure['surface_stable_id'] == 153,
            'native rejection is not back row153')
    require(failure['first_invalid_triangle']['triangle_index_in_surface'] in
            zero_ids_by_row[153],
            'failure receipt first face not found by exact lattice audit')

    predicted = np.load(PRED)
    native = np.load(NATIVE_ROW)
    pp = np.asarray(predicted['positions'], dtype='<f4')
    np0 = np.asarray(native['positions'], dtype='<f4')
    pf = np.asarray(predicted['faces'], dtype=np.int64)
    nf = np.asarray(native['faces'], dtype=np.int64)
    require(pp.shape == np0.shape == (4687, 3), 'row157 vertex shape changed')
    require(pf.shape == nf.shape == (9370, 3), 'row157 face shape changed')
    require(np.array_equal(pf, nf), 'row157 topology differs between predicted and native')
    pzero = lattice_zero_face_ids(pp, pf)
    nzero = lattice_zero_face_ids(np0, nf)
    require(pzero == [5187, 7656], 'default-pose row157 false-positive IDs changed')
    require(nzero == [], 'native step-0 row157 is no longer exact-zero-area clear')
    delta = np.asarray(np0, dtype=np.float64) - np.asarray(pp, dtype=np.float64)
    per_vertex = np.linalg.norm(delta, axis=1)
    native_row_audit = json.loads(NATIVE_ROW_JSON.read_text())
    accepted_receipt = json.loads(NATIVE_RECEIPT.read_text())
    fail_native = json.loads(FAILURE.read_text())
    require(native_row_audit['degenerate_face_ids'] == [], 'native audit JSON has degenerate row157 face')
    require(native_row_audit['raw_self_pair_count'] == 47, 'native self-pair diagnostic changed')
    require(int(accepted_receipt.get('accepted_step', 0)) == 0, 'accepted geometry is not step 0')
    metadata = json.loads(NATIVE_METADATA.read_text())
    invocation = json.loads(NATIVE_INVOCATION.read_text())
    invocation_argv = invocation['argv']
    native_dt = float(invocation_argv[invocation_argv.index('--muscle-step-seconds') + 1])
    native_steps = int(invocation_argv[invocation_argv.index('--muscle-step-count') + 1])
    native_duration = native_dt * native_steps

    header_text = HEADER.read_text()
    runner_text = RUNNER.read_text()
    require('require(requestedGeometrySteps.empty()||presentWindow' in header_text,
            'viewer-only geometry-export guard changed')
    require('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS' in header_text,
            'accepted-geometry step selector missing')
    require('duration must be an integer number of native steps' in runner_text,
            'native runner duration/step check changed')
    require('diagnostic without the native viewer or movie' in runner_text,
            'mechanics-only semantics changed')

    pins_end = {str(path): sha256(path) for path in paths}
    require(pins_start == pins_end, 'a pinned input changed during audit')

    report = {
        'schema': 'numi.human.back_preflight_native_false_positive_review.v1',
        'status': 'pass_for_bounded_audit',
        'execution': 'CPU-only exact Float32-lattice and metadata review; no native/GPU execution',
        'pins_unchanged_at_end': True,
        'input_sha256': pins_end,
        'rejected_step0_back_rows': {
            'pack_path': str(REJECT_PACK),
            'pack_sha256': pins_end[str(REJECT_PACK)],
            'pack_vertex_count': int(reject_header['vertices']),
            'pack_primitive_count': int(reject_header['primitives']),
            'exact_zero_area_face_count_total': total_zero,
            'rows': counts,
            'scope_note': 'Counts are for the six back rows in the actual rejected step-0 MRVPack; the first rejection receipt identifies row153 face4040.',
        },
        'row157_default_pose_vs_native_step0': {
            'predicted_path': str(PRED),
            'native_extracted_path': str(NATIVE_ROW),
            'vertices': int(len(pp)),
            'faces': int(len(pf)),
            'face_indices_identical': True,
            'default_pose_predictor_exact_zero_area_face_ids': pzero,
            'accepted_native_step0_exact_zero_area_face_ids': nzero,
            'max_vertex_displacement_m': float(np.max(per_vertex)),
            'rms_vertex_displacement_m': float(math.sqrt(float(np.mean(per_vertex * per_vertex)))),
            'max_axiswise_abs_displacement_m': np.max(np.abs(delta), axis=0).tolist(),
            'predictor_bounds_m': box(pp),
            'native_step0_bounds_m': box(np0),
            'predictor_area_stats': area_stats(pp, pf),
            'native_step0_area_stats': area_stats(np0, nf),
            'native_step0_row_self_pair_count': int(native_row_audit['raw_self_pair_count']),
            'native_step0_row_closed_oriented_manifold_candidate': native_row_audit['topology']['closed_oriented_manifold_candidate'],
            'native_run_duration_s': native_duration,
            'native_run_accepted_steps': native_steps,
            'native_run_exit_code': metadata.get('exit_code'),
            'native_run_accepted_step': accepted_receipt.get('accepted_step'),
            'interpretation': 'The default-pose preflight predicts two exact Float32-lattice zero-area faces, but the same face indices at accepted native step 0 have none. The coordinate sets differ substantially, so this is a cross-pose false rejection, not a same-pose arithmetic discrepancy. This positive control only addresses zero-area face rejection; the native row retains 47 raw self-pair witnesses and is not claimed fully self-clear.',
        },
        'minimum_reliable_existing_native_preflight': {
            'mode': 'existing native Metal viewer path with a single accepted step',
            'environment': {'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS': '0'},
            'runner_parameters': {'seconds': 0.001, 'dt_seconds': 0.001, 'expected_native_steps': 1},
            'mechanics_only': False,
            'reason': 'The existing Metal source allows at most eight explicitly requested export steps, but requires presentWindow for accepted-geometry export. The runner maps --mechanics-only to a non-viewer diagnostic, so this export check must use the normal viewer path. With seconds=dt=0.001, existing runner arithmetic produces one native step. This is a one-step native execution, not geometry-only validation.',
            'source_locations': {
                str(HEADER): {'step_selector_line': 83, 'viewer_export_guard_line': 919},
                str(RUNNER): {'positive_duration_line': 182, 'integer_step_count_line': 188, 'mechanics_only_argv_line': 202, 'dt_argument_line': 334, 'mechanics_only_help_line': 337},
            },
            'gpu_owner': 'root',
        },
    }
    output = OUT / 'report.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    print(json.dumps({
        'report': str(output),
        'status': report['status'],
        'allrow_exact_zero_area_counts': {k: v['exact_zero_area_face_count'] for k, v in counts.items()},
        'total': total_zero,
        'row157_predictor_zero_ids': pzero,
        'row157_native_zero_ids': nzero,
        'row157_native_self_pairs': native_row_audit['raw_self_pair_count'],
        'pins_unchanged_at_end': True,
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

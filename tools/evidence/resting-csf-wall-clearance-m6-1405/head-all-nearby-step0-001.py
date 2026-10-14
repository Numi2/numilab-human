#!/usr/bin/env python3
"""Bounded step-0 all-neighbor source candidate gate for the head/CSF row.

Uses the shared resting anatomy verifier for exact preparation, self, pair, and
containment predicates. The only reciprocal interface rule is the frozen dual
lineage pons-floor proof; all other typed targets are strict.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

A = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D = A / 'full-anatomy-delivery-audit-1378'
W = D / 'pons-fourth-ventricle-reconstruction-001'
ROOT = Path('/Users/n/numi-human-pons-floor-compiler-20261014')
SRC = D / 'forearm492-head016-vessel-seam-integrated-source-002/resting-thorax.nhanatomy'
CAND = W / 'cord-local-clearance-candidate-compile-001/resting-thorax-candidate.nhanatomy'
COMPILE = W / 'cord-local-clearance-candidate-compile-001/compile-report.json'
INTEGRATED = D / 'forearm492-head-cord-vessel-seam-integrated-source-004/resting-anatomy-receipt.json'
PACK = D / 'forearm492-head016-vessel-seam-native-emission-diagnostic-003/native-run/accepted-geometry/step-0.mrvpack'
CAPTURE_RECEIPT = PACK.with_suffix('.receipt.json')
C2 = W / 'cord-local-clearance-candidate-003/candidate-01.npz'
C2_REPORT = W / 'cord-local-clearance-candidate-003/report.json'
FLOOR = W / 'local-floor-candidate-005/pons-fourth-ventricle-floor-candidate.npz'
FLOOR_REPORT = W / 'local-floor-candidate-005/candidate-report.json'
DUAL = W / 'local-floor-candidate-005/dual-lineage.json'
FOCUSED = W / 'local-floor-candidate-005/focused-source-gate-011/report.json'
VERIFIER = ROOT / 'src/numilab_human/resting_anatomy_verify.py'
MODEL = ROOT / 'src/numilab_human/model.py'
OUT = W / 'head-all-nearby-step0-001'
PREFLIGHT = OUT / 'preflight.json'
PROGRESS = OUT / 'progress.json'
REPORT = OUT / 'report.json'
FAILURE = OUT / 'failure.json'

EXPECTED = {
    str(SRC): '2f8af1031b3c5b8cc26866f32935c9b695127a2f357732e566be3f14fdc2666d',
    str(CAND): '9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09',
    str(COMPILE): 'efc7efef96eba68da58cc8f8c7cc94ede25255a1a25a579f4ac11038a1f040c8',
    str(INTEGRATED): '98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d',
    str(PACK): '3b82ed8fff3c4eb2bd4dd8dc1f5ccb276fe97a00cb7aa82dff5db6956c2bbb39',
    str(CAPTURE_RECEIPT): '62deacd6a6ae9c0e14bafdd78538013d9a901e3217fb302b94927036e7ed061b',
    str(C2): 'd9798684d4412fd97093ea056d009e2594cbaa40855cab2e1e82b2b84c5bdab6',
    str(C2_REPORT): '3e04eb8ee9fcc653951b245f285d3a99e84c2b299a1605cb9694018ded5ff33f',
    str(FLOOR): '7c9f5fb245169ce033caf6919c1397b6084acab802c7a745792e282dd024c989',
    str(FLOOR_REPORT): '0c66ba44381b6dc9011b11b4801fd8d4f9ee9b51984345b7fc8a360696aefda3',
    str(DUAL): '65fd020482a41b6d85d2ed91bd4d1bf68cb436487af77e093ab0f0cd0fe56458',
    str(FOCUSED): '66792e15b47c922d4628ed5e00cca07629cda7f82b385d194dbb8474930c9d37',
    str(VERIFIER): 'de87c97732baed438a50cce4856b602b4abed4dc2c05230bc233004b5e84ba60',
    str(MODEL): '29d5664ce1485c520c8ab8cb088f496a82c68e570400030dcaea98b2c9d6450e',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def source_capture_map(key, captured, source_faces):
    positions, capture_faces, normals, gids = captured
    source_faces = np.asarray(source_faces, dtype=np.int64)
    capture_faces = np.asarray(capture_faces, dtype=np.int64)
    gids = np.asarray(gids, dtype=np.uint64)
    need(source_faces.shape == capture_faces.shape, f'{key} source/capture face shape differs')
    global_capture = gids[capture_faces].astype(np.int64)
    offsets = global_capture - source_faces
    need(np.all(offsets == offsets.flat[0]), f'{key} source/capture face corners do not share one GID offset')
    offset = int(offsets.flat[0])
    projected, proof = av.retain_bound_source_faces(
        key, captured, source_faces, offset, np.arange(len(source_faces), dtype=np.int64))
    gid_index = av.gid_index_map(gids)
    used = np.unique(source_faces.reshape(-1))
    local_to_capture = {}
    for local in used:
        gid = int(local) + offset
        need(gid in gid_index, f'{key} referenced source vertex has no capture GID: {local}')
        local_to_capture[int(local)] = gid_index[gid]
    mapped = np.asarray([[local_to_capture[int(v)] for v in tri] for tri in source_faces], dtype=np.int64)
    need(np.array_equal(mapped, capture_faces), f'{key} ordered source face corners do not map to capture slots')
    return offset, local_to_capture, proof


def f32_key(point):
    row = np.asarray(point, dtype=np.float32)
    return tuple(int(v) for v in av.ci.float32_point_lattice_key(row.tolist()))


def f32_point_m(point):
    return tuple(Fraction(v, 1 << 149) for v in f32_key(point))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def on_segment(point, a, b):
    return cross(sub(b, a), sub(point, a)) == (0, 0, 0) and all(
        min(a[i], b[i]) <= point[i] <= max(a[i], b[i]) for i in range(3))


def on_triangle_boundary(point, tri):
    return any(on_segment(point, tri[i], tri[(i+1) % 3]) for i in range(3))


def cyclic_reverse(first, second):
    for shift in range(3):
        expected = (first[shift], first[(shift-1) % 3], first[(shift-2) % 3])
        if tuple(second) == expected:
            return True
    return False


def build_case():
    sys.path.insert(0, str(ROOT / 'src'))
    from numilab_human import resting_anatomy_verify as av_module
    from numilab_human import model as model_module
    from numilab_human.resting_pleura_proxy import _parse_payload
    global av, model
    av, model = av_module, model_module

    got = {str(path): sha(path) for path in map(Path, EXPECTED)}
    need(got == EXPECTED, 'pinned input hash mismatch: ' + json.dumps(
        {k: [EXPECTED[k], got.get(k)] for k in EXPECTED if got.get(k) != EXPECTED[k]}, sort_keys=True))
    need(sha(Path(__file__)) == RUNNER_SHA, 'runner source changed after preflight')

    base_header, base_records, base_rows = _parse_payload(SRC.read_bytes())
    cand_header, cand_records, cand_rows = _parse_payload(CAND.read_bytes())
    need(len(base_rows) == len(cand_rows) == 525 and list(base_rows) == list(cand_rows), 'NHA row inventory changed')
    for sid in (323, 345):
        for rows, label in ((base_rows, 'base'), (cand_rows, 'candidate')):
            row = rows[sid]
            need((int(row['body_index']), int(row['layer'])) == (23, 1), f'{label} typed owner mismatch for row {sid}')
    base323, cand323 = base_rows[323], cand_rows[323]
    base345, cand345 = base_rows[345], cand_rows[345]
    base_xyz = np.asarray(base323['vertices6'][:, :3], dtype=np.float32)
    cand_xyz = np.asarray(cand323['vertices6'][:, :3], dtype=np.float32)
    base_faces = np.asarray(base323['faces'], dtype=np.int64)
    cand_faces = np.asarray(cand323['faces'], dtype=np.int64)
    need(base_xyz.shape == cand_xyz.shape == (9506, 3), 'row323 vertex array shape changed')
    need(np.array_equal(base_faces, cand_faces), 'C2 candidate changed row323 face order')
    need(np.array_equal(np.asarray(base345['vertices6']), np.asarray(cand345['vertices6']))
         and np.array_equal(np.asarray(base345['faces']), np.asarray(cand345['faces'])),
         'C2 source unexpectedly changed Pons row345')

    with np.load(C2, allow_pickle=False) as z:
        changed = np.asarray(z['changed_source_vertex_ids_i64'], dtype=np.int64).copy()
        saved_step0 = np.asarray(z['captured_step0_candidate_xyz_f32'], dtype=np.float32).copy()
    changed_expected = np.flatnonzero(np.any(base_xyz != cand_xyz, axis=1)).astype(np.int64)
    need(np.array_equal(changed, changed_expected), 'candidate changed-vertex IDs do not match compiled NHA delta')
    need(np.all((changed >= 0) & (changed < len(base_xyz))), 'changed vertex ID outside row323')
    used = np.unique(base_faces.reshape(-1))
    need(set(map(int, changed)).issubset(set(map(int, used))), 'changed source IDs include an unused slot')

    integrated = json.loads(INTEGRATED.read_text())
    payload = integrated.get('payload', {})
    integrated_payload_path = Path(payload.get('path', '')).resolve()
    need(payload.get('sha256') == EXPECTED[str(CAND)] and integrated_payload_path.is_file()
         and sha(integrated_payload_path) == EXPECTED[str(CAND)],
         'integrated source-004 receipt payload bytes do not match the compiled C2 NHA')
    qualification = integrated.get('qualification')
    need(isinstance(qualification, str) and qualification.strip(),
         'integrated source receipt lacks its pinned qualification statement')

    with np.load(FLOOR, allow_pickle=False) as floor:
        pons_xyz = np.asarray(floor['pons_vertices6_f32'], dtype=np.float32)
        pons_faces = np.asarray(floor['pons_faces_i64'], dtype=np.int64)
        vent_xyz = np.asarray(floor['fourth_ventricle_vertices6_f32'], dtype=np.float32)
        vent_faces = np.asarray(floor['fourth_ventricle_faces_i64'], dtype=np.int64)
        pons_parent = np.asarray(floor['pons_face_source_parent_i64'], dtype=np.int64)
        need(np.array_equal(pons_xyz, np.asarray(base345['vertices6'], dtype=np.float32)), 'floor source Pons array differs from base row345')
        need(np.array_equal(pons_faces, np.asarray(base345['faces'], dtype=np.int64)), 'floor source Pons face order differs from base row345')
        need(np.array_equal(vent_xyz, np.asarray(base323['vertices6'], dtype=np.float32)), 'floor source ventricle differs from base row323')
        need(np.array_equal(vent_faces, base_faces), 'floor source ventricle face order differs from base row323')
        cap_faces = np.asarray([int(row['cap_output_face']) for row in json.loads(DUAL.read_text())['cap_faces']], dtype=np.int64)
        cap_parent_faces = np.asarray([int(row['pons_output_face']) for row in json.loads(DUAL.read_text())['cap_faces']], dtype=np.int64)
        floor_pons_xyz = pons_xyz[:, :3].copy()
        floor_vent_xyz = vent_xyz[:, :3].copy()
    focused = json.loads(FOCUSED.read_text())
    need(focused.get('status') == 'focused_source_gate_pass_not_motion_or_native_admission', 'frozen pons-floor source gate did not pass')
    interface = focused['interface']
    need(interface['same_rigid_owner_body_index'] == 23 and interface['dual_face_lineage_verified']
         and interface['opposed_cap_orientation_verified'], 'frozen floor proof owner/lineage/orientation changed')
    need(interface['cap_faces'] == 3314 and interface['cap_parent_shared_indexed_feature_count'] == 50531
         and interface['reciprocal_floor_point_junction_count'] == 16596
         and interface['reciprocal_boundary_edge_count'] == 2848,
         'frozen floor proof cardinalities changed')
    dual = json.loads(DUAL.read_text())
    capstart = len(vent_faces) - len(cap_faces)
    need(capstart == 9362 and len(cap_faces) == 3314, 'cap output face range changed')

    capture_receipt = json.loads(CAPTURE_RECEIPT.read_text())
    need(capture_receipt.get('accepted_step') == 0, 'actual reference capture is not step zero')
    body_rows = [r for r in capture_receipt.get('accepted_body_poses', [])
                 if isinstance(r, dict) and r.get('body_index') == 23]
    need(len(body_rows) == 1, 'capture must have exactly one keyed body_index 23 pose')
    body = body_rows[0]
    pos = np.asarray(body.get('position_m'), dtype=np.float64)
    quat = np.asarray(body.get('quaternion_xyzw'), dtype=np.float64)
    need(pos.shape == (3,) and quat.shape == (4,) and np.isfinite(pos).all() and np.isfinite(quat).all(),
         'body23 pose dimensions are invalid')
    rot = np.asarray(model._myosim_matrix_from_quaternion_xyzw(quat.tolist()), dtype=np.float64)
    need(rot.shape == (3, 3) and np.isfinite(rot).all(), 'body23 rotation matrix invalid')

    surfaces, endpoint = av.load_capture(PACK, CAPTURE_RECEIPT)
    need(endpoint.get('accepted_step') == 0 and len(surfaces) == 875, 'actual capture keyset/count mismatch')
    need((51010, 323) in surfaces and (51010, 345) in surfaces, 'actual row323/Pons345 missing')
    vcap = surfaces[(51010, 323)]
    pcap = surfaces[(51010, 345)]
    voffset, vlocal_to_capture, vbind = source_capture_map((51010, 323), vcap, base_faces)
    poffset, plocal_to_capture, pbind = source_capture_map((51010, 345), pcap, np.asarray(base345['faces'], dtype=np.int64))
    used_sorted = np.unique(base_faces.reshape(-1))
    actual_base = np.asarray(vcap[0][[vlocal_to_capture[int(i)] for i in used_sorted]], dtype=np.float64)
    expected_base = base_xyz[used_sorted].astype(np.float64) @ rot.T + pos
    baseline_residual = np.linalg.norm(actual_base - expected_base, axis=1)
    baseline_max = float(np.max(baseline_residual, initial=0.0))
    need(baseline_max <= 2e-6, 'actual step0 row323 differs from source003 rigid rest by >2um')

    candidate_world = np.array(vcap[0], dtype=np.float32, copy=True)
    for source_slot in changed:
        cap_slot = vlocal_to_capture[int(source_slot)]
        delta_local = cand_xyz[int(source_slot)].astype(np.float64) - base_xyz[int(source_slot)].astype(np.float64)
        candidate_world[cap_slot] = (candidate_world[cap_slot].astype(np.float64) + delta_local @ rot.T).astype(np.float32)
    candidate_used_world = candidate_world[[vlocal_to_capture[int(i)] for i in used_sorted]]
    need(saved_step0.shape == candidate_used_world.shape, 'saved C2 step0 coordinates do not match used source slot order')
    step0_delta_error = np.linalg.norm(candidate_used_world.astype(np.float64) - saved_step0.astype(np.float64), axis=1)
    step0_error_max = float(np.max(step0_delta_error, initial=0.0))
    need(step0_error_max <= 2e-6, 'C2 transformed source delta differs from pinned step0 prediction by >2um')

    # Candidate edits must not touch any source vertex in the frozen floor's contact support.
    touched = set()
    for face_id in cap_faces:
        touched.update(map(int, vent_faces[int(face_id)]))
    dual_edges = dual['reciprocal_boundary_edges']
    for row in dual_edges:
        touched.update(map(int, vent_faces[int(row['ventricle_output_face'])]))
    for row in interface['cap_parent_shared_indexed_feature_proofs']:
        touched.update(map(int, vent_faces[int(row['cap_output_face'])]))
    for row in interface['reciprocal_floor_point_junctions']:
        touched.update(map(int, vent_faces[int(row['ventricle_output_face'])]))
    changed_touch = sorted(touched.intersection(map(int, changed)))
    need(not changed_touch, 'C2 changed a vertex in the frozen pons-floor contact support')

    # Verify actual Pons parent overlays against the fresh native capture, not source-coordinate proximity.
    actual_pons = np.asarray(pcap[0], dtype=np.float32)
    actual_vent = candidate_world
    exact_overlay_count = 0
    for cap_face, parent_face in zip(cap_faces, cap_parent_faces):
        ctri = actual_vent[np.asarray([vlocal_to_capture[int(i)] for i in vent_faces[int(cap_face)]], dtype=np.int64)]
        ptri = actual_pons[np.asarray([plocal_to_capture[int(i)] for i in pons_faces[int(parent_face)]], dtype=np.int64)]
        ck = tuple(f32_key(row) for row in ctri)
        pk = tuple(f32_key(row) for row in ptri)
        need(set(ck) == set(pk) and cyclic_reverse(pk, ck),
             f'actual step0 cap-parent overlay/orientation mismatch at {int(cap_face)}:{int(parent_face)}')
        exact_overlay_count += 1
    need(exact_overlay_count == 3314, 'actual cap-parent overlay proof incomplete')

    # Build a deterministic exact AABB target inventory over all 874 other typed surfaces.
    bmin = candidate_world.min(axis=0)
    bmax = candidate_world.max(axis=0)
    overlap = []
    disjoint = []
    for key, target in surfaces.items():
        if key == (51010, 323):
            continue
        tp = np.asarray(target[0], dtype=np.float32)
        lo, hi = tp.min(axis=0), tp.max(axis=0)
        if bool(np.all(bmin <= hi) and np.all(lo <= bmax)):
            overlap.append(key)
        else:
            disjoint.append(key)
    need(len(overlap) + len(disjoint) == 874, 'AABB inventory omitted a typed target')
    # Run the directly declared Pons floor interface first, then the spinal-cord source row,
    # then every remaining overlapping typed target in stable key order.
    order = []
    for key in ((51010, 345), (51012, 12)):
        if key in overlap:
            order.append(key)
    order.extend(sorted(key for key in overlap if key not in order))

    source_feature_counts = {
        'coincident_cap_parent': 3314,
        'cap_parent_shared_indexed_feature': 50531,
        'reciprocal_floor_point_junction': 16596,
        'retained_ventricle_reciprocal_edge': 15236,
    }
    return {
        'pins': got,
        'base_rows': base_rows,
        'candidate_rows': cand_rows,
        'surfaces': surfaces,
        'endpoint': endpoint,
        'capture_receipt': capture_receipt,
        'rot': rot,
        'pos': pos,
        'base_xyz': base_xyz,
        'cand_xyz': cand_xyz,
        'base_faces': base_faces,
        'changed': changed,
        'used_sorted': used_sorted,
        'candidate_world': candidate_world,
        'vlocal_to_capture': vlocal_to_capture,
        'plocal_to_capture': plocal_to_capture,
        'pons_faces': pons_faces,
        'vent_faces': vent_faces,
        'pons_parent': pons_parent,
        'dual': dual,
        'focused': focused,
        'overlap': order,
        'disjoint': sorted(disjoint),
        'preflight_summary': {
            'status': 'preflight_pass_no_exact_geometry_predicates',
            'runner_sha256': RUNNER_SHA,
            'capture_step': 0,
            'typed_surface_count': len(surfaces),
            'tested_changed_identity': [51010, 323],
            'all_target_count': len(surfaces)-1,
            'aabb_overlap_target_count': len(overlap),
            'aabb_disjoint_target_count': len(disjoint),
            'aabb_overlap_targets_ordered': [list(k) for k in order],
            'aabb_disjoint_targets': [list(k) for k in sorted(disjoint)],
            'row323_face_global_offset': voffset,
            'row345_face_global_offset': poffset,
            'row323_ordered_faces_bound': vbind['all_ordered_source_facets_match'],
            'row345_ordered_faces_bound': pbind['all_ordered_source_facets_match'],
            'row323_used_source_vertices': len(used_sorted),
            'row323_actual_emission_max_residual_m': baseline_max,
            'c2_changed_source_vertices': len(changed),
            'c2_actual_step0_delta_max_m': float(np.linalg.norm(candidate_world-vcap[0], axis=1).max()),
            'c2_saved_step0_prediction_max_error_m': step0_error_max,
            'frozen_interface_support_source_vertices': len(touched),
            'changed_vertices_touching_frozen_interface_support': changed_touch,
            'actual_step0_cap_parent_exact_reversed_overlays': exact_overlay_count,
            'frozen_focused_source_interface_counts': source_feature_counts,
            'full_focused_source_report_sha256': EXPECTED[str(FOCUSED)],
            'source004_receipt_sha256': EXPECTED[str(INTEGRATED)],
            'native_or_motion_admission': False,
        },
    }


def feature_points_from_maps(proof, pons_actual, vent_actual, pmap, vmap):
    expected = set()
    for mapping in proof.get('positive_source_feature_maps', []):
        p_local = int(mapping['pons_vertex'])
        need(p_local in pmap, 'frozen Pons feature vertex is absent from actual capture map')
        pslot = pmap[p_local]
        ppoint = pons_actual[pslot]
        for caprow in mapping.get('cap_vertices', []):
            v_local = int(caprow['vertex'])
            need(v_local in vmap, 'frozen cap feature vertex is absent from actual capture map')
            vslot = vmap[v_local]
            vpoint = vent_actual[vslot]
            need(f32_key(ppoint) == f32_key(vpoint), 'actual capture broke a frozen shared cap feature vertex')
            expected.add(f32_point_m(ppoint))
    return expected


def edge_segment_actual(edge_xyz, pons_points, vent_points, pons_actual, vent_actual,
                        pmap, vmap):
    endpoints = []
    for xyz in edge_xyz:
        key = f32_key(xyz)
        p_ids = pons_points.get(key, ())
        v_ids = vent_points.get(key, ())
        candidates = []
        for local in p_ids:
            if local in pmap:
                candidates.append(pons_actual[pmap[local]])
        for local in v_ids:
            if local in vmap:
                candidates.append(vent_actual[vmap[local]])
        need(candidates, 'reciprocal edge endpoint has no source-to-capture vertex map')
        keys = {f32_key(x) for x in candidates}
        need(len(keys) == 1, 'reciprocal edge endpoint emissions disagree across the shared owner')
        endpoints.append(f32_point_m(candidates[0]))
    need(endpoints[0] != endpoints[1], 'reciprocal floor edge collapsed in actual step0 output')
    return tuple(endpoints)


def audit_pons_interface(case, candidate_surface, pons_surface):
    from numilab_human import cardiac_cavity_intersections as ci
    floor_dir = W / 'local-floor-candidate-005'
    with np.load(FLOOR, allow_pickle=False) as z:
        pons_points_xyz = np.asarray(z['pons_vertices6_f32'][:, :3], dtype=np.float32)
        vent_points_xyz = np.asarray(z['fourth_ventricle_vertices6_f32'][:, :3], dtype=np.float32)
        pons_output_faces = np.asarray(z['pons_faces_i64'], dtype=np.int64)
        vent_output_faces = np.asarray(z['fourth_ventricle_faces_i64'], dtype=np.int64)
        pons_parent = np.asarray(z['pons_face_source_parent_i64'], dtype=np.int64)
    interface = case['focused']['interface']
    dual = case['dual']
    pons_actual = pons_surface.positions
    vent_actual = candidate_surface.positions
    pmap = case['plocal_to_capture']
    vmap = case['vlocal_to_capture']
    capstart = len(vent_output_faces) - len(dual['cap_faces'])

    overlay = {}
    for row in dual['cap_faces']:
        key = (int(row['pons_output_face']), int(row['cap_output_face']))
        need(key not in overlay, 'duplicate cap-parent proof pair')
        overlay[key] = row
    feature = {}
    for row in interface['cap_parent_shared_indexed_feature_proofs']:
        key = (int(row['pons_output_face']), int(row['cap_output_face']))
        need(key not in feature, 'duplicate cap shared-feature proof pair')
        feature[key] = row
    junction = defaultdict(list)
    for row in interface['reciprocal_floor_point_junctions']:
        junction[(int(row['pons_output_face']), int(row['ventricle_output_face']))].append(row)
    expected_direct = set(overlay) | set(feature) | set(junction)
    need(len(overlay) == 3314 and len(feature) == 50531 and len(junction) == 16596,
         'frozen floor proof direct pair identities changed')

    # Source-face edge endpoints are mapped to current emitted vertices exactly by Float32 identity.
    pons_point_map = defaultdict(list)
    for local, point in enumerate(pons_points_xyz):
        pons_point_map[f32_key(point)].append(local)
    vent_point_map = defaultdict(list)
    for local, point in enumerate(vent_points_xyz):
        vent_point_map[f32_key(point)].append(local)
    edge_support = defaultdict(list)
    for row in dual['reciprocal_boundary_edges']:
        vface = int(row['ventricle_output_face'])
        psource = int(row['cap_source_pons_face'])
        segment = edge_segment_actual(row['edge_xyz_f32_m'], pons_point_map, vent_point_map,
                                      pons_actual, vent_actual, pmap, vmap)
        edge_support[(psource, vface)].append(segment)

    # Exact contacts come from the shared verifier kernel. Every Pons hit must be a specific
    # source-proven overlay, a positive shared source feature, a reciprocal point, or a point
    # on its exact dual-lineage reciprocal edge.
    result = ci._audit_pair_prepared_first(pons_surface.index, candidate_surface.records, same_surface=False)
    pairs = [(int(a), int(b)) for a, b in result['triangle_pairs']]
    actual_pairs = set(pairs)
    need(expected_direct.issubset(actual_pairs), 'actual capture lost a declared floor source contact pair')
    counts = {'coincident_cap_parent': 0, 'cap_parent_shared_indexed_feature': 0,
              'reciprocal_floor_point_junction': 0, 'retained_ventricle_reciprocal_edge': 0}
    first_failure = None
    for ordinal, (pface, vface) in enumerate(pairs):
        points = pons_surface.intersection_points_meters(candidate_surface, pface, vface)
        if not points:
            first_failure = {'face_pair': [pface, vface], 'reason': 'pair engine returned no exact contact points'}
            break
        key = (pface, vface)
        ptri = tuple(f32_point_m(pons_actual[pons_surface.faces[pface, j]]) for j in range(3))
        vtri = tuple(f32_point_m(vent_actual[candidate_surface.faces[vface, j]]) for j in range(3))
        if key in overlay:
            if set(points) != set(ptri) or set(ptri) != set(vtri) or not cyclic_reverse(ptri, vtri):
                first_failure = {'face_pair': [pface, vface], 'reason': 'actual cap-parent contact is not the exact reversed full triangle',
                                 'points_m': [[str(c) for c in p] for p in points]}
                break
            counts['coincident_cap_parent'] += 1
        elif key in feature:
            expected = feature_points_from_maps(feature[key], pons_actual, vent_actual, pmap, vmap)
            if not expected or set(points) != expected or len(points) not in (1, 2):
                first_failure = {'face_pair': [pface, vface], 'reason': 'actual cap contact exceeds frozen exact shared-feature support',
                                 'expected_feature_points_m': [[str(c) for c in p] for p in sorted(expected)],
                                 'actual_points_m': [[str(c) for c in p] for p in points]}
                break
            counts['cap_parent_shared_indexed_feature'] += 1
        elif key in junction:
            expected = set()
            for item in junction[key]:
                for local in item.get('pons_shared_indexed_vertex_ids', []):
                    local = int(local)
                    need(local in pmap, 'reciprocal point proof vertex lacks actual capture mapping')
                    expected.add(f32_point_m(pons_actual[pmap[local]]))
            if len(points) != 1 or set(points) != expected:
                first_failure = {'face_pair': [pface, vface], 'reason': 'actual point contact is outside the frozen reciprocal vertex junction',
                                 'expected_points_m': [[str(c) for c in p] for p in sorted(expected)],
                                 'actual_points_m': [[str(c) for c in p] for p in points]}
                break
            point = points[0]
            if point not in ptri or point not in vtri:
                first_failure = {'face_pair': [pface, vface], 'reason': 'reciprocal junction is not a vertex of both actual triangles',
                                 'actual_points_m': [[str(c) for c in p] for p in points]}
                break
            counts['reciprocal_floor_point_junction'] += 1
        else:
            psource = int(pons_parent[pface])
            segments = edge_support.get((psource, vface), ())
            if not segments or not all(any(on_segment(point, a, b) for a, b in segments) for point in points):
                first_failure = {'face_pair': [pface, vface], 'pons_source_face': psource,
                                 'reason': 'actual retained-floor contact is not on a declared dual-lineage reciprocal edge',
                                 'actual_points_m': [[str(c) for c in p] for p in points]}
                break
            counts['retained_ventricle_reciprocal_edge'] += 1
        if ordinal and ordinal % 10000 == 0:
            write_atomic(PROGRESS, {'schema': 'numi.human.head_candidate_all_nearby_progress.v1',
                                    'status': 'pons_floor_interface_classification',
                                    'pair_count_total': len(pairs), 'pairs_classified': ordinal,
                                    'classified_counts': counts, 'elapsed_seconds': time.monotonic()-RUN_STARTED})
    if first_failure is not None:
        return {'passed': False, 'triangle_pair_count': len(pairs), 'classified_counts': counts,
                'first_failure': first_failure}
    expected_counts = {'coincident_cap_parent': 3314, 'cap_parent_shared_indexed_feature': 50531,
                       'reciprocal_floor_point_junction': 16596, 'retained_ventricle_reciprocal_edge': 15236}
    if counts != expected_counts or len(pairs) != sum(expected_counts.values()):
        return {'passed': False, 'triangle_pair_count': len(pairs), 'classified_counts': counts,
                'expected_counts': expected_counts, 'first_failure': {'reason': 'actual interface contact counts differ from frozen source proof'}}

    components = pons_surface.quotient_topology['face_components']
    need(len(components) >= 1, 'actual Pons quotient has no face components')
    main_ids = np.asarray(max(components, key=len), dtype=np.int64)
    pons_main = av.prepare_surface((51010, 345), pons_actual, pons_surface.faces[main_ids],
                                   source_vertex_ids=pons_surface.source_vertex_ids)
    vent_in_pons = av._contained_components(candidate_surface, pons_main, inspect_all_faces=True)
    if vent_in_pons.get('status') != 'checked_exact_rational_metres' or vent_in_pons.get('inside_component_witnesses'):
        return {'passed': False, 'triangle_pair_count': len(pairs), 'classified_counts': counts,
                'containment_failure': {'ventricle_outside_main_pons': vent_in_pons}}
    pons_in_vent = av._contained_components(pons_main, candidate_surface, inspect_all_faces=True)
    if pons_in_vent.get('status') != 'checked_exact_rational_metres' or pons_in_vent.get('inside_component_witnesses'):
        return {'passed': False, 'triangle_pair_count': len(pairs), 'classified_counts': counts,
                'containment_failure': {'pons_outside_csf': pons_in_vent}}
    return {'passed': True, 'triangle_pair_count': len(pairs), 'classified_counts': counts,
            'expected_counts': expected_counts, 'declared_interface': 'frozen dual-lineage pons-floor proof only',
            'pons_main_component_face_count': len(main_ids), 'small_pons_component_preserved': True,
            'containment': {'ventricle_outside_main_pons': vent_in_pons,
                            'pons_outside_csf': pons_in_vent}}


def compact_pair_result(result, target_surface, candidate_surface):
    out = {k: v for k, v in result.items()
           if k not in ('unexpected_triangle_pairs', 'declared_interface_triangle_pairs',
                        'declared_interface_feature_contacts', 'rejected_interface_feature_contacts')}
    unexpected = result.get('unexpected_triangle_pairs', [])
    out['unexpected_pair_count'] = len(unexpected)
    if unexpected:
        pair = tuple(map(int, unexpected[0]))
        out['first_unexpected_pair'] = list(pair)
        out['first_unexpected_points_m'] = [[str(v) for v in p]
            for p in candidate_surface.intersection_points_meters(target_surface, pair[0], pair[1])]
    return out


def run(max_seconds=1200):
    global RUN_STARTED
    if REPORT.exists() or PROGRESS.exists() or FAILURE.exists():
        raise SystemExit('preserve prior outputs; this output directory has already been used')
    saved_pre = json.loads(PREFLIGHT.read_text())
    need(saved_pre.get('status') == 'preflight_pass_no_exact_geometry_predicates', 'preflight did not pass')
    need(saved_pre.get('runner_sha256') == RUNNER_SHA, 'preflight is bound to different runner source')
    RUN_STARTED = time.monotonic()
    deadline = RUN_STARTED + max_seconds
    def stop(_signum, _frame):
        raise TimeoutError(f'total all-neighbor gate exceeded {max_seconds}s')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max_seconds)
    progress = {'schema': 'numi.human.head_candidate_all_nearby_progress.v1', 'status': 'started',
                'runner_sha256': RUNNER_SHA, 'pins': EXPECTED, 'stages': []}
    write_atomic(PROGRESS, progress)
    def stage(name, **fields):
        row = {'stage': name, 'elapsed_seconds': time.monotonic()-RUN_STARTED, **fields}
        progress['stages'].append(row); progress['status'] = name; write_atomic(PROGRESS, progress)
        if time.monotonic() > deadline:
            raise TimeoutError('all-neighbor deadline reached')
    try:
        case = build_case()
        # Rebind exact candidate generation to preflight before any topology predicate.
        need(case['preflight_summary']['aabb_overlap_targets_ordered'] == saved_pre['aabb_overlap_targets_ordered'],
             'preflight target inventory changed')
        stage('pinned_source_and_capture_loaded', source_binding_rows=2,
              all_typed_targets=len(case['surfaces'])-1,
              aabb_overlap_targets=len(case['overlap']), aabb_disjoint_targets=len(case['disjoint']))
        candidate = av.prepare_surface((51010, 323), case['candidate_world'],
                                       np.asarray(case['surfaces'][(51010, 323)][1], dtype=np.int64),
                                       source_vertex_ids=np.asarray(case['surfaces'][(51010, 323)][3]))
        self_result = candidate.self_audit()
        stage('candidate_self', count=self_result['count'], face_pair_count=len(self_result.get('triangle_pairs', [])))
        if self_result['count']:
            failure = {'failed_stage': 'candidate_self', 'typed_identity': [51010, 323],
                       'self_audit': self_result, 'status': 'first_geometry_failure'}
            write_atomic(FAILURE, failure); progress['status'] = 'failed_first_stage'; write_atomic(PROGRESS, progress); return 1

        results = []
        surfaces = case['surfaces']
        for target_key in case['overlap']:
            target_data = surfaces[target_key]
            target = av.prepare_surface(target_key, target_data[0], target_data[1],
                                        normals=None, source_vertex_ids=target_data[3])
            if target_key == (51010, 345):
                pair_result = audit_pons_interface(case, candidate, target)
                summary = pair_result
                if not pair_result['passed']:
                    failure = {'failed_stage': 'pons_floor_interface_and_containment',
                               'tested_identity': [51010, 323], 'target_identity': list(target_key),
                               'result': summary, 'status': 'first_geometry_failure'}
                    write_atomic(FAILURE, failure); stage('failed_first_target', target=list(target_key),
                                                           reason='pons-floor interface/classification or containment failed')
                    return 1
            else:
                pair_result = av.audit_surface_pair(candidate, target)
                summary = compact_pair_result(pair_result, target, candidate)
                if not pair_result.get('passed', False):
                    failure = {'failed_stage': 'strict_all_neighbor_pair',
                               'tested_identity': [51010, 323], 'target_identity': list(target_key),
                               'result': summary, 'status': 'first_geometry_failure'}
                    write_atomic(FAILURE, failure); stage('failed_first_target', target=list(target_key),
                                                           triangle_pair_count=pair_result.get('triangle_pair_count'),
                                                           unexpected_pair_count=summary.get('unexpected_pair_count'),
                                                           containment=pair_result.get('containment'))
                    return 1
            results.append({'target': list(target_key), 'passed': True,
                            'triangle_pair_count': summary.get('triangle_pair_count', 0),
                            'aabb_disjoint': False,
                            'classification_counts': summary.get('classified_counts'),
                            'containment_statuses': [r.get('status') for r in summary.get('containment', [])]})
            stage('target_passed', target=list(target_key), completed_targets=len(results),
                  total_overlap_targets=len(case['overlap']), triangle_pair_count=summary.get('triangle_pair_count', 0))
            del target

        report = {
            'schema': 'numi.human.head_candidate_all_nearby_step0.v1',
            'status': 'source_candidate_step0_all_nearby_pass_not_native_or_motion_admission',
            'candidate_admission': False, 'native_admission': False,
            'scope': 'current actual source003 accepted step0 geometry plus the row323 C2 delta; all other captured geometry left byte-exact',
            'tested_identity': [51010, 323], 'typed_target_universe_count': 874,
            'aabb_disjoint_target_count': len(case['disjoint']),
            'aabb_disjoint_targets': [list(k) for k in case['disjoint']],
            'aabb_overlap_target_count': len(case['overlap']),
            'aabb_overlap_targets': results,
            'candidate_self': self_result,
            'pins': EXPECTED,
            'preflight_sha256': sha(PREFLIGHT),
            'runner_sha256': RUNNER_SHA,
            'elapsed_seconds': time.monotonic()-RUN_STARTED,
            'limitations': ['step0 actual-reference check only', 'nine-pose gate was row323 versus row12 only',
                            'no all-neighbor motion captures', 'no native candidate run', 'no clinical or individual measurement claim'],
        }
        write_atomic(REPORT, report)
        progress['status'] = report['status']; progress['elapsed_seconds'] = report['elapsed_seconds']; write_atomic(PROGRESS, progress)
        return 0
    except BaseException as exc:
        failure = {'status': 'failed_or_interrupted', 'failed_stage': progress.get('status'),
                   'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.monotonic()-RUN_STARTED,
                   'pins': EXPECTED, 'runner_sha256': RUNNER_SHA}
        write_atomic(FAILURE, failure)
        progress['status'] = 'failed_or_interrupted'; progress['error'] = failure['error']; write_atomic(PROGRESS, progress)
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight-only', action='store_true')
    group.add_argument('--execute', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        if PREFLIGHT.exists():
            raise SystemExit('preserve prior preflight; output path already used')
        case = build_case()
        write_atomic(PREFLIGHT, case['preflight_summary'])
        print(json.dumps({'status': case['preflight_summary']['status'], 'preflight': str(PREFLIGHT),
                          'preflight_sha256': sha(PREFLIGHT), 'runner_sha256': RUNNER_SHA,
                          'overlap_target_count': len(case['overlap'])}, indent=2))
    else:
        raise SystemExit(run(args.max_seconds))


RUNNER_SHA = sha(Path(__file__))
if __name__ == '__main__':
    main()

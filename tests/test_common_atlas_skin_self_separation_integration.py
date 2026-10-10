from __future__ import annotations

import numpy as np
import pytest

import numilab_human.common_atlas_skin_clearance as clearance
from numilab_human.model import ImportError


def _clearance_case(*, sparse_source_ids=False):
    base_source = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0],
        [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0],
    ], dtype="<f4")
    if sparse_source_ids:
        global_ids = np.array([2, 4, 6, 7, 9, 11, 0, 1, 3], dtype=np.int64)
        source = np.zeros((12, 3), dtype="<f4")
        source[global_ids] = base_source
    else:
        global_ids = np.arange(len(base_source), dtype=np.int64)
        source = base_source.copy()
    faces = np.array([
        global_ids[[0, 1, 2]], global_ids[[3, 4, 5]], global_ids[[6, 7, 8]],
    ], dtype=np.int64)
    referenced = np.unique(faces)
    compact_faces = np.searchsorted(referenced, faces)
    maps = np.broadcast_to(np.eye(3), (2, len(referenced), 3, 3)).copy()
    captured = np.repeat(source[referenced][None, :, :], 2, axis=0).astype("<f4").astype(np.float64)
    targets = {
        (pose, "active"): np.array([
            [0.3, -0.1, -0.2], [0.3, 0.8, 0.2], [0.3, 0.1, -0.2],
        ], dtype="<f4").astype(np.float64)
        for pose in range(2)
    }
    targets.update({
        (pose, "ocular"): np.array([
            [5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0],
        ], dtype="<f4").astype(np.float64)
        for pose in range(2)
    })

    def scan(pose, world):
        skin = clearance._exact_surface_records(world, compact_faces)
        rows = {}
        for key in ("active", "ocular"):
            target = clearance._exact_surface_records(targets[(pose, key)], np.array([[0, 1, 2]]))
            pair = clearance._audit_pair(skin, target, same_surface=False)
            rows[key] = {
                "triangle_pairs": pair["triangle_pairs"],
                "count": int(pair["count"]),
                "aabb_candidate_pairs": int(pair["aabb_candidate_pairs"]),
                "degenerate_face_rows": [],
            }
        return rows

    baseline = [scan(pose, captured[pose]) for pose in range(2)]
    args = dict(
        source_positions=source,
        faces=faces,
        jacobians_by_pose=maps,
        accepted_skin_world_by_pose=captured,
        baseline_target_audits_by_pose=baseline,
        baseline_skin_self_pairs_by_pose=[0, 0],
        source_outward_face_signs=np.ones(len(faces), dtype=np.int8),
        scan_candidate_targets=scan,
        target_triangle_by_row=lambda pose, key, row: targets[(pose, key)],
        all_target_keys={"active", "ocular"},
        ocular_monitor_keys={"ocular"},
        fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
        bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        max_iterations=1,
        backtrack_count=1,
    )
    return args


def _fake_self_audit(pairs):
    return {
        "triangle_pairs": [[int(a), int(b)] for a, b in pairs],
        "count": len(pairs),
        "degenerate_face_rows": [],
        "aabb_candidate_pairs": 0,
    }


def _fake_proposal_record(kwargs, proposed):
    protected = np.unique(np.concatenate((
        kwargs["required_target_seed_vertex_ids"],
        kwargs["active_target_face_vertex_ids"],
        kwargs["fixed_support_vertex_ids"],
        kwargs["preserved_anchor_vertex_ids"],
    )))
    return {
        "status": "bounded_local_linearized_proposal",
        "proposal_source_positions_m_f32": proposed,
        "movable_vertex_ids": kwargs["movable_vertex_ids"],
        "protected_vertex_ids": protected,
        "constraint_rows_by_pose": [9] * len(kwargs["pose_samples"]),
        "pair_pose_receipts": [
            {"pose_index": sample_index, "face_pair": pair.tolist()}
            for sample_index, sample in enumerate(kwargs["pose_samples"])
            for pair in np.asarray(sample["self_pair_rows"], dtype=np.int64)
        ],
    }


def test_callsite_refuses_self_patch_touching_active_target_seed(monkeypatch):
    args = _clearance_case()
    args["fixed_source_vertex_ids"] = np.array([3, 4, 5], dtype=np.int64)
    scan_progress = []
    helper_calls = []

    def fake_incremental(**kwargs):
        return _fake_self_audit([(0, 1)])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        raise AssertionError("protected self-pair should be rejected before the proposal helper")

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)
    args["scan_progress_callback"] = scan_progress.append

    with pytest.raises(ImportError, match="could not admit iteration=1.*neither face.*eligible"):
        clearance.derive_shared_multipose_inferred_clearance(**args)

    assert helper_calls == []
    assert scan_progress == []


def test_callsite_rechecks_every_pose_after_local_proposal_before_target_scans(monkeypatch):
    args = _clearance_case()
    scan_progress = []
    target_calls = []
    helper_calls = []
    audit_call = 0
    original_scan = args["scan_candidate_targets"]

    def scan(pose, world):
        target_calls.append(pose)
        return original_scan(pose, world)

    def fake_incremental(**kwargs):
        nonlocal audit_call
        slot = audit_call % 4
        audit_call += 1
        # For one trial: initial pose 0 fails, initial pose 1 is clear;
        # after the local source change pose 0 is clear but pose 1 fails.
        if slot == 0 or slot == 3:
            return _fake_self_audit([(1, 2)])
        return _fake_self_audit([])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        proposed = np.asarray(kwargs["source_positions_m"], dtype="<f4").copy()
        proposed[5, 2] += np.float32(1.0e-5)
        return _fake_proposal_record(kwargs, proposed)

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)
    args["scan_candidate_targets"] = scan
    args["scan_progress_callback"] = scan_progress.append

    with pytest.raises(ImportError, match="could not admit iteration=1.*pose=1"):
        clearance.derive_shared_multipose_inferred_clearance(**args)

    assert len(helper_calls) == 1
    assert helper_calls[0]["self_patch_seed_vertex_ids"].tolist() == [3, 4, 5]
    assert not set(helper_calls[0]["self_patch_seed_vertex_ids"]) & set(
        helper_calls[0]["active_target_face_vertex_ids"])
    assert target_calls == []
    assert scan_progress == []


def test_callsite_reports_only_the_locally_repaired_source_to_target_scan(monkeypatch):
    import hashlib

    args = _clearance_case()
    scan_progress = []
    helper_calls = []
    audit_call = 0
    original_scan = args["scan_candidate_targets"]

    def scan(pose, world):
        return original_scan(pose, world)

    def fake_incremental(**kwargs):
        nonlocal audit_call
        slot = audit_call % 4
        audit_call += 1
        if slot == 0:
            return _fake_self_audit([(1, 2)])
        return _fake_self_audit([])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        proposed = np.asarray(kwargs["source_positions_m"], dtype="<f4").copy()
        proposed[5, 2] += np.float32(1.0e-5)
        return {
            **_fake_proposal_record(kwargs, proposed),
            "constraint_rows_by_pose": [9],
            "requested_max_vertex_l2_increment_mm": 0.01,
            "float32_max_vertex_l2_increment_mm": 0.01,
            "post_rounding_linearized_minimum_sat_slack_mm": 0.0,
            "post_rounding_predicted_supplied_pair_exact_count": 0,
        }

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)
    args["scan_candidate_targets"] = scan
    args["scan_progress_callback"] = scan_progress.append

    result, report = clearance.derive_shared_multipose_inferred_clearance(**args)

    assert report["final_nonocular_pair_count_by_pose"] == [0, 0]
    assert len(helper_calls) == 1
    assert scan_progress[0]["event"] == "trial_started"
    expected = helper_calls[0]["source_positions_m"].copy()
    expected[5, 2] += np.float32(1.0e-5)
    observed = np.asarray(scan_progress[0]["source_positions_f32"], dtype="<f4")
    assert np.array_equal(observed, expected)
    assert scan_progress[0]["source_positions_f32_sha256"] == hashlib.sha256(observed.tobytes()).hexdigest()
    assert [row["pose_index"] for row in scan_progress[1:]] == [0, 1]
    assert np.array_equal(result.astype("<f4"), observed)


def test_callsite_moves_only_eligible_side_when_other_pair_face_is_protected(monkeypatch):
    args = _clearance_case()
    scan_progress = []
    helper_calls = []
    audit_call = 0

    def fake_incremental(**kwargs):
        nonlocal audit_call
        # A single trial: pair 0/1 fails at pose 0, then both poses clear
        # after the local proposal. Face 0 is an active target core; face 1
        # is the sole eligible movable side.
        slot = audit_call % 4
        audit_call += 1
        return _fake_self_audit([(0, 1)] if slot == 0 else [])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        proposed = np.asarray(kwargs["source_positions_m"], dtype="<f4").copy()
        proposed[4, 2] += np.float32(1.0e-5)
        return {
            **_fake_proposal_record(kwargs, proposed),
            "constraint_rows_by_pose": [9],
            "requested_max_vertex_l2_increment_mm": 0.01,
            "float32_max_vertex_l2_increment_mm": 0.01,
            "post_rounding_linearized_minimum_sat_slack_mm": 0.0,
            "post_rounding_predicted_supplied_pair_exact_count": 0,
        }

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)
    args["scan_progress_callback"] = scan_progress.append

    result, report = clearance.derive_shared_multipose_inferred_clearance(**args)

    assert len(helper_calls) == 1
    call = helper_calls[0]
    assert call["self_patch_seed_vertex_ids"].tolist() == [3, 4, 5]
    assert not set(call["movable_vertex_ids"].tolist()) & {0, 1, 2}
    assert set(call["required_target_seed_vertex_ids"].tolist()) & {0, 1, 2}
    assert report["final_nonocular_pair_count_by_pose"] == [0, 0]
    details = report["iterations"][0]["local_self_separation"]
    selection = details["pair_seed_face_selection"][0]
    assert selection["triangle_pair"] == [0, 1]
    assert selection["eligible_seed_faces"] == [1]
    assert selection["selected_seed_face"] == 1
    assert details["protected_pair_side_face_rows"] == [0]
    assert scan_progress[0]["event"] == "trial_started"
    assert np.array_equal(result[0:3], call["source_positions_m"][0:3])


def test_pair_side_selection_maps_compact_rows_to_sparse_global_source_ids(monkeypatch):
    args = _clearance_case(sparse_source_ids=True)
    helper_calls = []
    audit_call = 0

    def fake_incremental(**kwargs):
        nonlocal audit_call
        slot = audit_call % 4
        audit_call += 1
        return _fake_self_audit([(0, 1)] if slot == 0 else [])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        proposed = np.asarray(kwargs["source_positions_m"], dtype="<f4").copy()
        proposed[9, 2] += np.float32(1.0e-5)
        return {
            **_fake_proposal_record(kwargs, proposed),
            "constraint_rows_by_pose": [9],
            "requested_max_vertex_l2_increment_mm": 0.01,
            "float32_max_vertex_l2_increment_mm": 0.01,
            "post_rounding_linearized_minimum_sat_slack_mm": 0.0,
            "post_rounding_predicted_supplied_pair_exact_count": 0,
        }

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)
    result, report = clearance.derive_shared_multipose_inferred_clearance(**args)

    assert len(helper_calls) == 1
    call = helper_calls[0]
    assert call["self_patch_seed_vertex_ids"].tolist() == [7, 9, 11]
    assert set(call["movable_vertex_ids"].tolist()).isdisjoint({2, 4, 6})
    assert np.array_equal(result[[2, 4, 6]], call["source_positions_m"][[2, 4, 6]])
    selection = report["iterations"][0]["local_self_separation"]["pair_seed_face_selection"][0]
    assert selection["selected_seed_face"] == 1


def test_callsite_refuses_seed_shared_with_another_protected_pair_side(monkeypatch):
    args = _clearance_case()
    original_source = args["source_positions"]
    extra = np.array([
        [2.0, -1.0, 0.0], [2.0, 0.0, 1.0],
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
    ], dtype="<f4")
    args["source_positions"] = np.vstack((original_source, extra))
    args["faces"] = np.array([
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [3, 9, 10], [11, 12, 13],
    ], dtype=np.int64)
    # Rebuild compact pose data and scan closure for the extended source mesh.
    referenced = np.unique(args["faces"])
    compact_faces = np.searchsorted(referenced, args["faces"])
    args["jacobians_by_pose"] = np.broadcast_to(
        np.eye(3), (2, len(referenced), 3, 3),
    ).copy()
    args["accepted_skin_world_by_pose"] = np.repeat(
        args["source_positions"][referenced][None, :, :], 2, axis=0,
    ).astype("<f4").astype(np.float64)
    original_targets = {
        (pose, key): args["target_triangle_by_row"](pose, key, 0)
        for pose in range(2) for key in ("active", "ocular")
    }

    def scan(pose, world):
        skin = clearance._exact_surface_records(world, compact_faces)
        rows = {}
        for key in ("active", "ocular"):
            target = clearance._exact_surface_records(
                original_targets[(pose, key)], np.array([[0, 1, 2]]),
            )
            pair = clearance._audit_pair(skin, target, same_surface=False)
            rows[key] = {
                "triangle_pairs": pair["triangle_pairs"],
                "count": int(pair["count"]),
                "aabb_candidate_pairs": int(pair["aabb_candidate_pairs"]),
                "degenerate_face_rows": [],
            }
        return rows

    args["baseline_target_audits_by_pose"] = [
        scan(pose, args["accepted_skin_world_by_pose"][pose]) for pose in range(2)
    ]
    args["source_outward_face_signs"] = np.ones(len(args["faces"]), dtype=np.int8)
    args["scan_candidate_targets"] = scan
    args["target_triangle_by_row"] = lambda pose, key, row: original_targets[(pose, key)]
    args["fixed_source_vertex_ids"] = np.array([9], dtype=np.int64)
    helper_calls = []
    audit_call = 0

    def fake_incremental(**kwargs):
        nonlocal audit_call
        slot = audit_call % 4
        audit_call += 1
        return _fake_self_audit([(0, 1), (3, 4)] if slot == 0 else [])

    def fake_proposal(**kwargs):
        helper_calls.append(kwargs)
        raise AssertionError("a seed overlapping a protected pair-side must reject before solve")

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)

    with pytest.raises(
        ImportError,
        match="could not admit iteration=1.*overlap vertices of a protected pair-side face",
    ):
        clearance.derive_shared_multipose_inferred_clearance(**args)

    assert helper_calls == []


def test_helper_sample_receipts_are_remapped_to_actual_accepted_pose(monkeypatch):
    args = _clearance_case()
    audit_call = 0

    def fake_incremental(**kwargs):
        nonlocal audit_call
        slot = audit_call % 4
        audit_call += 1
        # Only accepted pose 1 fails in the initial self check.
        return _fake_self_audit([(1, 2)] if slot == 1 else [])

    def fake_proposal(**kwargs):
        proposed = np.asarray(kwargs["source_positions_m"], dtype="<f4").copy()
        proposed[5, 2] += np.float32(1.0e-5)
        return {
            **_fake_proposal_record(kwargs, proposed),
            "constraint_rows_by_pose": [11],
            "requested_max_vertex_l2_increment_mm": 0.01,
            "float32_max_vertex_l2_increment_mm": 0.01,
            "post_rounding_linearized_minimum_sat_slack_mm": 0.0,
            "post_rounding_predicted_supplied_pair_exact_count": 0,
        }

    monkeypatch.setattr(clearance, "audit_incremental_skin_self_intersections", fake_incremental)
    monkeypatch.setattr(clearance, "_propose_local_self_separation_increment", fake_proposal)

    _, report = clearance.derive_shared_multipose_inferred_clearance(**args)
    details = report["iterations"][0]["local_self_separation"]
    assert details["failure_pose_indices"] == [1]
    assert details["constraint_rows_by_sample"] == [11]
    assert details["constraint_rows_by_pose"] == [0, 11]
    assert details["pair_pose_receipts"] == [{
        "pose_index": 1,
        "sample_index": 0,
        "face_pair": [1, 2],
    }]

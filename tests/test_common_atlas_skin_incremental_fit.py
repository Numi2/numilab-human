"""Exact self-audit cache follows accepted fitter state across backtracking."""
import hashlib

import numpy as np
import pytest

from numilab_human import common_atlas_skin_clearance as c
from numilab_human.model import ImportError


def test_backtracking_self_audit_reuses_only_accepted_geometry(monkeypatch):
    source = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.7, 0, .1], [.7, .2, .1], [.7, .05, .3],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    captures = np.stack([source, source])
    maps = np.broadcast_to(np.eye(3), (2, len(source), 3, 3)).copy()
    target_faces = np.array([[0, 1, 2]], dtype=np.int64)
    targets = {
        "1:1": np.array([[.2, .05, -.04], [.2, .2, .04], [.2, .1, -.04]], dtype="<f4"),
        "1:2": np.array([[.3, .05, -.2], [.3, .2, .2], [.3, .1, -.2]], dtype="<f4"),
        "2:1": np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype="<f4"),
    }

    def full_self(world):
        records = c._exact_surface_records(world, faces)
        return c._audit_pair(records, records, same_surface=True)

    def scan(pose, world):
        records = c._exact_surface_records(world, faces)
        return {
            key: c._audit_pair(
                records, c._exact_surface_records(triangle, target_faces), same_surface=False,
            )
            for key, triangle in targets.items()
        }

    expected_current = source.copy()
    checked_baselines = []
    checked_candidates = []
    events = []
    implementation = c.audit_incremental_skin_self_intersections

    def checked_incremental(**kwargs):
        # Both poses have identity maps. A rejected candidate must not replace
        # this reference; only the fitter's all-gates acceptance event does.
        np.testing.assert_array_equal(kwargs["baseline_world_positions"], expected_current)
        result = implementation(**kwargs)
        complete = full_self(kwargs["candidate_world_positions"])
        assert result["triangle_pairs"] == complete["triangle_pairs"]
        checked_baselines.append(result["baseline_world_f32_sha256"])
        checked_candidates.append(result["count"])
        return result

    def progress(event):
        nonlocal expected_current
        events.append(event)
        if event["status"] == "accepted":
            expected_current = event["source_positions_f32"].copy()

    monkeypatch.setattr(c, "audit_incremental_skin_self_intersections", checked_incremental)
    baseline = [scan(pose, captures[pose]) for pose in range(2)]
    assert [sum(row["count"] for row in audit.values()) for audit in baseline] == [2, 2]
    assert full_self(source)["count"] == 0

    # Full and half increments hit the second skin face. A quarter increment
    # clears the low target and is accepted; the remaining target cannot be
    # cleared through that face, so the next iteration must still reject.
    with pytest.raises(ImportError, match="could not admit iteration=2"):
        c.derive_shared_multipose_inferred_clearance(
            source_positions=source, faces=faces, jacobians_by_pose=maps,
            accepted_skin_world_by_pose=captures,
            baseline_target_audits_by_pose=baseline,
            baseline_skin_self_pairs_by_pose=[0, 0],
            source_outward_face_signs=np.ones(2, dtype=np.int8),
            scan_candidate_targets=scan,
            target_triangle_by_row=lambda pose, key, row: targets[key],
            all_target_keys=set(targets), ocular_monitor_keys={"2:1"},
            fixed_source_vertex_ids=np.array([3, 4, 5], dtype=np.int64),
            bed_plane_origins_by_pose=np.array([[0, 0, -1], [0, 0, -1]], dtype=float),
            bed_plane_normals_by_pose=np.array([[0, 0, 1], [0, 0, 1]], dtype=float),
            max_iterations=2, backtrack_count=7, progress_callback=progress,
        )

    accepted = [event for event in events if event["status"] == "accepted"]
    assert len(accepted) == 1
    assert accepted[0]["nonocular_pair_counts_candidate_by_pose"] == [1, 1]
    # Rejection wording may distinguish repair eligibility from exact scans.
    # The recorded intersecting pair, not that wording, proves this trial failed
    # the self-intersection gate without replacing the accepted baseline.
    assert any(
        event["status"] == "rejected"
        and any(row["triangle_pairs"] == [[0, 1]]
                for row in (event.get("candidate_diagnostics") or {}).get("self_pair_failures", []))
        for event in events
    )
    assert 1 in checked_candidates and 0 in checked_candidates
    expected_hashes = {
        hashlib.sha256(source.tobytes()).hexdigest(),
        hashlib.sha256(accepted[0]["source_positions_f32"].tobytes()).hexdigest(),
    }
    assert set(checked_baselines) == expected_hashes

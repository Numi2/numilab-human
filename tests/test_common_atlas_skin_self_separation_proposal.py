from __future__ import annotations

import numpy as np
import pytest

import numilab_human.common_atlas_skin_clearance as clearance
from numilab_human.common_atlas_skin_clearance import _propose_local_self_separation_increment
from numilab_human.model import ImportError


def _fixture(*, rejected_z: tuple[float, float, float] = (-0.00025, 0.00025, -0.00025),
             baseline_translation_m: float = 0.0005, world_origin_z: float = 0.0,
             source_world_offset_z: float = 0.0):
    baseline = np.array([
        [0.0, 0.0, world_origin_z], [1.0, 0.0, world_origin_z], [0.0, 1.0, world_origin_z],
        [0.1, 0.1, world_origin_z + rejected_z[0] + baseline_translation_m],
        [0.8, 0.1, world_origin_z + rejected_z[1] + baseline_translation_m],
        [0.1, 0.8, world_origin_z + rejected_z[2] + baseline_translation_m],
    ], dtype="<f4")
    rejected = baseline.copy()
    rejected[3:, 2] = world_origin_z + np.asarray(rejected_z, dtype="<f4")
    source = rejected.copy()
    source[:, 2] += np.float32(source_world_offset_z)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    jac = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 6, axis=0)
    pose = {
        "vertex_ids": np.arange(6, dtype=np.int64),
        "baseline_world_positions": baseline,
        "rejected_world_positions": rejected,
        "jacobians": jac,
        "self_pair_rows": np.array([[0, 1]], dtype=np.int64),
    }
    args = {
        "source_positions_m": source,
        "faces": faces,
        "movable_vertex_ids": np.array([3, 4, 5], dtype=np.int64),
        "self_patch_seed_vertex_ids": np.array([3, 4, 5], dtype=np.int64),
        "required_target_seed_vertex_ids": np.array([0], dtype=np.int64),
        "active_target_face_vertex_ids": np.array([0, 1, 2], dtype=np.int64),
        "fixed_support_vertex_ids": np.array([1], dtype=np.int64),
        "preserved_anchor_vertex_ids": np.array([2], dtype=np.int64),
        "pose_samples": [pose],
    }
    return args, baseline, rejected


def test_local_self_separation_uses_millimeters_positive_sat_direction_and_freezes_protected_rows():
    args, _, _ = _fixture()
    source_before = args["source_positions_m"].copy()
    result = _propose_local_self_separation_increment(**args)

    assert result["status"] == "bounded_local_linearized_proposal"
    proposed = result["proposal_source_positions_m_f32"]
    actual = result["source_increment_mm_f32_applied"]
    assert np.all(actual[:, 2] > 0.0)
    assert actual[0, 2] >= 0.5 - 2e-4 and actual[2, 2] >= 0.5 - 2e-4
    assert 0.0 < actual[1, 2] < 0.5
    assert 0.49 <= result["float32_max_vertex_l2_increment_mm"] <= 0.51
    component_cap = clearance._LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM
    assert np.max(np.abs(result["source_increment_mm_f32_applied"])) <= (
        component_cap + result["float32_component_bound_rounding_tolerance_mm_max"] + 1.0e-10
    )
    assert result["constraint_rows_by_pose"] == [9]
    assert result["pair_pose_receipts"][0]["accepted_sat_gap_mm"] == pytest.approx(0.25, abs=1e-6)
    assert result["post_rounding_linearized_minimum_sat_slack_mm"] >= -1e-7
    assert proposed[[0, 1, 2]].tobytes() == source_before[[0, 1, 2]].tobytes()
    assert args["source_positions_m"].tobytes() == source_before.tobytes()
    assert result["protected_vertex_ids"].tolist() == [0, 1, 2]
    assert result["exact_forward_or_full_mesh_acceptance_performed"] is False


def test_local_self_separation_combines_constraints_from_every_failing_pose():
    args, _, _ = _fixture()
    harder = args["pose_samples"][0]
    easier = {**harder, "rejected_world_positions": harder["rejected_world_positions"].copy()}
    easier["rejected_world_positions"][3:, 2] = np.array([-0.0001, 0.0004, -0.0001], dtype="<f4")
    args["pose_samples"] = [harder, easier]

    result = _propose_local_self_separation_increment(**args)

    assert result["constraint_rows_by_pose"] == [9, 9]
    assert len(result["pair_pose_receipts"]) == 2
    # The first pose requires +0.5 mm, while the second needs only +0.35 mm.
    moved = result["source_increment_mm_f32_applied"][:, 2]
    assert moved[0] >= 0.5 - 2e-4 and moved[2] >= 0.5 - 2e-4
    assert 0.0 < moved[1] < 0.5
    assert result["post_rounding_linearized_minimum_sat_slack_mm"] >= -1e-7


@pytest.mark.parametrize(
    ("protected_field", "protected_id"),
    [
        ("required_target_seed_vertex_ids", 0),
        ("active_target_face_vertex_ids", 0),
        ("fixed_support_vertex_ids", 1),
        ("preserved_anchor_vertex_ids", 2),
    ],
)
def test_local_self_separation_rejects_movable_required_or_fixed_ids(protected_field, protected_id):
    args, _, _ = _fixture()
    args["movable_vertex_ids"] = np.array([protected_id, 3, 4, 5], dtype=np.int64)
    args["required_target_seed_vertex_ids"] = np.empty(0, dtype=np.int64)
    args["active_target_face_vertex_ids"] = np.empty(0, dtype=np.int64)
    args["fixed_support_vertex_ids"] = np.empty(0, dtype=np.int64)
    args["preserved_anchor_vertex_ids"] = np.empty(0, dtype=np.int64)
    args[protected_field] = np.array([protected_id], dtype=np.int64)
    with pytest.raises(ImportError, match="overlaps a target seed"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_requires_exactly_clear_accepted_baseline_pairs():
    args, _, _ = _fixture(baseline_translation_m=0.0)
    with pytest.raises(ImportError, match="accepted-baseline identity precondition.*exactly clear"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rechecks_optimizer_component_bound(monkeypatch):
    import scipy.optimize

    args, _, _ = _fixture()
    real_minimize = scipy.optimize.minimize

    def component_bound_violation(*positional, **keywords):
        result = real_minimize(*positional, **keywords)
        result.x[2] = clearance._LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM + 0.01
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", component_bound_violation)
    with pytest.raises(ImportError, match="component bound"):
        _propose_local_self_separation_increment(**args)


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (1, "time limit reached", "stopped without a feasible solution.*status=1.*time limit reached"),
        (2, "infeasible model", "infeasible.*status=2.*infeasible model"),
        (4, "numerical difficulty", "solver failed without proving infeasibility.*status=4.*numerical difficulty"),
    ],
)
def test_local_self_separation_preserves_non_successful_highs_status_and_message(
    monkeypatch, status, message, expected
):
    from types import SimpleNamespace
    import scipy.optimize

    args, _, _ = _fixture()
    monkeypatch.setattr(
        scipy.optimize, "linprog",
        lambda *_positional, **_keywords: SimpleNamespace(success=False, status=status, message=message),
    )
    with pytest.raises(ImportError, match=expected):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_a_float32_rounding_that_breaks_sat_constraints():
    args, _, _ = _fixture(
        rejected_z=(0.0, 0.0, 0.0),
        baseline_translation_m=float(np.spacing(np.float32(0.1))),
        world_origin_z=0.1,
        source_world_offset_z=0.2,
    )
    with pytest.raises(ImportError, match="Float32-rounded"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_infeasible_fixed_one_millimeter_bound():
    args, _, _ = _fixture(baseline_translation_m=0.00125)
    with pytest.raises(ImportError, match="infeasible within the fixed 1 mm"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_nonfinite_jacobians_and_duplicate_pair_rows():
    args, _, _ = _fixture()
    args["pose_samples"][0]["jacobians"][0, 0, 0] = np.nan
    with pytest.raises(ImportError, match="non-finite"):
        _propose_local_self_separation_increment(**args)

    args, _, _ = _fixture()
    args["pose_samples"][0]["self_pair_rows"] = np.array([[0, 1], [0, 1]], dtype=np.int64)
    with pytest.raises(ImportError, match="malformed"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_noncanonical_pose_row_binding_and_excess_pose_count():
    args, _, _ = _fixture()
    args["pose_samples"][0]["vertex_ids"] = np.array([1, 0, 2, 3, 4, 5], dtype=np.int64)
    with pytest.raises(ImportError, match="must be sorted"):
        _propose_local_self_separation_increment(**args)

    args, _, _ = _fixture()
    args["pose_samples"] = args["pose_samples"] * 18
    with pytest.raises(ImportError, match="one to seventeen"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_pair_table_not_present_in_exact_rejected_geometry():
    args, _, _ = _fixture()
    args["pose_samples"][0]["rejected_world_positions"][3:, 2] += np.float32(0.001)
    with pytest.raises(ImportError, match="absent from its exact rejected geometry"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_degenerate_supplied_pair_geometry():
    args, _, _ = _fixture()
    args["pose_samples"][0]["baseline_world_positions"][2] = args["pose_samples"][0]["baseline_world_positions"][1]
    with pytest.raises(ImportError, match="degenerate|exactly degenerate"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_nonfinite_source_and_oversized_patch(monkeypatch):
    args, _, _ = _fixture()
    args["source_positions_m"][0, 0] = np.nan
    with pytest.raises(ImportError, match="source positions contain non-finite"):
        _propose_local_self_separation_increment(**args)

    args, _, _ = _fixture()
    monkeypatch.setattr(clearance, "_LOCAL_SELF_SEPARATION_MAX_MOVABLE_VERTICES", 2)
    with pytest.raises(ImportError, match="exceeds its bounded vertex count"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_optimizer_has_bounded_lp_and_slsqp_wall_time(monkeypatch):
    import scipy.optimize

    args, _, _ = _fixture()
    real_linprog = scipy.optimize.linprog
    observed = {}

    def bounded_linprog(*positional, **keywords):
        observed["lp_time_limit"] = keywords["options"]["time_limit"]
        return real_linprog(*positional, **keywords)

    def timeout_minimize(*_positional, **keywords):
        observed["callback"] = keywords["callback"]
        observed["options"] = keywords["options"]
        keywords["callback"](np.zeros(9, dtype=np.float64))
        raise AssertionError("deadline callback should stop SLSQP")

    monotonic_values = iter((10.0, 10.0, 10.0, 131.0))
    monkeypatch.setattr(clearance, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(scipy.optimize, "linprog", bounded_linprog)
    monkeypatch.setattr(scipy.optimize, "minimize", timeout_minimize)

    with pytest.raises(ImportError, match="SLSQP exceeded its 120 second wall-time budget"):
        _propose_local_self_separation_increment(**args)

    assert observed["lp_time_limit"] == pytest.approx(120.0)
    assert observed["options"]["maxiter"] == 1000
    assert callable(observed["callback"])

from __future__ import annotations

import json

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
    with pytest.raises(ImportError, match="component bound") as caught:
        _propose_local_self_separation_increment(**args)
    diagnostics = caught.value.local_self_separation_diagnostics
    assert diagnostics["rejection_stage"] == "requested"
    assert diagnostics["requested"]["component_pass"] is False
    assert diagnostics["requested"]["sat_pass"] is True
    assert diagnostics["requested"]["l2_pass"] is True
    assert diagnostics["requested"]["minimum_component_margin_mm"] == pytest.approx(-0.01, abs=1e-9)
    assert diagnostics["requested"]["minimum_component_margin_source_vertex_id"] == 3
    assert diagnostics["requested"]["minimum_component_margin_axis"] == 2
    assert "rounded" not in diagnostics
    json.dumps(diagnostics, allow_nan=False)


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


def test_local_self_separation_rounding_reserve_bounds_f32_displacement_error():
    cap = clearance._LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM
    below_quarter = np.nextafter(np.float32(0.25), np.float32(0.0))
    above_quarter = np.nextafter(np.float32(0.25), np.float32(np.inf))
    below_negative_quarter = np.nextafter(np.float32(-0.25), np.float32(0.0))
    above_negative_quarter = np.nextafter(np.float32(-0.25), np.float32(-np.inf))
    source = np.array([
        [0.0, -0.0, 0.25],
        [below_quarter, above_quarter, -0.25],
        [below_negative_quarter, above_negative_quarter, 8192.0],
        [-8192.0, 0.125, -0.125],
    ], dtype="<f4")
    reserve = clearance._local_self_separation_rounding_reserve_mm(source, cap)

    assert reserve.shape == source.shape
    assert reserve.dtype == np.float64
    assert np.isfinite(reserve).all()
    assert np.all(reserve >= 0.0)
    assert np.all(reserve < cap)

    for (row, column), source_f32 in np.ndenumerate(source):
        source_value = float(source_f32)
        targets = [
            source_value - cap / 1000.0,
            source_value,
            source_value + cap / 1000.0,
        ]
        # Exercise both half-ULP boundaries around the source and the reachable
        # endpoints, including the 0.25 m binade and the 8192 m scale.
        for center in list(targets):
            center_f32 = np.float32(center)
            for neighbor in (
                np.nextafter(center_f32, np.float32(-np.inf)),
                np.nextafter(center_f32, np.float32(np.inf)),
            ):
                targets.append((float(center_f32) + float(neighbor)) * 0.5)
        requests = [-cap, cap, 0.0]
        for target in targets:
            requested_mm = (target - source_value) * 1000.0
            if abs(requested_mm) <= cap:
                requests.append(requested_mm)
        for requested_mm in requests:
            actual_f32 = np.float32(np.float64(source_f32) + requested_mm / 1000.0)
            actual_mm = (float(actual_f32) - source_value) * 1000.0
            assert abs(actual_mm - requested_mm) <= reserve[row, column]


def test_local_self_separation_rounding_reserve_rejects_no_safe_component_range():
    source = np.full((1, 3), np.float32(32768.0), dtype="<f4")
    with pytest.raises(ImportError, match="leave no positive Float32-safe component range"):
        clearance._local_self_separation_rounding_reserve_mm(
            source, clearance._LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM,
        )


def test_local_self_separation_uses_reserve_to_clear_natural_float32_rounding_case():
    args, _, _ = _fixture(
        rejected_z=(0.0, 0.0, 0.0),
        baseline_translation_m=float(np.spacing(np.float32(0.1))),
        world_origin_z=0.1,
        source_world_offset_z=0.2,
    )
    # The movable source vertices are at 0.3 m while the rejected world pair is
    # at 0.1 m. The required correction is below a Float32 step at the source.
    assert np.all(args["source_positions_m"][3:, 2] == np.float32(0.3))
    result = _propose_local_self_separation_increment(**args)

    assert result["status"] == "bounded_local_linearized_proposal"
    assert result["rounding_reserve_max_component_mm"] > 0.0
    assert result["requested_reserved_linearized_minimum_sat_slack_mm"] >= -1.0e-7
    assert result["post_rounding_linearized_minimum_sat_slack_mm"] >= -1.0e-7
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0
    assert np.all(result["source_increment_mm_f32_applied"][:, 2] > 0.0)
    assert result["proposal_source_positions_m_f32"][3:, 2].tobytes() != (
        args["source_positions_m"][3:, 2].tobytes()
    )


def test_local_self_separation_rejects_requested_move_below_rounding_reserve(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize

    args, _, _ = _fixture(
        rejected_z=(0.0, 0.0, 0.0),
        baseline_translation_m=float(np.spacing(np.float32(0.1))),
        world_origin_z=0.1,
        source_world_offset_z=0.2,
    )
    gap_mm = float(np.spacing(np.float32(0.1))) * 1000.0

    def original_gap_move(*_positional, **_keywords):
        requested = np.zeros(9, dtype=np.float64)
        requested[2::3] = gap_mm
        return SimpleNamespace(success=True, x=requested, status=0, nit=1, message="fixture")

    monkeypatch.setattr(scipy.optimize, "minimize", original_gap_move)
    with pytest.raises(ImportError, match="Float32-reserved") as caught:
        _propose_local_self_separation_increment(**args)
    diagnostics = caught.value.local_self_separation_diagnostics
    assert diagnostics["rejection_stage"] == "requested"
    assert diagnostics["requested"]["sat_pass"] is True
    assert diagnostics["requested"]["reserved_sat_pass"] is False
    assert diagnostics["requested"]["minimum_sat_slack_mm"] >= -1.0e-7
    assert diagnostics["requested"]["minimum_reserved_sat_slack_mm"] < -1.0e-7
    assert diagnostics["requested"]["component_pass"] is True
    assert diagnostics["requested"]["l2_pass"] is True
    json.dumps(diagnostics, allow_nan=False)


def test_local_self_separation_keeps_rounded_failure_diagnostic_under_zero_reserve(monkeypatch):
    args, _, _ = _fixture(
        rejected_z=(0.0, 0.0, 0.0),
        baseline_translation_m=float(np.spacing(np.float32(0.1))),
        world_origin_z=0.1,
        source_world_offset_z=0.2,
    )
    monkeypatch.setattr(
        clearance,
        "_local_self_separation_rounding_reserve_mm",
        lambda source_positions_m, component_cap_mm: np.zeros(
            np.asarray(source_positions_m).shape, dtype=np.float64,
        ),
    )
    with pytest.raises(ImportError, match="Float32-rounded") as caught:
        _propose_local_self_separation_increment(**args)
    diagnostics = caught.value.local_self_separation_diagnostics
    assert diagnostics["rejection_stage"] == "rounded"
    assert all(diagnostics["requested"][key] for key in ("sat_pass", "component_pass", "l2_pass"))
    assert diagnostics["requested"]["reserved_sat_pass"] is True
    assert diagnostics["rounded"]["sat_pass"] is False
    assert diagnostics["rounded"]["component_pass"] is True
    assert diagnostics["rounded"]["l2_pass"] is True
    row = diagnostics["rounded"]["minimum_sat_constraint"]
    assert row["sample_index"] == 0 and row["face_pair"] == [0, 1]
    assert row["source_vertex_pair"][0] in [0, 1, 2]
    assert row["source_vertex_pair"][1] in [3, 4, 5]
    assert row["axis_unit_world"] == [0.0, 0.0, 1.0]
    assert diagnostics["float32_quantization_error_max_abs_component_mm"] > 0.0
    assert diagnostics["optimizer"]["success"] is True
    json.dumps(diagnostics, allow_nan=False)


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
def _preservation_fixture():
    baseline = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.55, 0.05, 0.00025], [0.85, 0.05, 0.00025], [0.55, 0.35, 0.00025],
        [0.05, 0.05, 0.00050], [0.25, 0.05, 0.00050], [0.05, 0.25, 0.00050],
    ], dtype="<f4")
    rejected = baseline.copy()
    rejected[3:6, 2] = np.array([-0.00025, 0.00025, -0.00025], dtype="<f4")
    rejected[6:9, 2] = np.float32(0.00025)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    pose = {
        "vertex_ids": np.arange(9, dtype=np.int64),
        "baseline_world_positions": baseline,
        "rejected_world_positions": rejected,
        "jacobians": np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 9, axis=0),
        "self_pair_rows": np.array([[0, 1]], dtype=np.int64),
        "clearance_preservation_pair_rows": np.array([[0, 2]], dtype=np.int64),
    }
    return {
        "source_positions_m": rejected.copy(),
        "faces": faces,
        "movable_vertex_ids": np.array([3, 4, 5, 6, 7, 8], dtype=np.int64),
        "self_patch_seed_vertex_ids": np.array([3, 4, 5], dtype=np.int64),
        "required_target_seed_vertex_ids": np.array([0], dtype=np.int64),
        "active_target_face_vertex_ids": np.array([0, 1, 2], dtype=np.int64),
        "fixed_support_vertex_ids": np.empty(0, dtype=np.int64),
        "preserved_anchor_vertex_ids": np.empty(0, dtype=np.int64),
        "pose_samples": [pose],
    }


def test_local_self_separation_combines_resolving_and_clearance_preservation_rows():
    args = _preservation_fixture()

    result = _propose_local_self_separation_increment(**args)

    assert result["status"] == "bounded_local_linearized_proposal"
    assert result["constraint_rows_by_pose"] == [18]
    assert [x["constraint_kind"] for x in result["pair_pose_receipts"]] == ["resolve", "preserve"]
    applied = result["source_increment_mm_f32_applied"]
    assert np.all(applied[:3, 2] > 0.0)
    preservation = next(x for x in result["pair_pose_receipts"] if x["constraint_kind"] == "preserve")
    assert preservation["baseline_sat_gap_mm"] == pytest.approx(0.5, abs=1e-6)
    assert preservation["current_sat_gap_mm"] == pytest.approx(0.25, abs=1e-6)
    assert preservation["preserved_sat_gap_floor_mm"] == pytest.approx(0.25, abs=1e-6)
    assert np.max(np.abs(applied[3:, 2])) < 1e-6
    assert result["post_rounding_linearized_minimum_sat_slack_mm"] >= -1e-7
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0


def test_local_self_separation_rejects_preservation_pair_that_is_not_currently_clear():
    args = _preservation_fixture()
    args["pose_samples"][0]["rejected_world_positions"][6:9, 2] = np.float32(0.0)

    with pytest.raises(ImportError, match="preservation pair is not clear"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_rejects_pair_marked_both_resolving_and_preserved():
    args = _preservation_fixture()
    args["pose_samples"][0]["clearance_preservation_pair_rows"] = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ImportError, match="both resolving and preserved"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_accepts_alternative_generated_baseline_axis():
    args, baseline, rejected = _fixture()
    # The two triangles are separated in both x and z in the accepted source,
    # but the rejected triangle moves into the first triangle. The default
    # largest-gap axis is x; the smaller positive z gap is also a valid SAT
    # separator and can resolve the pair with less local movement.
    baseline = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.1, 0.1, 0.00025], [2.8, 0.1, 0.00075], [2.1, 0.8, 0.00025],
    ], dtype="<f4")
    rejected = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.1, 0.1, -0.00025], [0.8, 0.1, 0.00025], [0.1, 0.8, -0.00025],
    ], dtype="<f4")
    args["source_positions_m"] = rejected.copy()
    args["pose_samples"][0]["baseline_world_positions"] = baseline
    args["pose_samples"][0]["rejected_world_positions"] = rejected
    args["pose_samples"][0]["self_pair_axis_overrides"] = [
        {"face_pair": [0, 1], "axis_world": [0.0, 0.0, 1.0]},
    ]

    result = _propose_local_self_separation_increment(**args)

    receipt = result["pair_pose_receipts"][0]
    assert receipt["sat_axis_selection"] == "caller_supplied_positive_baseline_axis"
    assert receipt["axis_unit_world"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert receipt["baseline_sat_gap_mm"] == pytest.approx(0.25, abs=1e-6)
    assert result["post_rounding_linearized_minimum_sat_slack_mm"] >= -1e-7
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        ([0.0, 1.0, 0.0], "no positive accepted-baseline separation"),
        ([0.0, 0.0, 0.0], "axis override is degenerate"),
        ([float("nan"), 0.0, 1.0], "axis override must be finite xyz"),
    ],
)
def test_local_self_separation_rejects_invalid_or_nonseparating_axis_override(axis, message):
    args, baseline, rejected = _fixture()
    baseline = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.1, 0.1, 0.00025], [2.8, 0.1, 0.00075], [2.1, 0.8, 0.00025],
    ], dtype="<f4")
    rejected = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.1, 0.1, -0.00025], [0.8, 0.1, 0.00025], [0.1, 0.8, -0.00025],
    ], dtype="<f4")
    args["source_positions_m"] = rejected.copy()
    args["pose_samples"][0]["baseline_world_positions"] = baseline
    args["pose_samples"][0]["rejected_world_positions"] = rejected
    args["pose_samples"][0]["self_pair_axis_overrides"] = [
        {"face_pair": [0, 1], "axis_world": axis},
    ]

    with pytest.raises(ImportError, match=message):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_keeps_default_axis_selection_unchanged():
    args, _, _ = _fixture()
    result = _propose_local_self_separation_increment(**args)
    assert result["pair_pose_receipts"][0]["sat_axis_selection"] == "accepted_baseline"
    assert result["pair_pose_receipts"][0]["sat_axis_index"] >= 0


def test_local_self_separation_rejects_axis_override_for_nonresolving_pair():
    args, _, _ = _fixture()
    args["pose_samples"][0]["self_pair_axis_overrides"] = [
        {"face_pair": [0, 1], "axis_world": [0.0, 0.0, 1.0]},
        {"face_pair": [0, 1], "axis_world": [0.0, 0.0, 1.0]},
    ]
    with pytest.raises(ImportError, match="duplicates a pair axis override"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_accepts_nonsat_direction_with_positive_baseline_gap():
    args, _, _ = _fixture()
    baseline = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.1, 0.1, 0.00025], [2.8, 0.1, 0.00075], [2.1, 0.8, 0.00025],
    ], dtype="<f4")
    rejected = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.1, 0.1, -0.00025], [0.8, 0.1, 0.00025], [0.1, 0.8, -0.00025],
    ], dtype="<f4")
    args["source_positions_m"] = rejected.copy()
    args["pose_samples"][0]["baseline_world_positions"] = baseline
    args["pose_samples"][0]["rejected_world_positions"] = rejected
    args["pose_samples"][0]["self_pair_axis_overrides"] = [
        {"face_pair": [0, 1], "axis_world": [0.000015, 0.0, 1.0]},
    ]

    result = _propose_local_self_separation_increment(**args)

    receipt = result["pair_pose_receipts"][0]
    assert receipt["sat_axis_selection"] == "caller_supplied_positive_baseline_axis"
    assert receipt["baseline_sat_gap_mm"] > 0.25
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0


def _shared_vertex_contact_fixture():
    source = np.array([
        [0.0, 0.0, 0.0],
        [-0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, 0.001],
        [0.0, -0.001, 0.0],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    baseline = source.copy()
    rejected = source.copy()
    rejected[4, 1] = np.float32(0.0002)
    jac = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(source), axis=0)
    pose = {
        "vertex_ids": np.arange(len(source), dtype=np.int64),
        "baseline_world_positions": baseline,
        "rejected_world_positions": rejected,
        "jacobians": jac,
        "self_pair_rows": np.empty((0, 2), dtype=np.int64),
        "shared_vertex_contact_pair_rows": np.array([[0, 1]], dtype=np.int64),
    }
    args = {
        "source_positions_m": source,
        "faces": faces,
        "movable_vertex_ids": np.array([3, 4], dtype=np.int64),
        "self_patch_seed_vertex_ids": np.array([3, 4], dtype=np.int64),
        "required_target_seed_vertex_ids": np.array([0], dtype=np.int64),
        "active_target_face_vertex_ids": np.array([0, 1, 2], dtype=np.int64),
        "fixed_support_vertex_ids": np.array([1], dtype=np.int64),
        "preserved_anchor_vertex_ids": np.array([2], dtype=np.int64),
        "pose_samples": [pose],
    }
    return args, source, rejected


def test_local_self_separation_resolves_pose_crossing_from_source_proven_shared_vertex_contact():
    from fractions import Fraction
    from numilab_human import cardiac_cavity_intersections as ci

    args, source, rejected = _shared_vertex_contact_fixture()
    source_before = source.copy()
    source_triangles = [
        tuple(ci.float32_point_lattice_key(source[int(vertex)]) for vertex in args["faces"][face])
        for face in range(2)
    ]
    source_points = ci.triangle_intersection_points(*source_triangles)
    common = tuple(Fraction(int(value)) for value in ci.float32_point_lattice_key(source[0]))
    assert source_points and all(tuple(point) == common for point in source_points)
    rejected_triangles = [
        tuple(ci.float32_point_lattice_key(rejected[int(vertex)]) for vertex in args["faces"][face])
        for face in range(2)
    ]
    assert any(tuple(point) != common for point in ci.triangle_intersection_points(*rejected_triangles))

    result = _propose_local_self_separation_increment(**args)

    assert result["status"] == "bounded_local_linearized_proposal"
    assert result["proposal_source_positions_m_f32"].tobytes() != source_before.tobytes()
    assert result["proposal_source_positions_m_f32"][[0, 1, 2]].tobytes() == source_before[[0, 1, 2]].tobytes()
    receipt = result["pair_pose_receipts"][0]
    assert receipt["constraint_kind"] == "shared_vertex_contact"
    assert receipt["shared_vertex_id"] == 0
    assert receipt["source_contact_floor_mm"] == 0.0
    assert receipt["source_intersection_points_only_shared_vertex"] is True
    assert receipt["source_one_sided_support_proof"] is True
    assert receipt["sat_axis_selection"] == "exact_source_shared_vertex_support_transformed_by_shared_jacobian"
    assert receipt["source_axis_selection"] == (
        "enumerated_exact_source_support_axes_ranked_by_fixed_support_feasibility_and_movable_response"
    )
    assert receipt["source_axis_candidate_count"] >= 1
    assert receipt["source_axis_index"] is not None
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0
    assert args["source_positions_m"].tobytes() == source_before.tobytes()


def test_local_self_separation_contact_mode_rejects_geometric_contact_without_shared_index():
    args, source, _ = _shared_vertex_contact_fixture()
    source_with_duplicate = np.vstack((source, source[0])).astype("<f4")
    args["source_positions_m"] = source_with_duplicate
    args["faces"] = np.array([[0, 1, 2], [5, 3, 4]], dtype=np.int64)
    pose = args["pose_samples"][0]
    pose["vertex_ids"] = np.arange(6, dtype=np.int64)
    pose["baseline_world_positions"] = np.vstack((pose["baseline_world_positions"], pose["baseline_world_positions"][0])).astype("<f4")
    pose["rejected_world_positions"] = np.vstack((pose["rejected_world_positions"], pose["rejected_world_positions"][0])).astype("<f4")
    pose["jacobians"] = np.concatenate((pose["jacobians"], pose["jacobians"][[0]]), axis=0)
    args["movable_vertex_ids"] = np.array([3, 4], dtype=np.int64)
    args["self_patch_seed_vertex_ids"] = np.array([3, 4], dtype=np.int64)
    with pytest.raises(ImportError, match="exactly one shared indexed vertex"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_contact_mode_rejects_source_intersection_beyond_shared_vertex():
    args, _, _ = _shared_vertex_contact_fixture()
    args["source_positions_m"][4, 1] = np.float32(0.0002)
    with pytest.raises(ImportError, match="exact source-only shared-vertex intersection"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_does_not_let_ordinary_pairs_use_zero_gap_contact_mode():
    args, _, _ = _fixture(baseline_translation_m=0.0)
    with pytest.raises(ImportError, match="accepted-baseline identity precondition.*exactly clear"):
        _propose_local_self_separation_increment(**args)


def test_local_self_separation_does_not_propose_when_only_shared_contact_vertex_can_move():
    args, _, _ = _shared_vertex_contact_fixture()
    args["movable_vertex_ids"] = np.array([0], dtype=np.int64)
    args["self_patch_seed_vertex_ids"] = np.array([0], dtype=np.int64)
    args["required_target_seed_vertex_ids"] = np.empty(0, dtype=np.int64)
    args["active_target_face_vertex_ids"] = np.empty(0, dtype=np.int64)
    args["fixed_support_vertex_ids"] = np.array([1, 2, 3, 4], dtype=np.int64)
    with pytest.raises(ImportError) as error:
        _propose_local_self_separation_increment(**args)
    assert "did not converge to a finite proposal" in str(error.value) or "no movable response" in str(error.value)



def test_local_self_separation_accepts_only_enumerated_shared_contact_source_axis_override():
    import copy

    args, _, _ = _shared_vertex_contact_fixture()
    automatic = _propose_local_self_separation_increment(**copy.deepcopy(args))
    automatic_receipt = automatic["pair_pose_receipts"][0]
    feasible_axes = automatic_receipt["fixed_support_feasible_source_axis_indices"]
    alternative = next(index for index in feasible_axes if index != automatic_receipt["source_axis_index"])

    forced_args = copy.deepcopy(args)
    forced_args["pose_samples"][0]["shared_vertex_contact_axis_overrides"] = [
        {"face_pair": [0, 1], "source_axis_index": alternative},
    ]
    forced = _propose_local_self_separation_increment(**forced_args)
    forced_receipt = forced["pair_pose_receipts"][0]

    assert forced_receipt["source_axis_selection"] == "caller_selected_exact_source_support_axis_index"
    assert forced_receipt["source_axis_override_index"] == alternative
    assert forced_receipt["source_axis_index"] == alternative
    assert forced["post_rounding_predicted_supplied_pair_exact_count"] == 0
    assert automatic["proposal_source_positions_m_f32"].tobytes() != forced["proposal_source_positions_m_f32"].tobytes()


def test_local_self_separation_rejects_unproven_shared_contact_axis_override():
    import copy

    args, _, _ = _shared_vertex_contact_fixture()
    args["pose_samples"][0]["shared_vertex_contact_axis_overrides"] = [
        {"face_pair": [0, 1], "source_axis_index": 99},
    ]
    with pytest.raises(ImportError, match="not an exact one-sided source support axis"):
        _propose_local_self_separation_increment(**args)

    wrong_pair_args, _, _ = _shared_vertex_contact_fixture()
    wrong_pair_args["pose_samples"][0]["shared_vertex_contact_axis_overrides"] = [
        {"face_pair": [0, 0], "source_axis_index": 0},
    ]
    with pytest.raises(ImportError, match="not a shared-vertex contact pair"):
        _propose_local_self_separation_increment(**wrong_pair_args)


def _external_clearance_fixture(*, parent_z=0.0005):
    target = np.array([
        [0.2, 0.2, 0.0], [0.7, 0.2, 0.0], [0.2, 0.7, 0.0],
    ], dtype="<f4")
    parent = np.array([
        [0.0, 0.0, parent_z], [1.0, 0.0, parent_z], [0.0, 1.0, parent_z],
    ], dtype="<f4")
    rejected = np.array([
        [0.0, 0.0, -0.00025], [1.0, 0.0, 0.00025], [0.0, 1.0, -0.00025],
    ], dtype="<f4")
    unrelated_parent = np.array([
        [10.0, 0.0, parent_z], [11.0, 0.0, parent_z], [10.0, 1.0, parent_z],
    ], dtype="<f4")
    accepted_parent_positions = np.concatenate((unrelated_parent, parent), axis=0)
    pose = {
        "vertex_ids": np.arange(3, dtype=np.int64),
        "baseline_world_positions": parent,
        "rejected_world_positions": rejected,
        "accepted_parent_world_positions": accepted_parent_positions,
        "accepted_parent_faces": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        "jacobians": np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0),
        "external_clearance_rows": [{
            "candidate_face_row": 0,
            "parent_face_row": 1,
            "target_semantic": 51005,
            "target_stable_id": 134,
            "target_face_id": 1478,
            "target_triangle_world_positions": target,
        }],
    }
    return {
        "source_positions_m": rejected.copy(),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
        "movable_vertex_ids": np.array([0, 1, 2], dtype=np.int64),
        "self_patch_seed_vertex_ids": np.empty(0, dtype=np.int64),
        "required_target_seed_vertex_ids": np.empty(0, dtype=np.int64),
        "active_target_face_vertex_ids": np.empty(0, dtype=np.int64),
        "fixed_support_vertex_ids": np.empty(0, dtype=np.int64),
        "preserved_anchor_vertex_ids": np.empty(0, dtype=np.int64),
        "pose_samples": [pose],
    }


def _many_external_clearance_args(row_count):
    import copy

    args = _external_clearance_fixture()
    template = args["pose_samples"][0]
    samples = []
    next_face_id = 0
    remaining = row_count
    while remaining:
        count = min(128, remaining)
        pose = copy.deepcopy(template)
        pose["external_clearance_rows"] = []
        for _ in range(count):
            row = dict(template["external_clearance_rows"][0])
            row["target_face_id"] = next_face_id
            next_face_id += 1
            pose["external_clearance_rows"].append(row)
        samples.append(pose)
        remaining -= count
    args["pose_samples"] = samples
    return args


def _many_self_preservation_args(row_count):
    vertices = []
    faces = []
    tri_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="<f4")
    tri_b = tri_a.copy()
    tri_b[:, 2] = np.float32(0.001)
    for face in range(46):
        tri = tri_a if face < 23 else tri_b
        start = len(vertices)
        vertices.extend(tri.tolist())
        faces.append([start, start + 1, start + 2])
    source = np.asarray(vertices, dtype="<f4")
    face_table = np.asarray(faces, dtype=np.int64)
    all_pairs = np.asarray([[a, b] for a in range(23) for b in range(23, 46)], dtype=np.int64)[:row_count]
    samples = []
    for start in range(0, row_count, 128):
        samples.append({
            "vertex_ids": np.arange(len(source), dtype=np.int64),
            "baseline_world_positions": source.copy(),
            "rejected_world_positions": source.copy(),
            "jacobians": np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(source), axis=0),
            "clearance_preservation_pair_rows": all_pairs[start:start + 128].copy(),
        })
    return {
        "source_positions_m": source,
        "faces": face_table,
        "movable_vertex_ids": np.arange(len(source), dtype=np.int64),
        "self_patch_seed_vertex_ids": np.arange(len(source), dtype=np.int64),
        "required_target_seed_vertex_ids": np.empty(0, dtype=np.int64),
        "active_target_face_vertex_ids": np.empty(0, dtype=np.int64),
        "fixed_support_vertex_ids": np.empty(0, dtype=np.int64),
        "preserved_anchor_vertex_ids": np.empty(0, dtype=np.int64),
        "pose_samples": samples,
    }


def test_legacy_self_only_pair_bound_remains_512_rows():
    args = _many_self_preservation_args(513)

    with pytest.raises(ImportError, match="legacy 512-row bound"):
        _propose_local_self_separation_increment(**args)


def test_external_clearance_opt_in_bound_accepts_514_joint_rows():
    args = _many_external_clearance_args(514)

    result = _propose_local_self_separation_increment(**args)

    assert clearance._LOCAL_SELF_SEPARATION_MAX_PAIR_POSE_ROWS == 512
    assert clearance._LOCAL_SELF_SEPARATION_MAX_EXTERNAL_PAIR_POSE_ROWS == 520
    assert result["constraint_rows_by_pose"] == [1152, 1152, 1152, 1152, 18]
    assert len(result["external_clearance_pair_receipts"]) == 514
    assert result["external_clearance_exact_predicted_pair_count"] == 0


def test_external_clearance_opt_in_bound_rejects_more_than_520_rows():
    args = _many_external_clearance_args(521)

    with pytest.raises(ImportError, match="opt-in 520-row external-clearance bound"):
        _propose_local_self_separation_increment(**args)


def test_external_clearance_floor_comes_from_exact_parent_and_target_triangles():
    args = _external_clearance_fixture()

    result = _propose_local_self_separation_increment(**args)

    receipt, = result["external_clearance_pair_receipts"]
    assert receipt["accepted_parent_exactly_clear"] is True
    assert receipt["rejected_candidate_exactly_intersects"] is True
    assert receipt["accepted_parent_sat_gap_mm"] == pytest.approx(0.5, abs=1e-6)
    assert receipt["minimum_gap_floor_mm"] == pytest.approx(0.001, abs=1e-12)
    assert receipt["accepted_parent_clearance_loss_mm"] == pytest.approx(0.499, abs=1e-6)
    assert receipt["candidate_face_row"] == 0
    assert receipt["accepted_parent_face_row"] == 1
    assert receipt["target_semantic"] == 51005
    assert receipt["target_stable_id"] == 134
    assert receipt["target_face_id"] == 1478
    assert result["external_clearance_exact_predicted_pair_count"] == 0
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0
    assert np.max(result["source_increment_mm_f32_applied"][:, 2]) <= 1.0


def test_external_clearance_rejects_intersecting_accepted_parent():
    args = _external_clearance_fixture(parent_z=0.0)

    with pytest.raises(ImportError, match="external accepted-parent face 0 already intersects target"):
        _propose_local_self_separation_increment(**args)


def test_local_self_resolution_and_external_clearance_constraints_are_jointly_applied():
    args, baseline, _ = _fixture()
    pose = args["pose_samples"][0]
    pose["accepted_parent_world_positions"] = baseline.copy()
    pose["accepted_parent_faces"] = args["faces"].copy()
    pose["external_clearance_rows"] = [{
        "candidate_face_row": 1,
        "parent_face_row": 1,
        "target_semantic": 51005,
        "target_stable_id": 134,
        "target_face_id": 0,
        "target_triangle_world_positions": baseline[args["faces"][0]].copy(),
    }]

    result = _propose_local_self_separation_increment(**args)

    assert len(result["pair_pose_receipts"]) == 1
    assert result["pair_pose_receipts"][0]["constraint_kind"] == "resolve"
    assert len(result["external_clearance_pair_receipts"]) == 1
    assert result["external_clearance_pair_receipts"][0]["rejected_candidate_exactly_intersects"] is True
    assert result["external_clearance_exact_predicted_pair_count"] == 0
    assert result["post_rounding_predicted_supplied_pair_exact_count"] == 0


def test_optional_external_clearance_does_not_change_default_self_proposal():
    args, _, _ = _fixture()
    legacy = _propose_local_self_separation_increment(**args)
    unchanged_args, _, _ = _fixture()
    unchanged_args["pose_samples"][0]["external_clearance_rows"] = []
    unchanged = _propose_local_self_separation_increment(**unchanged_args)

    assert legacy["proposal_source_positions_m_f32"].tobytes() == unchanged["proposal_source_positions_m_f32"].tobytes()
    assert legacy["source_increment_mm_f32_applied"].tobytes() == unchanged["source_increment_mm_f32_applied"].tobytes()
    assert legacy["constraint_rows_by_pose"] == unchanged["constraint_rows_by_pose"]
    assert legacy["pair_pose_receipts"] == unchanged["pair_pose_receipts"]
    assert unchanged["external_clearance_pair_receipts"] == []

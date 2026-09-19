from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import numilab_human.native_trace_refinement as refinement


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _machine_receipt(
    identity: str = "1" * 64,
    *,
    chip: str = "Apple M4 Pro",
    machine_name: str = "Mac mini",
) -> dict[str, str]:
    receipt = {
        "architecture": "arm64",
        "chip": chip,
        "machine_identity_sha256": identity,
        "machine_model": "Mac16,11",
        "machine_name": machine_name,
        "memory": "24 GB",
        "os_build": "25G72",
        "os_version": "26.6",
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _sample(index: int, timestep: float, scale: float) -> dict:
    q = [0.0] * refinement.Q_COUNT
    q[2] = 1.9 + scale * index * timestep
    q[6] = 1.0
    v = [scale * index * timestep] * refinement.V_COUNT
    impulse_work = scale * timestep if index else 0.0
    return {
        "step": index,
        "time_seconds": index * timestep,
        "q": q,
        "v": v,
        "normal_impulse": scale * timestep,
        "maximum_normal_contact_impulse_ns": 0.4 * scale * timestep,
        "maximum_tangential_contact_impulse_ns": 0.1 * scale * timestep,
        "maximum_source_limit_impulse_ns_or_nms": 0.2 * scale * timestep,
        "total_source_limit_absolute_impulse_ns_or_nms": 0.3 * scale * timestep,
        "maximum_equality_impulse": 0.5 * scale * timestep,
        "total_equality_impulse": 0.7 * scale * timestep,
        "post_projection_normal_contact_target_velocity_residual_m_s": 1.0e-7,
        "post_projection_source_limit_target_velocity_residual_m_s_or_rad_s": 2.0e-7,
        "post_projection_equality_target_velocity_residual_m_s_or_rad_s": 3.0e-9,
        "tendon_max_force_residual_n": scale * timestep * 0.002,
        "tendon_max_moment_residual_nm": scale * timestep * 0.0002,
        "muscle_virtual_work_j": -scale * timestep * 0.01,
        "passive_joint_potential_work_j": scale * timestep * 0.001,
        "support_virtual_work_j": scale * timestep * 0.002,
        "contact_normal_impulse_work_j": -impulse_work * 0.03,
        "contact_tangential_impulse_work_j": -impulse_work * 0.02,
        "equality_impulse_work_j": impulse_work * 0.01,
        "source_limit_impulse_work_j": -impulse_work * 0.005,
        "contact_normal_absolute_impulse_work_j": impulse_work * 0.04,
        "contact_tangential_absolute_impulse_work_j": impulse_work * 0.03,
        "equality_absolute_impulse_work_j": impulse_work * 0.02,
        "source_limit_absolute_impulse_work_j": impulse_work * 0.01,
        "passive_joint_energy_j": 0.02 + scale * index * timestep * 0.001,
        "maximum_normal_contact_impulse_index": 1,
        "maximum_tangential_contact_impulse_index": 2,
        "maximum_source_limit_impulse_dof": 3,
        "maximum_equality_impulse_index": 4,
        "free_force_acceleration": 100.0,
        "constraint_velocity_delta": scale * timestep,
        "pre_projection_velocity_delta": scale * timestep * 0.01,
        "published_velocity_delta": scale * timestep * 0.01,
    }


def _case(
    root: Path,
    name: str,
    source: str = "a" * 40,
    *,
    machine_identity: str = "1" * 64,
    chip: str = "Apple M4 Pro",
    machine_name: str = "Mac mini",
) -> Path:
    nanoseconds, steps = refinement.CASE_SPEC[name]
    timestep = nanoseconds * 1.0e-9
    directory = root / name
    directory.mkdir()
    samples = [_sample(index, timestep, 950.0) for index in range(steps + 1)]
    trace = {
        "schema": refinement.TRACE_SCHEMA,
        "work_scope": (
            "production_constraint_impulse_work_by_family;"
            "exact_coordinate_projection_is_an_unowned_overwrite_not_impulse_work"
        ),
        "endpoint_equivalent": "bitwise",
        "endpoint_max_q_delta": 0,
        "endpoint_max_v_delta": 0,
        "samples": samples,
    }
    for sample_field, total_field in refinement.TRACE_IMPULSE_WORK_TOTALS.items():
        trace[total_field] = sum(sample[sample_field] for sample in samples[1:])
    stdout = directory / "stdout.txt"
    stdout.write_text(
        "persistent_stand_trace=" + json.dumps(trace) + "\n",
        encoding="utf-8",
    )
    stderr = directory / "stderr.txt"
    stderr.write_text("", encoding="utf-8")
    summary = {
        "schema": refinement.CASE_SCHEMA,
        "name": name,
        "native_commit": source,
        "input_commit": "b" * 40,
        "binary_sha256": "c" * 64,
        "payload_sha256": {"rigid": "d" * 64},
        "launcher_sha256": "e" * 64,
        "physical_machine_receipt": _machine_receipt(
            machine_identity,
            chip=chip,
            machine_name=machine_name,
        ),
        "timestep_nanoseconds": nanoseconds,
        "step_count": steps,
        "duration_nanoseconds": refinement.COMMON_DURATION_NS,
        "exit_code": 0,
        "wall_seconds": 1.0,
        "stdout_sha256": _sha(stdout),
        "stderr_sha256": _sha(stderr),
        "stderr_nonbanner_lines": [],
        "required_metric_mismatches": {},
        "velocity_stage_diagnostics_complete": True,
        "constraint_impulse_work_complete": True,
        "compiled_static_balance": True,
    }
    (directory / "case-summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return directory


def _rewrite_trace(case: Path, mutate) -> None:
    stdout = case / "stdout.txt"
    trace = json.loads(stdout.read_text(encoding="utf-8").split("=", 1)[1])
    mutate(trace)
    stdout.write_text(
        "persistent_stand_trace=" + json.dumps(trace) + "\n",
        encoding="utf-8",
    )
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stdout_sha256"] = _sha(stdout)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_complete_source_bound_trace_comparison(tmp_path: Path) -> None:
    cases = [_case(tmp_path, name) for name in refinement.CASE_SPEC]
    report = refinement.compare_cases(cases)
    assert report["status"] == "diagnostic_complete"
    assert report["coverage"]["aggregate_equality_impulse"]
    assert report["coverage"]["constraint_stage_impulsive_work"]
    assert not report["coverage"]["complete_impulsive_work"]
    assert not report["coverage"]["complete_physical_energy_closure"]
    assert not report["coverage"]["complete_per_constraint_reaction_vectors"]
    assert report["coverage"]["identical_physical_machine_identity"]
    assert report["coverage"]["tendon_force_and_moment_residuals"]
    assert report["qualification"]["physical_m4_validation"]
    assert not report["qualification"]["force_convergence"]
    assert not report["qualification"]["sustained_standing"]
    assert report["physical_machine_receipt"]["machine_name"] == "Mac mini"
    assert len(report["comparisons"]) == 3
    assert all(
        row["maximum_normal_reaction_delta_n"] < 1.0e-9
        for row in report["comparisons"]
    )
    assert all(
        set(row["maximum_tendon_residual_deltas"])
        == set(refinement.TENDON_RESIDUAL_FIELDS)
        for row in report["comparisons"]
    )
    assert all(
        set(row["maximum_tendon_residuals"])
        == set(refinement.TENDON_RESIDUAL_FIELDS)
        for row in report["cases"]
    )


def test_mixed_source_is_rejected(tmp_path: Path) -> None:
    cases = [
        _case(tmp_path, name, "f" * 40 if name == "25us" else "a" * 40)
        for name in refinement.CASE_SPEC
    ]
    with pytest.raises(refinement.TraceRefinementError, match="mix native_commit"):
        refinement.compare_cases(cases)


def test_mixed_physical_machine_identity_is_rejected(tmp_path: Path) -> None:
    cases = [
        _case(
            tmp_path,
            name,
            machine_identity="2" * 64 if name == "25us" else "1" * 64,
        )
        for name in refinement.CASE_SPEC
    ]
    with pytest.raises(
        refinement.TraceRefinementError,
        match="mix physical_machine_receipt",
    ):
        refinement.compare_cases(cases)


def test_non_m4_machine_remains_diagnostic(tmp_path: Path) -> None:
    cases = [
        _case(tmp_path, name, chip="Apple M3 Pro")
        for name in refinement.CASE_SPEC
    ]
    report = refinement.compare_cases(cases)
    assert not report["qualification"]["physical_m4_validation"]
    assert not report["qualification"]["force_convergence"]
    assert not report["qualification"]["sustained_standing"]


def test_missing_physical_machine_receipt_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("physical_machine_receipt")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        refinement.TraceRefinementError,
        match="physical_machine_receipt is not an object",
    ):
        refinement.load_case(case)


def test_physical_machine_receipt_digest_is_bound(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["physical_machine_receipt"]["memory"] = "128 GB"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        refinement.TraceRefinementError,
        match="physical_machine_receipt digest mismatch",
    ):
        refinement.load_case(case)


def test_v2_case_without_machine_binding_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = "numi.human.current-refinement-case.v2"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(refinement.TraceRefinementError, match="schema mismatch"):
        refinement.load_case(case)


def test_changed_stdout_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    (case / "stdout.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(refinement.TraceRefinementError, match="hash mismatch"):
        refinement.load_case(case)


def test_partial_velocity_stage_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    _rewrite_trace(
        case,
        lambda trace: trace["samples"][1].pop("published_velocity_delta"),
    )
    with pytest.raises(refinement.TraceRefinementError, match="partial velocity-stage"):
        refinement.load_case(case)


def test_missing_constraint_impulse_work_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    _rewrite_trace(
        case,
        lambda trace: trace["samples"][1].pop("equality_impulse_work_j"),
    )
    with pytest.raises(refinement.TraceRefinementError, match="equality_impulse_work_j"):
        refinement.load_case(case)


def test_absolute_constraint_work_cannot_hide_signed_work(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    _rewrite_trace(
        case,
        lambda trace: trace["samples"][1].__setitem__(
            "contact_normal_absolute_impulse_work_j", 0.0
        ),
    )
    with pytest.raises(refinement.TraceRefinementError, match="hides signed work"):
        refinement.load_case(case)


def test_trace_constraint_work_total_must_match_samples(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    _rewrite_trace(
        case,
        lambda trace: trace.__setitem__("total_equality_impulse_work_j", 1.0),
    )
    with pytest.raises(refinement.TraceRefinementError, match="disagrees with samples"):
        refinement.load_case(case)


def test_case_without_constraint_work_completion_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["constraint_impulse_work_complete"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(refinement.TraceRefinementError, match="lacks constraint impulse work"):
        refinement.load_case(case)

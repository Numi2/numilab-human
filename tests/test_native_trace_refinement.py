from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import numilab_human.native_trace_refinement as refinement


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(index: int, timestep: float, scale: float) -> dict:
    q = [0.0] * refinement.Q_COUNT
    q[2] = 1.9 + scale * index * timestep
    q[6] = 1.0
    v = [scale * index * timestep] * refinement.V_COUNT
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
        "muscle_virtual_work_j": -scale * timestep * 0.01,
        "passive_joint_potential_work_j": scale * timestep * 0.001,
        "support_virtual_work_j": scale * timestep * 0.002,
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


def _case(root: Path, name: str, source: str = "a" * 40) -> Path:
    nanoseconds, steps = refinement.CASE_SPEC[name]
    timestep = nanoseconds * 1.0e-9
    directory = root / name
    directory.mkdir()
    trace = {
        "schema": refinement.TRACE_SCHEMA,
        "endpoint_equivalent": "bitwise",
        "endpoint_max_q_delta": 0,
        "endpoint_max_v_delta": 0,
        "samples": [_sample(index, timestep, 950.0) for index in range(steps + 1)],
    }
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
        "compiled_static_balance": True,
    }
    (directory / "case-summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return directory


def test_complete_source_bound_trace_comparison(tmp_path: Path) -> None:
    cases = [_case(tmp_path, name) for name in refinement.CASE_SPEC]
    report = refinement.compare_cases(cases)
    assert report["status"] == "diagnostic_complete"
    assert report["coverage"]["aggregate_equality_impulse"]
    assert not report["coverage"]["complete_per_constraint_reaction_vectors"]
    assert not report["qualification"]["force_convergence"]
    assert len(report["comparisons"]) == 3
    assert all(
        row["maximum_normal_reaction_delta_n"] < 1.0e-9
        for row in report["comparisons"]
    )


def test_mixed_source_is_rejected(tmp_path: Path) -> None:
    cases = [
        _case(tmp_path, name, "f" * 40 if name == "25us" else "a" * 40)
        for name in refinement.CASE_SPEC
    ]
    with pytest.raises(refinement.TraceRefinementError, match="mix native_commit"):
        refinement.compare_cases(cases)


def test_changed_stdout_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    (case / "stdout.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(refinement.TraceRefinementError, match="hash mismatch"):
        refinement.load_case(case)


def test_partial_velocity_stage_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path, "100us")
    stdout = case / "stdout.txt"
    trace = json.loads(stdout.read_text(encoding="utf-8").split("=", 1)[1])
    del trace["samples"][1]["published_velocity_delta"]
    stdout.write_text(
        "persistent_stand_trace=" + json.dumps(trace) + "\n",
        encoding="utf-8",
    )
    summary_path = case / "case-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stdout_sha256"] = _sha(stdout)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(refinement.TraceRefinementError, match="partial velocity-stage"):
        refinement.load_case(case)

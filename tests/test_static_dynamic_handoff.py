from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.static_dynamic_handoff import (
    DYNAMIC_PREFIX,
    DYNAMIC_SCHEMA,
    MUSCLE_COUNT,
    NV,
    STATIC_MUSCLE_PREFIX,
    STATIC_MUSCLE_SCHEMA,
    STATIC_REACTION_PREFIX,
    STATIC_REACTION_SCHEMA,
    audit,
)


def _static_records() -> tuple[dict, dict]:
    muscles = {
        "schema": STATIC_MUSCLE_SCHEMA,
        "activation_fp32": [0.2] * MUSCLE_COUNT,
        "reference_fiber_length_m": [0.1] * MUSCLE_COUNT,
        "actuator_force_n": [100.0] * MUSCLE_COUNT,
        "passive_actuator_force_n": [2.0] * MUSCLE_COUNT,
    }
    reactions = {
        "schema": STATIC_REACTION_SCHEMA,
        "muscle_force": [3.0] * NV,
        "passive_force": [0.5] * NV,
        "force_residual": [0.001] * NV,
    }
    return muscles, reactions


def _dynamic() -> dict:
    return {
        "schema": DYNAMIC_SCHEMA,
        "stage": "pre_step",
        "completed_steps": 0,
        "state_owner": "PersistentMetalHumanState.initial",
        "force_owner": "PersistentMetalHumanState.pre_step_force",
        "activation_fp32": [0.2] * MUSCLE_COUNT,
        "fiber_length_m": [0.1] * MUSCLE_COUNT,
        "actuator_force_n": [100.0] * MUSCLE_COUNT,
        "passive_actuator_force_n": [2.0] * MUSCLE_COUNT,
        "damped_equilibrium_residual": [0.0] * MUSCLE_COUNT,
        "generalized_muscle_force": [3.0] * NV,
        "generalized_passive_force": [0.5] * NV,
        "force_residual": [0.001] * NV,
        "metadata": {"fixture": True},
    }


def _write_log(path: Path, dynamic: dict | None = None) -> None:
    muscles, reactions = _static_records()
    lines = [
        STATIC_MUSCLE_PREFIX + json.dumps(muscles),
        STATIC_REACTION_PREFIX + json.dumps(reactions),
    ]
    if dynamic is not None:
        lines.append(DYNAMIC_PREFIX + json.dumps(dynamic))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _arguments(log: Path, output: Path, *, snapshot: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        log=log,
        dynamic_snapshot=snapshot,
        output=output,
        maximum_activation_delta=1.0e-7,
        fiber_absolute_tolerance=5.0e-7,
        fiber_relative_tolerance=5.0e-6,
        force_absolute_tolerance=5.0e-2,
        force_relative_tolerance=5.0e-5,
        residual_absolute_tolerance=1.0e-3,
        maximum_damped_equilibrium_residual=1.0e-5,
    )


def _run(tmp_path: Path, dynamic: dict | None) -> dict:
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    assert audit(_arguments(log, output)) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_exact_pre_step_handoff_passes(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _dynamic())
    assert receipt["status"] == "passed"
    assert receipt["coverage"]["muscles"] == 416
    assert receipt["coverage"]["generalized_coordinates"] == 128
    assert receipt["qualification"]["static_dynamic_handoff_parity"]
    assert receipt["comparisons"]["fiber_length"]["maximum_absolute_delta"] == 0.0


def test_missing_dynamic_snapshot_is_partial_not_silently_accepted(tmp_path: Path) -> None:
    receipt = _run(tmp_path, None)
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["pre_step_snapshot_present"]
    assert not receipt["qualification"]["static_dynamic_handoff_parity"]
    assert any("did not publish" in reason for reason in receipt["gate"]["reasons"])


def test_fiber_state_drift_identifies_worst_muscle(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["fiber_length_m"][37] += 0.001
    receipt = _run(tmp_path, dynamic)
    comparison = receipt["comparisons"]["fiber_length"]
    assert receipt["status"] == "partial"
    assert comparison["worst_index"] == 37
    assert not comparison["passed"]
    assert not receipt["qualification"]["activation_and_fiber_state_parity"]


def test_generalized_force_drift_identifies_worst_coordinate(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["generalized_muscle_force"][91] += 5.0
    receipt = _run(tmp_path, dynamic)
    comparison = receipt["comparisons"]["generalized_muscle_force"]
    assert comparison["worst_index"] == 91
    assert not receipt["qualification"]["generalized_force_parity"]
    assert any("generalized_muscle_force" in reason for reason in receipt["gate"]["reasons"])


def test_dynamic_fiber_equilibrium_residual_is_independently_gated(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["damped_equilibrium_residual"][12] = 0.01
    receipt = _run(tmp_path, dynamic)
    assert receipt["comparisons"]["actuator_force"]["passed"]
    assert not receipt["qualification"]["fiber_tendon_equilibrium_closed"]
    assert not receipt["qualification"]["static_dynamic_handoff_parity"]


def test_separate_snapshot_is_supported(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    snapshot = tmp_path / "handoff.json"
    output = tmp_path / "receipt.json"
    _write_log(log)
    snapshot.write_text(json.dumps(_dynamic()), encoding="utf-8")
    assert audit(_arguments(log, output, snapshot=snapshot)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["input"]["dynamic_snapshot"]["path"] == str(snapshot.resolve())


def test_embedded_and_separate_dynamic_snapshots_are_rejected(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    snapshot = tmp_path / "handoff.json"
    output = tmp_path / "receipt.json"
    _write_log(log, _dynamic())
    snapshot.write_text(json.dumps(_dynamic()), encoding="utf-8")
    with pytest.raises(ImportError, match="ambiguous"):
        audit(_arguments(log, output, snapshot=snapshot))


def test_malformed_dynamic_vector_is_rejected(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["actuator_force_n"].pop()
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="actuator_force_n"):
        audit(_arguments(log, output))


def test_post_step_snapshot_cannot_masquerade_as_handoff(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["stage"] = "post_step"
    dynamic["completed_steps"] = 1
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="pre_step"):
        audit(_arguments(log, output))

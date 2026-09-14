from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.static_dynamic_handoff import (
    ACCEPTED_FIBER_STATE_SOURCE,
    DYNAMIC_PREFIX,
    DYNAMIC_SCHEMA,
    GENERALIZED_FORCE_KEYS,
    GRAVITY_CONVENTION,
    MUSCLE_COUNT,
    NV,
    PASSIVE_BIAS_POLICY,
    SCHEMA,
    STATIC_MUSCLE_PREFIX,
    STATIC_MUSCLE_SCHEMA,
    STATIC_REACTION_PREFIX,
    STATIC_REACTION_SCHEMA,
    audit,
)


def _static_records(
    *,
    source_total: float = 100.0,
    zero_activation: float = 2.0,
    driven: float = 98.0,
) -> tuple[dict, dict]:
    muscles = {
        "schema": STATIC_MUSCLE_SCHEMA,
        "activation_fp32": [0.2] * MUSCLE_COUNT,
        "reference_fiber_length_m": [0.1] * MUSCLE_COUNT,
        "actuator_force_n": [source_total] * MUSCLE_COUNT,
        "passive_actuator_force_n": [zero_activation] * MUSCLE_COUNT,
        "driven_actuator_force_n": [driven] * MUSCLE_COUNT,
    }
    reactions = {
        "schema": STATIC_REACTION_SCHEMA,
        "muscle_force": [3.0] * NV,
        "equality_force": [2.0] * NV,
        "limit_force": [1.0] * NV,
        "support_force": [4.0] * NV,
        "passive_force": [0.5] * NV,
        "gravity_target": [9.0] * NV,
        "force_residual": [1.5] * NV,
    }
    return muscles, reactions


def _dynamic(
    *,
    source_total: float = 100.0,
    excluded_bias: float = 2.0,
    driven: float = 98.0,
) -> dict:
    return {
        "schema": DYNAMIC_SCHEMA,
        "stage": "pre_step",
        "completed_steps": 0,
        "passive_bias_policy": PASSIVE_BIAS_POLICY,
        "gravity_convention": GRAVITY_CONVENTION,
        "fiber_state_source": ACCEPTED_FIBER_STATE_SOURCE,
        "state_owner": "PersistentMetalHumanState.initial",
        "force_owner": "PersistentMetalHumanState.pre_step_force",
        "activation_fp32": [0.2] * MUSCLE_COUNT,
        "fiber_length_m": [0.1] * MUSCLE_COUNT,
        "source_total_actuator_force_n": [source_total] * MUSCLE_COUNT,
        "excluded_passive_bias_force_n": [excluded_bias] * MUSCLE_COUNT,
        "driven_actuator_force_n": [driven] * MUSCLE_COUNT,
        "damped_equilibrium_residual": [0.0] * MUSCLE_COUNT,
        "generalized_muscle_force": [3.0] * NV,
        "generalized_joint_equality_force": [2.0] * NV,
        "generalized_position_limit_force": [1.0] * NV,
        "generalized_support_force": [4.0] * NV,
        "generalized_passive_force": [0.5] * NV,
        "gravity_target": [9.0] * NV,
        "force_residual": [1.5] * NV,
        "metadata": {"fixture": True},
    }


def _write_log(
    path: Path,
    dynamic: dict | None = None,
    *,
    static_records: tuple[dict, dict] | None = None,
) -> None:
    muscles, reactions = static_records or _static_records()
    lines = [
        STATIC_MUSCLE_PREFIX + json.dumps(muscles),
        STATIC_REACTION_PREFIX + json.dumps(reactions),
    ]
    if dynamic is not None:
        lines.append(DYNAMIC_PREFIX + json.dumps(dynamic))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _arguments(
    log: Path, output: Path, *, snapshot: Path | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        log=log,
        dynamic_snapshot=snapshot,
        output=output,
        maximum_activation_delta=1.0e-7,
        fiber_absolute_tolerance=5.0e-7,
        fiber_relative_tolerance=5.0e-6,
        force_absolute_tolerance=5.0e-2,
        force_relative_tolerance=5.0e-5,
        decomposition_absolute_tolerance=1.0e-6,
        residual_absolute_tolerance=1.0e-3,
        assembly_absolute_tolerance=1.0e-4,
        maximum_damped_equilibrium_residual=1.0e-5,
    )


def _run(
    tmp_path: Path,
    dynamic: dict | None,
    *,
    static_records: tuple[dict, dict] | None = None,
) -> dict:
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic, static_records=static_records)
    assert audit(_arguments(log, output)) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_exact_v3_pre_step_handoff_passes(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _dynamic())
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["coverage"]["force_owners"] == list(GENERALIZED_FORCE_KEYS)
    assert receipt["coverage"]["gravity_convention"] == GRAVITY_CONVENTION
    assert receipt["coverage"]["accepted_fiber_state_source"] == ACCEPTED_FIBER_STATE_SOURCE
    qualification = receipt["qualification"]
    assert qualification["accepted_fiber_state_transport"]
    assert qualification["full_force_owner_parity"]
    assert qualification["static_force_assembly_closed"]
    assert qualification["dynamic_force_assembly_closed"]
    assert qualification["static_dynamic_handoff_parity"]
    assert receipt["force_assembly"]["static"]["maximum_absolute_delta"] == 0.0
    assert receipt["force_assembly"]["dynamic"]["maximum_absolute_delta"] == 0.0


def test_missing_dynamic_snapshot_is_partial(tmp_path: Path) -> None:
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
    assert comparison["worst_index"] == 37
    assert not comparison["passed"]
    assert not receipt["qualification"]["activation_and_fiber_state_parity"]


def test_accepted_fiber_source_is_mandatory(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["fiber_state_source"] = "zero_length_sentinel"
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="accepted static fibre"):
        audit(_arguments(log, output))


def test_driven_and_excluded_bias_drift_are_localized(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["driven_actuator_force_n"][42] += 5.0
    dynamic["source_total_actuator_force_n"][42] += 5.0
    receipt = _run(tmp_path, dynamic)
    assert receipt["comparisons"]["driven_actuator_force"]["worst_index"] == 42
    assert not receipt["qualification"]["per_muscle_force_parity"]

    dynamic = _dynamic()
    dynamic["excluded_passive_bias_force_n"][13] += 1.0
    dynamic["source_total_actuator_force_n"][13] += 1.0
    receipt = _run(tmp_path, dynamic)
    assert receipt["comparisons"]["excluded_passive_bias_force"]["worst_index"] == 13
    assert not receipt["qualification"]["per_muscle_force_parity"]


def test_source_force_decomposition_is_independently_gated(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["source_total_actuator_force_n"][9] += 1.0
    receipt = _run(tmp_path, dynamic)
    assert not receipt["source_force_decomposition"]["passed"]
    assert receipt["source_force_decomposition"]["worst_index"] == 9
    assert not receipt["qualification"]["source_force_decomposition_closed"]


def test_zero_activation_force_remains_diagnostic(tmp_path: Path) -> None:
    static = _static_records(source_total=100.0, zero_activation=5.0, driven=100.0)
    dynamic = _dynamic(source_total=100.0, excluded_bias=0.0, driven=100.0)
    receipt = _run(tmp_path, dynamic, static_records=static)
    assert receipt["status"] == "passed"
    assert receipt["metadata"]["static_zero_activation_force_role"] == "diagnostic_only"


@pytest.mark.parametrize(
    ("field", "comparison", "index", "delta", "residual_delta"),
    [
        ("generalized_muscle_force", "generalized_muscle_force", 91, 5.0, 5.0),
        ("generalized_joint_equality_force", "generalized_joint_equality_force", 47, 5.0, 5.0),
        ("generalized_position_limit_force", "generalized_position_limit_force", 103, 5.0, 5.0),
        ("generalized_support_force", "generalized_support_force", 2, 5.0, 5.0),
        ("generalized_passive_force", "generalized_passive_force", 67, 5.0, 5.0),
        ("gravity_target", "gravity_target", 8, 5.0, -5.0),
    ],
)
def test_each_force_owner_drift_is_localized_while_assembly_stays_closed(
    tmp_path: Path,
    field: str,
    comparison: str,
    index: int,
    delta: float,
    residual_delta: float,
) -> None:
    dynamic = _dynamic()
    dynamic[field][index] += delta
    dynamic["force_residual"][index] += residual_delta
    receipt = _run(tmp_path, dynamic)
    assert receipt["comparisons"][comparison]["worst_index"] == index
    assert not receipt["comparisons"][comparison]["passed"]
    assert receipt["force_assembly"]["dynamic"]["passed"]
    assert not receipt["qualification"]["full_force_owner_parity"]


def test_dynamic_force_assembly_corruption_is_rejected(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["force_residual"][7] += 1.0
    receipt = _run(tmp_path, dynamic)
    assert receipt["force_assembly"]["dynamic"]["worst_index"] == 7
    assert not receipt["qualification"]["dynamic_force_assembly_closed"]
    assert any("dynamic generalized force owners" in reason for reason in receipt["gate"]["reasons"])


def test_static_force_assembly_corruption_is_rejected(tmp_path: Path) -> None:
    muscles, reactions = _static_records()
    reactions["force_residual"][11] += 1.0
    receipt = _run(tmp_path, _dynamic(), static_records=(muscles, reactions))
    assert receipt["force_assembly"]["static"]["worst_index"] == 11
    assert not receipt["qualification"]["static_force_assembly_closed"]


def test_dynamic_fiber_equilibrium_residual_is_independently_gated(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["damped_equilibrium_residual"][12] = 0.01
    receipt = _run(tmp_path, dynamic)
    assert receipt["worst_damped_equilibrium_muscle"] == 12
    assert not receipt["qualification"]["fiber_tendon_equilibrium_closed"]


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


def test_embedded_and_separate_snapshots_are_rejected(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    snapshot = tmp_path / "handoff.json"
    output = tmp_path / "receipt.json"
    _write_log(log, _dynamic())
    snapshot.write_text(json.dumps(_dynamic()), encoding="utf-8")
    with pytest.raises(ImportError, match="ambiguous"):
        audit(_arguments(log, output, snapshot=snapshot))


def test_v2_snapshot_cannot_satisfy_v3_contract(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["schema"] = "numi.human.dynamic-handoff.v2"
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="schema mismatch"):
        audit(_arguments(log, output))


def test_gravity_convention_mismatch_is_rejected(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["gravity_convention"] = "force_residual=gravity+owners"
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="gravity convention"):
        audit(_arguments(log, output))


def test_malformed_force_owner_vector_is_rejected(tmp_path: Path) -> None:
    dynamic = _dynamic()
    dynamic["generalized_support_force"].pop()
    log = tmp_path / "native.log"
    output = tmp_path / "receipt.json"
    _write_log(log, dynamic)
    with pytest.raises(ImportError, match="generalized_support_force"):
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

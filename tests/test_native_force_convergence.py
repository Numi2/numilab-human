from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_force_convergence import (
    HANDOFF_PASSIVE_BIAS_POLICY,
    audit,
)


LINE = (
    'myosim_articulated_marker_visual=ok metal_pose_device="Apple M4 Pro" '
    'core_bodies=157 muscle_step_count=512 muscle_step_seconds=1.25e-05 '
    'persistent_completed_steps=512 persistent_max_acceleration=46673.2 '
    'persistent_max_penetration_m=1.9e-7 muscle_step_max_velocity_delta=0.472 '
    'muscle_step_max_configuration_delta=0.0015 compiled_stand_balanced=false '
    'compiled_stand_max_root_force_residual=776.8 compiled_stand_support_contacts=18 '
    'compiled_stand_active_support_contacts=2 compiled_stand_total_support_force_n=176 '
    'source_dynamic_force_parity_max_delta_n=0.01 muscle_force_metal_elapsed_ms=100 '
    'stand_deterministic_replay=not_requested\n'
)

HANDOFF_COUNTS = {
    "activation": 416,
    "fiber_length": 416,
    "source_total_actuator_force": 416,
    "driven_actuator_force": 416,
    "excluded_passive_bias_force": 416,
    "generalized_muscle_force": 128,
    "generalized_passive_force": 128,
    "force_residual": 128,
}


def _arguments(stdout: Path, output: Path, **overrides) -> argparse.Namespace:
    values = dict(
        stdout=stdout,
        stderr=None,
        replay_stdout=None,
        build_log=None,
        standing_state_receipt=None,
        require_standing_state_receipt=False,
        force_ledger_receipt=None,
        require_force_ledger_receipt=False,
        static_dynamic_handoff_receipt=None,
        require_static_dynamic_handoff_receipt=False,
        output=output,
        source_commit="fixture",
        binary_sha256="0" * 64,
        subject="one adult male source package",
        body_count=157,
        dof_count=128,
        q_count=129,
        exact_body_limit=32,
        exact_dof_limit=40,
        exact_q_limit=41,
        minimum_steps=512,
        maximum_acceleration=1000.0,
        maximum_velocity_delta=0.01,
        maximum_configuration_delta=1.0e-4,
        require_same_horizon_replay=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _converged_line() -> str:
    return LINE.replace(
        "persistent_max_acceleration=46673.2", "persistent_max_acceleration=1"
    ).replace(
        "muscle_step_max_velocity_delta=0.472", "muscle_step_max_velocity_delta=0.001"
    ).replace(
        "muscle_step_max_configuration_delta=0.0015",
        "muscle_step_max_configuration_delta=0.00001",
    ).replace(
        "compiled_stand_balanced=false", "compiled_stand_balanced=true"
    ).replace(
        "compiled_stand_max_root_force_residual=776.8",
        "compiled_stand_max_root_force_residual=0.001",
    )


def _state(path: Path, *, candidate: bool = True, transported: bool = True) -> None:
    path.write_text(json.dumps({
        "schema": "numi.human.standing-initial-state-audit.v1",
        "initial_state": {"sha256": "1" * 64},
        "state": {
            "uniform_maximal_activation": not candidate,
            "activation_nonzero_count": 237 if candidate else 416,
        },
        "qualification": {
            "standing_initial_state_candidate": candidate,
            "equilibrium_state_transport": transported,
        },
    }), encoding="utf-8")


def _ledger(path: Path, *, complete: bool = True) -> None:
    path.write_text(json.dumps({
        "schema": "numi.human.generalized-force-ledger.v1",
        "coverage": {"full_force_coverage": complete},
        "residual": {
            "maximum_assembly_error": 0.0 if complete else 1.0,
            "maximum_closure_ratio": 0.0 if complete else 1.0,
        },
        "worst_coordinates": [],
        "qualification": {"full_generalized_force_ledger": complete},
    }), encoding="utf-8")


def _comparison(count: int, *, passed: bool = True) -> dict:
    return {
        "count": count,
        "absolute_tolerance": 1.0e-7,
        "relative_tolerance": 0.0,
        "maximum_absolute_delta": 0.0 if passed else 1.0,
        "rms_absolute_delta": 0.0 if passed else 0.1,
        "maximum_normalized_error": 0.0 if passed else 2.0,
        "worst_index": 0,
        "worst_reference": 0.0,
        "worst_candidate": 0.0 if passed else 1.0,
        "worst_absolute_delta": 0.0 if passed else 1.0,
        "passed": passed,
    }


def _handoff_payload(*, complete: bool = True) -> dict:
    evidence = complete
    return {
        "schema": "numi.human.static-dynamic-handoff-audit.v2",
        "status": "passed" if evidence else "partial",
        "coverage": {
            "muscles": 416,
            "generalized_coordinates": 128,
            "static_muscle_state": True,
            "static_generalized_forces": True,
            "static_zero_activation_force_diagnostic": True,
            "passive_bias_policy": HANDOFF_PASSIVE_BIAS_POLICY,
            "dynamic_pre_step_state": evidence,
            "dynamic_state_owner": "PersistentMetalHumanState.initial" if evidence else None,
            "dynamic_force_owner": "PersistentMetalHumanState.pre_step_force" if evidence else None,
        },
        "thresholds": {
            "activation_absolute": 1.0e-7,
            "fiber_absolute_m": 5.0e-7,
            "fiber_relative": 5.0e-6,
            "force_absolute_n": 5.0e-2,
            "force_relative": 5.0e-5,
            "decomposition_absolute_n": 1.0e-6,
            "residual_absolute": 1.0e-3,
            "maximum_damped_equilibrium_residual": 1.0e-5,
        },
        "comparisons": {
            name: _comparison(count, passed=evidence)
            for name, count in HANDOFF_COUNTS.items()
        },
        "source_force_decomposition": _comparison(416, passed=evidence),
        "maximum_damped_equilibrium_residual": 0.0 if evidence else 0.01,
        "qualification": {
            "pre_step_snapshot_present": evidence,
            "activation_and_fiber_state_parity": evidence,
            "per_muscle_force_parity": evidence,
            "source_force_decomposition_closed": evidence,
            "generalized_force_parity": evidence,
            "fiber_tendon_equilibrium_closed": evidence,
            "static_dynamic_handoff_parity": complete,
        },
        "gate": {"reasons": [] if evidence else ["fixture incomplete"]},
    }


def _handoff(path: Path, *, complete: bool = True) -> None:
    path.write_text(json.dumps(_handoff_payload(complete=complete)), encoding="utf-8")


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.json"
    handoff = tmp_path / "handoff.json"
    _state(state)
    _ledger(ledger)
    _handoff(handoff)
    return state, ledger, handoff


def _audit(tmp_path: Path, line: str, **overrides) -> dict:
    stdout = tmp_path / "stdout.log"
    output = tmp_path / "receipt.json"
    stdout.write_text(line, encoding="utf-8")
    assert audit(_arguments(stdout, output, **overrides)) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_unconverged_native_horizon_remains_partial(tmp_path: Path) -> None:
    receipt = _audit(tmp_path, LINE)
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["mechanical_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_mechanical_convergence_does_not_claim_standing(tmp_path: Path) -> None:
    receipt = _audit(tmp_path, _converged_line())
    assert receipt["qualification"]["mechanical_force_convergence"]
    assert receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_standing_force_convergence_requires_v2_handoff(tmp_path: Path) -> None:
    state, ledger, handoff = _evidence(tmp_path)
    receipt = _audit(
        tmp_path,
        _converged_line(),
        standing_state_receipt=state,
        require_standing_state_receipt=True,
        force_ledger_receipt=ledger,
        require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )
    assert receipt["qualification"]["static_dynamic_handoff_admissible"]
    assert receipt["static_dynamic_handoff"]["source_force_decomposition_closed"]
    assert receipt["qualification"]["standing_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_legacy_scalar_cannot_replace_structured_handoff(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.json"
    _state(state)
    _ledger(ledger)
    receipt = _audit(
        tmp_path,
        _converged_line().replace(
            "source_dynamic_force_parity_max_delta_n=0.01",
            "source_dynamic_force_parity_max_delta_n=0",
        ),
        standing_state_receipt=state,
        force_ledger_receipt=ledger,
    )
    assert receipt["horizon"]["source_dynamic_force_parity_max_delta_n"] == 0.0
    assert not receipt["qualification"]["standing_force_convergence"]


def test_incomplete_required_handoff_cannot_pass(tmp_path: Path) -> None:
    state, ledger, handoff = _evidence(tmp_path)
    _handoff(handoff, complete=False)
    receipt = _audit(
        tmp_path,
        _converged_line(),
        standing_state_receipt=state,
        require_standing_state_receipt=True,
        force_ledger_receipt=ledger,
        require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )
    assert not receipt["qualification"]["force_convergence"]
    assert any("handoff" in reason for reason in receipt["gate"]["reasons"])


def _reject_mutation(tmp_path: Path, mutate) -> None:
    stdout = tmp_path / "stdout.log"
    handoff = tmp_path / "handoff.json"
    output = tmp_path / "receipt.json"
    stdout.write_text(_converged_line(), encoding="utf-8")
    payload = _handoff_payload()
    mutate(payload)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImportError, match="claims parity"):
        audit(_arguments(stdout, output, static_dynamic_handoff_receipt=handoff))


def test_forged_empty_comparisons_are_rejected(tmp_path: Path) -> None:
    _reject_mutation(tmp_path, lambda payload: payload.__setitem__("comparisons", {}))


def test_forged_decomposition_is_rejected(tmp_path: Path) -> None:
    _reject_mutation(
        tmp_path,
        lambda payload: payload["source_force_decomposition"].__setitem__("passed", False),
    )


def test_relaxed_decomposition_threshold_is_rejected(tmp_path: Path) -> None:
    _reject_mutation(
        tmp_path,
        lambda payload: payload["thresholds"].__setitem__(
            "decomposition_absolute_n", 1.0e6
        ),
    )


def test_wrong_passive_bias_policy_is_rejected(tmp_path: Path) -> None:
    _reject_mutation(
        tmp_path,
        lambda payload: payload["coverage"].__setitem__(
            "passive_bias_policy", "total_source_force"
        ),
    )


def test_missing_zero_activation_diagnostic_identity_is_rejected(
    tmp_path: Path,
) -> None:
    _reject_mutation(
        tmp_path,
        lambda payload: payload["coverage"].__setitem__(
            "static_zero_activation_force_diagnostic", False
        ),
    )


def test_unbound_standing_state_cannot_pass(tmp_path: Path) -> None:
    state, ledger, handoff = _evidence(tmp_path)
    _state(state, transported=False)
    receipt = _audit(
        tmp_path,
        _converged_line(),
        standing_state_receipt=state,
        require_standing_state_receipt=True,
        force_ledger_receipt=ledger,
        require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )
    assert not receipt["qualification"]["standing_state_admissible"]
    assert not receipt["qualification"]["force_convergence"]

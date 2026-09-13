from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_force_convergence import audit


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


def _arguments(stdout: Path, output: Path, **overrides) -> argparse.Namespace:
    values = dict(
        stdout=stdout, stderr=None, replay_stdout=None, build_log=None,
        standing_state_receipt=None, require_standing_state_receipt=False,
        force_ledger_receipt=None, require_force_ledger_receipt=False,
        static_dynamic_handoff_receipt=None,
        require_static_dynamic_handoff_receipt=False,
        output=output, source_commit="fixture", binary_sha256="0" * 64,
        subject="one adult male source package", body_count=157, dof_count=128, q_count=129,
        exact_body_limit=32, exact_dof_limit=40, exact_q_limit=41,
        minimum_steps=512, maximum_acceleration=1000.0,
        maximum_velocity_delta=0.01, maximum_configuration_delta=1.0e-4,
        require_same_horizon_replay=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _converged_line() -> str:
    return LINE.replace("persistent_max_acceleration=46673.2", "persistent_max_acceleration=1").replace(
        "muscle_step_max_velocity_delta=0.472", "muscle_step_max_velocity_delta=0.001"
    ).replace(
        "muscle_step_max_configuration_delta=0.0015", "muscle_step_max_configuration_delta=0.00001"
    ).replace("compiled_stand_balanced=false", "compiled_stand_balanced=true").replace(
        "compiled_stand_max_root_force_residual=776.8", "compiled_stand_max_root_force_residual=0.001"
    )


def _state_receipt(
    path: Path, *, candidate: bool, maximal: bool, equilibrium_transport: bool | None = None
) -> None:
    if equilibrium_transport is None:
        equilibrium_transport = candidate
    path.write_text(json.dumps({
        "schema": "numi.human.standing-initial-state-audit.v1",
        "initial_state": {"sha256": "1" * 64},
        "state": {
            "uniform_maximal_activation": maximal,
            "activation_nonzero_count": 237 if candidate else 416,
        },
        "qualification": {
            "standing_initial_state_candidate": candidate,
            "equilibrium_state_transport": equilibrium_transport,
        },
    }), encoding="utf-8")


def _force_ledger(path: Path, *, complete: bool) -> None:
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


def _handoff(path: Path, *, complete: bool, inconsistent_claim: bool = False) -> None:
    evidence = complete and not inconsistent_claim
    path.write_text(json.dumps({
        "schema": "numi.human.static-dynamic-handoff-audit.v1",
        "status": "passed" if complete else "partial",
        "coverage": {
            "muscles": 416,
            "generalized_coordinates": 128,
            "dynamic_pre_step_state": evidence,
        },
        "comparisons": {} if evidence else None,
        "maximum_damped_equilibrium_residual": 0.0 if evidence else None,
        "qualification": {
            "pre_step_snapshot_present": evidence,
            "activation_and_fiber_state_parity": evidence,
            "per_muscle_force_parity": evidence,
            "generalized_force_parity": evidence,
            "fiber_tendon_equilibrium_closed": evidence,
            "static_dynamic_handoff_parity": complete,
        },
    }), encoding="utf-8")


def test_native_force_audit_retains_partial_status(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(LINE, encoding="utf-8")
    stderr = tmp_path / "stderr"
    stderr.write_text("", encoding="utf-8")
    output = tmp_path / "receipt.json"
    assert audit(_arguments(stdout, output, stderr=stderr)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]
    assert receipt["exact_dense_stage"]["selected_path"] == "large_state_fallback"


def test_force_convergence_does_not_promote_sustained_standing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    output = tmp_path / "receipt.json"
    assert audit(_arguments(stdout, output)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["qualification"]["mechanical_force_convergence"]
    assert receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_standing_force_convergence_requires_state_ledger_and_handoff(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=True, maximal=False)
    _force_ledger(ledger, complete=True)
    _handoff(handoff, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state, require_standing_state_receipt=True,
        force_ledger_receipt=ledger, require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["qualification"]["standing_state_admissible"]
    assert receipt["qualification"]["force_ledger_admissible"]
    assert receipt["qualification"]["static_dynamic_handoff_admissible"]
    assert receipt["qualification"]["force_convergence"]
    assert receipt["qualification"]["standing_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_legacy_scalar_cannot_substitute_for_structured_handoff(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(
        _converged_line().replace("source_dynamic_force_parity_max_delta_n=0.01", "source_dynamic_force_parity_max_delta_n=0"),
        encoding="utf-8",
    )
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    _state_receipt(state, candidate=True, maximal=False)
    _force_ledger(ledger, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state,
        force_ledger_receipt=ledger,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["horizon"]["source_dynamic_force_parity_max_delta_n"] == 0.0
    assert not receipt["qualification"]["static_dynamic_handoff_admissible"]
    assert not receipt["qualification"]["standing_force_convergence"]


def test_legacy_scalar_may_be_absent_when_structured_handoff_is_used(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    line = _converged_line().replace("source_dynamic_force_parity_max_delta_n=0.01 ", "")
    stdout.write_text(line, encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=True, maximal=False)
    _force_ledger(ledger, complete=True)
    _handoff(handoff, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state,
        force_ledger_receipt=ledger,
        static_dynamic_handoff_receipt=handoff,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["horizon"]["source_dynamic_force_parity_max_delta_n"] is None
    assert receipt["qualification"]["standing_force_convergence"]


def test_stationary_but_unbound_state_cannot_pass_standing(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=True, maximal=False, equilibrium_transport=False)
    _force_ledger(ledger, complete=True)
    _handoff(handoff, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state, require_standing_state_receipt=True,
        force_ledger_receipt=ledger, require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["qualification"]["mechanical_force_convergence"]
    assert not receipt["qualification"]["standing_state_admissible"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert any("solved coupled equilibrium" in reason for reason in receipt["gate"]["reasons"])


def test_required_incomplete_force_ledger_cannot_pass(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=True, maximal=False)
    _force_ledger(ledger, complete=False)
    _handoff(handoff, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state, require_standing_state_receipt=True,
        force_ledger_receipt=ledger, require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["qualification"]["mechanical_force_convergence"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert any("force ledger" in reason for reason in receipt["gate"]["reasons"])


def test_required_incomplete_handoff_cannot_pass(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=True, maximal=False)
    _force_ledger(ledger, complete=True)
    _handoff(handoff, complete=False)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state, require_standing_state_receipt=True,
        force_ledger_receipt=ledger, require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert not receipt["qualification"]["static_dynamic_handoff_admissible"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert any("handoff" in reason for reason in receipt["gate"]["reasons"])


def test_inconsistent_claimed_handoff_is_rejected(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    handoff = tmp_path / "handoff.json"
    _handoff(handoff, complete=True, inconsistent_claim=True)
    output = tmp_path / "receipt.json"
    with pytest.raises(ImportError, match="claims parity"):
        audit(_arguments(stdout, output, static_dynamic_handoff_receipt=handoff))


def test_required_maximal_activation_state_cannot_pass(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text(_converged_line(), encoding="utf-8")
    state = tmp_path / "standing-state.json"
    ledger = tmp_path / "force-ledger.json"
    handoff = tmp_path / "handoff.json"
    _state_receipt(state, candidate=False, maximal=True, equilibrium_transport=True)
    _force_ledger(ledger, complete=True)
    _handoff(handoff, complete=True)
    output = tmp_path / "receipt.json"
    assert audit(_arguments(
        stdout, output,
        standing_state_receipt=state, require_standing_state_receipt=True,
        force_ledger_receipt=ledger, require_force_ledger_receipt=True,
        static_dynamic_handoff_receipt=handoff,
        require_static_dynamic_handoff_receipt=True,
    )) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["standing_force_convergence"]
    assert any("maximal-activation" in reason for reason in receipt["gate"]["reasons"])

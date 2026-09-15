from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


SCHEMA = "numi.human.native-force-convergence-audit.v1"
STANDING_STATE_SCHEMA = "numi.human.standing-initial-state-audit.v1"
FORCE_LEDGER_SCHEMA = "numi.human.generalized-force-ledger.v1"
STATIC_DYNAMIC_HANDOFF_SCHEMA = "numi.human.static-dynamic-handoff-audit.v3"
HANDOFF_PASSIVE_BIAS_POLICY = (
    "legacy_zero_activation_bias_excluded_compliant_tendon_force_retained"
)
HANDOFF_GRAVITY_CONVENTION = (
    "force_residual=muscle+equality+limit+support+passive-gravity_target"
)
HANDOFF_ACCEPTED_FIBER_STATE_SOURCE = (
    "compiled_equilibrium_reference_fiber_length"
)
_HANDOFF_FORCE_OWNERS = (
    "generalized_muscle_force",
    "generalized_joint_equality_force",
    "generalized_position_limit_force",
    "generalized_support_force",
    "generalized_passive_force",
    "gravity_target",
    "force_residual",
)
_HANDOFF_COUNTS = {
    "activation": 416,
    "fiber_length": 416,
    "source_total_actuator_force": 416,
    "driven_actuator_force": 416,
    "excluded_passive_bias_force": 416,
    "generalized_muscle_force": 128,
    "generalized_joint_equality_force": 128,
    "generalized_position_limit_force": 128,
    "generalized_support_force": 128,
    "generalized_passive_force": 128,
    "gravity_target": 128,
    "force_residual": 128,
}
_HANDOFF_THRESHOLD_CEILINGS = {
    "activation_absolute": 1.0e-7,
    "fiber_absolute_m": 5.0e-7,
    "fiber_relative": 5.0e-6,
    "force_absolute_n": 5.0e-2,
    "force_relative": 5.0e-5,
    "decomposition_absolute_n": 1.0e-6,
    "residual_absolute": 1.0e-3,
    "assembly_absolute": 1.0e-4,
    "maximum_damped_equilibrium_residual": 1.0e-5,
}


def _native_metrics(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ImportError(f"cannot read native stdout {path}: {error}") from error
    kinds = (
        "myosim_articulated_marker_visual",
        "myosim_articulated_bodyparts_bone_visual",
        "myosim_articulated_mechanics",
    )
    matches = [(kind, line) for line in lines for kind in kinds
               if line.startswith(kind + "=")]
    if len(matches) != 1:
        raise ImportError(f"{path} must contain exactly one native result line")
    kind, line = matches[0]
    try:
        tokens = shlex.split(line)
    except ValueError as error:
        raise ImportError(f"{path} contains a malformed native result") from error
    values: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in values:
            raise ImportError(f"native result contains duplicate metric {key}")
        values[key] = value
    if values.get(kind) != "ok" or sum(key in values for key in kinds) != 1:
        raise ImportError(f"{path} does not contain an unambiguous successful native result")
    if kind == "myosim_articulated_mechanics":
        if (values.get("rendering_performed") != "false" or
                values.get("visual_coverage_qualified") != "false"):
            raise ImportError("mechanics-only result must explicitly exclude visual qualification")
    return values


def _field(values: dict[str, str], name: str) -> str:
    value = values.get(name)
    if value is None or value == "":
        raise ImportError(f"native result is missing {name}")
    return value


def _number(values: dict[str, str], name: str) -> float:
    try:
        result = float(_field(values, name))
    except ValueError as error:
        raise ImportError(f"native result field {name} is not numeric") from error
    if not math.isfinite(result):
        raise ImportError(f"native result field {name} is not finite")
    return result


def _integer(values: dict[str, str], name: str) -> int:
    value = _number(values, name)
    if value != int(value):
        raise ImportError(f"native result field {name} is not integral")
    return int(value)


def _boolean(values: dict[str, str], name: str) -> bool:
    value = _field(values, name).lower()
    if value not in {"true", "false"}:
        raise ImportError(f"native result field {name} is not boolean")
    return value == "true"


def _metrics(values: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": _field(values, "metal_pose_device"),
        "core_bodies": _integer(values, "core_bodies"),
        "muscle_step_count": _integer(values, "muscle_step_count"),
        "timestep_seconds": _number(values, "muscle_step_seconds"),
        "persistent_completed_steps": _integer(values, "persistent_completed_steps"),
        "persistent_max_acceleration": _number(values, "persistent_max_acceleration"),
        "persistent_max_penetration_m": _number(
            values, "persistent_max_penetration_m"
        ),
        "muscle_step_max_velocity_delta": _number(
            values, "muscle_step_max_velocity_delta"
        ),
        "muscle_step_max_configuration_delta": _number(
            values, "muscle_step_max_configuration_delta"
        ),
        "compiled_stand_balanced": _boolean(values, "compiled_stand_balanced"),
        "compiled_stand_max_root_force_residual_n": _number(
            values, "compiled_stand_max_root_force_residual"
        ),
        "compiled_stand_support_contacts": _integer(
            values, "compiled_stand_support_contacts"
        ),
        "compiled_stand_active_support_contacts": _integer(
            values, "compiled_stand_active_support_contacts"
        ),
        "compiled_stand_total_support_force_n": _number(
            values, "compiled_stand_total_support_force_n"
        ),
        "muscle_force_metal_elapsed_ms": _number(
            values, "muscle_force_metal_elapsed_ms"
        ),
        "stand_deterministic_replay": _field(values, "stand_deterministic_replay"),
        "source_dynamic_force_parity_max_delta_n": (
            _number(values, "source_dynamic_force_parity_max_delta_n")
            if "source_dynamic_force_parity_max_delta_n" in values
            else None
        ),
    }
    if "compiled_stand_normalized_residual_rms" in values:
        result["compiled_stand_normalized_residual_rms"] = _number(
            values, "compiled_stand_normalized_residual_rms"
        )
    if "compiled_stand_max_activation" in values:
        result["compiled_stand_max_activation"] = _number(
            values, "compiled_stand_max_activation"
        )
    return result


def _replay_summary(path: Path | None, main: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"same_horizon": "not_supplied", "one_step_bitwise": False}
    replay = _metrics(_native_metrics(path))
    same_horizon = (
        replay["persistent_completed_steps"] == main["persistent_completed_steps"]
    )
    return {
        "same_horizon": (
            "bitwise"
            if same_horizon and replay["stand_deterministic_replay"] == "bitwise"
            else "not_proved"
        ),
        "one_step_bitwise": replay["stand_deterministic_replay"] == "bitwise",
        "steps": replay["persistent_completed_steps"],
        "stdout_sha256": sha256(path),
    }


def _load_receipt(
    path: Path, schema: str, label: str
) -> tuple[Path, dict[str, Any]]:
    resolved = path.resolve()
    try:
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != schema:
        raise ImportError(f"{label} schema mismatch")
    return resolved, receipt


def _standing_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved, receipt = _load_receipt(
        path, STANDING_STATE_SCHEMA, "standing-state receipt"
    )
    qualification = receipt.get("qualification")
    state = receipt.get("state")
    if not isinstance(qualification, dict) or not isinstance(state, dict):
        raise ImportError("standing-state receipt is missing qualification/state")
    candidate = qualification.get("standing_initial_state_candidate") is True
    equilibrium_transport = qualification.get("equilibrium_state_transport") is True
    uniform_maximal = state.get("uniform_maximal_activation")
    if type(uniform_maximal) is not bool:
        raise ImportError(
            "standing-state receipt has no boolean maximal-activation classification"
        )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "candidate": candidate,
        "equilibrium_state_transport": equilibrium_transport,
        "uniform_maximal_activation": uniform_maximal,
        "activation_nonzero_count": state.get("activation_nonzero_count"),
        "initial_state_sha256": (receipt.get("initial_state") or {}).get("sha256"),
    }


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0.0


def _force_ledger(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved, receipt = _load_receipt(
        path, FORCE_LEDGER_SCHEMA, "generalized-force ledger"
    )
    qualification = receipt.get("qualification")
    coverage = receipt.get("coverage")
    residual = receipt.get("residual")
    if not all(
        isinstance(value, dict)
        for value in (qualification, coverage, residual)
    ):
        raise ImportError(
            "generalized-force ledger is missing coverage/residual/qualification"
        )
    complete = qualification.get("full_generalized_force_ledger") is True
    assembly_error = residual.get("maximum_assembly_error")
    closure_ratio = residual.get("maximum_closure_ratio")
    if complete and not (
        coverage.get("full_force_coverage") is True
        and _finite_nonnegative(assembly_error)
        and _finite_nonnegative(closure_ratio)
    ):
        raise ImportError(
            "generalized-force ledger claims completion without finite coverage evidence"
        )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "complete": complete,
        "full_force_coverage": coverage.get("full_force_coverage") is True,
        "maximum_assembly_error": assembly_error,
        "maximum_closure_ratio": closure_ratio,
        "worst_coordinates": receipt.get("worst_coordinates", [])[:6],
    }


def _handoff_comparison(value: Any, expected_count: int) -> bool:
    if not isinstance(value, dict):
        return False
    worst_index = value.get("worst_index")
    maximum_normalized_error = value.get("maximum_normalized_error")
    return (
        value.get("count") == expected_count
        and type(worst_index) is int
        and 0 <= worst_index < expected_count
        and value.get("passed") is True
        and _finite_nonnegative(value.get("absolute_tolerance"))
        and _finite_nonnegative(value.get("relative_tolerance"))
        and _finite_nonnegative(value.get("maximum_absolute_delta"))
        and _finite_nonnegative(value.get("rms_absolute_delta"))
        and _finite_nonnegative(maximum_normalized_error)
        and maximum_normalized_error <= 1.0
        and type(value.get("worst_reference")) in (int, float)
        and math.isfinite(value["worst_reference"])
        and type(value.get("worst_candidate")) in (int, float)
        and math.isfinite(value["worst_candidate"])
        and _finite_nonnegative(value.get("worst_absolute_delta"))
    )


def _static_dynamic_handoff(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved, receipt = _load_receipt(
        path, STATIC_DYNAMIC_HANDOFF_SCHEMA, "static-dynamic handoff receipt"
    )
    qualification = receipt.get("qualification")
    coverage = receipt.get("coverage")
    thresholds = receipt.get("thresholds")
    comparisons = receipt.get("comparisons")
    decomposition = receipt.get("source_force_decomposition")
    force_assembly = receipt.get("force_assembly")
    gate = receipt.get("gate")
    if not all(
        isinstance(value, dict)
        for value in (
            qualification,
            coverage,
            thresholds,
            comparisons,
            decomposition,
            force_assembly,
            gate,
        )
    ):
        raise ImportError(
            "static-dynamic handoff receipt is missing coverage, thresholds, "
            "comparisons, decomposition, force assembly, gate, or qualification"
        )

    threshold_evidence = all(
        _finite_nonnegative(thresholds.get(name))
        and thresholds[name] <= ceiling
        for name, ceiling in _HANDOFF_THRESHOLD_CEILINGS.items()
    )
    comparison_evidence = (
        set(comparisons) == set(_HANDOFF_COUNTS)
        and all(
            _handoff_comparison(comparisons.get(name), count)
            for name, count in _HANDOFF_COUNTS.items()
        )
    )
    decomposition_evidence = _handoff_comparison(decomposition, 416)
    assembly_evidence = (
        force_assembly.get("gravity_convention") == HANDOFF_GRAVITY_CONVENTION
        and _handoff_comparison(force_assembly.get("static"), 128)
        and _handoff_comparison(force_assembly.get("dynamic"), 128)
    )
    force_owners = coverage.get("force_owners")
    coverage_evidence = (
        coverage.get("muscles") == 416
        and coverage.get("generalized_coordinates") == 128
        and coverage.get("static_muscle_state") is True
        and coverage.get("static_generalized_forces") is True
        and coverage.get("static_zero_activation_force_diagnostic") is True
        and coverage.get("passive_bias_policy") == HANDOFF_PASSIVE_BIAS_POLICY
        and coverage.get("gravity_convention") == HANDOFF_GRAVITY_CONVENTION
        and coverage.get("accepted_fiber_state_source")
            == HANDOFF_ACCEPTED_FIBER_STATE_SOURCE
        and force_owners == list(_HANDOFF_FORCE_OWNERS)
        and coverage.get("dynamic_pre_step_state") is True
        and isinstance(coverage.get("dynamic_state_owner"), str)
        and bool(coverage["dynamic_state_owner"])
        and isinstance(coverage.get("dynamic_force_owner"), str)
        and bool(coverage["dynamic_force_owner"])
    )
    qualification_evidence = (
        qualification.get("pre_step_snapshot_present") is True
        and qualification.get("accepted_fiber_state_transport") is True
        and qualification.get("activation_and_fiber_state_parity") is True
        and qualification.get("per_muscle_force_parity") is True
        and qualification.get("source_force_decomposition_closed") is True
        and qualification.get("full_force_owner_parity") is True
        and qualification.get("static_force_assembly_closed") is True
        and qualification.get("dynamic_force_assembly_closed") is True
        and qualification.get("generalized_force_parity") is True
        and qualification.get("fiber_tendon_equilibrium_closed") is True
    )
    maximum_equilibrium_residual = receipt.get(
        "maximum_damped_equilibrium_residual"
    )
    worst_equilibrium_muscle = receipt.get("worst_damped_equilibrium_muscle")
    equilibrium_evidence = (
        _finite_nonnegative(maximum_equilibrium_residual)
        and maximum_equilibrium_residual
            <= thresholds.get("maximum_damped_equilibrium_residual", -1.0)
        and type(worst_equilibrium_muscle) is int
        and 0 <= worst_equilibrium_muscle < 416
    )
    evidence = (
        receipt.get("status") == "passed"
        and coverage_evidence
        and threshold_evidence
        and comparison_evidence
        and decomposition_evidence
        and assembly_evidence
        and equilibrium_evidence
        and qualification_evidence
        and gate.get("reasons") == []
    )
    claimed = qualification.get("static_dynamic_handoff_parity") is True
    if claimed and not evidence:
        raise ImportError(
            "static-dynamic handoff receipt claims parity without complete supporting evidence"
        )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "complete": claimed and evidence,
        "passive_bias_policy": coverage.get("passive_bias_policy"),
        "gravity_convention": coverage.get("gravity_convention"),
        "accepted_fiber_state_source": coverage.get(
            "accepted_fiber_state_source"
        ),
        "force_owners": force_owners,
        "pre_step_snapshot_present": (
            qualification.get("pre_step_snapshot_present") is True
        ),
        "accepted_fiber_state_transport": (
            qualification.get("accepted_fiber_state_transport") is True
        ),
        "activation_and_fiber_state_parity": (
            qualification.get("activation_and_fiber_state_parity") is True
        ),
        "per_muscle_force_parity": (
            qualification.get("per_muscle_force_parity") is True
        ),
        "source_force_decomposition_closed": (
            qualification.get("source_force_decomposition_closed") is True
        ),
        "full_force_owner_parity": (
            qualification.get("full_force_owner_parity") is True
        ),
        "static_force_assembly_closed": (
            qualification.get("static_force_assembly_closed") is True
        ),
        "dynamic_force_assembly_closed": (
            qualification.get("dynamic_force_assembly_closed") is True
        ),
        "generalized_force_parity": (
            qualification.get("generalized_force_parity") is True
        ),
        "fiber_tendon_equilibrium_closed": (
            qualification.get("fiber_tendon_equilibrium_closed") is True
        ),
        "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
        "worst_damped_equilibrium_muscle": worst_equilibrium_muscle,
        "comparisons": comparisons,
        "source_force_decomposition": decomposition,
        "force_assembly": force_assembly,
    }


def audit(arguments: argparse.Namespace) -> int:
    stdout = arguments.stdout.resolve()
    metrics = _metrics(_native_metrics(stdout))
    thresholds = {
        "clock_seconds": 1.25e-5,
        "clock_tolerance_seconds": 1.0e-12,
        "minimum_steps": arguments.minimum_steps,
        "maximum_acceleration": arguments.maximum_acceleration,
        "maximum_velocity_delta": arguments.maximum_velocity_delta,
        "maximum_configuration_delta": arguments.maximum_configuration_delta,
    }
    reasons: list[str] = []
    clock_exact = (
        abs(metrics["timestep_seconds"] - thresholds["clock_seconds"])
        <= thresholds["clock_tolerance_seconds"]
    )
    if not clock_exact:
        reasons.append("the native timestep is not the required 12.5 us clock")
    horizon_complete = (
        metrics["persistent_completed_steps"] >= thresholds["minimum_steps"]
    )
    if not horizon_complete:
        reasons.append("the requested long horizon did not complete")
    static_balance = metrics["compiled_stand_balanced"]
    if not static_balance:
        reasons.append("compiled_stand_balanced=false")
    temporal = (
        metrics["persistent_max_acceleration"]
            <= thresholds["maximum_acceleration"]
        and metrics["muscle_step_max_velocity_delta"]
            <= thresholds["maximum_velocity_delta"]
        and metrics["muscle_step_max_configuration_delta"]
            <= thresholds["maximum_configuration_delta"]
    )
    if not temporal:
        reasons.append("temporal drift exceeds the declared engineering bounds")
    replay = _replay_summary(
        arguments.replay_stdout.resolve() if arguments.replay_stdout else None,
        metrics,
    )
    if (
        arguments.require_same_horizon_replay
        and replay["same_horizon"] != "bitwise"
    ):
        reasons.append("same-horizon deterministic replay was not supplied")

    standing_state = _standing_state(arguments.standing_state_receipt)
    standing_state_admissible = bool(
        standing_state
        and standing_state["candidate"]
        and standing_state["equilibrium_state_transport"]
    )
    if standing_state and standing_state["uniform_maximal_activation"]:
        reasons.append(
            "the supplied standing state is a uniform maximal-activation diagnostic"
        )
    if (
        standing_state
        and standing_state["candidate"]
        and not standing_state["equilibrium_state_transport"]
    ):
        reasons.append(
            "the supplied standing state is not bound to the solved coupled equilibrium"
        )
    if arguments.require_standing_state_receipt and standing_state is None:
        reasons.append("a prepared standing-state receipt is required")
    if (
        arguments.require_standing_state_receipt
        and not standing_state_admissible
    ):
        reasons.append("the prepared standing state is not equilibrium-admissible")

    force_ledger = _force_ledger(arguments.force_ledger_receipt)
    force_ledger_admissible = bool(force_ledger and force_ledger["complete"])
    if arguments.require_force_ledger_receipt and force_ledger is None:
        reasons.append("a complete generalized-force ledger is required")
    if (
        arguments.require_force_ledger_receipt
        and not force_ledger_admissible
    ):
        reasons.append(
            "the generalized-force ledger is incomplete or not force-closed"
        )

    handoff = _static_dynamic_handoff(
        arguments.static_dynamic_handoff_receipt
    )
    handoff_admissible = bool(handoff and handoff["complete"])
    if handoff is not None and not handoff_admissible:
        reasons.append(
            "the accepted static state is not proved identical to the complete "
            "persistent dynamic pre-step force state"
        )
    if (
        arguments.require_static_dynamic_handoff_receipt
        and handoff is None
    ):
        reasons.append("a complete static-dynamic handoff receipt is required")
    if (
        arguments.require_static_dynamic_handoff_receipt
        and not handoff_admissible
    ):
        reasons.append(
            "the static-dynamic handoff lacks accepted fibre transport, complete "
            "force-owner parity, or closed residual assembly"
        )

    exact_eligible = (
        arguments.body_count <= arguments.exact_body_limit
        and arguments.dof_count <= arguments.exact_dof_limit
        and arguments.q_count <= arguments.exact_q_limit
    )
    mechanical_force_convergence = (
        clock_exact and horizon_complete and static_balance and temporal
    )
    force_convergence = mechanical_force_convergence
    if arguments.require_standing_state_receipt:
        force_convergence = force_convergence and standing_state_admissible
    if arguments.require_force_ledger_receipt:
        force_convergence = force_convergence and force_ledger_admissible
    if arguments.require_static_dynamic_handoff_receipt:
        force_convergence = force_convergence and handoff_admissible
    standing_force_convergence = (
        mechanical_force_convergence
        and standing_state_admissible
        and force_ledger_admissible
        and handoff_admissible
    )

    receipt = {
        "schema": SCHEMA,
        "status": "passed" if force_convergence and not reasons else "partial",
        "source": {
            "commit": arguments.source_commit,
            "subject": arguments.subject,
            "body_count": arguments.body_count,
            "dof_count": arguments.dof_count,
            "q_count": arguments.q_count,
        },
        "binary": {
            "sha256": arguments.binary_sha256,
            "device": metrics["device"],
        },
        "horizon": metrics,
        "replay": replay,
        "standing_state": standing_state,
        "force_ledger": force_ledger,
        "static_dynamic_handoff": handoff,
        "exact_dense_stage": {
            "eligible": exact_eligible,
            "body_limit": arguments.exact_body_limit,
            "dof_limit": arguments.exact_dof_limit,
            "q_limit": arguments.exact_q_limit,
            "selected_path": (
                "exact_dense" if exact_eligible else "large_state_fallback"
            ),
        },
        "qualification": {
            "clock_exact": clock_exact,
            "horizon_complete": horizon_complete,
            "static_balance": static_balance,
            "temporal_convergence": temporal,
            "mechanical_force_convergence": mechanical_force_convergence,
            "standing_state_admissible": standing_state_admissible,
            "force_ledger_admissible": force_ledger_admissible,
            "static_dynamic_handoff_admissible": handoff_admissible,
            "force_convergence": force_convergence,
            "standing_force_convergence": standing_force_convergence,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
            "anatomical_supports_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "materials_resolved": False,
            "subject_calibration": False,
        },
        "gate": {
            "thresholds": thresholds,
            "require_standing_state_receipt": (
                arguments.require_standing_state_receipt
            ),
            "require_force_ledger_receipt": (
                arguments.require_force_ledger_receipt
            ),
            "require_static_dynamic_handoff_receipt": (
                arguments.require_static_dynamic_handoff_receipt
            ),
            "reasons": reasons,
        },
        "artifacts": {
            "stdout": {"path": str(stdout), "sha256": sha256(stdout)},
            "stderr": (
                {
                    "path": str(arguments.stderr.resolve()),
                    "sha256": sha256(arguments.stderr.resolve()),
                }
                if arguments.stderr
                else None
            ),
            "build_log": (
                {
                    "path": str(arguments.build_log.resolve()),
                    "sha256": sha256(arguments.build_log.resolve()),
                }
                if arguments.build_log
                else None
            ),
        },
    }
    write_json(arguments.output.resolve(), receipt)
    print(f"wrote {arguments.output.resolve()}")
    print(
        f"force_convergence={str(force_convergence).lower()} "
        f"status={receipt['status']}"
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--replay-stdout", type=Path)
    parser.add_argument("--build-log", type=Path)
    parser.add_argument("--standing-state-receipt", type=Path)
    parser.add_argument("--require-standing-state-receipt", action="store_true")
    parser.add_argument("--force-ledger-receipt", type=Path)
    parser.add_argument("--require-force-ledger-receipt", action="store_true")
    parser.add_argument("--static-dynamic-handoff-receipt", type=Path)
    parser.add_argument(
        "--require-static-dynamic-handoff-receipt", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--subject", default="one adult male source package")
    parser.add_argument("--body-count", type=int, default=157)
    parser.add_argument("--dof-count", type=int, default=128)
    parser.add_argument("--q-count", type=int, default=129)
    parser.add_argument("--exact-body-limit", type=int, default=32)
    parser.add_argument("--exact-dof-limit", type=int, default=40)
    parser.add_argument("--exact-q-limit", type=int, default=41)
    parser.add_argument("--minimum-steps", type=int, default=512)
    parser.add_argument("--maximum-acceleration", type=float, default=1000.0)
    parser.add_argument("--maximum-velocity-delta", type=float, default=0.01)
    parser.add_argument(
        "--maximum-configuration-delta", type=float, default=1.0e-4
    )
    parser.add_argument("--require-same-horizon-replay", action="store_true")
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit native exact-clock Human force convergence"
    )
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

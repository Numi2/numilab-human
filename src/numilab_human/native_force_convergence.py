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


def _native_metrics(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ImportError(f"cannot read native stdout {path}: {error}") from error
    line = next(
        (
            item
            for item in reversed(lines)
            if item.startswith("myosim_articulated_marker_visual=")
        ),
        None,
    )
    if line is None:
        raise ImportError(f"{path} has no myosim_articulated_marker_visual result line")
    values: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    if values.get("myosim_articulated_marker_visual") != "ok":
        raise ImportError(f"{path} does not contain a successful native result")
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
    result = {
        "device": _field(values, "metal_pose_device"),
        "core_bodies": _integer(values, "core_bodies"),
        "muscle_step_count": _integer(values, "muscle_step_count"),
        "timestep_seconds": _number(values, "muscle_step_seconds"),
        "persistent_completed_steps": _integer(values, "persistent_completed_steps"),
        "persistent_max_acceleration": _number(values, "persistent_max_acceleration"),
        "persistent_max_penetration_m": _number(values, "persistent_max_penetration_m"),
        "muscle_step_max_velocity_delta": _number(values, "muscle_step_max_velocity_delta"),
        "muscle_step_max_configuration_delta": _number(values, "muscle_step_max_configuration_delta"),
        "compiled_stand_balanced": _boolean(values, "compiled_stand_balanced"),
        "compiled_stand_max_root_force_residual_n": _number(
            values, "compiled_stand_max_root_force_residual"
        ),
        "compiled_stand_support_contacts": _integer(values, "compiled_stand_support_contacts"),
        "compiled_stand_active_support_contacts": _integer(
            values, "compiled_stand_active_support_contacts"
        ),
        "compiled_stand_total_support_force_n": _number(
            values, "compiled_stand_total_support_force_n"
        ),
        "source_dynamic_force_parity_max_delta_n": _number(
            values, "source_dynamic_force_parity_max_delta_n"
        ),
        "muscle_force_metal_elapsed_ms": _number(values, "muscle_force_metal_elapsed_ms"),
        "stand_deterministic_replay": _field(values, "stand_deterministic_replay"),
    }
    if "compiled_stand_normalized_residual_rms" in values:
        result["compiled_stand_normalized_residual_rms"] = _number(
            values, "compiled_stand_normalized_residual_rms"
        )
    if "compiled_stand_max_activation" in values:
        result["compiled_stand_max_activation"] = _number(values, "compiled_stand_max_activation")
    return result


def _replay_summary(path: Path | None, main: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"same_horizon": "not_supplied", "one_step_bitwise": False}
    replay_values = _native_metrics(path)
    replay = _metrics(replay_values)
    same_horizon = replay["persistent_completed_steps"] == main["persistent_completed_steps"]
    return {
        "same_horizon": "bitwise" if same_horizon and replay["stand_deterministic_replay"] == "bitwise" else "not_proved",
        "one_step_bitwise": replay["stand_deterministic_replay"] == "bitwise",
        "steps": replay["persistent_completed_steps"],
        "stdout_sha256": sha256(path),
    }


def _standing_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read standing-state receipt {resolved}: {error}") from error
    if receipt.get("schema") != STANDING_STATE_SCHEMA:
        raise ImportError("standing-state receipt schema mismatch")
    qualification = receipt.get("qualification")
    state = receipt.get("state")
    if not isinstance(qualification, dict) or not isinstance(state, dict):
        raise ImportError("standing-state receipt is missing qualification/state")
    candidate = qualification.get("standing_initial_state_candidate") is True
    uniform_maximal = state.get("uniform_maximal_activation")
    if type(uniform_maximal) is not bool:
        raise ImportError("standing-state receipt has no boolean maximal-activation classification")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "candidate": candidate,
        "uniform_maximal_activation": uniform_maximal,
        "activation_nonzero_count": state.get("activation_nonzero_count"),
        "initial_state_sha256": (receipt.get("initial_state") or {}).get("sha256"),
    }


def audit(arguments: argparse.Namespace) -> int:
    stdout = arguments.stdout.resolve()
    main_values = _native_metrics(stdout)
    metrics = _metrics(main_values)
    thresholds = {
        "clock_seconds": 1.25e-5,
        "clock_tolerance_seconds": 1.0e-12,
        "minimum_steps": arguments.minimum_steps,
        "maximum_acceleration": arguments.maximum_acceleration,
        "maximum_velocity_delta": arguments.maximum_velocity_delta,
        "maximum_configuration_delta": arguments.maximum_configuration_delta,
    }
    reasons: list[str] = []
    clock_exact = abs(metrics["timestep_seconds"] - thresholds["clock_seconds"]) <= thresholds["clock_tolerance_seconds"]
    if not clock_exact:
        reasons.append("the native timestep is not the required 12.5 us clock")
    complete = metrics["persistent_completed_steps"] >= thresholds["minimum_steps"]
    if not complete:
        reasons.append("the requested long horizon did not complete")
    static_balance = metrics["compiled_stand_balanced"]
    if not static_balance:
        reasons.append("compiled_stand_balanced=false")
    temporal = (
        metrics["persistent_max_acceleration"] <= thresholds["maximum_acceleration"]
        and metrics["muscle_step_max_velocity_delta"] <= thresholds["maximum_velocity_delta"]
        and metrics["muscle_step_max_configuration_delta"] <= thresholds["maximum_configuration_delta"]
    )
    if not temporal:
        reasons.append("temporal drift exceeds the declared engineering bounds")
    replay = _replay_summary(arguments.replay_stdout.resolve() if arguments.replay_stdout else None, metrics)
    if arguments.require_same_horizon_replay and replay["same_horizon"] != "bitwise":
        reasons.append("same-horizon deterministic replay was not supplied")

    standing_state = _standing_state(arguments.standing_state_receipt)
    standing_state_admissible = bool(standing_state and standing_state["candidate"])
    if standing_state and standing_state["uniform_maximal_activation"]:
        reasons.append("the supplied standing state is a uniform maximal-activation diagnostic")
    if arguments.require_standing_state_receipt and standing_state is None:
        reasons.append("a prepared standing-state receipt is required")
    if arguments.require_standing_state_receipt and not standing_state_admissible:
        reasons.append("the prepared standing state is not admissible")

    exact_eligible = (
        arguments.body_count <= arguments.exact_body_limit
        and arguments.dof_count <= arguments.exact_dof_limit
        and arguments.q_count <= arguments.exact_q_limit
    )
    force_convergence = clock_exact and complete and static_balance and temporal
    standing_force_convergence = force_convergence and standing_state_admissible
    if arguments.require_standing_state_receipt:
        force_convergence = force_convergence and standing_state_admissible

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
        "exact_dense_stage": {
            "eligible": exact_eligible,
            "body_limit": arguments.exact_body_limit,
            "dof_limit": arguments.exact_dof_limit,
            "q_limit": arguments.exact_q_limit,
            "selected_path": "exact_dense" if exact_eligible else "large_state_fallback",
        },
        "qualification": {
            "clock_exact": clock_exact,
            "horizon_complete": complete,
            "static_balance": static_balance,
            "temporal_convergence": temporal,
            "standing_state_admissible": standing_state_admissible,
            "force_convergence": force_convergence,
            "standing_force_convergence": standing_force_convergence,
            # A converged force horizon is necessary for standing, but it is
            # not a standing-behavior qualification. The latter needs the
            # separate accepted-root support/contact and behavior protocol.
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
            "require_standing_state_receipt": arguments.require_standing_state_receipt,
            "reasons": reasons,
        },
        "artifacts": {
            "stdout": {"path": str(stdout), "sha256": sha256(stdout)},
            "stderr": (
                {"path": str(arguments.stderr.resolve()), "sha256": sha256(arguments.stderr.resolve())}
                if arguments.stderr
                else None
            ),
            "build_log": (
                {"path": str(arguments.build_log.resolve()), "sha256": sha256(arguments.build_log.resolve())}
                if arguments.build_log
                else None
            ),
        },
    }
    write_json(arguments.output.resolve(), receipt)
    print(f"wrote {arguments.output.resolve()}")
    print(f"force_convergence={str(force_convergence).lower()} status={receipt['status']}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--replay-stdout", type=Path)
    parser.add_argument("--build-log", type=Path)
    parser.add_argument("--standing-state-receipt", type=Path)
    parser.add_argument("--require-standing-state-receipt", action="store_true")
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
    parser.add_argument("--maximum-configuration-delta", type=float, default=1.0e-4)
    parser.add_argument("--require-same-horizon-replay", action="store_true")
    parser.set_defaults(handler=audit)

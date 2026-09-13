from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


SCHEMA = "numi.human.static-dynamic-handoff-audit.v1"
STATIC_MUSCLE_SCHEMA = "numi.human.offline-muscle-state.v1"
STATIC_REACTION_SCHEMA = "numi.human.offline-reactions.v1"
DYNAMIC_SCHEMA = "numi.human.dynamic-handoff.v1"
STATIC_MUSCLE_PREFIX = "compiled_equilibrium_muscles="
STATIC_REACTION_PREFIX = "compiled_equilibrium_reactions="
DYNAMIC_PREFIX = "compiled_dynamic_handoff="
MUSCLE_COUNT = 416
NV = 128


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ImportError(f"cannot read native Human log {path}: {error}") from error
    if path.suffix == ".gz":
        try:
            raw = gzip.decompress(raw)
        except OSError as error:
            raise ImportError(f"cannot decompress native Human log {path}: {error}") from error
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImportError(f"native Human log is not UTF-8: {path}") from error


def _record(text: str, prefix: str, *, required: bool) -> dict[str, Any] | None:
    rows = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if not rows:
        if required:
            raise ImportError(f"native log has no {prefix[:-1]} record")
        return None
    _require(len(rows) == 1, f"native log must contain exactly one {prefix[:-1]} record")
    try:
        payload = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise ImportError(f"{prefix[:-1]} is not valid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{prefix[:-1]} must be an object")
    return payload


def _snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read dynamic handoff snapshot {path}: {error}") from error
    _require(isinstance(payload, dict), "dynamic handoff snapshot must be an object")
    return payload


def _vector(payload: dict[str, Any], key: str, count: int) -> list[float]:
    value = payload.get(key)
    _require(isinstance(value, list) and len(value) == count, f"{key} must contain {count} values")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(type(item) in (int, float) and math.isfinite(item), f"{key}[{index}] is not finite")
        result.append(float(item))
    return result


def _static_state(text: str) -> dict[str, list[float]]:
    muscles = _record(text, STATIC_MUSCLE_PREFIX, required=True)
    reactions = _record(text, STATIC_REACTION_PREFIX, required=True)
    assert muscles is not None and reactions is not None
    _require(muscles.get("schema") == STATIC_MUSCLE_SCHEMA, "static muscle-state schema mismatch")
    _require(reactions.get("schema") == STATIC_REACTION_SCHEMA, "static reaction schema mismatch")
    return {
        "activation_fp32": _vector(muscles, "activation_fp32", MUSCLE_COUNT),
        "fiber_length_m": _vector(muscles, "reference_fiber_length_m", MUSCLE_COUNT),
        "actuator_force_n": _vector(muscles, "actuator_force_n", MUSCLE_COUNT),
        "passive_actuator_force_n": _vector(muscles, "passive_actuator_force_n", MUSCLE_COUNT),
        "generalized_muscle_force": _vector(reactions, "muscle_force", NV),
        "generalized_passive_force": _vector(reactions, "passive_force", NV),
        "force_residual": _vector(reactions, "force_residual", NV),
    }


def _dynamic_state(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == DYNAMIC_SCHEMA, "dynamic handoff schema mismatch")
    _require(payload.get("stage") == "pre_step", "dynamic handoff must describe the pre_step stage")
    completed_steps = payload.get("completed_steps")
    _require(type(completed_steps) is int and completed_steps == 0,
             "dynamic handoff must be captured before completing a persistent step")
    state_owner = payload.get("state_owner")
    force_owner = payload.get("force_owner")
    _require(isinstance(state_owner, str) and state_owner, "dynamic handoff has no state_owner")
    _require(isinstance(force_owner, str) and force_owner, "dynamic handoff has no force_owner")
    return {
        "activation_fp32": _vector(payload, "activation_fp32", MUSCLE_COUNT),
        "fiber_length_m": _vector(payload, "fiber_length_m", MUSCLE_COUNT),
        "actuator_force_n": _vector(payload, "actuator_force_n", MUSCLE_COUNT),
        "passive_actuator_force_n": _vector(payload, "passive_actuator_force_n", MUSCLE_COUNT),
        "damped_equilibrium_residual": _vector(payload, "damped_equilibrium_residual", MUSCLE_COUNT),
        "generalized_muscle_force": _vector(payload, "generalized_muscle_force", NV),
        "generalized_passive_force": _vector(payload, "generalized_passive_force", NV),
        "force_residual": _vector(payload, "force_residual", NV),
        "state_owner": state_owner,
        "force_owner": force_owner,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }


def _comparison(
    reference: list[float], candidate: list[float], *, absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    _require(len(reference) == len(candidate) and reference,
             "parity vectors must be non-empty and have identical dimensions")
    rows: list[tuple[float, float, int, float, float]] = []
    squared = 0.0
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        delta = abs(actual - expected)
        tolerance = absolute_tolerance + relative_tolerance * max(abs(expected), abs(actual))
        normalized = delta / tolerance if tolerance > 0.0 else (0.0 if delta == 0.0 else math.inf)
        rows.append((normalized, delta, index, expected, actual))
        squared += delta * delta
    worst = max(rows, key=lambda row: (row[0], row[1], -row[2]))
    maximum_normalized = worst[0]
    return {
        "count": len(reference),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_delta": max(row[1] for row in rows),
        "rms_absolute_delta": math.sqrt(squared / len(reference)),
        "maximum_normalized_error": maximum_normalized if math.isfinite(maximum_normalized) else None,
        "worst_index": worst[2],
        "worst_reference": worst[3],
        "worst_candidate": worst[4],
        "worst_absolute_delta": worst[1],
        "passed": maximum_normalized <= 1.0,
    }


def _validate_threshold(value: float, name: str) -> float:
    _require(math.isfinite(value) and value >= 0.0, f"{name} must be finite and non-negative")
    return value


def audit(arguments: argparse.Namespace) -> int:
    source = arguments.log.resolve()
    text = _read_text(source)
    static = _static_state(text)
    embedded = _record(text, DYNAMIC_PREFIX, required=False)
    snapshot_path = arguments.dynamic_snapshot.resolve() if arguments.dynamic_snapshot else None
    _require(not (embedded is not None and snapshot_path is not None),
             "dynamic handoff is ambiguous: both the native log and --dynamic-snapshot provide it")
    dynamic_payload = _snapshot(snapshot_path) if snapshot_path is not None else embedded

    thresholds = {
        "activation_absolute": _validate_threshold(arguments.maximum_activation_delta, "maximum activation delta"),
        "fiber_absolute_m": _validate_threshold(arguments.fiber_absolute_tolerance, "fiber absolute tolerance"),
        "fiber_relative": _validate_threshold(arguments.fiber_relative_tolerance, "fiber relative tolerance"),
        "force_absolute_n": _validate_threshold(arguments.force_absolute_tolerance, "force absolute tolerance"),
        "force_relative": _validate_threshold(arguments.force_relative_tolerance, "force relative tolerance"),
        "residual_absolute": _validate_threshold(arguments.residual_absolute_tolerance, "residual absolute tolerance"),
        "maximum_damped_equilibrium_residual": _validate_threshold(
            arguments.maximum_damped_equilibrium_residual,
            "maximum damped equilibrium residual",
        ),
    }

    reasons: list[str] = []
    comparisons: dict[str, Any] | None = None
    dynamic: dict[str, Any] | None = None
    maximum_equilibrium_residual: float | None = None
    if dynamic_payload is None:
        reasons.append("the native owner did not publish a pre-step dynamic handoff snapshot")
    else:
        dynamic = _dynamic_state(dynamic_payload)
        comparisons = {
            "activation": _comparison(
                static["activation_fp32"], dynamic["activation_fp32"],
                absolute_tolerance=thresholds["activation_absolute"], relative_tolerance=0.0,
            ),
            "fiber_length": _comparison(
                static["fiber_length_m"], dynamic["fiber_length_m"],
                absolute_tolerance=thresholds["fiber_absolute_m"],
                relative_tolerance=thresholds["fiber_relative"],
            ),
            "actuator_force": _comparison(
                static["actuator_force_n"], dynamic["actuator_force_n"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "passive_actuator_force": _comparison(
                static["passive_actuator_force_n"], dynamic["passive_actuator_force_n"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_muscle_force": _comparison(
                static["generalized_muscle_force"], dynamic["generalized_muscle_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_passive_force": _comparison(
                static["generalized_passive_force"], dynamic["generalized_passive_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "force_residual": _comparison(
                static["force_residual"], dynamic["force_residual"],
                absolute_tolerance=thresholds["residual_absolute"],
                relative_tolerance=thresholds["force_relative"],
            ),
        }
        for name, result in comparisons.items():
            if not result["passed"]:
                reasons.append(f"{name} changes across the static-to-dynamic handoff")
        maximum_equilibrium_residual = max(abs(value) for value in dynamic["damped_equilibrium_residual"])
        if maximum_equilibrium_residual > thresholds["maximum_damped_equilibrium_residual"]:
            reasons.append("one or more dynamic muscle states violate damped fibre/tendon equilibrium")

    state_parity = bool(comparisons and comparisons["activation"]["passed"] and comparisons["fiber_length"]["passed"])
    muscle_force_parity = bool(
        comparisons
        and comparisons["actuator_force"]["passed"]
        and comparisons["passive_actuator_force"]["passed"]
    )
    generalized_force_parity = bool(
        comparisons
        and comparisons["generalized_muscle_force"]["passed"]
        and comparisons["generalized_passive_force"]["passed"]
        and comparisons["force_residual"]["passed"]
    )
    equilibrium_closed = bool(
        maximum_equilibrium_residual is not None
        and maximum_equilibrium_residual <= thresholds["maximum_damped_equilibrium_residual"]
    )
    complete = state_parity and muscle_force_parity and generalized_force_parity and equilibrium_closed

    receipt = {
        "schema": SCHEMA,
        "status": "passed" if complete else "partial",
        "input": {
            "native_log": {"path": str(source), "sha256": sha256(source)},
            "dynamic_snapshot": (
                {"path": str(snapshot_path), "sha256": sha256(snapshot_path)}
                if snapshot_path is not None else None
            ),
        },
        "coverage": {
            "muscles": MUSCLE_COUNT,
            "generalized_coordinates": NV,
            "static_muscle_state": True,
            "static_generalized_forces": True,
            "dynamic_pre_step_state": dynamic is not None,
            "dynamic_state_owner": dynamic["state_owner"] if dynamic else None,
            "dynamic_force_owner": dynamic["force_owner"] if dynamic else None,
        },
        "thresholds": thresholds,
        "comparisons": comparisons,
        "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
        "qualification": {
            "pre_step_snapshot_present": dynamic is not None,
            "activation_and_fiber_state_parity": state_parity,
            "per_muscle_force_parity": muscle_force_parity,
            "generalized_force_parity": generalized_force_parity,
            "fiber_tendon_equilibrium_closed": equilibrium_closed,
            "static_dynamic_handoff_parity": complete,
            "sustained_standing": False,
            "walking": False,
        },
        "gate": {"reasons": reasons},
        "metadata": dynamic["metadata"] if dynamic else {},
    }
    write_json(arguments.output.resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "dynamic_snapshot": dynamic is not None,
        "handoff_parity": complete,
        "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", type=Path, required=True,
                        help="native Human stdout/log with compiled equilibrium records")
    parser.add_argument("--dynamic-snapshot", type=Path,
                        help="optional numi.human.dynamic-handoff.v1 JSON when not embedded in the log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-activation-delta", type=float, default=1.0e-7)
    parser.add_argument("--fiber-absolute-tolerance", type=float, default=5.0e-7)
    parser.add_argument("--fiber-relative-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--force-absolute-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--force-relative-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--residual-absolute-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--maximum-damped-equilibrium-residual", type=float, default=1.0e-5)
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit all 416 muscle states and 128 generalized forces across the Human pre-step handoff"
    )
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

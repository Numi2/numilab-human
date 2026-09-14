from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


SCHEMA = "numi.human.static-dynamic-handoff-audit.v3"
STATIC_MUSCLE_SCHEMA = "numi.human.offline-muscle-state.v1"
STATIC_REACTION_SCHEMA = "numi.human.offline-reactions.v1"
DYNAMIC_SCHEMA = "numi.human.dynamic-handoff.v3"
STATIC_MUSCLE_PREFIX = "compiled_equilibrium_muscles="
STATIC_REACTION_PREFIX = "compiled_equilibrium_reactions="
DYNAMIC_PREFIX = "compiled_dynamic_handoff="
MUSCLE_COUNT = 416
NV = 128
PASSIVE_BIAS_POLICY = (
    "legacy_zero_activation_bias_excluded_compliant_tendon_force_retained"
)
GRAVITY_CONVENTION = (
    "force_residual=muscle+equality+limit+support+passive-gravity_target"
)
ACCEPTED_FIBER_STATE_SOURCE = "compiled_equilibrium_reference_fiber_length"
GENERALIZED_FORCE_KEYS = (
    "generalized_muscle_force",
    "generalized_joint_equality_force",
    "generalized_position_limit_force",
    "generalized_support_force",
    "generalized_passive_force",
    "gravity_target",
    "force_residual",
)


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
        _require(
            type(item) in (int, float) and math.isfinite(item),
            f"{key}[{index}] is not finite",
        )
        result.append(float(item))
    return result


def _static_state(text: str) -> dict[str, Any]:
    muscles = _record(text, STATIC_MUSCLE_PREFIX, required=True)
    reactions = _record(text, STATIC_REACTION_PREFIX, required=True)
    assert muscles is not None and reactions is not None
    _require(muscles.get("schema") == STATIC_MUSCLE_SCHEMA, "static muscle-state schema mismatch")
    _require(reactions.get("schema") == STATIC_REACTION_SCHEMA, "static reaction schema mismatch")

    source_total = _vector(muscles, "actuator_force_n", MUSCLE_COUNT)
    zero_activation = _vector(muscles, "passive_actuator_force_n", MUSCLE_COUNT)
    driven = _vector(muscles, "driven_actuator_force_n", MUSCLE_COUNT)
    return {
        "activation_fp32": _vector(muscles, "activation_fp32", MUSCLE_COUNT),
        "fiber_length_m": _vector(muscles, "reference_fiber_length_m", MUSCLE_COUNT),
        "source_total_actuator_force_n": source_total,
        "source_zero_activation_force_n": zero_activation,
        "driven_actuator_force_n": driven,
        "excluded_passive_bias_force_n": [
            total - active for total, active in zip(source_total, driven)
        ],
        "generalized_muscle_force": _vector(reactions, "muscle_force", NV),
        "generalized_joint_equality_force": _vector(reactions, "equality_force", NV),
        "generalized_position_limit_force": _vector(reactions, "limit_force", NV),
        "generalized_support_force": _vector(reactions, "support_force", NV),
        "generalized_passive_force": _vector(reactions, "passive_force", NV),
        "gravity_target": _vector(reactions, "gravity_target", NV),
        "force_residual": _vector(reactions, "force_residual", NV),
    }


def _dynamic_state(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == DYNAMIC_SCHEMA, "dynamic handoff schema mismatch")
    _require(payload.get("stage") == "pre_step", "dynamic handoff must describe the pre_step stage")
    _require(
        type(payload.get("completed_steps")) is int and payload.get("completed_steps") == 0,
        "dynamic handoff must be captured before completing a persistent step",
    )
    _require(
        payload.get("passive_bias_policy") == PASSIVE_BIAS_POLICY,
        "dynamic handoff passive-bias policy mismatch",
    )
    _require(
        payload.get("gravity_convention") == GRAVITY_CONVENTION,
        "dynamic handoff gravity convention mismatch",
    )
    _require(
        payload.get("fiber_state_source") == ACCEPTED_FIBER_STATE_SOURCE,
        "dynamic handoff did not transport the accepted static fibre state",
    )
    state_owner = payload.get("state_owner")
    force_owner = payload.get("force_owner")
    _require(isinstance(state_owner, str) and state_owner, "dynamic handoff has no state_owner")
    _require(isinstance(force_owner, str) and force_owner, "dynamic handoff has no force_owner")
    return {
        "activation_fp32": _vector(payload, "activation_fp32", MUSCLE_COUNT),
        "fiber_length_m": _vector(payload, "fiber_length_m", MUSCLE_COUNT),
        "source_total_actuator_force_n": _vector(
            payload, "source_total_actuator_force_n", MUSCLE_COUNT
        ),
        "excluded_passive_bias_force_n": _vector(
            payload, "excluded_passive_bias_force_n", MUSCLE_COUNT
        ),
        "driven_actuator_force_n": _vector(
            payload, "driven_actuator_force_n", MUSCLE_COUNT
        ),
        "damped_equilibrium_residual": _vector(
            payload, "damped_equilibrium_residual", MUSCLE_COUNT
        ),
        "generalized_muscle_force": _vector(payload, "generalized_muscle_force", NV),
        "generalized_joint_equality_force": _vector(
            payload, "generalized_joint_equality_force", NV
        ),
        "generalized_position_limit_force": _vector(
            payload, "generalized_position_limit_force", NV
        ),
        "generalized_support_force": _vector(payload, "generalized_support_force", NV),
        "generalized_passive_force": _vector(payload, "generalized_passive_force", NV),
        "gravity_target": _vector(payload, "gravity_target", NV),
        "force_residual": _vector(payload, "force_residual", NV),
        "fiber_state_source": payload["fiber_state_source"],
        "state_owner": state_owner,
        "force_owner": force_owner,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }


def _comparison(
    reference: list[float],
    candidate: list[float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    _require(
        len(reference) == len(candidate) and reference,
        "parity vectors must be non-empty and have identical dimensions",
    )
    rows: list[tuple[float, float, int, float, float]] = []
    squared = 0.0
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        delta = abs(actual - expected)
        tolerance = absolute_tolerance + relative_tolerance * max(
            abs(expected), abs(actual)
        )
        normalized = (
            delta / tolerance
            if tolerance > 0.0
            else (0.0 if delta == 0.0 else math.inf)
        )
        rows.append((normalized, delta, index, expected, actual))
        squared += delta * delta
    worst = max(rows, key=lambda row: (row[0], row[1], -row[2]))
    return {
        "count": len(reference),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_delta": max(row[1] for row in rows),
        "rms_absolute_delta": math.sqrt(squared / len(reference)),
        "maximum_normalized_error": worst[0] if math.isfinite(worst[0]) else None,
        "worst_index": worst[2],
        "worst_reference": worst[3],
        "worst_candidate": worst[4],
        "worst_absolute_delta": worst[1],
        "passed": worst[0] <= 1.0,
    }


def _reconstruct_force_residual(state: dict[str, Any]) -> list[float]:
    return [
        state["generalized_muscle_force"][index]
        + state["generalized_joint_equality_force"][index]
        + state["generalized_position_limit_force"][index]
        + state["generalized_support_force"][index]
        + state["generalized_passive_force"][index]
        - state["gravity_target"][index]
        for index in range(NV)
    ]


def _validate_threshold(value: float, name: str) -> float:
    _require(
        math.isfinite(value) and value >= 0.0,
        f"{name} must be finite and non-negative",
    )
    return value


def audit(arguments: argparse.Namespace) -> int:
    source = arguments.log.resolve()
    text = _read_text(source)
    static = _static_state(text)
    embedded = _record(text, DYNAMIC_PREFIX, required=False)
    snapshot_path = arguments.dynamic_snapshot.resolve() if arguments.dynamic_snapshot else None
    _require(
        not (embedded is not None and snapshot_path is not None),
        "dynamic handoff is ambiguous: both the native log and --dynamic-snapshot provide it",
    )
    dynamic_payload = _snapshot(snapshot_path) if snapshot_path is not None else embedded
    thresholds = {
        "activation_absolute": _validate_threshold(
            arguments.maximum_activation_delta, "maximum activation delta"
        ),
        "fiber_absolute_m": _validate_threshold(
            arguments.fiber_absolute_tolerance, "fiber absolute tolerance"
        ),
        "fiber_relative": _validate_threshold(
            arguments.fiber_relative_tolerance, "fiber relative tolerance"
        ),
        "force_absolute_n": _validate_threshold(
            arguments.force_absolute_tolerance, "force absolute tolerance"
        ),
        "force_relative": _validate_threshold(
            arguments.force_relative_tolerance, "force relative tolerance"
        ),
        "decomposition_absolute_n": _validate_threshold(
            arguments.decomposition_absolute_tolerance,
            "decomposition absolute tolerance",
        ),
        "residual_absolute": _validate_threshold(
            arguments.residual_absolute_tolerance, "residual absolute tolerance"
        ),
        "assembly_absolute": _validate_threshold(
            arguments.assembly_absolute_tolerance, "assembly absolute tolerance"
        ),
        "maximum_damped_equilibrium_residual": _validate_threshold(
            arguments.maximum_damped_equilibrium_residual,
            "maximum damped equilibrium residual",
        ),
    }

    reasons: list[str] = []
    comparisons: dict[str, Any] | None = None
    dynamic: dict[str, Any] | None = None
    decomposition: dict[str, Any] | None = None
    force_assembly: dict[str, Any] | None = None
    maximum_equilibrium_residual: float | None = None
    worst_equilibrium_muscle: int | None = None
    if dynamic_payload is None:
        reasons.append("the native owner did not publish a pre-step dynamic handoff snapshot")
    else:
        dynamic = _dynamic_state(dynamic_payload)
        comparisons = {
            "activation": _comparison(
                static["activation_fp32"],
                dynamic["activation_fp32"],
                absolute_tolerance=thresholds["activation_absolute"],
                relative_tolerance=0.0,
            ),
            "fiber_length": _comparison(
                static["fiber_length_m"],
                dynamic["fiber_length_m"],
                absolute_tolerance=thresholds["fiber_absolute_m"],
                relative_tolerance=thresholds["fiber_relative"],
            ),
            "source_total_actuator_force": _comparison(
                static["source_total_actuator_force_n"],
                dynamic["source_total_actuator_force_n"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "driven_actuator_force": _comparison(
                static["driven_actuator_force_n"],
                dynamic["driven_actuator_force_n"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "excluded_passive_bias_force": _comparison(
                static["excluded_passive_bias_force_n"],
                dynamic["excluded_passive_bias_force_n"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_muscle_force": _comparison(
                static["generalized_muscle_force"],
                dynamic["generalized_muscle_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_joint_equality_force": _comparison(
                static["generalized_joint_equality_force"],
                dynamic["generalized_joint_equality_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_position_limit_force": _comparison(
                static["generalized_position_limit_force"],
                dynamic["generalized_position_limit_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_support_force": _comparison(
                static["generalized_support_force"],
                dynamic["generalized_support_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "generalized_passive_force": _comparison(
                static["generalized_passive_force"],
                dynamic["generalized_passive_force"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "gravity_target": _comparison(
                static["gravity_target"],
                dynamic["gravity_target"],
                absolute_tolerance=thresholds["force_absolute_n"],
                relative_tolerance=thresholds["force_relative"],
            ),
            "force_residual": _comparison(
                static["force_residual"],
                dynamic["force_residual"],
                absolute_tolerance=thresholds["residual_absolute"],
                relative_tolerance=thresholds["force_relative"],
            ),
        }
        reconstructed_source = [
            active + excluded
            for active, excluded in zip(
                dynamic["driven_actuator_force_n"],
                dynamic["excluded_passive_bias_force_n"],
            )
        ]
        decomposition = _comparison(
            dynamic["source_total_actuator_force_n"],
            reconstructed_source,
            absolute_tolerance=thresholds["decomposition_absolute_n"],
            relative_tolerance=thresholds["force_relative"],
        )
        force_assembly = {
            "static": _comparison(
                static["force_residual"],
                _reconstruct_force_residual(static),
                absolute_tolerance=thresholds["assembly_absolute"],
                relative_tolerance=0.0,
            ),
            "dynamic": _comparison(
                dynamic["force_residual"],
                _reconstruct_force_residual(dynamic),
                absolute_tolerance=thresholds["assembly_absolute"],
                relative_tolerance=0.0,
            ),
            "gravity_convention": GRAVITY_CONVENTION,
        }
        if not decomposition["passed"]:
            reasons.append(
                "dynamic source force does not decompose into driven force plus excluded bias"
            )
        for name, result in comparisons.items():
            if not result["passed"]:
                reasons.append(f"{name} changes across the static-to-dynamic handoff")
        if not force_assembly["static"]["passed"]:
            reasons.append(
                "the static generalized force owners do not reconstruct the authoritative residual"
            )
        if not force_assembly["dynamic"]["passed"]:
            reasons.append(
                "the dynamic generalized force owners do not reconstruct the authoritative residual"
            )
        residual_rows = [
            (abs(value), index)
            for index, value in enumerate(dynamic["damped_equilibrium_residual"])
        ]
        maximum_equilibrium_residual, worst_equilibrium_muscle = max(residual_rows)
        if maximum_equilibrium_residual > thresholds["maximum_damped_equilibrium_residual"]:
            reasons.append(
                "one or more dynamic muscle states violate damped fibre/tendon equilibrium"
            )

    state_parity = bool(
        comparisons
        and comparisons["activation"]["passed"]
        and comparisons["fiber_length"]["passed"]
    )
    accepted_fiber_transport = bool(
        dynamic and dynamic["fiber_state_source"] == ACCEPTED_FIBER_STATE_SOURCE
    )
    muscle_force_parity = bool(
        comparisons
        and comparisons["source_total_actuator_force"]["passed"]
        and comparisons["driven_actuator_force"]["passed"]
        and comparisons["excluded_passive_bias_force"]["passed"]
    )
    full_force_owner_parity = bool(
        comparisons
        and all(comparisons[name]["passed"] for name in GENERALIZED_FORCE_KEYS)
    )
    static_assembly_closed = bool(force_assembly and force_assembly["static"]["passed"])
    dynamic_assembly_closed = bool(force_assembly and force_assembly["dynamic"]["passed"])
    generalized_force_parity = (
        full_force_owner_parity and static_assembly_closed and dynamic_assembly_closed
    )
    decomposition_closed = bool(decomposition and decomposition["passed"])
    equilibrium_closed = bool(
        maximum_equilibrium_residual is not None
        and maximum_equilibrium_residual
        <= thresholds["maximum_damped_equilibrium_residual"]
    )
    complete = (
        state_parity
        and accepted_fiber_transport
        and muscle_force_parity
        and decomposition_closed
        and generalized_force_parity
        and equilibrium_closed
    )
    receipt = {
        "schema": SCHEMA,
        "status": "passed" if complete else "partial",
        "input": {
            "native_log": {"path": str(source), "sha256": sha256(source)},
            "dynamic_snapshot": (
                {"path": str(snapshot_path), "sha256": sha256(snapshot_path)}
                if snapshot_path is not None
                else None
            ),
        },
        "coverage": {
            "muscles": MUSCLE_COUNT,
            "generalized_coordinates": NV,
            "static_muscle_state": True,
            "static_generalized_forces": True,
            "static_zero_activation_force_diagnostic": True,
            "passive_bias_policy": PASSIVE_BIAS_POLICY,
            "gravity_convention": GRAVITY_CONVENTION,
            "accepted_fiber_state_source": ACCEPTED_FIBER_STATE_SOURCE,
            "dynamic_pre_step_state": dynamic is not None,
            "dynamic_state_owner": dynamic["state_owner"] if dynamic else None,
            "dynamic_force_owner": dynamic["force_owner"] if dynamic else None,
            "force_owners": list(GENERALIZED_FORCE_KEYS),
        },
        "thresholds": thresholds,
        "comparisons": comparisons,
        "source_force_decomposition": decomposition,
        "force_assembly": force_assembly,
        "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
        "worst_damped_equilibrium_muscle": worst_equilibrium_muscle,
        "qualification": {
            "pre_step_snapshot_present": dynamic is not None,
            "accepted_fiber_state_transport": accepted_fiber_transport,
            "activation_and_fiber_state_parity": state_parity,
            "per_muscle_force_parity": muscle_force_parity,
            "source_force_decomposition_closed": decomposition_closed,
            "full_force_owner_parity": full_force_owner_parity,
            "static_force_assembly_closed": static_assembly_closed,
            "dynamic_force_assembly_closed": dynamic_assembly_closed,
            "generalized_force_parity": generalized_force_parity,
            "fiber_tendon_equilibrium_closed": equilibrium_closed,
            "static_dynamic_handoff_parity": complete,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "gate": {"reasons": reasons},
        "metadata": {
            **(dynamic["metadata"] if dynamic else {}),
            "static_zero_activation_force_role": "diagnostic_only",
        },
    }
    write_json(arguments.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "dynamic_snapshot": dynamic is not None,
                "handoff_parity": complete,
                "accepted_fiber_state_transport": accepted_fiber_transport,
                "full_force_owner_parity": full_force_owner_parity,
                "static_force_assembly_closed": static_assembly_closed,
                "dynamic_force_assembly_closed": dynamic_assembly_closed,
                "maximum_damped_equilibrium_residual": maximum_equilibrium_residual,
            },
            sort_keys=True,
        )
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="native Human stdout/log with compiled equilibrium records",
    )
    parser.add_argument(
        "--dynamic-snapshot",
        type=Path,
        help="optional numi.human.dynamic-handoff.v3 JSON when not embedded in the log",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-activation-delta", type=float, default=1.0e-7)
    parser.add_argument("--fiber-absolute-tolerance", type=float, default=5.0e-7)
    parser.add_argument("--fiber-relative-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--force-absolute-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--force-relative-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--decomposition-absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--residual-absolute-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--assembly-absolute-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--maximum-damped-equilibrium-residual", type=float, default=1.0e-5)
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all 416 muscle states and every 128-DoF force owner across "
            "the Human pre-step handoff"
        )
    )
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

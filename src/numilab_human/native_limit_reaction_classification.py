from __future__ import annotations

import argparse
import math
import re
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


SCHEMA = "numi.human.native-limit-reaction-classification.v1"
_INVALID_INDEX = 0xFFFFFFFF
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SUMMARY_PREFIXES = (
    "myosim_articulated_mechanics=",
    "myosim_articulated_marker_visual=",
    "myosim_articulated_bone_visual=",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError("native limit reaction classification: " + message)


def _summary(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ImportError(f"cannot read native stdout {path}: {error}") from error
    rows = [line for line in lines if line.startswith(_SUMMARY_PREFIXES)]
    _require(len(rows) == 1, "stdout must contain exactly one supported native summary")
    values: dict[str, str] = {}
    for token in shlex.split(rows[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        _require(key not in values, f"native summary repeats {key}")
        values[key] = value
    result_keys = [key for key in values if key.startswith("myosim_articulated_")]
    _require(len(result_keys) == 1 and values[result_keys[0]] == "ok",
             "native result did not complete successfully")
    return values


def _field(values: dict[str, str], name: str) -> str:
    value = values.get(name)
    _require(value not in {None, ""}, f"native summary is missing {name}")
    return str(value)


def _number(values: dict[str, str], name: str, *, nonnegative: bool = False) -> float:
    try:
        value = float(_field(values, name))
    except ValueError as error:
        raise ImportError(
            f"native limit reaction classification: {name} is not numeric"
        ) from error
    _require(math.isfinite(value), f"{name} is not finite")
    if nonnegative:
        _require(value >= 0.0, f"{name} is negative")
    return value


def _integer(values: dict[str, str], name: str) -> int:
    value = _number(values, name)
    _require(value == int(value), f"{name} is not integral")
    return int(value)


def _class(
    values: dict[str, str],
    *,
    count_name: str,
    maximum_name: str,
    dof_name: str,
) -> dict[str, Any]:
    count = _integer(values, count_name)
    maximum = _number(values, maximum_name, nonnegative=True)
    dof = _integer(values, dof_name)
    _require(0 <= count <= 128, f"{count_name} is outside [0, 128]")
    if count == 0:
        _require(maximum == 0.0, f"{maximum_name} must be zero when the class is empty")
        _require(dof == _INVALID_INDEX,
                 f"{dof_name} must be the invalid index when the class is empty")
    else:
        _require(maximum > 0.0, f"{maximum_name} must be positive for an active class")
        _require(0 <= dof < 128, f"{dof_name} is outside [0, 128)")
    return {
        "active_count": count,
        "maximum_reaction": maximum,
        "maximum_reaction_dof": None if dof == _INVALID_INDEX else dof,
    }


def compile_receipt(
    stdout: Path,
    *,
    source_commit: str | None = None,
    binary_sha256: str | None = None,
) -> dict[str, Any]:
    stdout = Path(stdout).resolve()
    values = _summary(stdout)

    _require(_field(values, "persistent_root_assistance") == "none",
             "root assistance must be disabled")
    _require(_integer(values, "core_bodies") == 157,
             "full 157-body source model was not executed")
    _require(_integer(values, "compiled_stand_recruited_muscles") == 416,
             "all 416 source muscles were not recruited")
    _require(_field(values, "compiled_stand_balanced") == "true",
             "compiled static standing state is not balanced")

    active_total = _integer(values, "compiled_stand_active_limits")
    _require(0 <= active_total <= 128,
             "compiled_stand_active_limits is outside [0, 128]")
    structural = _class(
        values,
        count_name="compiled_stand_active_structural_locks",
        maximum_name="compiled_stand_max_structural_lock_reaction",
        dof_name="compiled_stand_max_structural_lock_reaction_dof",
    )
    finite_range = _class(
        values,
        count_name="compiled_stand_active_finite_range_limits",
        maximum_name="compiled_stand_max_finite_range_limit_reaction",
        dof_name="compiled_stand_max_finite_range_limit_reaction_dof",
    )
    _require(
        structural["active_count"] + finite_range["active_count"] == active_total,
        "classified limit counts do not reconstruct the active total",
    )
    maximum_total = _number(
        values, "compiled_stand_max_limit_reaction", nonnegative=True
    )
    reconstructed_maximum = max(
        structural["maximum_reaction"], finite_range["maximum_reaction"]
    )
    scale = max(1.0, maximum_total, reconstructed_maximum)
    _require(
        abs(maximum_total - reconstructed_maximum) <= 1.0e-12 * scale,
        "classified reaction maxima do not reconstruct the overall maximum",
    )

    if source_commit is not None:
        _require(_COMMIT.fullmatch(source_commit) is not None,
                 "source commit must be a full lowercase Git SHA")
    if binary_sha256 is not None:
        _require(_SHA256.fullmatch(binary_sha256) is not None,
                 "binary SHA-256 must contain 64 lowercase hexadecimal characters")

    return {
        "schema": SCHEMA,
        "status": "partial",
        "source": {
            "stdout_path": str(stdout),
            "stdout_sha256": sha256(stdout),
            "native_commit": source_commit,
            "binary_sha256": binary_sha256,
        },
        "model": {
            "body_count": 157,
            "velocity_coordinate_count": 128,
            "recruited_muscle_count": 416,
            "root_assistance": "none",
            "compiled_static_balance": True,
        },
        "classification": {
            "active_total": active_total,
            "overall_maximum_reaction": maximum_total,
            "structural_locks": structural,
            "finite_range_stops": finite_range,
            "policy": (
                "A structural lock has an authored interval width no greater "
                "than the static position-limit tolerance. A finite-range stop "
                "has a wider authored interval and is a unilateral anatomical "
                "or mechanical boundary, not a fixed coordinate."
            ),
        },
        "qualification": {
            "reaction_classes_measured": True,
            "finite_range_stop_dependence_resolved": False,
            "generalized_force_convergence": False,
            "sustained_standing": False,
            "perturbation_recovery": False,
            "walking": False,
        },
        "boundary": (
            "This receipt classifies the compiled static unilateral reactions. "
            "It does not prove that finite-range reactions are physiologically "
            "valid, that runtime reactions reproduce them, or that standing is "
            "stable. Structural locks must not be penalized as anatomical stop "
            "dependence merely because they carry reaction force."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--binary-sha256")
    parser.add_argument("--output", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify native Human static limit reactions by authored range"
    )
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    receipt = compile_receipt(
        arguments.stdout,
        source_commit=arguments.source_commit,
        binary_sha256=arguments.binary_sha256,
    )
    write_json(arguments.output.resolve(), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

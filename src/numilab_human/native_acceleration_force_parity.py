from __future__ import annotations

import argparse
import math
import re
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


SCHEMA = "numi.human.native-acceleration-force-parity.v1"
# Standing admission limit, independent of the native success flag.
MAXIMUM_STATIC_BALANCE_RESIDUAL = 0.05
EXPECTED_SEMANTICS = (
    "unconstrained_same_operator_compiled_vs_metal_muscle_force"
)
_SUMMARY_PREFIXES = (
    "myosim_articulated_mechanics=",
    "myosim_articulated_marker_visual=",
    "myosim_articulated_bone_visual=",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError("native acceleration force parity: " + message)


def _summary(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ImportError(f"cannot read native stdout {path}: {error}") from error
    rows = [line for line in lines if line.startswith(_SUMMARY_PREFIXES)]
    _require(len(rows) == 1, "stdout must contain exactly one supported native summary")
    fields: dict[str, str] = {}
    for token in shlex.split(rows[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        _require(key not in fields, f"native summary repeats {key}")
        fields[key] = value
    result_keys = [key for key in fields if key.startswith("myosim_articulated_")]
    _require(len(result_keys) == 1 and fields[result_keys[0]] == "ok",
             "native result did not complete successfully")
    return fields


def _field(fields: dict[str, str], name: str) -> str:
    value = fields.get(name)
    _require(value not in {None, ""}, f"native summary is missing {name}")
    return str(value)


def _number(fields: dict[str, str], name: str, *, nonnegative: bool = False) -> float:
    raw = _field(fields, name)
    try:
        value = float(raw)
    except ValueError as error:
        raise ImportError(f"native acceleration force parity: {name} is not numeric") from error
    _require(math.isfinite(value), f"{name} is not finite")
    if nonnegative:
        _require(value >= 0.0, f"{name} is negative")
    return value


def _integer(fields: dict[str, str], name: str) -> int:
    value = _number(fields, name)
    _require(value == int(value), f"{name} is not integral")
    return int(value)


def compile_receipt(
    stdout: Path,
    *,
    source_commit: str | None = None,
    binary_sha256: str | None = None,
) -> dict[str, Any]:
    stdout = Path(stdout).resolve()
    fields = _summary(stdout)

    _require(_field(fields, "persistent_root_assistance") == "none",
             "root assistance must be disabled")
    _require(_integer(fields, "core_bodies") == 157,
             "full 157-body source model was not executed")
    _require(_integer(fields, "compiled_stand_recruited_muscles") == 416,
             "all 416 source muscles were not recruited")
    _require(_field(fields, "compiled_stand_balanced") == "true",
             "compiled static standing state is not balanced")
    static_residual = _number(
        fields, "compiled_stand_normalized_residual_rms", nonnegative=True
    )
    _require(static_residual <= MAXIMUM_STATIC_BALANCE_RESIDUAL,
             "compiled static balance residual exceeds the 0.05 admission limit")

    timestep = _number(fields, "muscle_step_seconds")
    _require(timestep > 0.0, "muscle_step_seconds must be positive")
    force_delta = _number(
        fields, "source_dynamic_force_parity_max_delta_n", nonnegative=True
    )
    velocity_delta = _number(
        fields,
        "source_dynamic_force_parity_max_delta_v_mixed_units",
        nonnegative=True,
    )
    acceleration_delta = _number(
        fields,
        "source_dynamic_force_parity_max_delta_acceleration_mixed_units",
        nonnegative=True,
    )
    dof = _integer(fields, "source_dynamic_force_parity_max_delta_dof")
    _require(0 <= dof < 128, "maximum acceleration-parity DoF is outside [0, 128)")
    semantics = _field(fields, "source_dynamic_force_parity_acceleration_semantics")
    _require(semantics == EXPECTED_SEMANTICS,
             "acceleration-parity semantics do not match the qualified diagnostic")

    scale = max(1.0, acceleration_delta, velocity_delta / timestep)
    _require(abs(acceleration_delta - velocity_delta / timestep) <= 1.0e-10 * scale,
             "reported acceleration delta is inconsistent with velocity delta / timestep")

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
            "compiled_static_residual_rms": static_residual,
            "maximum_static_balance_residual_rms": MAXIMUM_STATIC_BALANCE_RESIDUAL,
        },
        "diagnostic": {
            "timestep_seconds": timestep,
            "maximum_generalized_force_delta_n": force_delta,
            "maximum_velocity_increment_delta_mixed_units": velocity_delta,
            "maximum_acceleration_delta_mixed_units": acceleration_delta,
            "maximum_acceleration_delta_dof": dof,
            "semantics": semantics,
        },
        "qualification": {
            "static_balance_residual_verified": True,
            "same_operator_muscle_force_acceleration_parity_measured": True,
            "complete_dynamic_force_assembly": False,
            "generalized_force_convergence": False,
            "sustained_standing": False,
            "perturbation_recovery": False,
            "walking": False,
        },
        "boundary": (
            "This diagnostic compares compiled and Metal-evaluated muscle-force "
            "vectors through the same unconstrained mass, gravity and damping "
            "operator. It excludes contact, joint equalities, joint limits and "
            "their impulses. It therefore cannot independently qualify complete "
            "force convergence, standing, recovery or walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--binary-sha256")
    parser.add_argument("--output", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the native Human acceleration-space muscle-force parity diagnostic"
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

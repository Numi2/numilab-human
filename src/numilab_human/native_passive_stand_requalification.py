"""Bind the current native passive-equilibrium persistent-stand replay.

This receipt records the first native path that carries the complete static
joint-tissue equilibrium into the persistent Human release.  It is a bounded
12.5 microsecond replay on the one-adult source package.  The result improves
the force-balance handoff and dynamic acceleration, but it does not promote
the run to sustained standing, recovery, walking, anatomical contact loading,
or calibrated physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "Docs/media/native-passive-stand-20260915"
PASSIVE_STDOUT = EVIDENCE / "native-passive-512-stdout.txt"
BASELINE_STDOUT = EVIDENCE / "native-default-512-stdout.txt"
PASSIVE_STDERR = EVIDENCE / "native-passive-512-stderr.txt"
BASELINE_STDERR = EVIDENCE / "native-default-512-stderr.txt"
BUILD_LOG = EVIDENCE / "native-build.log"
CONFIGURE_LOG = EVIDENCE / "native-configure.log"
OUTPUT = EVIDENCE / "receipt-v1.json"
SCHEMA = "HumanPack.native-passive-stand-current-requalification.v1"

CURRENT_SOURCE = {
    "branch": "numi-human-passive-stand-20260915",
    "commit": "ef0fc708db0f4de1a07fca426e5a415f62e9da27",
    "base_commit": "c45fa9622f6c73b58febdc24a7115aecf3d7699f",
    "checkout": "/private/tmp/numi-human-native-passive-stand-20260915",
    "device": "Mac mini M4 Pro",
    "binary": "metalrobo_numilab_human_myosim_visual_probe",
    "binary_sha256": "4c02fc6bac0994cfb000a9f9812218775d6fe261c4c0ae914f58d51624b65b79",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native passive stand requalification: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path.read_bytes()


def _number(fields: dict[str, str], key: str) -> float:
    value = fields.get(key)
    _require(value is not None, f"native result is missing {key}")
    try:
        result = float(value)
    except ValueError as error:
        raise HumanImportError(f"native result field {key} is not numeric") from error
    _require(math.isfinite(result), f"native result field {key} is not finite")
    return result


def _integer(fields: dict[str, str], key: str) -> int:
    value = _number(fields, key)
    _require(value == int(value), f"native result field {key} is not integral")
    return int(value)


def _boolean(fields: dict[str, str], key: str) -> bool:
    value = fields.get(key)
    _require(value in {"true", "false"}, f"native result field {key} is not boolean")
    return value == "true"


def _optional_number(fields: dict[str, str], key: str) -> float | None:
    return None if key not in fields else _number(fields, key)


def _optional_integer(fields: dict[str, str], key: str) -> int | None:
    return None if key not in fields else _integer(fields, key)


def _optional_boolean(fields: dict[str, str], key: str) -> bool | None:
    return None if key not in fields else _boolean(fields, key)


def _fields(path: Path) -> dict[str, str]:
    text = _read_regular(path, "native stdout").decode("utf-8")
    line = next(
        (
            row
            for row in reversed(text.splitlines())
            if "persistent_metal_horizon=" in row
            and "myosim_articulated_" in row
        ),
        None,
    )
    _require(line is not None, f"{path} has no persistent Human result line")
    result: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    _require(result.get("persistent_metal_horizon") == "true", f"{path} is not a persistent stand result")
    _require(result.get("myosim_articulated_bodyparts_bone_visual") == "ok",
             f"{path} does not contain a successful native result")
    return result


def _metrics(path: Path) -> dict[str, Any]:
    fields = _fields(path)
    return {
        "device": fields["metal_pose_device"],
        "core_bodies": _integer(fields, "core_bodies"),
        "timestep_seconds": _number(fields, "muscle_step_seconds"),
        "muscle_step_count": _integer(fields, "muscle_step_count"),
        "persistent_completed_steps": _integer(fields, "persistent_completed_steps"),
        "muscle_activation": _number(fields, "muscle_activation"),
        "persistent_max_acceleration_mps2": _number(fields, "persistent_max_acceleration"),
        "persistent_max_penetration_m": _number(fields, "persistent_max_penetration_m"),
        "compiled_stand_normalized_residual_rms": _number(fields, "compiled_stand_normalized_residual_rms"),
        "compiled_stand_initial_normalized_residual_rms": _number(fields, "compiled_stand_initial_normalized_residual_rms"),
        "compiled_stand_max_acceleration_residual": _number(fields, "compiled_stand_max_acceleration_residual"),
        "compiled_stand_max_root_force_residual_n": _number(fields, "compiled_stand_max_root_force_residual"),
        "compiled_stand_max_root_acceleration_residual": _number(fields, "compiled_stand_max_root_acceleration_residual"),
        "compiled_stand_balanced": _boolean(fields, "compiled_stand_balanced"),
        "compiled_stand_support_contacts": _integer(fields, "compiled_stand_support_contacts"),
        "compiled_stand_active_support_contacts": _integer(fields, "compiled_stand_active_support_contacts"),
        "compiled_stand_total_support_force_n": _number(fields, "compiled_stand_total_support_force_n"),
        "source_constraint_preload": fields.get("source_constraint_preload"),
        "source_constraint_preload_max_n": _number(fields, "source_constraint_preload_max_n"),
        "source_dynamic_force_parity_max_delta_n": _number(fields, "source_dynamic_force_parity_max_delta_n"),
        "persistent_source_passive_joint_tissue": _optional_boolean(fields, "persistent_source_passive_joint_tissue"),
        "persistent_passive_coordinate_couplings": _optional_integer(fields, "persistent_passive_coordinate_couplings"),
        "persistent_passive_coordinate_force_max_n": _optional_number(fields, "persistent_passive_coordinate_force_max_n"),
        "stand_deterministic_replay": fields.get("stand_deterministic_replay"),
        "tendon_borrowed_consumer": fields.get("tendon_borrowed_consumer"),
        "tendon_rollback": fields.get("tendon_rollback"),
        "persistent_root_assistance": fields.get("persistent_root_assistance"),
        "muscle_step_max_velocity_delta": _number(fields, "muscle_step_max_velocity_delta"),
        "muscle_step_max_configuration_delta": _number(fields, "muscle_step_max_configuration_delta"),
        "initial_fiber_equilibration_iterations": _integer(fields, "initial_fiber_equilibration_iterations"),
    }


def compile_receipt(*, passive_stdout: Path = PASSIVE_STDOUT,
                    baseline_stdout: Path = BASELINE_STDOUT,
                    passive_stderr: Path = PASSIVE_STDERR,
                    baseline_stderr: Path = BASELINE_STDERR,
                    build_log: Path = BUILD_LOG,
                    configure_log: Path = CONFIGURE_LOG) -> dict[str, Any]:
    passive = _metrics(Path(passive_stdout))
    baseline = _metrics(Path(baseline_stdout))
    _read_regular(Path(passive_stderr), "passive stderr")
    _read_regular(Path(baseline_stderr), "baseline stderr")
    build = _read_regular(Path(build_log), "native build log")
    configure = _read_regular(Path(configure_log), "native configure log")
    _require(b"[131/131] Linking OBJCXX executable" in build,
             "native build log does not prove the Human probe link")
    _require(b"Build files have been written" in configure,
             "native configure log does not prove the isolated build")

    _require(passive["device"] == "Apple M4 Pro" and baseline["device"] == "Apple M4 Pro",
             "native device changed")
    _require(passive["core_bodies"] == 157 and passive["core_bodies"] == baseline["core_bodies"],
             "full source body count changed")
    _require(passive["timestep_seconds"] == 0.0000125 and
             passive["muscle_step_count"] == 512 and
             passive["persistent_completed_steps"] == 512,
             "canonical 12.5 microsecond horizon changed")
    _require(passive["persistent_source_passive_joint_tissue"] and
             passive["persistent_passive_coordinate_couplings"] == 40 and
             passive["source_constraint_preload"] ==
             "static_equality_plus_position_limit_plus_passive_joint_tissue",
             "passive joint-tissue owner was not admitted")
    _require(passive["compiled_stand_balanced"] and
             passive["compiled_stand_normalized_residual_rms"] <= 1.0e-4 and
             passive["compiled_stand_max_acceleration_residual"] <= 1.0e-3,
             "complete static generalized balance did not close")
    _require(passive["compiled_stand_active_support_contacts"] == 6 and
             passive["compiled_stand_support_contacts"] == 10 and
             passive["compiled_stand_total_support_force_n"] > 900.0,
             "support contact or load evidence changed")
    _require(passive["persistent_max_penetration_m"] == 0.0 and
             passive["persistent_root_assistance"] == "none" and
             passive["stand_deterministic_replay"] == "bitwise" and
             passive["tendon_borrowed_consumer"] == "same_command_buffer_exact_snapshot" and
             passive["tendon_rollback"] == "consumer_rejection_preserved_result",
             "bounded release transaction did not close")
    _require(passive["persistent_max_acceleration_mps2"] < baseline["persistent_max_acceleration_mps2"],
             "passive equilibrium did not improve dynamic acceleration")
    improvement = baseline["persistent_max_acceleration_mps2"] - passive["persistent_max_acceleration_mps2"]
    return {
        "schema": SCHEMA,
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            **CURRENT_SOURCE,
            "passive_stdout_sha256": _sha(Path(passive_stdout)),
            "baseline_stdout_sha256": _sha(Path(baseline_stdout)),
            "passive_stderr_sha256": _sha(Path(passive_stderr)),
            "baseline_stderr_sha256": _sha(Path(baseline_stderr)),
            "build_log_sha256": _sha(Path(build_log)),
            "configure_log_sha256": _sha(Path(configure_log)),
        },
        "command": {
            "timestep_seconds": passive["timestep_seconds"],
            "step_count": passive["muscle_step_count"],
            "muscle_activation": passive["muscle_activation"],
            "persistent_metal_stand": True,
            "persistent_source_passive_joint_tissue": True,
            "stand_root_assistance": "none",
            "stand_deterministic_replay": True,
        },
        "results": {
            "passive": passive,
            "baseline": baseline,
            "persistent_max_acceleration_improvement_mps2": improvement,
            "persistent_max_acceleration_improvement_fraction": improvement / baseline["persistent_max_acceleration_mps2"],
        },
        "qualification": {
            "complete_static_generalized_balance": True,
            "bounded_exact_clock_release": True,
            "source_support_load_and_replay": True,
            "sustained_standing": False,
            "perturbation_recovery": False,
            "walking": False,
            "anatomical_support_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
        },
        "boundary": (
            "The isolated Apple M4 Pro branch carries the source passive joint/tissue "
            "couplings into the persistent stand preload. At the canonical 12.5 us "
            "clock it closes the complete static generalized balance and improves the "
            "bounded 512-step release. This remains one adult source data, linearized "
            "experimental upper-joint tissue, source support witnesses and a bounded "
            "explicit release. It does not qualify sustained standing, recovery, walking, "
            "anatomical foot registration, activation calibration, blood-to-tissue mass "
            "transfer, calibrated materials, or subject-specific validation."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "receipt output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "receipt output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_receipt(
        passive_stdout=arguments.passive_stdout,
        baseline_stdout=arguments.baseline_stdout,
        passive_stderr=arguments.passive_stderr,
        baseline_stderr=arguments.baseline_stderr,
        build_log=arguments.build_log,
        configure_log=arguments.configure_log,
    )
    receipt_sha = _immutable_write(arguments.receipt.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "receipt": str(arguments.receipt.resolve()),
                      "sha256": receipt_sha, "status": result["status"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--passive-stdout", type=Path, default=PASSIVE_STDOUT)
    parser.add_argument("--baseline-stdout", type=Path, default=BASELINE_STDOUT)
    parser.add_argument("--passive-stderr", type=Path, default=PASSIVE_STDERR)
    parser.add_argument("--baseline-stderr", type=Path, default=BASELINE_STDERR)
    parser.add_argument("--build-log", type=Path, default=BUILD_LOG)
    parser.add_argument("--configure-log", type=Path, default=CONFIGURE_LOG)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

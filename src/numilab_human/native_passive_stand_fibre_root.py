"""Validate the native persistent-stand fibre-root initialization repair.

This receipt is deliberately narrower than the common-duration refinement. It
records the source change that carries the accepted static fibre lengths into
the Metal horizon and snaps a stationary constitutive root when float-length
roundoff would otherwise become artificial fibre speed. The long canonical
release remains open, so this compiler never promotes standing or force
convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical

ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "Docs/media/native-passive-stand-fibre-root-20260915"
SCHEMA = "HumanPack.native-passive-stand-fibre-root-current-requalification.v1"
SOURCE_COMMIT = "f45fcfdc80a05c2226d638227295add7f789c55c"
BINARY_SHA256 = "62648144c20f386b25f625d503fd101f5c846b9ca8b0dac0649e4a064fb04919"
SOURCE_PATCH = CASE_ROOT / "source.patch"
CASE_SPEC = (
    ("12p5us-64", 1.25e-5, 64, 0.0008),
    ("12p5us-512", 1.25e-5, 512, 0.0064),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native passive stand fibre root: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _field(text: str, key: str, *, cast: type = float) -> Any:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\"[^\"]*\"|[^\s]+)", text)
    _require(match is not None, f"native output lacks {key}")
    raw = match.group(1).strip('"')
    try:
        return cast(raw)
    except (TypeError, ValueError) as error:
        raise HumanImportError(f"native output field {key} is invalid") from error


def _case(case_root: Path, name: str, timestep: float, steps: int,
          duration: float) -> dict[str, Any]:
    stdout = case_root / f"{name}-stdout.txt"
    stderr = case_root / f"{name}-stderr.txt"
    _require(stdout.is_file() and not stdout.is_symlink(), f"{name} stdout is missing")
    _require(stderr.is_file() and not stderr.is_symlink(), f"{name} stderr is missing")
    text = stdout.read_text(encoding="utf-8")
    timing = stderr.read_text(encoding="utf-8").strip().splitlines()
    _require(timing and all(line.startswith(prefix) for line, prefix in zip(
        timing, ("real ", "user ", "sys ")
    )), f"{name} stderr lacks bounded execution timing")
    _require(abs(_field(text, "muscle_step_seconds") - timestep) <= 1.0e-12,
             f"{name} timestep differs")
    _require(_field(text, "muscle_step_count", cast=int) == steps,
             f"{name} step count differs")
    _require(abs(timestep * steps - duration) <= 1.0e-12,
             f"{name} duration differs")
    _require(_field(text, "persistent_metal_horizon", cast=str) == "true",
             f"{name} is not a persistent horizon")
    _require(_field(text, "persistent_source_passive_joint_tissue", cast=str) == "true",
             f"{name} does not carry source passive tissue")
    _require(_field(text, "compiled_stand_balanced", cast=str) == "true",
             f"{name} compiled stand is not balanced")
    _require(_field(text, "stand_deterministic_replay", cast=str) == "bitwise",
             f"{name} replay is not bitwise")
    _require(_field(text, "source_support_metal_device", cast=str) == "Apple M4 Pro",
             f"{name} is not physical-Mac evidence")
    penetration = _field(text, "persistent_max_penetration_m")
    _require(penetration == 0.0, f"{name} has nonzero penetration")
    completed = _field(text, "persistent_completed_steps", cast=int)
    _require(completed == steps, f"{name} completed-step count differs")
    return {
        "name": name,
        "timestep_seconds": timestep,
        "timestep_nanoseconds": int(round(timestep * 1.0e9)),
        "step_count": steps,
        "duration_seconds": duration,
        "persistent_completed_steps": completed,
        "persistent_max_acceleration_mps2": _field(text, "persistent_max_acceleration"),
        "persistent_max_penetration_m": penetration,
        "compiled_stand_normalized_residual_rms": _field(
            text, "compiled_stand_normalized_residual_rms"
        ),
        "initial_fiber_equilibration_iterations": _field(
            text, "initial_fiber_equilibration_iterations", cast=int
        ),
        "initial_fiber_equilibration_max_length_delta_m": _field(
            text, "initial_fiber_equilibration_max_length_delta_m"
        ),
        "initial_fiber_equilibration_max_velocity_mps": _field(
            text, "initial_fiber_equilibration_max_velocity_mps"
        ),
        "source_dynamic_force_parity_max_delta_n": _field(
            text, "source_dynamic_force_parity_max_delta_n"
        ),
        "stand_one_step_max_v_error": _field(text, "stand_one_step_max_v_error"),
        "muscle_step_max_velocity_delta": _field(text, "muscle_step_max_velocity_delta"),
        "muscle_step_max_configuration_delta": _field(
            text, "muscle_step_max_configuration_delta"
        ),
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout": _relative(stdout),
        "stderr": _relative(stderr),
    }


def compile_requalification(*, case_root: Path = CASE_ROOT) -> dict[str, Any]:
    case_root = Path(case_root).resolve()
    _require(case_root.is_relative_to(ROOT), "case root resolves outside the repository")
    patch = SOURCE_PATCH if case_root == CASE_ROOT else case_root / "source.patch"
    _require(patch.is_file() and not patch.is_symlink(), "source patch is missing")
    cases = [_case(case_root, *spec) for spec in CASE_SPEC]
    _require(_sha256(patch) == "3c71da40114aa07ecf66a68dabbdbf4b49f974fc8eb1a10f07f1374c3113ce56",
             "source patch fingerprint differs")
    short, long = cases
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.native-passive-stand-fibre-root.1",
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            "branch": "numi-human-passive-stand-20260915",
            "commit": SOURCE_COMMIT,
            "binary": "metalrobo_numilab_human_myosim_visual_probe",
            "binary_sha256": BINARY_SHA256,
            "device": "Mac mini M4 Pro",
            "source_patch": _relative(patch),
            "source_patch_sha256": _sha256(patch),
        },
        "repair": {
            "static_fibre_lengths_carried_into_runtime": True,
            "stationary_zero_velocity_root_roundoff_guard": True,
            "initial_fibre_equilibration_retained": True,
            "initial_fibre_equilibration_not_a_dynamic_step": True,
        },
        "cases": cases,
        "qualification": {
            "physical_mac_replay": True,
            "exact_12_5_us_clock": True,
            "complete_static_generalized_balance": True,
            "zero_penetration": True,
            "bitwise_replay": True,
            "short_horizon_release_bounded": short["persistent_max_acceleration_mps2"] < 1.0,
            "long_horizon_release_bounded": long["persistent_max_acceleration_mps2"] < 1.0,
            "force_convergence": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
            "anatomical_support_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
        },
        "blocker": {
            "id": "force_convergence",
            "status": "open",
            "reason": (
                "The stationary-root repair bounds the 0.8 ms 12.5 us release at "
                f"{short['persistent_max_acceleration_mps2']:.12g} m/s2, but the same "
                f"source state reaches {long['persistent_max_acceleration_mps2']:.12g} "
                "m/s2 over 6.4 ms. Long-horizon state/force coupling remains open."
            ),
        },
        "boundary": (
            "This physical Mac mini receipt closes the runtime fibre-seed ownership "
            "defect and retains exact static balance, zero penetration and bitwise "
            "replay. It does not qualify long-horizon force convergence, sustained "
            "standing, recovery, walking, anatomical contact, activation calibration, "
            "blood transfer, materials or subject calibration."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-root", type=Path, default=CASE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_requalification(case_root=arguments.case_root)
    digest = immutable_write(arguments.output.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "output": str(arguments.output.resolve()),
                      "sha256": digest,
                      "force_convergence": result["qualification"]["force_convergence"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        raise SystemExit(run(parser.parse_args()))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"native passive stand fibre root: {error}\n")

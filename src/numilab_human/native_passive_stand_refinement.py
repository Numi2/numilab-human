"""Validate common-duration native passive-stand timestep refinement.

The four native runs share the same pose search, source passive coupling,
support payload, and physical duration.  This compiler records the measured
release traces and fails the force-convergence gate when the release changes
materially with the timestep.
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
CASE_ROOT = ROOT / "Docs/media/native-passive-stand-refinement-20260915"
SCHEMA = "HumanPack.native-passive-stand-refinement-current-requalification.v1"
SOURCE_COMMIT = "7625ec565e086faf0dcd349846dadc2d22d65e86"
BINARY_SHA256 = "e78e6efd31471eba0839ebf75674d64395e1e804a9751be7f52c1fb420113fe5"
CASE_SPEC = (
    ("100us", 1.0e-4, 64),
    ("50us", 5.0e-5, 128),
    ("25us", 2.5e-5, 256),
    ("12p5us", 1.25e-5, 512),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native passive stand refinement: " + message)


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


def _case(case_root: Path, name: str, timestep: float, expected_steps: int) -> dict[str, Any]:
    stdout = case_root / name / "stdout.txt"
    stderr = case_root / name / "stderr.txt"
    _require(stdout.is_file() and not stdout.is_symlink(), f"{name} stdout is missing")
    _require(stderr.is_file() and not stderr.is_symlink(), f"{name} stderr is missing")
    text = stdout.read_text(encoding="utf-8")
    _require(not stderr.read_text(encoding="utf-8").strip(), f"{name} native stderr is not empty")
    parsed_step = _field(text, "muscle_step_seconds")
    _require(abs(parsed_step - timestep) <= 1.0e-12, f"{name} timestep differs")
    steps = _field(text, "muscle_step_count", cast=int)
    _require(steps == expected_steps, f"{name} step count differs")
    _require(_field(text, "persistent_metal_horizon", cast=str) == "true",
             f"{name} is not a persistent horizon")
    _require(_field(text, "persistent_source_passive_joint_tissue", cast=str) == "true",
             f"{name} does not carry source passive tissue")
    _require(_field(text, "compiled_stand_balanced", cast=str) == "true",
             f"{name} compiled stand is not balanced")
    _require(_field(text, "stand_deterministic_replay", cast=str) == "bitwise",
             f"{name} replay is not bitwise")
    _require(_field(text, "source_support_metal_device", cast=str) == 'Apple M4 Pro',
             f"{name} is not physical-Mac evidence")
    penetration = _field(text, "persistent_max_penetration_m")
    _require(penetration == 0.0, f"{name} has nonzero penetration")
    return {
        "name": name,
        "timestep_seconds": timestep,
        "timestep_nanoseconds": int(round(timestep * 1.0e9)),
        "step_count": steps,
        "duration_seconds": timestep * steps,
        "persistent_completed_steps": _field(text, "persistent_completed_steps", cast=int),
        "persistent_max_acceleration_mps2": _field(text, "persistent_max_acceleration"),
        "persistent_max_penetration_m": penetration,
        "compiled_stand_normalized_residual_rms": _field(text, "compiled_stand_normalized_residual_rms"),
        "compiled_stand_balanced": True,
        "source_dynamic_force_parity_max_delta_n": _field(text, "source_dynamic_force_parity_max_delta_n"),
        "muscle_step_max_velocity_delta": _field(text, "muscle_step_max_velocity_delta"),
        "muscle_step_max_configuration_delta": _field(text, "muscle_step_max_configuration_delta"),
        "stand_deterministic_replay": "bitwise",
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout": _relative(stdout),
        "stderr": _relative(stderr),
    }


def compile_refinement(*, case_root: Path = CASE_ROOT) -> dict[str, Any]:
    case_root = Path(case_root).resolve()
    _require(case_root.is_relative_to(ROOT), "case root resolves outside the repository")
    cases = [_case(case_root, name, timestep, steps) for name, timestep, steps in CASE_SPEC]
    durations = [row["duration_seconds"] for row in cases]
    _require(max(durations) - min(durations) <= 1.0e-15,
             "common-duration refinement cases differ in physical duration")
    static_residuals = [row["compiled_stand_normalized_residual_rms"] for row in cases]
    _require(max(static_residuals) - min(static_residuals) <= 1.0e-12,
             "refinement cases do not share the compiled static balance")
    peaks = [row["persistent_max_acceleration_mps2"] for row in cases]
    peak_minimum = min(peaks)
    peak_maximum = max(peaks)
    peak_range_over_minimum = (peak_maximum - peak_minimum) / peak_minimum
    convergence_tolerance = 0.05
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.native-passive-stand-refinement.1",
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            "branch": "numi-human-passive-stand-20260915",
            "commit": SOURCE_COMMIT,
            "binary": "metalrobo_numilab_human_myosim_visual_probe",
            "binary_sha256": BINARY_SHA256,
            "device": "Mac mini M4 Pro",
        },
        "common_duration": {
            "duration_seconds": durations[0],
            "case_count": len(cases),
            "timestep_nanoseconds": [row["timestep_nanoseconds"] for row in cases],
            "step_counts": [row["step_count"] for row in cases],
            "static_normalized_residual_rms": static_residuals,
            "static_residual_spread": max(static_residuals) - min(static_residuals),
        },
        "cases": cases,
        "convergence": {
            "maximum_peak_acceleration_mps2": peak_maximum,
            "minimum_peak_acceleration_mps2": peak_minimum,
            "peak_acceleration_range_over_minimum": peak_range_over_minimum,
            "peak_acceleration_relative_tolerance": convergence_tolerance,
            "peak_acceleration_converged": peak_range_over_minimum <= convergence_tolerance,
            "force_convergence": peak_range_over_minimum <= convergence_tolerance,
        },
        "qualification": {
            "physical_mac_replay": True,
            "exact_clock_cases": True,
            "common_duration": True,
            "complete_static_generalized_balance": True,
            "zero_penetration": all(row["persistent_max_penetration_m"] == 0.0 for row in cases),
            "bitwise_replay": all(row["stand_deterministic_replay"] == "bitwise" for row in cases),
            "force_convergence": peak_range_over_minimum <= convergence_tolerance,
            "anatomical_support_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "blocker": {
            "id": "force_convergence",
            "status": "proved" if peak_range_over_minimum <= convergence_tolerance else "open",
            "reason": (
                "Common-duration release peaks are not timestep-converged: "
                f"range/minimum={peak_range_over_minimum:.12g} exceeds "
                f"the {convergence_tolerance:.12g} tolerance."
            ),
        },
        "boundary": (
            "Four physical Mac mini M4 Pro passive-source releases use the same pose "
            "search, support and tendon/equality payloads over a common 6.4 ms duration. "
            "They prove exact-clock execution, complete static balance, zero penetration "
            "and bitwise replay. The dynamic release is not timestep-converged, so this "
            "receipt does not promote force convergence, sustained standing, recovery, "
            "walking, anatomical contact, activation calibration, blood transfer, materials "
            "or subject calibration."
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
    result = compile_refinement(case_root=arguments.case_root)
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
        parser.exit(2, f"native passive stand refinement: {error}\n")

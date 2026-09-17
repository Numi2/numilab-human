"""Validate common-duration native passive-stand timestep refinement.

A new run manifest binds source, binary, payloads, and all four output pairs.
Peak acceleration consistency is a diagnostic, never proof of force convergence.
Historical evidence remains readable without assigning its identity to new runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical

ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "Docs/media/native-passive-stand-refinement-20260915"
SCHEMA = "HumanPack.native-passive-stand-refinement-current-requalification.v1"
MANIFEST_SCHEMA = "numi.human.native-refinement-run.v1"
SOURCE_COMMIT = "7625ec565e086faf0dcd349846dadc2d22d65e86"
BINARY_SHA256 = "e78e6efd31471eba0839ebf75674d64395e1e804a9751be7f52c1fb420113fe5"
CASE_SPEC = (("100us", 1.0e-4, 64), ("50us", 5.0e-5, 128),
             ("25us", 2.5e-5, 256), ("12p5us", 1.25e-5, 512))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native passive stand refinement: " + message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _field(text: str, key: str, *, cast: type = float) -> Any:
    matches = re.findall(rf'(?:^|\s){re.escape(key)}=("[^"]*"|[^\s]+)', text)
    _require(len(matches) == 1, f"native output lacks unique {key}")
    try:
        result = cast(matches[0].strip('"'))
    except (TypeError, ValueError) as error:
        raise HumanImportError(f"native output field {key} is invalid") from error
    if cast is float:
        _require(math.isfinite(result), f"native output field {key} is not finite")
    return result


def _case(case_root: Path, name: str, timestep: float, expected_steps: int,
          *, expected_device: str = "Apple M4 Pro", allow_validation_banner: bool = False) -> dict[str, Any]:
    stdout, stderr = (case_root / name / filename for filename in ("stdout.txt", "stderr.txt"))
    for path in (stdout, stderr):
        _require(path.is_file() and not path.is_symlink(), f"{name} {path.name} is missing or redirected")
    text = stdout.read_text(encoding="utf-8")
    stderr_lines = [line.strip() for line in stderr.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Metal's validation-enabled banner is not a failure; arbitrary validation
    # warnings or native errors are NEVER discarded with it.
    if allow_validation_banner:
        banner = re.compile(r"^(?:\d{4}-\d{2}-\d{2} [\d:.]+ \S+\[\d+:\d+\] )?Metal API Validation Enabled$")
        stderr_lines = [line for line in stderr_lines if not banner.fullmatch(line)]
    _require(not stderr_lines, f"{name} native stderr is not empty")
    _require(abs(_field(text, "muscle_step_seconds") - timestep) <= 1.0e-12,
             f"{name} timestep differs")
    steps = _field(text, "muscle_step_count", cast=int)
    completed = _field(text, "persistent_completed_steps", cast=int)
    _require(steps == expected_steps and completed == expected_steps,
             f"{name} requested/completed step count differs")
    for key, required in (("persistent_metal_horizon", "true"),
                          ("persistent_source_passive_joint_tissue", "true"),
                          ("compiled_stand_balanced", "true"),
                          ("stand_deterministic_replay", "bitwise"),
                          ("source_support_metal_device", expected_device)):
        _require(_field(text, key, cast=str) == required, f"{name} invalid {key}")
    penetration = _field(text, "persistent_max_penetration_m")
    _require(penetration == 0.0, f"{name} has nonzero penetration")
    metrics = {
        "persistent_max_acceleration_mps2": "persistent_max_acceleration",
        "compiled_stand_normalized_residual_rms": "compiled_stand_normalized_residual_rms",
        "source_dynamic_force_parity_max_delta_n": "source_dynamic_force_parity_max_delta_n",
        "muscle_step_max_velocity_delta": "muscle_step_max_velocity_delta",
        "muscle_step_max_configuration_delta": "muscle_step_max_configuration_delta",
    }
    values = {name: _field(text, key) for name, key in metrics.items()}
    _require(all(value >= 0.0 for value in values.values()), f"{name} negative norm/peak diagnostic")
    stage_fields = {
        "maximum_free_acceleration_mixed_units": "persistent_max_free_acceleration",
        "maximum_free_acceleration_dof": "persistent_max_free_acceleration_dof",
        "maximum_constraint_delta_v_mixed_units": "persistent_max_constraint_delta_v",
        "maximum_constraint_delta_v_dof": "persistent_max_constraint_delta_v_dof",
        "maximum_pre_projection_delta_v_mixed_units": "persistent_max_pre_projection_delta_v",
        "maximum_pre_projection_delta_v_dof": "persistent_max_pre_projection_delta_v_dof",
        "maximum_published_delta_v_mixed_units": "persistent_max_published_delta_v",
        "maximum_published_delta_v_dof": "persistent_max_published_delta_v_dof",
    }
    stage_tokens = [f"{key}=" in text for key in (
        "persistent_max_acceleration_semantics", *stage_fields.values())]
    _require(not any(stage_tokens) or all(stage_tokens),
             f"{name} has a partial velocity-stage diagnostic record")
    velocity_stage = None
    if all(stage_tokens):
        semantics = _field(text, "persistent_max_acceleration_semantics", cast=str)
        _require(semantics == "pre_projection_total_delta_v_divided_by_timestep",
                 f"{name} acceleration semantics changed")
        stage: dict[str, Any] = {}
        for output_name, native_name in stage_fields.items():
            cast = int if output_name.endswith("_dof") else float
            stage[output_name] = _field(text, native_name, cast=cast)
        for key, value in stage.items():
            if key.endswith("_dof"):
                _require(0 <= value < 128, f"{name} invalid {key}")
            else:
                _require(value >= 0.0, f"{name} negative {key}")
        legacy_rate = values["persistent_max_acceleration_mps2"]
        reconstructed_rate = stage["maximum_pre_projection_delta_v_mixed_units"] / timestep
        rate_tolerance = max(1.0e-5, 2.0e-5 * max(legacy_rate, reconstructed_rate, 1.0))
        _require(abs(legacy_rate - reconstructed_rate) <= rate_tolerance,
                 f"{name} legacy acceleration does not reconstruct pre-projection delta-v")
        velocity_stage = {
            "schema": "numi.human.velocity-stage-diagnostics.v1",
            "legacy_acceleration_semantics": semantics,
            **stage,
            "reconstructed_pre_projection_rate_mixed_units": reconstructed_rate,
            "legacy_rate_reconstruction_tolerance": rate_tolerance,
        }
    return {"name": name, "timestep_seconds": timestep,
            "timestep_nanoseconds": int(round(timestep * 1.0e9)), "step_count": steps,
            "duration_seconds": timestep * completed, "persistent_completed_steps": completed,
            **values, "velocity_stage_diagnostics": velocity_stage,
            "persistent_max_penetration_m": penetration, "compiled_stand_balanced": True,
            "stand_deterministic_replay": "bitwise", "stdout_sha256": _sha256(stdout),
            "stderr_sha256": _sha256(stderr), "stdout": _relative(stdout), "stderr": _relative(stderr)}


def _hash(value: Any, digits: int, label: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{digits}}}", value) is not None,
             f"{label} is not a complete lowercase hash")
    return value


def _manifest(path: Path, case_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), "run manifest is missing or redirected")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HumanImportError(f"invalid run manifest: {error}") from error
    _require(isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA,
             "run manifest schema mismatch")
    source = manifest.get("source")
    _require(isinstance(source, dict), "run manifest has no source")
    _hash(source.get("commit"), 40, "native commit")
    _hash(source.get("binary_sha256"), 64, "native binary")
    _require(source.get("worktree_clean") is True, "new refinement requires a clean native source")
    _require(source.get("root_assistance") == "none", "new refinement must be assistance-free")
    _require(isinstance(source.get("device"), str) and source["device"].startswith("Apple "),
             "new refinement has no Apple device identity")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict) and
             {"binary", "rigid", "muscle", "tendon", "support", "equalities"} <= set(artifacts),
             "run manifest is missing native program/payload identities")
    for name, record in artifacts.items():
        _require(isinstance(record, dict) and isinstance(record.get("path"), str), f"invalid artifact {name}")
        expected = _hash(record.get("sha256"), 64, f"artifact {name}")
        target = Path(record["path"])
        if not target.is_absolute():
            target = path.parent / target
        _require(target.is_file() and not target.is_symlink() and _sha256(target) == expected,
                 f"artifact changed or missing: {name}")
    _require(artifacts["binary"]["sha256"] == source["binary_sha256"], "native binary identity mismatch")
    runs = manifest.get("cases")
    _require(isinstance(runs, dict) and set(runs) == {case[0] for case in CASE_SPEC},
             "run manifest must bind exactly four refinement cases")
    for name, timestep, steps in CASE_SPEC:
        row = runs[name]
        _require(isinstance(row, dict) and row.get("exit_code") == 0 and type(row.get("exit_code")) is int,
                 f"{name} process did not finish successfully")
        _require(row.get("source_commit") == source["commit"] and
                 row.get("binary_sha256") == source["binary_sha256"], f"{name} source/binary differs")
        _require(row.get("artifact_hashes") == {key: value["sha256"] for key, value in artifacts.items()},
                 f"{name} artifact identities differ")
        _require(type(row.get("timestep_nanoseconds")) is int and
                 row["timestep_nanoseconds"] == int(round(timestep * 1e9)) and
                 type(row.get("steps")) is int and row["steps"] == steps, f"{name} execution grid differs")
        for stream in ("stdout", "stderr"):
            expected = _hash(row.get(f"{stream}_sha256"), 64, f"{name} {stream}")
            _require(_sha256(case_root / name / f"{stream}.txt") == expected, f"{name} {stream} changed")
    return dict(source), {"path": str(path), "sha256": _sha256(path), "artifacts": artifacts}


def compile_refinement(*, case_root: Path = CASE_ROOT,
                       run_manifest: Path | None = None) -> dict[str, Any]:
    case_root = Path(case_root).resolve()
    source = {"branch": "numi-human-passive-stand-20260915", "commit": SOURCE_COMMIT,
              "binary": "metalrobo_numilab_human_myosim_visual_probe", "binary_sha256": BINARY_SHA256,
              "device": "Mac mini M4 Pro"}
    binding = None
    device = "Apple M4 Pro"
    if run_manifest is None:
        _require(case_root == CASE_ROOT.resolve(), "a new case root requires --run-manifest; historical source identity cannot be reused")
    else:
        source, binding = _manifest(Path(run_manifest).absolute(), case_root)
        device = source["device"]
    cases = [_case(case_root, name, timestep, steps, expected_device=device,
                   allow_validation_banner=binding is not None)
             for name, timestep, steps in CASE_SPEC]
    durations = [row["duration_seconds"] for row in cases]
    _require(max(durations) - min(durations) <= 1.0e-15, "cases differ in physical duration")
    static = [row["compiled_stand_normalized_residual_rms"] for row in cases]
    _require(max(static) - min(static) <= 1.0e-12, "cases do not share the compiled static balance")
    peaks = [row["persistent_max_acceleration_mps2"] for row in cases]
    low, high = min(peaks), max(peaks)
    ratio = (high - low) / low if low > 0.0 else (0.0 if high == 0.0 else None)
    tolerance = 0.05
    peak_consistent = ratio is not None and ratio <= tolerance
    stage_presence = [row["velocity_stage_diagnostics"] is not None for row in cases]
    _require(not any(stage_presence) or all(stage_presence),
             "velocity-stage diagnostics differ across refinement cases")
    stage_convergence = None
    if all(stage_presence):
        free_acceleration = [
            row["velocity_stage_diagnostics"]["maximum_free_acceleration_mixed_units"]
            for row in cases
        ]
        free_low, free_high = min(free_acceleration), max(free_acceleration)
        free_ratio = ((free_high - free_low) / free_low if free_low > 0.0
                      else (0.0 if free_high == 0.0 else None))
        constraint_delta_v = [row["velocity_stage_diagnostics"]["maximum_constraint_delta_v_mixed_units"]
                              for row in cases]
        published_delta_v = [row["velocity_stage_diagnostics"]["maximum_published_delta_v_mixed_units"]
                             for row in cases]
        stage_convergence = {
            "free_force_acceleration_range_over_minimum": free_ratio,
            "free_force_acceleration_relative_tolerance": tolerance,
            "free_force_acceleration_converged": free_ratio is not None and free_ratio <= tolerance,
            "maximum_free_force_acceleration_mixed_units": free_high,
            "free_force_acceleration_semantics": (
                "unconstrained M_effective^-1 applied-force acceleration before contact, equality, "
                "and joint-limit projection; not a constrained physical body acceleration"
            ),
            "maximum_constraint_delta_v_mixed_units": max(constraint_delta_v),
            "minimum_constraint_delta_v_mixed_units": min(constraint_delta_v),
            "maximum_published_delta_v_mixed_units": max(published_delta_v),
            "minimum_published_delta_v_mixed_units": min(published_delta_v),
            "constraint_events_temporally_converged": False,
        }
    return {
        "schema": SCHEMA, "compiler": "numilab-human.native-passive-stand-refinement.4",
        "status": "partial", "subject": "one adult male source package", "source": source,
        "run_binding": binding,
        "common_duration": {"duration_seconds": durations[0], "case_count": len(cases),
                            "timestep_nanoseconds": [row["timestep_nanoseconds"] for row in cases],
                            "step_counts": [row["step_count"] for row in cases],
                            "static_normalized_residual_rms": static, "static_residual_spread": max(static)-min(static)},
        "cases": cases,
        "convergence": {"maximum_peak_acceleration_mps2": high, "minimum_peak_acceleration_mps2": low,
                        "peak_acceleration_range_over_minimum": ratio,
                        "peak_acceleration_relative_tolerance": tolerance,
                        "peak_acceleration_converged": peak_consistent,
                        "legacy_peak_acceleration_is_constraint_rate_mixed": all(stage_presence),
                        "velocity_stage_diagnostics_available": all(stage_presence),
                        "velocity_stage": stage_convergence,
                        "force_convergence": False},
        "qualification": {"physical_mac_replay": True, "exact_clock_cases": True, "common_duration": True,
                          "complete_static_generalized_balance": True, "zero_penetration": True,
                          "bitwise_replay": True, "force_convergence": False,
                          "anatomical_support_loading": False, "activation_calibration": False,
                          "blood_mass_transfer": False, "material_calibration": False,
                          "subject_calibration": False, "sustained_standing": False, "recovery": False, "walking": False},
        "blocker": {"id": "force_convergence", "status": "open", "reason": (
            ((("Free-force acceleration is grid-consistent, but it is unconstrained and can contain "
               "forces subsequently balanced by equality, contact, and joint-limit reactions. "
               "Compare same-time state, complete reactions, complementarity, and impulsive work.")
              if stage_convergence and stage_convergence["free_force_acceleration_converged"]
              else "Separated velocity stages are available, but unconstrained free-force acceleration "
                   "or the constrained trajectory is not yet timestep-converged.")
             if all(stage_presence) else
             "Matching legacy acceleration peaks alone do not establish force convergence; compare "
             "same-time state, constraint reactions, complementarity, and work on the bound source.")
            if peak_consistent else f"Common-duration acceleration peaks are inconsistent: range/minimum={ratio!r}, tolerance={tolerance}.")},
        "boundary": "Bounded source-bound release diagnostics only. Static reaction references are not runtime force evidence. "
                    "Neither peak agreement, zero penetration nor replay establishes temporal force convergence or biological validation.",
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
    parser.add_argument("--run-manifest", type=Path, help="numi.human.native-refinement-run.v1 identities for a new native run")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_refinement(case_root=arguments.case_root, run_manifest=getattr(arguments, "run_manifest", None))
    digest = immutable_write(arguments.output.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "output": str(arguments.output.resolve()), "sha256": digest,
                     "force_convergence": result["qualification"]["force_convergence"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        raise SystemExit(run(parser.parse_args()))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"native passive stand refinement: {error}\n")

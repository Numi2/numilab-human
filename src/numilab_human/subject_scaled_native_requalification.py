"""Bind the physical-M4 replay of the subject-scaled NHRIGID2 input.

The replay proves that the mass/inertia handoff is consumable by the published
native source tuple.  It is deliberately a bounded 64-step result: it does
not promote a uniform mass candidate to segment calibration, full generalized
force convergence, sustained standing, recovery, walking, or organ/tissue
ownership.
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
EVIDENCE = ROOT / "Docs/media/native-subject-scaled-runtime-20260915"
STDOUT = EVIDENCE / "native.stdout.txt"
STDERR = EVIDENCE / "native.stderr.txt"
OUTPUT = EVIDENCE / "receipt-v1.json"
SCALED_INPUT = ROOT / "Docs/media/subject-scaled-runtime-input-20260915/myosim-fullbody-core-reference.nhrigid"
SCALED_INPUT_RECEIPT = ROOT / "Docs/media/subject-scaled-runtime-input-20260915/receipt-v1.json"
SCALING_RECEIPT = ROOT / "Docs/media/subject-mass-scaling-20260915/receipt-v1.json"
SOURCE_PACKAGE_RECEIPT = ROOT / "Docs/media/native-runtime-source-package-20260915/source-package-receipt-v1.json"
SCHEMA = "HumanPack.native-subject-scaled-runtime-requalification.v1"

NATIVE_SOURCE = {
    "immutable_tag": "human-native-step281-rank-audit-20260915",
    "resolved_commit": "337741b51bfc4a5552a837dacb4d0b5c4268d298",
    "binary": "metalrobo_numilab_human_myosim_visual_probe",
    "binary_sha256": "b85669fefaf414a28a9eb44531985dcd83fe549eb5b8d6c7747773bedb3e15bc",
    "device": "Apple M4 Pro",
    "checkout": "/private/tmp/numi-human-rank-audit-public-verify-build",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("subject-scaled native requalification: " + message)


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


def _fields(path: Path) -> dict[str, str]:
    text = _read_regular(path, "native stdout").decode("utf-8")
    line = next(
        (row for row in reversed(text.splitlines())
         if "persistent_metal_horizon=" in row and
         "myosim_articulated_marker_visual=" in row),
        None,
    )
    _require(line is not None, "stdout has no persistent Human result line")
    result: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    _require(result.get("myosim_articulated_marker_visual") == "ok",
             "native result did not complete the articulated visual probe")
    _require(result.get("persistent_metal_horizon") == "true",
             "native result is not a persistent horizon")
    return result


def _json_line(path: Path, key: str) -> dict[str, Any]:
    text = _read_regular(path, "native stdout").decode("utf-8")
    line = next((row for row in text.splitlines() if row.startswith(key + "=")), None)
    _require(line is not None, f"stdout is missing {key}")
    try:
        value = json.loads(line.split("=", 1)[1])
    except json.JSONDecodeError as error:
        raise HumanImportError(f"stdout {key} is not JSON") from error
    _require(isinstance(value, dict), f"stdout {key} is not an object")
    return value


def _views(path: Path) -> dict[str, Any]:
    text = _read_regular(path, "native stdout").decode("utf-8")
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("view="):
            continue
        fields: dict[str, str] = {}
        for token in shlex.split(line):
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        name = fields.get("view")
        if name:
            result[name] = {key: int(fields[key]) for key in (
                "body_pixels", "bone_pixels", "muscle_site_pixels", "muscle_route_pixels",
                "skin_shell_pixels", "organ_surface_pixels", "vessel_surface_pixels",
            ) if key in fields}
    _require(set(result) == {"front", "oblique", "side", "rear"},
             "native visual view set is incomplete")
    return result


def _metrics(path: Path) -> dict[str, Any]:
    fields = _fields(path)
    views = _views(path)
    audit = _json_line(path, "persistent_dynamic_force_audit")
    trace = _json_line(path, "persistent_stand_trace")
    _require(audit.get("schema") == "numi.human.persistent-dynamic-force-audit.v1",
             "dynamic-force audit schema changed")
    _require(trace.get("schema") == "numi.human.persistent-stand-trace.v2",
             "persistent trace schema changed")
    _require(trace.get("endpoint_equivalent") == "bitwise" and
             trace.get("endpoint_max_q_delta") == 0 and
             trace.get("endpoint_max_v_delta") == 0,
             "persistent trace replay is not bitwise")
    return {
        "device": fields["metal_pose_device"],
        "core_bodies": _integer(fields, "core_bodies"),
        "muscle_step_seconds": _number(fields, "muscle_step_seconds"),
        "muscle_step_count": _integer(fields, "muscle_step_count"),
        "persistent_completed_steps": _integer(fields, "persistent_completed_steps"),
        "muscle_activation": _number(fields, "muscle_activation"),
        "muscle_activation_scope": fields.get("muscle_activation_scope"),
        "compiled_stand_recruited_muscles": _integer(fields, "compiled_stand_recruited_muscles"),
        "compiled_stand_active_muscles": _integer(fields, "compiled_stand_active_muscles"),
        "compiled_stand_support_contacts": _integer(fields, "compiled_stand_support_contacts"),
        "compiled_stand_active_support_contacts": _integer(fields, "compiled_stand_active_support_contacts"),
        "compiled_stand_total_support_force_n": _number(fields, "compiled_stand_total_support_force_n"),
        "compiled_stand_balanced": _boolean(fields, "compiled_stand_balanced"),
        "compiled_stand_normalized_residual_rms": _number(fields, "compiled_stand_normalized_residual_rms"),
        "compiled_stand_initial_normalized_residual_rms": _number(fields, "compiled_stand_initial_normalized_residual_rms"),
        "compiled_stand_max_root_force_residual_n": _number(fields, "compiled_stand_max_root_force_residual"),
        "compiled_stand_max_root_acceleration_residual": _number(fields, "compiled_stand_max_root_acceleration_residual"),
        "initial_fiber_equilibration_iterations": _integer(fields, "initial_fiber_equilibration_iterations"),
        "initial_fiber_equilibration_max_length_delta_m": _number(fields, "initial_fiber_equilibration_max_length_delta_m"),
        "persistent_max_acceleration_mps2": _number(fields, "persistent_max_acceleration"),
        "persistent_max_penetration_m": _number(fields, "persistent_max_penetration_m"),
        "persistent_root_assistance": fields.get("persistent_root_assistance"),
        "stand_deterministic_replay": fields.get("stand_deterministic_replay"),
        "source_dynamic_force_parity_max_delta_n": _number(fields, "source_dynamic_force_parity_max_delta_n"),
        "persistent_dynamic_force_residual_max_n": _number(fields, "persistent_dynamic_force_residual_max_n"),
        "tendon_step_max_force_residual_n": _number(fields, "tendon_step_max_force_residual_n"),
        "tendon_step_max_moment_residual_nm": _number(fields, "tendon_step_max_moment_residual_nm"),
        "source_support_witnesses": _integer(fields, "source_support_witnesses"),
        "source_support_active_contacts": _integer(fields, "source_support_active_contacts"),
        "source_support_min_plane_gap_m": _number(fields, "source_support_min_plane_gap_m"),
        "stand_joint_equalities": _integer(fields, "stand_joint_equalities"),
        "stand_max_equality_position_error": _number(fields, "stand_max_equality_position_error"),
        "stand_max_equality_velocity_error": _number(fields, "stand_max_equality_velocity_error"),
        "views": views,
        "dynamic_force_audit_max_abs_residual_n": float(audit["maximum_abs_residual_n"]),
        "dynamic_force_audit_rows": len(audit.get("rows", [])),
        "trace_samples": len(trace.get("samples", [])),
        "trace_endpoint_equivalent": trace["endpoint_equivalent"],
        "trace_endpoint_max_q_delta": trace["endpoint_max_q_delta"],
        "trace_endpoint_max_v_delta": trace["endpoint_max_v_delta"],
    }


def compile_receipt(*, stdout: Path = STDOUT, stderr: Path = STDERR) -> dict[str, Any]:
    stdout = Path(stdout)
    stderr = Path(stderr)
    metrics = _metrics(stdout)
    _read_regular(stderr, "native stderr")
    _require(metrics["device"] == "Apple M4 Pro", "native device changed")
    _require(metrics["core_bodies"] == 157, "full source body count changed")
    _require(metrics["muscle_step_seconds"] == 0.0000125 and
             metrics["muscle_step_count"] == 64 and
             metrics["persistent_completed_steps"] == 64,
             "canonical 12.5 microsecond horizon changed")
    _require(metrics["compiled_stand_recruited_muscles"] == 416 and
             metrics["muscle_activation_scope"] == "all_source_muscles",
             "source muscle population changed")
    _require(metrics["persistent_root_assistance"] == "none" and
             metrics["persistent_max_penetration_m"] == 0.0 and
             metrics["stand_deterministic_replay"] == "bitwise",
             "bounded replay invariants did not close")
    _require(metrics["source_support_active_contacts"] == 6 and
             metrics["source_support_witnesses"] == 10,
             "source support witness set changed")
    _require(metrics["dynamic_force_audit_rows"] == 128,
             "complete dynamic force audit was not emitted")
    scaled_receipt = json.loads(_read_regular(SCALED_INPUT_RECEIPT, "scaled input receipt"))
    scaling_receipt = json.loads(_read_regular(SCALING_RECEIPT, "scaling receipt"))
    package_receipt = json.loads(_read_regular(SOURCE_PACKAGE_RECEIPT, "source package receipt"))
    _require(scaled_receipt.get("output", {}).get("sha256") == _sha(SCALED_INPUT),
             "scaled input receipt does not match binary")
    _require(package_receipt.get("fresh_public_build", {}).get("binary_sha256") == NATIVE_SOURCE["binary_sha256"],
             "native public binary identity changed")
    return {
        "schema": SCHEMA,
        "status": "partial",
        "subject": "one adult male source package",
        "source": NATIVE_SOURCE,
        "inputs": {
            "scaled_rigid": {"path": str(SCALED_INPUT.relative_to(ROOT)), "bytes": SCALED_INPUT.stat().st_size,
                              "sha256": _sha(SCALED_INPUT), "receipt_sha256": _sha(SCALED_INPUT_RECEIPT)},
            "mass_scaling_candidate": {"path": str(SCALING_RECEIPT.relative_to(ROOT)),
                                        "sha256": _sha(SCALING_RECEIPT),
                                        "schema": scaling_receipt.get("schema")},
            "muscle": package_receipt["inputs"]["muscle"],
            "tendon": package_receipt["inputs"]["tendon"],
            "support_contact": package_receipt["inputs"]["support_contact"],
            "joint_equalities": package_receipt["inputs"]["joint_equalities"],
        },
        "native_capture": {
            "stdout": {"path": str(stdout.relative_to(ROOT)), "bytes": stdout.stat().st_size, "sha256": _sha(stdout)},
            "stderr": {"path": str(stderr.relative_to(ROOT)), "bytes": stderr.stat().st_size, "sha256": _sha(stderr)},
        },
        "command": {
            "timestep_seconds": metrics["muscle_step_seconds"],
            "step_count": metrics["muscle_step_count"],
            "muscle_activation": metrics["muscle_activation"],
            "persistent_metal_stand": True,
            "persistent_source_passive_joint_tissue": True,
            "persistent_stand_trace": True,
            "stand_contact_iterations": 64,
            "stand_deterministic_replay": True,
            "root_assistance": "none",
        },
        "results": metrics,
        "qualification": {
            "scaled_binary_consumed_by_native_runtime": True,
            "bounded_12p5_us_m4_replay": True,
            "bitwise_replay": True,
            "zero_penetration": True,
            "scalar_mass_target_input": True,
            "segment_composition_calibration": False,
            "geometry_calibration": False,
            "inertia_calibration": False,
            "full_generalized_force_convergence": False,
            "anatomical_support_loading": False,
            "activation_calibration": False,
            "organ_blood_tissue_fat_muscle_ownership": False,
            "sustained_standing": False,
            "perturbation_recovery": False,
            "walking": False,
            "subject_prediction_validation": False,
        },
        "blockers": [
            {"id": "segment_composition_geometry_inertia", "status": "open",
             "reason": "The input applies uniform mass-only scaling with fixed geometry and inferred diagonal inertia."},
            {"id": "full_generalized_force_convergence", "status": "open",
             "reason": "The native result is a 64-step bounded replay with a 0.03299692183 N dynamic audit residual; it is not temporal convergence."},
            {"id": "anatomical_support_loading", "status": "open",
             "reason": "The replay consumes source support witnesses, not calibrated plantar anatomy, friction, or load transfer."},
            {"id": "organ_blood_tissue_fat_muscle_ownership", "status": "open",
             "reason": "The native visual result reports zero bone, soft-tissue, organ, vessel, and skin surface pixels for this source route."},
            {"id": "standing_recovery_walking", "status": "open",
             "reason": "No sustained standing, perturbation recovery, or walking horizon was executed."},
        ],
        "boundary": (
            "This receipt proves that the immutable subject-scaled NHRIGID2 binary "
            "is consumed by the public Apple M4 Pro native probe with the exact "
            "muscle/tendon/contact/equality tuple. It records a bounded 64-step, "
            "12.5 us, bitwise replay with zero penetration and no root assistance. "
            "It does not qualify segment composition, geometry or inertia, full "
            "generalized force convergence, anatomical support/loading, activation "
            "calibration, organ/blood/tissue/fat/muscle ownership, sustained standing, "
            "recovery, walking, or subject prediction."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "receipt output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "receipt output is immutable")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, default=STDOUT)
    parser.add_argument("--stderr", type=Path, default=STDERR)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_receipt(stdout=arguments.stdout, stderr=arguments.stderr)
    receipt_sha = _immutable_write(arguments.receipt.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "receipt": str(arguments.receipt.resolve()), "sha256": receipt_sha}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"subject-scaled native requalification: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

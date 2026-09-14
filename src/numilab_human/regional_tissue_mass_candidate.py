"""Admit the bounded native regional tissue-mass partition as a candidate.

The current native tissue artifact is a source-bound costal partition, not a
whole-human mass model.  This compiler joins the immutable binding, cooked
mass compilation, and the current Apple M4 Pro requalification.  It checks
mass conservation, frame-rebase evidence, and payload identity while keeping
production ownership, calibration, fat, skeletal-muscle volume, loaded
thorax mechanics, and behavior closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.regional-tissue-mass-candidate.v1"
PROFILE = ROOT / "config/regional-tissue-mass-candidate.v1.json"


class RegionalTissueMassError(HumanImportError):
    """A regional tissue mass candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegionalTissueMassError("regional tissue mass candidate: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} is not positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result >= 0.0, f"{label} is negative")
    return result


def _sha256(path: Path, label: str) -> str:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    digest = _sha256(path, label)
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise RegionalTissueMassError(f"cannot read {label}") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, digest


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _safe_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
             and ".." not in Path(value).parts and "\\" not in value,
             f"{label} path is unsafe")
    return value


def _vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} is not length three")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _tensor(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 9, f"{label} is not length nine")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_json(path, "regional tissue mass profile")
    _require(path.read_bytes() == canonical(profile) + b"\n",
             "regional tissue mass profile is not canonical")
    required = {"schema", "id", "subject", "mass_compilation", "binding",
                "binding_payload", "native_requalification", "expected", "boundary"}
    _require(set(profile) == required, "regional tissue mass profile fields differ")
    _require(profile["schema"] == "numi.human.regional-tissue-mass-candidate.v1",
             "unsupported regional tissue mass profile schema")
    _require(profile["id"] == "costal_native_mass_partition",
             "unsupported regional tissue mass profile")
    _require(profile["subject"] == "one adult male source package",
             "regional tissue mass subject changed")
    for key in ("mass_compilation", "binding", "binding_payload", "native_requalification"):
        _safe_relative(profile[key], key)
    _require(isinstance(profile["expected"], dict), "regional tissue mass expected values missing")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "regional tissue mass boundary is missing")
    return profile, profile_sha


def _check_expected(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, bool):
        _require(value is expected, f"{label} disagrees with the pinned profile")
    elif isinstance(expected, int):
        _require(type(value) is int and value == expected,
                 f"{label} disagrees with the pinned profile")
    else:
        actual = _finite(value, label)
        target = _finite(expected, f"expected {label}")
        tolerance = max(1e-12, abs(target) * 1e-12)
        _require(abs(actual - target) <= tolerance,
                 f"{label} disagrees with the pinned profile")


def _load_binding(path: Path, payload_path: Path) -> tuple[dict[str, Any], str, str]:
    binding, binding_sha = _read_json(path, "costal tissue binding")
    _require(binding.get("schema") == "HumanPack.costal-tissue-binding.v1",
             "unsupported costal tissue binding schema")
    payload = binding.get("payload")
    _require(isinstance(payload, dict), "costal tissue binding payload is missing")
    _require(payload.get("file") == payload_path.name,
             "costal tissue payload filename disagrees")
    payload_sha = _sha256(payload_path, "costal tissue payload")
    _require(payload.get("sha256") == payload_sha,
             "costal tissue payload hash disagrees with binding")
    _require(type(payload.get("bytes")) is int and payload["bytes"] == payload_path.stat().st_size,
             "costal tissue payload size disagrees with binding")
    _require(payload_path.read_bytes()[:8] == b"NHTBIND1", "costal tissue payload magic changed")
    mass_ownership = binding.get("mass_ownership")
    _require(isinstance(mass_ownership, dict)
             and mass_ownership.get("native_partition_required") is True
             and mass_ownership.get("frame_rebase_required") is True,
             "costal tissue mass ownership contract changed")
    _require(mass_ownership.get("donor_assumption")
             == "costal_volume_is_included_in_source_gross_torso_inertia",
             "costal tissue donor assumption changed")
    regions = binding.get("regions")
    _require(isinstance(regions, list) and len(regions) == 14,
             "costal tissue source region count changed")
    members: set[str] = set()
    for row in regions:
        _require(isinstance(row, dict), "costal tissue source region is malformed")
        member = row.get("member_id")
        _require(isinstance(member, str) and member and member not in members,
                 "costal tissue member identity is missing or duplicated")
        members.add(member)
        for key in ("donor_body", "rib_body", "sternal_body"):
            _require(type(row.get(key)) is int and row[key] == 20,
                     f"costal tissue {key} changed")
    qualification = binding.get("qualification")
    _require(isinstance(qualification, dict)
             and qualification.get("production_owner_fraction") == 0.0
             and qualification.get("accepted_root_integration") is False
             and qualification.get("rib_sternum_relative_articulation") is False
             and qualification.get("tissue_calibration") is False,
             "costal tissue binding has been promoted")
    return binding, binding_sha, payload_sha


def _load_compilation(path: Path, expected: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document, document_sha = _read_json(path, "native tissue mass compilation")
    _require(document.get("schema") == "numi.human.tissue-mass-compilation.v1",
             "unsupported native tissue mass compilation schema")
    _require(document.get("status") == "compiled_requires_v5_runtime_admission",
             "native tissue mass compilation status changed")
    for key in ("matter_world_fingerprint", "cooked_nodes", "cooked_tetrahedra", "attachments",
                "negative_admission_cases"):
        _check_expected(document.get(key), expected[key], key)
    _require(document.get("metal_executed") is True
             and document.get("metal_device") == "Apple M4 Pro",
             "native tissue mass compilation did not execute on the pinned device")
    _check_expected(document.get("metal_replay_cases"), expected["native_replay_cases"],
                    "metal_replay_cases")
    _check_expected(document.get("rebase_pose_cases"), expected["rebase_pose_cases"],
                    "rebase_pose_cases")
    _require(document.get("whole_body_dynamic_mass_matrix_qualified") is False,
             "whole-body dynamic mass matrix was unexpectedly promoted")
    _check_expected(document.get("production_owner_fraction"), expected["production_owner_fraction"],
                    "production_owner_fraction")
    _require(document.get("rib_sternum_articulation_qualified") is False,
             "rib/sternum articulation was unexpectedly promoted")
    partitions = document.get("partitions")
    _require(isinstance(partitions, list) and len(partitions) == 1,
             "native tissue mass compilation must contain one regional partition")
    partition = partitions[0]
    _require(isinstance(partition, dict), "native tissue mass partition is malformed")
    _check_expected(partition.get("donor_body"), expected["donor_body"], "donor_body")
    _check_expected(partition.get("node_count"), expected["cooked_nodes"], "partition node_count")
    source_mass = _positive(partition.get("source_mass_kg"), "source mass")
    tissue_mass = _positive(partition.get("tissue_mass_kg"), "tissue mass")
    remaining_mass = _nonnegative(partition.get("remaining_mass_kg"), "remaining mass")
    tolerance = _positive(expected["mass_tolerance_kg"], "mass tolerance")
    _require(abs(source_mass - tissue_mass - remaining_mass) <= tolerance,
             "native tissue mass partition is not conservative")
    for key in ("remaining_com_offset_m", "tissue_first_moment_kg_m"):
        _vector(partition.get(key), key)
    _tensor(partition.get("tissue_second_moment_kg_m2"), "tissue second moment")
    packed_error = _nonnegative(partition.get("packed_moment_relative_error"),
                                "packed moment relative error")
    _require(packed_error <= 1e-5, "packed tissue moment error exceeds the candidate gate")
    for key in ("maximum_attachment_rest_error_m", "maximum_rebase_point_error_m",
                "maximum_rebase_velocity_error_m_s", "maximum_rebase_jacobian_error",
                "maximum_metal_point_error_m", "maximum_metal_jacobian_error",
                "maximum_metal_donor_mass_matrix_scaled_error"):
        _nonnegative(document.get(key), key)
    _check_expected(tissue_mass, expected["tissue_mass_kg"], "tissue_mass_kg")
    _check_expected(source_mass, expected["source_mass_kg"], "source_mass_kg")
    _check_expected(remaining_mass, expected["remaining_mass_kg"], "remaining_mass_kg")
    return document, document_sha, partition


def _load_requalification(path: Path, payload_sha: str, partition: dict[str, Any],
                          expected: dict[str, Any]) -> tuple[dict[str, Any], str]:
    document, document_sha = _read_json(path, "native tissue requalification")
    _require(document.get("schema") == "HumanPack.costal-tissue-native-requalification.v1",
             "unsupported native tissue requalification schema")
    inputs = document.get("inputs")
    _require(isinstance(inputs, dict) and inputs.get("binding_sha256") == payload_sha,
             "native requalification is not bound to the current tissue payload")
    native = document.get("native")
    _require(isinstance(native, dict) and native.get("device") == "Apple M4 Pro",
             "native requalification device changed")
    qualification = document.get("qualification")
    _require(isinstance(qualification, dict)
             and qualification.get("native_metal_replay") == "pass"
             and qualification.get("registered_costal_tissue_mass_and_rebase") == "pass"
             and qualification.get("experimental_material_calibration") is False
             and qualification.get("loaded_thorax_convergence") is False
             and qualification.get("whole_body_dynamic_mass_matrix") is False
             and qualification.get("standing_recovery_walking") is False,
             "native requalification boundary changed")
    results = document.get("results")
    _require(isinstance(results, dict) and results.get("mass_partition") == "pass",
             "native requalification mass partition did not pass")
    for key in ("cooked_nodes", "cooked_tetrahedra", "attachments", "rebase_pose_cases"):
        _check_expected(results.get(key), expected[key], f"requalification {key}")
    _check_expected(results.get("metal_replay_cases"), expected["native_replay_cases"],
                    "requalification metal_replay_cases")
    _check_expected(results.get("tissue_mass_kg"), partition["tissue_mass_kg"],
                    "requalification tissue mass")
    _check_expected(results.get("residual_torso_mass_kg"), partition["remaining_mass_kg"],
                    "requalification residual torso mass")
    return document, document_sha


def compile_candidate(*, profile: Path = PROFILE, mass_compilation: Path | None = None,
                      binding: Path | None = None, binding_payload: Path | None = None,
                      native_requalification: Path | None = None) -> dict[str, Any]:
    profile_path = Path(profile)
    profile_doc, profile_sha = _profile(profile_path)
    expected = profile_doc["expected"]
    paths = {
        "mass_compilation": Path(mass_compilation or ROOT / profile_doc["mass_compilation"]),
        "binding": Path(binding or ROOT / profile_doc["binding"]),
        "binding_payload": Path(binding_payload or ROOT / profile_doc["binding_payload"]),
        "native_requalification": Path(native_requalification or ROOT / profile_doc["native_requalification"]),
    }
    compilation, compilation_sha, partition = _load_compilation(paths["mass_compilation"], expected)
    binding_doc, binding_sha, payload_sha = _load_binding(paths["binding"], paths["binding_payload"])
    requalification, requalification_sha = _load_requalification(
        paths["native_requalification"], payload_sha, partition, expected,
    )
    source_mass = partition["source_mass_kg"]
    tissue_mass = partition["tissue_mass_kg"]
    remaining_mass = partition["remaining_mass_kg"]
    conservation_residual = source_mass - tissue_mass - remaining_mass
    identity = {
        "profile_sha256": profile_sha,
        "mass_compilation_sha256": compilation_sha,
        "binding_sha256": binding_sha,
        "binding_payload_sha256": payload_sha,
        "native_requalification_sha256": requalification_sha,
        "matter_world_fingerprint": compilation["matter_world_fingerprint"],
        "partition": partition,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.regional-tissue-mass-candidate.1",
        "status": "partial",
        "subject": profile_doc["subject"],
        "source": {
            "profile": _relative(profile_path),
            "profile_sha256": profile_sha,
            "mass_compilation": _relative(paths["mass_compilation"]),
            "mass_compilation_sha256": compilation_sha,
            "binding": _relative(paths["binding"]),
            "binding_sha256": binding_sha,
            "binding_payload": _relative(paths["binding_payload"]),
            "binding_payload_sha256": payload_sha,
            "native_requalification": _relative(paths["native_requalification"]),
            "native_requalification_sha256": requalification_sha,
        },
        "identity_sha256": hashlib.sha256(canonical(identity)).hexdigest(),
        "region": {
            "id": "costal_tissue_body20",
            "source_member_count": len(binding_doc["regions"]),
            "donor_body": partition["donor_body"],
            "cooked_nodes": partition["node_count"],
            "cooked_tetrahedra": compilation["cooked_tetrahedra"],
            "attachments": compilation["attachments"],
            "source_mass_kg": source_mass,
            "tissue_mass_kg": tissue_mass,
            "remaining_mass_kg": remaining_mass,
            "tissue_fraction_of_donor": tissue_mass / source_mass,
            "conservation_residual_kg": conservation_residual,
            "remaining_com_offset_m": partition["remaining_com_offset_m"],
            "tissue_first_moment_kg_m": partition["tissue_first_moment_kg_m"],
            "tissue_second_moment_kg_m2": partition["tissue_second_moment_kg_m2"],
            "packed_moment_relative_error": partition["packed_moment_relative_error"],
            "production_owner_fraction": compilation["production_owner_fraction"],
        },
        "native_evidence": {
            "metal_device": compilation["metal_device"],
            "metal_replay_cases": compilation["metal_replay_cases"],
            "rebase_pose_cases": compilation["rebase_pose_cases"],
            "maximum_attachment_rest_error_m": compilation["maximum_attachment_rest_error_m"],
            "maximum_rebase_point_error_m": compilation["maximum_rebase_point_error_m"],
            "maximum_rebase_velocity_error_m_s": compilation["maximum_rebase_velocity_error_m_s"],
            "maximum_rebase_jacobian_error": compilation["maximum_rebase_jacobian_error"],
            "maximum_metal_point_error_m": compilation["maximum_metal_point_error_m"],
            "maximum_metal_jacobian_error": compilation["maximum_metal_jacobian_error"],
            "maximum_metal_donor_mass_matrix_scaled_error": compilation[
                "maximum_metal_donor_mass_matrix_scaled_error"
            ],
            "negative_admission_cases": compilation["negative_admission_cases"],
        },
        "ownership": {
            "regional_tissue_mass_candidate": True,
            "production_mechanical_mass_owner": False,
            "whole_body_dynamic_mass_matrix_owner": False,
            "fat_mass_owner": False,
            "skeletal_muscle_tissue_partition_owner": False,
            "skin_mass_owner": False,
            "blood_mass_owner": False,
        },
        "qualification": {
            "source_binding_identity_bound": True,
            "native_cooked_mass_partition": True,
            "mass_conservation": abs(conservation_residual) <= expected["mass_tolerance_kg"],
            "com_frame_rebase": True,
            "native_metal_replay": True,
            "regional_tissue_mass_candidate": True,
            "production_mechanical_mass_owner": False,
            "whole_body_dynamic_mass_matrix": False,
            "loaded_thorax_convergence": False,
            "independent_rib_sternum_articulation": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "fat_geometry_and_mass": False,
            "skeletal_muscle_tissue_partition": False,
            "material_calibration": False,
            "subject_calibration": False,
            "standing_recovery_walking": False,
        },
        "boundary": profile_doc["boundary"],
    }
    canonical(result)
    return result


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(profile=arguments.profile)
    output = arguments.output.resolve()
    receipt_sha = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": receipt_sha,
        "status": result["status"],
        "donor_body": result["region"]["donor_body"],
        "tissue_mass_kg": result["region"]["tissue_mass_kg"],
        "production_mechanical_mass_owner": False,
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"regional-tissue-mass: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

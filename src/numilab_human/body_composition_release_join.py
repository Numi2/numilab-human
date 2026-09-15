"""Join the current one-male organ, blood, tissue, fat, muscle and runtime graph.

This is an evidence-graph release join.  It makes the cross-domain identity
and ownership boundaries machine-checkable without turning source geometry,
zeroth-moment transport, or a bounded native replay into physical owners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/body-composition-release-join.v1.json"
EXTENSION = ROOT / "Docs/media/body-composition-extension-join-20260915/receipt-v1.json"
BLOOD_TISSUE = ROOT / "Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json"
FAT = ROOT / "Docs/media/fat-source-absence-candidate-20260915/receipt-v1.json"
NATIVE = ROOT / "Docs/media/native-subject-scaled-runtime-20260915/receipt-v1.json"
SCHEMA = "HumanPack.body-composition-release-join.v1"


class ReleaseJoinError(HumanImportError):
    """The cross-domain one-male release graph cannot be joined."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseJoinError("body composition release join: " + message)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise ReleaseJoinError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _input(path: Path, document: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(document.get("schema")),
            "file_sha256": digest}


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "release profile")
    _require(profile.get("schema") == "numi.human.body-composition-release-join.v1",
             "unsupported release profile schema")
    _require(profile.get("id") == "one_adult_male_cross_domain_release_graph",
             "unsupported release profile")
    for key in ("extension_receipt", "blood_tissue_receipt", "fat_receipt", "native_receipt"):
        value = profile.get(key)
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    _require(isinstance(profile.get("boundary"), str) and profile["boundary"].strip(),
             "release boundary is missing")
    return profile, digest


def compile_candidate(*, profile: Path = PROFILE, extension: Path = EXTENSION,
                      blood_tissue: Path = BLOOD_TISSUE, fat: Path = FAT,
                      native: Path = NATIVE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    paths = {
        "extension": Path(extension), "blood_tissue": Path(blood_tissue),
        "fat": Path(fat), "native": Path(native),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for key, path in paths.items():
        documents[key], hashes[key] = _read(path, f"{key} receipt")

    extension_doc = documents["extension"]
    _require(extension_doc.get("schema") == "HumanPack.body-composition-extension-join.v1"
             and extension_doc.get("status") == "partial",
             "composition extension receipt changed")
    _require(extension_doc.get("subject") == "one adult male source package",
             "composition extension subject changed")
    extension_qualification = extension_doc.get("qualification", {})
    _require(extension_qualification.get("cross_extension_source_hashes_closed") is True
             and extension_qualification.get("physical_owner_count") == 0
             and extension_qualification.get("muscle_route_volume_incidence_bound") is True,
             "composition extension owner boundary changed")
    _require(extension_qualification.get("skeletal_muscle_tissue_mass") is False
             and extension_qualification.get("fat_geometry_and_mass") is False,
             "composition extension promoted a tissue owner")

    blood_doc = documents["blood_tissue"]
    _require(blood_doc.get("schema") == "HumanPack.organ-blood-mass-transfer-candidate.v1"
             and blood_doc.get("status") == "partial",
             "blood/tissue receipt changed")
    blood_qualification = blood_doc.get("qualification", {})
    _require(blood_qualification.get("regional_blood_transport_bound") is True
             and blood_qualification.get("regional_tissue_candidate_mass_bound") is True
             and blood_qualification.get("blood_tissue_zeroth_moment_mass_transfer") is True
             and blood_qualification.get("mass_and_volume_conservation") is True
             and blood_qualification.get("accepted_step_rollback") is True,
             "blood/tissue conservation handoff is incomplete")
    for key in ("anatomical_vessel_lumen", "mechanical_blood_mass_owner",
                "mechanical_tissue_mass_owner", "organ_mechanics",
                "material_density_calibrated", "subject_calibration"):
        _require(blood_qualification.get(key) is False,
                 f"blood/tissue receipt promoted {key}")

    fat_doc = documents["fat"]
    _require(fat_doc.get("schema") == "HumanPack.fat-source-absence-candidate.v1"
             and fat_doc.get("status") == "partial"
             and fat_doc.get("subject") == "one adult male source package",
             "fat absence receipt changed")
    fat_qualification = fat_doc.get("qualification", {})
    _require(fat_qualification.get("fat_source_absence_bound") is True
             and fat_qualification.get("fat_physical_volume_owner") is False
             and fat_qualification.get("fat_mechanical_mass_owner") is False,
             "fat absence boundary changed")
    fat_counts = fat_doc.get("counts", {})
    _require(fat_counts.get("fat_surface_count") == 0
             and fat_counts.get("fat_volume_candidate_count") == 0
             and fat_counts.get("fat_mass_candidate_count") == 0,
             "fat source absence counts changed")

    native_doc = documents["native"]
    _require(native_doc.get("schema") == "HumanPack.native-subject-scaled-runtime-requalification.v1"
             and native_doc.get("status") == "partial"
             and native_doc.get("subject") == "one adult male source package",
             "native subject receipt changed")
    native_qualification = native_doc.get("qualification", {})
    _require(native_qualification.get("scaled_binary_consumed_by_native_runtime") is True
             and native_qualification.get("bounded_12p5_us_m4_replay") is True
             and native_qualification.get("bitwise_replay") is True,
             "native subject replay is not bound")
    for key in ("full_generalized_force_convergence", "anatomical_support_loading",
                "activation_calibration", "organ_blood_tissue_fat_muscle_ownership",
                "sustained_standing", "perturbation_recovery", "walking",
                "subject_prediction_validation"):
        _require(native_qualification.get(key) is False,
                 f"native receipt promoted {key}")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.body-composition-release-join.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            key: _input(paths[key], documents[key], hashes[key]) for key in paths
        } | {"profile": {"path": _relative(Path(profile)),
                         "schema": profile_doc["schema"], "file_sha256": profile_sha}},
        "domain_counts": {
            "muscle_source_routes": extension_doc["extensions"]["muscle_route_volume"]["source_route_count"],
            "muscle_closed_geometry_candidates": extension_doc["extensions"]["muscle_route_volume"]["closed_geometry_volume_owner_count"],
            "muscle_routes_without_surface_binding": extension_doc["extensions"]["muscle_route_volume"]["routes_without_surface_binding"],
            "organ_blood_tissue_beds": blood_doc["counts"]["bed_count"],
            "organ_blood_source_owners": blood_doc["counts"]["source_blood_owner_count"],
            "organ_blood_source_members": blood_doc["counts"]["source_members_bound"],
            "fat_surfaces": fat_counts["fat_surface_count"],
            "fat_volume_candidates": fat_counts["fat_volume_candidate_count"],
            "fat_mass_candidates": fat_counts["fat_mass_candidate_count"],
            "native_recruited_muscles": native_doc["results"]["compiled_stand_recruited_muscles"],
            "native_active_support_contacts": native_doc["results"]["compiled_stand_active_support_contacts"],
        },
        "ownership": {
            "physical_owner_count": 0,
            "organ_physical_volume_owner": False,
            "mechanical_blood_mass_owner": False,
            "mechanical_tissue_mass_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "fat_mechanical_mass_owner": False,
            "whole_body_dynamic_mass_matrix_owner": False,
        },
        "qualification": {
            "one_adult_male_source_graph_bound": True,
            "organ_blood_tissue_zeroth_moment_bound": True,
            "blood_tissue_conservation_and_rollback_bound": True,
            "muscle_route_volume_identity_bound": True,
            "fat_source_absence_bound": True,
            "subject_scaled_native_replay_bound": True,
            "cross_domain_owner_nonduplication": True,
            "organ_mechanics": False,
            "anatomical_vessel_lumen": False,
            "physical_tissue_volume": False,
            "mechanical_blood_mass": False,
            "mechanical_tissue_mass": False,
            "skeletal_muscle_mass": False,
            "fat_geometry_and_mass": False,
            "activation_force_transfer": False,
            "activation_calibration": False,
            "material_calibration": False,
            "subject_calibration": False,
            "full_generalized_force_convergence": False,
            "standing": False,
            "recovery": False,
            "walking": False,
            "integrated_human_qualification": False,
        },
        "blockers": [
            {"id": "anatomical-organ-and-vessel-mechanics", "status": "open",
             "reason": "The joined organ/blood graph has no anatomical lumen, capillary geometry, organ mechanics, or calibrated tissue exchange."},
            {"id": "physical-soft-tissue-ownership", "status": "open",
             "reason": "Muscle volume, tissue candidates, blood mass, fat, and skin remain nonphysical source-bound candidates with zero production owners."},
            {"id": "subject-material-activation-calibration", "status": "open",
             "reason": "Density, constitutive materials, activation/force transfer, and held-out subject calibration are unresolved."},
            {"id": "whole-body-behavior", "status": "open",
             "reason": "The native handoff is bounded replay only; generalized convergence, sustained standing, recovery, and walking are unqualified."},
        ],
        "boundary": profile_doc["boundary"],
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument("--blood-tissue", type=Path, default=BLOOD_TISSUE)
    parser.add_argument("--fat", type=Path, default=FAT)
    parser.add_argument("--native", type=Path, default=NATIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(profile=arguments.profile, extension=arguments.extension,
                               blood_tissue=arguments.blood_tissue, fat=arguments.fat,
                               native=arguments.native)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (ReleaseJoinError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"body-composition-release-join: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

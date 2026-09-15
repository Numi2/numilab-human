"""Bind organ/blood transfer and regional tissue mass to the Human bridge.

The v7 bridge extension binds the route-level muscle mass candidate.  This
receipt adds the existing exact-clock blood/tissue transfer and the native
costal tissue partition to the same one-male source graph.  It is a provenance
and conservation hand-off; it does not assign anatomical lumen, tissue
mechanical mass, organ mechanics, materials, or subject calibration.
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
PROFILE = ROOT / "config/human-source-evidence-tissue-extension.v1.json"
BASE = ROOT / "Docs/media/human-source-evidence-bridge-20260915/receipt-v7.json"
BLOOD = ROOT / "Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json"
ORGAN_MASS = ROOT / "Docs/media/tissue-mass-candidate-20260914/receipt-v2.json"
REGIONAL_TISSUE = ROOT / (
    "Docs/media/regional-tissue-mass-candidate-20260914/receipt-v1.json"
)
SCHEMA = "HumanPack.human-source-evidence-tissue-extension.v1"


class TissueExtensionError(HumanImportError):
    """The organ/blood/tissue receipts cannot be joined to the bridge."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TissueExtensionError("Human source-evidence tissue extension: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise TissueExtensionError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _input(path: Path, value: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(value.get("schema")),
            "file_sha256": digest}


def _canonical_digest(value: dict[str, Any]) -> str:
    """Return the no-newline canonical digest used by legacy source receipts."""
    return hashlib.sha256(canonical(value)).hexdigest()


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "tissue extension profile", canonical_required=True)
    required = {
        "schema", "id", "base_extension", "blood_transfer", "organ_mass", "regional_tissue",
        "expected_clock_nanoseconds", "expected_bed_count", "expected_blood_owner_count",
        "expected_source_member_count", "expected_tissue_mass_kg", "expected_region_count",
        "expected_cooked_nodes", "expected_cooked_tetrahedra", "boundary",
    }
    _require(set(profile) == required, "tissue extension profile fields differ")
    _require(profile["schema"] == "numi.human.human-source-evidence-tissue-extension.v1",
             "unsupported tissue extension profile schema")
    _require(profile["id"] == "bind_blood_transfer_to_regional_tissue_and_muscle_bridge",
             "unsupported tissue extension profile")
    for key in ("base_extension", "blood_transfer", "organ_mass", "regional_tissue"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    for key, expected in (
        ("expected_clock_nanoseconds", 12500),
        ("expected_bed_count", 7),
        ("expected_blood_owner_count", 6),
        ("expected_source_member_count", 14),
        ("expected_region_count", 14),
        ("expected_cooked_nodes", 13516),
        ("expected_cooked_tetrahedra", 46278),
    ):
        _require(profile[key] == expected, f"{key} differs from the source contract")
    _require(type(profile["expected_tissue_mass_kg"]) in (int, float)
             and profile["expected_tissue_mass_kg"] > 0.0,
             "expected tissue mass is invalid")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "tissue extension boundary is missing")
    return profile, digest


def compile_candidate(*, base: Path = BASE, blood_transfer: Path = BLOOD,
                      organ_mass: Path = ORGAN_MASS,
                      regional_tissue: Path = REGIONAL_TISSUE,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    base_path, blood_path, organ_mass_path, tissue_path = (
        Path(base), Path(blood_transfer), Path(organ_mass), Path(regional_tissue)
    )
    base_doc, base_sha = _read(base_path, "muscle bridge extension")
    blood_doc, blood_sha = _read(blood_path, "blood-transfer receipt")
    organ_mass_doc, organ_mass_sha = _read(organ_mass_path, "organ tissue-mass receipt")
    tissue_doc, tissue_sha = _read(tissue_path, "regional tissue receipt")

    _require(base_doc.get("schema") == "HumanPack.human-source-evidence-bridge-extension.v1"
             and base_doc.get("status") == "partial", "base bridge extension changed")
    base_graph = base_doc.get("source_graph", {})
    base_qualification = base_doc.get("qualification", {})
    base_muscle = base_doc.get("domains", {}).get("muscle", {})
    _require(isinstance(base_graph, dict) and isinstance(base_qualification, dict)
             and isinstance(base_muscle, dict), "base bridge extension is incomplete")
    _require(base_graph.get("physical_owner_count") == 0
             and base_graph.get("cross_domain_hashes_closed") is True
             and base_qualification.get("route_mass_partition_bound") is True
             and base_qualification.get("integrated_human_qualification") is False,
             "base bridge extension boundary changed")

    _require(blood_doc.get("schema") == "HumanPack.organ-blood-mass-transfer-candidate.v1"
             and blood_doc.get("status") == "partial", "blood-transfer receipt changed")
    blood_counts = blood_doc.get("counts", {})
    blood_clock = blood_doc.get("clock", {})
    blood_conservation = blood_doc.get("conservation", {})
    blood_qualification = blood_doc.get("qualification", {})
    blood_source = blood_doc.get("source", {})
    _require(blood_counts.get("bed_count") == profile_doc["expected_bed_count"]
             and blood_counts.get("region_count") == profile_doc["expected_bed_count"]
             and blood_counts.get("source_blood_owner_count") == profile_doc["expected_blood_owner_count"]
             and blood_counts.get("source_members_bound") == 329
             and blood_clock.get("nanoseconds") == profile_doc["expected_clock_nanoseconds"]
             and blood_doc.get("accepted_steps") == 511
             and blood_doc.get("rejected_steps") == 1
             and blood_conservation.get("mass_conserved") is True
             and blood_conservation.get("volume_conserved") is True,
             "blood-transfer counts, clock or conservation changed")
    for key in ("regional_blood_transport_bound", "regional_tissue_candidate_mass_bound",
                "blood_tissue_zeroth_moment_mass_transfer", "bidirectional_transfer_schedule",
                "accepted_step_rollback", "mass_and_volume_conservation"):
        _require(blood_qualification.get(key) is True, f"blood-transfer lacks {key}")
    for key in ("anatomical_exchange_owner", "anatomical_vessel_lumen",
                "physical_tissue_volume_owner", "mechanical_blood_mass_owner",
                "mechanical_tissue_mass_owner", "organ_mechanics", "material_density_calibrated",
                "subject_calibration", "standing_walking"):
        _require(blood_qualification.get(key) is False, f"blood-transfer boundary changed for {key}")

    _require(organ_mass_doc.get("schema") == "HumanPack.tissue-mass-composition-candidate.v1"
             and organ_mass_doc.get("status") == "partial", "organ tissue-mass receipt changed")
    organ_counts = organ_mass_doc.get("counts", {})
    organ_qualification = organ_mass_doc.get("qualification", {})
    _require(organ_counts.get("region_count") == 18
             and organ_counts.get("source_members") == 378
             and organ_counts.get("organ_surface_mass_candidates") == 342
             and organ_counts.get("shared_source_members") == 8
             and organ_counts.get("unresolved_members") == 36,
             "organ tissue-mass counts changed")
    for key in ("source_moment_identity_bound", "candidate_mass_closes_against_density",
                "candidate_zeroth_first_second_moments"):
        _require(organ_qualification.get(key) is True, f"organ tissue-mass lacks {key}")
    for key in ("physical_volume_authority", "mechanical_mass_owner_assigned",
                "blood_mass_transfer", "organ_mechanics", "material_calibration",
                "subject_calibration", "standing_walking"):
        _require(organ_qualification.get(key) is False,
                 f"organ tissue-mass boundary changed for {key}")

    _require(tissue_doc.get("schema") == "HumanPack.regional-tissue-mass-candidate.v1"
             and tissue_doc.get("status") == "partial"
             and tissue_doc.get("subject") == "one adult male source package",
             "regional tissue receipt changed")
    tissue_region = tissue_doc.get("region", {})
    tissue_ownership = tissue_doc.get("ownership", {})
    tissue_qualification = tissue_doc.get("qualification", {})
    _require(tissue_region.get("id") == "costal_tissue_body20"
             and tissue_region.get("source_member_count") == profile_doc["expected_source_member_count"]
             and tissue_region.get("cooked_nodes") == profile_doc["expected_cooked_nodes"]
             and tissue_region.get("cooked_tetrahedra") == profile_doc["expected_cooked_tetrahedra"]
             and tissue_region.get("source_mass_kg") > tissue_region.get("tissue_mass_kg") > 0.0
             and abs(float(tissue_region.get("tissue_mass_kg")) -
                     float(profile_doc["expected_tissue_mass_kg"])) <= 1.0e-12
             and tissue_region.get("conservation_residual_kg") is not None,
             "regional tissue partition changed")
    _require(tissue_ownership.get("regional_tissue_mass_candidate") is True
             and tissue_ownership.get("production_mechanical_mass_owner") is False
             and tissue_ownership.get("whole_body_dynamic_mass_matrix_owner") is False
             and tissue_ownership.get("blood_mass_owner") is False,
             "regional tissue ownership boundary changed")
    for key in ("source_binding_identity_bound", "native_cooked_mass_partition",
                "mass_conservation", "com_frame_rebase", "native_metal_replay",
                "regional_tissue_mass_candidate"):
        _require(tissue_qualification.get(key) is True, f"regional tissue lacks {key}")
    for key in ("production_mechanical_mass_owner", "whole_body_dynamic_mass_matrix",
                "loaded_thorax_convergence", "activation_calibration", "blood_mass_transfer",
                "fat_geometry_and_mass", "skeletal_muscle_tissue_partition",
                "material_calibration", "subject_calibration", "standing_recovery_walking"):
        _require(tissue_qualification.get(key) is False,
                 f"regional tissue boundary changed for {key}")

    _require(blood_source.get("tissue_mass_candidate_sha256") == _canonical_digest(organ_mass_doc)
             and blood_source.get("tissue_mass_schema") == organ_mass_doc.get("schema"),
             "blood-transfer and organ tissue-mass hashes diverge")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.human-source-evidence-tissue-extension.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "base_extension": _input(base_path, base_doc, base_sha),
            "blood_transfer": _input(blood_path, blood_doc, blood_sha),
            "organ_mass": _input(organ_mass_path, organ_mass_doc, organ_mass_sha),
            "regional_tissue": _input(tissue_path, tissue_doc, tissue_sha),
            "profile": {"path": _relative(Path(profile)), "schema": profile_doc["schema"],
                        "file_sha256": profile_sha},
        },
        "base_extension": {
            "schema": base_doc["schema"],
            "sha256": base_sha,
            "route_mass_partition_bound": True,
            "physical_owner_count": 0,
        },
        "domains": {
            "muscle": {
                "source_route_count": base_muscle["source_route_count"],
                "closed_surface_count": base_muscle["closed_surface_count"],
                "route_incidence_count": base_muscle["route_incidence_count"],
                "candidate_mass_kg": base_muscle["candidate_mass_kg"],
                "physical_volume_owner": False,
                "mechanical_mass_owner": False,
                "activation_force_transfer": False,
            },
            "organ_blood": {
                "clock_nanoseconds": blood_clock["nanoseconds"],
                "bed_count": blood_counts["bed_count"],
                "source_blood_owner_count": blood_counts["source_blood_owner_count"],
                "source_members_bound": blood_counts["source_members_bound"],
                "accepted_steps": blood_doc["accepted_steps"],
                "rejected_steps": blood_doc["rejected_steps"],
                "mass_conserved": blood_conservation["mass_conserved"],
                "volume_conserved": blood_conservation["volume_conserved"],
                "anatomical_vessel_lumen": False,
                "mechanical_blood_mass_owner": False,
                "mechanical_tissue_mass_owner": False,
            },
            "organ_tissue": {
                "region_count": organ_counts["region_count"],
                "source_member_count": organ_counts["source_members"],
                "organ_surface_mass_candidate_count": organ_counts["organ_surface_mass_candidates"],
                "shared_source_member_count": organ_counts["shared_source_members"],
                "unresolved_member_count": organ_counts["unresolved_members"],
                "candidate_mass_closes_against_density": True,
                "physical_volume_authority": False,
                "mechanical_mass_owner_assigned": False,
            },
            "regional_tissue": {
                "region_id": tissue_region["id"],
                "source_member_count": tissue_region["source_member_count"],
                "cooked_nodes": tissue_region["cooked_nodes"],
                "cooked_tetrahedra": tissue_region["cooked_tetrahedra"],
                "source_mass_kg": tissue_region["source_mass_kg"],
                "tissue_mass_kg": tissue_region["tissue_mass_kg"],
                "conservation_residual_kg": tissue_region["conservation_residual_kg"],
                "regional_tissue_mass_candidate": True,
                "production_mechanical_mass_owner": False,
            },
        },
        "source_graph": {
            "base_extension_sha256": base_sha,
            "blood_transfer_sha256": blood_sha,
            "organ_mass_file_sha256": organ_mass_sha,
            "organ_mass_source_sha256": _canonical_digest(organ_mass_doc),
            "regional_tissue_sha256": tissue_sha,
            "cross_domain_hashes_closed": True,
            "physical_owner_count": 0,
        },
        "qualification": {
            "base_extension_bound": True,
            "route_mass_partition_bound": True,
            "blood_transfer_bound": True,
            "organ_tissue_mass_bound": True,
            "regional_tissue_mass_bound": True,
            "exact_clock_bound": True,
            "mass_and_volume_conservation_bound": True,
            "physical_owner_count": 0,
            "anatomical_vessel_lumen": False,
            "physical_tissue_volume_owner": False,
            "organ_tissue_physical_volume_authority": False,
            "mechanical_blood_mass_owner": False,
            "mechanical_tissue_mass_owner": False,
            "production_mechanical_mass_owner": False,
            "organ_mechanics": False,
            "activation_calibration": False,
            "fat_geometry_and_mass": False,
            "material_calibration": False,
            "subject_calibration": False,
            "force_convergence": False,
            "standing": False,
            "recovery": False,
            "walking": False,
            "integrated_human_qualification": False,
        },
        "blockers": [
            {"id": "anatomical_blood_tissue_mass", "status": "open",
             "reason": "Exact-clock zeroth-moment transfer, the 18-region organ tissue candidate and a native regional tissue partition are bound, but lumen, capillary exchange, physical tissue volume and mechanical mass owners remain absent."},
            {"id": "muscle_mass_and_activation", "status": "open",
             "reason": "The route-level muscle candidate remains bookkeeping-only; physical mass, active force transfer and activation calibration remain absent."},
            {"id": "force_convergence_and_behavior", "status": "open",
             "reason": "The parent bridge still reports unresolved whole-body equilibrium, materials, subject calibration, standing, recovery and walking."},
        ],
        "boundary": profile_doc["boundary"],
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(base=arguments.base, blood_transfer=arguments.blood_transfer,
                               organ_mass=arguments.organ_mass,
                               regional_tissue=arguments.regional_tissue,
                               profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output)}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--blood-transfer", type=Path, default=BLOOD)
    parser.add_argument("--organ-mass", type=Path, default=ORGAN_MASS)
    parser.add_argument("--regional-tissue", type=Path, default=REGIONAL_TISSUE)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (TissueExtensionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"human-source-evidence-tissue-extension: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

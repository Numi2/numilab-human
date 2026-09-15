"""Join the current Human runtime evidence to the composition hand-off graph.

The current evidence receipt already binds native mechanics, regional blood and
zeroth-moment blood/tissue transfer.  The composition extension binds the same
one-male source graph to registered foot proxies and the muscle route/volume
incidence table.  This bridge also binds an explicit receipt proving that no
adipose source layer is present.  It checks that the receipts refer to the
exact v14 composition hash and keeps every physical-owner, material, fat, and
behavior gate fail-closed.
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
PROFILE = ROOT / "config/human-source-evidence-bridge.v1.json"
CURRENT_EVIDENCE = ROOT / "Docs/media/current-human-evidence-join-20260915/receipt-v3.json"
COMPOSITION_EXTENSION = ROOT / (
    "Docs/media/body-composition-extension-join-20260915/receipt-v1.json"
)
BLOOD_MASS_TRANSFER = ROOT / "Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json"
MUSCLE_TISSUE_MASS = ROOT / "Docs/media/muscle-tissue-mass-candidate-20260915/receipt-v1.json"
FAT_SOURCE_ABSENCE = ROOT / "Docs/media/fat-source-absence-candidate-20260915/receipt-v1.json"
SCHEMA = "HumanPack.human-source-evidence-bridge.v1"


class EvidenceBridgeError(HumanImportError):
    """The current Human evidence cannot be joined to the source graph."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceBridgeError("Human source evidence bridge: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise EvidenceBridgeError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _input(path: Path, value: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(value.get("schema")),
            "file_sha256": digest}


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "bridge profile", canonical_required=True)
    required = {
        "schema", "id", "current_evidence", "composition_extension", "blood_mass_transfer",
        "muscle_tissue_mass_candidate", "fat_source_absence_candidate",
        "expected_bed_count", "expected_blood_owner_count", "expected_closed_muscle_volume_count",
        "expected_muscle_route_count",
        "expected_foot_proxy_count", "boundary",
    }
    _require(set(profile) == required, "bridge profile fields differ")
    _require(profile["schema"] == "numi.human.human-source-evidence-bridge.v1",
             "unsupported bridge profile schema")
    _require(profile["id"] == "current_runtime_to_composition_graph",
             "unsupported bridge profile")
    for key in ("current_evidence", "composition_extension", "blood_mass_transfer",
                "muscle_tissue_mass_candidate", "fat_source_absence_candidate"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_bed_count": 7,
        "expected_blood_owner_count": 6,
        "expected_closed_muscle_volume_count": 60,
        "expected_muscle_route_count": 416,
        "expected_foot_proxy_count": 30,
    }
    for key, expected_value in expected.items():
        _require(profile[key] == expected_value, f"{key} differs from the source contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "bridge boundary is missing")
    return profile, digest


def compile_bridge(*, current_evidence: Path = CURRENT_EVIDENCE,
                   composition_extension: Path = COMPOSITION_EXTENSION,
                   blood_mass_transfer: Path = BLOOD_MASS_TRANSFER,
                   muscle_tissue_mass: Path = MUSCLE_TISSUE_MASS,
                   fat_source_absence: Path = FAT_SOURCE_ABSENCE,
                   profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    current_path = Path(current_evidence)
    extension_path = Path(composition_extension)
    blood_path = Path(blood_mass_transfer)
    muscle_mass_path = Path(muscle_tissue_mass)
    fat_absence_path = Path(fat_source_absence)
    current, current_sha = _read(current_path, "current evidence receipt")
    extension, extension_sha = _read(extension_path, "composition extension receipt")
    blood_doc, blood_sha = _read(blood_path, "blood mass-transfer receipt")
    muscle_mass_doc, muscle_mass_sha = _read(muscle_mass_path, "muscle tissue-mass receipt")
    fat_absence_doc, fat_absence_sha = _read(fat_absence_path, "fat source-absence receipt")

    _require(current.get("schema") == "HumanPack.current-human-evidence-join.v2"
             and current.get("status") == "partial",
             "current evidence receipt changed")
    current_inputs = current.get("inputs")
    current_qualification = current.get("qualification")
    current_ownership = current.get("ownership")
    _require(isinstance(current_inputs, dict) and isinstance(current_qualification, dict)
             and isinstance(current_ownership, dict),
             "current evidence receipt is incomplete")
    _require(current_qualification.get("source_identity_graph_bound") is True
             and current_qualification.get("integrated_human_qualification") is False,
             "current evidence source or integrated boundary changed")
    _require(current_ownership and all(value == 0 for value in current_ownership.values()),
             "current evidence contains a physical owner")
    _require(current_qualification.get("regional_blood_transport") is True
             and current_qualification.get("oxygen_amount_exchange") is True
             and current_qualification.get("blood_tissue_mass_transfer_candidate") is True
             and current_qualification.get("force_convergence") is False,
             "current evidence lacks the regional organ/blood hand-offs")
    current_blood_input = current_inputs.get("blood_mass_transfer")
    _require(isinstance(current_blood_input, dict)
             and current_blood_input.get("file_sha256") == blood_sha
             and current_blood_input.get("schema") == blood_doc.get("schema"),
             "runtime and blood mass-transfer hashes diverge")
    _require(blood_doc.get("schema") == "HumanPack.organ-blood-mass-transfer-candidate.v1"
             and blood_doc.get("status") == "partial"
             and blood_doc.get("clock", {}).get("nanoseconds") == 12500
             and blood_doc.get("counts", {}).get("bed_count") == profile_doc["expected_bed_count"]
             and blood_doc.get("counts", {}).get("source_blood_owner_count") == profile_doc["expected_blood_owner_count"]
             and blood_doc.get("accepted_steps") == 511
             and blood_doc.get("rejected_steps") == 1
             and blood_doc.get("conservation", {}).get("mass_conserved") is True
             and blood_doc.get("conservation", {}).get("volume_conserved") is True,
             "blood mass-transfer receipt changed")
    blood_qualification = blood_doc.get("qualification", {})
    for key in ("mechanical_blood_mass_owner", "mechanical_tissue_mass_owner",
                "anatomical_exchange_owner", "anatomical_vessel_lumen",
                "material_density_calibrated", "organ_mechanics", "subject_calibration"):
        _require(blood_qualification.get(key) is False,
                 f"blood mass-transfer boundary changed for {key}")

    _require(muscle_mass_doc.get("schema") == "HumanPack.muscle-tissue-mass-candidate.v1"
             and muscle_mass_doc.get("status") == "partial"
             and muscle_mass_doc.get("counts", {}).get("source_muscle_surface_count") == 148
             and muscle_mass_doc.get("counts", {}).get("closed_volume_count") == profile_doc["expected_closed_muscle_volume_count"]
             and muscle_mass_doc.get("counts", {}).get("unadmitted_surface_count") == 88
             and muscle_mass_doc.get("counts", {}).get("mechanical_mass_owner_count") == 0,
             "muscle tissue-mass receipt changed")
    muscle_mass_qualification = muscle_mass_doc.get("qualification", {})
    _require(muscle_mass_qualification.get("candidate_mass_budget") is True
             and muscle_mass_qualification.get("skeletal_muscle_tissue_mass_candidate") is True
             and muscle_mass_qualification.get("skeletal_muscle_tissue_mass_owner") is False
             and muscle_mass_qualification.get("disjoint_volume_partition") is False,
             "muscle tissue-mass boundary changed")

    _require(fat_absence_doc.get("schema") == "HumanPack.fat-source-absence-candidate.v1"
             and fat_absence_doc.get("status") == "partial"
             and fat_absence_doc.get("subject") == "one adult male source package",
             "fat source-absence receipt changed")
    fat_counts = fat_absence_doc.get("counts", {})
    _require(fat_counts.get("fat_surface_count") == 0
             and fat_counts.get("fat_volume_candidate_count") == 0
             and fat_counts.get("fat_mass_candidate_count") == 0
             and fat_counts.get("fat_physical_volume_owner_count") == 0
             and fat_counts.get("fat_mechanical_mass_owner_count") == 0,
             "fat source-absence counts changed")
    fat_qualification = fat_absence_doc.get("qualification", {})
    _require(fat_qualification.get("fat_source_absence_bound") is True
             and fat_qualification.get("fat_geometry_present") is False
             and fat_qualification.get("fat_mass_candidate") is False
             and fat_qualification.get("fat_mechanical_mass_owner") is False
             and fat_qualification.get("integrated_human_qualification") is False,
             "fat source-absence boundary changed")

    _require(extension.get("schema") == "HumanPack.body-composition-extension-join.v1"
             and extension.get("status") == "partial",
             "composition extension receipt changed")
    extension_inputs = extension.get("inputs")
    extension_base = extension.get("base_receipt")
    extension_qualification = extension.get("qualification")
    extensions = extension.get("extensions")
    _require(isinstance(extension_inputs, dict) and isinstance(extension_base, dict)
             and isinstance(extension_qualification, dict) and isinstance(extensions, dict),
             "composition extension receipt is incomplete")
    _require(extension_base.get("schema") == "HumanPack.body-composition-integration-candidate.v1"
             and extension_base.get("physical_owner_count") == 0
             and extension_base.get("source_identity_graph_bound") is True,
             "composition extension base boundary changed")
    current_base = current_inputs.get("body_composition")
    extension_base_input = extension_inputs.get("base_composition")
    _require(isinstance(current_base, dict) and isinstance(extension_base_input, dict)
             and current_base.get("schema") == extension_base.get("schema")
             and current_base.get("file_sha256") == extension_base_input.get("file_sha256")
             and current_base.get("file_sha256") == extension_base.get("sha256"),
             "runtime and composition base hashes diverge")
    for key in ("base_source_identity_graph_bound", "foot_contact_proxy_bound",
                "muscle_route_volume_incidence_bound", "cross_extension_source_hashes_closed"):
        _require(extension_qualification.get(key) is True,
                 f"composition extension lacks {key}")
    for key in ("anatomical_supports_loading", "dynamic_foot_contact",
                "skeletal_muscle_tissue_volume", "skeletal_muscle_tissue_mass",
                "activation_calibration", "material_calibration", "subject_calibration",
                "standing", "recovery", "walking", "integrated_human_qualification"):
        _require(extension_qualification.get(key) is False,
                 f"composition extension boundary changed for {key}")

    blood = current.get("blood_tissue_mass_transfer")
    regional = current.get("regional_transport")
    _require(isinstance(blood, dict) and isinstance(regional, dict),
             "current blood summaries are incomplete")
    _require(blood.get("accepted_steps") == 511 and blood.get("rejected_steps") == 1
             and blood.get("mass_conserved") is True and blood.get("volume_conserved") is True,
             "blood/tissue mass-transfer summary changed")
    _require(regional.get("regional_bed_count") == profile_doc["expected_bed_count"],
             "regional bed count changed")

    foot = extensions.get("foot_contact_proxy")
    muscle = extensions.get("muscle_route_volume")
    _require(isinstance(foot, dict) and isinstance(muscle, dict),
             "composition extension domains are incomplete")
    _require(foot.get("proxy_count") == profile_doc["expected_foot_proxy_count"]
             and muscle.get("source_route_count") == profile_doc["expected_muscle_route_count"]
             and muscle.get("closed_geometry_volume_owner_count") == profile_doc["expected_closed_muscle_volume_count"]
             and muscle.get("volume_receipt_sha256") ==
             muscle_mass_doc.get("inputs", {}).get("muscle_volume_receipt", {}).get("file_sha256"),
             "composition extension counts changed")

    qualification = {
        "source_runtime_identity_bound": True,
        "organ_blood_transport_candidate_bound": True,
        "blood_tissue_mass_transfer_candidate_bound": True,
        "foot_contact_proxy_bound": True,
        "muscle_route_volume_incidence_bound": True,
        "physical_owner_count": 0,
        "anatomical_supports_loading": False,
        "dynamic_foot_contact": False,
        "activation_calibration": False,
        "anatomical_blood_mass_transfer": False,
        "mechanical_blood_mass_owner": False,
        "skeletal_muscle_tissue_volume": False,
        "skeletal_muscle_tissue_mass": False,
        "skeletal_muscle_tissue_mass_candidate_bound": True,
        "fat_source_absence_bound": True,
        "fat_geometry_and_mass": False,
        "organ_mechanics": False,
        "material_calibration": False,
        "subject_calibration": False,
        "force_convergence": False,
        "standing": False,
        "sustained_standing": False,
        "recovery": False,
        "walking": False,
        "integrated_human_qualification": False,
    }
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.human-source-evidence-bridge.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "current_evidence": _input(current_path, current, current_sha),
            "composition_extension": _input(extension_path, extension, extension_sha),
            "blood_mass_transfer": _input(blood_path, blood_doc, blood_sha),
            "muscle_tissue_mass": _input(muscle_mass_path, muscle_mass_doc, muscle_mass_sha),
            "fat_source_absence": _input(fat_absence_path, fat_absence_doc, fat_absence_sha),
            "profile": {"path": _relative(Path(profile)), "schema": profile_doc["schema"],
                        "file_sha256": profile_sha},
        },
        "source_graph": {
            "base_composition_sha256": extension_base["sha256"],
            "current_evidence_body_composition_sha256": current_base["file_sha256"],
            "cross_domain_hashes_closed": True,
            "physical_owner_count": 0,
        },
        "domains": {
            "organ_blood": {
                "regional_bed_count": regional["regional_bed_count"],
                "source_blood_owner_count": blood_doc["counts"]["source_blood_owner_count"],
                "accepted_mass_transfer_steps": blood["accepted_steps"],
                "rejected_mass_transfer_steps": blood["rejected_steps"],
                "mass_conserved": blood["mass_conserved"],
                "volume_conserved": blood["volume_conserved"],
                "anatomical_owner": False,
                "mechanical_mass_owner": False,
            },
            "contact": {
                "foot_body_count": foot["foot_body_count"],
                "proxy_count": foot["proxy_count"],
                "active_support_witness_count": foot["active_support_witness_count"],
                "dynamic_contact": False,
            },
            "muscle": {
                "source_route_count": muscle["source_route_count"],
                "closed_geometry_volume_owner_count": muscle["closed_geometry_volume_owner_count"],
                "routes_with_closed_geometry": muscle["routes_with_closed_geometry"],
                "routes_without_surface_binding": muscle["routes_without_surface_binding"],
                "volume_partition_owner": False,
                "activation_force_transfer": False,
                "candidate_mass_kg": muscle_mass_doc["totals"]["candidate_mass_kg"],
                "candidate_density_kg_per_m3": muscle_mass_doc["density"]["candidate_kg_per_m3"],
                "skeletal_muscle_tissue_mass_candidate": True,
                "skeletal_muscle_tissue_mass_owner": False,
            },
            "fat": {
                "fat_surface_count": fat_counts["fat_surface_count"],
                "fat_volume_candidate_count": fat_counts["fat_volume_candidate_count"],
                "fat_mass_candidate_count": fat_counts["fat_mass_candidate_count"],
                "fat_source_absence_bound": True,
                "fat_physical_volume_owner_count": fat_counts["fat_physical_volume_owner_count"],
                "fat_mechanical_mass_owner_count": fat_counts["fat_mechanical_mass_owner_count"],
            },
        },
        "qualification": qualification,
        "blockers": [
            {"id": "force_convergence", "status": "open",
             "reason": "Static force rows and dynamic components are diagnosed, but common-duration state/force convergence remains unqualified."},
            {"id": "anatomical_supports_loading", "status": "open",
             "reason": "Foot proxy bounds and support identities are joined; collider admission, exclusions, friction and calibrated loading remain open."},
            {"id": "activation_calibration", "status": "open",
             "reason": "The source recruitment vector is joined to muscle incidence, but measured activation and held-out force data are absent."},
            {"id": "blood_mass_transfer", "status": "open",
             "reason": "Regional zeroth-moment transfer conserves and rolls back on the exact clock; anatomical lumen, tissue exchange ownership and mechanical blood mass remain open."},
            {"id": "materials_and_fat", "status": "open",
             "reason": "Material calibration and subject density remain unresolved; the source graph now explicitly proves that no adipose geometry or mass layer is available, so fat geometry/mass and skeletal-muscle tissue partition are not admitted."},
            {"id": "standing_recovery_walking", "status": "open",
             "reason": "The joined release is bounded and replayable, but sustained standing, perturbation recovery and walking remain unqualified."},
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
    result = compile_bridge(current_evidence=arguments.current_evidence,
                            composition_extension=arguments.composition_extension,
                            blood_mass_transfer=arguments.blood_mass_transfer,
                            muscle_tissue_mass=arguments.muscle_tissue_mass,
                            fat_source_absence=arguments.fat_source_absence,
                            profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output),
                      "physical_owner_count": 0}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--current-evidence", type=Path, default=CURRENT_EVIDENCE)
    parser.add_argument("--composition-extension", type=Path, default=COMPOSITION_EXTENSION)
    parser.add_argument("--blood-mass-transfer", type=Path, default=BLOOD_MASS_TRANSFER)
    parser.add_argument("--muscle-tissue-mass", type=Path, default=MUSCLE_TISSUE_MASS)
    parser.add_argument("--fat-source-absence", type=Path, default=FAT_SOURCE_ABSENCE)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (EvidenceBridgeError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"human-source-evidence-bridge: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

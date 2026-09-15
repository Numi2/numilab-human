"""Join the one-male mass and ownership candidates without double counting.

The Human source graph contains several useful mass-like quantities: compiled
rigid-body mass, organ surface candidates, regional blood/tissue state, vessel
surface moments, a costal tissue partition, and a muscle-volume allocation.
They are not one disjoint mass model.  This compiler makes the scopes and the
non-additivity rule explicit, while retaining the physical-owner gates false.
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
SCHEMA = "HumanPack.body-composition-mass-ledger.v1"

MYOSIM = ROOT / "Docs/media/myosim-mass-owner-20260915/receipt-v1.json"
SCALING = ROOT / "Docs/media/subject-mass-scaling-20260915/receipt-v1.json"
RUNTIME_INPUT = ROOT / "Docs/media/subject-scaled-runtime-input-20260915/receipt-v1.json"
SUBJECT = ROOT / "Docs/media/addbiomechanics-subject-binding-20260915/receipt-v1.json"
ORGAN = ROOT / "Docs/media/tissue-mass-candidate-20260915/receipt-v1.json"
BLOOD_TRANSFER = ROOT / "Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json"
CARDIAC = ROOT / "Docs/media/cardiac-blood-mass-candidate-20260914/receipt.json"
REGIONAL_TISSUE = ROOT / "Docs/media/regional-tissue-mass-candidate-20260914/receipt-v1.json"
MUSCLE = ROOT / "Docs/media/muscle-route-mass-partition-candidate-20260915/receipt-v1.json"
FAT = ROOT / "Docs/media/fat-source-absence-candidate-20260915/receipt-v1.json"
VESSEL = ROOT / "Docs/media/vessel-mass-moment-candidate-20260915/receipt-v1.json"


class MassLedgerError(HumanImportError):
    """The one-male mass ledger cannot be joined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MassLedgerError("body composition mass ledger: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise MassLedgerError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _input(path: Path, document: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(document.get("schema")),
            "file_sha256": digest}


def _schema(document: dict[str, Any], expected: str, label: str) -> None:
    _require(document.get("schema") == expected,
             f"{label} schema is {document.get('schema')!r}, expected {expected!r}")


def _false(document: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    qualification = document.get("qualification", {})
    _require(isinstance(qualification, dict), f"{label} qualification is missing")
    for key in keys:
        _require(qualification.get(key) is False, f"{label} promoted {key}")


def compile_ledger(
    *,
    myosim: Path = MYOSIM,
    scaling: Path = SCALING,
    runtime_input: Path = RUNTIME_INPUT,
    subject: Path = SUBJECT,
    organ: Path = ORGAN,
    blood_transfer: Path = BLOOD_TRANSFER,
    cardiac: Path = CARDIAC,
    regional_tissue: Path = REGIONAL_TISSUE,
    muscle: Path = MUSCLE,
    fat: Path = FAT,
    vessel: Path = VESSEL,
) -> dict[str, Any]:
    paths = {
        "myosim": Path(myosim), "scaling": Path(scaling),
        "runtime_input": Path(runtime_input), "subject": Path(subject),
        "organ": Path(organ), "blood_transfer": Path(blood_transfer),
        "cardiac": Path(cardiac), "regional_tissue": Path(regional_tissue),
        "muscle": Path(muscle), "fat": Path(fat), "vessel": Path(vessel),
    }
    labels = {
        "myosim": "MyoSim mass owner", "scaling": "subject mass scaling",
        "runtime_input": "subject-scaled runtime input", "subject": "subject binding",
        "organ": "organ tissue mass candidate", "blood_transfer": "blood/tissue transfer",
        "cardiac": "cardiac blood candidate", "regional_tissue": "regional tissue candidate",
        "muscle": "muscle route mass partition", "fat": "fat source absence",
        "vessel": "vessel mass moments",
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for key, path in paths.items():
        documents[key], hashes[key] = _read(path, labels[key])

    myosim_doc = documents["myosim"]
    _schema(myosim_doc, "HumanPack.myosim-rigid-body-mass-owner-candidate.v1", labels["myosim"])
    _require(myosim_doc.get("status") == "partial", "MyoSim mass owner changed status")
    myosim_mass = myosim_doc.get("rigid_body_mass", {})
    _require(myosim_mass.get("body_count") == 103 and
             myosim_mass.get("mass_bearing_body_count") == 96 and
             myosim_mass.get("zero_mass_body_count") == 7,
             "MyoSim mass owner body counts changed")
    source_mass_kg = _finite(myosim_mass.get("total_mass_kg"), "source rigid-body mass")
    _require(math.isclose(source_mass_kg, 97.13195176621342, rel_tol=0.0, abs_tol=1.0e-12),
             "source rigid-body mass changed")
    _false(myosim_doc, (
        "anatomical_organ_mass_owner", "mechanical_blood_mass_owner",
        "skeletal_muscle_tissue_mass_owner", "fat_volume_and_mass_owner",
        "skin_volume_and_mass_owner", "soft_tissue_material_calibration",
        "subject_calibration", "whole_body_dynamic_mass_matrix_owner",
    ), labels["myosim"])

    scaling_doc = documents["scaling"]
    _schema(scaling_doc, "HumanPack.subject-mass-scaling-candidate.v1", labels["scaling"])
    scaling = scaling_doc.get("scaling", {})
    scaling_qualification = scaling_doc.get("qualification", {})
    _require(scaling_doc.get("status") == "partial" and
             scaling_qualification.get("target_mass_closure") is True and
             scaling_qualification.get("mechanical_runtime_admitted") is False,
             "subject mass scaling boundary changed")
    target_mass_kg = _finite(scaling.get("target_mass_kg"), "target subject mass")
    scaled_mass_kg = _finite(scaling.get("scaled_mass_kg"), "scaled subject mass")
    closure_error_kg = _finite(scaling.get("mass_closure_error_kg"), "scalar mass closure")
    _require(target_mass_kg == 65.5 and abs(closure_error_kg) <= 1.0e-12 and
             abs(scaled_mass_kg - target_mass_kg) <= 1.0e-12,
             "scalar subject mass closure changed")

    runtime_doc = documents["runtime_input"]
    _schema(runtime_doc, "HumanPack.subject-scaled-runtime-input.v1", labels["runtime_input"])
    _require(runtime_doc.get("status") == "partial" and
             runtime_doc.get("qualification", {}).get("scaled_mass_closure") is True and
             runtime_doc.get("qualification", {}).get("subject_mass_calibration") is False,
             "subject-scaled runtime boundary changed")
    runtime_source = runtime_doc.get("source", {})
    _require(runtime_source.get("magic") == "NHRIGID2" and
             runtime_source.get("source_body_count") == 103 and
             runtime_source.get("engine_body_count") == 157 and
             runtime_source.get("nq") == 129 and runtime_source.get("nv") == 128,
             "subject-scaled runtime layout changed")

    subject_doc = documents["subject"]
    _schema(subject_doc, "HumanPack.addbiomechanics-subject-binding.v1", labels["subject"])
    subject_info = subject_doc.get("subject", {})
    _require(subject_doc.get("status") == "partial" and
             subject_info.get("id") == "Falisse2017:subject_1" and
             subject_info.get("sex") == "male" and subject_info.get("age_years") == 43 and
             subject_info.get("height_m") == 1.78 and subject_info.get("mass_kg") == target_mass_kg,
             "one-male subject identity changed")
    _require(subject_doc.get("qualification", {}).get("subject_mass_identity_match") is False,
             "subject binding silently promoted mass identity")

    organ_doc = documents["organ"]
    _schema(organ_doc, "HumanPack.tissue-mass-composition-candidate.v1", labels["organ"])
    _require(organ_doc.get("status") == "partial" and
             organ_doc.get("counts", {}).get("source_members") == 378 and
             organ_doc.get("counts", {}).get("organ_surface_mass_candidates") == 349 and
             organ_doc.get("counts", {}).get("shared_source_members") == 8,
             "organ candidate coverage changed")
    organ_mass_kg = _finite(organ_doc.get("totals", {}).get("candidate_surface_mass_kg"),
                            "organ candidate mass")
    _require(organ_mass_kg > 0.0, "organ candidate mass is missing")
    _false(organ_doc, ("mechanical_mass_owner_assigned", "physical_volume_authority",
                       "interdomain_disjointness_qualified", "material_calibration",
                       "subject_calibration"), labels["organ"])

    transfer_doc = documents["blood_transfer"]
    _schema(transfer_doc, "HumanPack.organ-blood-mass-transfer-candidate.v1", labels["blood_transfer"])
    transfer_qualification = transfer_doc.get("qualification", {})
    _require(transfer_doc.get("status") == "partial" and
             transfer_doc.get("clock", {}).get("nanoseconds") == 12500 and
             transfer_doc.get("counts", {}).get("bed_count") == 7 and
             transfer_doc.get("accepted_steps") == 511 and
             transfer_doc.get("conservation", {}).get("mass_conserved") is True and
             transfer_doc.get("conservation", {}).get("volume_conserved") is True and
             transfer_qualification.get("blood_tissue_zeroth_moment_mass_transfer") is True,
             "blood/tissue transfer evidence changed")
    _false(transfer_doc, ("mechanical_blood_mass_owner", "mechanical_tissue_mass_owner",
                          "anatomical_exchange_owner", "anatomical_vessel_lumen",
                          "material_density_calibrated", "subject_calibration"),
           labels["blood_transfer"])
    transfer_totals = transfer_doc.get("final_totals", {})
    blood_mass_kg = _finite(transfer_totals.get("blood_mass_kg"), "regional blood mass")
    tissue_mass_kg = _finite(transfer_totals.get("tissue_mass_kg"), "regional transfer tissue mass")
    transfer_mass_kg = _finite(transfer_totals.get("mass_kg"), "regional transfer mass")
    _require(math.isclose(transfer_mass_kg, blood_mass_kg + tissue_mass_kg,
                          rel_tol=0.0, abs_tol=1.0e-12),
             "blood/tissue transfer total does not close")

    cardiac_doc = documents["cardiac"]
    _schema(cardiac_doc, "HumanPack.cardiac-blood-mass-candidate.v1", labels["cardiac"])
    cardiac_budget = cardiac_doc.get("mass_budget", {})
    cardiac_mass_kg = _finite(cardiac_budget.get("candidate_mass_kg"), "cardiac candidate mass")
    _require(cardiac_doc.get("qualification", {}).get("mechanical_mass_assigned") is False and
             cardiac_doc.get("qualification", {}).get("material_density_calibrated") is False,
             "cardiac blood candidate promoted an owner")

    regional_doc = documents["regional_tissue"]
    _schema(regional_doc, "HumanPack.regional-tissue-mass-candidate.v1", labels["regional_tissue"])
    regional_mass_kg = _finite(regional_doc.get("region", {}).get("tissue_mass_kg"),
                               "regional tissue mass")
    _require(regional_doc.get("ownership", {}).get("production_mechanical_mass_owner") is False and
             regional_doc.get("ownership", {}).get("whole_body_dynamic_mass_matrix_owner") is False,
             "regional tissue promoted an owner")

    muscle_doc = documents["muscle"]
    _schema(muscle_doc, "HumanPack.muscle-route-mass-partition-candidate.v1", labels["muscle"])
    muscle_totals = muscle_doc.get("totals", {})
    muscle_mass_kg = _finite(muscle_totals.get("allocated_candidate_mass_kg"),
                              "muscle candidate mass")
    muscle_volume_m3 = _finite(muscle_totals.get("allocated_candidate_volume_m3"),
                                "muscle candidate volume")
    _require(muscle_totals.get("candidate_partition_is_disjoint") is True and
             muscle_totals.get("candidate_is_mechanical_mass") is False and
             muscle_doc.get("qualification", {}).get("activation_force_transfer") is False,
             "muscle candidate partition changed")

    fat_doc = documents["fat"]
    _schema(fat_doc, "HumanPack.fat-source-absence-candidate.v1", labels["fat"])
    fat_counts = fat_doc.get("counts", {})
    _require(fat_doc.get("qualification", {}).get("fat_source_absence_bound") is True and
             all(fat_counts.get(key) == 0 for key in (
                 "fat_surface_count", "fat_volume_candidate_count", "fat_mass_candidate_count",
                 "fat_physical_volume_owner_count", "fat_mechanical_mass_owner_count")),
             "fat source absence changed")

    vessel_doc = documents["vessel"]
    _schema(vessel_doc, "HumanPack.vessel-mass-moment-owner-candidate.v1", labels["vessel"])
    vessel_totals = vessel_doc.get("totals", {})
    vessel_mass_kg = _finite(vessel_totals.get("mass_kg"), "vessel surface candidate mass")
    _require(vessel_totals.get("owner_count") == 6 and
             vessel_doc.get("qualification", {}).get("source_surface_is_lumen") is False and
             vessel_doc.get("qualification", {}).get("anatomical_blood_mass_owner") is False,
             "vessel candidate promoted an owner")

    source_target_delta_kg = source_mass_kg - target_mass_kg
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.body-composition-mass-ledger.1",
        "status": "partial",
        "subject": {
            "id": subject_info["id"], "age_years": subject_info["age_years"],
            "sex": subject_info["sex"], "height_m": subject_info["height_m"],
            "target_mass_kg": target_mass_kg,
        },
        "inputs": {key: _input(paths[key], documents[key], hashes[key]) for key in paths},
        "scalar_mass_closure": {
            "source_rigid_body_mass_kg": source_mass_kg,
            "target_subject_mass_kg": target_mass_kg,
            "source_to_target_delta_kg": source_target_delta_kg,
            "scaled_mass_kg": scaled_mass_kg,
            "closure_error_kg": closure_error_kg,
            "mass_factor": _finite(scaling.get("mass_factor"), "mass factor"),
            "target_closed": True,
            "mechanical_subject_calibrated": False,
        },
        "candidate_scopes": {
            "rigid_body_source_owner": {
                "mass_kg": source_mass_kg, "body_count": myosim_mass["body_count"],
                "mass_bearing_body_count": myosim_mass["mass_bearing_body_count"],
                "admitted_to_rigid_body_dynamics": True,
                "subject_scaled_handoff_admitted_to_production": False,
                "authority": "compiled source owner; the subject-scaled handoff remains a candidate",
            },
            "organ_surface_candidates": {
                "mass_kg": organ_mass_kg, "source_members": 378,
                "computed_candidates": 349, "physical_owner": False,
            },
            "blood_tissue_transfer": {
                "blood_mass_kg": blood_mass_kg, "tissue_mass_kg": tissue_mass_kg,
                "mass_kg": transfer_mass_kg, "beds": 7, "accepted_steps": 511,
                "physical_owner": False,
            },
            "cardiac_hydraulic_candidate": {
                "mass_kg": cardiac_mass_kg, "physical_owner": False,
            },
            "regional_costal_tissue": {
                "mass_kg": regional_mass_kg, "physical_owner": False,
            },
            "skeletal_muscle_route_partition": {
                "mass_kg": muscle_mass_kg, "volume_m3": muscle_volume_m3,
                "physical_owner": False, "activation_force_owner": False,
            },
            "vessel_surface_moments": {
                "mass_kg": vessel_mass_kg, "owner_count": 6,
                "lumen_or_tube_owner": False,
            },
            "fat": {
                "surface_count": 0, "volume_candidate_count": 0,
                "mass_candidate_count": 0, "physical_owner": False,
            },
        },
        "scope_policy": {
            "candidate_scopes_are_disjoint": False,
            "candidate_mass_sum_kg": None,
            "candidate_mass_sum_status": "forbidden_until_interdomain_partition",
            "overlap_reasons": [
                "organ surface and blood/tissue scopes share source organ identities",
                "cardiac and vessel surface candidates are hydraulic/surface views, not disjoint tissue owners",
                "regional tissue may overlap the compiled donor-body row until a whole-body partition exists",
                "skeletal muscle candidate volume has unresolved density and no production mass owner",
            ],
            "candidate_mass_admitted_to_dynamics": False,
        },
        "ownership": {
            "rigid_body_source_owner_count": myosim_mass["body_count"],
            "organ_physical_volume_owner_count": 0,
            "mechanical_blood_mass_owner_count": 0,
            "mechanical_tissue_mass_owner_count": 0,
            "skeletal_muscle_tissue_mass_owner_count": 0,
            "fat_mechanical_mass_owner_count": 0,
            "whole_body_dynamic_mass_matrix_owner_count": 0,
        },
        "qualification": {
            "one_male_identity_bound": True,
            "rigid_body_source_mass_owner_bound": True,
            "scalar_subject_target_closure": True,
            "candidate_scope_non_additivity_checked": True,
            "candidate_mass_admitted_to_dynamics": False,
            "organ_candidate_moment_bound": True,
            "blood_tissue_conservation_bound": True,
            "muscle_candidate_partition_bound": True,
            "fat_source_absence_bound": True,
            "organ_physical_volume": False,
            "mechanical_blood_mass": False,
            "mechanical_tissue_mass": False,
            "skeletal_muscle_tissue_mass": False,
            "fat_geometry_and_mass": False,
            "material_calibration": False,
            "activation_force_transfer": False,
            "subject_calibration": False,
            "whole_body_mass_partition": False,
            "integrated_human_qualification": False,
        },
        "blockers": [
            {"id": "whole-body-mass-partition", "status": "open",
             "reason": "The scalar 65.5 kg target closes, but segment composition, inertia, and disjoint organ/blood/fat/muscle/tissue ownership are not calibrated."},
            {"id": "physical-soft-tissue-owners", "status": "open",
             "reason": "Candidate organ, blood, vessel, regional tissue, skeletal-muscle and fat scopes have no production physical owners."},
            {"id": "activation-and-material-calibration", "status": "open",
             "reason": "Density/material and measured activation-to-force transfer remain unresolved and are barred from rigid-body dynamics."},
        ],
        "boundary": (
            "One 43-year-old male source package is identity-bound and its scalar "
            "65.5 kg target closes under an explicit uniform mass-only handoff. "
            "Organ, blood, vessel, regional tissue, skeletal-muscle and fat records "
            "remain separately scoped candidates; their mass-like values are not "
            "summed and are not admitted to production dynamics. Segment composition, "
            "inertia, anatomical support/loading, activation, material calibration, "
            "organ mechanics, sustained standing, recovery, and walking remain open."
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


# Keep the private helper spelling used by the other candidate compilers and
# their focused tests.
_immutable_write = immutable_write


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--myosim", type=Path, default=MYOSIM)
    parser.add_argument("--scaling", type=Path, default=SCALING)
    parser.add_argument("--runtime-input", type=Path, default=RUNTIME_INPUT)
    parser.add_argument("--subject", type=Path, default=SUBJECT)
    parser.add_argument("--organ", type=Path, default=ORGAN)
    parser.add_argument("--blood-transfer", type=Path, default=BLOOD_TRANSFER)
    parser.add_argument("--cardiac", type=Path, default=CARDIAC)
    parser.add_argument("--regional-tissue", type=Path, default=REGIONAL_TISSUE)
    parser.add_argument("--muscle", type=Path, default=MUSCLE)
    parser.add_argument("--fat", type=Path, default=FAT)
    parser.add_argument("--vessel", type=Path, default=VESSEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_ledger(
        myosim=arguments.myosim, scaling=arguments.scaling,
        runtime_input=arguments.runtime_input, subject=arguments.subject,
        organ=arguments.organ, blood_transfer=arguments.blood_transfer,
        cardiac=arguments.cardiac, regional_tissue=arguments.regional_tissue,
        muscle=arguments.muscle, fat=arguments.fat, vessel=arguments.vessel,
    )
    output = arguments.output.resolve()
    digest = immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(output), "sha256": digest,
                      "target_mass_kg": result["scalar_mass_closure"]["target_subject_mass_kg"],
                      "candidate_mass_admitted_to_dynamics": False}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (MassLedgerError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"body-composition-mass-ledger: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

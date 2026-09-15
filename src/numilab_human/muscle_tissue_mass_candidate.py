"""Compile an explicit, non-production skeletal-muscle mass budget.

The source geometry audit admits 60 single-closed muscle surfaces.  This
compiler applies a profile-supplied engineering density only to produce a
reproducible candidate mass budget.  It does not create a disjoint FEM volume,
partition the 88 other muscle surfaces, replace MyoSim rigid-body mass, or
assign a mechanical owner.
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
PROFILE = ROOT / "config/muscle-tissue-mass-candidate.v1.json"
VOLUME_RECEIPT = ROOT / "Docs/media/muscle-geometric-volume-candidate-20260914/receipt-v1.json"
SCHEMA = "HumanPack.muscle-tissue-mass-candidate.v1"


class MuscleMassError(HumanImportError):
    """A skeletal-muscle mass candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MuscleMassError("muscle tissue mass: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise MuscleMassError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "muscle mass profile", canonical_required=True)
    required = {
        "schema", "id", "volume_receipt", "density_kg_per_m3", "density_provenance",
        "expected_closed_volume_count", "expected_source_surface_count",
        "expected_unadmitted_surface_count", "boundary",
    }
    _require(set(profile) == required, "muscle mass profile fields differ")
    _require(profile["schema"] == "numi.human.muscle-tissue-mass-candidate.v1",
             "unsupported muscle mass profile schema")
    _require(profile["id"] == "source_closed_muscle_candidate_mass_budget",
             "unsupported muscle mass profile")
    volume_receipt = profile["volume_receipt"]
    _require(isinstance(volume_receipt, str) and volume_receipt.strip()
             and not Path(volume_receipt).is_absolute()
             and ".." not in Path(volume_receipt).parts and "\\" not in volume_receipt,
             "volume receipt path is unsafe")
    for key, expected in (("expected_closed_volume_count", 60),
                          ("expected_source_surface_count", 148),
                          ("expected_unadmitted_surface_count", 88)):
        _require(profile[key] == expected, f"{key} differs from the source contract")
    density = profile["density_kg_per_m3"]
    _require(type(density) in (int, float) and math.isfinite(float(density))
             and float(density) > 0.0, "density candidate is invalid")
    _require(isinstance(profile["density_provenance"], str)
             and "unresolved" in profile["density_provenance"],
             "density provenance must remain unresolved")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "muscle mass boundary is missing")
    return profile, digest


def compile_candidate(*, volume_receipt: Path = VOLUME_RECEIPT,
                       profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    volume_path = Path(volume_receipt)
    volume_doc, volume_sha = _read(volume_path, "muscle volume receipt")
    _require(volume_doc.get("schema") == "HumanPack.muscle-geometric-volume-candidate.v1"
             and volume_doc.get("status") == "partial", "muscle volume receipt changed")
    coverage = volume_doc.get("coverage", {})
    geometry = volume_doc.get("geometry", {})
    owners = volume_doc.get("owners")
    qualification = volume_doc.get("qualification", {})
    _require(isinstance(coverage, dict) and isinstance(geometry, dict)
             and isinstance(owners, list) and isinstance(qualification, dict),
             "muscle volume receipt is incomplete")
    _require(coverage.get("closed_muscle_component_count") == profile_doc["expected_closed_volume_count"]
             and coverage.get("source_muscle_surface_count") == profile_doc["expected_source_surface_count"]
             and coverage.get("closed_multi_component_count") +
             coverage.get("topology_defective_muscle_component_count") == profile_doc["expected_unadmitted_surface_count"],
             "muscle volume coverage changed")
    _require(geometry.get("volume_units") == "m^3"
             and type(geometry.get("closed_muscle_volume_total_m3")) in (int, float)
             and math.isfinite(float(geometry["closed_muscle_volume_total_m3"])),
             "muscle volume total is invalid")
    for key in ("physical_volume_owner", "skeletal_muscle_tissue_mass_owner",
                "material_calibration", "volumetric_active_force_owner"):
        _require(qualification.get(key) is False,
                 f"muscle volume receipt promoted {key}")
    _require(len(owners) == profile_doc["expected_closed_volume_count"],
             "muscle volume owner count changed")
    density = float(profile_doc["density_kg_per_m3"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_volume = 0.0
    total_mass = 0.0
    for owner in owners:
        _require(isinstance(owner, dict), "muscle volume owner row is malformed")
        member_id = owner.get("member_id")
        volume = owner.get("volume_m3")
        _require(isinstance(member_id, str) and member_id and member_id not in seen,
                 "muscle source member identity is invalid or duplicated")
        _require(type(volume) in (int, float) and math.isfinite(float(volume))
                 and float(volume) > 0.0, f"muscle source volume is invalid: {member_id}")
        _require(owner.get("physical_volume_owner") is False
                 and owner.get("mechanical_mass_owner") is False
                 and owner.get("material_owner") is False
                 and owner.get("volumetric_active_force_owner") is False,
                 f"muscle volume owner boundary changed: {member_id}")
        member_volume = float(volume)
        member_mass = density * member_volume
        seen.add(member_id)
        total_volume += member_volume
        total_mass += member_mass
        rows.append({
            "stable_id": owner.get("stable_id"),
            "member_id": member_id,
            "member_sha256": owner.get("member_sha256"),
            "source_volume_m3": member_volume,
            "density_candidate_kg_per_m3": density,
            "candidate_mass_kg": member_mass,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "mass_admission": "candidate_budget_only_unresolved_density_and_partition",
        })
    rows.sort(key=lambda row: row["stable_id"])
    _require(math.isclose(total_volume, float(geometry["closed_muscle_volume_total_m3"]),
                          rel_tol=2e-14, abs_tol=1e-15),
             "muscle source volumes do not reproduce the geometry receipt")
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.muscle-tissue-mass-candidate.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "muscle_volume_receipt": {"path": _relative(volume_path),
                                       "schema": volume_doc["schema"],
                                       "file_sha256": volume_sha},
            "profile": {"path": _relative(Path(profile)), "schema": profile_doc["schema"],
                        "file_sha256": profile_sha},
        },
        "density": {
            "candidate_kg_per_m3": density,
            "provenance": profile_doc["density_provenance"],
            "subject_calibrated": False,
        },
        "counts": {
            "source_muscle_surface_count": coverage["source_muscle_surface_count"],
            "closed_volume_count": len(rows),
            "unadmitted_surface_count": profile_doc["expected_unadmitted_surface_count"],
            "physical_volume_owner_count": 0,
            "mechanical_mass_owner_count": 0,
        },
        "candidates": rows,
        "totals": {
            "closed_source_volume_m3": total_volume,
            "candidate_mass_kg": total_mass,
            "candidate_is_mechanical_mass": False,
            "candidate_is_disjoint_partition": False,
        },
        "qualification": {
            "source_volume_identity_bound": True,
            "candidate_density_explicit": True,
            "candidate_mass_budget": True,
            "skeletal_muscle_tissue_volume": True,
            "skeletal_muscle_tissue_mass_candidate": True,
            "skeletal_muscle_tissue_mass_owner": False,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "disjoint_volume_partition": False,
            "material_calibration": False,
            "activation_force_transfer": False,
            "activation_calibration": False,
            "fat_geometry_and_mass": False,
            "subject_calibration": False,
            "standing": False,
            "walking": False,
        },
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
    result = compile_candidate(volume_receipt=arguments.volume_receipt, profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output),
                      "closed_volume_count": result["counts"]["closed_volume_count"],
                      "candidate_mass_kg": result["totals"]["candidate_mass_kg"],
                      "mechanical_mass_owner_count": 0}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--volume-receipt", type=Path, default=VOLUME_RECEIPT)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (MuscleMassError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"muscle-tissue-mass: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

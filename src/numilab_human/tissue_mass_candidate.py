"""Compile a source-bound organ/tissue mass composition candidate.

The pinned organ moments are geometric surface integrals.  This compiler may
apply an explicitly labelled engineering density to produce a deterministic
zeroth/first/second *candidate* moment, but it never turns an overlapping
surface inventory into a mechanical mass owner.  Blood, skeletal muscle, fat,
vessel wall, tendon/fascia, and skin retain separate ownership boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.tissue-mass-composition-candidate.v1"
MOMENTS = ROOT / "Docs/media/organ-geometry-moments-20260913/moments.json"
PROFILE = ROOT / "config/numi-human-tissue-mass-candidate.v1.json"
SUPPORTED_MOMENT_SCHEMAS = {
    "HumanPack.organ-geometry-moments.v1",
    "HumanPack.organ-geometry-component-moments.v1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError("tissue mass candidate: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} is not finite")
    return float(value)


def _canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ImportError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _matrix(value: Any, label: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} is not 3x3")
    result = []
    for row_index, row in enumerate(value):
        _require(isinstance(row, list) and len(row) == 3, f"{label}[{row_index}] is not length 3")
        result.append([_finite(item, f"{label}[{row_index}][{column}]") for column, item in enumerate(row)])
    return result


def _scale(matrix: list[list[float]], factor: float) -> list[list[float]]:
    return [[factor * value for value in row] for row in matrix]


def _vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} is not length 3")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def compile_candidate(*, moments: Path = MOMENTS, profile: Path = PROFILE) -> dict[str, Any]:
    moments_doc, moments_sha = _canonical_json(Path(moments), "organ moments")
    profile_doc, profile_sha = _canonical_json(Path(profile), "tissue mass profile")
    _require(moments_doc.get("schema") in SUPPORTED_MOMENT_SCHEMAS,
             "unsupported organ moments schema")
    _require(profile_doc.get("schema") == "numi.human.tissue-mass-candidate.v1",
             "unsupported tissue mass profile schema")
    region_classes = profile_doc.get("region_classes")
    classes = profile_doc.get("classes")
    _require(isinstance(region_classes, dict) and isinstance(classes, dict),
             "profile classes are missing")
    _require(set(region_classes) == {row.get("id") for row in moments_doc.get("regions", [])},
             "profile does not cover every source region exactly")
    _require(isinstance(moments_doc.get("members"), list), "organ moments have no members")
    member_regions: dict[str, list[str]] = {}
    for region in moments_doc["regions"]:
        _require(isinstance(region, dict), "organ moments region is malformed")
        region_id = region.get("id")
        for member_id in region.get("member_ids", []):
            _require(isinstance(member_id, str) and member_id, "region member id is invalid")
            member_regions.setdefault(member_id, []).append(region_id)

    # A seam member can be declared by more than one anatomical region (for
    # example a chamber boundary or a vessel continuation).  Counting it in
    # every region would manufacture mass.  Keep the identity visible and
    # leave the shared row unresolved until a disjoint volume owner exists.
    shared_members = {
        member_id for member_id, regions in member_regions.items() if len(regions) > 1
    }

    candidates: list[dict[str, Any]] = []
    unresolved = 0
    computed = 0
    candidate_mass = 0.0
    candidate_first = [0.0, 0.0, 0.0]
    candidate_second = [[0.0] * 3 for _ in range(3)]
    for member in sorted(moments_doc["members"], key=lambda row: row.get("member_id", "")):
        _require(isinstance(member, dict), "organ moments member is malformed")
        member_id = member.get("member_id")
        regions = member_regions.get(member_id)
        _require(regions, f"member is not assigned to a region: {member_id}")
        region_id = regions[0]
        class_id = region_classes[region_id]
        class_record = classes.get(class_id)
        _require(isinstance(class_record, dict), f"profile class is missing: {class_id}")
        density = class_record.get("density_kg_per_m3")
        moment = member.get("source_surface_moments")
        admissible = (
            class_id == "organ_parenchyma"
            and member.get("moment_status") in {
                "computed_single_closed_component",
                "computed_disjoint_closed_component_sum",
            }
            and isinstance(moment, dict)
            and density is not None
        )
        row: dict[str, Any] = {
            "member_id": member_id,
            "region_id": region_id,
            "region_ids": sorted(regions),
            "tissue_class": class_id,
            "source_sha256": member.get("source_sha256"),
            "source_moment_status": member.get("moment_status"),
            "candidate_surface_volume_m3": None,
            "density_kg_per_m3": None,
            "candidate_mass_kg": None,
            "candidate_first_mass_moment_kg_m": None,
            "candidate_central_second_mass_moment_kg_m2": None,
            "mechanical_mass_owner": class_record.get("mechanical_mass_owner"),
            "physical_volume_owner": class_record.get("physical_volume_owner"),
            "mass_admission": "unresolved",
        }
        if member_id in shared_members:
            row["mass_admission"] = "shared_source_member_overlap_unresolved"
            unresolved += 1
        elif admissible:
            density_value = _finite(density, f"{class_id} density")
            _require(density_value > 0.0, f"{class_id} density is not positive")
            volume = abs(_finite(moment.get("absolute_signed_volume_m3"), f"{member_id} volume"))
            centroid = _vector(moment.get("centroid_source_frame_m"), f"{member_id} centroid")
            central = _matrix(moment.get("central_second_volume_moment_m5"), f"{member_id} central moment")
            mass = density_value * volume
            first = [mass * value for value in centroid]
            central_mass = _scale(central, density_value)
            row.update(
                candidate_surface_volume_m3=volume,
                density_kg_per_m3=density_value,
                candidate_mass_kg=mass,
                candidate_first_mass_moment_kg_m=first,
                candidate_central_second_mass_moment_kg_m2=central_mass,
                mass_admission="surface_candidate_only_physical_volume_unresolved",
            )
            computed += 1
            candidate_mass += mass
            for index in range(3):
                candidate_first[index] += first[index]
                for column in range(3):
                    candidate_second[index][column] += central_mass[index][column]
        else:
            unresolved += 1
        candidates.append(row)

    source_counts = moments_doc.get("counts", {})
    _require(computed > 0 and len(candidates) == source_counts.get("member_count"),
             "candidate rows do not cover the source moments")
    source = {
        "subject": profile_doc.get("subject"),
        "moments": str(Path(moments).relative_to(ROOT)) if Path(moments).is_relative_to(ROOT) else str(moments),
        "moments_sha256": moments_sha,
        "profile": str(Path(profile).relative_to(ROOT)) if Path(profile).is_relative_to(ROOT) else str(profile),
        "profile_sha256": profile_sha,
    }
    if moments_doc["schema"] != "HumanPack.organ-geometry-moments.v1":
        source["moments_schema"] = moments_doc["schema"]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.tissue-mass-candidate.1",
        "status": "partial",
        "source": source,
        "counts": {
            "source_members": len(candidates),
            "organ_surface_mass_candidates": computed,
            "unresolved_members": unresolved,
            "shared_source_members": len(shared_members),
            "region_count": len(region_classes),
        },
        "classes": profile_doc["classes"],
        "candidates": candidates,
        "totals": {
            "candidate_surface_mass_kg": candidate_mass,
            "candidate_first_mass_moment_kg_m": candidate_first,
            "candidate_central_second_mass_moment_kg_m2": candidate_second,
            "aggregate_is_physical": False,
            "physical_mass_owner_count": 0,
            "blood_mass_owner_count": 0,
            "fat_mass_owner_count": 0,
            "skeletal_muscle_tissue_mass_owner_count": 0,
        },
        "qualification": {
            "source_moment_identity_bound": True,
            "candidate_zeroth_first_second_moments": True,
            "candidate_mass_closes_against_density": True,
            "interdomain_disjointness_qualified": False,
            "physical_volume_authority": False,
            "mechanical_mass_owner_assigned": False,
            "blood_mass_transfer": False,
            "fat_geometry_and_mass": False,
            "skeletal_muscle_tissue_partition": False,
            "material_calibration": False,
            "subject_calibration": False,
            "organ_mechanics": False,
            "standing_walking": False,
        },
        "boundary": (
            "The source organ surfaces receive an explicit, unresolved density "
            "candidate and deterministic mass moments for closed parenchymal "
            "components. Overlap, topology defects, lumen/wall identity, fat, "
            "skeletal-muscle tissue partition, physical mass ownership, organ "
            "mechanics, blood transfer, and calibration remain closed gates."
        ),
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
    result = compile_candidate(moments=arguments.moments, profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                      "status": result["status"], **result["counts"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moments", type=Path, default=MOMENTS)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

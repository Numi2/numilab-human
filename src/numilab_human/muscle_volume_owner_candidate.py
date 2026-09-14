"""Bind admitted closed muscle surface volumes to source identities.

The geometry audit already recomputes algebraic volumes for single closed
BodyParts3D muscle components.  This compiler turns those measurements into a
stable, source-member keyed geometric-volume handoff.  It deliberately does
not promote the measurements to a FEM volume, density, mechanical mass,
constitutive material, active-force, or subject-calibrated owner.
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
AUDIT = ROOT / "Docs/media/muscle-surface-geometry-audit-20260914/receipt-v1.json"
SCHEMA = "HumanPack.muscle-geometric-volume-candidate.v1"


class MuscleVolumeError(HumanImportError):
    """A muscle geometric-volume candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MuscleVolumeError("muscle geometric volume: " + message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise MuscleVolumeError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, _sha256(raw)


def compile_candidate(*, audit: Path = AUDIT) -> dict[str, Any]:
    audit = Path(audit)
    document, audit_sha = _read(audit, "muscle geometry audit")
    _require(document.get("schema") == "HumanPack.muscle-surface-geometry-audit.v1",
             "unsupported muscle geometry audit schema")
    _require(document.get("status") == "partial", "geometry audit status changed")
    source = document.get("source")
    counts = document.get("counts")
    geometry = document.get("geometry")
    rows = document.get("surfaces")
    _require(isinstance(source, dict) and isinstance(counts, dict)
             and isinstance(geometry, dict) and isinstance(rows, list),
             "geometry audit is incomplete")
    _require(counts.get("muscle_surface_count") == 148
             and counts.get("tendon_surface_count") == 2
             and counts.get("source_surface_count") == 150,
             "source surface coverage changed")
    _require(counts.get("single_closed_component_count") == 60
             and counts.get("closed_multi_component_count") == 6
             and counts.get("topology_defective_count") == 84,
             "topology admission counts changed")
    _require(counts.get("physical_volume_owner_count") == 0
             and counts.get("mechanical_mass_owner_count") == 0,
             "upstream audit unexpectedly promoted a physical owner")

    owners: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), "surface row is malformed")
        if row.get("status") != "single_closed_component":
            continue
        _require(row.get("layer") == "muscle", "a non-muscle closed component was admitted")
        member = row.get("member_id")
        stable_id = row.get("stable_id")
        volume = row.get("algebraic_volume_candidate_m3")
        _require(isinstance(member, str) and member and member not in seen_members,
                 "closed muscle member identity is invalid or duplicated")
        _require(type(stable_id) is int and stable_id > 0,
                 f"closed muscle stable id is invalid: {member}")
        _require(type(volume) in (int, float) and math.isfinite(float(volume)) and float(volume) > 0.0,
                 f"closed muscle volume is invalid: {member}")
        _require(row.get("physical_volume_owner") is None
                 and row.get("mechanical_mass_owner") is None
                 and row.get("material_owner") is None
                 and row.get("volumetric_active_force_owner") is None,
                 f"geometry audit already owns a physical field: {member}")
        member_sha = row.get("member_sha256")
        _require(isinstance(member_sha, str) and len(member_sha) == 64,
                 f"closed muscle source hash is invalid: {member}")
        seen_members.add(member)
        owners.append({
            "stable_id": stable_id,
            "member_id": member,
            "member_sha256": member_sha,
            "owner_id": f"muscle-volume:bodyparts3d:{member}",
            "volume_m3": float(volume),
            "ownership_scope": "single_closed_source_surface_geometry",
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "material_owner": False,
            "volumetric_active_force_owner": False,
        })
    owners.sort(key=lambda row: row["stable_id"])
    _require(len(owners) == 60, "closed muscle owner count changed")
    total = math.fsum(row["volume_m3"] for row in owners)
    audit_total = geometry.get("algebraic_volume_total_m3")
    _require(type(audit_total) in (int, float) and math.isfinite(float(audit_total)),
             "geometry audit total volume is invalid")
    _require(math.isclose(total, float(audit_total), rel_tol=2e-14, abs_tol=1e-15),
             "closed muscle volumes do not reproduce the audit total")
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.muscle-geometric-volume-candidate.1",
        "status": "partial",
        "source": {
            "audit": _relative(audit),
            "audit_sha256": audit_sha,
            "surface_manifest": source.get("surface_manifest"),
            "surface_manifest_sha256": source.get("surface_manifest_sha256"),
            "source_archive": source.get("source_archive"),
            "source_archive_sha256": source.get("source_archive_sha256"),
            "nhtiss4_payload_sha256": source.get("nhtiss4_payload_sha256"),
            "subject": "one adult male source package",
        },
        "coverage": {
            "source_muscle_surface_count": 148,
            "closed_muscle_component_count": len(owners),
            "closed_multi_component_count": counts["closed_multi_component_count"],
            "topology_defective_muscle_component_count": 82,
            "owner_fraction_of_muscle_surfaces": len(owners) / 148.0,
        },
        "geometry": {
            "closed_muscle_volume_total_m3": total,
            "volume_units": "m^3",
            "frame": document.get("coordinate_frame"),
        },
        "owners": owners,
        "qualification": {
            "source_surface_member_identity_bound": True,
            "single_closed_geometry_volume_bound": True,
            "physical_volume_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "material_calibration": False,
            "volumetric_active_force_owner": False,
            "activation_force_transfer": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "This record binds 60 single closed BodyParts3D muscle surface "
            "volume measurements to immutable source member identities. It is "
            "a geometric candidate only: it does not create a watertight FEM "
            "volume, assign density or mechanical mass, provide constitutive "
            "material, transfer active force, or calibrate one adult male. The "
            "six closed multi-component and 82 topology-defective muscle "
            "surfaces remain unadmitted, as do all tendon, fat and skin owners."
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
    return _sha256(payload)


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(audit=arguments.audit)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "status": result["status"],
        **result["coverage"],
        "closed_muscle_volume_total_m3": result["geometry"]["closed_muscle_volume_total_m3"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"muscle-geometric-volume: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

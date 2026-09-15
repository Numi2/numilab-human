"""Partition candidate muscle mass across the source route incidence graph.

The source route-volume join identifies 60 closed muscle surfaces and 82
route incidences, but a shared surface must not be counted once per route.
This compiler applies an explicit equal-incidence engineering split so the
candidate mass and volume budget can be inspected per route.  It does not
claim a measured fibre partition, create a physical volume owner, transfer
active force, or calibrate density/materials.
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
PROFILE = ROOT / "config/muscle-route-mass-partition-candidate.v1.json"
MASS_RECEIPT = ROOT / "Docs/media/muscle-tissue-mass-candidate-20260915/receipt-v1.json"
ROUTE_JOIN = ROOT / "Docs/media/muscle-route-volume-join-candidate-20260915/receipt-v1.json"
SCHEMA = "HumanPack.muscle-route-mass-partition-candidate.v1"
ROUTES = 416
CLOSED_SURFACES = 60
INCIDENCES = 82
UNBOUND_ROUTES = 238


class MusclePartitionError(HumanImportError):
    """A route-level candidate mass partition cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MusclePartitionError("muscle route mass partition: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} is not positive")
    return result


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        raise MusclePartitionError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "partition profile", canonical_required=True)
    required = {
        "schema", "id", "mass_receipt", "route_join_receipt", "share_policy",
        "expected_route_count", "expected_closed_surface_count",
        "expected_incidence_count", "expected_unbound_route_count", "boundary",
    }
    _require(set(profile) == required, "partition profile fields differ")
    _require(profile["schema"] == "numi.human.muscle-route-mass-partition-candidate.v1",
             "unsupported partition profile schema")
    _require(profile["id"] == "equal_incidence_route_mass_partition",
             "unsupported partition profile")
    for key in ("mass_receipt", "route_join_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_route_count": ROUTES,
        "expected_closed_surface_count": CLOSED_SURFACES,
        "expected_incidence_count": INCIDENCES,
        "expected_unbound_route_count": UNBOUND_ROUTES,
    }
    for key, value in expected.items():
        _require(profile[key] == value, f"{key} differs from the source contract")
    _require(profile["share_policy"] == "equal_incidence",
             "only the explicit equal-incidence policy is admitted")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "partition boundary is missing")
    return profile, digest


def compile_candidate(*, mass_receipt: Path = MASS_RECEIPT,
                      route_join: Path = ROUTE_JOIN,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    mass_path, route_path = Path(mass_receipt), Path(route_join)
    mass_doc, mass_sha = _read(mass_path, "muscle tissue mass receipt")
    route_doc, route_sha = _read(route_path, "muscle route-volume receipt")
    _require(mass_doc.get("schema") == "HumanPack.muscle-tissue-mass-candidate.v1"
             and mass_doc.get("status") == "partial", "muscle mass receipt changed")
    _require(route_doc.get("schema") == "HumanPack.muscle-route-volume-join-candidate.v1"
             and route_doc.get("status") == "partial", "route-volume receipt changed")

    mass_counts = mass_doc.get("counts", {})
    mass_qualification = mass_doc.get("qualification", {})
    _require(mass_counts.get("closed_volume_count") == CLOSED_SURFACES
             and mass_counts.get("source_muscle_surface_count") == 148
             and mass_counts.get("unadmitted_surface_count") == 88,
             "muscle mass coverage changed")
    for key in ("candidate_mass_budget", "source_volume_identity_bound"):
        _require(mass_qualification.get(key) is True, f"muscle mass lacks {key}")
    for key in ("skeletal_muscle_tissue_mass_owner", "mechanical_mass_owner",
                "activation_force_transfer", "material_calibration"):
        _require(mass_qualification.get(key) is False,
                 f"muscle mass promoted {key}")

    mass_rows: dict[str, dict[str, Any]] = {}
    for row in mass_doc.get("candidates", []):
        _require(isinstance(row, dict), "muscle mass row is malformed")
        member = row.get("member_id")
        _require(isinstance(member, str) and member and member not in mass_rows,
                 "muscle mass member identity is duplicated")
        member_sha = row.get("member_sha256")
        _require(isinstance(member_sha, str) and len(member_sha) == 64,
                 f"muscle mass member hash is invalid: {member}")
        mass_rows[member] = {
            "member_sha256": member_sha,
            "stable_id": row.get("stable_id"),
            "volume_m3": _positive(row.get("source_volume_m3"), f"{member} volume"),
            "mass_kg": _positive(row.get("candidate_mass_kg"), f"{member} mass"),
        }
    _require(len(mass_rows) == CLOSED_SURFACES, "muscle mass rows do not cover closed surfaces")

    route_counts = route_doc.get("counts", {})
    route_qualification = route_doc.get("qualification", {})
    _require(route_counts.get("source_route_count") == ROUTES
             and route_counts.get("closed_geometry_volume_owner_count") == CLOSED_SURFACES
             and route_counts.get("closed_geometry_route_incidence_count") == INCIDENCES
             and route_counts.get("routes_without_surface_binding") == UNBOUND_ROUTES,
             "route-volume coverage changed")
    for key in ("source_route_identity_bound", "closed_geometry_volume_identity_joined",
                "shared_surface_incidence_explicit", "unbound_routes_retained"):
        _require(route_qualification.get(key) is True, f"route-volume lacks {key}")
    _require(route_qualification.get("volume_partition_owner") is False
             and route_qualification.get("skeletal_muscle_tissue_mass_owner") is False,
             "route-volume receipt promoted a physical owner")

    route_rows = route_doc.get("route_rows")
    surface_rows = route_doc.get("closed_geometry_surface_rows")
    _require(isinstance(route_rows, list) and len(route_rows) == ROUTES
             and isinstance(surface_rows, list) and len(surface_rows) == CLOSED_SURFACES,
             "route-volume rows are incomplete")
    routes: dict[int, dict[str, Any]] = {}
    for row in route_rows:
        _require(isinstance(row, dict), "route row is malformed")
        index = row.get("source_actuator_index")
        _require(type(index) is int and 0 <= index < ROUTES and index not in routes,
                 "route identity is duplicated or out of range")
        routes[index] = row

    allocations: list[dict[str, Any]] = []
    route_allocations: dict[int, list[dict[str, Any]]] = {index: [] for index in range(ROUTES)}
    seen_surfaces: set[str] = set()
    for surface in sorted(surface_rows, key=lambda row: row.get("stable_id", -1)):
        _require(isinstance(surface, dict), "closed surface row is malformed")
        member = surface.get("member_id")
        _require(isinstance(member, str) and member not in seen_surfaces,
                 "closed surface identity is duplicated")
        seen_surfaces.add(member)
        source = mass_rows.get(member)
        _require(source is not None, f"closed surface has no mass candidate: {member}")
        _require(surface.get("member_sha256") == source["member_sha256"],
                 f"surface and mass hashes differ: {member}")
        route_ids = surface.get("source_actuator_indices")
        _require(isinstance(route_ids, list) and route_ids,
                 f"closed surface has no route incidence: {member}")
        _require(len(route_ids) == len(set(route_ids)),
                 f"closed surface repeats a route incidence: {member}")
        _require(all(type(index) is int and index in routes for index in route_ids),
                 f"closed surface route identity is invalid: {member}")
        _require(math.isclose(_positive(surface.get("volume_m3"), f"{member} joined volume"),
                              source["volume_m3"], rel_tol=2e-14, abs_tol=1e-15),
                 f"surface and mass volumes differ: {member}")
        incidence_count = len(route_ids)
        fraction = 1.0 / incidence_count
        for index in sorted(route_ids):
            allocation = {
                "source_actuator_index": index,
                "member_id": member,
                "member_sha256": source["member_sha256"],
                "stable_id": source["stable_id"],
                "incidence_count": incidence_count,
                "allocation_fraction": fraction,
                "candidate_volume_m3": source["volume_m3"] * fraction,
                "candidate_mass_kg": source["mass_kg"] * fraction,
                "physical_volume_owner": False,
                "mechanical_mass_owner": False,
                "volumetric_active_force_owner": False,
            }
            allocations.append(allocation)
            route_allocations[index].append(allocation)
    _require(len(seen_surfaces) == CLOSED_SURFACES, "closed surface coverage changed")
    _require(len(allocations) == INCIDENCES, "route incidence count changed")

    route_budget: list[dict[str, Any]] = []
    for index in range(ROUTES):
        row = routes[index]
        entries = sorted(route_allocations[index], key=lambda item: (item["stable_id"], item["member_id"]))
        route_budget.append({
            "source_actuator_index": index,
            "name": row.get("name"),
            "source_surface_count": row.get("surface_count"),
            "candidate_surface_count": len(entries),
            "candidate_surface_ids": [entry["member_id"] for entry in entries],
            "candidate_volume_m3": math.fsum(entry["candidate_volume_m3"] for entry in entries),
            "candidate_mass_kg": math.fsum(entry["candidate_mass_kg"] for entry in entries),
            "mass_partition_status": "equal_incidence_candidate" if entries else "no_closed_surface",
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "volumetric_active_force_owner": False,
        })

    source_volume = math.fsum(row["volume_m3"] for row in mass_rows.values())
    source_mass = math.fsum(row["mass_kg"] for row in mass_rows.values())
    allocated_volume = math.fsum(row["candidate_volume_m3"] for row in allocations)
    allocated_mass = math.fsum(row["candidate_mass_kg"] for row in allocations)
    volume_residual = allocated_volume - source_volume
    mass_residual = allocated_mass - source_mass
    _require(abs(volume_residual) <= 2.0e-15, "allocated candidate volume does not close")
    _require(abs(mass_residual) <= 2.0e-12, "allocated candidate mass does not close")
    routes_with_budget = sum(bool(entries) for entries in route_allocations.values())
    routes_without_surface_binding = route_counts["routes_without_surface_binding"]
    _require(routes_without_surface_binding == UNBOUND_ROUTES,
             "source route surface-binding count changed")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.muscle-route-mass-partition-candidate.1",
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "muscle_mass_receipt": _relative(mass_path),
            "muscle_mass_receipt_sha256": mass_sha,
            "route_volume_receipt": _relative(route_path),
            "route_volume_receipt_sha256": route_sha,
        },
        "policy": {
            "name": profile_doc["share_policy"],
            "description": "Each closed source surface candidate is divided equally across its explicit route incidences.",
            "measured_partition": False,
        },
        "counts": {
            "source_route_count": ROUTES,
            "closed_surface_count": CLOSED_SURFACES,
            "route_incidence_count": len(allocations),
            "routes_with_candidate_budget": routes_with_budget,
            "routes_without_closed_geometry": ROUTES - routes_with_budget,
            "routes_without_surface_binding": routes_without_surface_binding,
            "unadmitted_surface_count": 88,
        },
        "surface_allocations": allocations,
        "route_budgets": route_budget,
        "totals": {
            "source_candidate_volume_m3": source_volume,
            "allocated_candidate_volume_m3": allocated_volume,
            "candidate_volume_residual_m3": volume_residual,
            "source_candidate_mass_kg": source_mass,
            "allocated_candidate_mass_kg": allocated_mass,
            "candidate_mass_residual_kg": mass_residual,
            "candidate_partition_is_disjoint": True,
            "candidate_is_mechanical_mass": False,
        },
        "qualification": {
            "source_route_identity_bound": True,
            "source_surface_mass_bound": True,
            "equal_incidence_candidate_partition": True,
            "candidate_mass_and_volume_close": True,
            "unbound_routes_retained": True,
            "skeletal_muscle_tissue_mass_candidate": True,
            "skeletal_muscle_tissue_mass_owner": False,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "volumetric_active_force_owner": False,
            "activation_force_transfer": False,
            "activation_calibration": False,
            "material_calibration": False,
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
    result = compile_candidate(mass_receipt=arguments.mass_receipt,
                               route_join=arguments.route_join,
                               profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "status": result["status"],
        "sha256": digest,
        "output": str(output),
        "route_incidence_count": result["counts"]["route_incidence_count"],
        "allocated_candidate_mass_kg": result["totals"]["allocated_candidate_mass_kg"],
        "routes_without_surface_binding": result["counts"]["routes_without_surface_binding"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mass-receipt", type=Path, default=MASS_RECEIPT)
    parser.add_argument("--route-join", type=Path, default=ROUTE_JOIN)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (MusclePartitionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"muscle-route-mass-partition: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

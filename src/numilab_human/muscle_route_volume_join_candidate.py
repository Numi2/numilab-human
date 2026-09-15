"""Join source muscle routes to admitted geometric-volume candidates.

The route/surface candidate and the single-closed geometry candidate are both
source-bound, but their separate receipts do not prove which routes have an
available volume measurement.  This compiler performs that identity join and
keeps shared surface incidence explicit.  It does not partition a surface
between routes or promote geometry to tissue mass, constitutive material, or
active force authority.
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
PROFILE = ROOT / "config/muscle-route-volume-join-candidate.v1.json"
SURFACES = ROOT / "Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json"
VOLUMES = ROOT / "Docs/media/muscle-geometric-volume-candidate-20260914/receipt-v1.json"
SCHEMA = "HumanPack.muscle-route-volume-join-candidate.v1"
ROUTES = 416
MUSCLE_SURFACES = 148
TENDON_SURFACES = 2
CLOSED_VOLUMES = 60


class MuscleJoinError(HumanImportError):
    """A route-to-volume identity join cannot be compiled."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MuscleJoinError("muscle route-volume join: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError) as error:
        raise MuscleJoinError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "join profile", canonical_required=True)
    required = {
        "schema", "id", "surface_receipt", "volume_receipt", "expected_route_count",
        "expected_muscle_surface_count", "expected_tendon_surface_count",
        "expected_closed_volume_count", "expected_routes_without_surface_binding",
        "boundary",
    }
    _require(set(profile) == required, "join profile fields differ")
    _require(profile["schema"] == "numi.human.muscle-route-volume-join-candidate.v1",
             "unsupported join profile schema")
    _require(profile["id"] == "source_route_to_closed_geometry_incidence",
             "unsupported join profile")
    for key in ("surface_receipt", "volume_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_route_count": ROUTES,
        "expected_muscle_surface_count": MUSCLE_SURFACES,
        "expected_tendon_surface_count": TENDON_SURFACES,
        "expected_closed_volume_count": CLOSED_VOLUMES,
        "expected_routes_without_surface_binding": 238,
    }
    for key, expected_value in expected.items():
        _require(profile[key] == expected_value, f"{key} differs from the source contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "join boundary is missing")
    return profile, digest


def compile_candidate(*, surfaces: Path = SURFACES, volumes: Path = VOLUMES,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    surface_path = Path(surfaces)
    volume_path = Path(volumes)
    surface, surface_sha = _read(surface_path, "muscle surface receipt")
    volume, volume_sha = _read(volume_path, "muscle volume receipt")
    _require(surface.get("schema") == "HumanPack.soft-tissue-surface-candidate.v1",
             "unsupported muscle surface receipt")
    _require(volume.get("schema") == "HumanPack.muscle-geometric-volume-candidate.v1",
             "unsupported muscle volume receipt")
    _require(surface.get("status") == "partial" and volume.get("status") == "partial",
             "upstream muscle receipt status changed")
    surface_counts = surface.get("counts")
    volume_coverage = volume.get("coverage")
    _require(isinstance(surface_counts, dict) and isinstance(volume_coverage, dict),
             "upstream muscle counts are incomplete")
    _require(surface_counts.get("source_route_count") == ROUTES
             and surface_counts.get("muscle_surface_count") == MUSCLE_SURFACES
             and surface_counts.get("tendon_surface_count") == TENDON_SURFACES
             and surface_counts.get("routes_without_surface_binding")
             == profile_doc["expected_routes_without_surface_binding"],
             "source route/surface counts changed")
    _require(volume_coverage.get("source_muscle_surface_count") == MUSCLE_SURFACES
             and volume_coverage.get("closed_muscle_component_count") == CLOSED_VOLUMES,
             "source geometric-volume counts changed")

    route_rows = surface.get("route_rows")
    surface_rows = surface.get("surface_rows")
    volume_rows = volume.get("owners")
    _require(isinstance(route_rows, list) and len(route_rows) == ROUTES
             and isinstance(surface_rows, list) and len(surface_rows) == MUSCLE_SURFACES + TENDON_SURFACES
             and isinstance(volume_rows, list) and len(volume_rows) == CLOSED_VOLUMES,
             "upstream muscle identity rows are incomplete")

    routes_by_id: dict[int, dict[str, Any]] = {}
    for row in route_rows:
        _require(isinstance(row, dict), "route row is malformed")
        index = row.get("source_actuator_index")
        _require(type(index) is int and 0 <= index < ROUTES and index not in routes_by_id,
                 "route identity is invalid or duplicated")
        for key in ("mechanical_mass_owner", "physical_volume_owner", "volumetric_active_force_owner"):
            _require(row.get(key) is None, f"route {index} already owns {key}")
        routes_by_id[index] = row

    surfaces_by_id: dict[str, dict[str, Any]] = {}
    surface_routes: dict[str, list[int]] = {}
    for row in surface_rows:
        _require(isinstance(row, dict), "surface row is malformed")
        member = row.get("member_id")
        member_sha = row.get("member_sha256")
        layer = row.get("layer")
        _require(isinstance(member, str) and member and member not in surfaces_by_id,
                 "surface member identity is invalid or duplicated")
        _require(isinstance(member_sha, str) and len(member_sha) == 64,
                 f"surface hash is invalid for {member}")
        _require(layer in {"muscle", "tendon"}, f"surface layer is invalid for {member}")
        matches = row.get("matched_muscles")
        _require(isinstance(matches, list) and matches, f"surface has no route identity for {member}")
        identities: list[int] = []
        for match in matches:
            _require(isinstance(match, dict), f"surface route match is malformed for {member}")
            index = match.get("source_actuator_index")
            nodes = match.get("source_route_node_count")
            _require(type(index) is int and index in routes_by_id,
                     f"surface route identity is invalid for {member}")
            _require(type(nodes) is int and nodes >= 2,
                     f"surface route node count is invalid for {member}")
            identities.append(index)
        _require(len(set(identities)) == len(identities),
                 f"surface repeats a route identity for {member}")
        surfaces_by_id[member] = row
        surface_routes[member] = sorted(identities)

    volume_by_member: dict[str, dict[str, Any]] = {}
    for row in volume_rows:
        _require(isinstance(row, dict), "volume owner row is malformed")
        member = row.get("member_id")
        _require(isinstance(member, str) and member and member not in volume_by_member,
                 "volume member identity is invalid or duplicated")
        _require(member in surfaces_by_id and surfaces_by_id[member].get("layer") == "muscle",
                 f"volume member is not a muscle surface identity: {member}")
        _require(row.get("member_sha256") == surfaces_by_id[member].get("member_sha256"),
                 f"surface and volume hashes differ for {member}")
        value = row.get("volume_m3")
        _require(type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0.0,
                 f"volume candidate is invalid for {member}")
        for key in ("physical_volume_owner", "mechanical_mass_owner", "material_owner",
                    "volumetric_active_force_owner"):
            _require(row.get(key) is False, f"volume candidate promotes {key} for {member}")
        volume_by_member[member] = row

    volume_surface_rows = []
    route_incidence: dict[int, list[dict[str, Any]]] = {index: [] for index in routes_by_id}
    for member, row in sorted(volume_by_member.items(), key=lambda item: item[1]["stable_id"]):
        matched_routes = surface_routes[member]
        incidence = {
            "member_id": member,
            "member_sha256": row["member_sha256"],
            "stable_id": row["stable_id"],
            "volume_m3": float(row["volume_m3"]),
            "source_actuator_indices": matched_routes,
            "shared_route_incidence": len(matched_routes) > 1,
            "partitioned_between_routes": False,
        }
        volume_surface_rows.append(incidence)
        for index in matched_routes:
            route_incidence[index].append({
                "member_id": member,
                "stable_id": row["stable_id"],
                "volume_m3": float(row["volume_m3"]),
            })

    joined_routes = []
    for index in range(ROUTES):
        row = routes_by_id[index]
        incidences = sorted(route_incidence[index], key=lambda item: item["stable_id"])
        joined_routes.append({
            "source_actuator_index": index,
            "name": row.get("name"),
            "surface_count": row.get("surface_count"),
            "closed_geometry_surface_count": len(incidences),
            "closed_geometry_surface_ids": [item["member_id"] for item in incidences],
            "closed_geometry_volume_incidence_m3": math.fsum(item["volume_m3"] for item in incidences),
            "volume_partition_owner": False,
            "mechanical_mass_owner": None,
            "volumetric_active_force_owner": None,
        })
    routes_with_closed_geometry = sum(bool(row["closed_geometry_surface_ids"]) for row in joined_routes)
    route_incidence_count = sum(len(row["closed_geometry_surface_ids"]) for row in joined_routes)
    _require(routes_with_closed_geometry == 78 and route_incidence_count == 82,
             "closed geometry route incidence changed")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.muscle-route-volume-join-candidate.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "surface_receipt": _relative(surface_path),
            "surface_receipt_sha256": surface_sha,
            "volume_receipt": _relative(volume_path),
            "volume_receipt_sha256": volume_sha,
            "subject": "one adult male source package",
        },
        "counts": {
            "source_route_count": ROUTES,
            "muscle_surface_count": MUSCLE_SURFACES,
            "tendon_surface_count": TENDON_SURFACES,
            "closed_geometry_volume_owner_count": len(volume_surface_rows),
            "routes_with_surface_binding": surface_counts["routes_with_surface_binding"],
            "routes_without_surface_binding": surface_counts["routes_without_surface_binding"],
            "routes_with_closed_geometry": routes_with_closed_geometry,
            "closed_geometry_route_incidence_count": route_incidence_count,
        },
        "closed_geometry_surface_rows": volume_surface_rows,
        "route_rows": joined_routes,
        "qualification": {
            "source_route_identity_bound": True,
            "muscle_surface_identity_bound": True,
            "closed_geometry_volume_identity_joined": True,
            "unbound_routes_retained": True,
            "shared_surface_incidence_explicit": True,
            "volume_partition_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "material_calibration": False,
            "activation_force_transfer": False,
            "activation_calibration": False,
            "fat_geometry_owner": False,
            "skin_geometry_owner": False,
            "subject_calibration": False,
            "standing_walking": False,
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
    result = compile_candidate(surfaces=arguments.surfaces, volumes=arguments.volumes,
                               profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "status": result["status"],
        "source_route_count": result["counts"]["source_route_count"],
        "closed_geometry_volume_owner_count": result["counts"]["closed_geometry_volume_owner_count"],
        "routes_with_closed_geometry": result["counts"]["routes_with_closed_geometry"],
        "sha256": digest,
        "output": str(output),
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surfaces", type=Path, default=SURFACES)
    parser.add_argument("--volumes", type=Path, default=VOLUMES)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (MuscleJoinError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"muscle-route-volume-join: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

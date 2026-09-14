"""Compile a source-bound muscle-surface and soft-tissue ownership candidate.

The native reference contains 416 source muscle routes.  The NHTISS4 package
contains 148 muscle surfaces and two tendon surfaces with sparse route-body
bindings.  This compiler joins those identities, retains the 238 routes with
no emitted surface, and records that no fat geometry, tissue volume authority,
or active volumetric force owner exists.  It is a provenance/coverage gate;
it does not turn a visual surface into a continuum or a mechanical mass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.soft-tissue-surface-candidate.v1"
PROFILE = ROOT / "config/soft-tissue-surface-candidate.v1.json"


class SurfaceCandidateError(HumanImportError):
    """A source soft-tissue surface candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SurfaceCandidateError("soft tissue surface candidate: " + message)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SurfaceCandidateError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_json(path, "surface profile")
    _require(path.read_bytes() == canonical(profile) + b"\n",
             "surface profile is not canonical")
    required = {
        "schema", "id", "reference_manifest", "surface_manifest",
        "expected_route_count", "expected_surface_count", "expected_muscle_surface_count",
        "expected_tendon_surface_count", "boundary",
    }
    _require(set(profile) == required, "surface profile fields differ")
    _require(profile["schema"] == "numi.human.soft-tissue-surface-candidate.v1",
             "unsupported surface profile schema")
    _require(profile["id"] == "myosim_nhtiss4_route_surface_coverage",
             "unsupported surface profile")
    for key in (
        "expected_route_count", "expected_surface_count",
        "expected_muscle_surface_count", "expected_tendon_surface_count",
    ):
        value = profile[key]
        _require(type(value) is int and value > 0, f"{key} is invalid")
    _require(profile["expected_muscle_surface_count"]
             + profile["expected_tendon_surface_count"]
             == profile["expected_surface_count"],
             "surface count profile does not add up")
    for key in ("reference_manifest", "surface_manifest"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "surface profile boundary is missing")
    return profile, profile_sha


def _reference_rows(reference: dict[str, Any], expected: int) -> dict[int, dict[str, Any]]:
    _require(reference.get("schema") == "numi.human.myosim-fullbody-reference.v1",
             "unsupported MyoSim reference manifest")
    model = reference.get("model")
    _require(isinstance(model, dict) and model.get("source_muscle_count") == expected,
             "MyoSim reference muscle count differs")
    muscles = reference.get("muscles")
    _require(isinstance(muscles, list) and len(muscles) == expected,
             "MyoSim reference muscle table is incomplete")
    result: dict[int, dict[str, Any]] = {}
    for row in muscles:
        _require(isinstance(row, dict), "MyoSim reference muscle row is malformed")
        index = row.get("source_actuator_index")
        name = row.get("name")
        _require(type(index) is int and 0 <= index < expected and index not in result,
                 "MyoSim source actuator index is invalid or duplicated")
        _require(isinstance(name, str) and name.strip(),
                 f"MyoSim source muscle {index} has no name")
        for key in ("oracle_length_m", "oracle_force_n_at_activation_0_5"):
            value = row.get(key)
            _require(type(value) in (int, float) and math.isfinite(float(value)),
                     f"MyoSim source muscle {name} has invalid {key}")
        architecture = row.get("compliant_architecture")
        _require(isinstance(architecture, dict),
                 f"MyoSim source muscle {name} has no compliant architecture")
        result[index] = row
    _require(set(result) == set(range(expected)),
             "MyoSim source actuator indices do not cover the expected route table")
    return result


def compile_candidate(*, profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    reference_path = (ROOT / profile_doc["reference_manifest"]).resolve()
    surface_path = (ROOT / profile_doc["surface_manifest"]).resolve()
    _require(reference_path.is_relative_to(ROOT) and surface_path.is_relative_to(ROOT),
             "surface manifests resolve outside repository")
    reference, reference_sha = _read_json(reference_path, "MyoSim reference manifest")
    surface, surface_sha = _read_json(surface_path, "NHTISS4 surface manifest")
    expected_routes = profile_doc["expected_route_count"]
    muscles = _reference_rows(reference, expected_routes)

    _require(surface.get("schema")
             == "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
             "unsupported NHTISS4 surface manifest")
    coverage = surface.get("coverage")
    payload = surface.get("payload")
    source = surface.get("source")
    _require(isinstance(coverage, dict) and isinstance(payload, dict)
             and isinstance(source, dict), "NHTISS4 surface manifest is incomplete")
    _require(coverage.get("authored_myosim_muscle_count") == expected_routes,
             "NHTISS4 authored route count differs")
    expected_surfaces = profile_doc["expected_surface_count"]
    _require(coverage.get("emitted_surface_count") == expected_surfaces
             and coverage.get("configured_surface_count") == expected_surfaces,
             "NHTISS4 surface count differs")
    _require(payload.get("magic") == "NHTISS4"
             and payload.get("surface_count") == expected_surfaces
             and type(payload.get("sha256")) is str
             and len(payload["sha256"]) == 64,
             "NHTISS4 payload identity is incomplete")
    source_surfaces = source.get("surfaces")
    _require(isinstance(source_surfaces, list) and len(source_surfaces) == expected_surfaces,
             "NHTISS4 source surface rows are incomplete")

    expected_muscles = profile_doc["expected_muscle_surface_count"]
    expected_tendons = profile_doc["expected_tendon_surface_count"]
    layer_counts = {"muscle": 0, "tendon": 0}
    surface_ids: set[int] = set()
    member_ids: set[str] = set()
    route_surfaces: dict[int, list[dict[str, Any]]] = {index: [] for index in muscles}
    surface_rows: list[dict[str, Any]] = []
    for surface_row in source_surfaces:
        _require(isinstance(surface_row, dict), "NHTISS4 source surface row is malformed")
        stable_id = surface_row.get("stable_id")
        member_id = surface_row.get("member_id")
        layer = surface_row.get("layer")
        _require(type(stable_id) is int and stable_id > 0 and stable_id not in surface_ids,
                 "NHTISS4 stable surface ID is invalid or duplicated")
        _require(isinstance(member_id, str) and member_id and member_id not in member_ids,
                 "NHTISS4 source surface member ID is invalid or duplicated")
        _require(layer in layer_counts, f"NHTISS4 surface {stable_id} has an invalid layer")
        member_sha = surface_row.get("member_sha256")
        _require(isinstance(member_sha, str) and len(member_sha) == 64
                 and all(character in "0123456789abcdef" for character in member_sha),
                 f"NHTISS4 surface {stable_id} has invalid source hash")
        for key in ("triangle_count", "vertex_count"):
            _require(type(surface_row.get(key)) is int and surface_row[key] > 0,
                     f"NHTISS4 surface {stable_id} has invalid {key}")
        matched = surface_row.get("matched_muscles")
        _require(isinstance(matched, list) and matched,
                 f"NHTISS4 surface {stable_id} has no route identity")
        matched_rows = []
        for match in matched:
            _require(isinstance(match, dict), f"NHTISS4 surface {stable_id} route row is malformed")
            index = match.get("source_actuator_index")
            name = match.get("name")
            route_nodes = match.get("source_route_node_count")
            _require(type(index) is int and index in muscles and name == muscles[index]["name"],
                     f"NHTISS4 surface {stable_id} route identity is not source-bound")
            _require(type(route_nodes) is int and route_nodes >= 2,
                     f"NHTISS4 surface {stable_id} route node count is invalid")
            matched_rows.append({"source_actuator_index": index, "name": name,
                                 "source_route_node_count": route_nodes})
            route_surfaces[index].append({"stable_id": stable_id, "layer": layer})
        surface_ids.add(stable_id)
        member_ids.add(member_id)
        layer_counts[layer] += 1
        surface_rows.append({
            "stable_id": stable_id,
            "member_id": member_id,
            "member_sha256": member_sha,
            "label": surface_row.get("label"),
            "layer": layer,
            "triangle_count": surface_row["triangle_count"],
            "vertex_count": surface_row["vertex_count"],
            "matched_muscles": sorted(matched_rows,
                                       key=lambda row: row["source_actuator_index"]),
            "route_binding": surface_row.get("route_binding"),
        })
    _require(layer_counts["muscle"] == expected_muscles
             and layer_counts["tendon"] == expected_tendons,
             "NHTISS4 muscle/tendon layer counts differ")

    route_rows = []
    for index in sorted(muscles):
        row = muscles[index]
        bound = sorted(route_surfaces[index], key=lambda item: item["stable_id"])
        route_rows.append({
            "source_actuator_index": index,
            "name": row["name"],
            "route_nodes": row.get("route_nodes"),
            "oracle_length_m": row["oracle_length_m"],
            "oracle_force_n_at_activation_0_5": row["oracle_force_n_at_activation_0_5"],
            "surface_count": len(bound),
            "surface_ids": [item["stable_id"] for item in bound],
            "surface_layers": sorted({item["layer"] for item in bound}),
            "physical_volume_owner": None,
            "mechanical_mass_owner": None,
            "volumetric_active_force_owner": None,
        })
    bound_count = sum(bool(row["surface_count"]) for row in route_rows)
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.soft-tissue-surface-candidate.1",
        "status": "partial",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "reference_manifest": _relative(reference_path),
            "reference_manifest_sha256": reference_sha,
            "surface_manifest": _relative(surface_path),
            "surface_manifest_sha256": surface_sha,
            "nhtiss4_payload_sha256": payload["sha256"],
        },
        "counts": {
            "source_route_count": expected_routes,
            "surface_count": expected_surfaces,
            "muscle_surface_count": layer_counts["muscle"],
            "tendon_surface_count": layer_counts["tendon"],
            "routes_with_surface_binding": bound_count,
            "routes_without_surface_binding": expected_routes - bound_count,
            "fat_surface_count": 0,
            "skin_surface_count": 0,
            "physical_volume_owner_count": 0,
            "mechanical_mass_owner_count": 0,
            "volumetric_active_force_owner_count": 0,
        },
        "surface_rows": sorted(surface_rows, key=lambda row: row["stable_id"]),
        "route_rows": route_rows,
        "qualification": {
            "source_route_identity_bound": True,
            "muscle_surface_identity_bound": True,
            "tendon_surface_identity_bound": True,
            "unbound_routes_retained": True,
            "fat_geometry_present": False,
            "skin_geometry_present": False,
            "physical_tissue_volume_owner": False,
            "mechanical_mass_owner": False,
            "volumetric_active_muscle_owner": False,
            "muscle_force_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": profile_doc["boundary"],
    }


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
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
    result = compile_candidate(profile=arguments.profile)
    output = arguments.output.resolve()
    receipt_sha = _write_immutable(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": receipt_sha,
        "status": result["status"],
        **result["counts"],
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
        parser.exit(2, f"soft-tissue-surface: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

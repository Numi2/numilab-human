"""Bind the route-level muscle mass partition to the Human evidence bridge.

The published v6 source-evidence bridge already joins the one-male runtime,
organ/blood transfer, contact proxies, route/volume incidence, activation and
the explicit unresolved tissue candidates.  This extension adds the new
equal-incidence route-mass partition to that immutable bridge and keeps the
partition's physical, mechanical and calibration owners fail-closed.
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
PROFILE = ROOT / "config/human-source-evidence-bridge-extension.v1.json"
BASE_BRIDGE = ROOT / "Docs/media/human-source-evidence-bridge-20260915/receipt-v6.json"
ROUTE_PARTITION = ROOT / (
    "Docs/media/muscle-route-mass-partition-candidate-20260915/receipt-v1.json"
)
ROUTE_VOLUME = ROOT / (
    "Docs/media/muscle-route-volume-join-candidate-20260915/receipt-v1.json"
)
SCHEMA = "HumanPack.human-source-evidence-bridge-extension.v1"


class BridgeExtensionError(HumanImportError):
    """The route-mass partition cannot be bound to the published bridge."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeExtensionError("Human source-evidence bridge extension: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise BridgeExtensionError(f"{label} is not valid JSON") from error
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
    profile, digest = _read(path, "extension profile", canonical_required=True)
    required = {
        "schema", "id", "base_bridge", "route_partition", "route_volume",
        "expected_source_route_count", "expected_closed_surface_count",
        "expected_route_incidence_count", "expected_routes_with_candidate_budget",
        "expected_routes_without_surface_binding", "expected_unadmitted_surface_count",
        "expected_candidate_mass_kg", "expected_candidate_volume_m3", "boundary",
    }
    _require(set(profile) == required, "extension profile fields differ")
    _require(profile["schema"] == "numi.human.human-source-evidence-bridge-extension.v1",
             "unsupported extension profile schema")
    _require(profile["id"] == "bind_route_mass_partition_to_source_bridge",
             "unsupported extension profile")
    for key in ("base_bridge", "route_partition", "route_volume"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_source_route_count": 416,
        "expected_closed_surface_count": 60,
        "expected_route_incidence_count": 82,
        "expected_routes_with_candidate_budget": 78,
        "expected_routes_without_surface_binding": 238,
        "expected_unadmitted_surface_count": 88,
    }
    for key, expected_value in expected.items():
        _require(profile[key] == expected_value, f"{key} differs from the source contract")
    for key in ("expected_candidate_mass_kg", "expected_candidate_volume_m3"):
        _require(isinstance(profile[key], (int, float)) and profile[key] > 0,
                 f"{key} is invalid")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "extension boundary is missing")
    return profile, digest


def compile_candidate(*, base_bridge: Path = BASE_BRIDGE,
                      route_partition: Path = ROUTE_PARTITION,
                      route_volume: Path = ROUTE_VOLUME,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    base_path, partition_path, volume_path = (
        Path(base_bridge), Path(route_partition), Path(route_volume)
    )
    base, base_sha = _read(base_path, "base bridge receipt")
    partition, partition_sha = _read(partition_path, "route-mass partition receipt")
    volume, volume_sha = _read(volume_path, "route-volume receipt")

    _require(base.get("schema") == "HumanPack.human-source-evidence-bridge.v1"
             and base.get("status") == "partial", "base bridge receipt changed")
    base_graph = base.get("source_graph", {})
    base_qualification = base.get("qualification", {})
    base_inputs = base.get("inputs", {})
    base_muscle = base.get("domains", {}).get("muscle", {})
    _require(isinstance(base_graph, dict) and isinstance(base_qualification, dict)
             and isinstance(base_inputs, dict) and isinstance(base_muscle, dict),
             "base bridge receipt is incomplete")
    _require(base_graph.get("cross_domain_hashes_closed") is True
             and base_graph.get("physical_owner_count") == 0
             and base_qualification.get("integrated_human_qualification") is False,
             "base bridge qualification boundary changed")
    _require(base_muscle.get("source_route_count") == profile_doc["expected_source_route_count"]
             and base_muscle.get("closed_geometry_volume_owner_count") == profile_doc["expected_closed_surface_count"]
             and base_muscle.get("routes_with_closed_geometry") == profile_doc["expected_routes_with_candidate_budget"],
             "base bridge muscle counts changed")

    _require(volume.get("schema") == "HumanPack.muscle-route-volume-join-candidate.v1"
             and volume.get("status") == "partial", "route-volume receipt changed")
    volume_counts = volume.get("counts", {})
    volume_qualification = volume.get("qualification", {})
    _require(volume_counts.get("source_route_count") == profile_doc["expected_source_route_count"]
             and volume_counts.get("closed_geometry_volume_owner_count") == profile_doc["expected_closed_surface_count"]
             and volume_counts.get("closed_geometry_route_incidence_count") == profile_doc["expected_route_incidence_count"]
             and volume_counts.get("routes_with_closed_geometry") == profile_doc["expected_routes_with_candidate_budget"]
             and volume_counts.get("routes_without_surface_binding") == profile_doc["expected_routes_without_surface_binding"],
             "route-volume counts changed")
    for key in ("source_route_identity_bound", "muscle_surface_identity_bound",
                "closed_geometry_volume_identity_joined", "unbound_routes_retained",
                "shared_surface_incidence_explicit"):
        _require(volume_qualification.get(key) is True, f"route-volume lacks {key}")
    for key in ("volume_partition_owner", "skeletal_muscle_tissue_mass_owner",
                "activation_force_transfer", "activation_calibration", "subject_calibration"):
        _require(volume_qualification.get(key) is False, f"route-volume boundary changed for {key}")

    _require(partition.get("schema") == "HumanPack.muscle-route-mass-partition-candidate.v1"
             and partition.get("status") == "partial", "route-mass partition receipt changed")
    partition_counts = partition.get("counts", {})
    partition_source = partition.get("source", {})
    partition_totals = partition.get("totals", {})
    partition_qualification = partition.get("qualification", {})
    _require(partition_counts.get("source_route_count") == profile_doc["expected_source_route_count"]
             and partition_counts.get("closed_surface_count") == profile_doc["expected_closed_surface_count"]
             and partition_counts.get("route_incidence_count") == profile_doc["expected_route_incidence_count"]
             and partition_counts.get("routes_with_candidate_budget") == profile_doc["expected_routes_with_candidate_budget"]
             and partition_counts.get("routes_without_surface_binding") == profile_doc["expected_routes_without_surface_binding"]
             and partition_counts.get("unadmitted_surface_count") == profile_doc["expected_unadmitted_surface_count"],
             "route-mass partition counts changed")
    _require(partition_source.get("route_volume_receipt_sha256") == volume_sha
             and partition_source.get("muscle_mass_receipt_sha256") ==
             base_inputs.get("muscle_tissue_mass", {}).get("file_sha256"),
             "route-mass source hashes diverge from the base graph")
    _require(partition_qualification.get("source_route_identity_bound") is True
             and partition_qualification.get("source_surface_mass_bound") is True
             and partition_qualification.get("equal_incidence_candidate_partition") is True
             and partition_qualification.get("candidate_mass_and_volume_close") is True
             and partition_qualification.get("unbound_routes_retained") is True,
             "route-mass partition identity or closure changed")
    for key in ("physical_volume_owner", "mechanical_mass_owner", "volumetric_active_force_owner",
                "skeletal_muscle_tissue_mass_owner", "activation_force_transfer",
                "activation_calibration", "material_calibration", "subject_calibration",
                "standing", "walking"):
        _require(partition_qualification.get(key) is False,
                 f"route-mass partition boundary changed for {key}")

    expected_mass = float(profile_doc["expected_candidate_mass_kg"])
    expected_volume = float(profile_doc["expected_candidate_volume_m3"])
    candidate_mass = float(partition_totals.get("allocated_candidate_mass_kg", -1.0))
    candidate_volume = float(partition_totals.get("allocated_candidate_volume_m3", -1.0))
    _require(abs(candidate_mass - expected_mass) <= 1.0e-12
             and abs(candidate_volume - expected_volume) <= 1.0e-15
             and partition_totals.get("candidate_mass_residual_kg") == 0.0
             and partition_totals.get("candidate_volume_residual_m3") == 0.0
             and partition_totals.get("candidate_partition_is_disjoint") is True
             and partition_totals.get("candidate_is_mechanical_mass") is False,
             "route-mass totals are not closed")
    _require(abs(float(base_muscle.get("candidate_mass_kg", -1.0)) - candidate_mass) <= 1.0e-12,
             "base bridge candidate mass diverges")

    qualification = {
        "base_bridge_bound": True,
        "route_volume_incidence_bound": True,
        "route_mass_partition_bound": True,
        "equal_incidence_candidate_partition": True,
        "candidate_mass_and_volume_close": True,
        "skeletal_muscle_tissue_mass_candidate_bound": True,
        "physical_owner_count": 0,
        "physical_volume_owner": False,
        "mechanical_mass_owner": False,
        "volumetric_active_force_owner": False,
        "skeletal_muscle_tissue_mass_owner": False,
        "activation_force_transfer": False,
        "activation_calibration": False,
        "fat_geometry_and_mass": False,
        "material_calibration": False,
        "subject_calibration": False,
        "force_convergence": False,
        "standing": False,
        "recovery": False,
        "walking": False,
        "integrated_human_qualification": False,
    }
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.human-source-evidence-bridge-extension.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "base_bridge": _input(base_path, base, base_sha),
            "route_mass_partition": _input(partition_path, partition, partition_sha),
            "route_volume": _input(volume_path, volume, volume_sha),
            "profile": {"path": _relative(Path(profile)), "schema": profile_doc["schema"],
                        "file_sha256": profile_sha},
        },
        "base_bridge": {
            "schema": base["schema"],
            "sha256": base_sha,
            "cross_domain_hashes_closed": True,
            "physical_owner_count": 0,
            "integrated_human_qualification": False,
        },
        "domains": {
            "muscle": {
                "source_route_count": partition_counts["source_route_count"],
                "closed_surface_count": partition_counts["closed_surface_count"],
                "route_incidence_count": partition_counts["route_incidence_count"],
                "routes_with_candidate_budget": partition_counts["routes_with_candidate_budget"],
                "routes_without_surface_binding": partition_counts["routes_without_surface_binding"],
                "unadmitted_surface_count": partition_counts["unadmitted_surface_count"],
                "candidate_mass_kg": candidate_mass,
                "candidate_volume_m3": candidate_volume,
                "candidate_partition_is_disjoint": True,
                "physical_volume_owner": False,
                "mechanical_mass_owner": False,
                "volumetric_active_force_owner": False,
                "activation_force_transfer": False,
                "activation_calibration": False,
            }
        },
        "source_graph": {
            "base_bridge_sha256": base_sha,
            "route_volume_sha256": volume_sha,
            "route_mass_partition_sha256": partition_sha,
            "cross_domain_hashes_closed": True,
            "physical_owner_count": 0,
        },
        "qualification": qualification,
        "blockers": [
            {"id": "muscle_mass_owner", "status": "open",
             "reason": "The equal-incidence partition closes a candidate mass and volume budget, but physical volume, mechanical mass and active force owners remain absent."},
            {"id": "activation_calibration", "status": "open",
             "reason": "Route mass is bound to the source graph, but activation-force transfer and measured activation calibration remain absent."},
            {"id": "force_convergence", "status": "open",
             "reason": "The parent bridge still reports unresolved complete generalized equilibrium and temporal force convergence."},
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
    result = compile_candidate(base_bridge=arguments.base_bridge,
                               route_partition=arguments.route_partition,
                               route_volume=arguments.route_volume,
                               profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output)}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-bridge", type=Path, default=BASE_BRIDGE)
    parser.add_argument("--route-partition", type=Path, default=ROUTE_PARTITION)
    parser.add_argument("--route-volume", type=Path, default=ROUTE_VOLUME)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (BridgeExtensionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"human-source-evidence-bridge-extension: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

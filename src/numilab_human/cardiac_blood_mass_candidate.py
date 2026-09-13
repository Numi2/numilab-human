"""Bind cardiac cavity ownership candidates to source blood mass budgets.

The source partition contains two disjoint right-heart geometric conventions,
while CVSim21 remains the hydraulic volume authority.  This module binds both
conventions to the four hydraulic chambers and an explicit unresolved density
candidate.  It records the volume scale needed to compare surface geometry to
hydraulic storage but never promotes that scale to a physical volume owner,
mechanical mass, or selected biological interface.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import cardiac_cavity_ownership as ownership
from . import cvsim21
from . import cvsim21_blood_mass_step as blood_mass
from .model import ImportError as HumanImportError


ROOT = Path(__file__).resolve().parents[2]
PARTITION = ROOT / "Docs/media/cardiac-partition-20260912/partition.json"
BRIDGE = ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json"
BODY_LINKS = ROOT / "Docs/media/organ-cardiac-cavity-body-link-20260913/body-links.json"
CVSIM_CONFIG = ROOT / "config/cvsim21-source.v1.json"
BLOOD_OWNER = ROOT / "config/cvsim21-blood-mass-owner.v1.json"
SCHEMA = "HumanPack.cardiac-blood-mass-candidate.v1"
EXPECTED_CANDIDATES = ("right_atrium_priority", "right_ventricle_priority")
EXPECTED_LABELS = ("right_atrium", "right_ventricle", "left_atrium", "left_ventricle")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac blood mass candidate: " + message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise HumanImportError("cardiac blood mass candidate requires finite JSON") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value) + b"\n").hexdigest()


def file_digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = cvsim21.read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise HumanImportError(f"cannot read candidate input {path}") from error
    require(isinstance(value, dict), f"candidate input is not an object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)),
            f"{label} is not finite")
    return float(value)


def positive(value: Any, label: str) -> float:
    result = finite(value, label)
    require(result > 0.0, f"{label} is not positive")
    return result


def decode_rational(value: Any, label: str) -> float:
    require(isinstance(value, list) and len(value) == 2
            and all(isinstance(item, str) for item in value),
            f"{label} is not an encoded rational")
    try:
        numerator = int(value[0], 16)
        denominator = int(value[1], 16)
    except ValueError as error:
        raise HumanImportError(f"cardiac blood mass candidate: {label} is malformed") from error
    require(denominator > 0, f"{label} has a nonpositive denominator")
    result = float(Fraction(numerator, denominator))
    require(math.isfinite(result) and result > 0.0, f"{label} is not positive")
    return result


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _load_partition(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    document = read_json(path)
    require(document.get("schema") == "HumanPack.cardiac-cavity-partition-candidates.v1",
            "unsupported cardiac partition schema")
    require(document.get("selection") is None and document.get("physical_stepping") is False,
            "partition candidate is selected or stepped")
    qualification = document.get("qualification")
    require(isinstance(qualification, dict)
            and qualification.get("both_emitted_candidates_have_disjoint_interiors") is True
            and qualification.get("blood_tissue_mass_partition") is False
            and qualification.get("anatomical_valve_interface_selected") is False,
            "partition boundary changed")
    candidates = document.get("candidates")
    require(isinstance(candidates, dict) and set(candidates) == set(EXPECTED_CANDIDATES),
            "partition candidates differ")
    volumes: dict[str, float] = {}
    for candidate_id in EXPECTED_CANDIDATES:
        candidate = candidates[candidate_id]
        require(isinstance(candidate, dict)
                and candidate.get("four_cavity_interiors_disjoint") is True
                and candidate.get("biological_selection") is False
                and candidate.get("mechanical_mass_assigned") is False,
                f"candidate boundary changed: {candidate_id}")
        regions = candidate.get("emitted_float64_audit", {}).get("per_region")
        require(isinstance(regions, dict) and set(regions) == {"right_atrium", "right_ventricle"},
                f"candidate regions differ: {candidate_id}")
        for label, row in regions.items():
            volumes[f"{candidate_id}:{label}"] = decode_rational(
                row.get("moments", {}).get("volume_m3"),
                f"{candidate_id}:{label} surface volume",
            )
    return document, volumes


def _load_bridge(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = read_json(path)
    require(document.get("schema") == "HumanPack.organ-blood-cavity-bridge.v1",
            "unsupported organ/blood cavity bridge schema")
    qualification = document.get("qualification")
    require(isinstance(qualification, dict)
            and qualification.get("physical_volume_authority_assigned") is False
            and qualification.get("blood_mass_assigned") is False
            and qualification.get("cavity_domains_disjoint") is False,
            "cavity bridge boundary changed")
    rows = document.get("bindings")
    require(isinstance(rows, list) and len(rows) == 4, "cavity bridge must contain four bindings")
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("source_label")
        require(label in EXPECTED_LABELS and label not in by_label,
                "cavity bridge labels are incomplete or duplicated")
        positive(row.get("hydraulic_initial_volume_m3"), f"{label} hydraulic volume")
        positive(row.get("source_surface_integral_volume_m3"), f"{label} source surface volume")
        require(isinstance(row.get("hydraulic_volume_authority"), str)
                and row["hydraulic_volume_authority"].startswith("source_blood:CVSim21:v"),
                f"{label} hydraulic authority is not CVSim21")
        by_label[label] = row
    require(set(by_label) == set(EXPECTED_LABELS), "cavity bridge labels do not cover four chambers")
    return document, by_label


def compile_candidate(*, partition: Path = PARTITION, bridge: Path = BRIDGE,
                      body_links: Path = BODY_LINKS,
                      cvsim_config: Path = CVSIM_CONFIG,
                      blood_owner: Path = BLOOD_OWNER) -> dict[str, Any]:
    partition = Path(partition)
    bridge = Path(bridge)
    body_links = Path(body_links)
    cvsim_config = Path(cvsim_config)
    blood_owner = Path(blood_owner)
    partition_document, partition_volumes = _load_partition(partition)
    bridge_document, bridge_by_label = _load_bridge(bridge)
    ownership_document = ownership.compile_ownership(
        partition=partition, bridge=bridge, body_links=body_links,
    )
    native, lowering = cvsim21.compile_source(config=read_json(cvsim_config))
    owner_document = read_json(blood_owner)
    validated_owner = blood_mass._validate_owner(native, owner_document)
    require(validated_owner["density"] > 0.0, "candidate density is not positive")
    require(owner_document["density_provenance"]["kind"] == "engineering_candidate_unresolved",
            "candidate density has been promoted to calibrated source data")

    ownership_by_candidate = {
        row["candidate_id"]: row for row in ownership_document["candidates"]
    }
    candidate_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    hydraulic_total = math.fsum(row["hydraulic_initial_volume_m3"]
                                for row in bridge_by_label.values())
    expected_hydraulic_total = math.fsum(
        row["initial_volume_m3"] for row in native["compartments"]
        if row["stable_identifier"] in {16, 17, 20, 21}
    )
    require(math.isclose(hydraulic_total, expected_hydraulic_total,
                         rel_tol=0.0, abs_tol=1e-18),
            "bridge and CVSim cardiac hydraulic totals disagree")
    for candidate_id in EXPECTED_CANDIDATES:
        candidate_meta = ownership_by_candidate[candidate_id]
        rows: list[dict[str, Any]] = []
        candidate_surface_total = 0.0
        candidate_mass_total = 0.0
        for label in EXPECTED_LABELS:
            bridge_row = bridge_by_label[label]
            surface_volume = (
                partition_volumes[f"{candidate_id}:{label}"]
                if label in {"right_atrium", "right_ventricle"}
                else bridge_row["source_surface_integral_volume_m3"]
            )
            if label in {"right_atrium", "right_ventricle"}:
                # The priority region is unchanged from the original source;
                # the other region is the explicit closure alternative.
                priority = candidate_meta["priority_region"]
                if label == priority:
                    require(abs(surface_volume - bridge_row["source_surface_integral_volume_m3"])
                            <= 1e-15,
                            f"{candidate_id} priority volume changed source identity")
            hydraulic_volume = bridge_row["hydraulic_initial_volume_m3"]
            scale = hydraulic_volume / surface_volume
            mass = validated_owner["density"] * hydraulic_volume
            candidate_surface_total += surface_volume
            candidate_mass_total += mass
            rows.append({
                "candidate_id": candidate_id,
                "source_label": label,
                "semantic_id": bridge_row["semantic_id"],
                "compartment_stable_identifier": bridge_row["compartment_stable_identifier"],
                "hydraulic_volume_authority": bridge_row["hydraulic_volume_authority"],
                "source_surface_integral_volume_m3": surface_volume,
                "hydraulic_initial_volume_m3": hydraulic_volume,
                "hydraulic_to_surface_scale": scale,
                "candidate_mass_kg": mass,
                "candidate_receiver_id": f"cardiac_candidate:{candidate_id}:{label}",
                "physical_volume_owner": None,
                "mechanical_mass_owner": None,
                "tissue_exchange_owner": None,
            })
        candidate_rows.extend(rows)
        candidate_summaries.append({
            "candidate_id": candidate_id,
            "priority_region": candidate_meta["priority_region"],
            "geometry_sha256": candidate_meta["geometry_sha256"],
            "source_surface_total_m3": candidate_surface_total,
            "hydraulic_initial_total_m3": hydraulic_total,
            "candidate_mass_total_kg": candidate_mass_total,
            "mass_conserved_against_density": math.isclose(
                candidate_mass_total, validated_owner["density"] * hydraulic_total,
                rel_tol=0.0, abs_tol=1e-12,
            ),
            "physical_volume_owner_assigned": False,
            "selection": False,
        })
    identity = {
        "partition_sha256": file_digest(partition),
        "bridge_sha256": file_digest(bridge),
        "body_links_sha256": file_digest(body_links),
        "ownership_identity_sha256": ownership_document["identity_sha256"],
        "native_content_sha256": owner_document["native_content_sha256"],
        "owner_graph_sha256": digest(owner_document),
        "source_lowering_manifest_sha256": digest(lowering),
        "candidate_rows": candidate_rows,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.cardiac-blood-mass-candidate.1",
        "inputs": {
            "partition": {"path": _relative(partition), "sha256": identity["partition_sha256"]},
            "bridge": {"path": _relative(bridge), "sha256": identity["bridge_sha256"]},
            "body_links": {"path": _relative(body_links), "sha256": identity["body_links_sha256"]},
            "cvsim_config": {"path": _relative(cvsim_config), "sha256": file_digest(cvsim_config)},
            "blood_owner": {"path": _relative(blood_owner), "sha256": file_digest(blood_owner)},
        },
        "identity_sha256": digest(identity),
        "source_native_sha256": native["source_graph_sha256"],
        "native_content_sha256": owner_document["native_content_sha256"],
        "density": {
            "value_kg_per_m3": validated_owner["density"],
            "provenance": owner_document["density_provenance"],
            "calibrated": False,
        },
        "selection": None,
        "candidates": candidate_summaries,
        "bindings": candidate_rows,
        "mass_budget": {
            "hydraulic_cardiac_volume_m3": hydraulic_total,
            "candidate_mass_kg": validated_owner["density"] * hydraulic_total,
            "both_candidates_share_hydraulic_budget": all(
                row["mass_conserved_against_density"] for row in candidate_summaries
            ),
            "physical_volume_owner_assigned": False,
            "mechanical_mass_owner_assigned": False,
            "tissue_exchange_assigned": False,
        },
        "qualification": {
            "source_candidate_geometry_bound": True,
            "two_disjoint_right_heart_candidates": True,
            "four_chamber_hydraulic_binding": True,
            "aggregate_mass_budget_conserved": True,
            "unresolved_density_candidate_explicit": True,
            "biological_interface_selected": False,
            "physical_volume_owner_assigned": False,
            "mechanical_mass_assigned": False,
            "two_way_blood_tissue_transfer": False,
            "activation_calibration": False,
            "material_density_calibrated": False,
            "physiological_calibration": False,
            "standing": False,
            "walking": False,
        },
        "boundary": (
            "Two exact source-preserving right-heart geometry candidates are bound to "
            "the four CVSim hydraulic chamber authorities and an explicit unresolved "
            "density candidate. Hydraulic-to-surface scale factors are comparison "
            "metadata only; no candidate is selected and no anatomical physical-volume, "
            "mechanical mass, tissue exchange, activation, material, calibration, or "
            "behavior claim is admitted."
        ),
    }
    canonical(result)
    return result


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", type=Path, default=PARTITION)
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--body-links", type=Path, default=BODY_LINKS)
    parser.add_argument("--cvsim-config", type=Path, default=CVSIM_CONFIG)
    parser.add_argument("--blood-owner", type=Path, default=BLOOD_OWNER)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    result = compile_candidate(
        partition=args.partition, bridge=args.bridge, body_links=args.body_links,
        cvsim_config=args.cvsim_config, blood_owner=args.blood_owner,
    )
    output = args.output.resolve()
    sha = immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": sha,
                      "candidates": len(result["candidates"]), "selection": None,
                      "candidate_mass_kg": result["mass_budget"]["candidate_mass_kg"],
                      "physical_volume_owner_assigned": False}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"cardiac-blood-mass-candidate: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

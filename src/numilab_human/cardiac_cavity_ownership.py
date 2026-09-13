"""Bind disjoint cardiac cavity geometry candidates to the Human body frame.

The partition compiler already constructs two exact source-preserving
right-heart ownership candidates.  This owner joins those candidates to the
four-cavity/body-link receipt without selecting a biological valve interface
or assigning physical blood mass.
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
SCHEMA = "HumanPack.organ-cardiac-cavity-ownership-candidates.v1"
PARTITION = ROOT / "Docs/media/cardiac-partition-20260912/partition.json"
BRIDGE = ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json"
BODY_LINKS = ROOT / "Docs/media/organ-cardiac-cavity-body-link-20260913/body-links.json"
EXPECTED_CANDIDATES = ("right_atrium_priority", "right_ventricle_priority")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac cavity ownership admission: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    value = read_json(path)
    require(isinstance(value, dict), f"input is not an object: {path}")
    return value


def _hex(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value),
            f"{label} is not a lowercase SHA-256")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{label} is not finite and nonnegative")
    return float(value)


def compile_ownership(*, partition: Path = PARTITION, bridge: Path = BRIDGE,
                      body_links: Path = BODY_LINKS) -> dict[str, Any]:
    partition = Path(partition)
    bridge = Path(bridge)
    body_links = Path(body_links)
    source = _read(partition)
    require(source.get("schema") == "HumanPack.cardiac-cavity-partition-candidates.v1",
            "unsupported cardiac partition schema")
    require(source.get("selection") is None and source.get("physical_stepping") is False,
            "partition artifact already selected or stepped a candidate")
    require(source.get("native_payload_changed") is False and
            source.get("hydraulic_parameters_changed") is False and
            _finite_nonnegative(source.get("added_mechanical_mass_kg"), "added mechanical mass") == 0.0,
            "partition artifact changes a physical owner")
    qualification = source.get("qualification")
    require(isinstance(qualification, dict) and
            qualification.get("both_emitted_candidates_have_disjoint_interiors") is True and
            qualification.get("source_union_preserved_exactly_in_rational_construction") is True and
            qualification.get("source_exclusive_regions_preserved_exactly_in_rational_construction") is True and
            qualification.get("anatomical_valve_interface_selected") is False and
            qualification.get("blood_tissue_mass_partition") is False,
            "partition qualification boundary changed")
    candidates = source.get("candidates")
    require(isinstance(candidates, dict) and tuple(candidates) == EXPECTED_CANDIDATES,
            "partition candidate identity or ordering changed")
    candidate_rows: list[dict[str, Any]] = []
    expected_priorities = {
        "right_atrium_priority": "right_atrium",
        "right_ventricle_priority": "right_ventricle",
    }
    for candidate_id in EXPECTED_CANDIDATES:
        candidate = candidates[candidate_id]
        require(isinstance(candidate, dict), f"candidate is malformed: {candidate_id}")
        require(candidate.get("four_cavity_interiors_disjoint") is True and
                candidate.get("biological_selection") is False and
                candidate.get("mechanical_mass_assigned") is False,
                f"candidate boundary changed: {candidate_id}")
        require(candidate.get("priority_region") == expected_priorities[candidate_id] and
                isinstance(candidate.get("ownership_convention"), str) and
                candidate["ownership_convention"].strip(),
                f"candidate ownership convention changed: {candidate_id}")
        candidate_rows.append({
            "candidate_id": candidate_id,
            "priority_region": candidate.get("priority_region"),
            "ownership_convention": candidate.get("ownership_convention"),
            "geometry_sha256": _hex(candidate.get("geometry_sha256"), candidate_id + " geometry"),
            "four_cavity_interiors_disjoint": True,
            "biological_selection": False,
            "mechanical_mass_assigned": False,
        })

    bridge_doc = _read(bridge)
    require(bridge_doc.get("schema") == "HumanPack.organ-blood-cavity-bridge.v1",
            "unsupported organ/blood bridge schema")
    bridge_reference = bridge_doc.get("cvsim_cavity_reference")
    require(isinstance(bridge_reference, dict), "bridge has no CVSim cavity identity")
    source_geometry_sha = _hex(source.get("source_geometry_sha256"), "partition source geometry")
    require(bridge_reference.get("cavity_geometry_sha256") == source_geometry_sha,
            "partition source geometry disagrees with the bridge")
    require(bridge_doc.get("qualification", {}).get("cavity_domains_disjoint") is False,
            "bridge already promotes disjoint cavity ownership")

    body_doc = _read(body_links)
    require(body_doc.get("schema") == "HumanPack.organ-cardiac-cavity-body-link-registration.v1",
            "unsupported cardiac cavity body-link schema")
    body_qualification = body_doc.get("qualification", {})
    require(body_qualification.get("body_link_registration") is True and
            body_qualification.get("physical_volume_authority_assigned") is False and
            body_qualification.get("blood_mass_assigned") is False,
            "cardiac cavity body-link boundary changed")
    rows = body_doc.get("bindings")
    require(isinstance(rows, list) and len(rows) == 4 and
            {row.get("member_id") for row in rows} == {"FJ2422", "FJ2423", "FJ2424", "FJ2425"} and
            all(row.get("myosim_body") == "torso" and row.get("source_body_id") == 9 and
                row.get("core_body_index") == 20 and row.get("body_link_registration") is True
                for row in rows),
            "cardiac cavity body-frame bindings changed")

    shared_volume_m3 = _finite_nonnegative(source.get("shared_source_volume_m3"),
                                            "shared source volume")
    shared_volume_ml = _finite_nonnegative(source.get("shared_source_volume_ml"),
                                           "shared source volume mL")
    require(abs(shared_volume_m3 * 1.0e6 - shared_volume_ml) <= 1.0e-12,
            "shared source volume units disagree")

    identity = {
        "partition_sha256": _sha256(partition.read_bytes()),
        "bridge_sha256": _sha256(bridge.read_bytes()),
        "body_links_sha256": _sha256(body_links.read_bytes()),
        "source_geometry_sha256": source_geometry_sha,
        "candidate_rows": candidate_rows,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.cardiac-cavity-ownership.1",
        "source": {
            "partition": str(partition.relative_to(ROOT)) if partition.is_relative_to(ROOT) else str(partition),
            "partition_sha256": identity["partition_sha256"],
            "bridge": str(bridge.relative_to(ROOT)) if bridge.is_relative_to(ROOT) else str(bridge),
            "bridge_sha256": identity["bridge_sha256"],
            "body_links": str(body_links.relative_to(ROOT)) if body_links.is_relative_to(ROOT) else str(body_links),
            "body_links_sha256": identity["body_links_sha256"],
            "source_geometry_sha256": source_geometry_sha,
            "shared_source_volume_m3": shared_volume_m3,
            "shared_source_volume_ml": shared_volume_ml,
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "candidates": candidate_rows,
        "selection": None,
        "qualification": {
            "source_union_preserved": True,
            "source_exclusive_regions_preserved": True,
            "geometric_candidate_interiors_disjoint": True,
            "body_frame_registered": True,
            "biological_valve_interface_selected": False,
            "cavity_domains_disjoint_in_original_source": False,
            "physical_volume_authority_assigned": False,
            "blood_mass_assigned": False,
            "material_density_calibrated": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_blood_tissue_transfer": False,
            "physiological_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Both exact source-preserving right-heart ownership candidates are "
            "hash-bound to the four-cavity bridge and torso source/core body links. "
            "No candidate is biologically selected; the original source domains "
            "remain overlapping, and no physical volume, density, blood mass, "
            "pressure reaction, tissue exchange, calibration, or behavior is "
            "admitted."
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
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=PARTITION)
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--body-links", type=Path, default=BODY_LINKS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_ownership(partition=args.partition, bridge=args.bridge,
                                   body_links=args.body_links)
        digest = immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "candidates": len(result["candidates"]),
                          "selection": None, "blood_mass_assigned": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"cardiac cavity ownership admission: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

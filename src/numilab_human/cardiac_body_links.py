"""Bind the four source cardiac cavity surfaces to the exact MyoSim body frame.

This is a provenance subgate for the organ/blood bridge.  It does not promote
the atlas surface integral to a physical cavity volume, blood mass, wall
material, pressure reaction, or calibrated physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, load_anatomy, read_json
from .vessel_body_links import load_body_catalog

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-cardiac-cavity-body-link-registration.v1"
MAP = ROOT / "config/bodyparts3d-myosim-cardiac-cavity-map.v1.json"
BRIDGE = ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json"
HUMAN_MANIFEST = ROOT / "Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json"
EXPECTED = (
    ("FJ2424", "cavity of right atrium", "FMA11359"),
    ("FJ2423", "cavity of right ventricle", "FMA9291"),
    ("FJ2425", "cavity of left atrium", "FMA9465"),
    ("FJ2422", "cavity of left ventricle", "FMA9466"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac cavity body-link registration: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} is malformed")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    value = read_json(path)
    require(isinstance(value, dict), f"input is not an object: {path}")
    return value


def compile_body_links(*, sources: Path = ROOT / "Sources",
                       source_lock: Path = ROOT / "sources.lock.json",
                       cavity_map: Path = MAP,
                       bridge: Path = BRIDGE,
                       human_manifest: Path = HUMAN_MANIFEST) -> dict[str, Any]:
    mapping = _read_json(Path(cavity_map))
    require(mapping.get("schema") == "numi.human.bodyparts3d-myosim-cardiac-cavity-map.v1",
            "unsupported cardiac cavity map schema")
    entries = mapping.get("entries")
    require(isinstance(entries, list) and len(entries) == 4, "cardiac cavity map must contain four entries")
    by_member: dict[str, dict[str, Any]] = {}
    anatomy = load_anatomy(Path(sources), Path(source_lock))
    tables = anatomy.get("tables", {})
    for entry in entries:
        require(isinstance(entry, dict), "cardiac cavity map entry is malformed")
        required = {"concept_id", "source_name", "member_id", "hierarchy", "layer", "myosim_body"}
        require(required <= entry.keys(), "cardiac cavity map entry is incomplete")
        require(entry["layer"] == "cardiac_cavity" and entry["hierarchy"] == "part_of",
                "cardiac cavity map layer or hierarchy changed")
        member = entry["member_id"]
        require(isinstance(member, str) and member not in by_member, "duplicate cavity member")
        relation = tables.get(entry["hierarchy"], {})
        key = (entry["concept_id"], entry["source_name"])
        require(key in relation and member in relation[key],
                f"cardiac cavity source relation drifted: {member}")
        by_member[member] = entry
    require(tuple((x["member_id"], x["source_name"], x["concept_id"]) for x in entries) == EXPECTED,
            "cardiac cavity map identity or ordering changed")

    bridge_doc = _read_json(Path(bridge))
    require(bridge_doc.get("schema") == "HumanPack.organ-blood-cavity-bridge.v1",
            "unsupported organ/blood bridge schema")
    bridge_bindings = bridge_doc.get("bindings")
    require(isinstance(bridge_bindings, list) and len(bridge_bindings) == 4,
            "organ/blood bridge must contain four cavities")
    bridge_by_member = {row.get("member_id"): row for row in bridge_bindings if isinstance(row, dict)}
    require(set(bridge_by_member) == {x[0] for x in EXPECTED}, "bridge cavity member coverage changed")
    catalog = load_body_catalog(Path(human_manifest))
    torso = catalog["bodies"].get("torso")
    require(torso is not None and torso["source_body_id"] == 9 and torso["core_body_index"] == 20,
            "pinned MyoSim torso body identity changed")

    linked: list[dict[str, Any]] = []
    for member, source_name, concept_id in EXPECTED:
        entry = by_member[member]
        row = bridge_by_member[member]
        require(row.get("semantic_id") == "FMA:" + concept_id[3:],
                f"bridge semantic identity changed: {member}")
        require(row.get("source_moment_status") == "computed_single_closed_component",
                f"bridge source moment is not admitted: {member}")
        source_member_sha = row.get("source_member_sha256")
        require(isinstance(source_member_sha, str) and len(source_member_sha) == 64
                and all(character in "0123456789abcdef" for character in source_member_sha),
                f"bridge source member hash is malformed: {member}")
        require(row.get("physical_volume_owner") is None and row.get("mechanical_mass_owner") is None,
                f"bridge already promoted a physical owner: {member}")
        require(entry["myosim_body"] == "torso", f"cardiac cavity body link changed: {member}")
        linked.append({
            "member_id": member,
            "source_name": source_name,
            "hydraulic_source_label": row.get("source_label"),
            "semantic_id": concept_id,
            "organ_region_ids": row.get("organ_region_ids", []),
            "source_member_sha256": source_member_sha,
            "myosim_body": "torso",
            "source_body_id": torso["source_body_id"],
            "source_record_index": torso["source_record_index"],
            "core_body_index": torso["core_body_index"],
            "default_com_position_world_m": _vector(torso["default_com_position_world_m"], 3, "torso COM"),
            "default_inertial_quaternion_world_xyzw": _vector(
                torso["default_inertial_quaternion_world_xyzw"], 4, "torso quaternion"
            ),
            "body_link_registration": True,
            "world_or_body_frame_registration": False,
            "physical_volume_owner": None,
            "mechanical_mass_owner": None,
            "density_kg_per_m3": None,
            "pressure_gradient_momentum_transfer": False,
            "two_way_blood_tissue_transfer": False,
            "subject_calibration": False,
        })

    identity = {
        "cavity_map_sha256": _sha256(Path(cavity_map).read_bytes()),
        "bridge_sha256": _sha256(Path(bridge).read_bytes()),
        "human_manifest_sha256": _sha256(Path(human_manifest).read_bytes()),
        "rigid_payload_sha256": catalog["sha256"],
        "source_archive_sha256": catalog["archive_sha256"],
        "bindings": linked,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-cardiac-cavity-body-links.1",
        "source": {
            "cavity_map": str(Path(cavity_map).relative_to(ROOT)) if Path(cavity_map).is_relative_to(ROOT) else str(cavity_map),
            "bridge": str(Path(bridge).relative_to(ROOT)) if Path(bridge).is_relative_to(ROOT) else str(bridge),
            "bridge_sha256": identity["bridge_sha256"],
            "human_manifest": str(Path(human_manifest).relative_to(ROOT)) if Path(human_manifest).is_relative_to(ROOT) else str(human_manifest),
            "human_manifest_sha256": identity["human_manifest_sha256"],
            "rigid_payload_sha256": catalog["sha256"],
            "source_archive_sha256": catalog["archive_sha256"],
            "body_count": catalog["body_count"],
            "source_body_count": catalog["source_count"],
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "bindings": linked,
        "qualification": {
            "exact_source_member_hash_binding": True,
            "exact_cavity_semantic_binding": True,
            "body_link_registration": True,
            "body_link_transform_registered": True,
            "world_or_body_frame_registration": False,
            "cavity_domains_disjoint": bool(bridge_doc.get("qualification", {}).get("cavity_domains_disjoint", False)),
            "physical_volume_authority_assigned": False,
            "blood_mass_assigned": False,
            "material_density_calibrated": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_blood_tissue_transfer": False,
            "physiological_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Four exact source cardiac cavity members are hash-bound to the CVSim bridge "
            "and the named MyoSim torso source/core body frame. This closes body-frame "
            "bookkeeping only; the bridge remains non-disjoint and no physical cavity "
            "volume, blood mass, material density, pressure reaction, tissue exchange, "
            "subject calibration, or behavior is admitted."
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
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--cavity-map", type=Path, default=MAP)
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--human-manifest", type=Path, default=HUMAN_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_body_links(sources=args.sources, source_lock=args.source_lock,
                                    cavity_map=args.cavity_map, bridge=args.bridge,
                                    human_manifest=args.human_manifest)
        digest = immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "bindings": len(result["bindings"]),
                          "body_link_registration": True, "blood_mass_assigned": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"cardiac cavity body-link registration: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

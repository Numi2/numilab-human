"""Inventory the pinned BodyParts3D organ and vessel surface members.

The physiology authoring graph names 18 source regions, but naming a region is
not the same thing as admitting a volumetric or tubular mechanical field.  This
module resolves every declared BodyParts3D member, verifies the archive and
source hierarchy, performs the same exact authored-coordinate seam quotient
used by the cardiac importer, and records topology defects without repairing
or assigning a physical owner.

The output is an immutable source inventory.  It is intentionally not a FEM
mesh, a blood volume, a body registration, or a qualification receipt for
organ mechanics.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

from .cardiac_cavity_geometry import analyze_topology, exact_coordinate_quotient, parse_obj
from .model import ImportError as HumanImportError
from .physiology import canonical, load_anatomy, read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-geometry-inventory.v1"
TEMPLATE = ROOT / "config/physiology-organ-network-template.v1.json"
ARCHIVE = "partof_BP3D_4.0_obj_99.zip"
ARCHIVE_MEMBER_ROOT = "partof_BP3D_4.0_obj_99"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ geometry: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_bounds(vertices: Iterable[Iterable[float]]) -> dict[str, list[float]]:
    points = [tuple(float(value) for value in point) for point in vertices]
    require(points and all(len(point) == 3 for point in points), "member has no coordinates")
    require(all(math.isfinite(value) for point in points for value in point), "member has nonfinite coordinates")
    return {
        "minimum_m": [min(point[index] for point in points) * 0.001 for index in range(3)],
        "maximum_m": [max(point[index] for point in points) * 0.001 for index in range(3)],
    }


def _topology_summary(raw: dict[str, Any], quotient: dict[str, Any]) -> dict[str, Any]:
    topology = quotient["topology"]
    return {
        "raw_vertex_count": raw["vertex_count"],
        "raw_face_count": raw["face_count"],
        "raw_boundary_edge_count": raw["boundary_edge_count"],
        "quotient_vertex_count": len(quotient["vertices_m"]),
        "quotient_face_count": len(quotient["triangles"]),
        "identified_vertex_count": quotient["identified_vertex_count"],
        "boundary_edge_count": topology["boundary_edge_count"],
        "nonmanifold_edge_count": len(topology["nonmanifold_edges"]),
        "orientation_defect_edge_count": len(topology["orientation_defect_edges"]),
        "degenerate_face_count": len(topology["degenerate_face_ids"]),
        "duplicate_face_count": len(topology["duplicate_face_ids"]),
        "vertex_manifold_defect_count": len(topology["vertex_manifold_defect_ids"]),
        "face_component_count": topology["face_component_count"],
        "euler_characteristic": topology["euler_characteristic"],
        "closed_oriented_manifold_candidate": topology["closed_oriented_manifold_candidate"],
        "self_intersection_status": "not_checked",
        "interdomain_overlap_status": "not_checked",
    }


def _member_record(member_id: str, data: bytes) -> dict[str, Any]:
    member = f"{ARCHIVE_MEMBER_ROOT}/{member_id}.obj"
    parsed = parse_obj(data, member)
    raw = analyze_topology(parsed["vertices_mm"], parsed["triangles"])
    quotient = exact_coordinate_quotient(parsed)
    topology = _topology_summary(raw, quotient)
    return {
        "member_id": member_id,
        "archive_member": member,
        "sha256": _sha256(data),
        "bytes": len(data),
        "header_comments": parsed["comments"],
        "bounds": _finite_bounds(parsed["vertices_mm"]),
        "topology": topology,
        "admission": "source_surface_inventory_only",
        "repair_applied": False,
        "physical_volume_m3": None,
        "mechanical_mass_kg": None,
        "body_registration": False,
    }


def _locked_archive(sources: Path, source_lock: Path) -> tuple[Path, dict[str, Any], str]:
    lock = read_json(source_lock)
    metadata = lock.get("sources", {}).get("bodyparts3d_4")
    require(isinstance(metadata, dict), "missing BodyParts3D source lock")
    files = metadata.get("files", {})
    record = files.get(ARCHIVE)
    require(isinstance(record, dict), "missing BodyParts3D geometry archive lock")
    expected = record.get("sha256")
    expected_bytes = record.get("bytes")
    require(isinstance(expected, str) and len(expected) == 64, "invalid geometry archive hash lock")
    archive = sources / ARCHIVE
    require(archive.is_file() and not archive.is_symlink(), "missing or redirected geometry archive")
    data = archive.read_bytes()
    require(_sha256(data) == expected, "geometry archive hash mismatch")
    if expected_bytes is not None:
        require(len(data) == expected_bytes, "geometry archive byte count mismatch")
    return archive, {"file": ARCHIVE, "sha256": expected, "bytes": len(data)}, _sha256(data)


def inventory(*, sources: Path = ROOT / "Sources", source_lock: Path = ROOT / "sources.lock.json",
              template: Path = TEMPLATE) -> dict[str, Any]:
    """Resolve all 18 authored regions to exact source surface-member records."""
    sources = Path(sources)
    source_lock = Path(source_lock)
    template = Path(template)
    graph = read_json(template)
    require(graph.get("schema") == "HumanPack.physiology.v1", "unsupported physiology template")
    regions = graph.get("regions")
    require(isinstance(regions, list) and regions, "physiology template has no regions")
    anatomy = load_anatomy(sources, source_lock)
    archive_path, archive_record, archive_sha = _locked_archive(sources, source_lock)
    declared: dict[str, dict[str, Any]] = {}
    member_regions: dict[str, list[str]] = {}
    for region in regions:
        require(isinstance(region, dict), "invalid region record")
        required = {"id", "semantic_id", "concept_id", "source_name", "hierarchy", "member_ids", "selection"}
        require(required <= region.keys(), "region record is incomplete")
        region_id = region["id"]
        require(isinstance(region_id, str) and region_id and region_id not in declared, "duplicate or invalid region ID")
        concept = region["concept_id"]
        hierarchy = region["hierarchy"]
        source_members = anatomy["tables"].get(hierarchy, {}).get((concept, region["source_name"]))
        require(source_members is not None, f"region {region_id} is absent from the pinned hierarchy")
        members = region["member_ids"]
        require(isinstance(members, list) and members and len(set(members)) == len(members),
                f"region {region_id} has invalid member IDs")
        require(region["selection"] == "complete_membership" and set(members) == source_members,
                f"region {region_id} does not retain complete source membership")
        for member_id in members:
            require(isinstance(member_id, str) and member_id and member_id.isascii(),
                    f"region {region_id} has an invalid member identity")
            member_regions.setdefault(member_id, []).append(region_id)
        declared[region_id] = region
    records: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = {info.filename for info in archive.infolist()}
        for member_id in sorted(member_regions):
            member = f"{ARCHIVE_MEMBER_ROOT}/{member_id}.obj"
            require(member in names, f"missing archive member {member}")
            data = archive.read(member)
            records[member_id] = _member_record(member_id, data)
    region_records = []
    for region_id in sorted(declared):
        region = declared[region_id]
        members = [records[member] for member in sorted(region["member_ids"])]
        closed = sum(1 for member in members if member["topology"]["closed_oriented_manifold_candidate"])
        region_records.append({
            "id": region_id,
            "semantic_id": region["semantic_id"],
            "concept_id": region["concept_id"],
            "source_name": region["source_name"],
            "hierarchy": region["hierarchy"],
            "member_ids": sorted(region["member_ids"]),
            "member_count": len(members),
            "closed_quotient_member_count": closed,
            "open_or_defective_quotient_member_count": len(members) - closed,
            "surface_members": members,
            "source_overlap_status": "not_checked",
            "region_volume_status": "not_admitted",
            "mechanical_owner_status": "unassigned",
            "body_registration_status": "unregistered",
        })
    duplicate_members = [
        {"member_id": member_id, "region_ids": sorted(region_ids)}
        for member_id, region_ids in sorted(member_regions.items()) if len(region_ids) > 1
    ]
    closed_count = sum(member["topology"]["closed_oriented_manifold_candidate"] for member in records.values())
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-geometry.1",
        "template": {"path": str(template), "sha256": _sha256(template.read_bytes())},
        "anatomy_source": anatomy["source"],
        "archive": archive_record,
        "archive_sha256_recomputed": archive_sha,
        "coordinate_frame": "BodyParts3D 4.0 source frame",
        "source_units": "mm",
        "output_units": "m",
        "regions": region_records,
        "member_region_overlaps": duplicate_members,
        "counts": {
            "region_count": len(region_records),
            "member_count": len(records),
            "declared_membership_count": sum(len(region["member_ids"]) for region in region_records),
            "closed_quotient_member_count": closed_count,
            "open_or_defective_quotient_member_count": len(records) - closed_count,
            "region_with_open_or_defective_member_count": sum(
                bool(region["open_or_defective_quotient_member_count"]) for region in region_records
            ),
        },
        "qualification": {
            "source_membership_verified": True,
            "source_bytes_verified": True,
            "exact_coordinate_seam_quotient_performed": True,
            "watertight_union_performed": False,
            "self_intersection_checked": False,
            "interdomain_overlap_checked": False,
            "body_frame_registered": False,
            "material_assigned": False,
            "blood_mass_assigned": False,
            "physical_stepping": False,
            "physiological_calibration": False,
        },
        "boundary": (
            "Pinned 18-region source membership and 386 OBJ members are inventoried. "
            "The seven seam-quotient members with topology defects are retained. "
            "No surface is repaired, unioned, registered, assigned a material or mass, "
            "or advanced by a physical solver."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = inventory(sources=args.sources, source_lock=args.source_lock, template=args.template)
        output = args.output.resolve()
        sha = _immutable_write(output, result)
        print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": sha,
                          **result["counts"]}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"organ geometry: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

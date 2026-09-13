"""Compute source-frame surface moments for the pinned organ inventory.

This is a source-data increment after ``organ_geometry``.  It integrates the
closed, single-component BodyParts3D surfaces without assigning a physical
volume, density, blood owner, body transform, or material.  Defective and
multi-component surfaces remain explicit non-admissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

from . import cvsim21_anatomy
from .cardiac_cavity_geometry import exact_coordinate_quotient, parse_obj
from .model import ImportError as HumanImportError
from .organ_geometry import (
    ARCHIVE,
    ARCHIVE_MEMBER_ROOT,
    ROOT,
    _locked_archive,
    _topology_summary,
    inventory,
)
from .physiology import canonical, read_json

SCHEMA = "HumanPack.organ-geometry-moments.v1"
TEMPLATE = ROOT / "config/physiology-organ-network-template.v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ geometry moments: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _moment_status(topology: dict[str, Any]) -> str:
    if not topology["closed_oriented_manifold_candidate"]:
        return "source_topology_defective"
    if topology["face_component_count"] != 1 or topology["unused_vertex_ids"]:
        return "not_single_closed_component"
    return "computed_single_closed_component"


def compile_moments(*, sources: Path = ROOT / "Sources",
                    source_lock: Path = ROOT / "sources.lock.json",
                    template: Path = TEMPLATE) -> dict[str, Any]:
    """Recompute source-frame moments for every unique inventory member."""
    sources = Path(sources)
    source_lock = Path(source_lock)
    template = Path(template)
    source_inventory = inventory(sources=sources, source_lock=source_lock, template=template)
    archive_path, archive_record, archive_sha = _locked_archive(sources, source_lock)
    inventory_members = {
        member["member_id"]: member
        for region in source_inventory["regions"]
        for member in region["surface_members"]
    }
    require(len(inventory_members) == source_inventory["counts"]["member_count"],
            "inventory member identities are not unique")
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = {info.filename for info in archive.infolist()}
        for member_id in sorted(inventory_members):
            expected = inventory_members[member_id]
            archive_member = f"{ARCHIVE_MEMBER_ROOT}/{member_id}.obj"
            require(archive_member in names, f"missing archive member {archive_member}")
            data = archive.read(archive_member)
            require(_sha256(data) == expected["sha256"], f"source member hash drift: {member_id}")
            parsed = parse_obj(data, archive_member)
            raw = cvsim21_anatomy.geometry.analyze_topology(parsed["vertices_mm"], parsed["triangles"])
            quotient = exact_coordinate_quotient(parsed)
            actual_topology = _topology_summary(raw, quotient)
            require(actual_topology == expected["topology"], f"source topology drift: {member_id}")
            topology = quotient["topology"]
            status = _moment_status(topology)
            moments: dict[str, Any] | None = None
            if status == "computed_single_closed_component":
                moments = cvsim21_anatomy.geometry_moments(
                    quotient["vertices_m"], quotient["triangles"]
                )
                require(moments["status"] == "algebraic_geometry_moments_not_physiological_volume_or_mass",
                        f"unexpected moment status: {member_id}")
                require(math.isfinite(moments["absolute_signed_volume_m3"])
                        and moments["absolute_signed_volume_m3"] > 0,
                        f"invalid source surface volume: {member_id}")
                require(moments["physical_volume_m3"] is None
                        and moments["mechanical_mass_kg"] is None,
                        f"source moment promoted to mass: {member_id}")
            records.append({
                "member_id": member_id,
                "archive_member": archive_member,
                "source_sha256": expected["sha256"],
                "moment_status": status,
                "face_component_count": topology["face_component_count"],
                "unused_quotient_vertex_count": len(topology["unused_vertex_ids"]),
                "source_surface_moments": moments,
                "physical_volume_m3": None,
                "mechanical_mass_kg": None,
                "body_registration": False,
            })

    by_id = {row["member_id"]: row for row in records}
    regions = []
    for region in source_inventory["regions"]:
        members = [by_id[member_id] for member_id in region["member_ids"]]
        counts: dict[str, int] = {}
        for member in members:
            counts[member["moment_status"]] = counts.get(member["moment_status"], 0) + 1
        regions.append({"id": region["id"], "member_ids": list(region["member_ids"]),
                       "member_count": len(members), "moment_status_counts": dict(sorted(counts.items()))})
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["moment_status"]] = status_counts.get(record["moment_status"], 0) + 1
    computed = status_counts.get("computed_single_closed_component", 0)
    require(computed == 357 and status_counts.get("source_topology_defective", 0) == 7
            and status_counts.get("not_single_closed_component", 0) == 14,
            f"unexpected source moment status counts: {status_counts}")
    inventory_sha = _sha256(canonical(source_inventory) + b"\n")
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-geometry-moments.1",
        "inventory_schema": source_inventory["schema"],
        "inventory_sha256": inventory_sha,
        "template": {"path": str(template), "sha256": _sha256(template.read_bytes())},
        "archive": archive_record,
        "archive_sha256_recomputed": archive_sha,
        "coordinate_frame": source_inventory["coordinate_frame"],
        "source_units": source_inventory["source_units"],
        "output_units": source_inventory["output_units"],
        "counts": {
            **source_inventory["counts"],
            "moment_computed_member_count": computed,
            "not_single_closed_component_member_count": status_counts.get("not_single_closed_component", 0),
            "source_topology_defective_member_count": status_counts.get("source_topology_defective", 0),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "regions": regions,
        "members": records,
        "qualification": {
            "source_inventory_recomputed": True,
            "single_closed_surface_integrals_computed": True,
            "watertight_union_performed": False,
            "self_intersection_checked": False,
            "interdomain_overlap_checked": False,
            "body_frame_registered": False,
            "material_assigned": False,
            "blood_mass_assigned": False,
            "physical_volume_authority_assigned": False,
            "physical_stepping": False,
            "physiological_calibration": False,
        },
        "boundary": (
            "Source-frame algebraic volumes and first/second spatial moments are retained "
            "only for 357 single closed components. Seven topology-defective and fourteen "
            "multi-component closed candidates remain unintegrable. No result is a body "
            "registered organ volume, a hydraulic blood volume, a density or a mass owner."
        ),
    }
    canonical(result)
    return result


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
        result = compile_moments(sources=args.sources, source_lock=args.source_lock, template=args.template)
        output = args.output.resolve()
        digest = _immutable_write(output, result)
        print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                          **result["counts"]}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"organ geometry moments: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

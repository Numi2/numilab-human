"""Compute conservative source-frame moments for disconnected organ members.

The BodyParts3D archive contains a small set of members whose exact seam
quotient is made of more than one closed surface component.  A disconnected
member can be integrated without inventing a watertight union only when every
component is individually closed and their axis-aligned source bounds are
disjoint.  This compiler records that algebraic component sum as source
geometry.  It never assigns a physical organ volume, density, blood owner,
material, body transform, or mechanical mass.
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
from .cardiac_cavity_geometry import analyze_topology, exact_coordinate_quotient, parse_obj
from .model import ImportError as HumanImportError
from .organ_geometry import (
    ARCHIVE_MEMBER_ROOT,
    ROOT,
    _locked_archive,
    _topology_summary,
    inventory,
)
from .physiology import canonical

SCHEMA = "HumanPack.organ-geometry-component-moments.v1"
TEMPLATE = ROOT / "config/physiology-organ-network-template.v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ geometry component moments: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aabb(vertices: list[list[float]]) -> dict[str, list[float]]:
    require(vertices and all(len(point) == 3 for point in vertices), "component has no coordinates")
    require(all(math.isfinite(value) for point in vertices for value in point),
            "component has nonfinite coordinates")
    return {
        "minimum_m": [min(point[index] for point in vertices) for index in range(3)],
        "maximum_m": [max(point[index] for point in vertices) for index in range(3)],
    }


def _aabbs_disjoint(bounds: list[dict[str, list[float]]]) -> bool:
    """Return true only when every pair is separated on at least one axis."""
    for left_index, left in enumerate(bounds):
        for right in bounds[left_index + 1:]:
            separated = any(
                left["maximum_m"][axis] <= right["minimum_m"][axis]
                or right["maximum_m"][axis] <= left["minimum_m"][axis]
                for axis in range(3)
            )
            if not separated:
                return False
    return True


def _component_geometry(quotient: dict[str, Any], face_ids: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_triangles = quotient["triangles"]
    vertex_ids = sorted({vertex for face_id in face_ids for vertex in source_triangles[face_id]})
    mapping = {source: index for index, source in enumerate(vertex_ids)}
    vertices = [quotient["vertices_m"][source] for source in vertex_ids]
    triangles = [[mapping[vertex] for vertex in source_triangles[face_id]] for face_id in face_ids]
    topology = analyze_topology(vertices, triangles)
    return {"vertices_m": vertices, "triangles": triangles}, topology


def _component_moment(geometry: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any] | None:
    if not topology["closed_oriented_manifold_candidate"] \
            or topology["face_component_count"] != 1 \
            or topology["unused_vertex_ids"]:
        return None
    moments = cvsim21_anatomy.geometry_moments(geometry["vertices_m"], geometry["triangles"])
    require(moments["status"] == "algebraic_geometry_moments_not_physiological_volume_or_mass",
            "unexpected component moment status")
    require(math.isfinite(moments["absolute_signed_volume_m3"])
            and moments["absolute_signed_volume_m3"] > 0,
            "invalid component source surface volume")
    require(moments["physical_volume_m3"] is None and moments["mechanical_mass_kg"] is None,
            "component moment promoted to mass")
    return moments


def _sum_moments(component_moments: list[dict[str, Any]]) -> dict[str, Any]:
    require(component_moments, "cannot sum empty component moments")
    volume = math.fsum(moment["absolute_signed_volume_m3"] for moment in component_moments)
    first = [math.fsum(moment["first_volume_moment_m4"][axis] for moment in component_moments)
             for axis in range(3)]
    second = [[math.fsum(moment["second_volume_moment_m5"][row][column]
                         for moment in component_moments)
               for column in range(3)] for row in range(3)]
    require(math.isfinite(volume) and volume > 0, "invalid aggregate source surface volume")
    centroid = [value / volume for value in first]
    central = [[math.fsum((second[row][column],
                           -volume * centroid[row] * centroid[column]))
                for column in range(3)] for row in range(3)]
    require(all(math.isfinite(value) for row in central for value in row),
            "nonfinite aggregate central volume moment")
    scale = max(abs(value) for row in central for value in row)
    require(scale > 0, "zero aggregate central volume moment")
    normalized = [[value / scale for value in row] for row in central]
    require(all(normalized[index][index] > 0 for index in range(3)),
            "nonpositive aggregate central volume moment")
    require(all(normalized[row][row] * normalized[column][column]
                - normalized[row][column] ** 2 > 0
                for row in range(3) for column in range(row)),
            "nonpositive aggregate central volume moment minor")
    determinant = (
        normalized[0][0] * (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1])
        - normalized[0][1] * (normalized[1][0] * normalized[2][2] - normalized[1][2] * normalized[2][0])
        + normalized[0][2] * (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0])
    )
    require(determinant > 0, "nonpositive aggregate central volume moment determinant")
    inertia = [[(math.fsum(central[axis][axis] for axis in range(3))
                 if row == column else 0.0) - central[row][column]
                for column in range(3)] for row in range(3)]
    return {
        "method": "disjoint_closed_component_sum_of_oriented_surface_tetrahedral_integrals",
        "status": "algebraic_geometry_moments_not_physiological_volume_or_mass",
        "signed_volume_m3": volume,
        "absolute_signed_volume_m3": volume,
        "source_winding": "component_absolute_volume_sum",
        "centroid_source_frame_m": centroid,
        "first_volume_moment_m4": first,
        "second_volume_moment_m5": second,
        "central_second_volume_moment_m5": central,
        "inertia_per_unit_density_m5": inertia,
        "physical_volume_m3": None,
        "density_kg_per_m3": None,
        "mechanical_mass_kg": None,
        "self_intersection_qualified": False,
        "interdomain_disjointness_qualified": False,
    }


def _component_details(quotient: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    details: list[dict[str, Any]] = []
    moments: list[dict[str, Any]] = []
    all_closed = True
    for component_index, face_ids in enumerate(quotient["topology"]["face_components"]):
        geometry, topology = _component_geometry(quotient, face_ids)
        bounds = _aabb(geometry["vertices_m"])
        component_moment = _component_moment(geometry, topology)
        if component_moment is None:
            all_closed = False
        else:
            moments.append(component_moment)
        details.append({
            "component_index": component_index,
            "source_face_ids": list(face_ids),
            "vertex_count": len(geometry["vertices_m"]),
            "face_count": len(geometry["triangles"]),
            "aabb_m": bounds,
            "closed_oriented_manifold_candidate": component_moment is not None,
            "topology": {
                "boundary_edge_count": topology["boundary_edge_count"],
                "nonmanifold_edge_count": len(topology["nonmanifold_edges"]),
                "orientation_defect_edge_count": len(topology["orientation_defect_edges"]),
                "degenerate_face_count": len(topology["degenerate_face_ids"]),
                "vertex_manifold_defect_count": len(topology["vertex_manifold_defect_ids"]),
                "unused_vertex_count": len(topology["unused_vertex_ids"]),
            },
        })
    return details, moments, all_closed


def compile_component_moments(*, sources: Path = ROOT / "Sources",
                              source_lock: Path = ROOT / "sources.lock.json",
                              template: Path = TEMPLATE) -> dict[str, Any]:
    """Recompute exact source moments, summing only safely separated components."""
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
            status = "source_topology_defective"
            moments: dict[str, Any] | None = None
            component_details: list[dict[str, Any]] = []
            component_moments: list[dict[str, Any]] = []
            component_aabb_disjoint: bool | None = None
            if topology["closed_oriented_manifold_candidate"] and topology["face_component_count"] == 1 \
                    and not topology["unused_vertex_ids"]:
                status = "computed_single_closed_component"
                moments = cvsim21_anatomy.geometry_moments(quotient["vertices_m"], quotient["triangles"])
                require(moments["physical_volume_m3"] is None and moments["mechanical_mass_kg"] is None,
                        f"source moment promoted to mass: {member_id}")
            elif topology["closed_oriented_manifold_candidate"] and topology["face_component_count"] > 1:
                component_details, component_moments, all_closed = _component_details(quotient)
                component_aabb_disjoint = _aabbs_disjoint([detail["aabb_m"] for detail in component_details])
                if all_closed and component_aabb_disjoint:
                    status = "computed_disjoint_closed_component_sum"
                    moments = _sum_moments(component_moments)
                elif all_closed:
                    status = "not_disjoint_component_bounds"
                else:
                    status = "source_topology_defective"
            record = {
                "member_id": member_id,
                "archive_member": archive_member,
                "source_sha256": expected["sha256"],
                "moment_status": status,
                "face_component_count": topology["face_component_count"],
                "unused_quotient_vertex_count": len(topology["unused_vertex_ids"]),
                "component_aabb_disjoint_checked": component_aabb_disjoint is not None,
                "component_aabb_disjoint": component_aabb_disjoint,
                "component_details": component_details or None,
                "source_component_moments": component_moments or None,
                "source_surface_moments": moments,
                "physical_volume_m3": None,
                "mechanical_mass_kg": None,
                "body_registration": False,
            }
            records.append(record)

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
    computed_single = status_counts.get("computed_single_closed_component", 0)
    computed_sum = status_counts.get("computed_disjoint_closed_component_sum", 0)
    require(computed_single == 357 and computed_sum == 7
            and status_counts.get("not_disjoint_component_bounds", 0) == 7
            and status_counts.get("source_topology_defective", 0) == 7,
            f"unexpected component moment status counts: {status_counts}")
    inventory_sha = _sha256(canonical(source_inventory) + b"\n")
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-geometry-component-moments.1",
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
            "moment_computed_member_count": computed_single + computed_sum,
            "single_closed_component_member_count": computed_single,
            "disjoint_closed_component_sum_member_count": computed_sum,
            "not_disjoint_component_bounds_member_count": status_counts.get("not_disjoint_component_bounds", 0),
            "source_topology_defective_member_count": status_counts.get("source_topology_defective", 0),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "regions": regions,
        "members": records,
        "qualification": {
            "source_inventory_recomputed": True,
            "single_closed_surface_integrals_computed": True,
            "disjoint_closed_component_sums_computed": True,
            "component_aabb_disjointness_checked": True,
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
            "for 357 single closed members and seven members whose individually closed "
            "components have disjoint source bounds. Seven multi-component members with "
            "overlapping bounds and seven source-topology-defective members remain "
            "unintegrated. No result is a watertight union, a body-registered organ "
            "volume, a hydraulic blood volume, a density, a material, or a mechanical mass owner."
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
        result = compile_component_moments(sources=args.sources, source_lock=args.source_lock, template=args.template)
        output = args.output.resolve()
        digest = _immutable_write(output, result)
        print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                          **result["counts"]}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"organ geometry component moments: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

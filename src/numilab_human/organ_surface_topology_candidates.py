"""Build exact-support topology candidates for defective organ source surfaces.

This repairs only duplicate/opposite source facets, exact zero-area faces, and
conforming straight seams already supported by ``surface_topology_repair``.
The raw BodyParts3D archive remains unchanged. Closed topology here is not
self-intersection, anatomical, physical-volume, or mechanics qualification.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

from . import model
from . import surface_topology_repair as topology_repair
from .cardiac_cavity_geometry import analyze_topology, parse_obj
from .organ_geometry import TEMPLATE, inventory
from .physiology import canonical

SCHEMA = "numi.human.organ-surface-topology-candidates.v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_keys(keys: list[tuple[Fraction, Fraction, Fraction]]) -> list[tuple[Fraction, Fraction, Fraction]]:
    seen: dict[tuple[Fraction, Fraction, Fraction], int] = {}
    result = []
    for key in keys:
        if key not in seen:
            seen[key] = len(result)
            result.append(key)
    return result


def _json_topology(topology: dict[str, Any]) -> dict[str, Any]:
    return topology_repair.compact_topology(topology)


def _rational_coordinates(keys: list[tuple[Fraction, Fraction, Fraction]]) -> list[list[list[int]]]:
    return [[[value.numerator, value.denominator] for value in point] for point in keys]


def _component_bounds(vertices: list[list[float]], keys: list[tuple[Fraction, Fraction, Fraction]],
                      faces: list[list[int]], topology: dict[str, Any]) -> dict[str, Any]:
    bounds = []
    for face_ids in topology["face_components"]:
        vertex_ids = sorted({vertex for face_id in face_ids for vertex in faces[face_id]})
        local_vertex = {source: index for index, source in enumerate(vertex_ids)}
        component_faces = [[local_vertex[vertex] for vertex in faces[face_id]] for face_id in face_ids]
        component_vertices = [vertices[vertex_id] for vertex_id in vertex_ids]
        component_topology = analyze_topology(component_vertices, component_faces)
        exact_points = [keys[vertex_id] for vertex_id in vertex_ids]
        bounds.append({
            "closed_oriented_manifold_candidate": component_topology["closed_oriented_manifold_candidate"],
            "minimum_mm": _rational_coordinates([[min(point[axis] for point in exact_points)
                                                   for axis in range(3)]])[0],
            "maximum_mm": _rational_coordinates([[max(point[axis] for point in exact_points)
                                                   for axis in range(3)]])[0],
        })

    overlap_pairs = []
    for left_index, left in enumerate(bounds):
        for right_index in range(left_index + 1, len(bounds)):
            right = bounds[right_index]
            separated = any(
                Fraction(*left["maximum_mm"][axis]) <= Fraction(*right["minimum_mm"][axis])
                or Fraction(*right["maximum_mm"][axis]) <= Fraction(*left["minimum_mm"][axis])
                for axis in range(3)
            )
            if not separated:
                overlap_pairs.append([left_index, right_index])
    return {
        "component_count": len(bounds),
        "all_components_closed_oriented_manifold_candidates": all(
            row["closed_oriented_manifold_candidate"] for row in bounds
        ),
        "component_aabb_pairwise_disjoint": not overlap_pairs,
        "component_aabb_overlap_pairs": overlap_pairs,
        "component_aabb_bounds_mm_exact": bounds,
    }


def _repair_member(sources: Path, member: dict[str, Any]) -> dict[str, Any]:
    archive_member = member["archive_member"]
    hierarchy = "part_of" if archive_member.startswith("partof_") else "is_a"
    _, exact_archive_member, data = model._bodyparts_obj_member(sources, hierarchy, member["member_id"])
    if exact_archive_member != archive_member or _sha256(data) != member["sha256"]:
        raise model.ImportError(f"organ surface topology candidate: source member drift: {member['member_id']}")

    parsed = parse_obj(data, archive_member)
    vertices = parsed["vertices_mm"]
    keys = parsed["coordinate_keys"]
    faces = parsed["triangles"]
    face_ancestry = list(range(len(faces)))
    vertex_ancestry = list(range(len(vertices)))
    passes = []

    for pass_index in range(1, 3):
        result = topology_repair.repair(vertices, keys, faces)
        passes.append({
            "pass": pass_index,
            "before": result["before"],
            "after": result["after"],
            "removed_exact_zero_area_face_count": len(result["removed_exact_zero_area_source_faces"]),
            "cancelled_opposite_face_pair_count": len(result["cancelled_opposite_source_face_pairs"]),
            "conforming_seam_subdivision_count": len(result["conforming_seam_subdivisions"]),
            "retained_source_support_preserved": result["retained_source_support_preserved"],
            "exact_signed_integral_preserved": result["exact_signed_integral_preserved"],
            "vertex_coordinates_modified": result["vertex_coordinates_modified"],
        })
        face_ancestry = [face_ancestry[index] for index in result["source_face_ids"]]
        vertex_ancestry = [vertex_ancestry[index] for index in result["source_vertex_ids"]]
        quotient_keys = _unique_keys(keys)
        keys = [quotient_keys[index] for index in result["source_quotient_vertex_ids"]]
        vertices = result["vertices"]
        faces = result["triangles"]
        if result["after"]["closed_oriented_manifold_candidate"]:
            break

    final_topology = analyze_topology(vertices, faces)
    if not final_topology["closed_oriented_manifold_candidate"]:
        raise model.ImportError(
            f"organ surface topology candidate: unresolved source topology: {member['member_id']}"
        )
    if len(face_ancestry) != len(faces) or len(vertex_ancestry) != len(vertices) or len(keys) != len(vertices):
        raise model.ImportError(f"organ surface topology candidate: ancestry coverage: {member['member_id']}")
    components = _component_bounds(vertices, keys, faces, final_topology)

    return {
        "member_id": member["member_id"],
        "archive_member": archive_member,
        "hierarchy": hierarchy,
        "source_member_sha256": member["sha256"],
        "source_vertex_count": len(parsed["vertices_mm"]),
        "source_face_count": len(parsed["triangles"]),
        "regions": [],
        "passes": passes,
        "candidate_topology": _json_topology(final_topology),
        "component_geometry": components,
        "candidate_vertex_count": len(vertices),
        "candidate_face_count": len(faces),
        "source_vertex_ids": vertex_ancestry,
        "source_face_ids": face_ancestry,
        "coordinate_unit": "mm",
        "coordinate_keys_rational": _rational_coordinates(keys),
        "triangles": faces,
        "source_support_preserved": all(p["retained_source_support_preserved"] for p in passes),
        "exact_signed_integral_preserved": all(p["exact_signed_integral_preserved"] for p in passes),
        "vertex_coordinates_modified": any(p["vertex_coordinates_modified"] for p in passes),
        "self_intersections": "not_checked",
        "physical_volume": False,
        "mechanics": False,
    }


def build_candidates(*, sources: Path, source_lock: Path, template: Path = TEMPLATE) -> tuple[dict[str, Any], bytes]:
    sources = Path(sources)
    source_lock = Path(source_lock)
    template = Path(template)
    source_inventory = inventory(sources=sources, source_lock=source_lock, template=template)
    members: dict[str, dict[str, Any]] = {}
    regions_by_member: dict[str, set[str]] = defaultdict(set)
    for region in source_inventory["regions"]:
        for surface in region["surface_members"]:
            if not surface["topology"]["closed_oriented_manifold_candidate"]:
                members.setdefault(surface["member_id"], surface)
                regions_by_member[surface["member_id"]].add(region["id"])

    candidates = []
    for member_id in sorted(members):
        candidate = _repair_member(sources, members[member_id])
        candidate["regions"] = sorted(regions_by_member[member_id])
        candidates.append(candidate)

    unresolved = [row["member_id"] for row in candidates
                  if not row["candidate_topology"]["closed_oriented_manifold_candidate"]]
    if unresolved or len(candidates) != source_inventory["counts"]["open_or_defective_quotient_member_count"]:
        raise model.ImportError("organ surface topology candidate: incomplete candidate set")

    payload = {
        "schema": SCHEMA,
        "source_inventory_sha256": _sha256(canonical(source_inventory) + b"\n"),
        "source_archive": source_inventory["archive"],
        "source_members": candidates,
        "raw_source_archives_modified": False,
        "new_coordinates_added": False,
        "source_support_preserved": all(row["source_support_preserved"] for row in candidates),
        "exact_signed_integral_preserved": all(row["exact_signed_integral_preserved"] for row in candidates),
        "physical_volume": False,
        "mechanics": False,
        "evidence_boundary": (
            "Exact-source topology candidates only. Repairs cancel coincident opposite-oriented source facets, "
            "remove exact zero-area faces, and subdivide conforming straight seams using source coordinates. "
            "Self-intersection, inter-component disjointness, anatomy, subject registration, physical volume, "
            "constitutive calibration, and mechanics remain unqualified."
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    return payload, encoded

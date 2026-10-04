"""Exact intersection and containment checks for repaired organ surface candidates."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from . import cardiac_cavity_intersections as intersections
from . import model
from .cardiac_cavity_geometry import analyze_topology

SCHEMA = "numi.human.organ-surface-embeddedness-audit.v1"


def _hash_pairs(pairs: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bounds(vertices: list[tuple[int, int, int]]) -> list[list[int]]:
    return [[min(point[axis] for point in vertices), max(point[axis] for point in vertices)]
            for axis in range(3)]


def _component(vertices: list[tuple[int, int, int]], faces: list[list[int]], face_ids: list[int]) -> dict[str, Any]:
    source_vertex_ids = sorted({vertex for face_id in face_ids for vertex in faces[face_id]})
    local = {source: index for index, source in enumerate(source_vertex_ids)}
    local_faces = [[local[vertex] for vertex in faces[face_id]] for face_id in face_ids]
    local_vertices = [vertices[index] for index in source_vertex_ids]
    records = intersections._records(local_vertices, local_faces)
    report = intersections._audit_pair(records, records, same_surface=True)
    candidate_pairs = [[face_ids[left], face_ids[right]] for left, right in report["triangle_pairs"]]
    return {
        "candidate_face_ids": face_ids,
        "vertex_count": len(local_vertices),
        "face_count": len(local_faces),
        "topology": analyze_topology(local_vertices, local_faces),
        "records": records,
        "bounds": _bounds(local_vertices),
        "self_intersection_count": report["count"],
        "self_intersection_pairs_sha256": _hash_pairs(candidate_pairs),
        "self_intersection_pairs_sample": candidate_pairs[:16],
        "self_aabb_candidate_pairs": report["aabb_candidate_pairs"],
    }


def audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "numi.human.organ-surface-topology-candidates.v1":
        raise model.ImportError("organ surface embeddedness audit: unsupported candidate payload")
    results = []
    for member in payload.get("source_members", []):
        keys = [tuple(Fraction(*value) for value in point)
                for point in member["coordinate_keys_rational"]]
        denominator = math.lcm(*(value.denominator for point in keys for value in point))
        vertices = [tuple(value.numerator * (denominator // value.denominator) for value in point)
                    for point in keys]
        faces = member["triangles"]
        topology = analyze_topology(vertices, faces)
        if not topology["closed_oriented_manifold_candidate"]:
            raise model.ImportError(f"organ surface embeddedness audit: topology drift: {member['member_id']}")
        components = [_component(vertices, faces, face_ids)
                      for face_ids in topology["face_components"]]
        intersections_between = []
        containment = []
        unresolved = []
        for left_index, left in enumerate(components):
            for right_index in range(left_index + 1, len(components)):
                right = components[right_index]
                left_bounds, right_bounds = left["bounds"], right["bounds"]
                if any(left_bounds[axis][1] < right_bounds[axis][0]
                       or right_bounds[axis][1] < left_bounds[axis][0]
                       for axis in range(3)):
                    continue
                cross = intersections._audit_pair(left["records"], right["records"], same_surface=False)
                if cross["count"]:
                    candidate_pairs = [
                        [left["candidate_face_ids"][a], right["candidate_face_ids"][b]]
                        for a, b in cross["triangle_pairs"]
                    ]
                    intersections_between.append({
                        "components": [left_index, right_index],
                        "triangle_intersection_count": cross["count"],
                        "triangle_intersection_pairs_sha256": _hash_pairs(candidate_pairs),
                        "triangle_intersection_pairs_sample": candidate_pairs[:16],
                        "aabb_candidate_pairs": cross["aabb_candidate_pairs"],
                    })
                    continue
                for child_index, child, parent_index, parent in (
                    (left_index, left, right_index, right),
                    (right_index, right, left_index, left),
                ):
                    location = intersections.point_location(child["records"][0][0][0], parent["records"])["location"]
                    if location == "inside":
                        containment.append({"component": child_index, "container": parent_index})
                    elif location != "outside":
                        unresolved.append({"components": [left_index, right_index], "location": location})

        self_count = sum(component["self_intersection_count"] for component in components)
        all_components_closed = all(
            component["topology"]["closed_oriented_manifold_candidate"] for component in components
        )
        if self_count or intersections_between:
            status = "self_intersecting_source_candidate"
        elif containment:
            status = "nested_component_source_candidate"
        elif unresolved:
            status = "component_relation_unresolved"
        elif all_components_closed:
            status = "embedded_disjoint_component_union_candidate"
        else:
            status = "component_topology_unresolved"
        results.append({
            "member_id": member["member_id"],
            "regions": member["regions"],
            "component_count": len(components),
            "component_self_intersection_count": self_count,
            "component_self_intersection_pairs": [
                {
                    "component_index": index,
                    "count": component["self_intersection_count"],
                    "pairs_sha256": component["self_intersection_pairs_sha256"],
                    "pairs_sample": component["self_intersection_pairs_sample"],
                    "aabb_candidate_pairs": component["self_aabb_candidate_pairs"],
                }
                for index, component in enumerate(components)
                if component["self_intersection_count"]
            ],
            "intersecting_component_pairs": intersections_between,
            "nested_component_pairs": containment,
            "unresolved_component_pairs": unresolved,
            "status": status,
            "physical_volume": False,
            "mechanics": False,
        })
    if len(results) != len(payload.get("source_members", [])):
        raise model.ImportError("organ surface embeddedness audit: incomplete candidate list")
    return {
        "schema": SCHEMA,
        "candidate_payload_schema": payload["schema"],
        "candidate_count": len(results),
        "results": results,
        "exact_predicates": "rational source-coordinate triangle intersections and ray parity; no tolerance",
        "evidence_boundary": (
            "Exact embeddedness and component containment of topology-repaired source-surface candidates. "
            "No anatomical validation, subject binding, physical volume, tissue calibration, or mechanics."
        ),
    }

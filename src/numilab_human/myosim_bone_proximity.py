"""Audit MyoSim endpoint proximity to MyoSim's own compiled bone meshes.

This source-environment tool answers a deliberately narrow question: was an
authored terminal muscle site close to a mesh attached to the same mechanics
body before BodyParts3D registration was involved?  It does not register the
two anatomy sources and it does not promote an endpoint to a surface law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .myosim_export import export_fullbody


AUDIT_SCHEMA = "numi.human.myosim-source-bone-proximity.v1"
WORKLIST_SCHEMA = "numi.human.bodyparts-registration-worklist.v1"
TENDON_SCHEMA = "numi.human.tendon-attachment-envelope-payload.v2"
TENDON_SCHEMAS = {
    TENDON_SCHEMA,
    "numi.human.tendon-attachment-envelope-payload.v3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wxyz_rotation_matrix(quaternion: Any, np: Any) -> Any:
    """Return an explicit finite rotation matrix for a MuJoCo wxyz quaternion."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not bool(np.all(np.isfinite(values))):
        raise RuntimeError("compiled mesh geometry has a non-finite quaternion")
    norm = float(np.linalg.norm(values))
    if not norm > 0.0:
        raise RuntimeError("compiled mesh geometry has a degenerate quaternion")
    w, x, y, z = (float(value) / norm for value in values)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _point_segment_distance_squared(point: Any, start: Any, end: Any, np: Any) -> Any:
    edge = end - start
    denominator = np.einsum("ij,ij->i", edge, edge)
    numerator = np.einsum("ij,ij->i", point - start, edge)
    parameter = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-30,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = start + parameter[:, None] * edge
    difference = point - closest
    return np.einsum("ij,ij->i", difference, difference)


def _point_triangle_distances_squared(point: Any, triangles: Any, np: Any) -> Any:
    """Vectorized exact point/triangle distance, including degenerate faces."""
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    ab = b - a
    ac = c - a
    ap = point - a
    ab_ab = np.einsum("ij,ij->i", ab, ab)
    ab_ac = np.einsum("ij,ij->i", ab, ac)
    ac_ac = np.einsum("ij,ij->i", ac, ac)
    ap_ab = np.einsum("ij,ij->i", ap, ab)
    ap_ac = np.einsum("ij,ij->i", ap, ac)
    denominator = ab_ab * ac_ac - ab_ac * ab_ac
    usable = np.abs(denominator) > 1.0e-30
    bary_b = np.divide(
        ac_ac * ap_ab - ab_ac * ap_ac,
        denominator,
        out=np.zeros_like(denominator),
        where=usable,
    )
    bary_c = np.divide(
        ab_ab * ap_ac - ab_ac * ap_ab,
        denominator,
        out=np.zeros_like(denominator),
        where=usable,
    )
    inside = usable & (bary_b >= 0.0) & (bary_c >= 0.0) & (bary_b + bary_c <= 1.0)
    projected = a + bary_b[:, None] * ab + bary_c[:, None] * ac
    plane_difference = point - projected
    plane_distance = np.einsum("ij,ij->i", plane_difference, plane_difference)
    edge_distance = np.minimum.reduce(
        (
            _point_segment_distance_squared(point, a, b, np),
            _point_segment_distance_squared(point, b, c, np),
            _point_segment_distance_squared(point, c, a, np),
        )
    )
    return np.where(inside, plane_distance, edge_distance)


def _closest_point_on_triangle(point: Any, triangle: Any, np: Any) -> tuple[Any, list[float]]:
    """Return a deterministic closest point and barycentric coordinates."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [1.0, 0.0, 0.0]
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, [0.0, 1.0, 0.0]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        return a + fraction * ab, [1.0 - fraction, fraction, 0.0]
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, [0.0, 0.0, 1.0]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return a + fraction * ac, [1.0 - fraction, 0.0, fraction]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + fraction * (c - b), [0.0, 1.0 - fraction, fraction]
    denominator = va + vb + vc
    if abs(denominator) <= 1.0e-30:
        # Degenerate triangles are uncommon in the source assets. Choose the
        # closest edge deterministically rather than returning NaN provenance.
        candidates: list[tuple[float, int, Any, list[float]]] = []
        for ordinal, (start, end, bary_start, bary_end) in enumerate(
            ((a, b, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
             (b, c, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
             (c, a, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
        ):
            edge = end - start
            edge_norm = float(np.dot(edge, edge))
            fraction = 0.0 if edge_norm <= 1.0e-30 else max(
                0.0, min(1.0, float(np.dot(point - start, edge)) / edge_norm)
            )
            closest = start + fraction * edge
            barycentric = [
                (1.0 - fraction) * bary_start[index] + fraction * bary_end[index]
                for index in range(3)
            ]
            candidates.append((float(np.dot(point - closest, point - closest)), ordinal, closest, barycentric))
        _, _, closest, barycentric = min(candidates, key=lambda value: (value[0], value[1]))
        return closest, barycentric
    inverse = 1.0 / denominator
    bary_b = vb * inverse
    bary_c = vc * inverse
    bary_a = 1.0 - bary_b - bary_c
    return a + bary_b * ab + bary_c * ac, [bary_a, bary_b, bary_c]


def _compiled_meshes_by_body(model: Any, mujoco: Any, np: Any) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != mesh_type:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        mesh_id = int(model.geom_dataid[geom_id])
        if body_id <= 0 or mesh_id < 0:
            continue
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count], dtype=float
        )
        rotation = _wxyz_rotation_matrix(model.geom_quat[geom_id], np)
        position = np.asarray(model.geom_pos[geom_id], dtype=float)
        # ``einsum`` avoids a NumPy 2.2/MuJoCo strided-view matmul warning
        # observed on otherwise finite compiled mesh arrays.
        vertices = np.einsum("ki,ji->kj", vertices, rotation) + position
        faces = np.asarray(
            model.mesh_face[face_address : face_address + face_count], dtype=int
        )
        if vertices.shape != (vertex_count, 3) or faces.shape != (face_count, 3):
            raise RuntimeError(f"compiled mesh {mesh_id} has malformed topology")
        if not bool(np.all(np.isfinite(vertices))):
            raise RuntimeError(f"compiled mesh {mesh_id} contains non-finite vertices")
        if face_count and (int(np.min(faces)) < 0 or int(np.max(faces)) >= vertex_count):
            raise RuntimeError(f"compiled mesh {mesh_id} has out-of-range face indices")
        result[body_id].append(
            {
                "geom_id": geom_id,
                "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "mesh_id": mesh_id,
                "mesh_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
                "vertices": vertices,
                "faces": faces,
            }
        )
    for meshes in result.values():
        meshes.sort(key=lambda mesh: (int(mesh["geom_id"]), int(mesh["mesh_id"])))
    return dict(result)


def audit_source_bone_proximity(sources: Path, maximum_distance_m: float = 0.012) -> dict[str, Any]:
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise RuntimeError("maximum source-bone distance must be finite and positive")
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "source-bone proximity audit requires the pinned MyoSim/MuJoCo environment"
        ) from error

    exported = export_fullbody(sources)
    model = build_model("myofullbody")
    meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    bodies = {int(body["id"]): body for body in exported["bodies"]}
    sites = {int(site["id"]): site for site in exported["sites"]}
    endpoints: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    body_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for muscle in sorted(exported["muscles"], key=lambda value: int(value["id"])):
        route = muscle["route"]
        for endpoint_name, route_node_index in (("origin", 0), ("insertion", len(route) - 1)):
            route_node = route[route_node_index]
            if route_node.get("kind") != "site":
                raise RuntimeError(f"terminal route node for {muscle['name']} is not a site")
            site_id = int(route_node["source_id"])
            site = sites.get(site_id)
            if site is None:
                raise RuntimeError(f"terminal source site {site_id} was not exported")
            body_id = int(site["body"])
            body = bodies.get(body_id)
            if body is None:
                raise RuntimeError(f"terminal source site {site_id} has unknown body {body_id}")
            point = np.asarray(site["position_body_m"], dtype=float)
            candidates: list[dict[str, Any]] = []
            for mesh in meshes_by_body.get(body_id, []):
                faces = mesh["faces"]
                if not len(faces):
                    continue
                triangles = mesh["vertices"][faces]
                distances_squared = _point_triangle_distances_squared(point, triangles, np)
                triangle_index = int(np.argmin(distances_squared))
                closest, barycentric = _closest_point_on_triangle(
                    point, triangles[triangle_index], np
                )
                distance_m = float(np.linalg.norm(point - closest))
                candidates.append(
                    {
                        "source_geom_id": int(mesh["geom_id"]),
                        "source_geom_name": mesh["geom_name"],
                        "source_mesh_id": int(mesh["mesh_id"]),
                        "source_mesh_name": mesh["mesh_name"],
                        "vertex_count": int(len(mesh["vertices"])),
                        "triangle_count": int(len(faces)),
                        "nearest_triangle_index": triangle_index,
                        "nearest_point_body_m": [float(value) for value in closest],
                        "nearest_barycentric": [float(value) for value in barycentric],
                        "distance_m": distance_m,
                    }
                )
            candidates.sort(
                key=lambda value: (
                    float(value["distance_m"]),
                    int(value["source_geom_id"]),
                    int(value["source_mesh_id"]),
                    int(value["nearest_triangle_index"]),
                )
            )
            nearest = candidates[0] if candidates else None
            if nearest is None:
                classification = "source_body_has_no_mesh"
            elif float(nearest["distance_m"]) <= maximum_distance_m:
                classification = "source_model_bone_adjacent"
            else:
                classification = "source_model_not_bone_adjacent"
            class_counts[classification] += 1
            body_counts[body_id][classification] += 1
            endpoints.append(
                {
                    "source_actuator_index": int(muscle["id"]),
                    "muscle": str(muscle["name"]),
                    "endpoint": endpoint_name,
                    "route_node_index": int(route_node_index),
                    "source_site_id": site_id,
                    "source_site_name": str(site["name"]),
                    "source_site_position_body_m": [float(value) for value in point],
                    "source_body_id": body_id,
                    "source_body_name": str(body["name"]),
                    "classification": classification,
                    "nearest_source_bone_mesh": nearest,
                    "same_body_mesh_candidate_count": len(candidates),
                }
            )

    expected = 2 * len(exported["muscles"])
    if len(endpoints) != expected:
        raise RuntimeError(f"source audit produced {len(endpoints)} endpoints, expected {expected}")
    body_summary = []
    for body_id in sorted(body_counts):
        counts = body_counts[body_id]
        nearest_distances = [
            float(endpoint["nearest_source_bone_mesh"]["distance_m"])
            for endpoint in endpoints
            if int(endpoint["source_body_id"]) == body_id
            and endpoint["nearest_source_bone_mesh"] is not None
        ]
        body_summary.append(
            {
                "source_body_id": body_id,
                "source_body_name": str(bodies[body_id]["name"]),
                "endpoint_count": sum(counts.values()),
                "classification_counts": dict(sorted(counts.items())),
                "minimum_distance_m": min(nearest_distances) if nearest_distances else None,
                "maximum_distance_m": max(nearest_distances) if nearest_distances else None,
            }
        )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "source_mechanics_geometry_audit_only",
        "source": exported["source"],
        "model": {
            "name": exported["model"]["name"],
            "muscle_count": len(exported["muscles"]),
            "endpoint_count": len(endpoints),
            "compiled_mesh_geom_count": sum(len(meshes) for meshes in meshes_by_body.values()),
            "body_count_with_mesh": len(meshes_by_body),
        },
        "gate": {
            "maximum_source_bone_distance_m": maximum_distance_m,
            "comparison": "nearest exact triangle on a compiled mesh geom attached to the same source body",
        },
        "summary": {
            "classification_counts": dict(sorted(class_counts.items())),
            "bodies": body_summary,
        },
        "endpoints": endpoints,
        "evidence_boundary": (
            "This artifact measures authored terminal sites against compiled MyoSim mesh geometry in the "
            "same source body frame. It does not register BodyParts3D, infer anatomical enthesis semantics, "
            "move a source endpoint, or admit an NHTENDON2 surface force law."
        ),
    }


def registration_worklist(
    source_audit: dict[str, Any],
    tendon_manifest: dict[str, Any],
    *,
    source_audit_file: Path | None = None,
    tendon_manifest_file: Path | None = None,
) -> dict[str, Any]:
    if source_audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("registration worklist requires a MyoSim source-bone proximity v1 audit")
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise ValueError("registration worklist requires an NHTENDON2 or NHTENDON3 manifest")
    audit_source = source_audit.get("source")
    tendon_source = tendon_manifest.get("source")
    if not isinstance(audit_source, dict) or not isinstance(tendon_source, dict):
        raise ValueError("registration worklist inputs have incomplete provenance")
    archive_sha = audit_source.get("archive_sha256")
    if not isinstance(archive_sha, str) or archive_sha != tendon_source.get("myosim_archive_sha256"):
        raise ValueError("source audit and tendon manifest do not share the pinned MyoSim archive")
    audit_endpoints = source_audit.get("endpoints")
    tendon_endpoints = tendon_manifest.get("endpoints")
    if not isinstance(audit_endpoints, list) or not isinstance(tendon_endpoints, list):
        raise ValueError("registration worklist inputs have no endpoint arrays")
    audit_index: dict[tuple[int, str], dict[str, Any]] = {}
    for endpoint in audit_endpoints:
        key = (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"]))
        if key in audit_index:
            raise ValueError(f"source audit repeats endpoint identity {key}")
        audit_index[key] = endpoint
    tendon_index: dict[tuple[int, str], dict[str, Any]] = {}
    for endpoint in tendon_endpoints:
        key = (int(endpoint["muscle_index"]), str(endpoint["endpoint"]))
        if key in tendon_index:
            raise ValueError(f"tendon manifest repeats endpoint identity {key}")
        tendon_index[key] = endpoint
    if set(audit_index) != set(tendon_index):
        missing_audit = sorted(set(tendon_index) - set(audit_index))
        missing_tendon = sorted(set(audit_index) - set(tendon_index))
        raise ValueError(
            f"source/tendon endpoint identities differ: audit missing {missing_audit[:3]}, "
            f"tendon missing {missing_tendon[:3]}"
        )

    work_items: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    body_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for key in sorted(tendon_index):
        tendon = tendon_index[key]
        audit = audit_index[key]
        if str(tendon["muscle"]) != str(audit["muscle"]):
            raise ValueError(f"source/tendon muscle name mismatch for endpoint {key}")
        if tendon.get("attachment_mode") in {
            "registered_bone_distributed_envelope",
            "registered_bone_migrated_distributed_envelope",
            "registered_source_surface_distributed_envelope",
            "registered_source_composite_surface_distributed_envelope",
        }:
            disposition_counts["already_surface_admitted"] += 1
            continue
        reason = str(tendon.get("admission_reason"))
        source_class = str(audit.get("classification"))
        if reason == "source_thorax_non_rib_component_endpoint":
            disposition = "source_thorax_non_rib_component_endpoint"
        elif reason == "source_model_non_bone_endpoint":
            disposition = "source_model_non_bone_endpoint"
        elif reason in {
            "surface_distance_exceeds_gate",
            "semantic_enthesis_representative_distance_exceeds_gate",
        }:
            if source_class == "source_model_bone_adjacent":
                disposition = "bodyparts_registration_candidate"
            elif source_class == "source_model_not_bone_adjacent":
                disposition = "source_model_non_bone_endpoint"
            else:
                disposition = "source_body_mesh_missing"
        elif reason == "surface_patch_conditioning_failed_after_topology_aware_exact_surface_points":
            disposition = "surface_patch_conditioning_backlog"
        elif reason == "body_has_no_registered_bone_surface":
            disposition = "bodyparts_surface_mapping_missing"
        elif reason == "body_has_multiple_bone_members_without_semantic_enthesis_map":
            disposition = "semantic_bone_member_resolution_needed"
        else:
            disposition = "other_point_fallback"
        disposition_counts[disposition] += 1
        body_key = (int(audit["source_body_id"]), str(audit["source_body_name"]))
        body_counts[body_key][disposition] += 1
        nearest = audit.get("nearest_source_bone_mesh")
        work_items.append(
            {
                "disposition": disposition,
                "source_actuator_index": int(audit["source_actuator_index"]),
                "muscle": str(audit["muscle"]),
                "endpoint": str(audit["endpoint"]),
                "source_site_id": int(audit["source_site_id"]),
                "source_body_id": int(audit["source_body_id"]),
                "source_body_name": str(audit["source_body_name"]),
                "source_route_node_ordinal": int(audit["route_node_index"]),
                "core_site_index": int(tendon["source_site_index"]),
                "core_body_index": int(tendon["body_index"]),
                "core_route_node_index": int(tendon["route_node_index"]),
                "source_model_classification": source_class,
                "source_model_bone_distance_m": (
                    float(nearest["distance_m"]) if isinstance(nearest, dict) else None
                ),
                "current_admission_reason": reason,
            }
        )
    work_items.sort(
        key=lambda item: (
            str(item["disposition"]),
            str(item["source_body_name"]),
            int(item["source_actuator_index"]),
            str(item["endpoint"]),
        )
    )
    body_summary = [
        {
            "source_body_id": body_id,
            "source_body_name": body_name,
            "work_item_count": sum(counts.values()),
            "disposition_counts": dict(sorted(counts.items())),
        }
        for (body_id, body_name), counts in sorted(body_counts.items(), key=lambda value: value[0])
    ]
    input_records: dict[str, Any] = {
        "myosim_archive_sha256": archive_sha,
        "bodyparts3d_bone_payload": tendon_source.get("bodyparts3d_bone_payload"),
        "myosim_muscle_payload_sha256": tendon_source.get("myosim_muscle_payload_sha256"),
    }
    if source_audit_file is not None:
        input_records["source_audit"] = {
            "file": source_audit_file.name,
            "sha256": _sha256(source_audit_file),
        }
    if tendon_manifest_file is not None:
        input_records["tendon_manifest"] = {
            "file": tendon_manifest_file.name,
            "sha256": _sha256(tendon_manifest_file),
        }
    total = len(tendon_endpoints)
    admitted = int(disposition_counts.get("already_surface_admitted", 0))
    return {
        "schema": WORKLIST_SCHEMA,
        "status": "fail_closed_registration_and_endpoint_semantics_worklist",
        "source": input_records,
        "summary": {
            "endpoint_count": total,
            "already_surface_admitted_count": admitted,
            "point_fallback_count": total - admitted,
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "bodies": body_summary,
        },
        "work_items": work_items,
        "recommended_order": [
            {
                "step": 1,
                "disposition": "bodyparts_registration_candidate",
                "action": "fit and validate named regional source-mesh correspondences without moving MyoSim sites",
            },
            {
                "step": 2,
                "disposition": "surface_patch_conditioning_backlog",
                "action": "improve exact-surface quadrature or retain the point law without relaxing force amplification",
            },
            {
                "step": 3,
                "disposition": "bodyparts_surface_mapping_missing",
                "action": "add a named bone surface only where the source endpoint is anatomically bone-owned",
            },
            {
                "step": 4,
                "disposition": "semantic_bone_member_resolution_needed",
                "action": "review and pin one exact BodyParts3D member identity",
            },
            {
                "step": 5,
                "disposition": "source_thorax_non_rib_component_endpoint",
                "action": "bind to a named costal-cartilage, sternum, or fascia mechanics component after exact tissue classification",
            },
            {
                "step": 6,
                "disposition": "source_model_non_bone_endpoint",
                "action": "classify aponeurosis, fascia, or other body-owned soft-tissue attachment before any surface mechanics",
            },
        ],
        "evidence_boundary": (
            "A registration candidate is close to a same-body MyoSim source mesh but rejected by the current "
            "BodyParts3D surface distance gate. It is a bounded correspondence target, not an admitted enthesis. "
            "A source-model non-bone endpoint must not be repaired by warping a bone toward it. Original MuJoCo "
            "body/site/route IDs and remapped Core body/site/route indices are retained as separate namespaces."
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-distance", type=float, default=0.012)
    arguments = parser.parse_args(argv)
    try:
        result = audit_source_bone_proximity(
            arguments.sources.resolve(), arguments.maximum_distance
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human MyoSim source-bone proximity: {error}", file=sys.stderr)
        return 2
    _write_json(arguments.output, result)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

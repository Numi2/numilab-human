"""Reproduce the bounded registered row-308 precision repair and row-310 proxy.

This is a source-registration operation over one exact retained NHANAT1 ABI5
payload. It does not alter the native runtime or qualify transformed geometry.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .resting_anatomy_interface_patch import (
    HEADER,
    RECORD,
    normals,
    parse_payload,
    signed_volume,
    topology_report,
)
from .resting_pleura_proxy import build_candidate as build_pleura_candidate
from .resting_lobe_fan_refinement import refine_shared_vertex_fans

BASE_PAYLOAD_SHA256 = "90caf4727bebc6889c9b23f4124c795861bed969cae2ec9bbe3127f4b1a07f32"
BASE_RECEIPT_SHA256 = "5dbebdbc50b31e8660985989ab84bf7cba59b348c57b9b105ca041e64352bd39"
FACE_ORIGIN_SHA256 = "c23e2fcb357d8b8be54f0022c37ebfa7ec8606bc1c96f26724528ac9a67793e9"
RESPIRATORY_OWNER_SHA256 = "980b368d8de0cc6cd7e25a93e61e5b6f304275dbd18511207da8cc9919410bb5"
SOURCE308_STABLE_ID = 308
PLEURA_STABLE_ID = 310
COLLAPSE_REMOVE_VERTEX = 37249
COLLAPSE_KEEP_VERTEX = 37247
EXPECTED_REMOVED_FACE_ORIGINS = (74286, 74326)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _connected_component_count(vertex_count: int, faces: np.ndarray) -> int:
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in np.asarray(faces, dtype=np.int64):
        union(int(a), int(b))
        union(int(b), int(c))
    used = np.unique(np.asarray(faces, dtype=np.int64).reshape(-1))
    return len({find(int(index)) for index in used})


def _edge_count(faces: np.ndarray) -> int:
    f = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    return int(len(np.unique(np.sort(edges, axis=1), axis=0)))


def collapse_midpoint_edge(vertices6, faces, face_origins, *, remove_vertex: int,
                            keep_vertex: int, position_policy: str = "midpoint",
                            max_endpoint_displacement_m: float | None = None,
                            max_abs_volume_delta_m3: float = 1e-12):
    """Collapse one manifold edge to its midpoint or retained endpoint."""
    v = np.asarray(vertices6, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    origins = np.asarray(face_origins, dtype=np.int64)
    if (v.ndim != 2 or v.shape[1] != 6 or f.ndim != 2 or f.shape[1] != 3
            or origins.ndim != 1 or len(origins) != len(f) or not len(f)
            or not np.isfinite(v).all()):
        raise ValueError("malformed edge-collapse inputs")
    if f.min() < 0 or f.max() >= len(v) or np.any(origins < 0):
        raise ValueError("edge-collapse indices or face origins are out of range")
    u, w = int(remove_vertex), int(keep_vertex)
    if u == w or min(u, w) < 0 or max(u, w) >= len(v):
        raise ValueError("edge-collapse endpoint indices are invalid")
    edge_faces = np.flatnonzero(np.sum(np.isin(f, (u, w)), axis=1) == 2)
    if len(edge_faces) != 2:
        raise ValueError(f"edge must have exactly two incident faces, got {len(edge_faces)}")

    def neighbors(vertex: int) -> set[int]:
        incident = f[np.any(f == vertex, axis=1)]
        return set(map(int, incident.reshape(-1))) - {vertex}

    common_neighbors = neighbors(u) & neighbors(w)
    opposite_vertices: set[int] = set()
    for face_index in edge_faces:
        opposite_vertices.update(map(int, f[face_index][~np.isin(f[face_index], (u, w))]))
    if len(common_neighbors) != 2 or common_neighbors != opposite_vertices:
        raise ValueError("edge collapse fails the two-manifold link condition")

    p, q = v[u, :3].astype(np.float64), v[w, :3].astype(np.float64)
    edge_length = float(np.linalg.norm(p - q))
    if position_policy == "midpoint":
        collapse_position = ((p + q) * 0.5).astype(np.float32)
    elif position_policy == "keep":
        collapse_position = v[w, :3].copy()
    else:
        raise ValueError("position_policy must be 'midpoint' or 'keep'")
    endpoint_displacements = [
        float(np.linalg.norm(collapse_position.astype(np.float64) - p)),
        float(np.linalg.norm(collapse_position.astype(np.float64) - q)),
    ]
    maximum_displacement = max(endpoint_displacements)
    if max_endpoint_displacement_m is not None:
        if not np.isfinite(max_endpoint_displacement_m) or max_endpoint_displacement_m < 0:
            raise ValueError("max_endpoint_displacement_m must be finite and non-negative")
        if maximum_displacement > float(max_endpoint_displacement_m):
            raise ValueError("edge collapse exceeds the caller endpoint-displacement bound")
    if not np.isfinite(max_abs_volume_delta_m3) or max_abs_volume_delta_m3 < 0:
        raise ValueError("max_abs_volume_delta_m3 must be finite and non-negative")

    replaced = f.copy()
    replaced[replaced == u] = w
    keep = ((replaced[:, 0] != replaced[:, 1]) &
            (replaced[:, 1] != replaced[:, 2]) &
            (replaced[:, 2] != replaced[:, 0]))
    deleted = np.flatnonzero(~keep)
    if not np.array_equal(deleted, edge_faces):
        raise ValueError("edge collapse would delete faces beyond the two edge incident faces")
    kept_old_faces = np.flatnonzero(keep)
    kept_faces = replaced[keep]
    kept_origins = origins[keep]
    used = np.unique(kept_faces.reshape(-1))
    old_to_new_vertex = np.full(len(v), -1, dtype=np.int64)
    old_to_new_vertex[used] = np.arange(len(used), dtype=np.int64)
    out_faces = old_to_new_vertex[kept_faces]
    out_vertices = v[used].copy()
    survivor = int(old_to_new_vertex[w])
    if survivor < 0 or old_to_new_vertex[u] >= 0:
        raise ValueError("edge collapse did not remove exactly the selected endpoint")
    out_vertices[survivor, :3] = collapse_position
    out_vertices[:, 3:6] = normals(out_vertices[:, :3].astype(np.float64), out_faces).astype(np.float32)

    source_topology = topology_report(f)
    result_topology = topology_report(out_faces)
    source_edge_count = _edge_count(f)
    source_euler = len(v) - source_edge_count + len(f)
    source_components = _connected_component_count(len(v), f)
    source_duplicates = len(f) - len(np.unique(np.sort(f, axis=1), axis=0))
    source_tri = v[f, :3].astype(np.float64)
    source_twice_area = np.linalg.norm(
        np.cross(source_tri[:, 1] - source_tri[:, 0], source_tri[:, 2] - source_tri[:, 0]), axis=1
    )
    tri = out_vertices[out_faces, :3].astype(np.float64)
    twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if (source_topology["boundary_edge_count"] or source_topology["nonmanifold_edge_count"]
            or source_topology["orientation_error_edge_count"] or source_duplicates
            or source_components != 1 or source_euler > 2 or (2 - source_euler) % 2
            or np.any(source_twice_area <= 0) or len(np.unique(f.reshape(-1))) != len(v)):
        raise ValueError("edge-collapse source is not one closed oriented connected surface of valid genus")
    if (result_topology["boundary_edge_count"] or result_topology["nonmanifold_edge_count"]
            or result_topology["orientation_error_edge_count"] or np.any(twice_area <= 0)):
        raise ValueError("edge-collapse result is not closed, oriented, and nondegenerate")
    duplicate_faces = len(out_faces) - len(np.unique(np.sort(out_faces, axis=1), axis=0))
    edge_count = _edge_count(out_faces)
    euler = len(out_vertices) - edge_count + len(out_faces)
    components = _connected_component_count(len(out_vertices), out_faces)
    if duplicate_faces or components != source_components or euler != source_euler:
        raise ValueError("edge-collapse result does not preserve connected topology and Euler characteristic")
    if euler > 2 or (2 - euler) % 2:
        raise ValueError("edge-collapse result has an invalid closed orientable Euler characteristic")

    incident_star = np.flatnonzero(np.any((f == u) | (f == w), axis=1))
    star_vertex_ids = np.unique(f[incident_star].reshape(-1))
    star_points = v[star_vertex_ids, :3].astype(np.float64)
    star_diameter = float(np.linalg.norm(
        star_points[:, None, :] - star_points[None, :, :], axis=2).max())
    old_incident_survivors = np.flatnonzero(
        keep & (np.any(f == u, axis=1) | np.any(f == w, axis=1)))
    new_local_faces = old_to_new_vertex[replaced[old_incident_survivors]]
    old_tri = v[f[old_incident_survivors], :3].astype(np.float64)
    new_tri = out_vertices[new_local_faces, :3].astype(np.float64)
    old_normal = np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0])
    new_normal = np.cross(new_tri[:, 1] - new_tri[:, 0], new_tri[:, 2] - new_tri[:, 0])
    normal_dots = np.einsum("ij,ij->i", old_normal, new_normal)
    if np.any(normal_dots <= 0):
        raise ValueError("edge collapse reverses a surviving face in the changed local star")
    volume_before = signed_volume(v[:, :3].astype(np.float64), f)
    volume_after = signed_volume(out_vertices[:, :3].astype(np.float64), out_faces)
    volume_delta = volume_after - volume_before
    if abs(volume_delta) > float(max_abs_volume_delta_m3):
        raise ValueError("edge collapse exceeds the caller signed-volume bound")
    old_to_new_face = np.full(len(f), -1, dtype=np.int64)
    old_to_new_face[kept_old_faces] = np.arange(len(kept_old_faces), dtype=np.int64)
    affected_output_faces = old_to_new_face[old_incident_survivors]
    if bool((affected_output_faces < 0).any()):
        raise ValueError("edge-collapse changed-face lineage is inconsistent")

    report = {
        "operation": "one topology-preserving midpoint edge collapse",
        "removed_vertex_before_compaction": u,
        "retained_vertex_before_compaction": w,
        "retained_vertex_after_compaction": survivor,
        "edge_length_m": edge_length,
        "collapse_position_policy": position_policy,
        "collapse_position_f32_m": collapse_position.astype(float).tolist(),
        "endpoint_displacements_m": endpoint_displacements,
        "maximum_endpoint_displacement_m": maximum_displacement,
        "changed_star_diameter_m": star_diameter,
        "changed_star_bound_interpretation": (
            "coarse support-diameter bound: all old and new changed triangles lie in the convex hull of the old edge star"
        ),
        "common_neighbors": sorted(common_neighbors),
        "opposite_vertices": sorted(opposite_vertices),
        "deleted_face_indices": deleted.astype(int).tolist(),
        "deleted_face_origins": origins[deleted].astype(int).tolist(),
        "surviving_changed_face_count": int(len(old_incident_survivors)),
        "minimum_surviving_changed_face_normal_dot": float(normal_dots.min()),
        "source_topology": {
            "vertex_count": int(len(v)), "face_count": int(len(f)),
            "genus": int((2 - source_euler) // 2),
            "edge_count": source_edge_count,
            "boundary_edges": source_topology["boundary_edge_count"],
            "nonmanifold_edges": source_topology["nonmanifold_edge_count"],
            "orientation_error_edges": source_topology["orientation_error_edge_count"],
            "duplicate_faces": int(source_duplicates), "connected_components": source_components,
            "euler_characteristic": int(source_euler),
        },
        "candidate_topology": {
            "vertex_count": int(len(out_vertices)), "face_count": int(len(out_faces)),
            "genus": int((2 - euler) // 2),
            "edge_count": edge_count,
            "boundary_edges": result_topology["boundary_edge_count"],
            "nonmanifold_edges": result_topology["nonmanifold_edge_count"],
            "orientation_error_edges": result_topology["orientation_error_edge_count"],
            "duplicate_faces": int(duplicate_faces), "connected_components": components,
            "euler_characteristic": int(euler), "zero_area_faces": 0,
            "minimum_triangle_area_m2": float(twice_area.min() * 0.5),
        },
        "signed_volume_before_m3": volume_before,
        "signed_volume_after_m3": volume_after,
        "signed_volume_delta_m3": volume_delta,
        "source_face_indices_in_changed_one_ring": incident_star.astype(int).tolist(),
        "surviving_changed_face_lineage": [
            {"output_face_index": int(new), "previous_face_index": int(old),
             "previous_face_origin": int(origins[old])}
            for old, new in zip(old_incident_survivors, affected_output_faces)
        ],
        "deleted_face_origins": origins[deleted].astype(int).tolist(),
        "face_origins_preserved_for_surviving_faces": True,
    }
    return out_vertices, out_faces, kept_origins, report


def _serialize_payload(header, record_order, rows) -> bytes:
    raw_records = []
    vertex_blocks = []
    index_blocks = []
    vertex_count = index_count = 0
    for sid in record_order:
        row = rows[int(sid)]
        vertices = np.ascontiguousarray(row["vertices6"], dtype="<f4")
        faces = np.asarray(row["faces"], dtype=np.int64)
        if (vertices.ndim != 2 or vertices.shape[1] != 6 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(f"invalid geometry for stable anatomy ID {sid}")
        raw_records.append(RECORD.pack(
            int(row["body_index"]), vertex_count, len(vertices), index_count, faces.size,
            int(sid), int(row["layer"]), int(row["flags"]),
        ))
        vertex_blocks.append(vertices.tobytes())
        index_blocks.append((faces.reshape(-1) + vertex_count).astype("<u4").tobytes())
        row["vertices6"] = vertices
        row["faces"] = faces
        vertex_count += len(vertices)
        index_count += faces.size
    output_header = HEADER.pack(
        header[0], header[1], len(record_order), vertex_count, index_count, header[5], header[6]
    )
    return output_header + b"".join(raw_records) + b"".join(vertex_blocks) + b"".join(index_blocks)


def _load_pinned_respiratory_owner(path: Path):
    if sha256(path) != RESPIRATORY_OWNER_SHA256:
        raise ValueError("respiratory Kuhn-field owner SHA does not match the retained source")
    module_name = "numilab_human._pinned_resting_respiratory_conforming_field"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the pinned respiratory Kuhn-field owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _derive_basal_effective_area(rows, kuhn_basis):
    details = []
    total = 0.0
    for sid in (305, 306, 307, 308, 309):
        row = rows[sid]
        xyz = np.asarray(row["vertices6"], dtype=np.float32)[:, :3].astype(np.float64)
        faces = np.asarray(row["faces"], dtype=np.int64)
        weights = np.asarray(kuhn_basis(xyz)[0], dtype=np.float64)
        tri = xyz[faces]
        w = weights[faces]
        velocity = np.zeros_like(tri)
        velocity[:, :, 1] = -w
        derivative = (
            np.einsum("ij,ij->i", velocity[:, 0], np.cross(tri[:, 1], tri[:, 2]))
            + np.einsum("ij,ij->i", tri[:, 0], np.cross(velocity[:, 1], tri[:, 2]))
            + np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], velocity[:, 2]))
        ) / 6.0
        area = float(derivative.sum(dtype=np.float64))
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"invalid source-derived basal area for lobe {sid}")
        total += area
        details.append({"lung_stable_id": sid, "effective_area_m2_float64": area,
                        "effective_area_m2_runtime_float32": float(np.float32(area))})
    return total, details


def _source_precision_screen(rows):
    """Measure the final serialized-F32 lobe altitude risk after every edit."""
    result = {}
    for sid in range(305, 310):
        xyz = np.asarray(rows[sid]["vertices6"], dtype="<f4")[:, :3].astype(np.float64)
        faces = np.asarray(rows[sid]["faces"], dtype=np.int64)
        tri = xyz[faces]
        edge_lengths = np.stack((
            np.linalg.norm(tri[:, 1]-tri[:, 2], axis=1),
            np.linalg.norm(tri[:, 2]-tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 0]-tri[:, 1], axis=1),
        ), axis=1)
        twice_area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
        if (not np.isfinite(edge_lengths).all() or not np.isfinite(twice_area).all()
                or bool((edge_lengths <= 0.0).any())):
            raise ValueError(f"final source lobe {sid} has invalid triangle geometry")
        altitude = twice_area[:, None] / edge_lengths
        face_minimum = altitude.min(axis=1)
        result[str(sid)] = {
            "vertex_count": int(len(xyz)),
            "face_count": int(len(faces)),
            "minimum_triangle_altitude_m": float(face_minimum.min()),
            "zero_area_face_count": int(np.count_nonzero(twice_area == 0.0)),
            "face_count_below_128nm": int(np.count_nonzero(face_minimum < 128e-9)),
            "face_count_below_512nm": int(np.count_nonzero(face_minimum < 512e-9)),
            "face_count_below_1um": int(np.count_nonzero(face_minimum < 1e-6)),
            "basis": "minimum of the three double-precision altitudes computed from final serialized Float32 source positions",
        }
    return result


def _source_volume_rows(rows):
    details = []
    total = 0.0
    for sid in (305, 306, 307, 308, 309):
        row = rows[sid]
        volume = signed_volume(
            np.asarray(row["vertices6"], dtype=np.float32)[:, :3].astype(np.float64),
            np.asarray(row["faces"], dtype=np.int64),
        )
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError(f"invalid outward signed volume for lobe {sid}")
        details.append({"lung_stable_id": sid, "signed_volume_m3": volume,
                        "enclosed_volume_m3": abs(volume)})
        total += abs(volume)
    return total, details


def build_candidate(base_payload: Path, base_receipt_path: Path, face_origin_path: Path,
                    respiratory_owner_path: Path, output_dir: Path):
    base_payload = Path(base_payload)
    base_receipt_path = Path(base_receipt_path)
    face_origin_path = Path(face_origin_path)
    respiratory_owner_path = Path(respiratory_owner_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing candidate output: {output_dir}")
    for path, expected, label in (
        (base_payload, BASE_PAYLOAD_SHA256, "565 anatomy payload"),
        (base_receipt_path, BASE_RECEIPT_SHA256, "565 anatomy receipt"),
        (face_origin_path, FACE_ORIGIN_SHA256, "row-308 source face-origin map"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} SHA does not match the retained input")
    receipt_raw = base_receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    if (receipt.get("schema") != "numi.human.resting-anatomy-receipt.v1"
            or receipt.get("payload", {}).get("sha256") != BASE_PAYLOAD_SHA256):
        raise ValueError("source receipt does not bind the exact integrated 565 payload")
    header, original_rows = parse_payload(base_payload)
    row_records = [RECORD.unpack_from(base_payload.read_bytes(), HEADER.size + i * RECORD.size)
                   for i in range(int(header[2]))]
    record_order = [int(record[5]) for record in row_records]
    receipt_ids = set(map(int, receipt.get("provenance", {}).get("source_id_map", {})))
    if (len(set(record_order)) != len(record_order) or set(record_order) != set(original_rows)
            or not {305, 306, 307, 308, 309, 310, 311}.issubset(receipt_ids)
            or set(record_order) - receipt_ids != {22}):
        raise ValueError("NHA records and source receipt IDs disagree with the pinned 565 exception")
    row308 = original_rows.get(SOURCE308_STABLE_ID)
    row310 = original_rows.get(PLEURA_STABLE_ID)
    if (row308 is None or row310 is None or int(row308["body_index"]) != 20
            or int(row308["layer"]) != 7 or int(row310["body_index"]) != 20
            or int(row310["layer"]) != 8):
        raise ValueError("retained 565 payload has unexpected row-308/310 ownership")
    source_map = receipt["provenance"]["source_id_map"]
    if (source_map["308"].get("name") != "Superior lobe of left lung"
            or source_map["310"].get("name") != "Pleura"):
        raise ValueError("retained stable-ID source identities do not match the repair owner")
    face_origins = np.load(face_origin_path, allow_pickle=False).astype(np.int64, copy=False)
    if len(face_origins) != len(row308["faces"]):
        raise ValueError("pinned row-308 face-origin input does not match source face count")

    vertices_after, faces_after, origins_after, collapse_report = collapse_midpoint_edge(
        row308["vertices6"], row308["faces"], face_origins,
        remove_vertex=COLLAPSE_REMOVE_VERTEX, keep_vertex=COLLAPSE_KEEP_VERTEX,
    )
    if tuple(sorted(collapse_report["deleted_face_origins"])) != EXPECTED_REMOVED_FACE_ORIGINS:
        raise ValueError("midpoint collapse did not remove the two diagnosed source face origins")
    if collapse_report["edge_length_m"] > 5e-8:
        raise ValueError("diagnosed row-308 edge is no longer sub-50-nm")

    output_dir.mkdir(parents=True)
    pre_dir = output_dir / "pre-pleura"
    final_dir = output_dir / "final"
    pre_dir.mkdir()
    rows = {sid: dict(row) for sid, row in original_rows.items()}
    rows[308] = dict(row308)
    rows[308]["vertices6"] = vertices_after
    rows[308]["faces"] = faces_after
    pre_blob = _serialize_payload(header, record_order, rows)
    pre_payload_path = pre_dir / "resting-thorax.nhanatomy"
    pre_payload_path.write_bytes(pre_blob)
    pre_payload_sha = sha256(pre_payload_path)

    pre_receipt = json.loads(receipt_raw)
    pre_receipt["payload"].update({
        "path": str(pre_payload_path), "input_payload_sha256": BASE_PAYLOAD_SHA256,
        "sha256": pre_payload_sha, "vertex_count": int(HEADER.unpack_from(pre_blob)[3]),
        "index_count": int(HEADER.unpack_from(pre_blob)[4]),
    })
    pre_receipt["functional_bindings"]["anatomy_payload_sha256"] = pre_payload_sha
    repair = pre_receipt["provenance"]["source_id_map"]["308"].setdefault("repair", {})
    repair["bounded_source_precision_conditioning"] = {
        "method": collapse_report["operation"],
        "retained_vertex_before_compaction": COLLAPSE_KEEP_VERTEX,
        "removed_vertex_before_compaction": COLLAPSE_REMOVE_VERTEX,
        "removed_source_face_origins": EXPECTED_REMOVED_FACE_ORIGINS,
        "maximum_endpoint_displacement_m": collapse_report["maximum_endpoint_displacement_m"],
        "source_face_origin_sha256": FACE_ORIGIN_SHA256,
    }
    total_volume, volume_rows = _source_volume_rows(rows)
    pre_receipt["thorax_source_volume_m3"]["five_lung_envelopes"] = [
        row["enclosed_volume_m3"] for row in volume_rows
    ]
    pre_receipt["thorax_source_volume_m3"]["sum"] = total_volume
    pre_receipt_path = pre_dir / "resting-anatomy-receipt.json"
    pre_receipt_path.write_text(json.dumps(pre_receipt, indent=2, sort_keys=True) + "\n")

    respiratory_owner = _load_pinned_respiratory_owner(respiratory_owner_path)
    area64, area_rows = _derive_basal_effective_area(rows, respiratory_owner.kuhn_basis)
    area32 = float(np.float32(area64))
    resp = pre_receipt["functional_bindings"].get("respiratory_geometry_binding")
    if not isinstance(resp, dict):
        raise ValueError("source receipt lacks its respiratory geometry binding")
    resp["diaphragm_effective_area_m2"] = area32
    resp["per_lobe_effective_area_m2"] = [
        {"lung_stable_id": x["lung_stable_id"],
         "effective_area_m2": x["effective_area_m2_runtime_float32"]}
        for x in area_rows
    ]
    area_update = resp.setdefault("source_refinement_area_update", {})
    area_update.update({
        "basis": "Exact analytic derivative of each closed lobe signed tetrahedral volume under the pinned Kuhn basal field, evaluated in Float64 on current serialized Float32 rows.",
        "source_owner_path": str(respiratory_owner_path),
        "source_owner_sha256": RESPIRATORY_OWNER_SHA256,
        "candidate_payload_sha256": pre_payload_sha,
        "per_lobe_derivatives_m2": {
            str(x["lung_stable_id"]): x["effective_area_m2_float64"] for x in area_rows
        },
        "refined_area_m2_float64": area64,
        "refined_area_m2_runtime_float32": area32,
    })
    # Refresh the pre-stage receipt now that the binding has been derived.
    pre_receipt_path.write_text(json.dumps(pre_receipt, indent=2, sort_keys=True) + "\n")
    build_pleura_candidate(pre_payload_path, pre_receipt_path, final_dir)

    final_payload_path = final_dir / "resting-thorax.nhanatomy"
    final_receipt_path = final_dir / "resting-anatomy-receipt.json"
    final_header, final_rows = parse_payload(final_payload_path)
    if set(final_rows) != set(original_rows):
        raise ValueError("pleura owner changed the stable anatomy surface inventory")
    def geometry_bytes(row):
        return (
            np.ascontiguousarray(row["vertices6"], dtype="<f4").tobytes()
            + np.ascontiguousarray(row["faces"], dtype="<i8").tobytes()
        )

    for sid in original_rows:
        if sid in (308, 310):
            continue
        if geometry_bytes(final_rows[sid]) != geometry_bytes(original_rows[sid]):
            raise ValueError(f"candidate changed unrelated stable anatomy row bytes {sid}")
    if (not np.array_equal(final_rows[308]["vertices6"], vertices_after)
            or not np.array_equal(final_rows[308]["faces"], faces_after)):
        raise ValueError("pleura owner changed repaired row 308")

    # Exact per-face source lineage for the derived external boundary.
    def point_key(point):
        return tuple(int(x) for x in np.ascontiguousarray(np.asarray(point, dtype="<f4")).view("<u4"))

    def face_key(points):
        return tuple(sorted(point_key(point) for point in points))

    source_faces = collections.defaultdict(list)
    for sid in (305, 306, 307, 308, 309):
        source_xyz = final_rows[sid]["vertices6"][:, :3]
        for face_index, triangle in enumerate(final_rows[sid]["faces"]):
            source_faces[face_key(source_xyz[triangle])].append((sid, face_index))
    pleura_xyz = final_rows[310]["vertices6"][:, :3]
    lineage = np.empty((len(final_rows[310]["faces"]), 2), dtype="<i4")
    for face_index, triangle in enumerate(final_rows[310]["faces"]):
        matches = source_faces.get(face_key(pleura_xyz[triangle]), [])
        if len(matches) != 1:
            raise ValueError(f"derived pleura face {face_index} has ambiguous source lineage: {matches[:4]}")
        lineage[face_index] = matches[0]
    lineage_path = output_dir / "row-310-face-lineage.npy"
    np.save(lineage_path, lineage, allow_pickle=False)
    origin_output_path = output_dir / "row-308-face-origin.npy"
    np.save(origin_output_path, origins_after.astype("<i8"), allow_pickle=False)

    final_receipt = json.loads(final_receipt_path.read_text())
    final_receipt["provenance"]["lung_source_midpoint_edge_repair"] = {
        "algorithm": "registered_lung_midpoint_edge_collapse_then_existing_pleura_proxy_v1",
        "input_payload_path": str(base_payload), "input_payload_sha256": BASE_PAYLOAD_SHA256,
        "input_receipt_path": str(base_receipt_path), "input_receipt_sha256": BASE_RECEIPT_SHA256,
        "row_308_face_origin_input_path": str(face_origin_path),
        "row_308_face_origin_input_sha256": FACE_ORIGIN_SHA256,
        "row_308_face_origin_output_path": str(origin_output_path),
        "row_308_face_origin_output_sha256": sha256(origin_output_path),
        "respiratory_kuhn_owner_path": str(respiratory_owner_path),
        "respiratory_kuhn_owner_sha256": RESPIRATORY_OWNER_SHA256,
        "row_308_operation": collapse_report,
        "row_310_owner": "numilab_human.resting_pleura_proxy.build_candidate",
        "row_310_face_lineage_path": str(lineage_path),
        "row_310_face_lineage_sha256": sha256(lineage_path),
        "row_310_face_lineage_face_count": int(len(lineage)),
        "row_310_face_lineage_counts_by_lobe": {
            str(sid): int(np.count_nonzero(lineage[:, 0] == sid)) for sid in (305, 306, 307, 308, 309)
        },
        "respiratory_area_derivation": {
            "effective_area_m2_float64": area64,
            "effective_area_m2_runtime_float32": area32,
            "per_lobe": area_rows,
        },
        "source_license": "CC-BY-SA-4.0; inherited source attribution and share-alike obligations apply",
        "qualification_limit": (
            "This source-space repair does not establish native transformed-pose self-intersection clearance; "
            "the native Float32 305/308 seam audit remains a separate retained failure gate."
        ),
    }
    final_receipt_path.write_text(json.dumps(final_receipt, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "numi.human.lung-source-edge-repair-reproduction.v1",
        "status": "source_candidate_built_native_pose_gate_separate",
        "source_payload_path": str(base_payload), "source_payload_sha256": BASE_PAYLOAD_SHA256,
        "source_receipt_path": str(base_receipt_path), "source_receipt_sha256": BASE_RECEIPT_SHA256,
        "face_origin_input_path": str(face_origin_path), "face_origin_input_sha256": FACE_ORIGIN_SHA256,
        "respiratory_owner_path": str(respiratory_owner_path),
        "respiratory_owner_sha256": RESPIRATORY_OWNER_SHA256,
        "pre_pleura_payload_path": str(pre_payload_path), "pre_pleura_payload_sha256": sha256(pre_payload_path),
        "final_payload_path": str(final_payload_path), "final_payload_sha256": sha256(final_payload_path),
        "final_receipt_path": str(final_receipt_path), "final_receipt_sha256": sha256(final_receipt_path),
        "row_308_face_origin_output_path": str(origin_output_path),
        "row_308_face_origin_output_sha256": sha256(origin_output_path),
        "row_310_face_lineage_path": str(lineage_path),
        "row_310_face_lineage_sha256": sha256(lineage_path),
        "row_308": collapse_report,
        "row_310_triangle_count": int(len(final_rows[310]["faces"])),
        "row_310_lineage_counts_by_lobe": {
            str(sid): int(np.count_nonzero(lineage[:, 0] == sid)) for sid in (305, 306, 307, 308, 309)
        },
        "respiratory_area": {
            "effective_area_m2_float64": area64,
            "effective_area_m2_runtime_float32": area32,
            "per_lobe": area_rows,
        },
        "unmodified_surface_geometry_byte_identity": True,
        "native_rendering_qualification": "not performed by source builder",
    }
    report_path = output_dir / "build-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report



PRECISION_PAYLOAD_SHA256 = "c6adae6522a2f7e5a8bd661035d464d686c843e96ab04ffc3ccd9aa1371569c1"
PRECISION_RECEIPT_SHA256 = "64c3b182d4ebc13bed468b72871b269887f8aecc2e7dc43622eb90d20bb0545e"
FAN_REFINEMENT_INPUT_SHA256 = "21599a732e5bc414a37c86e74cc69079c39ff5eb6a2e8ff6036949384a180142"
FAN_REFINEMENT_AUDIT_SHA256 = "0bde320d123412ebe0c5f4f5baf42919b06356f6818f571cbf021331ec000253"
FAN_REFINEMENT_TARGET_FACES = ((305, 40225), (308, 46153))
FAN_REFINEMENT_SHARED_VERTEX_M = (
    -0.0013838123995810747, 0.04402492940425873, -0.0619685985147953,
)



def _triangle_altitudes(xyz, faces):
    tri = np.asarray(xyz, dtype=np.float64)[faces]
    longest = np.maximum.reduce([np.linalg.norm(tri[:, (i+1)%3] - tri[:, i], axis=1)
                                 for i in range(3)])
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    return np.divide(area2, longest, out=np.zeros_like(area2), where=longest > 0)


def _condition_precision_lobes(original_rows, *, minimum_altitude_m, selection_altitude_m=128e-9):
    """Bounded source conditioning; shared interlobar patches flip together."""
    from .surface_precision_retriangulation import (
        build_surface_adjacency, flip_interior_edge, validate_closed_oriented_surface,
    )
    selection_altitude = float(selection_altitude_m)
    if not np.isfinite(selection_altitude) or selection_altitude <= 0:
        raise ValueError("selection altitude must be finite and positive")
    rows = {sid: dict(row) for sid, row in original_rows.items()}
    adjacency, lineage, before_topology, before_volume = {}, {}, {}, {}
    positions, lookup = {}, collections.defaultdict(set)

    def point_key(point):
        return np.ascontiguousarray(point, dtype="<f4").tobytes()

    def face_key(sid, triangle):
        return tuple(sorted(point_key(p) for p in positions[sid][triangle]))

    for sid in range(305, 310):
        positions[sid] = original_rows[sid]["vertices6"][:, :3]
        rows[sid]["faces"] = original_rows[sid]["faces"].copy()
        f = rows[sid]["faces"]
        before_topology[sid] = validate_closed_oriented_surface(positions[sid], f)
        adjacency[sid] = build_surface_adjacency(f)
        before_volume[sid] = signed_volume(positions[sid].astype(np.float64), f)
        lineage[sid] = {}
        for fi, triangle in enumerate(f):
            lookup[face_key(sid, triangle)].add((sid, fi))
    # Row 311 is not a lobe to retriangulate here, but its exact reciprocal
    # diaphragm faces must participate in admission. Any lobe flip touching
    # this interface is protected; a paired repair needs a diaphragm-aware
    # owner that updates its registration ranges as well.
    if 311 in original_rows:
        positions[311] = original_rows[311]["vertices6"][:, :3]
        rows[311]["faces"] = original_rows[311]["faces"].copy()
        for fi, triangle in enumerate(rows[311]["faces"]):
            lookup[face_key(311, triangle)].add((311, fi))
    initial_shared = {key: tuple(sorted(value)) for key, value in lookup.items() if len(value) > 1}
    lobe_ids = tuple(range(305, 310))
    initial_counts = {sid: int((_triangle_altitudes(positions[sid], rows[sid]["faces"]) < selection_altitude).sum())
                      for sid in lobe_ids}
    seeds = sorted((float(h), sid, int(fi))
                   for sid in lobe_ids
                   for fi, h in enumerate(_triangle_altitudes(positions[sid], rows[sid]["faces"]))
                   if h < selection_altitude)
    operations, blocked = [], []

    def apply(sid, candidate, report):
        f = rows[sid]["faces"]
        ids = [r["output_face_index"] for r in report["output_face_lineage"]]
        parent_ids = sorted(set().union(*(lineage[sid].get(fi, {fi}) for fi in ids)))
        for fi in ids:
            lookup[face_key(sid, f[fi])].remove((sid, fi))
        rows[sid]["faces"] = candidate
        for fi in ids:
            lookup[face_key(sid, candidate[fi])].add((sid, fi))
            lineage[sid][fi] = set(parent_ids)
        for change in report["edge_incidence_updates"]:
            edge = tuple(change["edge"])
            if change["after_face_indices"]:
                adjacency[sid].edge_faces[edge] = tuple(change["after_face_indices"])
            else:
                adjacency[sid].edge_faces.pop(edge, None)
        for action in ("remove", "add"):
            for a, b in report["vertex_neighbor_updates"][action]:
                neighbors = set(adjacency[sid].vertex_neighbors[a])
                (neighbors.discard if action == "remove" else neighbors.add)(b)
                adjacency[sid].vertex_neighbors[a] = frozenset(neighbors)
        report["input_source_face_ancestry"] = parent_ids
        report["stable_id"] = sid

    for altitude, sid, seed in seeds:
        f, xyz = rows[sid]["faces"], positions[sid]
        if _triangle_altitudes(xyz, f[seed:seed+1])[0] >= selection_altitude:
            continue
        errors = []
        triangle = f[seed]
        edges = sorted((tuple(sorted((int(triangle[i]), int(triangle[(i+1)%3]))))
                        for i in range(3)),
                       key=lambda edge: -float(np.linalg.norm(xyz[edge[0]].astype(float)-xyz[edge[1]])))
        succeeded = False
        for edge in edges:
            pair = adjacency[sid].edge_faces.get(edge, ())
            if len(pair) != 2:
                errors.append("not two incident faces")
                continue
            partners = [lookup[face_key(sid, f[fi])] - {(sid, fi)} for fi in pair]
            mirrored = None
            if any(partners):
                if not all(len(p) == 1 for p in partners):
                    errors.append("partly shared or ambiguous patch")
                    continue
                (other, oi), (other2, oj) = (next(iter(p)) for p in partners)
                if other != other2 or other == sid:
                    errors.append("shared patch has different lobe owners")
                    continue
                if other not in adjacency:
                    errors.append(f"shared patch with protected non-lobe stable id {other}")
                    continue
                of = rows[other]["faces"]
                coordinates = {}
                for vi in np.unique(of[[oi, oj]]):
                    coordinates.setdefault(point_key(positions[other][vi]), []).append(int(vi))
                matched = [coordinates.get(point_key(xyz[vi]), []) for vi in edge]
                if any(len(m) != 1 for m in matched):
                    errors.append("shared edge does not have unique counterpart vertices")
                    continue
                mirrored = (other, (oi, oj), tuple(m[0] for m in matched))
            proposed = []
            try:
                candidate, detail = flip_interior_edge(
                    xyz, f, edge=edge, face_indices=pair,
                    minimum_altitude_m=minimum_altitude_m, adjacency=adjacency[sid])
                proposed.append((sid, candidate, detail))
                if mirrored:
                    other, other_pair, other_edge = mirrored
                    candidate_other, detail_other = flip_interior_edge(
                        positions[other], rows[other]["faces"], edge=other_edge,
                        face_indices=other_pair, minimum_altitude_m=minimum_altitude_m,
                        adjacency=adjacency[other])
                    proposed.append((other, candidate_other, detail_other))
                for psid, candidate, detail in proposed:
                    if abs(detail["signed_volume_delta_m3"]) > 1e-12:
                        raise ValueError("local signed-volume change exceeds 1e-12 m3")
                    ids = [r["output_face_index"] for r in detail["output_face_lineage"]]
                    if not mirrored and any(lookup.get(face_key(psid, candidate[fi]), set()) - {(psid, fi)}
                                            for fi in ids):
                        raise ValueError("flip would create a new shared patch")
                if mirrored:
                    def pair_keys(entry):
                        psid, candidate, detail = entry
                        return {face_key(psid, candidate[r["output_face_index"]])
                                for r in detail["output_face_lineage"]}
                    if pair_keys(proposed[0]) != pair_keys(proposed[1]):
                        raise ValueError("paired flip does not retain identical shared faces")
            except ValueError as error:
                errors.append(str(error))
                continue
            for psid, candidate, detail in proposed:
                apply(psid, candidate, detail)
            operations.append({"paired_interlobar_flip": mirrored is not None,
                               "source_seed": [sid, seed], "flips": [x[2] for x in proposed]})
            succeeded = True
            break
        if not succeeded:
            blocked.append({"stable_id": sid, "face_index": seed,
                            "source_altitude_m": altitude, "rejections": errors})

    final_shared = {key: tuple(sorted(value)) for key, value in lookup.items() if len(value) > 1}
    # Shared triangulations may change, but reciprocal lobe ownership counts must not.
    def shared_counts(shared):
        return collections.Counter(tuple(sid for sid, fi in owners) for owners in shared.values())
    if shared_counts(initial_shared) != shared_counts(final_shared):
        raise ValueError("precision repair changed interlobar shared-face ownership counts")
    per_lobe = {}
    for sid in lobe_ids:
        f = rows[sid]["faces"]
        topology = validate_closed_oriented_surface(positions[sid], f)
        if topology != before_topology[sid]:
            raise ValueError(f"precision repair changed closed lobe topology {sid}")
        delta = signed_volume(positions[sid].astype(float), f) - before_volume[sid]
        if abs(delta) > 1e-12:
            raise ValueError(f"precision repair changed lobe {sid} volume by more than 1e-12 m3")
        altitude = _triangle_altitudes(positions[sid], f)
        remaining = np.flatnonzero(altitude < selection_altitude)
        vertices = original_rows[sid]["vertices6"].copy()
        vertices[:, 3:6] = normals(positions[sid].astype(float), f).astype(np.float32)
        rows[sid]["vertices6"] = vertices
        per_lobe[str(sid)] = {
            "source_risk_face_count": initial_counts[sid],
            "remaining_risk_faces": [{"face_index": int(i), "altitude_m": float(altitude[i])} for i in remaining],
            "minimum_altitude_m": float(altitude.min()), "signed_volume_delta_m3": delta,
            "topology_before_and_after": topology,
            "changed_face_ancestry": {str(i): sorted(parents) for i, parents in lineage[sid].items()},
        }
    return rows, {
        "selection_altitude_m": selection_altitude,
        "replacement_minimum_altitude_m": minimum_altitude_m,
        "selection_basis": (f"source altitude screen at {selection_altitude:.9g} m and replacement floor "
                            f"{minimum_altitude_m:.9g} m. This is a preparation selection, not proof of all transformed Float32 poses."),
        "operations": operations, "blocked_attempts": blocked, "lobes": per_lobe,
        "protected_non_lobe_reciprocal_owner_ids": [311] if 311 in original_rows else [],
        "shared_lobe_face_ownership_counts_preserved": True,
    }


def _collapse_short_precision_slivers(original_rows, *, altitude_limit_m=128e-9,
                                      max_edge_length_m=1e-6,
                                      max_endpoint_displacement_m=1e-6,
                                      max_abs_volume_delta_m3=1e-12,
                                      max_operations=256, protected_owner_ids=(), protected_vertex_coordinates=()):
    """Collapse only diagnosed short sliver edges, propagating exact shared owners.

    Row 310 is a derived pleura and is rebuilt after this operation. Rows 305-309
    and diaphragm row 311 are source owners: a collapse touching an exact shared
    face is applied to every exact owner as one transaction. A lone shared point
    is never moved; a local endpoint may be collapsed into that fixed point.
    """
    owner_ids = tuple(sid for sid in (*range(305, 310), 311) if sid in original_rows)
    lobe_ids = tuple(range(305, 310))
    if not all(sid in original_rows for sid in lobe_ids):
        raise ValueError("sliver collapse requires all five lung lobes")
    if (not np.isfinite(altitude_limit_m) or altitude_limit_m <= 0
            or not np.isfinite(max_edge_length_m) or max_edge_length_m <= 0
            or not np.isfinite(max_endpoint_displacement_m) or max_endpoint_displacement_m <= 0
            or not np.isfinite(max_abs_volume_delta_m3) or max_abs_volume_delta_m3 < 0
            or type(max_operations) is not int or max_operations < 0):
        raise ValueError("sliver collapse bounds must be finite and positive")

    from .resting_anatomy_interface_patch import signed_volume as _signed_volume
    rows = {sid: dict(row) for sid, row in original_rows.items()}
    face_ancestry = {sid: [[fi] for fi in range(len(rows[sid]["faces"]))] for sid in owner_ids}
    initial_volume = {sid: _signed_volume(rows[sid]["vertices6"][:, :3].astype(np.float64), rows[sid]["faces"])
                      for sid in owner_ids}
    initial_area = {}
    for sid in owner_ids:
        xyz = rows[sid]["vertices6"][:, :3].astype(np.float64)
        tri = xyz[rows[sid]["faces"]]
        initial_area[sid] = float(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1).sum() * .5)

    def point_key(point):
        return np.ascontiguousarray(point, dtype="<f4").tobytes()

    protected_point_keys = {point_key(point) for point in protected_vertex_coordinates}

    def oriented_key(xyz, face):
        return tuple(point_key(xyz[int(vi)]) for vi in face)

    def opposite(a, b):
        return any(tuple(a[(shift-i) % 3] for i in range(3)) == b for shift in range(3))

    def shared_map(state):
        grouped = collections.defaultdict(list)
        for sid in owner_ids:
            if sid not in state:
                continue
            xyz = state[sid]["vertices6"][:, :3]
            for fi, face in enumerate(state[sid]["faces"]):
                grouped[tuple(sorted(oriented_key(xyz, face)))].append((sid, fi, oriented_key(xyz, face)))
        return grouped

    def patch_summary(grouped, pair):
        selected = [(key, owners) for key, owners in grouped.items()
                    if len(owners) == 2 and tuple(sorted((owners[0][0], owners[1][0]))) == pair]
        points = sorted({point for key, _ in selected for point in key})
        point_index = {key: index for index, key in enumerate(points)}
        edge_counts = collections.Counter()
        parent = list(range(len(points)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        triangles = []
        area = 0.0
        for key, owners in selected:
            tri = tuple(point_index[p] for p in owners[0][2])
            triangles.append(tri)
            for i in range(3):
                edge_counts[tuple(sorted((tri[i], tri[(i+1) % 3])))] += 1
            xyz = np.asarray([np.frombuffer(p, dtype="<f4").astype(np.float64) for p in owners[0][2]])
            area += float(np.linalg.norm(np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])) * .5)
        for tri in triangles:
            a, b, c = map(find, tri)
            parent[b] = a
            parent[c] = a
        components = len({find(index) for index in {v for tri in triangles for v in tri}})
        edges = len(edge_counts)
        boundary = sum(value == 1 for value in edge_counts.values())
        nonmanifold = sum(value > 2 for value in edge_counts.values())
        euler = len(points) - edges + len(triangles)
        return {"face_count": len(selected), "vertex_count": len(points), "edge_count": edges,
                "components": components, "boundary_edge_count": boundary,
                "nonmanifold_edge_count": nonmanifold, "euler_characteristic": euler,
                "area_m2": area}

    def validate_common(state, before_map, affected):
        after_map = shared_map(state)
        before_pairs = {tuple(sorted((owners[0][0], owners[1][0])))
                        for owners in before_map.values() if len(owners) == 2}
        after_pairs = {tuple(sorted((owners[0][0], owners[1][0])))
                       for owners in after_map.values() if len(owners) == 2}
        if after_pairs != before_pairs:
            raise ValueError("collapse changed the set of exact shared-face owner pairs")
        for key, owners in after_map.items():
            if len(owners) > 2:
                raise ValueError("collapse created a multiply owned exact shared face")
            if len(owners) == 2 and not opposite(owners[0][2], owners[1][2]):
                raise ValueError("collapse broke opposite winding on an exact shared face")
        touched_pairs = {tuple(sorted((a, b))) for a in affected for b in affected if a < b}
        patch_changes = {}
        for pair in sorted(before_pairs & after_pairs):
            before_keys = {key for key, owners in before_map.items()
                           if len(owners) == 2 and tuple(sorted((owners[0][0], owners[1][0]))) == pair}
            after_keys = {key for key, owners in after_map.items()
                          if len(owners) == 2 and tuple(sorted((owners[0][0], owners[1][0]))) == pair}
            if pair not in touched_pairs:
                if before_keys != after_keys:
                    raise ValueError(f"collapse unexpectedly changed untouched shared patch {pair}")
                continue
            before = patch_summary(before_map, pair)
            after = patch_summary(after_map, pair)
            topo_keys = ("components", "boundary_edge_count", "nonmanifold_edge_count",
                         "euler_characteristic")
            if any(before[k] != after[k] for k in topo_keys):
                raise ValueError(f"collapse changed shared patch topology for owner pair {pair}")
            patch_changes[str(pair)] = {
                "before": before, "after": after,
                "area_delta_m2": after["area_m2"] - before["area_m2"],
            }
        return patch_changes

    operations, blocked = [], []
    attempted_states = set()
    for _ in range(max_operations):
        before_map = shared_map(rows)
        vertex_indices = {}
        vertex_owners = collections.defaultdict(set)
        for owner in owner_ids:
            by_key = collections.defaultdict(list)
            for vertex_index, point in enumerate(rows[owner]["vertices6"][:, :3]):
                key = point_key(point)
                by_key[key].append(vertex_index)
                vertex_owners[key].add(owner)
            vertex_indices[owner] = by_key
        seeds = []
        for sid in lobe_ids:
            xyz = rows[sid]["vertices6"][:, :3]
            faces = rows[sid]["faces"]
            altitudes = _triangle_altitudes(xyz, faces)
            for fi in np.flatnonzero(altitudes < altitude_limit_m):
                tri = faces[int(fi)]
                for i in range(3):
                    edge = tuple(sorted((int(tri[i]), int(tri[(i+1) % 3]))))
                    length = float(np.linalg.norm(xyz[edge[0]].astype(np.float64)-xyz[edge[1]].astype(np.float64)))
                    if length <= max_edge_length_m:
                        seeds.append((float(altitudes[fi]), length, sid, int(fi), edge))
        if not seeds:
            break
        seeds.sort()
        committed = False
        for seed_altitude, edge_length, sid, seed_face, edge in seeds:
            global_edge_key = (sid, tuple(sorted(point_key(rows[sid]["vertices6"][i, :3]) for i in edge)))
            faces = rows[sid]["faces"]
            incident = np.flatnonzero(np.sum(np.isin(faces, edge), axis=1) == 2)
            if len(incident) != 2:
                blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                "reason": "edge no longer has exactly two incident faces"})
                continue
            incident_signature = tuple(sorted(
                tuple(sorted(oriented_key(rows[sid]["vertices6"][:, :3], faces[int(fi)])))
                for fi in incident))
            attempt_key = (global_edge_key, incident_signature)
            if attempt_key in attempted_states:
                continue
            attempted_states.add(attempt_key)
            peer_ids = set()
            ambiguous = False
            for fi in incident:
                key = tuple(sorted(oriented_key(rows[sid]["vertices6"][:, :3], faces[int(fi)])))
                owners = before_map.get(key, [])
                if len(owners) > 2:
                    ambiguous = True
                    break
                peer_ids.update(owner[0] for owner in owners if owner[0] != sid)
            if ambiguous:
                blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                "reason": "multiply owned incident face"})
                continue
            protected = sorted(peer_ids.intersection(set(map(int, protected_owner_ids))))
            if protected:
                blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                "peer_owners": sorted(peer_ids),
                                "reason": f"collapse touches protected reciprocal owner(s) {protected}"})
                continue
            endpoint_keys = [point_key(rows[sid]["vertices6"][i, :3]) for i in edge]
            if any(key in protected_point_keys for key in endpoint_keys):
                blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                "reason": "collapse touches a protected source vertex"})
                continue
            endpoint_owners = [set() for _ in endpoint_keys]
            for endpoint_index, key in enumerate(endpoint_keys):
                endpoint_owners[endpoint_index] = set(vertex_owners.get(key, ())) - {sid}
            policy = "midpoint"
            remove_key, keep_key = endpoint_keys
            if peer_ids:
                affected = {sid, *peer_ids}
                if any(not owners.issubset(affected) for owners in endpoint_owners):
                    blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                    "reason": "shared endpoint has an unpropagated owner"})
                    continue
            elif endpoint_owners[0] or endpoint_owners[1]:
                if bool(endpoint_owners[0]) == bool(endpoint_owners[1]):
                    blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                    "reason": "both endpoints are shared without a reciprocal edge owner"})
                    continue
                policy = "keep"
                if endpoint_owners[0]:
                    keep_key, remove_key = endpoint_keys[0], endpoint_keys[1]
                else:
                    keep_key, remove_key = endpoint_keys[1], endpoint_keys[0]
                affected = {sid}
            else:
                affected = {sid}

            plans = []
            try:
                for owner in sorted(affected):
                    remove_hits = vertex_indices[owner].get(remove_key, ())
                    keep_hits = vertex_indices[owner].get(keep_key, ())
                    if len(remove_hits) != 1 or len(keep_hits) != 1:
                        raise ValueError(f"owner {owner} lacks unique collapse endpoints")
                    remove_vertex = remove_hits[0]
                    keep_vertex = keep_hits[0]
                    candidate = collapse_midpoint_edge(
                        rows[owner]["vertices6"], rows[owner]["faces"],
                        np.arange(len(rows[owner]["faces"]), dtype=np.int64),
                        remove_vertex=remove_vertex, keep_vertex=keep_vertex,
                        position_policy=policy,
                        max_endpoint_displacement_m=max_endpoint_displacement_m,
                        max_abs_volume_delta_m3=max_abs_volume_delta_m3,
                    )
                    vertices, candidate_faces, origins, detail = candidate
                    old_ancestry = face_ancestry[owner]
                    new_ancestry = [old_ancestry[int(origin)] for origin in origins]
                    plans.append((owner, vertices, candidate_faces, new_ancestry, detail))
                candidate_rows = {key: dict(value) for key, value in rows.items()}
                for owner, vertices, candidate_faces, new_ancestry, detail in plans:
                    candidate_rows[owner]["vertices6"] = vertices
                    candidate_rows[owner]["faces"] = candidate_faces
                patch_changes = validate_common(candidate_rows, before_map, affected)
                after_volumes = {owner: _signed_volume(candidate_rows[owner]["vertices6"][:, :3].astype(np.float64),
                                                       candidate_rows[owner]["faces"]) for owner in owner_ids}
                if any(abs(after_volumes[owner]-initial_volume[owner]) > max_abs_volume_delta_m3
                       for owner in owner_ids):
                    raise ValueError("cumulative per-owner signed-volume change exceeds bound")
                before_lung_total = sum(initial_volume[s] for s in lobe_ids)
                after_lung_total = sum(after_volumes[s] for s in lobe_ids)
                if abs(after_lung_total-before_lung_total) > max_abs_volume_delta_m3:
                    raise ValueError("cumulative total lung signed-volume change exceeds bound")
            except ValueError as error:
                attempted_states.add(attempt_key)
                blocked.append({"stable_id": sid, "face_index": seed_face, "edge": list(edge),
                                "seed_altitude_m": seed_altitude, "edge_length_m": edge_length,
                                "peer_owners": sorted(peer_ids), "reason": str(error)})
                continue

            changed_owners = []
            for owner, vertices, candidate_faces, new_ancestry, detail in plans:
                rows[owner]["vertices6"] = vertices
                rows[owner]["faces"] = candidate_faces
                face_ancestry[owner] = new_ancestry
                changed_owners.append({"stable_id": owner, **detail})
            operations.append({
                "seed_lobe_stable_id": sid, "seed_face_index_before_collapse": seed_face,
                "seed_altitude_m": seed_altitude, "edge_length_m": edge_length,
                "endpoint_position_policy": policy,
                "endpoint_coordinate_bits_before": [list(np.frombuffer(key, dtype="<f4").astype(float)) for key in endpoint_keys],
                "affected_owner_ids": sorted(affected), "owner_operations": changed_owners,
                "reciprocal_patch_changes": patch_changes,
                "all_shared_face_keys_opposite_wound_after": True,
                "full_changed_star_owner_propagation_checked": True,
            })
            committed = True
            break
        if not committed:
            break

    remaining = {}
    per_owner = {}
    for sid in owner_ids:
        xyz = rows[sid]["vertices6"][:, :3].astype(np.float64)
        faces = rows[sid]["faces"]
        tri = xyz[faces]
        twice_area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
        area = float(twice_area.sum() * .5)
        volume = _signed_volume(xyz, faces)
        if sid in lobe_ids:
            altitudes = _triangle_altitudes(rows[sid]["vertices6"][:, :3], faces)
            remaining[str(sid)] = [
                {"face_index": int(fi), "altitude_m": float(altitudes[fi])}
                for fi in np.flatnonzero(altitudes < altitude_limit_m)
            ]
        per_owner[str(sid)] = {
            "face_count_before": len(original_rows[sid]["faces"]),
            "face_count_after": len(faces),
            "vertex_count_before": len(original_rows[sid]["vertices6"]),
            "vertex_count_after": len(rows[sid]["vertices6"]),
            "signed_volume_before_m3": initial_volume[sid],
            "signed_volume_after_m3": volume,
            "signed_volume_delta_m3": volume-initial_volume[sid],
            "surface_area_before_m2": initial_area[sid],
            "surface_area_after_m2": area,
            "surface_area_delta_m2": area-initial_area[sid],
        }
    return rows, {
        "selection_altitude_m": altitude_limit_m,
        "maximum_edge_length_m": max_edge_length_m,
        "maximum_endpoint_displacement_m": max_endpoint_displacement_m,
        "maximum_abs_volume_delta_m3": max_abs_volume_delta_m3,
        "protected_reciprocal_owner_ids": sorted(map(int, protected_owner_ids)),
        "protected_source_vertex_coordinates_f32_m": [np.asarray(x, dtype="<f4").astype(float).tolist()
                                                      for x in protected_vertex_coordinates],
        "derived_pleura_row310_rebuilt_by_existing_owner": True,
        "operations": operations, "blocked_attempts": blocked,
        "remaining_subthreshold_faces": remaining, "per_owner": per_owner,
        "exact_shared_owner_maps_and_opposite_winding_validated": True,
        "native_transformed_pose_qualification": "not performed",
    }, face_ancestry


def _relocate_subthreshold_vertices(original_rows, *, selection_altitude_m=128e-9,
                                    altitude_floor_m=512e-9, requested_altitude_m=540e-9,
                                    max_vertex_displacement_m=1e-6,
                                    max_abs_volume_delta_m3=1e-12, protected_vertex_coordinates=(),
                                    max_operations=128):
    """Raise selected source slivers with bounded, owner-propagated tangent moves.

    This changes source coordinates only; connectivity and native acceptance
    tolerances remain untouched. Exact Float32 duplicate points move together.
    """
    from .surface_precision_retriangulation import validate_closed_oriented_surface

    lobes = tuple(range(305, 310))
    owners = tuple(sid for sid in (*lobes, 311) if sid in original_rows)
    if not all(sid in original_rows for sid in lobes):
        raise ValueError("sliver relocation requires all five lung lobes")
    bounds = (selection_altitude_m, altitude_floor_m, requested_altitude_m,
              max_vertex_displacement_m, max_abs_volume_delta_m3)
    if (not all(np.isfinite(x) for x in bounds) or selection_altitude_m <= 0
            or altitude_floor_m < selection_altitude_m or requested_altitude_m < altitude_floor_m
            or max_vertex_displacement_m <= 0 or max_abs_volume_delta_m3 < 0
            or type(max_operations) is not int or max_operations < 0):
        raise ValueError("sliver relocation bounds are invalid")
    rows = {sid: dict(row) for sid, row in original_rows.items()}
    source_xyz = {sid: rows[sid]["vertices6"][:, :3].copy() for sid in owners}
    source_faces = {sid: rows[sid]["faces"].copy() for sid in owners}

    def key(point):
        return np.ascontiguousarray(point, dtype="<f4").tobytes()

    protected_point_keys = {key(point) for point in protected_vertex_coordinates}

    def face_orientation(vertices6, face):
        return tuple(key(vertices6[int(i), :3]) for i in face)

    def paired_faces(state):
        groups = collections.defaultdict(list)
        for sid in owners:
            xyz, faces = state[sid]["vertices6"], state[sid]["faces"]
            for fi, face in enumerate(faces):
                orient = face_orientation(xyz, face)
                groups[tuple(sorted(orient))].append((sid, fi, orient))
        paired = {}
        for group in groups.values():
            if len(group) > 2:
                raise ValueError("multiply owned exact face during tangent relocation")
            if len(group) == 2:
                a, b = group
                if a[0] == b[0] or not any(
                        tuple(a[2][(shift-i) % 3] for i in range(3)) == b[2]
                        for shift in range(3)):
                    raise ValueError("invalid exact reciprocal face or winding")
                paired[tuple(sorted(((a[0], a[1]), (b[0], b[1]))))] = (a[2], b[2])
        return paired

    before_pairs = paired_faces(rows)
    initial_vol = {sid: signed_volume(rows[sid]["vertices6"][:, :3].astype(np.float64),
                                      rows[sid]["faces"]) for sid in owners}
    current_vol = dict(initial_vol)
    initial_area = {}
    for sid in owners:
        xyz = rows[sid]["vertices6"][:, :3].astype(np.float64)
        tri = xyz[rows[sid]["faces"]]
        initial_area[sid] = float(np.linalg.norm(
            np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1).sum() * 0.5)

    def point_index():
        out = {}
        for sid in owners:
            mapping = collections.defaultdict(list)
            for vi, point in enumerate(rows[sid]["vertices6"][:, :3]):
                mapping[key(point)].append(int(vi))
            out[sid] = mapping
        return out

    def altitude(points):
        tri = np.asarray(points, dtype=np.float64)
        longest = max(float(np.linalg.norm(tri[1]-tri[0])),
                      float(np.linalg.norm(tri[2]-tri[1])),
                      float(np.linalg.norm(tri[0]-tri[2])))
        return 0.0 if longest <= 0 else float(
            np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) / longest)

    def tangent_proposals(sid, fi):
        face = rows[sid]["faces"][fi]
        points = rows[sid]["vertices6"][face, :3].astype(np.float64)
        area_vector = np.cross(points[1]-points[0], points[2]-points[0])
        area_norm = float(np.linalg.norm(area_vector))
        if not np.isfinite(area_norm) or area_norm <= 0:
            return [], "seed face has no finite oriented area"
        face_normal = area_vector / area_norm
        choices = []
        angles = (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60, 90, -90)
        for slot in range(3):
            other = [i for i in range(3) if i != slot]
            start = points[other[0]]
            edge = points[other[1]] - start
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 0:
                continue
            edge_hat = edge / edge_length
            offset = points[slot] - start
            perpendicular = offset - float(np.dot(offset, edge_hat)) * edge_hat
            if float(np.linalg.norm(perpendicular)) <= 0:
                continue
            tangent = np.cross(face_normal, edge_hat)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 0:
                continue
            tangent /= tangent_norm
            if float(np.dot(offset, tangent)) < 0:
                tangent = -tangent
            old_point = rows[sid]["vertices6"][int(face[slot]), :3].astype(np.float64)
            if key(old_point) in protected_point_keys:
                continue
            for angle_degrees in angles:
                angle = np.deg2rad(float(angle_degrees))
                direction = np.cos(angle) * tangent + np.sin(angle) * face_normal
                direction /= np.linalg.norm(direction)
                lo, hi = 0.0, float(max_vertex_displacement_m)
                def candidate_at(distance):
                    proposed = (old_point + distance * direction).astype(np.float32)
                    triangle = points.copy()
                    triangle[slot] = proposed.astype(np.float64)
                    return proposed, altitude(triangle)
                proposed_hi, achieved_hi = candidate_at(hi)
                if achieved_hi < altitude_floor_m:
                    continue
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    _, achieved_mid = candidate_at(mid)
                    if achieved_mid >= requested_altitude_m:
                        hi = mid
                    else:
                        lo = mid
                proposed, achieved = candidate_at(hi)
                actual = float(np.linalg.norm(proposed.astype(np.float64) - old_point))
                if achieved >= altitude_floor_m and actual <= max_vertex_displacement_m:
                    choices.append((actual, slot, proposed, achieved, angle_degrees))
        return sorted(choices, key=lambda item: (item[0], abs(item[4]), item[4], item[1])), ""

    operations, blocked, attempted = [], [], set()
    for _ in range(max_operations):
        index = point_index()
        seeds = []
        for sid in lobes:
            heights = _triangle_altitudes(rows[sid]["vertices6"][:, :3], rows[sid]["faces"])
            seeds.extend((float(heights[fi]), sid, int(fi))
                         for fi in np.flatnonzero(heights < selection_altitude_m))
        seeds.sort()
        if not seeds:
            break
        accepted = False
        for _, sid, fi in seeds:
            face = rows[sid]["faces"][fi]
            before_alt = altitude(rows[sid]["vertices6"][face, :3])
            if before_alt >= altitude_floor_m:
                continue
            signature = (sid, fi, tuple(key(rows[sid]["vertices6"][int(v), :3]) for v in face))
            if signature in attempted:
                continue
            attempted.add(signature)
            choices, reason = tangent_proposals(sid, fi)
            if not choices:
                blocked.append({"stable_id": sid, "face_index": fi, "altitude_m": before_alt,
                                "reason": reason or "no bounded tangent proposal reaches the Float32 floor"})
                continue
            failures = []
            for movement, slot, proposed, achieved, angle_degrees in choices:
                old = rows[sid]["vertices6"][int(face[slot]), :3].copy()
                old_key = key(old)
                indices = {owner: index[owner].get(old_key, []) for owner in owners}
                affected = {owner for owner, ids in indices.items() if ids}
                if sid not in affected:
                    failures.append("selected point missing from its owner")
                    continue
                displacement = {}
                for owner, ids in indices.items():
                    if ids:
                        delta = np.linalg.norm(
                            proposed.astype(np.float64)-source_xyz[owner][ids].astype(np.float64), axis=1)
                        displacement[owner] = float(delta.max(initial=0.0))
                if any(value > max_vertex_displacement_m for value in displacement.values()):
                    failures.append("cumulative displacement exceeds bound")
                    continue
                new_key = key(proposed)
                if any(vi not in indices[owner] for owner in affected
                       for vi in index[owner].get(new_key, ())):
                    failures.append("proposed point aliases an unrelated vertex")
                    continue
                trial = {}
                for owner in affected:
                    trial[owner] = rows[owner]["vertices6"].copy()
                    trial[owner][indices[owner], :3] = proposed
                valid = True
                for owner in affected:
                    faces = rows[owner]["faces"]
                    incident = np.flatnonzero(np.any(np.isin(faces, indices[owner]), axis=1))
                    before = rows[owner]["vertices6"][faces[incident], :3].astype(np.float64)
                    after = trial[owner][faces[incident], :3].astype(np.float64)
                    old_normals = np.cross(before[:, 1]-before[:, 0], before[:, 2]-before[:, 0])
                    new_normals = np.cross(after[:, 1]-after[:, 0], after[:, 2]-after[:, 0])
                    if (not np.isfinite(new_normals).all()
                            or np.any(np.einsum("ij,ij->i", old_normals, new_normals) <= 0)
                            or np.any(np.linalg.norm(new_normals, axis=1) <= 0)):
                        valid = False
                        failures.append(f"owner {owner} changed star reverses or degenerates")
                        break
                if not valid:
                    continue
                volume = dict(current_vol)
                for owner in affected:
                    volume[owner] = signed_volume(trial[owner][:, :3].astype(np.float64),
                                                  rows[owner]["faces"])
                if any(abs(volume[o]-initial_vol[o]) > max_abs_volume_delta_m3 for o in owners):
                    failures.append("per-owner cumulative volume exceeds bound")
                    continue
                if abs(sum(volume[o]-initial_vol[o] for o in lobes)) > max_abs_volume_delta_m3:
                    failures.append("cumulative lung volume exceeds bound")
                    continue
                for owner in affected:
                    rows[owner]["vertices6"] = trial[owner]
                    current_vol[owner] = volume[owner]
                operations.append({
                    "stable_id": sid, "seed_face_index": fi,
                    "seed_altitude_before_m": before_alt,
                    "seed_altitude_after_m": altitude(rows[sid]["vertices6"][face, :3]),
                    "moved_vertex_slot": slot, "affected_owner_ids": sorted(affected),
                    "source_coordinate_f32_m": old.astype(float).tolist(),
                    "candidate_coordinate_f32_m": proposed.astype(float).tolist(),
                    "max_displacement_from_relocation_input_by_owner_m": {
                        str(o): displacement[o] for o in sorted(displacement)},
                    "changed_star_orientation_checked": True,
                    "exact_coordinate_owner_propagation_checked": True,
                    "per_owner_cumulative_volume_delta_m3": {
                        str(o): volume[o]-initial_vol[o] for o in owners},
                })
                accepted = True
                break
            if not accepted:
                blocked.append({"stable_id": sid, "face_index": fi, "altitude_m": before_alt,
                                "reason": "; ".join(dict.fromkeys(failures)) or "all tangent proposals rejected"})
            else:
                break
        if not accepted:
            break

    changed_owners = {
        sid for sid in owners
        if not np.array_equal(rows[sid]["vertices6"][:, :3], source_xyz[sid])
    }
    for sid in changed_owners:
        rows[sid]["vertices6"][:, 3:6] = normals(
            rows[sid]["vertices6"][:, :3].astype(np.float64), rows[sid]["faces"]
        ).astype(np.float32)
    after_pairs = paired_faces(rows)
    if set(after_pairs) != set(before_pairs):
        raise ValueError("tangent relocation changed reciprocal face owner/face-index pairs")
    for identity, (left, right) in before_pairs.items():
        new_left, new_right = after_pairs[identity]
        if not any(tuple(new_left[(shift-i) % 3] for i in range(3)) == new_right
                   for shift in range(3)):
            raise ValueError("tangent relocation broke reciprocal face winding")

    per_owner, remaining = {}, {}
    for sid in owners:
        if not np.array_equal(rows[sid]["faces"], source_faces[sid]):
            raise ValueError("tangent relocation changed face connectivity")
        topology = validate_closed_oriented_surface(rows[sid]["vertices6"][:, :3], rows[sid]["faces"])
        if not topology["closed_oriented"]:
            raise ValueError(f"tangent relocation invalidated owner {sid} topology")
        xyz = rows[sid]["vertices6"][:, :3].astype(np.float64)
        triangles = xyz[rows[sid]["faces"]]
        area = float(np.linalg.norm(
            np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1).sum() * .5)
        per_owner[str(sid)] = {
            "signed_volume_before_m3": initial_vol[sid],
            "signed_volume_after_m3": current_vol[sid],
            "signed_volume_delta_m3": current_vol[sid]-initial_vol[sid],
            "surface_area_before_m2": initial_area[sid],
            "surface_area_after_m2": area,
            "surface_area_delta_m2": area-initial_area[sid],
            "face_connectivity_unchanged": True, "topology_after": topology,
        }
        if sid in lobes:
            heights = _triangle_altitudes(rows[sid]["vertices6"][:, :3], rows[sid]["faces"])
            remaining[str(sid)] = [
                {"face_index": int(fi), "altitude_m": float(heights[fi])}
                for fi in np.flatnonzero(heights < altitude_floor_m)
            ]
    return rows, {
        "algorithm": "exact-owner-propagated-bounded-tangent-source-vertex-relocation-v1",
        "selection_altitude_m": float(selection_altitude_m),
        "altitude_floor_m": float(altitude_floor_m),
        "requested_replacement_altitude_m": float(requested_altitude_m),
        "maximum_vertex_displacement_m": float(max_vertex_displacement_m),
        "maximum_abs_volume_delta_m3": float(max_abs_volume_delta_m3),
        "protected_source_vertex_coordinates_f32_m": [np.asarray(x, dtype="<f4").astype(float).tolist()
                                                      for x in protected_vertex_coordinates],
        "operations": operations, "blocked_attempts": blocked,
        "remaining_subfloor_faces": remaining, "per_owner": per_owner,
        "exact_reciprocal_face_owner_indices_preserved": True,
        "native_transformed_pose_qualification": "not performed",
    }




def _geometry_bytes(row):
    return (np.ascontiguousarray(row["vertices6"], dtype="<f4").tobytes()
            + np.ascontiguousarray(row["faces"], dtype="<i8").tobytes())


def _pleura_face_lineage(rows):
    """Map each derived face to one current lobe face by exact position bits."""
    def key(points):
        bits = np.ascontiguousarray(points, dtype="<f4").view("<u4").reshape(-1, 3)
        return tuple(sorted(tuple(map(int, point)) for point in bits))

    lookup = collections.defaultdict(list)
    for sid in range(305, 310):
        xyz = rows[sid]["vertices6"][:, :3]
        for index, face in enumerate(rows[sid]["faces"]):
            lookup[key(xyz[face])].append((sid, index))
    result = []
    xyz = rows[310]["vertices6"][:, :3]
    for index, face in enumerate(rows[310]["faces"]):
        matches = lookup.get(key(xyz[face]), [])
        if len(matches) != 1:
            raise ValueError(f"derived pleura face {index} lacks unique lobe lineage")
        result.append(matches[0])
    return np.asarray(result, dtype="<i4")



def _reconstruct_diaphragm_interface_registration(receipt, rows):
    """Rebuild stale patch ranges from unique exact source-coordinate face pairs.

    This operation only reorders row 311 faces so each existing exact reciprocal
    lobe patch is contiguous for the current receipt schema. It does not move
    vertices or change the set or winding of any triangle.
    """
    interface = receipt["provenance"]["diaphragm_lung_interface"]
    old_rows = json.loads(json.dumps(interface.get("interface_rows", [])))
    old_summary = {
        key: interface.get(key) for key in (
            "added_reversed_interface_face_count",
            "added_reversed_interface_surface_area_m2",
            "reciprocal_triangle_count_after_joint_repair",
        )
    }
    d_vertices = rows[311]["vertices6"][:, :3]
    d_faces = rows[311]["faces"]
    old_face_count = len(d_faces)

    def point_key(point):
        return np.ascontiguousarray(point, dtype="<f4").tobytes()

    def oriented(xyz, face):
        return tuple(point_key(xyz[int(index)]) for index in face)

    def face_key(oriented_face):
        return tuple(sorted(oriented_face))

    def opposite(a, b):
        return any(tuple(a[(shift-i) % 3] for i in range(3)) == b
                   for shift in range(3))

    diaphragm_by_key = collections.defaultdict(list)
    for face_index, face in enumerate(d_faces):
        orientation = oriented(d_vertices, face)
        diaphragm_by_key[face_key(orientation)].append((face_index, orientation))

    lobe_by_key = {}
    lobe_face_orientations = {}
    for stable_id in range(305, 310):
        vertices = rows[stable_id]["vertices6"][:, :3]
        faces = rows[stable_id]["faces"]
        mapping = collections.defaultdict(list)
        orientations = {}
        for face_index, face in enumerate(faces):
            orientation = oriented(vertices, face)
            key = face_key(orientation)
            mapping[key].append((face_index, orientation))
            orientations[face_index] = orientation
        lobe_by_key[stable_id] = mapping
        lobe_face_orientations[stable_id] = orientations

    owner_by_diaphragm_face = {}
    groups = {}
    patch_metrics = {}
    for stable_id in range(305, 310):
        selected = []
        for key in diaphragm_by_key.keys() & lobe_by_key[stable_id].keys():
            d_entries = diaphragm_by_key[key]
            l_entries = lobe_by_key[stable_id][key]
            if len(d_entries) != 1 or len(l_entries) != 1:
                raise ValueError(
                    f"exact diaphragm/lobe face key is ambiguous for stable ID {stable_id}")
            d_index, d_orientation = d_entries[0]
            l_index, l_orientation = l_entries[0]
            if not opposite(d_orientation, l_orientation):
                raise ValueError(
                    f"exact diaphragm/lobe face key has same winding for stable ID {stable_id}")
            previous = owner_by_diaphragm_face.setdefault(d_index, stable_id)
            if previous != stable_id:
                raise ValueError(
                    f"one diaphragm face is exactly shared by multiple lobe owners: {previous}, {stable_id}")
            selected.append((d_index, l_index, l_orientation))
        if selected:
            groups[stable_id] = sorted(selected)
        elif stable_id in (305, 306, 307, 308):
            raise ValueError(f"no exact reciprocal diaphragm faces found for source lobe {stable_id}")

    def compress(indices):
        ranges = []
        for index in sorted(set(map(int, indices))):
            if not ranges or index != ranges[-1][1]:
                ranges.append([index, index+1])
            else:
                ranges[-1][1] = index+1
        return ranges

    def patch_topology(selected):
        point_ids = {}
        triangles = []
        area = 0.0
        for _, _, orientation in selected:
            tri = tuple(orientation)
            triangles.append(tri)
            for point in tri:
                point_ids.setdefault(point, len(point_ids))
            xyz = np.asarray([np.frombuffer(point, dtype="<f4").astype(np.float64)
                              for point in tri])
            area += float(np.linalg.norm(np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])) * 0.5)
        edge_counts = collections.Counter()
        parent = list(range(len(point_ids)))
        def find(value):
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]
        for tri in triangles:
            ids = [point_ids[point] for point in tri]
            for index in range(3):
                a, b = ids[index], ids[(index+1) % 3]
                edge_counts[tuple(sorted((a, b)))] += 1
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        vertices = len(point_ids)
        edges = len(edge_counts)
        boundary_keys = [edge for edge, count in edge_counts.items() if count == 1]
        boundary_edges = len(boundary_keys)
        boundary_vertices = len({vertex for edge in boundary_keys for vertex in edge})
        nonmanifold_edges = sum(value > 2 for value in edge_counts.values())
        components = len({find(index) for index in range(vertices)})
        euler = vertices - edges + len(triangles)
        return {
            "face_count": len(triangles), "vertex_count": vertices,
            "edge_count": edges, "connected_components": components,
            "boundary_edge_count": boundary_edges,
            "boundary_vertex_count": boundary_vertices,
            "nonmanifold_edge_count": nonmanifold_edges,
            "euler_characteristic": euler, "area_m2": area,
        }

    new_rows = []
    old_rows_by_owner = {int(row["lung_stable_id"]): row for row in old_rows}
    offset = 0
    for stable_id in sorted(groups):
        selected = groups[stable_id]
        lobe_face_ids = [entry[1] for entry in selected]
        d_face_ids = [entry[0] for entry in selected]
        metrics = patch_topology(selected)
        if (metrics["connected_components"] != 1
                or metrics["nonmanifold_edge_count"] != 0
                or metrics["boundary_edge_count"] == 0):
            raise ValueError(f"reconstructed reciprocal patch topology is invalid for lobe {stable_id}")
        row = dict(old_rows_by_owner.get(stable_id, {}))
        row.update({
            "diaphragm_stable_id": 311,
            "lung_stable_id": stable_id,
            "diaphragm_patch_face_start": offset,
            "diaphragm_patch_face_count": len(d_face_ids),
            "registered_lung_face_index_ranges": compress(lobe_face_ids),
            "registered_lung_face_count": len(lobe_face_ids),
            "patch_area_m2": metrics["area_m2"],
            "shared_boundary_edge_count": metrics["boundary_edge_count"],
            "shared_boundary_vertex_count": metrics["boundary_vertex_count"],
            "shared_reciprocal_child_face_count": len(d_face_ids),
            "orientation_relation": "reversed triangle winding across exact coincident material-interface faces",
            "volume_owner_stable_id": stable_id,
            "source_geometry_reconstructed_patch_topology": metrics,
        })
        new_rows.append(row)
        offset += len(d_face_ids)
        patch_metrics[str(stable_id)] = metrics

    assigned = set(owner_by_diaphragm_face)
    if len(assigned) != sum(len(group) for group in groups.values()):
        raise ValueError("reconstructed diaphragm face ownership is not one-to-one")
    remaining_faces = [index for index in range(old_face_count) if index not in assigned]
    new_to_old = np.asarray(
        [entry[0] for stable_id in sorted(groups) for entry in groups[stable_id]]
        + remaining_faces, dtype="<i8")
    if len(new_to_old) != old_face_count or set(map(int, new_to_old)) != set(range(old_face_count)):
        raise ValueError("diaphragm face reindexing is not a full permutation")

    before_topology = topology_report(rows[311]["faces"])
    rows[311]["faces"] = rows[311]["faces"][new_to_old].copy()
    after_topology = topology_report(rows[311]["faces"])
    topology_keys = (
        "edge_count", "boundary_edge_count", "boundary_loop_count",
        "boundary_branch_vertex_count", "nonmanifold_edge_count",
        "orientation_error_edge_count",
    )
    if any(before_topology.get(key) != after_topology.get(key) for key in topology_keys):
        raise ValueError("diaphragm face reordering changed topology")

    old_validation = {"status": "failed_stale_face_ranges"}
    old_validation["declared_patches"] = []
    for row in old_rows:
        stable_id = int(row["lung_stable_id"])
        start = int(row["diaphragm_patch_face_start"])
        count = int(row["diaphragm_patch_face_count"])
        d_match_count = 0
        if 0 <= start and 0 < count <= old_face_count and start + count <= old_face_count:
            for face in d_faces[start:start+count]:
                key = face_key(oriented(d_vertices, face))
                if key in lobe_by_key.get(stable_id, {}):
                    d_match_count += 1
        old_validation["declared_patches"].append({
            "lung_stable_id": stable_id,
            "old_diaphragm_range": [start, start+count],
            "old_diaphragm_faces_with_exact_lobe_counterparts": d_match_count,
            "old_diaphragm_face_count": count,
        })
    interface["historical_stale_registration"] = {
        "status": "failed_exact_source_geometry_validation",
        "payload_sha256": receipt.get("payload", {}).get("sha256"),
        "original_interface_rows": old_rows,
        "original_summary_counts": old_summary,
        "validation": old_validation,
    }
    interface["interface_rows"] = new_rows
    total_faces = sum(int(row["diaphragm_patch_face_count"]) for row in new_rows)
    total_area = sum(float(row["patch_area_m2"]) for row in new_rows)
    interface["added_reversed_interface_face_count"] = total_faces
    interface["reciprocal_triangle_count_after_joint_repair"] = total_faces
    interface["added_reversed_interface_surface_area_m2"] = total_area
    interface["qualification"] = (
        "exact source-coordinate reciprocal patch registration reconstructed after rejecting stale inherited ranges; "
        "native transformed-pose area, triangle, intersection, and functional-volume audits remain separate")
    receipt.setdefault("qualification", {})["diaphragm_lung_interface"] = (
        "source-pair ranges were rebuilt from unique exact Float32 coordinate keys with opposite winding; "
        "native full-cycle qualification remains pending")
    interface["source_geometry_registration_reconstruction"] = {
        "status": "reconstructed_from_exact_source_float32_geometry",
        "method": "unique exact coordinate-key face pairs with one diaphragm face, one lobe face, and opposed winding",
        "source_geometry_changed_by_registration_reconstruction": False,
        "row311_face_triangles_and_coordinates_changed": False,
        "row311_face_order_reindexed_for_existing_contiguous_range_fields": True,
        "patches": patch_metrics,
        "row311_topology_before_and_after_face_reindex": {
            key: before_topology.get(key) for key in topology_keys
        },
        "row311_source_face_order_mapping_required": True,
    }
    return new_to_old, {
        "old_registration_status": "failed_exact_source_geometry_validation",
        "reconstructed_patch_faces_by_lobe": {
            str(sid): int(len(groups[sid])) for sid in sorted(groups)},
        "reconstructed_patch_topology": patch_metrics,
        "row311_face_count": old_face_count,
        "row311_face_order_reindexed": True,
        "row311_face_triangles_and_coordinates_unchanged": True,
        "row311_topology_before_and_after_face_reindex": {
            key: before_topology.get(key) for key in topology_keys
        },
        "historical_registration_validation": old_validation,
    }


def _refresh_diaphragm_interface_registration(receipt, rows):
    """Rebind exact registered lung face ranges after index-compacting collapses."""
    interface = receipt["provenance"]["diaphragm_lung_interface"]
    diaphragm = rows[311]
    d_xyz, d_faces = diaphragm["vertices6"][:, :3], diaphragm["faces"]

    def point_key(point):
        return np.ascontiguousarray(point, dtype="<f4").tobytes()

    def oriented(xyz, face):
        return tuple(point_key(xyz[int(v)]) for v in face)

    def opposite(a, b):
        return any(tuple(a[(shift-i) % 3] for i in range(3)) == b for shift in range(3))

    total_count, total_area = 0, 0.0
    patch_check = {}
    for row in interface["interface_rows"]:
        sid = int(row["lung_stable_id"])
        start, count = int(row["diaphragm_patch_face_start"]), int(row["diaphragm_patch_face_count"])
        if start < 0 or count <= 0 or start + count > len(d_faces):
            raise ValueError(f"registered diaphragm face range is invalid for lobe {sid}")
        lung = rows[sid]
        l_xyz, l_faces = lung["vertices6"][:, :3], lung["faces"]
        lookup = collections.defaultdict(list)
        for fi, face in enumerate(l_faces):
            key = tuple(sorted(oriented(l_xyz, face)))
            lookup[key].append((fi, oriented(l_xyz, face)))
        matched, areas = [], []
        for dface in d_faces[start:start+count]:
            d_oriented = oriented(d_xyz, dface)
            key = tuple(sorted(d_oriented))
            candidates = lookup.get(key, ())
            if len(candidates) != 1:
                raise ValueError(f"registered diaphragm face for lobe {sid} has {len(candidates)} exact lung matches")
            fi, l_oriented = candidates[0]
            if not opposite(d_oriented, l_oriented):
                raise ValueError(f"registered diaphragm face for lobe {sid} lost opposite winding")
            matched.append(int(fi))
            tri = l_xyz[l_faces[fi]].astype(np.float64)
            areas.append(float(np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) * 0.5))
        if len(set(matched)) != count:
            raise ValueError(f"registered diaphragm face range for lobe {sid} is not one-to-one")
        ranges = []
        for fi in sorted(matched):
            if not ranges or fi != ranges[-1][1]:
                ranges.append([fi, fi+1])
            else:
                ranges[-1][1] = fi+1
        area = float(sum(areas))
        row["registered_lung_face_index_ranges"] = ranges
        row["registered_lung_face_count"] = count
        row["patch_area_m2"] = area
        total_count += count
        total_area += area
        patch_check[str(sid)] = {
            "matched_face_count": count, "opposite_winding_face_count": count,
        }
    interface["added_reversed_interface_face_count"] = total_count
    interface["reciprocal_triangle_count_after_joint_repair"] = total_count
    interface["added_reversed_interface_surface_area_m2"] = total_area
    interface.setdefault("conforming_refinement", {})["reciprocal_patch_check"] = patch_check
    return {
        "registered_reciprocal_face_count": total_count,
        "registered_reciprocal_area_m2": total_area,
        "row311_face_indices_unchanged": True,
        "exact_coordinate_and_opposite_winding_checked": True,
    }


def _compose_source_face_ancestry(original_rows, flip_reports, collapse_ancestry,
                                   fan_ancestry, final_rows, initial_face_order_by_owner=None):
    """Compose final face parents back to input owner rows."""
    result = {}
    initial_face_order_by_owner = initial_face_order_by_owner or {}
    first_report, second_report = flip_reports
    for sid in (*range(305, 310), 311):
        if sid not in original_rows:
            continue
        source_count = len(original_rows[sid]["faces"])
        first_changed = first_report.get("lobes", {}).get(str(sid), {}).get("changed_face_ancestry", {})
        second_changed = second_report.get("lobes", {}).get(str(sid), {}).get("changed_face_ancestry", {})
        initial_order = initial_face_order_by_owner.get(sid)
        if initial_order is None:
            first_output_to_source = {fi: [fi] for fi in range(source_count)}
        else:
            if len(initial_order) != source_count or set(map(int, initial_order)) != set(range(source_count)):
                raise ValueError(f"initial face order for owner {sid} is not a source permutation")
            first_output_to_source = {int(new): [int(old)]
                                      for new, old in enumerate(initial_order)}
        for output_face, source_faces in first_changed.items():
            first_output_to_source[int(output_face)] = sorted(set(map(int, source_faces)))
        second_output_to_first = {fi: [fi] for fi in range(source_count)}
        for output_face, input_faces in second_changed.items():
            second_output_to_first[int(output_face)] = sorted(set(map(int, input_faces)))

        collapse_output_to_source = []
        for pre_collapse_faces in collapse_ancestry[sid]:
            source_faces = set()
            for post_first_face in pre_collapse_faces:
                for after_first_face in second_output_to_first.get(
                        int(post_first_face), [int(post_first_face)]):
                    source_faces.update(first_output_to_source.get(
                        int(after_first_face), [int(after_first_face)]))
            collapse_output_to_source.append(sorted(source_faces))

        parent_to_children = fan_ancestry.get(
            sid, {fi: [fi] for fi in range(len(collapse_output_to_source))})
        final_to_prefan = {}
        for prefan_face, children in parent_to_children.items():
            for child in children:
                final_to_prefan.setdefault(int(child), []).append(int(prefan_face))
        source_to_final = [[] for _ in range(source_count)]
        for final_face in range(len(final_rows[sid]["faces"])):
            prefan_faces = final_to_prefan.get(final_face, [final_face])
            source_faces = set()
            for prefan_face in prefan_faces:
                if prefan_face < 0 or prefan_face >= len(collapse_output_to_source):
                    raise ValueError(f"face ancestry escapes intermediate owner {sid}")
                source_faces.update(collapse_output_to_source[prefan_face])
            for source_face in source_faces:
                if source_face < 0 or source_face >= source_count:
                    raise ValueError(f"face ancestry escapes input owner {sid}")
                source_to_final[source_face].append(final_face)
        mapped = {child for children in source_to_final for child in children}
        if mapped != set(range(len(final_rows[sid]["faces"]))):
            raise ValueError(f"face ancestry does not cover every final face on owner {sid}")
        result[sid] = [sorted(set(children)) for children in source_to_final]
    return result



def build_precision_candidate(base_payload: Path, base_receipt_path: Path,
                              respiratory_owner_path: Path, output_dir: Path):
    """Build a source-conditioned lung/pleura candidate through existing owners."""
    from .resting_anatomy import _copy_checked_sidecar
    from .surface_precision_retriangulation import validate_closed_oriented_surface

    base_payload, base_receipt_path = Path(base_payload), Path(base_receipt_path)
    respiratory_owner_path = Path(respiratory_owner_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace candidate output: {output_dir}")
    output_dir.mkdir(parents=True)
    module_paths = {
        "resting_lung_edge_repair.py": Path(__file__).resolve(),
        "resting_lobe_fan_refinement.py": Path(__file__).with_name("resting_lobe_fan_refinement.py").resolve(),
        "resting_anatomy_conforming_refinement.py": Path(__file__).with_name("resting_anatomy_conforming_refinement.py").resolve(),
    }
    module_hashes_at_launch = {name: {"path": str(path), "sha256": sha256(path)}
                               for name, path in module_paths.items()}
    source_pin_path = output_dir / "implementation-module-pins.json"
    source_pins = {"status": "started", "modules_at_launch": module_hashes_at_launch}
    source_pin_path.write_text(json.dumps(source_pins, indent=2, sort_keys=True)+"\n")
    if sha256(base_payload) != PRECISION_PAYLOAD_SHA256:
        raise ValueError("precision repair requires the pinned anatomy payload")
    if sha256(base_receipt_path) != PRECISION_RECEIPT_SHA256:
        raise ValueError("precision repair requires the pinned anatomy receipt")
    original_receipt = json.loads(base_receipt_path.read_text())
    if (original_receipt.get("schema") != "numi.human.resting-anatomy-receipt.v1"
            or original_receipt.get("payload", {}).get("sha256") != PRECISION_PAYLOAD_SHA256
            or original_receipt.get("functional_bindings", {}).get("anatomy_payload_sha256")
            != PRECISION_PAYLOAD_SHA256):
        raise ValueError("precision receipt does not bind the pinned source anatomy")
    respiratory_owner = _load_pinned_respiratory_owner(respiratory_owner_path)
    raw = base_payload.read_bytes()
    header, original_rows = parse_payload(base_payload)
    record_order = [int(RECORD.unpack_from(raw, HEADER.size+i*RECORD.size)[5])
                    for i in range(int(header[2]))]
    if any(int(original_rows[sid]["body_index"]) != 20 for sid in range(305, 311)):
        raise ValueError("lung anatomy ownership differs from the pinned source")

    receipt = json.loads(json.dumps(original_receipt))
    conditioning_rows = {sid: dict(row) for sid, row in original_rows.items()}
    row311_face_order, interface_reconstruction = _reconstruct_diaphragm_interface_registration(
        receipt, conditioning_rows)
    diaphragm_order_path = output_dir / "row-311-source-face-order.npy"
    interface = receipt["provenance"]["diaphragm_lung_interface"]

    # Run the retained position-preserving flip owner at the measured risk
    # screens, then only short, topology-checked collapses. Row311 remains the
    # reciprocal diaphragm owner; source-point relocation propagates into it
    # without changing its face inventory or registered face segment.
    rows1, flip_report1 = _condition_precision_lobes(
        conditioning_rows, minimum_altitude_m=1e-6, selection_altitude_m=1e-6)
    rows2, flip_report2 = _condition_precision_lobes(
        rows1, minimum_altitude_m=512e-9, selection_altitude_m=1e-6)
    collapse_rows, collapse_report, collapse_ancestry = _collapse_short_precision_slivers(
        rows2, altitude_limit_m=128e-9, max_edge_length_m=2e-6,
        max_endpoint_displacement_m=1e-6, max_abs_volume_delta_m3=1e-12,
        max_operations=256, protected_owner_ids=(311,),
        protected_vertex_coordinates=(FAN_REFINEMENT_SHARED_VERTEX_M,))
    rows3, relocation_report = _relocate_subthreshold_vertices(
        collapse_rows, selection_altitude_m=128e-9, altitude_floor_m=512e-9,
        requested_altitude_m=540e-9, max_vertex_displacement_m=1e-6,
        max_abs_volume_delta_m3=1e-12,
        protected_vertex_coordinates=(FAN_REFINEMENT_SHARED_VERTEX_M,), max_operations=128)

    interface_refresh_before_fan = _refresh_diaphragm_interface_registration(receipt, rows3)
    fan_lobes = {sid: (rows3[sid]["vertices6"], rows3[sid]["faces"])
                 for sid in range(305, 310)}
    expected_vertex = np.asarray(FAN_REFINEMENT_SHARED_VERTEX_M, dtype="<f4")
    dynamic_seed_faces = []
    for sid in (305, 308):
        vertices6, faces = fan_lobes[sid]
        matches = [
            int(fi) for fi, face in enumerate(faces)
            if int(np.count_nonzero(np.all(
                vertices6[face, :3] == expected_vertex.reshape(1, 3), axis=1))) == 1
        ]
        if not matches:
            raise ValueError(f"fan refinement source point is absent from lobe {sid}")
        dynamic_seed_faces.append((sid, matches[0]))
    refined_lobes, fan_ancestry, fan_report = refine_shared_vertex_fans(
        fan_lobes, seed_faces=tuple(dynamic_seed_faces),
        expected_shared_vertex_m=FAN_REFINEMENT_SHARED_VERTEX_M)
    diaphragm_rows = receipt["provenance"]["diaphragm_lung_interface"]["interface_rows"]
    for sid_text, detail in fan_report["per_lobe"].items():
        sid = int(sid_text)
        patch_faces = set()
        for interface_row in diaphragm_rows:
            if int(interface_row["lung_stable_id"]) == sid:
                for start, end in interface_row["registered_lung_face_index_ranges"]:
                    patch_faces.update(range(int(start), int(end)))
        overlap = patch_faces.intersection(detail["affected_source_faces"])
        if overlap:
            raise ValueError(f"fan star for lobe {sid} touches an unpropagated diaphragm patch: {sorted(overlap)}")
    for sid, (vertices6, faces) in refined_lobes.items():
        rows3[sid]["vertices6"], rows3[sid]["faces"] = vertices6, faces
    interface_refresh_after_fan = _refresh_diaphragm_interface_registration(receipt, rows3)
    final_source_precision_screen = _source_precision_screen(rows3)

    source_face_ancestry = _compose_source_face_ancestry(
        original_rows, (flip_report1, flip_report2), collapse_ancestry,
        fan_ancestry, rows3, initial_face_order_by_owner={311: row311_face_order})
    affected_source_faces = {}
    for sid_text, detail in fan_report["per_lobe"].items():
        sid = int(sid_text)
        affected = set(detail["affected_source_faces"])
        affected_source_faces[str(sid)] = {
            str(source_face): source_face_ancestry[sid][source_face]
            for source_face in range(len(source_face_ancestry[sid]))
            if any(child in affected for child in source_face_ancestry[sid][source_face])
        }
    fan_report["source_face_ancestry_from_input"] = affected_source_faces
    fan_report["source_pair_audit"] = {
        "report_sha256": FAN_REFINEMENT_AUDIT_SHA256,
        "accepted_pose_step": 4991,
        "row310_parent_faces": [40219, 220228],
        "source_lobe_parent_faces": [[305, 40225], [308, 46153]],
        "exact_intersection_dimension": "segment",
        "exact_intersection_diameter_m": 0.0013237775581397117,
        "qualification": "retained accepted-pose witness; new native exact-pose audit is required after this source change",
    }

    before_area, _ = _derive_basal_effective_area(
        original_rows, respiratory_owner.kuhn_basis)
    after_area, area_rows = _derive_basal_effective_area(
        {sid: rows3[sid] for sid in range(305, 310)}, respiratory_owner.kuhn_basis)
    before_volume_rows = {
        sid: signed_volume(original_rows[sid]["vertices6"][:, :3].astype(np.float64),
                           original_rows[sid]["faces"])
        for sid in range(305, 310)
    }
    after_volume_rows = {
        sid: signed_volume(rows3[sid]["vertices6"][:, :3].astype(np.float64),
                           rows3[sid]["faces"])
        for sid in range(305, 310)
    }
    volume_deltas = {sid: after_volume_rows[sid]-before_volume_rows[sid] for sid in range(305, 310)}
    if any(abs(delta) > 1e-12 for delta in volume_deltas.values()):
        raise ValueError("conditioned geometry exceeds the per-lobe signed-volume bound")
    before_total_volume = sum(before_volume_rows.values())
    total_volume = sum(after_volume_rows.values())
    if abs(total_volume-before_total_volume) > 1e-12:
        raise ValueError("conditioned geometry exceeds the aggregate signed-volume bound")

    np.save(diaphragm_order_path, row311_face_order, allow_pickle=False)
    interface["source_geometry_registration_reconstruction"]["row311_source_face_order_sidecar"] = {
        "path": str(diaphragm_order_path),
        "sha256": sha256(diaphragm_order_path),
        "encoding": "new row311 face index to input source row311 face index",
    }
    ancestry_path = output_dir / "lobe-source-face-ancestry.npz"
    ancestry_arrays = {}
    ancestry_summary = {}
    for sid, source_map in sorted(source_face_ancestry.items()):
        offsets = np.zeros(len(source_map)+1, dtype="<i8")
        children = []
        for source_face, candidate_faces in enumerate(source_map):
            children.extend(sorted(set(map(int, candidate_faces))))
            offsets[source_face+1] = len(children)
        ancestry_arrays[f"row{sid}_source_face_offsets"] = offsets
        ancestry_arrays[f"row{sid}_candidate_face_ids"] = np.asarray(children, dtype="<i8")
        ancestry_summary[str(sid)] = {
            "source_face_count": len(source_map),
            "candidate_face_count": len(rows3[sid]["faces"]),
            "deleted_source_face_ids": [fi for fi, child in enumerate(source_map) if not child],
            "source_faces_with_multiple_children": sum(len(set(child)) > 1 for child in source_map),
        }
    np.savez_compressed(ancestry_path, **ancestry_arrays)
    fan_report["source_face_ancestry_sidecar"] = {
        "path": str(ancestry_path), "sha256": sha256(ancestry_path),
        "encoding": "CSR source face to final candidate faces for rows 305-309 and 311; empty child ranges record collapsed source faces",
        "candidate_faces_may_have_multiple_source_parents": True,
    }

    pre_dir, final_dir = output_dir / "pre-pleura", output_dir / "final"
    pre_dir.mkdir()
    pre_payload = pre_dir / "resting-thorax.nhanatomy"
    pre_payload.write_bytes(_serialize_payload(header, record_order, rows3))
    pre_sha = sha256(pre_payload)
    receipt["payload"].update(path=str(pre_payload), sha256=pre_sha,
                              input_payload_sha256=PRECISION_PAYLOAD_SHA256)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = pre_sha
    common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    common["anatomy_payload_sha256"] = pre_sha
    for key_name in ("map", "polynomials", "domain_boxes"):
        sidecar = common[key_name]
        source = Path(sidecar["path"])
        if not source.is_absolute():
            source = base_receipt_path.parent / source
        sidecar["path"] = str(source)

    old_volume = receipt["thorax_source_volume_m3"].pop("candidate_geometry_derivation", None)
    old_source_qualification = receipt.setdefault("qualification", {}).get("source_geometry_candidate")
    volume_rows = []
    for sid in range(305, 310):
        topo = validate_closed_oriented_surface(rows3[sid]["vertices6"][:, :3], rows3[sid]["faces"])
        if not topo["closed_oriented"] or topo["zero_area_face_count"]:
            raise ValueError(f"conditioned lobe {sid} is not closed and nondegenerate")
        volume_rows.append({
            "stable_id": sid, "enclosed_volume_m3": after_volume_rows[sid],
            "signed_volume_delta_from_input_m3": volume_deltas[sid],
            "vertex_count": len(rows3[sid]["vertices6"]), "face_count": len(rows3[sid]["faces"]),
            "topology": topo,
        })
    receipt["thorax_source_volume_m3"]["candidate_geometry_derivation"] = {
        "basis": "signed tetrahedral volume on final serialized Float32 lobe positions",
        "row_geometry_byte_sha256": {
            str(sid): hashlib.sha256(_geometry_bytes(rows3[sid])).hexdigest()
            for sid in range(305, 310)},
        "per_lobe": volume_rows,
        "input_lobe_volumes_m3": {str(sid): before_volume_rows[sid] for sid in range(305, 310)},
        "aggregate_signed_volume_delta_m3": total_volume-before_total_volume,
        "geometry_byte_order": "little-endian float32 vertices6 followed by little-endian int64 local faces",
    }
    receipt["thorax_source_volume_m3"]["five_lung_envelopes"] = [
        row["enclosed_volume_m3"] for row in volume_rows]
    receipt["thorax_source_volume_m3"]["sum"] = total_volume

    respiratory = receipt["functional_bindings"]["respiratory_geometry_binding"]
    old_area_derivation = respiratory.pop("source_refinement_area_update", None)
    respiratory["diaphragm_effective_area_m2"] = float(np.float32(after_area))
    respiratory["per_lobe_effective_area_m2"] = [
        {"lung_stable_id": row["lung_stable_id"],
         "effective_area_m2": row["effective_area_m2_float64"]}
        for row in area_rows]
    area_bit_identity = np.float32(before_area) == np.float32(after_area)
    respiratory["source_refinement_area_update"] = {
        "basis": "analytic signed-volume derivative on final serialized Float32 source vertices under the retained Kuhn weights",
        "kuhn_basis_source_path": str(respiratory_owner_path),
        "kuhn_basis_source_sha256": RESPIRATORY_OWNER_SHA256,
        "input_area_m2_float64": before_area, "candidate_area_m2_float64": after_area,
        "candidate_area_delta_m2": after_area-before_area,
        "candidate_runtime_area_m2_float32": float(np.float32(after_area)),
        "input_runtime_area_bit_identity": bool(area_bit_identity),
        "per_lobe_derivatives_m2": {
            str(row["lung_stable_id"]): row["effective_area_m2_float64"] for row in area_rows},
    }

    operations = {
        "position_preserving_flip_pass_1": flip_report1,
        "position_preserving_flip_pass_2": flip_report2,
        "synchronized_short_edge_collapse": collapse_report,
        "bounded_tangent_vertex_relocation": relocation_report,
        "shared_vertex_fan_refinement": fan_report,
        "diaphragm_interface_reconstruction": interface_reconstruction,
        "diaphragm_interface_refresh_before_fan": interface_refresh_before_fan,
        "diaphragm_interface_refresh_after_fan": interface_refresh_after_fan,
        "source_face_ancestry_summary": ancestry_summary,
        "post_fan_final_source_precision_screen": final_source_precision_screen,
    }
    receipt["qualification"]["source_geometry_candidate"] = (
        "Bounded source preparation: position-preserving interior flips, topology-checked short-edge collapses, "
        "exact-owner-propagated Float32 tangent relocations, a source-anchored fan refinement, and the existing "
        "pleura union owner. Runtime transformed-pose area and intersection gates remain strict and separate.")
    receipt["provenance"]["lung_source_precision_retriangulation"] = {
        "input_payload_path": str(base_payload), "input_payload_sha256": PRECISION_PAYLOAD_SHA256,
        "input_receipt_path": str(base_receipt_path), "input_receipt_sha256": PRECISION_RECEIPT_SHA256,
        "selection_altitude_m": 1e-6,
        "relocation_screen_altitude_m": 128e-9,
        "relocation_source_floor_m": 512e-9,
        "maximum_declared_collapse_endpoint_displacement_m": 1e-6,
        "maximum_declared_relocation_displacement_m": 1e-6,
        "maximum_abs_per_owner_and_aggregate_volume_delta_m3": 1e-12,
        "runtime_area_and_triangle_validators_were_not_relaxed": True,
        "input_runtime_area_bit_identity": bool(area_bit_identity),
        "row311_face_connectivity_unchanged_after_registration_reindex": True,
        "row311_face_order_reindexed_by_source_geometry_registration": True,
        "source_point_relocations_are_explicitly_recorded": True,
        "preexisting_source_face_deletions_are_recorded_in_ancestry_sidecar": True,
        "unresolved_source_faces_below_512nm": {
            sid: final_source_precision_screen[str(sid)]["face_count_below_512nm"]
            for sid in range(305, 310)},
        "pre_fan_relocation_stage_faces_below_512nm": {
            sid: len(relocation_report["remaining_subfloor_faces"][str(sid)])
            for sid in range(305, 310)},
        "post_fan_final_source_precision_screen": final_source_precision_screen,
        "old_source_volume_derivation": old_volume,
        "old_source_geometry_qualification": old_source_qualification,
        "lobe_operations": operations,
        "source_license": "CC-BY-SA-4.0; existing source attribution and share-alike obligations retained",
        "qualification_limit": "This is a derived source candidate. Native transformed-pose exact triangle, area, and functional-volume audits are still required.",
    }
    pre_receipt = pre_dir / "resting-anatomy-receipt.json"
    pre_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    build_pleura_candidate(pre_payload, pre_receipt, final_dir)

    final_payload = final_dir / "resting-thorax.nhanatomy"
    final_receipt_path = final_dir / "resting-anatomy-receipt.json"
    _, final_rows = parse_payload(final_payload)
    if set(final_rows) != set(original_rows):
        raise ValueError("source conditioning changed the anatomy inventory")
    for sid in original_rows:
        if sid in range(305, 312):
            continue
        if _geometry_bytes(final_rows[sid]) != _geometry_bytes(original_rows[sid]):
            raise ValueError(f"source conditioning changed unrelated anatomy row {sid}")
    for sid in range(305, 310):
        if _geometry_bytes(final_rows[sid]) != _geometry_bytes(rows3[sid]):
            raise ValueError(f"pleura owner changed conditioned lobe row {sid}")
    if not np.array_equal(final_rows[311]["faces"], rows3[311]["faces"]):
        raise ValueError("pleura owner changed diaphragm row 311 face connectivity")
    if not np.array_equal(final_rows[311]["vertices6"], rows3[311]["vertices6"]):
        raise ValueError("pleura owner changed diaphragm row 311 vertices or propagated normals")
    final_interface = json.loads(final_receipt_path.read_text())
    _refresh_diaphragm_interface_registration(final_interface, final_rows)

    lineage = _pleura_face_lineage(final_rows)
    lineage_path = output_dir / "row-310-face-lineage.npy"
    np.save(lineage_path, lineage, allow_pickle=False)
    final_sha = sha256(final_payload)
    final_receipt = json.loads(final_receipt_path.read_text())
    final_common = final_receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    if final_common["anatomy_payload_sha256"] != pre_sha:
        raise ValueError("inherited common cardiac field has unexpected intermediate anatomy identity")
    final_common["anatomy_payload_sha256"] = final_sha
    for key_name in ("map", "polynomials", "domain_boxes"):
        sidecar = final_common[key_name]
        source = Path(sidecar["path"])
        if not source.is_absolute():
            source = base_receipt_path.parent / source
        destination = final_dir / source.name
        _copy_checked_sidecar(source, destination, sidecar["sha256"])
        sidecar["path"] = destination.name
    final_receipt["provenance"]["lung_source_precision_retriangulation"] = (
        receipt["provenance"]["lung_source_precision_retriangulation"])
    repair = final_receipt["provenance"]["lung_source_precision_retriangulation"]
    module_hashes_at_completion = {name: {"path": str(path), "sha256": sha256(path)}
                                   for name, path in module_paths.items()}
    if module_hashes_at_completion != module_hashes_at_launch:
        source_pins.update(status="failed_source_changed_during_build",
                            modules_at_completion=module_hashes_at_completion)
        source_pin_path.write_text(json.dumps(source_pins, indent=2, sort_keys=True)+"\n")
        raise ValueError("candidate implementation module changed during source build")
    source_pins.update(status="completed", modules_at_completion=module_hashes_at_completion,
                       implementation_unchanged_during_build=True)
    source_pin_path.write_text(json.dumps(source_pins, indent=2, sort_keys=True)+"\n")
    repair.update(
        output_payload_sha256=final_sha,
        row_310_face_lineage_path=str(lineage_path),
        row_310_face_lineage_sha256=sha256(lineage_path),
        row_310_face_lineage_face_count=len(lineage),
        source_face_ancestry_sidecar_path=str(ancestry_path),
        source_face_ancestry_sidecar_sha256=sha256(ancestry_path),
        implementation_module_pins_path=str(source_pin_path),
        implementation_module_pins_sha256=sha256(source_pin_path),
        implementation_module_hashes=source_pins,
        unrelated_surface_geometry_byte_identity=True,
        row311_face_connectivity_unchanged_after_registration_reindex=True,
    )
    _refresh_diaphragm_interface_registration(final_receipt, final_rows)
    final_receipt_path.write_text(json.dumps(final_receipt, indent=2, sort_keys=True)+"\n")
    final_report = {
        "schema": "numi.human.lung-source-edge-repair-reproduction.v2",
        "status": "bounded_source_candidate_built_native_pose_gate_separate",
        "source_payload_sha256": PRECISION_PAYLOAD_SHA256,
        "source_receipt_sha256": PRECISION_RECEIPT_SHA256,
        "pre_pleura_payload_path": str(pre_payload), "pre_pleura_payload_sha256": pre_sha,
        "final_payload_path": str(final_payload), "final_payload_sha256": final_sha,
        "final_receipt_path": str(final_receipt_path), "final_receipt_sha256": sha256(final_receipt_path),
        "input_respiratory_area_m2": before_area, "candidate_respiratory_area_m2": after_area,
        "respiratory_area_delta_m2": after_area-before_area,
        "input_runtime_area_bit_identity": bool(area_bit_identity),
        "per_lobe_signed_volume_delta_m3": {str(sid): volume_deltas[sid] for sid in range(305,310)},
        "aggregate_lobe_volume_delta_m3": total_volume-before_total_volume,
        "remaining_lobe_faces_below_512nm": {
            str(sid): final_source_precision_screen[str(sid)]["face_count_below_512nm"]
            for sid in range(305,310)},
        "pre_fan_relocation_stage_faces_below_512nm": {
            str(sid): len(relocation_report["remaining_subfloor_faces"][str(sid)])
            for sid in range(305,310)},
        "post_fan_final_source_precision_screen": final_source_precision_screen,
        "row311_face_connectivity_unchanged_after_registration_reindex": True,
        "row311_face_order_changed_by_registration_reindex": True,
        "row311_source_face_order_sidecar_path": str(diaphragm_order_path),
        "row311_source_face_order_sidecar_sha256": sha256(diaphragm_order_path),
        "source_face_ancestry_sidecar_path": str(ancestry_path),
        "source_face_ancestry_sidecar_sha256": sha256(ancestry_path),
        "implementation_module_pins_path": str(source_pin_path),
        "implementation_module_pins_sha256": sha256(source_pin_path),
        "implementation_module_hashes": source_pins,
        "row_310_face_count": len(lineage),
        "unrelated_surface_geometry_byte_identity": True,
        "diaphragm_interface_reconstruction": interface_reconstruction,
        "native_transformed_pose_qualification": "not performed",
        "lobe_operations": operations,
    }
    (output_dir/"build-report.json").write_text(json.dumps(final_report,indent=2,sort_keys=True)+"\n")
    return final_report




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-payload", required=True, type=Path)
    parser.add_argument("--base-receipt", required=True, type=Path)
    parser.add_argument("--face-origin", type=Path)
    parser.add_argument("--precision-flips", action="store_true", help="repair pinned 852 source slivers with position-preserving flips")
    parser.add_argument("--respiratory-owner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.precision_flips:
        report = build_precision_candidate(args.base_payload, args.base_receipt,
                                           args.respiratory_owner, args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.face_origin is None:
        parser.error("--face-origin is required for the historical midpoint collapse")
    print(json.dumps(build_candidate(
        args.base_payload, args.base_receipt, args.face_origin,
        args.respiratory_owner, args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

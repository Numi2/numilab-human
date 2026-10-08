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
                            keep_vertex: int):
    """Collapse one manifold edge at its Float32 midpoint; retain face lineage."""
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
    midpoint = ((p + q) * 0.5).astype(np.float32)
    endpoint_displacements = [
        float(np.linalg.norm(midpoint.astype(np.float64) - p)),
        float(np.linalg.norm(midpoint.astype(np.float64) - q)),
    ]

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
    out_vertices[survivor, :3] = midpoint
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
            or source_components != 1 or source_euler != 2 or np.any(source_twice_area <= 0)
            or len(np.unique(f.reshape(-1))) != len(v)):
        raise ValueError("edge-collapse source is not one closed oriented genus-zero surface")
    if (result_topology["boundary_edge_count"] or result_topology["nonmanifold_edge_count"]
            or result_topology["orientation_error_edge_count"] or np.any(twice_area <= 0)):
        raise ValueError("edge-collapse result is not closed, oriented, and nondegenerate")
    duplicate_faces = len(out_faces) - len(np.unique(np.sort(out_faces, axis=1), axis=0))
    edge_count = _edge_count(out_faces)
    euler = len(out_vertices) - edge_count + len(out_faces)
    components = _connected_component_count(len(out_vertices), out_faces)
    if duplicate_faces or components != 1 or euler != 2:
        raise ValueError("edge-collapse result is not one closed genus-zero surface")

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
    if abs(volume_delta) > 1e-12:
        raise ValueError("edge collapse changes registered lobe volume beyond 1e-12 m3")

    report = {
        "operation": "one topology-preserving midpoint edge collapse",
        "removed_vertex_before_compaction": u,
        "retained_vertex_before_compaction": w,
        "retained_vertex_after_compaction": survivor,
        "edge_length_m": edge_length,
        "midpoint_position_f32_m": midpoint.astype(float).tolist(),
        "endpoint_displacements_m": endpoint_displacements,
        "maximum_endpoint_displacement_m": max(endpoint_displacements),
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
            "edge_count": source_edge_count,
            "boundary_edges": source_topology["boundary_edge_count"],
            "nonmanifold_edges": source_topology["nonmanifold_edge_count"],
            "orientation_error_edges": source_topology["orientation_error_edge_count"],
            "duplicate_faces": int(source_duplicates), "connected_components": source_components,
            "euler_characteristic": int(source_euler),
        },
        "candidate_topology": {
            "vertex_count": int(len(out_vertices)), "face_count": int(len(out_faces)),
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



def _triangle_altitudes(xyz, faces):
    tri = np.asarray(xyz, dtype=np.float64)[faces]
    longest = np.maximum.reduce([np.linalg.norm(tri[:, (i+1)%3] - tri[:, i], axis=1)
                                 for i in range(3)])
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    return np.divide(area2, longest, out=np.zeros_like(area2), where=longest > 0)


def _condition_precision_lobes(original_rows, *, minimum_altitude_m):
    """Bounded source conditioning; shared interlobar patches flip together."""
    from .surface_precision_retriangulation import (
        build_surface_adjacency, flip_interior_edge, validate_closed_oriented_surface,
    )
    selection_altitude = 128e-9
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
    initial_shared = {key: tuple(sorted(value)) for key, value in lookup.items() if len(value) > 1}
    initial_counts = {sid: int((_triangle_altitudes(positions[sid], rows[sid]["faces"]) < selection_altitude).sum())
                      for sid in positions}
    seeds = sorted((float(h), sid, int(fi))
                   for sid in positions
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
    for sid in positions:
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
        "selection_basis": "128 nm source risk screen above twice the diagnosed 59.6 nm world-y ULP. This is an engineering selection, not a proof of all transformed Float32 poses.",
        "operations": operations, "blocked_attempts": blocked, "lobes": per_lobe,
        "shared_lobe_face_ownership_counts_preserved": True,
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


def build_precision_candidate(base_payload: Path, base_receipt_path: Path,
                              respiratory_owner_path: Path, output_dir: Path):
    """Repair the pinned 852 source slivers, retaining the existing owners.

    128 nm is a source preparation selection bound, not an area
    tolerance in the native validator. All vertex positions remain unchanged.
    Runtime Float32 transformed-pose qualification is still required.
    """
    from .resting_anatomy import _copy_checked_sidecar

    base_payload, base_receipt_path = Path(base_payload), Path(base_receipt_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace candidate output: {output_dir}")
    if sha256(base_payload) != PRECISION_PAYLOAD_SHA256:
        raise ValueError("precision repair requires the retained 852 anatomy payload")
    if sha256(base_receipt_path) != PRECISION_RECEIPT_SHA256:
        raise ValueError("precision repair requires the retained 852 anatomy receipt")
    original_receipt = json.loads(base_receipt_path.read_text())
    if (original_receipt.get("schema") != "numi.human.resting-anatomy-receipt.v1"
            or original_receipt.get("payload", {}).get("sha256") != PRECISION_PAYLOAD_SHA256
            or original_receipt.get("functional_bindings", {}).get("anatomy_payload_sha256")
            != PRECISION_PAYLOAD_SHA256):
        raise ValueError("precision input receipt does not bind the source anatomy")
    respiratory_owner = _load_pinned_respiratory_owner(Path(respiratory_owner_path))
    raw = base_payload.read_bytes()
    header, original_rows = parse_payload(base_payload)
    record_order = [int(RECORD.unpack_from(raw, HEADER.size + i * RECORD.size)[5])
                    for i in range(int(header[2]))]
    if any(int(original_rows[sid]["body_index"]) != 20 for sid in range(305, 311)):
        raise ValueError("precision repair source lung ownership differs")
    rows, operations = _condition_precision_lobes(original_rows, minimum_altitude_m=1e-6)
    # Derive again from the serialized Float32 geometry, not from expected deltas.
    before_area, _ = _derive_basal_effective_area(original_rows, respiratory_owner.kuhn_basis)
    after_area, area_rows = _derive_basal_effective_area(rows, respiratory_owner.kuhn_basis)
    if np.float32(before_area) != np.float32(after_area):
        raise ValueError("precision repair changed the runtime diaphragm area; needs separate qualification")
    before_volume, _ = _source_volume_rows(original_rows)
    total_volume, volume_rows = _source_volume_rows(rows)
    if abs(total_volume - before_volume) > 1e-12:
        raise ValueError("precision repair changed total lobe volume by more than 1e-12 m3")

    output_dir.mkdir(parents=True)
    pre_dir, final_dir = output_dir / "pre-pleura", output_dir / "final"
    pre_dir.mkdir()
    pre_payload = pre_dir / "resting-thorax.nhanatomy"
    pre_payload.write_bytes(_serialize_payload(header, record_order, rows))
    pre_sha = sha256(pre_payload)
    receipt = json.loads(json.dumps(original_receipt))
    receipt["payload"].update(path=str(pre_payload), sha256=pre_sha,
                              input_payload_sha256=PRECISION_PAYLOAD_SHA256)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = pre_sha
    pre_common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    pre_common["anatomy_payload_sha256"] = pre_sha
    for key in ("map", "polynomials", "domain_boxes"):
        sidecar = pre_common[key]
        sidecar_path = Path(sidecar["path"])
        if not sidecar_path.is_absolute():
            sidecar_path = base_receipt_path.parent / sidecar_path
        sidecar["path"] = str(sidecar_path)
    previous_volume_derivation = receipt["thorax_source_volume_m3"].pop("candidate_geometry_derivation", None)
    previous_qualification = receipt.setdefault("qualification", {}).get("source_geometry_candidate")
    receipt["qualification"]["source_geometry_candidate"] = (
        "Position-preserving bounded lung edge flips followed by the existing pleura union owner. "
        "Source topology and volume checks pass; native transformed-pose area/intersection gates are separate. "
        "The pre-pleura directory is a preparation intermediate and is not the launchable result.")
    receipt["thorax_source_volume_m3"]["candidate_geometry_derivation"] = {
        "basis": "Signed tetrahedral volumes on current serialized Float32 lobe geometry",
        "row_geometry_byte_sha256": {str(sid): hashlib.sha256(_geometry_bytes(rows[sid])).hexdigest() for sid in range(305, 310)},
        "per_lobe": volume_rows,
        "geometry_byte_order": "little-endian float32 vertices6 then little-endian int64 local triangle indices",
    }
    receipt["thorax_source_volume_m3"]["five_lung_envelopes"] = [
        r["enclosed_volume_m3"] for r in volume_rows]
    receipt["thorax_source_volume_m3"]["sum"] = total_volume
    resp = receipt["functional_bindings"]["respiratory_geometry_binding"]
    previous_derivation = resp.pop("source_refinement_area_update", None)
    resp["diaphragm_effective_area_m2"] = float(np.float32(after_area))
    resp["per_lobe_effective_area_m2"] = [
        {"lung_stable_id": r["lung_stable_id"], "effective_area_m2": r["effective_area_m2_float64"]}
        for r in area_rows]
    resp["source_refinement_area_update"] = {
        "basis": "Analytic signed-volume derivative on the current serialized Float32 positions and faces under the retained Kuhn vertex weights, accumulated in Float64.",
        "kuhn_basis_source_path": str(respiratory_owner_path),
        "kuhn_basis_source_sha256": RESPIRATORY_OWNER_SHA256,
        "previous_area_m2_float64": before_area,
        "refined_area_m2_float64": after_area,
        "refined_area_m2_runtime_float32": float(np.float32(after_area)),
        "per_lobe_derivatives_m2": {str(r["lung_stable_id"]): r["effective_area_m2_float64"] for r in area_rows},
        "runtime_area_bit_identity": True,
    }
    receipt["provenance"]["lung_source_precision_retriangulation"] = {
        "input_payload_path": str(base_payload),
        "input_payload_sha256": PRECISION_PAYLOAD_SHA256,
        "input_receipt_path": str(base_receipt_path),
        "input_receipt_sha256": PRECISION_RECEIPT_SHA256,
        "source_preparation_selection_altitude_m": 128e-9,
        "replacement_minimum_altitude_m": 1e-6,
        "selection_bound_is_not_a_runtime_area_tolerance": True,
        "all_lobe_vertex_positions_byte_identical": True,
        "previous_respiratory_area_derivation": previous_derivation,
        "previous_volume_derivation": previous_volume_derivation,
        "previous_source_geometry_qualification": previous_qualification,
        "lobe_operations": operations,
        "source_license": "CC-BY-SA-4.0; existing source attribution and share-alike obligations retained",
        "qualification_limit": "Changed triangles need not conform to the original Kuhn cell partition. The same field is evaluated at unchanged vertices; transformed geometry and functional volume residuals require native validation.",
    }
    pre_receipt = pre_dir / "resting-anatomy-receipt.json"
    pre_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    build_pleura_candidate(pre_payload, pre_receipt, final_dir)
    final_payload = final_dir / "resting-thorax.nhanatomy"
    final_receipt_path = final_dir / "resting-anatomy-receipt.json"
    _, final_rows = parse_payload(final_payload)
    if set(final_rows) != set(original_rows):
        raise ValueError("precision repair changed anatomy inventory")
    for sid in original_rows:
        if sid in range(305, 311):
            continue
        if _geometry_bytes(final_rows[sid]) != _geometry_bytes(original_rows[sid]):
            raise ValueError(f"precision repair changed unrelated anatomy row {sid}")
    for sid in range(305, 310):
        if _geometry_bytes(final_rows[sid]) != _geometry_bytes(rows[sid]):
            raise ValueError(f"pleura owner changed repaired lobe {sid}")
        if (np.ascontiguousarray(final_rows[sid]["vertices6"][:, :3], dtype="<f4").tobytes()
                != np.ascontiguousarray(original_rows[sid]["vertices6"][:, :3], dtype="<f4").tobytes()):
            raise ValueError(f"precision repair moved a source position in lobe {sid}")
    lineage = _pleura_face_lineage(final_rows)
    lineage_path = output_dir / "row-310-face-lineage.npy"
    np.save(lineage_path, lineage, allow_pickle=False)
    final_sha = sha256(final_payload)
    final_receipt = json.loads(final_receipt_path.read_text())
    common = final_receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    if common["anatomy_payload_sha256"] != pre_sha:
        raise ValueError("inherited common cardiac field has unexpected anatomy identity")
    common["anatomy_payload_sha256"] = final_sha
    for key in ("map", "polynomials", "domain_boxes"):
        sidecar = common[key]
        source = Path(sidecar["path"])
        if not source.is_absolute():
            source = base_receipt_path.parent / source
        destination = final_dir / source.name
        _copy_checked_sidecar(source, destination, sidecar["sha256"])
        sidecar["path"] = destination.name
    repair = final_receipt["provenance"]["lung_source_precision_retriangulation"]
    repair.update(output_payload_sha256=final_sha,
                  row_310_face_lineage_path=str(lineage_path),
                  row_310_face_lineage_sha256=sha256(lineage_path),
                  row_310_face_lineage_face_count=len(lineage),
                  unrelated_surface_geometry_byte_identity=True)
    final_receipt_path.write_text(json.dumps(final_receipt, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": "numi.human.lung-source-edge-repair-reproduction.v1",
        "status": "source_candidate_built_native_pose_gate_separate",
        "source_payload_sha256": PRECISION_PAYLOAD_SHA256,
        "source_receipt_sha256": PRECISION_RECEIPT_SHA256,
        "final_payload_path": str(final_payload),
        "final_payload_sha256": final_sha,
        "final_receipt_path": str(final_receipt_path),
        "final_receipt_sha256": sha256(final_receipt_path),
        "respiratory_area_before_m2": before_area,
        "respiratory_area_after_m2": after_area,
        "runtime_area_bit_identity": True,
        "lobe_volume_delta_m3": total_volume - before_volume,
        "lobe_operations": operations,
        "row_310_face_count": len(lineage),
        "unrelated_surface_geometry_byte_identity": True,
        "all_lobe_vertex_positions_byte_identical": True,
        "native_rendering_qualification": "not performed by source builder",
    }
    (output_dir / "build-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


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

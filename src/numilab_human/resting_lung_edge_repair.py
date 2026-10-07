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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-payload", required=True, type=Path)
    parser.add_argument("--base-receipt", required=True, type=Path)
    parser.add_argument("--face-origin", required=True, type=Path)
    parser.add_argument("--respiratory-owner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_candidate(
        args.base_payload, args.base_receipt, args.face_origin,
        args.respiratory_owner, args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

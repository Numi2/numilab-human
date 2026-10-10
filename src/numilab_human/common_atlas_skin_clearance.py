from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
import mmap
import struct
from time import monotonic
from pathlib import Path
from typing import Any

import numpy as np

from . import model as human
from . import cardiac_cavity_intersections as ci
from .cardiac_cavity_geometry import analyze_topology
from .cardiac_cavity_intersections import (
    _audit_pair,
    _audit_pair_prepared_first,
    _prepare_surface_aabb,
    _records,
    float32_point_lattice_key,
)

_PACK_HEADER = struct.Struct("<8sIIQQ32s24s")
_PACK_DIRECTORY = struct.Struct("<IIQQQII32s")
_BONE_HEADER = struct.Struct("<8s5I32s")
_BONE_RECORD = struct.Struct("<6I8fI")
_BONE_VERTEX = struct.Struct("<6f")
_EXPECTED_SURFACE_COUNTS = {
    51004: 185, 51005: 148, 51006: 2, 51010: 166, 51011: 67,
    51012: 1, 51020: 98, 51021: 97, 51022: 85, 51023: 5, 51024: 1, 51025: 4,
}
_EXPECTED_ALL_SURFACE_COUNT = 859
_EXPECTED_ALL_WITNESS_COUNT = 8863
_EXPECTED_CLEARANCE_SURFACE_COUNT = 842
_EXPECTED_CLEARANCE_WITNESS_COUNT = 4448
_EXPECTED_OCULAR_WITNESS_COUNT = 4415
_EXPECTED_ORIENTATION_FACE_COUNT = 980
_OCULAR_MONITOR_KEYS = {(51010, stable_id) for stable_id in range(381, 398)}
_DIRECTION_SMOOTHING_ITERATIONS = 8
_GEODESIC_SEED_BATCH_SIZE = 16
_MIN_CANDIDATE_DIRECTION_PROJECTION = 0.5
_ROTATED_ACTIVE_FACE_BASE_DOT = 0.75
_CANDIDATE_WINDING_OFFSET_M = 0.0005
_MIN_CANDIDATE_WINDING_CONTRAST = 0.5


def _sha(path: Path) -> str:
    return human.sha256(path)


def _rotation(q: Any) -> np.ndarray:
    return np.asarray(human._myosim_matrix_from_quaternion_xyzw(np.asarray(q, dtype=np.float32).astype(float).tolist()), dtype=np.float64)


def _load_target_inventory(inventory_path: Path):
    rows = list(csv.DictReader(Path(inventory_path).open(newline="")))
    keys = set()
    expected_counts = {}
    populations = {}
    for row in rows:
        key = (int(row["second_semantic"]), int(row["second_stable_id"]))
        if key in keys or row.get("audit_complete") != "True":
            raise human.ImportError("clearance target inventory has duplicate or incomplete surface rows")
        keys.add(key)
        expected_counts[key] = int(row["intersecting_triangle_pairs"])
        populations[key[0]] = populations.get(key[0], 0) + 1
    if populations != _EXPECTED_SURFACE_COUNTS or len(keys) != _EXPECTED_ALL_SURFACE_COUNT:
        raise human.ImportError(f"clearance target inventory surface partition changed: {populations}")
    if sum(expected_counts.values()) != _EXPECTED_ALL_WITNESS_COUNT:
        raise human.ImportError("clearance all-surface inventory intersection total changed")
    if not _OCULAR_MONITOR_KEYS.issubset(keys):
        raise human.ImportError("clearance inventory omits one of the separately monitored ocular interfaces")
    clearance_keys = keys - _OCULAR_MONITOR_KEYS
    if (len(clearance_keys) != _EXPECTED_CLEARANCE_SURFACE_COUNT
            or sum(expected_counts[key] for key in clearance_keys) != _EXPECTED_CLEARANCE_WITNESS_COUNT
            or sum(expected_counts[key] for key in _OCULAR_MONITOR_KEYS) != _EXPECTED_OCULAR_WITNESS_COUNT):
        raise human.ImportError("clearance and ocular-monitor surface partition changed")
    return keys, clearance_keys, expected_counts, populations


def _pack_surfaces(pack_path: Path, target_keys: set[tuple[int, int]]):
    with pack_path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    header = _PACK_HEADER.unpack_from(mapped, 0)
    if header[0] != b"MRVPACK2" or header[1] != 2:
        mapped.close()
        raise human.ImportError("clearance MRVPACK ABI is unsupported")
    sections = {}
    for row in range(header[2]):
        item = _PACK_DIRECTORY.unpack_from(mapped, _PACK_HEADER.size + row * _PACK_DIRECTORY.size)
        sections[item[0]] = item
    try:
        vertex_row, index_row, primitive_row = sections[2], sections[3], sections[4]
    except KeyError as error:
        mapped.close()
        raise human.ImportError("clearance MRVPACK lacks geometry sections") from error
    if (vertex_row[5], index_row[5], primitive_row[5]) != (80, 4, 64):
        mapped.close()
        raise human.ImportError("clearance MRVPACK geometry strides are unsupported")
    vertex_count, index_count, primitive_count = vertex_row[4], index_row[4], primitive_row[4]
    packed_vertices = np.ndarray((vertex_count, 20), dtype="<f4", buffer=mapped, offset=vertex_row[2])
    positions = packed_vertices[:, :3].copy()
    packed_indices = np.ndarray((index_count,), dtype="<u4", buffer=mapped, offset=index_row[2])
    surfaces = {}
    for row in range(primitive_count):
        offset = primitive_row[2] + row * 64
        first, count = struct.unpack_from("<2I", mapped, offset)
        semantic, stable_id, body_id, _ = struct.unpack_from("<4I", mapped, offset + 16)
        key = (int(semantic), int(stable_id))
        if (key[0] == 51004 and 1 <= key[1] <= 185) or key in target_keys or key == (51007, 1):
            surfaces[key] = {"faces": packed_indices[first:first + count].reshape(-1, 3).copy(), "body": int(body_id)}
    mapped.close()
    if (51007, 1) not in surfaces or not target_keys.issubset(surfaces):
        raise human.ImportError("clearance MRVPACK omits the skin or a surface from the pinned 859-surface inventory")
    return positions, surfaces, {"vertices": int(vertex_count), "primitives": int(primitive_count), "indices": int(index_count)}


def _fit_registered_bone_poses(pack_positions: np.ndarray, surfaces: dict, bone_path: Path, manifest_path: Path):
    raw = bone_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    header = _BONE_HEADER.unpack_from(raw, 0)
    magic, abi, record_count, vertex_count, index_count, fingerprint, archive_sha = header
    if magic != b"NHBONES1" or abi != 3:
        raise human.ImportError("clearance pose reconstruction requires NHBONES ABI 3")
    anchors = manifest.get("source", {}).get("anchors")
    if not isinstance(anchors, list) or len(anchors) != record_count or record_count != 185:
        raise human.ImportError("clearance NHBONES manifest does not exactly cover the source records")
    if manifest.get("source", {}).get("myosim_source_archive_sha256") != archive_sha.hex():
        raise human.ImportError("clearance NHBONES source archive differs from its manifest")
    expected_size = _BONE_HEADER.size + record_count * _BONE_RECORD.size + vertex_count * _BONE_VERTEX.size + index_count * 4
    if len(raw) != expected_size:
        raise human.ImportError("clearance NHBONES payload length is inconsistent")
    records = [_BONE_RECORD.unpack_from(raw, _BONE_HEADER.size + index * _BONE_RECORD.size) for index in range(record_count)]
    vertices_offset = _BONE_HEADER.size + record_count * _BONE_RECORD.size
    bone_vertices = np.ndarray((vertex_count, 6), dtype="<f4", buffer=raw, offset=vertices_offset)
    indices_offset = vertices_offset + vertex_count * _BONE_VERTEX.size
    bone_indices = np.ndarray((index_count,), dtype="<u4", buffer=raw, offset=indices_offset)
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {}
    stable_to_body: dict[int, int] = {}
    for source_row, (record, anchor) in enumerate(zip(records, anchors)):
        body_id, first_vertex, member_vertices, first_index, member_indices, stable_id, tx, ty, tz, qx, qy, qz, qw, scale, source_index = record
        if stable_id != source_row + 1 or source_index != anchor.get("source_record_index") or body_id != anchor.get("core_body_index"):
            raise human.ImportError(f"clearance NHBONES row {source_row + 1} differs from its pinned source anchor")
        key = (51004, int(stable_id))
        surface = surfaces.get(key)
        if surface is None or surface["body"] != int(body_id):
            raise human.ImportError(f"clearance accepted pack does not contain NHBONES member {stable_id} on its source body")
        global_faces = surface["faces"]
        if global_faces.size == 0:
            raise human.ImportError(f"clearance accepted pack has empty bone member {stable_id}")
        base = int(global_faces.min())
        source_faces = bone_indices[first_index:first_index + member_indices].reshape(-1, 3) - first_vertex
        if (len(source_faces) != len(global_faces) or int(global_faces.max()) - base + 1 != member_vertices
                or not np.array_equal(global_faces - base, source_faces)):
            raise human.ImportError(f"clearance bone topology/order differs for source member {stable_id}")
        local = bone_vertices[first_vertex:first_vertex + member_vertices, :3].astype(np.float64)
        rotation = _rotation((qx, qy, qz, qw))
        owner_local = np.array([tx, ty, tz], dtype=np.float64) + float(scale) * (local @ rotation.T)
        world = pack_positions[base:base + member_vertices].astype(np.float64)
        if world.shape != owner_local.shape:
            raise human.ImportError(f"clearance accepted bone vertex slice is malformed for member {stable_id}")
        grouped.setdefault(int(body_id), []).append((owner_local, world, str(anchor.get("member_id"))))
        stable_to_body[int(stable_id)] = int(body_id)
    poses, residuals = {}, {}
    for body_id, members in grouped.items():
        source = np.concatenate([item[0] for item in members], axis=0)
        target = np.concatenate([item[1] for item in members], axis=0)
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_mean - rotation @ source_mean
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        determinant = float(np.linalg.det(rotation))
        if determinant < 0.999999 or determinant > 1.000001 or float(errors.max()) > 1.0e-6:
            raise human.ImportError(f"clearance captured owner pose fit is not rigid/accurate for body {body_id}")
        poses[body_id] = (rotation, translation)
        residuals[str(body_id)] = {
            "member_count": len(members), "vertex_count": int(len(errors)),
            "maximum_rigid_fit_residual_um": float(errors.max() * 1.0e6), "determinant": determinant,
            "members": [item[2] for item in members],
        }
    if len(poses) != 86:
        raise human.ImportError(f"clearance pose reconstruction recovered {len(poses)} owners, expected 86")
    return poses, stable_to_body, residuals, {
        "abi": int(abi), "record_count": int(record_count), "vertex_count": int(vertex_count),
        "index_count": int(index_count), "fingerprint32": f"{fingerprint:08x}",
        "source_archive_sha256": archive_sha.hex(),
    }


def _exact_surface_records(vertices: np.ndarray, faces: np.ndarray):
    lattice = [float32_point_lattice_key(point) for point in vertices]
    return _records(lattice, faces)


def _prepare_closed_clearance_target(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    allow_nested_enclosure: bool = False,
) -> dict[str, Any]:
    """Prepare a captured target, optionally using its proven outer envelope.

    The exact-coordinate quotient only rejoins bit-identical Float32 xyz rows.
    In explicit nested-enclosure mode every component remains in the returned
    full target and its exact self/cross audits. Only the private inside/ray
    surface is restricted to the uniquely proven outer component; this means
    "external-skin-enclosure-only", not that inner components are cavities or
    disposable markers.
    """
    if type(allow_nested_enclosure) is not bool:
        raise human.ImportError("clearance nested-enclosure option must be a boolean")
    raw_vertices = np.asarray(vertices)
    raw_faces = np.asarray(faces)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or raw_vertices.dtype.kind not in "fiu":
        raise human.ImportError("clearance target vertices must be a numeric Nx3 array")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu":
        raise human.ImportError("clearance target faces must be an integer Fx3 array")
    if not len(raw_vertices) or not len(raw_faces):
        raise human.ImportError("clearance target must contain vertices and faces")
    if len(raw_faces) > 200_000 or len(raw_vertices) > 100_000:
        raise human.ImportError("clearance target exceeds the bounded mesh size")
    try:
        packed_vertices = np.ascontiguousarray(raw_vertices, dtype="<f4")
    except (OverflowError, ValueError, TypeError) as error:
        raise human.ImportError("clearance target coordinates are not representable as Float32") from error
    original_numeric = np.asarray(raw_vertices, dtype=np.float64)
    if not np.isfinite(original_numeric).all() or not np.isfinite(packed_vertices).all():
        raise human.ImportError("clearance target coordinates must be finite")
    if not np.array_equal(original_numeric, packed_vertices.astype(np.float64)):
        raise human.ImportError("clearance target coordinates must already be exact Float32 values")
    if np.any(raw_faces < 0) or np.any(raw_faces >= len(packed_vertices)):
        raise human.ImportError("clearance target face index is outside its vertex table")
    source_faces = np.ascontiguousarray(raw_faces, dtype="<u8")

    source_to_quotient = np.empty(len(packed_vertices), dtype=np.int64)
    unique_vertices = []
    coordinate_ids: dict[bytes, int] = {}
    for source_id, point in enumerate(packed_vertices):
        key = point.tobytes()
        quotient_id = coordinate_ids.get(key)
        if quotient_id is None:
            quotient_id = len(unique_vertices)
            coordinate_ids[key] = quotient_id
            unique_vertices.append(point.copy())
        source_to_quotient[source_id] = quotient_id
    quotient_vertices = np.ascontiguousarray(unique_vertices, dtype="<f4")
    quotient_faces = np.ascontiguousarray(source_to_quotient[source_faces.astype(np.int64)], dtype=np.int64)
    input_hashes = {
        "vertices_f32_sha256": hashlib.sha256(packed_vertices.tobytes()).hexdigest(),
        "faces_u64_sha256": hashlib.sha256(source_faces.tobytes()).hexdigest(),
    }
    quotient_hashes = {
        "vertices_f32_sha256": hashlib.sha256(quotient_vertices.tobytes()).hexdigest(),
        "faces_i64_sha256": hashlib.sha256(quotient_faces.astype("<i8", copy=False).tobytes()).hexdigest(),
    }
    topology = analyze_topology(quotient_vertices.astype(np.float64).tolist(), quotient_faces.tolist())
    records = tuple(_records(
        [float32_point_lattice_key(point) for point in quotient_vertices],
        quotient_faces.tolist(),
    ))
    self_audit = _audit_pair(records, records, same_surface=True)
    components = topology["face_components"]
    all_faces_closed_oriented = (
        topology["closed_oriented_manifold_candidate"]
        and not topology["unused_vertex_ids"]
        and bool(components)
        and all(components)
    )
    report: dict[str, Any] = {
        "quotient_method": "exact_bitwise_float32_xyz_identification_no_tolerance_or_geometry_change",
        "source_coordinates_modified": False,
        "faces_added": 0,
        "input_vertex_count": int(len(packed_vertices)),
        "quotient_vertex_count": int(len(quotient_vertices)),
        "identified_duplicate_coordinate_records": int(len(packed_vertices) - len(quotient_vertices)),
        "face_count": int(len(quotient_faces)),
        "input_hashes": input_hashes,
        "quotient_hashes": quotient_hashes,
        "topology": topology,
        "self_intersection_audit": self_audit,
        "closed_connected_oriented": bool(all_faces_closed_oriented and len(components) == 1),
        "all_components_closed_oriented_unused_free": bool(all_faces_closed_oriented),
        "embedded_closed_target": False,
        "inside_semantics": "unqualified",
        "external_skin_enclosure_only": False,
        "component_face_rows": [
            {"component_id": component_id, "face_rows": [int(row) for row in face_rows],
             "face_count": int(len(face_rows))}
            for component_id, face_rows in enumerate(components)
        ],
        "component_pair_proofs": [],
    }

    def reject(message: str):
        error = human.ImportError(message)
        error.target_report = report
        raise error

    if not all_faces_closed_oriented:
        reject("clearance target components must all be closed, oriented, and unused-vertex-free")
    if self_audit["count"] != 0:
        reject("clearance target must be exact-quotient and self-intersection-free")
    if len(components) > 1 and not allow_nested_enclosure:
        reject("clearance target must be closed, oriented, and connected unless explicit nested enclosure is enabled")

    lattice_vertices = [float32_point_lattice_key(point) for point in quotient_vertices]
    component_records = []
    component_representatives = []
    component_vertex_ids = []
    for component_id, face_rows in enumerate(components):
        face_rows = [int(row) for row in face_rows]
        component_faces = quotient_faces[np.asarray(face_rows, dtype=np.int64)]
        vertex_ids = np.unique(component_faces)
        local_faces = np.searchsorted(vertex_ids, component_faces)
        local_vertices = quotient_vertices[vertex_ids]
        component_topology = analyze_topology(
            local_vertices.astype(np.float64).tolist(), local_faces.astype(np.int64).tolist())
        if (not component_topology["closed_oriented_manifold_candidate"]
                or component_topology["face_component_count"] != 1
                or component_topology["unused_vertex_ids"]):
            reject(f"clearance target component {component_id} is not independently closed and oriented")
        component_records.append(tuple(records[row] for row in face_rows))
        component_representatives.append(lattice_vertices[int(component_faces[0, 0])])
        component_vertex_ids.append(vertex_ids)

    relation_by_pair: dict[tuple[int, int], str] = {}
    for first_id in range(len(components)):
        for second_id in range(first_id + 1, len(components)):
            first_records = component_records[first_id]
            second_records = component_records[second_id]
            cross_audit = _audit_pair(first_records, second_records, same_surface=False)
            if cross_audit["count"] != 0:
                report["component_pair_proofs"].append({
                    "first_component_id": first_id,
                    "second_component_id": second_id,
                    "cross_component_intersection_count": int(cross_audit["count"]),
                    "cross_component_triangle_pairs": cross_audit["triangle_pairs"],
                    "relation": "intersecting_or_touching",
                })
                reject("clearance target components must be pairwise nonintersecting")
            first_in_second = ci.point_location(
                component_representatives[first_id], second_records).get("location")
            second_in_first = ci.point_location(
                component_representatives[second_id], first_records).get("location")
            if first_in_second not in ("inside", "outside") or second_in_first not in ("inside", "outside"):
                report["component_pair_proofs"].append({
                    "first_component_id": first_id,
                    "second_component_id": second_id,
                    "cross_component_intersection_count": 0,
                    "first_representative_in_second": first_in_second,
                    "second_representative_in_first": second_in_first,
                    "relation": "ambiguous",
                })
                reject("clearance target component containment is boundary or indeterminate")
            if first_in_second == "inside" and second_in_first == "outside":
                relation = "second_contains_first"
            elif second_in_first == "inside" and first_in_second == "outside":
                relation = "first_contains_second"
            elif first_in_second == "outside" and second_in_first == "outside":
                relation = "disjoint"
            else:
                relation = "ambiguous"
                report["component_pair_proofs"].append({
                    "first_component_id": first_id,
                    "second_component_id": second_id,
                    "cross_component_intersection_count": 0,
                    "first_representative_in_second": first_in_second,
                    "second_representative_in_first": second_in_first,
                    "relation": relation,
                })
                reject("clearance target component containment is ambiguous")
            relation_by_pair[(first_id, second_id)] = relation
            report["component_pair_proofs"].append({
                "first_component_id": first_id,
                "second_component_id": second_id,
                "cross_component_intersection_count": 0,
                "first_representative_in_second": first_in_second,
                "second_representative_in_first": second_in_first,
                "relation": relation,
            })

    if len(components) == 1:
        outer_component_id = 0
        report["inside_semantics"] = "full_closed_target"
    else:
        outer_candidates = []
        for candidate_id in range(len(components)):
            contains_all = True
            for other_id in range(len(components)):
                if candidate_id == other_id:
                    continue
                pair = tuple(sorted((candidate_id, other_id)))
                relation = relation_by_pair[pair]
                candidate_contains = (
                    relation == "first_contains_second" if candidate_id < other_id
                    else relation == "second_contains_first")
                if not candidate_contains:
                    contains_all = False
                    break
            if contains_all:
                outer_candidates.append(candidate_id)
        if len(outer_candidates) != 1:
            reject("clearance target needs exactly one component enclosing every other component")
        outer_component_id = outer_candidates[0]
        report["inside_semantics"] = "external_skin_enclosure_only"
        report["external_skin_enclosure_only"] = True

    outer_face_rows = [int(row) for row in components[outer_component_id]]
    outer_vertex_ids = component_vertex_ids[outer_component_id]
    interior_records = component_records[outer_component_id]
    report["outer_component_id"] = int(outer_component_id)
    report["outer_component_face_rows"] = outer_face_rows
    report["outer_component_face_count"] = int(len(outer_face_rows))
    report["interior_face_count"] = int(len(interior_records))
    report["embedded_closed_target"] = True

    quotient_vertices.setflags(write=False)
    quotient_faces.setflags(write=False)
    interior_aabb_min = quotient_vertices[outer_vertex_ids].min(axis=0).copy()
    interior_aabb_max = quotient_vertices[outer_vertex_ids].max(axis=0).copy()
    interior_aabb_min.setflags(write=False)
    interior_aabb_max.setflags(write=False)
    return {
        "vertices": quotient_vertices,
        "faces": quotient_faces,
        "records": records,
        "_interior_records": interior_records,
        "_interior_point_locator": ci.prepare_point_location(interior_records),
        "_interior_face_rows": tuple(outer_face_rows),
        "report": report,
        "_inside_aabb_min": interior_aabb_min,
        "_inside_aabb_max": interior_aabb_max,
    }


def _require_prepared_closed_target(target: dict[str, Any]) -> None:
    if (not isinstance(target, dict)
            or not isinstance(target.get("report"), dict)
            or target["report"].get("embedded_closed_target") is not True
            or not isinstance(target.get("records"), tuple)
            or not target["records"]
            or not isinstance(target.get("_interior_records"), tuple)
            or not target["_interior_records"]):
        raise human.ImportError("clearance operation requires an admissible prepared closed target")
    locator = target.get("_interior_point_locator")
    if (not isinstance(locator, ci.PreparedPointLocation)
            or locator.records != target["_interior_records"]):
        raise human.ImportError("clearance target point-location snapshot does not match its interior records")


def _closed_target_inside_vertices(positions: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    """Return captured Float32 point IDs strictly inside the selected enclosure."""
    _require_prepared_closed_target(target)
    points = np.asarray(positions)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu":
        raise human.ImportError("clearance skin positions must be a numeric Nx3 array")
    try:
        packed = np.ascontiguousarray(points, dtype="<f4")
    except (OverflowError, ValueError, TypeError) as error:
        raise human.ImportError("clearance skin positions are not representable as Float32") from error
    numeric = np.asarray(points, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isfinite(packed).all():
        raise human.ImportError("clearance skin positions must be finite")
    if not np.array_equal(numeric, packed.astype(np.float64)):
        raise human.ImportError("clearance skin positions must already be exact Float32 values")
    lower = np.asarray(target["_inside_aabb_min"], dtype=np.float64)
    upper = np.asarray(target["_inside_aabb_max"], dtype=np.float64)
    inside = []
    for point_id, point in enumerate(packed):
        numeric_point = point.astype(np.float64)
        if np.any(numeric_point < lower) or np.any(numeric_point > upper):
            continue
        result = ci.point_location_prepared(float32_point_lattice_key(point), target["_interior_point_locator"])
        location = result.get("location")
        if location == "inside":
            inside.append(point_id)
        elif location not in ("outside", "boundary"):
            raise human.ImportError(
                f"clearance target point location is indeterminate for captured skin vertex {point_id}"
            )
    return np.asarray(inside, dtype=np.int64)


def _closed_target_exit_source_scalar(
    point: np.ndarray,
    mapped_direction: np.ndarray,
    target: dict[str, Any],
    margin_m: float = 0.00025,
) -> float:
    """Find an inside-to-outside ray exit plus a world-distance margin.

    mapped_direction is the actual world displacement per unit source scalar
    (for example, a captured LBS Jacobian applied to a source direction). Ray
    intersections use its exact binary64 rational value on the target's 2**149
    Float32 lattice. The margin is converted once by its norm; direction is
    never normalized before finding the source scalar.
    """
    _require_prepared_closed_target(target)
    try:
        original_point = np.asarray(point, dtype=np.float64).reshape(3)
        packed_point = np.ascontiguousarray(original_point, dtype="<f4")
        direction = np.asarray(mapped_direction, dtype=np.float64).reshape(3)
    except (ValueError, TypeError, OverflowError) as error:
        raise human.ImportError("clearance ray requires xyz point and direction") from error
    if (not np.isfinite(original_point).all() or not np.isfinite(packed_point).all()
            or not np.array_equal(original_point, packed_point.astype(np.float64))):
        raise human.ImportError("clearance ray start must be an exact Float32 point")
    if not np.isfinite(direction).all():
        raise human.ImportError("clearance mapped direction must be finite")
    margin = float(margin_m)
    if not math.isfinite(margin) or margin <= 0.0:
        raise human.ImportError("clearance ray margin must be finite and positive")
    speed = math.hypot(*(float(component) for component in direction))
    if not math.isfinite(speed) or speed <= 0.0:
        raise human.ImportError("clearance mapped direction must be nonzero")
    point_lattice = float32_point_lattice_key(packed_point)
    target_records = target["_interior_records"]
    if ci.point_location_prepared(point_lattice, target["_interior_point_locator"]).get("location") != "inside":
        raise human.ImportError("clearance ray start must be strictly inside the target")

    direction_fraction = tuple(Fraction.from_float(float(value)) for value in direction)
    # Binary64 denominators are powers of two. Keep each triangle test on the
    # existing integer lattice; only accepted hit scalars need Fraction objects.
    direction_denominator = max(component.denominator for component in direction_fraction)
    direction_integer = tuple(
        component.numerator * (direction_denominator // component.denominator)
        for component in direction_fraction)
    lattice_denominator = ci._FLOAT32_LATTICE_DENOMINATOR
    hits = set()
    for record in target_records:
        triangle = record[0]
        normal = ci._cross(ci._sub(triangle[1], triangle[0]), ci._sub(triangle[2], triangle[0]))
        numerator = ci._dot(normal, ci._sub(triangle[0], point_lattice))
        denominator = ci._dot(normal, direction_integer)
        if denominator == 0:
            if numerator == 0:
                raise human.ImportError("clearance ray is coplanar with a target face")
            continue
        if numerator * denominator <= 0:
            continue
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
        hit_numerator = tuple(
            point_lattice[axis] * denominator + numerator * direction_integer[axis]
            for axis in range(3))
        if ci._inside(ci._signs(hit_numerator, denominator, triangle, normal)):
            hits.add(Fraction(numerator * direction_denominator,
                              denominator * lattice_denominator))
    if not hits:
        raise human.ImportError("clearance ray found no positive target boundary crossing")

    margin_source_scalar = margin / speed
    for hit_scalar in sorted(hits):
        candidate_scalar = float(hit_scalar) + margin_source_scalar
        if not math.isfinite(candidate_scalar) or candidate_scalar <= 0.0:
            raise human.ImportError("clearance ray exit scalar is not finite and positive")
        endpoint = (packed_point.astype(np.float64) + candidate_scalar * direction).astype("<f4")
        endpoint_location = ci.point_location_prepared(
            float32_point_lattice_key(endpoint), target["_interior_point_locator"]
        ).get("location")
        if endpoint_location == "outside":
            return candidate_scalar
        if endpoint_location not in ("inside", "boundary"):
            raise human.ImportError("clearance ray margin endpoint has indeterminate target location")
    raise human.ImportError("clearance ray margin did not reach a proven outside Float32 endpoint")


def _float32_skin_aabb_index(skin_records):
    """Build a float-coordinate AABB tree from exact Float32 skin records.

    Exact records use the common 2**149 integer lattice. Every skin bound came
    from a Float32 coordinate, so scaling it back by 2**-149 is exactly
    representable as binary64 and preserves closed-box comparisons.
    """
    boxes = []
    for record in skin_records:
        lower = tuple(math.ldexp(float(value), -149) for value in record[1])
        upper = tuple(math.ldexp(float(value), -149) for value in record[2])
        for original, restored in zip((*record[1], *record[2]), (*lower, *upper)):
            if not math.isfinite(restored) or float(np.float32(restored)) != restored:
                raise human.ImportError("skin AABB bound is not an exact finite Float32 coordinate")
            if float32_point_lattice_key((restored, 0.0, 0.0))[0] != original:
                raise human.ImportError("skin AABB bound did not round-trip from the exact Float32 lattice")
        boxes.append((None, lower, upper, int(record[3]), ()))
    return _prepare_surface_aabb(boxes)


def _candidate_target_rows_for_skin_aabb(
    faces: np.ndarray,
    canonical_positions: np.ndarray,
    skin_float_index,
    *,
    chunk_size: int = 65_536,
) -> tuple[list[int], int]:
    """Select target rows whose closed Float32 boxes overlap a skin box.

    The union-box test is a cheap conservative first pass. The existing exact
    AABB tree then yields the precise candidate set and pair count; exact
    integer records are constructed only for selected target rows.
    """
    if skin_float_index.root is None:
        return [], 0
    raw_faces = np.asarray(faces)
    # The caller canonicalizes and finite-checks the complete target pool once
    # before entering this per-surface loop. Rechecking it here would rescan
    # millions of vertices for each of the 859 target surfaces.
    positions = np.asarray(canonical_positions)
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu"
            or not len(raw_faces) or raw_faces.min() < 0 or raw_faces.max() >= len(positions)):
        raise human.ImportError("target broadphase faces are malformed or outside the vertex inventory")
    if (positions.ndim != 2 or positions.shape[1] != 3 or positions.dtype.kind != "f"
            or positions.dtype.itemsize != 4 or not positions.flags.c_contiguous):
        raise human.ImportError("target broadphase positions must be prevalidated contiguous Float32 xyz rows")

    union_lower, union_upper = skin_float_index.root[:2]
    union_lower_array = np.asarray(union_lower, dtype=np.float64)
    union_upper_array = np.asarray(union_upper, dtype=np.float64)
    candidate_rows: set[int] = set()
    aabb_candidate_pairs = 0
    for start in range(0, len(raw_faces), chunk_size):
        stop = min(start + chunk_size, len(raw_faces))
        triangles = np.asarray(positions[raw_faces[start:stop]], dtype=np.float64)
        lower = triangles.min(axis=1)
        upper = triangles.max(axis=1)
        union_overlap = np.all((upper >= union_lower_array) & (lower <= union_upper_array), axis=1)
        local_rows = np.flatnonzero(union_overlap)
        queries = (
            (None, tuple(map(float, lower[local])), tuple(map(float, upper[local])), start + int(local), ())
            for local in local_rows
        )
        for target_row, _skin_row in ci._query_surface_aabb(queries, skin_float_index):
            candidate_rows.add(int(target_row[3]))
            aabb_candidate_pairs += 1
    return sorted(candidate_rows), aabb_candidate_pairs


def _target_intersection_audit(
    skin_records,
    target_faces: dict[tuple[int, int], np.ndarray],
    pack_positions: np.ndarray,
    *,
    _validated_target_geometry_cache: set[str] | None = None,
):
    """Audit every target, optionally reusing a prior full exact validation.

    Without the private fit-owned cache this retains the legacy full-record
    conversion and exact-degeneracy validation for every target face. A cache
    entry is added only after that complete scan succeeds and is keyed by the
    full target topology/Float32 geometry digest. Subsequent calls with the
    same digest use closed Float32 AABBs to select target rows before exact
    conversion; every selected pair still goes through the existing integer
    predicate and exact AABB tree. A digest miss always returns to the legacy
    full scan, so a non-overlapping degenerate or changed target cannot skip
    validation.
    """
    cache = _validated_target_geometry_cache
    if cache is not None and not isinstance(cache, set):
        raise human.ImportError("target geometry validation cache must be a fit-owned set")
    target_geometry_sha = (
        _target_geometry_f32_sha256(pack_positions, target_faces) if cache is not None else None
    )
    fast_path = cache is not None and target_geometry_sha in cache
    if fast_path:
        # The digest is over canonical Float32 coordinates. Preserve the exact
        # record constructor's rejection of non-Float32 source values even if
        # they happen to quantize to an already validated digest.
        raw_positions = np.asarray(pack_positions)
        packed_positions = _float32_xyz_rows(pack_positions, 'target world positions')
        if raw_positions.dtype.kind != 'f' or raw_positions.dtype.itemsize != 4:
            referenced = np.unique(np.concatenate([
                np.asarray(target_faces[key], dtype=np.int64).reshape(-1) for key in sorted(target_faces)
            ]))
            if not np.array_equal(raw_positions[referenced], packed_positions[referenced]):
                raise human.ImportError('cached target coordinates are not exact Float32 values')
    per_surface = {}
    skin_index = _prepare_surface_aabb(skin_records)
    skin_float_index = _float32_skin_aabb_index(skin_records) if fast_path else None
    for key in sorted(target_faces):
        faces = np.asarray(target_faces[key])
        if fast_path:
            candidate_rows, broadphase_aabb_count = _candidate_target_rows_for_skin_aabb(
                faces, packed_positions, skin_float_index,
            )
            selected_faces = faces[candidate_rows]
            unique = np.unique(selected_faces)
            compact_faces = np.searchsorted(unique, selected_faces)
            compact_records = _exact_surface_records(packed_positions[unique], compact_faces)
            target_records = [
                (row[0], row[1], row[2], int(candidate_rows[position]), row[4])
                for position, row in enumerate(compact_records)
            ]
            audit = _audit_pair_prepared_first(skin_index, target_records)
            if int(audit["aabb_candidate_pairs"]) != broadphase_aabb_count:
                raise human.ImportError(
                    "Float32 target broadphase candidate count differs from the exact integer AABB tree"
                )
        else:
            unique = np.unique(faces)
            compact_faces = np.searchsorted(unique, faces)
            target_records = _exact_surface_records(pack_positions[unique], compact_faces)
            audit = _audit_pair_prepared_first(skin_index, target_records)
        per_surface[f"{key[0]}:{key[1]}"] = {
            "triangle_pairs": audit["triangle_pairs"], "count": audit["count"],
            "aabb_candidate_pairs": audit["aabb_candidate_pairs"],
        }
    if cache is not None and not fast_path:
        # The full exact record conversion above rejected degenerates in every
        # target surface. Retain only a content identity, never the records.
        if target_geometry_sha is None:
            raise human.ImportError("full target validation did not bind a geometry digest")
        if len(cache) >= 64 and target_geometry_sha not in cache:
            cache.clear()
        cache.add(target_geometry_sha)
    return per_surface



def _float32_xyz_rows(points: np.ndarray, label: str) -> np.ndarray:
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu":
        raise human.ImportError(f"{label} must be a numeric Nx3 coordinate array")
    if not np.isfinite(raw).all():
        raise human.ImportError(f"{label} contains non-finite coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        packed = np.ascontiguousarray(raw, dtype="<f4")
    if not np.isfinite(packed).all():
        raise human.ImportError(f"{label} is non-finite after Float32 packing")
    return packed


def _float32_xyz_sha256(points: np.ndarray, label: str = "Float32 xyz") -> str:
    packed = _float32_xyz_rows(points, label)
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


def _face_index_sha256(faces: np.ndarray) -> str:
    rows = np.asarray(faces)
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in "iu" or rows.size == 0:
        raise human.ImportError("face identity requires a nonempty integer Fx3 array")
    packed = np.ascontiguousarray(rows, dtype="<i8")
    digest = hashlib.sha256(struct.pack("<QQ", len(packed), packed.shape[1]))
    digest.update(packed.tobytes(order="C"))
    return digest.hexdigest()


def _baseline_target_pair_table_sha256(baseline_target_audits: dict[str, dict[str, Any]]) -> str:
    if not isinstance(baseline_target_audits, dict) or not baseline_target_audits:
        raise human.ImportError("baseline target pair identity requires complete target audits")
    canonical_rows = []
    for key in sorted(baseline_target_audits):
        row = baseline_target_audits[key]
        if not isinstance(key, str) or not isinstance(row, dict):
            raise human.ImportError("baseline target pair identity has malformed target rows")
        raw_pairs = row.get("triangle_pairs")
        if not isinstance(raw_pairs, (list, tuple)):
            raise human.ImportError("baseline target pair identity omits exact pairs")
        pairs = []
        for pair in raw_pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int for value in pair)):
                raise human.ImportError("baseline target pair identity has malformed face pairs")
            pairs.append([int(pair[0]), int(pair[1])])
        pairs.sort()
        if (len({tuple(pair) for pair in pairs}) != len(pairs)
                or type(row.get("count")) is not int or row["count"] != len(pairs)):
            raise human.ImportError("baseline target pair identity count or uniqueness mismatch")
        canonical_rows.append([key, pairs])
    encoded = json.dumps(canonical_rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _target_geometry_f32_sha256(
    target_positions: np.ndarray,
    target_faces_by_key: dict[tuple[int, int], np.ndarray],
) -> str:
    xyz = _float32_xyz_rows(target_positions, "target world positions")
    if not isinstance(target_faces_by_key, dict) or not target_faces_by_key:
        raise human.ImportError("target geometry identity requires complete target-face coverage")
    digest = hashlib.sha256(struct.pack("<Q", len(target_faces_by_key)))
    seen = set()
    for key in sorted(target_faces_by_key):
        if (not isinstance(key, tuple) or len(key) != 2
                or any(type(value) is not int or value < 0 for value in key)):
            raise human.ImportError("target geometry identity has a malformed semantic/stable key")
        if key in seen:
            raise human.ImportError("target geometry identity has duplicate surface keys")
        seen.add(key)
        raw = np.asarray(target_faces_by_key[key])
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu" or raw.size == 0:
            raise human.ImportError("target geometry identity has malformed face rows")
        if raw.min() < 0 or raw.max() >= len(xyz):
            raise human.ImportError("target geometry identity face is outside the vertex inventory")
        faces = np.ascontiguousarray(raw, dtype="<i8")
        digest.update(struct.pack("<IIQ", key[0], key[1], len(faces)))
        digest.update(faces.tobytes(order="C"))
        digest.update(np.ascontiguousarray(xyz[faces], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def audit_incremental_skin_target_intersections(
    *,
    baseline_world_positions: np.ndarray,
    candidate_world_positions: np.ndarray,
    baseline_faces: np.ndarray,
    candidate_faces: np.ndarray,
    baseline_target_audits: dict[str, dict[str, Any]],
    target_faces_by_key: dict[tuple[int, int], np.ndarray],
    target_positions: np.ndarray,
    expected_baseline_world_f32_sha256: str,
    expected_face_index_sha256: str,
    expected_target_geometry_f32_sha256: str,
    expected_baseline_target_audits_sha256: str,
    _validated_target_geometry_cache: set[str] | None = None,
) -> dict[str, Any]:
    """Reuse exact baseline target pairs for unchanged Float32 skin faces.

    Baseline exact audits must be bound by the caller to the supplied baseline
    world-coordinate hash and target snapshot. Candidate and baseline topology
    must be row-identical. Every face whose packed Float32 xyz changes is then
    tested against every supplied target with the existing exact predicate and
    target AABB preparation; unchanged face-pair witnesses are retained verbatim.
    The reported AABB count covers only fresh queries for changed face rows.
    A caller-owned private cache may be passed to amortize full exact target
    degeneracy validation across trials; on its first digest miss, the exact
    target audit still validates every face before retaining the digest. The
    default remains uncached and preserves legacy full validation per call.
    """
    baseline = _float32_xyz_rows(baseline_world_positions, "baseline skin world positions")
    candidate = _float32_xyz_rows(candidate_world_positions, "candidate skin world positions")
    target_xyz = _float32_xyz_rows(target_positions, "target world positions")
    if baseline.shape != candidate.shape:
        raise human.ImportError("incremental target audit baseline/candidate vertex coverage differs")
    if (not isinstance(expected_baseline_world_f32_sha256, str)
            or len(expected_baseline_world_f32_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_baseline_world_f32_sha256)):
        raise human.ImportError("incremental target audit requires a lowercase baseline Float32 identity hash")
    baseline_sha = hashlib.sha256(baseline.tobytes(order="C")).hexdigest()
    if baseline_sha != expected_baseline_world_f32_sha256:
        raise human.ImportError("incremental target audit baseline world identity does not match its pin")

    def checked_faces(value, vertex_count: int, label: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu" or raw.size == 0:
            raise human.ImportError(f"{label} must be a nonempty integer Fx3 face array")
        if raw.min() < 0 or raw.max() >= vertex_count:
            raise human.ImportError(f"{label} references a vertex outside the declared inventory")
        return np.ascontiguousarray(raw, dtype="<i8")

    base_faces = checked_faces(baseline_faces, len(baseline), "baseline skin faces")
    next_faces = checked_faces(candidate_faces, len(candidate), "candidate skin faces")
    if base_faces.shape != next_faces.shape or not np.array_equal(base_faces, next_faces):
        raise human.ImportError("incremental target audit refuses changed face topology or row identity")
    base_face_sha = hashlib.sha256(struct.pack("<QQ", len(base_faces), base_faces.shape[1]))
    base_face_sha.update(base_faces.tobytes(order="C"))
    base_face_sha = base_face_sha.hexdigest()
    if (not isinstance(expected_face_index_sha256, str)
            or len(expected_face_index_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_face_index_sha256)
            or base_face_sha != expected_face_index_sha256):
        raise human.ImportError("incremental target audit baseline face identity does not match its pin")
    # The exact record constructor also fails closed on Float32-degenerate faces.
    _exact_surface_records(baseline, base_faces)
    _exact_surface_records(candidate, next_faces)

    if not isinstance(target_faces_by_key, dict) or not target_faces_by_key:
        raise human.ImportError("incremental target audit requires complete target-face coverage")
    target_faces = {}
    target_key_strings = set()
    for key, face_rows in target_faces_by_key.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or any(type(value) is not int or value < 0 for value in key)):
            raise human.ImportError("incremental target audit has a malformed semantic/stable target key")
        key_string = f"{key[0]}:{key[1]}"
        if key_string in target_key_strings:
            raise human.ImportError("incremental target audit has duplicate canonical target keys")
        target_key_strings.add(key_string)
        target_faces[key] = checked_faces(face_rows, len(target_xyz), f"target faces {key_string}")
    if not isinstance(baseline_target_audits, dict) or set(baseline_target_audits) != target_key_strings:
        raise human.ImportError("incremental target audit baseline target coverage differs from the full target inventory")
    target_geometry_sha = _target_geometry_f32_sha256(target_xyz, target_faces)
    if (not isinstance(expected_target_geometry_f32_sha256, str)
            or len(expected_target_geometry_f32_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_target_geometry_f32_sha256)
            or target_geometry_sha != expected_target_geometry_f32_sha256):
        raise human.ImportError("incremental target audit target geometry does not match its baseline pin")

    baseline_pairs: dict[str, set[tuple[int, int]]] = {}
    target_face_counts = {f"{key[0]}:{key[1]}": len(rows) for key, rows in target_faces.items()}
    for key_string, row in baseline_target_audits.items():
        if not isinstance(row, dict) or not isinstance(row.get("triangle_pairs"), (list, tuple)):
            raise human.ImportError(f"baseline target audit is malformed for {key_string}")
        pairs = []
        for pair in row["triangle_pairs"]:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int for value in pair)):
                raise human.ImportError(f"baseline target audit has a malformed face pair for {key_string}")
            face_row, target_row = pair
            if not (0 <= face_row < len(base_faces) and 0 <= target_row < target_face_counts[key_string]):
                raise human.ImportError(f"baseline target audit face pair is outside coverage for {key_string}")
            pairs.append((face_row, target_row))
        pair_set = set(pairs)
        if (len(pair_set) != len(pairs) or type(row.get("count")) is not int
                or row["count"] != len(pairs) or row.get("degenerate_face_rows")):
            raise human.ImportError(f"baseline target audit count, uniqueness, or degeneracy check failed for {key_string}")
        baseline_pairs[key_string] = pair_set
    baseline_pair_table_sha = _baseline_target_pair_table_sha256(baseline_target_audits)
    if (not isinstance(expected_baseline_target_audits_sha256, str)
            or len(expected_baseline_target_audits_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_baseline_target_audits_sha256)
            or baseline_pair_table_sha != expected_baseline_target_audits_sha256):
        raise human.ImportError("incremental target audit baseline exact-pair table does not match its pin")

    baseline_bits = baseline.view("<u4").reshape(baseline.shape)
    candidate_bits = candidate.view("<u4").reshape(candidate.shape)
    face_bits_differ = np.any(baseline_bits[base_faces] != candidate_bits[next_faces], axis=(1, 2))
    changed_face_rows = np.flatnonzero(face_bits_differ).astype(np.int64)
    changed_set = set(map(int, changed_face_rows))
    changed_records = _exact_surface_records(candidate, next_faces[changed_face_rows])
    fresh = _target_intersection_audit(
        changed_records, target_faces, target_xyz,
        _validated_target_geometry_cache=_validated_target_geometry_cache,
    )
    if set(fresh) != target_key_strings:
        raise human.ImportError("incremental target audit changed target coverage during exact changed-face scan")

    target_audits: dict[str, dict[str, Any]] = {}
    change_sets: dict[str, dict[str, list[list[int]]]] = {}
    reused_total = fresh_aabb_total = fresh_pair_total = 0
    for key_string in sorted(target_key_strings):
        old = baseline_pairs[key_string]
        reused = {pair for pair in old if pair[0] not in changed_set}
        fresh_row = fresh[key_string]
        local_pairs = _triangle_pair_set(fresh_row)
        if int(fresh_row.get("count", -1)) != len(local_pairs):
            raise human.ImportError(f"fresh changed-face target scan is malformed for {key_string}")
        remapped = set()
        for local_face, target_face in local_pairs:
            if not 0 <= local_face < len(changed_face_rows):
                raise human.ImportError(f"fresh target scan returned an invalid changed-face row for {key_string}")
            remapped.add((int(changed_face_rows[local_face]), int(target_face)))
        if len(remapped) != len(local_pairs):
            raise human.ImportError(f"fresh changed-face target scan did not map uniquely for {key_string}")
        combined = reused | remapped
        if len(combined) != len(reused) + len(remapped):
            raise human.ImportError("incremental target audit duplicated a pair across changed and reused faces")
        added = combined - old
        removed = old - combined
        unchanged = combined & old
        pairs_sorted = sorted(combined)
        aabb_count = int(fresh_row.get("aabb_candidate_pairs", -1))
        if aabb_count < len(local_pairs):
            raise human.ImportError(f"fresh changed-face AABB work count is malformed for {key_string}")
        target_audits[key_string] = {
            "triangle_pairs": [[int(a), int(b)] for a, b in pairs_sorted],
            "count": len(pairs_sorted),
            "aabb_candidate_pairs": aabb_count,
            "reused_baseline_exact_pair_count": len(reused),
            "fresh_changed_face_exact_pair_count": len(remapped),
            "fresh_changed_face_aabb_candidate_pairs": aabb_count,
            "aabb_work_scope": "changed_skin_faces_only",
        }
        change_sets[key_string] = {
            "added": [[int(a), int(b)] for a, b in sorted(added)],
            "removed": [[int(a), int(b)] for a, b in sorted(removed)],
            "unchanged": [[int(a), int(b)] for a, b in sorted(unchanged)],
        }
        reused_total += len(reused)
        fresh_pair_total += len(remapped)
        fresh_aabb_total += aabb_count

    return {
        "schema": "numi.human.skin-incremental-exact-target-audit.v1",
        "baseline_world_f32_sha256": baseline_sha,
        "candidate_world_f32_sha256": hashlib.sha256(candidate.tobytes(order="C")).hexdigest(),
        "face_index_sha256": base_face_sha,
        "target_geometry_f32_sha256": target_geometry_sha,
        "baseline_target_audits_sha256": baseline_pair_table_sha,
        "skin_face_count": int(len(base_faces)),
        "target_surface_count": len(target_key_strings),
        "changed_skin_face_rows": [int(value) for value in changed_face_rows],
        "unchanged_skin_face_count": int(len(base_faces) - len(changed_face_rows)),
        "reused_baseline_exact_pair_count": reused_total,
        "fresh_changed_face_exact_pair_count": fresh_pair_total,
        "fresh_changed_face_aabb_candidate_pairs": fresh_aabb_total,
        "aabb_work_scope": "new exact AABB work for changed skin faces against every supplied target; unchanged baseline pairs are reused without AABB retest",
        "target_audits": target_audits,
        "pair_changes_by_target": change_sets,
    }


def _checked_baseline_self_pair_rows(
    baseline_self_audit: dict[str, Any], face_count: int,
) -> list[tuple[int, int]]:
    """Validate the structural contract of a caller-verified exact self audit."""
    if not isinstance(baseline_self_audit, dict):
        raise human.ImportError("incremental self audit baseline result must be an exact owner audit mapping")
    raw_pairs = baseline_self_audit.get("triangle_pairs")
    if not isinstance(raw_pairs, (list, tuple)):
        raise human.ImportError("incremental self audit baseline result omits its exact pair rows")
    pairs: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise human.ImportError("incremental self audit baseline has a malformed face pair")
        first, second = pair
        if not (0 <= first < second < face_count):
            raise human.ImportError("incremental self audit baseline face pair is noncanonical or outside coverage")
        pairs.append((first, second))
    if (len(set(pairs)) != len(pairs) or pairs != sorted(pairs)
            or type(baseline_self_audit.get("count")) is not int
            or baseline_self_audit["count"] != len(pairs)):
        raise human.ImportError("incremental self audit baseline count, order, or uniqueness is invalid")
    if baseline_self_audit.get("degenerate_face_rows"):
        raise human.ImportError("incremental self audit baseline contains degenerate faces")
    for key in ("aabb_candidate_pairs", "allowed_shared_vertex_or_edge_pairs"):
        value = baseline_self_audit.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise human.ImportError(f"incremental self audit baseline {key} is malformed")
    return pairs


def _baseline_self_pair_table_sha256(
    baseline_self_audit: dict[str, Any], face_count: int,
) -> str:
    pairs = _checked_baseline_self_pair_rows(baseline_self_audit, face_count)
    digest = hashlib.sha256(b"HumanIncrementalExactSelfPairTable.v1\x00")
    digest.update(struct.pack("<QQ", face_count, len(pairs)))
    for first, second in pairs:
        digest.update(struct.pack("<QQ", first, second))
    return digest.hexdigest()


def audit_incremental_skin_self_intersections(
    *,
    baseline_world_positions: np.ndarray,
    candidate_world_positions: np.ndarray,
    baseline_faces: np.ndarray,
    candidate_faces: np.ndarray,
    baseline_self_audit: dict[str, Any],
    expected_baseline_world_f32_sha256: str,
    expected_face_index_sha256: str,
    expected_baseline_self_pair_table_sha256: str,
) -> dict[str, Any]:
    """Reuse exact baseline self-pairs untouched by changed faces.

    The caller must first obtain and provenance-bind ``baseline_self_audit``
    from the exact same-surface owner for the supplied baseline Float32 xyz and
    face-row table. This function validates its complete, canonical pair table
    and the three supplied identity pins, but does not repeat the full baseline
    narrow phase. Candidate topology and face-row order must be identical.

    Every face incident to a bitwise-changed Float32 vertex is queried against
    the full candidate surface with the existing exact AABB index. Unordered
    face-row pairs are canonicalized before the existing exact predicate is
    called, so changed-to-changed pairs are tested once and the complete
    candidate exact pair set is the union of untouched baseline pairs and
    freshly audited changed-face pairs. AABB and shared-adjacency counts in
    the result describe only this fresh changed-face work.
    """
    baseline = _float32_xyz_rows(baseline_world_positions, "baseline skin world positions")
    candidate = _float32_xyz_rows(candidate_world_positions, "candidate skin world positions")
    if baseline.shape != candidate.shape:
        raise human.ImportError("incremental self audit baseline/candidate vertex coverage differs")
    if not len(baseline):
        raise human.ImportError("incremental self audit requires a nonempty shared vertex table")

    def checked_faces(value, vertex_count: int, label: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu" or raw.size == 0:
            raise human.ImportError(f"{label} must be a nonempty integer Fx3 face array")
        if raw.min() < 0 or raw.max() >= vertex_count:
            raise human.ImportError(f"{label} references a vertex outside the declared inventory")
        return np.ascontiguousarray(raw, dtype="<i8")

    base_faces = checked_faces(baseline_faces, len(baseline), "baseline skin faces")
    next_faces = checked_faces(candidate_faces, len(candidate), "candidate skin faces")
    if base_faces.shape != next_faces.shape or not np.array_equal(base_faces, next_faces):
        raise human.ImportError("incremental self audit refuses changed face topology or row identity")

    baseline_sha = hashlib.sha256(baseline.tobytes(order="C")).hexdigest()
    if (not isinstance(expected_baseline_world_f32_sha256, str)
            or len(expected_baseline_world_f32_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_baseline_world_f32_sha256)
            or baseline_sha != expected_baseline_world_f32_sha256):
        raise human.ImportError("incremental self audit baseline world identity does not match its pin")
    face_sha = _face_index_sha256(base_faces)
    if (not isinstance(expected_face_index_sha256, str)
            or len(expected_face_index_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_face_index_sha256)
            or face_sha != expected_face_index_sha256):
        raise human.ImportError("incremental self audit face identity does not match its pin")

    _exact_surface_records(baseline, base_faces)
    candidate_records = _exact_surface_records(candidate, next_faces)
    baseline_pairs = _checked_baseline_self_pair_rows(baseline_self_audit, len(base_faces))
    baseline_pairs_sha = _baseline_self_pair_table_sha256(baseline_self_audit, len(base_faces))
    if (not isinstance(expected_baseline_self_pair_table_sha256, str)
            or len(expected_baseline_self_pair_table_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_baseline_self_pair_table_sha256)
            or baseline_pairs_sha != expected_baseline_self_pair_table_sha256):
        raise human.ImportError("incremental self audit baseline exact-pair table does not match its pin")

    baseline_bits = baseline.view("<u4").reshape(baseline.shape)
    candidate_bits = candidate.view("<u4").reshape(candidate.shape)
    changed_vertex_mask = np.any(baseline_bits != candidate_bits, axis=1)
    changed_vertex_rows = np.flatnonzero(changed_vertex_mask).astype(np.int64)
    changed_face_rows = np.flatnonzero(np.any(changed_vertex_mask[base_faces], axis=1)).astype(np.int64)
    changed_set = set(map(int, changed_face_rows))

    full_index = ci._prepare_surface_aabb(candidate_records)
    changed_records = [candidate_records[int(row)] for row in changed_face_rows]
    canonical_candidates: dict[tuple[int, int], tuple] = {}
    raw_occurrences = 0
    for changed_record, full_record in ci._query_surface_aabb(changed_records, full_index):
        first, second = int(changed_record[3]), int(full_record[3])
        if first == second:
            continue
        raw_occurrences += 1
        key = (min(first, second), max(first, second))
        canonical_candidates.setdefault(key, (candidate_records[key[0]], candidate_records[key[1]]))
    fresh = ci._audit_record_pairs(
        (canonical_candidates[key] for key in sorted(canonical_candidates)), same_surface=True,
    )
    fresh_pairs = [tuple(map(int, pair)) for pair in fresh["triangle_pairs"]]
    if (fresh["count"] != len(fresh_pairs) or len(set(fresh_pairs)) != len(fresh_pairs)
            or fresh_pairs != sorted(fresh_pairs)
            or any(first not in changed_set and second not in changed_set for first, second in fresh_pairs)):
        raise human.ImportError("incremental self audit fresh exact pair table is malformed")

    reused_pairs = [pair for pair in baseline_pairs if pair[0] not in changed_set and pair[1] not in changed_set]
    reused_set = set(reused_pairs)
    fresh_set = set(fresh_pairs)
    if reused_set & fresh_set:
        raise human.ImportError("incremental self audit duplicated a pair across reused and fresh work")
    candidate_pair_set = reused_set | fresh_set
    candidate_pairs = sorted(candidate_pair_set)
    baseline_set = set(baseline_pairs)

    return {
        "schema": "numi.human.skin-incremental-exact-self-audit.v1",
        "baseline_world_f32_sha256": baseline_sha,
        "candidate_world_f32_sha256": hashlib.sha256(candidate.tobytes(order="C")).hexdigest(),
        "face_index_sha256": face_sha,
        "baseline_self_pair_table_sha256": baseline_pairs_sha,
        "skin_face_count": int(len(base_faces)),
        "changed_skin_vertex_rows": [int(value) for value in changed_vertex_rows],
        "changed_skin_face_rows": [int(value) for value in changed_face_rows],
        "unchanged_skin_face_count": int(len(base_faces) - len(changed_face_rows)),
        "triangle_pairs": [[int(first), int(second)] for first, second in candidate_pairs],
        "count": len(candidate_pairs),
        "reused_baseline_exact_pair_count": len(reused_pairs),
        "fresh_changed_face_exact_pair_count": len(fresh_pairs),
        "fresh_changed_face_aabb_candidate_pairs": int(fresh["aabb_candidate_pairs"]),
        "fresh_changed_face_allowed_shared_vertex_or_edge_pair_count": int(
            fresh["allowed_shared_vertex_or_edge_pairs"]
        ),
        "fresh_changed_face_aabb_raw_occurrences_before_dedup": raw_occurrences,
        "added_exact_pair_rows": [[int(a), int(b)] for a, b in sorted(candidate_pair_set - baseline_set)],
        "removed_exact_pair_rows": [[int(a), int(b)] for a, b in sorted(baseline_set - candidate_pair_set)],
        "unchanged_exact_pair_rows": [[int(a), int(b)] for a, b in sorted(candidate_pair_set & baseline_set)],
        "aabb_work_scope": "fresh changed skin faces against the full candidate surface; untouched baseline exact pairs are reused",
        "baseline_audit_binding": "caller-validated exact same-surface owner audit; the supplied world, face, and pair-table pins were checked",
    }




_ACCEPTED_SELF_AUDIT_ROW_SCHEMA = "numi.human.accepted-skin-self-audit-row.v1"


def _module_source_sha256(module: Any, label: str) -> str:
    source_path = getattr(module, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        raise human.ImportError(f"{label} source path is unavailable for self-audit identity")
    path = Path(source_path).resolve()
    if not path.is_file():
        raise human.ImportError(f"{label} source file is unavailable for self-audit identity")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_skin_self_audits_sha256(rows: Any) -> str:
    try:
        packed = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise human.ImportError("accepted self-audit rows are not canonical JSON data") from error
    return hashlib.sha256(packed).hexdigest()


def _serialize_accepted_skin_self_audits(
    audits: Any, world_positions_by_pose: np.ndarray, source_positions_f32_sha256: str,
    faces: np.ndarray,
) -> list[dict[str, Any]]:
    """Bind already-computed complete zero-self rows for an accepted trial event."""
    worlds = np.asarray(world_positions_by_pose)
    face_rows = np.asarray(faces)
    if worlds.ndim != 3 or worlds.shape[2] != 3 or not isinstance(audits, (list, tuple)):
        raise human.ImportError("accepted self-audit event needs one world and audit row per pose")
    if len(audits) != len(worlds) or not len(worlds):
        raise human.ImportError("accepted self-audit event has incomplete pose coverage")
    if (not isinstance(source_positions_f32_sha256, str) or len(source_positions_f32_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in source_positions_f32_sha256)):
        raise human.ImportError("accepted self-audit event has an invalid source-position identity")
    face_sha = _face_index_sha256(face_rows)
    owner_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    predicate_sha = _module_source_sha256(ci, "exact self-intersection predicate")
    result = []
    for pose, audit in enumerate(audits):
        if not isinstance(audit, dict):
            raise human.ImportError(f"accepted self-audit row is malformed at pose={pose}")
        pairs = audit.get("triangle_pairs")
        if not isinstance(pairs, (list, tuple)):
            raise human.ImportError(f"accepted self-audit pair table is absent at pose={pose}")
        normalized_pairs = []
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int for value in pair)):
                raise human.ImportError(f"accepted self-audit pair row is malformed at pose={pose}")
            normalized_pairs.append([pair[0], pair[1]])
        count = audit.get("count")
        if type(count) is not int or count != len(normalized_pairs) or count != 0:
            raise human.ImportError(f"accepted trial self audit is not exact-zero at pose={pose}")
        raw_degenerates = audit.get("degenerate_face_rows", [])
        if not isinstance(raw_degenerates, (list, tuple)) or raw_degenerates:
            raise human.ImportError(f"accepted trial self audit has degenerate faces at pose={pose}")
        normalized = {"triangle_pairs": normalized_pairs, "count": count, "degenerate_face_rows": []}
        pair_sha = _baseline_self_pair_table_sha256(normalized, len(face_rows))
        result.append({
            "schema": _ACCEPTED_SELF_AUDIT_ROW_SCHEMA,
            "pose_index": int(pose),
            "source_positions_f32_sha256": source_positions_f32_sha256,
            "candidate_world_f32_sha256": _float32_xyz_sha256(worlds[pose]),
            "face_index_sha256": face_sha,
            "skin_face_count": int(len(face_rows)),
            "triangle_pairs": normalized_pairs,
            "count": count,
            "degenerate_face_rows": [],
            "self_pair_table_sha256": pair_sha,
            "clearance_owner_source_sha256": owner_sha,
            "exact_predicate_source_sha256": predicate_sha,
            "complete_exact_pair_table": True,
            "same_surface": True,
        })
    return result


def _validate_resumed_skin_self_audits(
    rows: Any, *, pose_count: int, resume_world: np.ndarray, source_sha256: str,
    faces: np.ndarray,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(rows, (list, tuple)) or len(rows) != pose_count:
        raise human.ImportError("resume self-audit proof does not cover every accepted pose")
    proof_sha = _accepted_skin_self_audits_sha256(rows)
    face_sha = _face_index_sha256(faces)
    owner_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    predicate_sha = _module_source_sha256(ci, "exact self-intersection predicate")
    required = {
        "schema", "pose_index", "source_positions_f32_sha256", "candidate_world_f32_sha256",
        "face_index_sha256", "skin_face_count", "triangle_pairs", "count", "degenerate_face_rows",
        "self_pair_table_sha256", "clearance_owner_source_sha256", "exact_predicate_source_sha256",
        "complete_exact_pair_table", "same_surface",
    }
    validated = []
    for pose, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise human.ImportError(f"resume self-audit proof row is malformed at pose={pose}")
        if (row["schema"] != _ACCEPTED_SELF_AUDIT_ROW_SCHEMA
                or type(row["pose_index"]) is not int or row["pose_index"] != pose
                or row["source_positions_f32_sha256"] != source_sha256
                or row["candidate_world_f32_sha256"] != _float32_xyz_sha256(resume_world[pose])
                or row["face_index_sha256"] != face_sha
                or type(row["skin_face_count"]) is not int or row["skin_face_count"] != len(faces)
                or row["clearance_owner_source_sha256"] != owner_sha
                or row["exact_predicate_source_sha256"] != predicate_sha
                or row["complete_exact_pair_table"] is not True
                or row["same_surface"] is not True):
            raise human.ImportError(f"resume self-audit proof identity mismatch at pose={pose}")
        if type(row["count"]) is not int or row["count"] != 0:
            raise human.ImportError(f"resume self-audit proof is not zero-pair at pose={pose}")
        if row["degenerate_face_rows"] != []:
            raise human.ImportError(f"resume self-audit proof has degenerate faces at pose={pose}")
        normalized = {
            "triangle_pairs": row["triangle_pairs"],
            "count": row["count"],
            "degenerate_face_rows": row["degenerate_face_rows"],
        }
        actual_table_sha = _baseline_self_pair_table_sha256(normalized, len(faces))
        if row["self_pair_table_sha256"] != actual_table_sha:
            raise human.ImportError(f"resume self-audit pair-table hash mismatch at pose={pose}")
        _checked_baseline_self_pair_rows(normalized, len(faces))
        validated.append(normalized)
    return validated, proof_sha

def _sum_target_pairs(audit: dict[str, Any]) -> int:
    return sum(int(row["count"]) for row in audit.values())


def _triangle_pair_set(audit_row: dict[str, Any]) -> set[tuple[int, int]]:
    return {tuple(map(int, pair)) for pair in audit_row["triangle_pairs"]}


def _smooth_vertex_directions(vertex_normals: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Use the existing Human skin smoother on current candidate normals.

    The smoother chooses the direction of the next bounded correction. Source
    winding and accepted-to-candidate face orientation remain separate checks.
    """
    normals = np.asarray(vertex_normals, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if normals.ndim != 2 or normals.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise human.ImportError("direction smoothing requires Nx3 normals and Fx3 mesh faces")
    if not np.isfinite(normals).all() or not np.isfinite(triangles).all():
        raise human.ImportError("direction smoothing received non-finite input")
    smoothed = np.asarray(human._bodyparts_skin_smooth_visual_normals(
        normals.tolist(), [tuple(int(value) for value in row) for row in triangles],
        iterations=_DIRECTION_SMOOTHING_ITERATIONS,
    ), dtype=np.float64)
    if smoothed.shape != normals.shape or not np.isfinite(smoothed).all():
        raise human.ImportError("Human skin normal smoothing returned malformed extension directions")
    lengths = np.linalg.norm(smoothed, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise human.ImportError("Human skin normal smoothing became singular")
    smoothed /= lengths[:, None]
    if np.any(np.einsum("ij,ij->i", smoothed, normals) <= 0.0):
        raise human.ImportError("Human skin normal smoothing reversed a local source normal")
    return smoothed


def _apply_captured_world_delta(captured: np.ndarray, world_delta: np.ndarray) -> np.ndarray:
    """Apply inferred motion to the exact accepted float32 skin baseline.

    The recovered rigid poses are used to form a per-vertex Jacobian, but their
    sub-micrometer Kabsch residual must not perturb untouched accepted vertices.
    """
    base = np.asarray(captured, dtype=np.float64)
    delta = np.asarray(world_delta, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3 or delta.shape != base.shape:
        raise human.ImportError("captured-world displacement arrays have incompatible shape")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise human.ImportError("captured-world displacement contains non-finite values")
    candidate = (base + delta).astype("<f4").astype(np.float64)
    unchanged = np.all(delta == 0.0, axis=1)
    if not np.array_equal(candidate[unchanged], base[unchanged]):
        raise human.ImportError("zero-displacement vertices changed from the accepted float32 skin baseline")
    return candidate


def _triangle_normal_translation_to_separate(
    skin_triangle: np.ndarray, target_triangle: np.ndarray, direction: np.ndarray,
) -> float:
    """Return the first positive translation of a triangle along direction that separates two triangles.

    This is the exact continuous separating-axis interval for two finite triangles, not
    the maximum opponent-vertex distance to the skin plane. The returned distance is
    a translation parameter along the supplied unit direction.
    """
    skin = np.asarray(skin_triangle, dtype=np.float64)
    target = np.asarray(target_triangle, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction))
    if skin.shape != (3, 3) or target.shape != (3, 3) or direction_length <= 1.0e-15:
        raise human.ImportError("triangle separation received malformed geometry or a zero direction")
    direction = direction / direction_length
    skin_edges = [skin[(index + 1) % 3] - skin[index] for index in range(3)]
    target_edges = [target[(index + 1) % 3] - target[index] for index in range(3)]
    skin_normal = np.cross(skin_edges[0], skin_edges[1])
    target_normal = np.cross(target_edges[0], target_edges[1])
    axes = [
        skin_normal,
        target_normal,
        *(np.cross(skin_edge, target_edge) for skin_edge in skin_edges for target_edge in target_edges),
        *(np.cross(edge, skin_normal) for edge in skin_edges),
        *(np.cross(edge, target_normal) for edge in target_edges),
    ]
    first_exit = np.inf
    valid_axes = 0
    projection_tolerance_m = 1.0e-10
    for axis in axes:
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1.0e-14:
            continue
        axis = axis / axis_length
        skin_projection = skin @ axis
        target_projection = target @ axis
        skin_min, skin_max = float(skin_projection.min()), float(skin_projection.max())
        target_min, target_max = float(target_projection.min()), float(target_projection.max())
        if (skin_max < target_min - projection_tolerance_m
                or target_max < skin_min - projection_tolerance_m):
            raise human.ImportError("triangle separating-axis input is already disjoint before translation")
        speed = float(np.dot(axis, direction))
        if abs(speed) <= 1.0e-12:
            if (skin_max < target_min - projection_tolerance_m
                    or target_max < skin_min - projection_tolerance_m):
                raise human.ImportError("zero-direction separating axis is ambiguous for a claimed intersecting pair")
            continue
        valid_axes += 1
        upper = ((target_max - skin_min) / speed if speed > 0.0
                 else (target_min - skin_max) / speed)
        if upper < -projection_tolerance_m:
            raise human.ImportError("triangle separating-axis intervals do not contain the accepted intersection")
        first_exit = min(first_exit, max(0.0, upper))
    if valid_axes == 0 or not np.isfinite(first_exit):
        raise human.ImportError("triangle separating-axis calculation found no direction-sensitive axis")
    return float(first_exit)


_LOCAL_SELF_SEPARATION_MAX_POSES = 17
_LOCAL_SELF_SEPARATION_MAX_SOURCE_VERTICES = 1_000_000
_LOCAL_SELF_SEPARATION_MAX_SOURCE_FACES = 2_000_000
_LOCAL_SELF_SEPARATION_MAX_MOVABLE_VERTICES = 256
_LOCAL_SELF_SEPARATION_MAX_PAIR_POSE_ROWS = 512
_LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM = 1.0 / math.sqrt(3.0)
_LOCAL_SELF_SEPARATION_INCREMENT_CAP_MM = 1.0
_LOCAL_SELF_SEPARATION_GRAPH_LAMBDA = 5.0
_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM = 1.0e-7
_LOCAL_SELF_SEPARATION_OPTIMIZER_BUDGET_SECONDS = 120.0


def _exact_coordinate_tied_source_vertex_ids(source_positions: np.ndarray) -> np.ndarray:
    """Return every source row in a repeated Float32 coordinate group.

    These rows are conservatively kept outside inferred local patches so a
    normal/material seam record is never moved independently of an exact
    coordinate copy.
    """
    raw = np.asarray(source_positions)
    if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind != "f"
            or raw.dtype.itemsize != 4 or not np.isfinite(raw).all()):
        raise human.ImportError("coordinate-tie protection requires finite Float32 Nx3 source positions")
    source = np.ascontiguousarray(raw, dtype="<f4")
    if len(source) < 2:
        return np.empty(0, dtype=np.int64)
    _, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
    return np.flatnonzero(counts[inverse] > 1).astype(np.int64)


def _free_self_pair_seed_vertex_ids(
    face_vertex_ids: np.ndarray,
    protected_vertex_ids: np.ndarray,
) -> np.ndarray:
    """Return only face vertices that can seed an inferred local movement."""
    face_ids = np.asarray(face_vertex_ids)
    protected = np.asarray(protected_vertex_ids)
    if (face_ids.ndim != 1 or face_ids.dtype.kind not in "iu"
            or protected.ndim != 1 or protected.dtype.kind not in "iu"):
        raise human.ImportError("self-pair seed eligibility requires integer vertex ID rows")
    return np.setdiff1d(np.unique(face_ids.astype(np.int64)),
                        np.unique(protected.astype(np.int64)), assume_unique=True)


def _local_self_separation_rounding_reserve_mm(
    source_positions_m: np.ndarray, component_cap_mm: float,
) -> np.ndarray:
    """Bound requested-to-stored displacement error over the entire source box.

    Source and result coordinates are Float32, while addition and displacement
    measurement use binary64. Half the largest adjacent Float32 spacing over
    the reachable interval covers round-to-nearest storage. The binary64 term
    conservatively covers division, coordinate addition, subtraction, and
    conversion back to millimetres, including construction of this bound.
    """
    source = np.asarray(source_positions_m, dtype=np.float64)
    cap = float(component_cap_mm)
    if not np.isfinite(source).all() or not math.isfinite(cap) or cap <= 0.0:
        raise human.ImportError("local self-separation rounding reserve requires finite coordinates and a positive cap")
    with np.errstate(over="ignore", invalid="ignore"):
        arithmetic_mm = np.nextafter(
            8.0 * np.finfo(np.float64).eps * (np.abs(source) * 1000.0 + 2.0 * cap),
            np.inf,
        )
        # Include the binary64 evaluation error when choosing the endpoint
        # spacing; looking only at the starting Float32 value misses binades.
        endpoint_m = np.nextafter(
            np.abs(source) + cap / 1000.0 + arithmetic_mm / 1000.0, np.inf,
        )
        endpoint_f32 = endpoint_m.astype("<f4")
        up = np.nextafter(endpoint_f32, np.float32(np.inf)).astype(np.float64)
        down = np.nextafter(endpoint_f32, np.float32(-np.inf)).astype(np.float64)
        endpoint = endpoint_f32.astype(np.float64)
        spacing_m = np.maximum(up - endpoint, endpoint - down)
        reserve = np.nextafter(500.0 * spacing_m + arithmetic_mm, np.inf)
    if not np.isfinite(reserve).all() or np.any(reserve >= cap):
        raise human.ImportError(
            "local self-separation source coordinates leave no positive Float32-safe component range"
        )
    return reserve


def _propose_local_self_separation_increment(
    *,
    source_positions_m: np.ndarray,
    faces: np.ndarray,
    movable_vertex_ids: np.ndarray,
    self_patch_seed_vertex_ids: np.ndarray,
    required_target_seed_vertex_ids: np.ndarray,
    active_target_face_vertex_ids: np.ndarray,
    fixed_support_vertex_ids: np.ndarray,
    preserved_anchor_vertex_ids: np.ndarray,
    pose_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a bounded local proposal for supplied exact self-pair failures.

    A pose sample contains sorted ``vertex_ids``; matching Float32 Vx3
    ``baseline_world_positions`` and ``rejected_world_positions``; a Vx3x3
    dimensionless Jacobian mapping source metres to world metres; and integer
    ``self_pair_rows`` indexing ``faces``. Optional
    ``clearance_preservation_pair_rows`` identify pairs that are exactly clear
    both in the accepted baseline and current candidate; their accepted-baseline
    SAT gaps join the resolving rows in the same QP. Resolving and preserving
    rows must be disjoint. A resolving row may include a caller-derived
    self_pair_axis_overrides entry naming one pair and a finite world
    direction; it is used only when its accepted-baseline projection gap is
    positive. A positive interval gap along any direction is a separating
    witness, even when that direction is not a standard SAT axis. The
    constraints preserve the selected positive gap. A conservative
    quantization reserve tightens only the solver constraints;
    the original accepted gap and displacement limits remain the postchecks.

    After rounding source coordinates to Float32, this rechecks the linearized
    SAT rows and runs the exact triangle predicate on each supplied pair in its
    first-order predicted Float32 pose. It does not run the production forward
    model or full-mesh audit; the caller must rerun those gates. The caller
    must also bind each baseline pose to its accepted receipt/source identity;
    this helper checks that the supplied baseline pair rows are exactly clear,
    but does not establish their external provenance.
    """
    from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

    source_raw = np.asarray(source_positions_m)
    if (source_raw.ndim != 2 or source_raw.shape[1] != 3 or source_raw.size == 0
            or source_raw.dtype.kind != "f" or source_raw.dtype.itemsize != 4):
        raise human.ImportError("local self-separation source positions must be a nonempty Float32 Nx3 table")
    source = np.ascontiguousarray(source_raw, dtype="<f4")
    if not np.isfinite(source).all():
        raise human.ImportError("local self-separation source positions contain non-finite values")
    vertex_count = len(source)
    if vertex_count > _LOCAL_SELF_SEPARATION_MAX_SOURCE_VERTICES:
        raise human.ImportError("local self-separation source vertex count exceeds its bound")

    faces_raw = np.asarray(faces)
    if (faces_raw.ndim != 2 or faces_raw.shape[1] != 3 or faces_raw.size == 0
            or faces_raw.dtype.kind not in "iu"):
        raise human.ImportError("local self-separation faces must be a nonempty integer Fx3 table")
    if faces_raw.min() < 0 or faces_raw.max() >= vertex_count:
        raise human.ImportError("local self-separation face index is outside the source table")
    face_rows = np.ascontiguousarray(faces_raw, dtype=np.int64)
    face_count = len(face_rows)
    if face_count > _LOCAL_SELF_SEPARATION_MAX_SOURCE_FACES:
        raise human.ImportError("local self-separation face count exceeds its bound")

    def ids_checked(value: np.ndarray, label: str, *, required: bool = False) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or (required and len(raw) == 0):
            raise human.ImportError(f"local self-separation {label} must be a one-dimensional integer ID table")
        ids = np.asarray(raw, dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or (len(ids) and (ids.min() < 0 or ids.max() >= vertex_count)):
            raise human.ImportError(f"local self-separation {label} has duplicate or out-of-range IDs")
        return np.sort(ids)

    movable = ids_checked(movable_vertex_ids, "movable vertices", required=True)
    patch_seeds = ids_checked(self_patch_seed_vertex_ids, "self-patch seeds", required=True)
    target_seeds = ids_checked(required_target_seed_vertex_ids, "required target seeds")
    active_target = ids_checked(active_target_face_vertex_ids, "active target-face vertices")
    fixed_support = ids_checked(fixed_support_vertex_ids, "fixed support vertices")
    anchors = ids_checked(preserved_anchor_vertex_ids, "preserved anchors")
    if len(movable) > _LOCAL_SELF_SEPARATION_MAX_MOVABLE_VERTICES:
        raise human.ImportError("local self-separation patch exceeds its bounded vertex count")
    if not np.all(np.isin(patch_seeds, movable)):
        raise human.ImportError("local self-separation patch omits a required self-pair seed vertex")
    protected = np.unique(np.concatenate((target_seeds, active_target, fixed_support, anchors)))
    if np.intersect1d(movable, protected).size:
        raise human.ImportError("local self-separation patch overlaps a target seed, active target, support, or anchor")

    if (not isinstance(pose_samples, (list, tuple)) or not pose_samples
            or len(pose_samples) > _LOCAL_SELF_SEPARATION_MAX_POSES):
        raise human.ImportError("local self-separation requires one to seventeen failing pose samples")
    checked: list[dict[str, Any]] = []
    total_pairs = 0
    for pose_index, sample in enumerate(pose_samples):
        if not isinstance(sample, dict):
            raise human.ImportError(f"local self-separation pose {pose_index} is not a mapping")
        raw_vertex_ids = np.asarray(sample.get("vertex_ids"))
        vertex_ids = ids_checked(raw_vertex_ids, f"pose {pose_index} vertex IDs", required=True)
        if not np.array_equal(np.asarray(raw_vertex_ids, dtype=np.int64), vertex_ids):
            raise human.ImportError(f"local self-separation pose {pose_index} vertex IDs must be sorted to bind world/Jacobian rows")
        if len(vertex_ids) > _LOCAL_SELF_SEPARATION_MAX_SOURCE_VERTICES:
            raise human.ImportError(f"local self-separation pose {pose_index} vertex count exceeds its bound")
        base_raw = np.asarray(sample.get("baseline_world_positions"))
        trial_raw = np.asarray(sample.get("rejected_world_positions"))
        if (base_raw.shape != (len(vertex_ids), 3) or trial_raw.shape != base_raw.shape
                or base_raw.dtype.kind != "f" or base_raw.dtype.itemsize != 4
                or trial_raw.dtype.kind != "f" or trial_raw.dtype.itemsize != 4):
            raise human.ImportError(f"local self-separation pose {pose_index} needs matching Float32 Vx3 worlds")
        baseline = np.ascontiguousarray(base_raw, dtype="<f4")
        rejected = np.ascontiguousarray(trial_raw, dtype="<f4")
        jac = np.asarray(sample.get("jacobians"), dtype=np.float64)
        if jac.shape != (len(vertex_ids), 3, 3):
            raise human.ImportError(f"local self-separation pose {pose_index} Jacobians need shape Vx3x3")
        if not np.isfinite(baseline).all() or not np.isfinite(rejected).all() or not np.isfinite(jac).all():
            raise human.ImportError(f"local self-separation pose {pose_index} contains non-finite inputs")
        def checked_pair_rows(value: Any, label: str) -> np.ndarray:
            raw = np.asarray(value)
            if raw.size == 0:
                return np.empty((0, 2), dtype=np.int64)
            if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu":
                raise human.ImportError(f"local self-separation pose {pose_index} {label} must be an integer pair table")
            pairs = np.asarray(raw, dtype=np.int64)
            if (pairs.min() < 0 or pairs.max() >= face_count or np.any(pairs[:, 0] >= pairs[:, 1])
                    or len(np.unique(pairs, axis=0)) != len(pairs)):
                raise human.ImportError(f"local self-separation pose {pose_index} {label} are malformed")
            return pairs

        resolving_pairs = checked_pair_rows(sample.get("self_pair_rows", np.empty((0, 2), dtype=np.int64)),
                                             "self-pair rows")
        preserving_pairs = checked_pair_rows(
            sample.get("clearance_preservation_pair_rows", np.empty((0, 2), dtype=np.int64)),
            "clearance-preservation pair rows",
        )
        if not len(resolving_pairs) and not len(preserving_pairs):
            raise human.ImportError(f"local self-separation pose {pose_index} needs at least one pair row")
        if set(map(tuple, resolving_pairs)) & set(map(tuple, preserving_pairs)):
            raise human.ImportError(f"local self-separation pose {pose_index} pair is both resolving and preserved")
        raw_axis_overrides = sample.get("self_pair_axis_overrides", [])
        if not isinstance(raw_axis_overrides, (list, tuple)):
            raise human.ImportError(f"local self-separation pose {pose_index} axis overrides must be a list")
        axis_overrides: dict[tuple[int, int], np.ndarray] = {}
        resolving_set = set(map(tuple, resolving_pairs))
        for override in raw_axis_overrides:
            if not isinstance(override, dict) or set(override) != {"face_pair", "axis_world"}:
                raise human.ImportError(f"local self-separation pose {pose_index} axis override needs face_pair and axis_world")
            pair_raw = np.asarray(override["face_pair"])
            axis_raw = np.asarray(override["axis_world"], dtype=np.float64)
            if (pair_raw.shape != (2,) or pair_raw.dtype.kind not in "iu"
                    or int(pair_raw[0]) >= int(pair_raw[1])
                    or tuple(map(int, pair_raw)) not in resolving_set):
                raise human.ImportError(f"local self-separation pose {pose_index} axis override is not a resolving pair")
            pair_key = tuple(map(int, pair_raw))
            if pair_key in axis_overrides:
                raise human.ImportError(f"local self-separation pose {pose_index} duplicates a pair axis override")
            if axis_raw.shape != (3,) or not np.isfinite(axis_raw).all():
                raise human.ImportError(f"local self-separation pose {pose_index} axis override must be finite xyz")
            axis_norm = float(np.linalg.norm(axis_raw))
            if not math.isfinite(axis_norm) or axis_norm <= 1.0e-14:
                raise human.ImportError(f"local self-separation pose {pose_index} axis override is degenerate")
            axis_overrides[pair_key] = axis_raw / axis_norm
        pairs = np.concatenate((resolving_pairs, preserving_pairs), axis=0)
        pair_kinds = np.asarray(
            ["resolve"] * len(resolving_pairs) + ["preserve"] * len(preserving_pairs),
            dtype=object,
        )
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs, pair_kinds = pairs[order], pair_kinds[order]
        if len(pairs) > 128:
            raise human.ImportError(f"local self-separation pose {pose_index} exceeds its pair bound")
        pair_vertices = face_rows[pairs]
        if not np.all(np.isin(pair_vertices, vertex_ids)):
            raise human.ImportError(f"local self-separation pose {pose_index} omits a paired face vertex")
        if np.any(np.any(pair_vertices[:, 0, :, None] == pair_vertices[:, 1, None, :], axis=(1, 2))):
            raise human.ImportError(f"local self-separation pair table includes adjacent faces")
        unique_pair_faces = np.unique(pairs.reshape(-1))
        local_pair_faces = np.searchsorted(vertex_ids, face_rows[unique_pair_faces])
        if not np.array_equal(vertex_ids[local_pair_faces], face_rows[unique_pair_faces]):
            raise human.ImportError(f"local self-separation pose {pose_index} face-to-world row mapping is invalid")
        baseline_records = _exact_surface_records(baseline, local_pair_faces)
        baseline_record_by_face = {int(face): baseline_records[index]
                                   for index, face in enumerate(unique_pair_faces)}
        rejected_records = _exact_surface_records(rejected, local_pair_faces)
        exact_record_by_face = {int(face): rejected_records[index]
                                for index, face in enumerate(unique_pair_faces)}
        for pair_index, (first_face, second_face) in enumerate(pairs):
            if ci.triangle_intersection_points(baseline_record_by_face[int(first_face)][0],
                                               baseline_record_by_face[int(second_face)][0]):
                raise human.ImportError(
                    f"local self-separation pose {pose_index} accepted-baseline identity precondition failed: "
                    "a supplied pair is not exactly clear"
                )
            trial_intersects = ci.triangle_intersection_points(
                exact_record_by_face[int(first_face)][0], exact_record_by_face[int(second_face)][0],
            )
            if pair_kinds[pair_index] == "resolve" and not trial_intersects:
                raise human.ImportError(f"local self-separation pose {pose_index} supplied a pair absent from its exact rejected geometry")
            if pair_kinds[pair_index] == "preserve" and trial_intersects:
                raise human.ImportError(f"local self-separation pose {pose_index} preservation pair is not clear in its current geometry")
        total_pairs += len(pairs)
        checked.append({"vertex_ids": vertex_ids, "baseline": baseline, "rejected": rejected,
                        "jacobian": jac, "pairs": pairs, "pair_kinds": pair_kinds,
                        "axis_overrides": axis_overrides})
    if total_pairs > _LOCAL_SELF_SEPARATION_MAX_PAIR_POSE_ROWS:
        raise human.ImportError("local self-separation pair/pose rows exceed the bounded count")

    variable_index = np.full(vertex_count, -1, dtype=np.int32)
    variable_index[movable] = np.arange(len(movable), dtype=np.int32)
    nvar = 3 * len(movable)
    constraints: list[np.ndarray] = []
    rhs_rows: list[float] = []
    constraint_receipts: list[dict[str, Any]] = []
    pair_receipts: list[dict[str, Any]] = []
    rows_by_pose: list[int] = []
    for pose_index, sample in enumerate(checked):
        local_index = np.full(vertex_count, -1, dtype=np.int32)
        local_index[sample["vertex_ids"]] = np.arange(len(sample["vertex_ids"]), dtype=np.int32)
        baseline = sample["baseline"].astype(np.float64)
        rejected = sample["rejected"].astype(np.float64)
        jac = sample["jacobian"]
        start = len(constraints)
        for pair_index, (face_a, face_b) in enumerate(sample["pairs"]):
            face_a, face_b = int(face_a), int(face_b)
            pair_kind = str(sample["pair_kinds"][pair_index])
            ids_a, ids_b = face_rows[face_a], face_rows[face_b]
            local_a, local_b = local_index[ids_a], local_index[ids_b]
            tri_a, tri_b = baseline[local_a], baseline[local_b]
            trial_a, trial_b = rejected[local_a], rejected[local_b]
            trial_edges_a = np.roll(trial_a, -1, axis=0) - trial_a
            trial_edges_b = np.roll(trial_b, -1, axis=0) - trial_b

            def sat_axes(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                edges_a = np.roll(points_a, -1, axis=0) - points_a
                edges_b = np.roll(points_b, -1, axis=0) - points_b
                normal_a = np.cross(edges_a[0], edges_a[1])
                normal_b = np.cross(edges_b[0], edges_b[1])
                if min(np.linalg.norm(normal_a), np.linalg.norm(normal_b)) <= 1.0e-14:
                    raise human.ImportError("local self-separation supplied pair contains a degenerate SAT triangle")
                axes = np.asarray([
                    normal_a, normal_b,
                    *[np.cross(a, b) for a in edges_a for b in edges_b],
                    *[np.cross(edge, normal_a) for edge in edges_a],
                    *[np.cross(edge, normal_b) for edge in edges_b],
                ], dtype=np.float64)
                lengths = np.linalg.norm(axes, axis=1)
                valid = lengths > 1.0e-14
                if not np.any(valid):
                    raise human.ImportError("local self-separation pair has no valid SAT axis")
                axes = axes[valid] / lengths[valid, None]
                return axes, edges_a, edges_b

            baseline_axes, _, _ = sat_axes(tri_a, tri_b)
            if pair_kind == "preserve":
                current_axes, _, _ = sat_axes(trial_a, trial_b)
                axes = np.concatenate((baseline_axes, current_axes), axis=0)
            else:
                axes = baseline_axes
            override = sample["axis_overrides"].get((face_a, face_b)) if pair_kind == "resolve" else None
            if override is not None and float(np.max(np.abs(axes @ override), initial=0.0)) < 1.0 - 1.0e-10:
                axes = np.concatenate((axes, override[None, :]), axis=0)
            base_proj_a, base_proj_b = tri_a @ axes.T, tri_b @ axes.T
            base_gap_ab = base_proj_b.min(axis=0) - base_proj_a.max(axis=0)
            base_gap_ba = base_proj_a.min(axis=0) - base_proj_b.max(axis=0)
            current_proj_a, current_proj_b = trial_a @ axes.T, trial_b @ axes.T
            current_gap_ab = current_proj_b.min(axis=0) - current_proj_a.max(axis=0)
            current_gap_ba = current_proj_a.min(axis=0) - current_proj_b.max(axis=0)
            best_index: int | None = None
            best_floor_m = -math.inf
            best_axis = np.zeros(3, dtype=np.float64)
            best_baseline_gap_m = -math.inf
            best_current_gap_m = -math.inf
            if override is not None:
                alignment = np.abs(axes @ override)
                best_index = int(np.argmax(alignment))
                if float(alignment[best_index]) < 1.0 - 1.0e-10:
                    raise human.ImportError("resolving-pair axis override could not be matched to its appended direction")
                if base_gap_ab[best_index] >= base_gap_ba[best_index]:
                    best_axis = axes[best_index]
                    best_baseline_gap_m = float(base_gap_ab[best_index])
                    best_current_gap_m = float(current_gap_ab[best_index])
                else:
                    best_axis = -axes[best_index]
                    best_baseline_gap_m = float(base_gap_ba[best_index])
                    best_current_gap_m = float(current_gap_ba[best_index])
                if best_baseline_gap_m <= 0.0:
                    raise human.ImportError("resolving-pair axis override has no positive accepted-baseline separation")
                best_floor_m = best_baseline_gap_m
            else:
                for axis_index in range(len(axes)):
                    if base_gap_ab[axis_index] >= base_gap_ba[axis_index]:
                        axis = axes[axis_index]
                        baseline_gap_m = float(base_gap_ab[axis_index])
                        current_gap_m = float(current_gap_ab[axis_index])
                    else:
                        axis = -axes[axis_index]
                        baseline_gap_m = float(base_gap_ba[axis_index])
                        current_gap_m = float(current_gap_ba[axis_index])
                    if pair_kind == "preserve":
                        floor_m = min(baseline_gap_m, current_gap_m)
                        if baseline_gap_m > 0.0 and current_gap_m > 0.0 and floor_m > best_floor_m:
                            best_index = axis_index
                            best_floor_m = floor_m
                            best_axis = axis
                            best_baseline_gap_m = baseline_gap_m
                            best_current_gap_m = current_gap_m
                    else:
                        if baseline_gap_m > best_baseline_gap_m:
                            best_index = axis_index
                            best_floor_m = baseline_gap_m
                            best_axis = axis
                            best_baseline_gap_m = baseline_gap_m
                            best_current_gap_m = current_gap_m
            if best_index is None or not math.isfinite(best_floor_m) or best_floor_m <= 0.0:
                if pair_kind == "preserve":
                    raise human.ImportError("clearance-preservation pair has no common positive baseline/current SAT axis")
                raise human.ImportError("accepted self-pair does not have positive SAT separation")
            axis = best_axis
            gap_m = best_baseline_gap_m
            current_gap_m = best_current_gap_m
            gap_mm = best_floor_m * 1000.0
            now_a, now_b = rejected[local_a] * 1000.0, rejected[local_b] * 1000.0
            max_rhs = -math.inf
            for i, source_a in enumerate(ids_a):
                for j, source_b in enumerate(ids_b):
                    row = np.zeros(nvar, dtype=np.float64)
                    for source_id, local_row, sign in (
                        (int(source_a), int(local_a[i]), -1.0),
                        (int(source_b), int(local_b[j]), 1.0),
                    ):
                        variable = int(variable_index[source_id])
                        if variable >= 0:
                            for dim in range(3):
                                row[3 * variable + dim] += sign * float(np.dot(axis, jac[local_row, :, dim]))
                    current_mm = float(np.dot(axis, now_b[j] - now_a[i]))
                    required_mm = gap_mm - current_mm
                    max_rhs = max(max_rhs, required_mm)
                    if not np.any(row):
                        if required_mm > 1.0e-8:
                            raise human.ImportError("self-pair violates accepted SAT gap with no movable response")
                        continue
                    constraints.append(row)
                    rhs_rows.append(required_mm)
                    constraint_receipts.append({
                        "sample_index": pose_index, "constraint_kind": pair_kind, "face_pair": [face_a, face_b],
                        "source_vertex_pair": [int(source_a), int(source_b)],
                        "axis_unit_world": axis.tolist(),
                        "accepted_sat_gap_mm": gap_mm,
                        "baseline_sat_gap_mm": gap_m * 1000.0,
                        "current_sat_gap_mm": current_gap_m * 1000.0,
                        "preserved_sat_gap_floor_mm": gap_mm,
                        "required_projection_change_mm": required_mm,
                    })
            pair_receipts.append({
                "pose_index": pose_index, "constraint_kind": pair_kind, "face_pair": [face_a, face_b],
                "accepted_sat_gap_mm": gap_mm, "baseline_sat_gap_mm": gap_m * 1000.0,
                "current_sat_gap_mm": current_gap_m * 1000.0, "preserved_sat_gap_floor_mm": gap_mm,
                "sat_axis_selection": ("common_positive_baseline_current" if pair_kind == "preserve"
                                       else ("caller_supplied_positive_baseline_axis" if override is not None
                                             else "accepted_baseline")),
                "sat_axis_index": int(best_index),
                "axis_unit_world": best_axis.tolist(),
                "maximum_required_projection_change_mm": max_rhs,
            })
        rows_by_pose.append(len(constraints) - start)
    if not constraints:
        raise human.ImportError("local self-separation produced no movable SAT constraints")
    matrix = np.ascontiguousarray(np.vstack(constraints), dtype=np.float64)
    rhs = np.asarray(rhs_rows, dtype=np.float64)
    if matrix.shape != (len(rhs), nvar) or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise human.ImportError("local self-separation SAT system is malformed or non-finite")

    # Identity plus the attempt002 source-mesh graph-Laplacian penalty.
    graph_edges = np.concatenate((face_rows[:, [0, 1]], face_rows[:, [1, 2]], face_rows[:, [2, 0]]), axis=0)
    graph_edges.sort(axis=1)
    graph_edges = np.unique(graph_edges, axis=0)
    graph_edges = graph_edges[np.any(np.isin(graph_edges, movable), axis=1)]
    q_matrix = np.eye(nvar, dtype=np.float64)
    lam = _LOCAL_SELF_SEPARATION_GRAPH_LAMBDA
    for first, second in graph_edges:
        a, b = int(variable_index[int(first)]), int(variable_index[int(second)])
        for dim in range(3):
            if a >= 0:
                q_matrix[3 * a + dim, 3 * a + dim] += lam
            if b >= 0:
                q_matrix[3 * b + dim, 3 * b + dim] += lam
            if a >= 0 and b >= 0:
                q_matrix[3 * a + dim, 3 * b + dim] -= lam
                q_matrix[3 * b + dim, 3 * a + dim] -= lam
    if not np.isfinite(q_matrix).all() or float(np.linalg.eigvalsh(q_matrix).min()) <= 0.0:
        raise human.ImportError("local self-separation source graph objective is not positive definite")

    cap = _LOCAL_SELF_SEPARATION_COMPONENT_CAP_MM
    rounding_reserve_mm = _local_self_separation_rounding_reserve_mm(source[movable], cap)
    reserve_flat = rounding_reserve_mm.reshape(-1)
    solver_cap = np.nextafter(cap - reserve_flat, -np.inf)
    # A(x + e) >= rhs for every componentwise |e| <= reserve. Keep rhs
    # unchanged for the measured Float32 postcheck; do not reserve twice.
    solver_rhs = np.nextafter(rhs + np.abs(matrix) @ reserve_flat, np.inf)
    if not np.isfinite(solver_rhs).all() or np.any(solver_cap <= 0.0):
        raise human.ImportError("local self-separation Float32-safe solver bounds are invalid")
    bounds = list(zip(-solver_cap, solver_cap))
    optimizer_deadline = monotonic() + _LOCAL_SELF_SEPARATION_OPTIMIZER_BUDGET_SECONDS

    class _OptimizerWallTimeExceeded(Exception):
        pass

    def remaining_optimizer_seconds() -> float:
        remaining = optimizer_deadline - monotonic()
        if remaining <= 0.0:
            raise human.ImportError("local self-separation optimizer exceeded its 120 second wall-time budget")
        return remaining

    feasible = linprog(
        np.zeros(nvar), A_ub=-matrix, b_ub=-solver_rhs, bounds=bounds, method="highs",
        options={"time_limit": remaining_optimizer_seconds()},
    )
    if not feasible.success:
        lp_status = int(getattr(feasible, "status", -1))
        lp_message = str(getattr(feasible, "message", "no solver message"))
        if lp_status == 2:
            raise human.ImportError(
                "local self-separation constraints with a conservative Float32 reserve are infeasible within the fixed 1 mm vertex bound "
                f"(HiGHS status={lp_status}; message={lp_message})"
            )
        if lp_status == 1 and monotonic() >= optimizer_deadline:
            raise human.ImportError(
                "local self-separation LP exceeded its 120 second wall-time budget "
                f"(HiGHS status={lp_status}; message={lp_message})"
            )
        if lp_status == 1:
            raise human.ImportError(
                "local self-separation LP stopped without a feasible solution "
                f"(HiGHS status={lp_status}; message={lp_message})"
            )
        raise human.ImportError(
            "local self-separation LP solver failed without proving infeasibility "
            f"(HiGHS status={lp_status}; message={lp_message})"
        )

    def check_optimizer_deadline(_x: np.ndarray) -> None:
        if monotonic() >= optimizer_deadline:
            raise _OptimizerWallTimeExceeded

    try:
        remaining_optimizer_seconds()
        result = minimize(
            lambda x: 0.5 * float(x @ (q_matrix @ x)), np.asarray(feasible.x, dtype=np.float64),
            jac=lambda x: q_matrix @ x, method="SLSQP",
            bounds=Bounds(-solver_cap, solver_cap),
            constraints=[LinearConstraint(matrix, solver_rhs, np.full(len(rhs), np.inf))],
            callback=check_optimizer_deadline,
            options={"maxiter": 1000, "ftol": 1.0e-10, "disp": False},
        )
    except _OptimizerWallTimeExceeded as error:
        raise human.ImportError("local self-separation SLSQP exceeded its 120 second wall-time budget") from error
    if monotonic() >= optimizer_deadline:
        raise human.ImportError("local self-separation SLSQP exceeded its 120 second wall-time budget")
    if not result.success or not np.isfinite(result.x).all():
        raise human.ImportError("local self-separation QP did not converge to a finite proposal")
    requested = np.asarray(result.x, dtype=np.float64).reshape(len(movable), 3)
    requested_slack = matrix @ result.x - rhs
    requested_reserved_slack = matrix @ result.x - solver_rhs
    requested_l2 = np.linalg.norm(requested, axis=1)

    def reject_with_gate_diagnostics(
        message: str, *, stage: str, rounded: np.ndarray | None = None,
        rounding_allowance: np.ndarray | None = None,
    ) -> None:
        # Keep original admission metrics distinct from the solver reserve:
        # a reserved solver failure must not masquerade as an original SAT miss.
        def scalar(value: float) -> float | None:
            value = float(value)
            return value if math.isfinite(value) else None

        def describe(delta: np.ndarray, component_allowance: np.ndarray | float,
                     l2_allowance: float) -> dict[str, Any]:
            slack = matrix @ delta.reshape(-1) - rhs
            l2 = np.linalg.norm(delta, axis=1)
            component_margin = cap + component_allowance + 1.0e-10 - np.abs(delta)
            finite_slack = np.isfinite(slack)
            minimum_row = int(np.argmin(slack)) if finite_slack.all() else None
            max_vertex = int(np.argmax(l2)) if np.isfinite(l2).all() else None
            component_index = (np.unravel_index(int(np.argmin(component_margin)),
                                               component_margin.shape)
                               if np.isfinite(component_margin).all() else None)
            details = {
                "finite_increment": bool(np.isfinite(delta).all()),
                "finite_sat_slack": bool(finite_slack.all()),
                "sat_pass": bool(finite_slack.all() and
                                 float(slack.min()) >= -_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM),
                "component_pass": bool(np.isfinite(component_margin).all() and
                                       np.all(component_margin >= 0.0)),
                "l2_pass": bool(np.isfinite(l2).all() and np.all(
                    l2 <= _LOCAL_SELF_SEPARATION_INCREMENT_CAP_MM + l2_allowance)),
                "minimum_sat_slack_mm": scalar(slack.min()),
                "sat_tolerance_mm": _LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM,
                "minimum_sat_constraint": ({"row_index": minimum_row,
                                            **constraint_receipts[minimum_row]}
                                           if minimum_row is not None else None),
                "max_abs_component_mm": scalar(np.abs(delta).max()),
                "component_base_cap_mm": cap,
                "component_rounding_allowance_max_mm": scalar(np.max(component_allowance)),
                "component_numerical_allowance_mm": 1.0e-10,
                "minimum_component_margin_mm": scalar(component_margin.min()),
                "minimum_component_margin_source_vertex_id": (
                    int(movable[component_index[0]]) if component_index is not None else None),
                "minimum_component_margin_axis": (
                    int(component_index[1]) if component_index is not None else None),
                "max_vertex_l2_mm": scalar(l2.max()),
                "max_l2_source_vertex_id": int(movable[max_vertex]) if max_vertex is not None else None,
                "l2_cap_mm": _LOCAL_SELF_SEPARATION_INCREMENT_CAP_MM,
                "l2_numerical_allowance_mm": l2_allowance,
            }
            return details

        details: dict[str, Any] = {
            "rejection_stage": stage,
            "constraint_count": len(rhs),
            "variable_count": nvar,
            "optimizer": {
                "success": bool(result.success),
                "status": int(getattr(result, "status", -1)),
                "message": str(getattr(result, "message", "")),
                "iterations": int(getattr(result, "nit", -1)),
                "objective": scalar(0.5 * float(result.x @ (q_matrix @ result.x))),
            },
            "requested": describe(requested, 0.0, 1.0e-10),
        }
        reserved_component_margin = solver_cap.reshape(len(movable), 3) + 1.0e-10 - np.abs(requested)
        details["rounding_reserve_max_component_mm"] = scalar(rounding_reserve_mm.max())
        details["requested"].update({
            "reserved_sat_pass": bool(np.isfinite(requested_reserved_slack).all() and
                                      float(requested_reserved_slack.min()) >=
                                      -_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM),
            "minimum_reserved_sat_slack_mm": scalar(requested_reserved_slack.min()),
            "reserved_component_pass": bool(np.isfinite(reserved_component_margin).all() and
                                            np.all(reserved_component_margin >= 0.0)),
            "minimum_reserved_component_margin_mm": scalar(reserved_component_margin.min()),
        })
        if rounded is not None:
            assert rounding_allowance is not None
            details["rounded"] = describe(rounded, rounding_allowance, 1.0e-9)
            quantization = rounded - requested
            details["float32_quantization_error_max_abs_component_mm"] = scalar(np.abs(quantization).max())
            details["float32_quantization_error_max_vertex_l2_mm"] = scalar(
                np.linalg.norm(quantization, axis=1).max())
            details["finite_component_rounding_allowance"] = bool(np.isfinite(rounding_allowance).all())
        error = human.ImportError(message)
        error.local_self_separation_diagnostics = details
        raise error

    if (not np.isfinite(requested).all()
            or np.any(np.abs(requested) > cap + 1.0e-10)
            or np.any(np.abs(requested.reshape(-1)) > solver_cap + 1.0e-10)
            or not np.isfinite(requested_reserved_slack).all()
            or float(requested_reserved_slack.min()) < -_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM
            or not np.isfinite(requested_slack).all()
            or float(requested_slack.min()) < -_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM
            or np.any(requested_l2 > _LOCAL_SELF_SEPARATION_INCREMENT_CAP_MM + 1.0e-10)):
        reject_with_gate_diagnostics(
            "local self-separation optimizer failed the Float32-reserved linear constraint, component bound, or 1 mm L2 bound",
            stage="requested",
        )

    proposed = source.copy()
    proposed[movable] = np.asarray(source[movable].astype(np.float64) + requested / 1000.0, dtype="<f4")
    actual_mm = (proposed[movable].astype(np.float64) - source[movable].astype(np.float64)) * 1000.0
    actual_l2 = np.linalg.norm(actual_mm, axis=1)
    actual_component = np.abs(actual_mm)
    rounded_f32 = proposed[movable]
    next_up = np.nextafter(rounded_f32, np.float32(np.inf))
    next_down = np.nextafter(rounded_f32, np.float32(-np.inf))
    component_rounding_tolerance_mm = 500.0 * np.maximum(
        np.abs(next_up.astype(np.float64) - rounded_f32.astype(np.float64)),
        np.abs(rounded_f32.astype(np.float64) - next_down.astype(np.float64)),
    )
    rounded_slack = matrix @ actual_mm.reshape(-1) - rhs
    if (not np.isfinite(actual_mm).all() or np.any(actual_l2 > _LOCAL_SELF_SEPARATION_INCREMENT_CAP_MM + 1.0e-9)
            or not np.isfinite(component_rounding_tolerance_mm).all()
            or np.any(actual_component > cap + component_rounding_tolerance_mm + 1.0e-10)
            or not np.isfinite(rounded_slack).all()
            or float(rounded_slack.min()) < -_LOCAL_SELF_SEPARATION_LINEAR_TOLERANCE_MM):
        reject_with_gate_diagnostics(
            "Float32-rounded local proposal fails its SAT constraints, component bound, or 1 mm L2 bound",
            stage="rounded", rounded=actual_mm,
            rounding_allowance=component_rounding_tolerance_mm,
        )
    if proposed[protected].tobytes() != source[protected].tobytes():
        raise human.ImportError("local self-separation changed a protected target, support, or anchor row")
    outside = np.setdiff1d(np.arange(vertex_count, dtype=np.int64), movable, assume_unique=True)
    if proposed[outside].tobytes() != source[outside].tobytes():
        raise human.ImportError("local self-separation changed source rows outside its movable patch")

    for pose_index, sample in enumerate(checked):
        pose_variable = variable_index[sample["vertex_ids"]]
        pose_delta_m = np.zeros((len(sample["vertex_ids"]), 3), dtype=np.float64)
        movable_rows = pose_variable >= 0
        pose_delta_m[movable_rows] = actual_mm[pose_variable[movable_rows]] / 1000.0
        predicted_world = np.ascontiguousarray(
            (sample["rejected"].astype(np.float64)
             + np.einsum("vij,vj->vi", sample["jacobian"], pose_delta_m)).astype("<f4"),
        )
        unique_pair_faces = np.unique(sample["pairs"].reshape(-1))
        local_pair_faces = np.searchsorted(sample["vertex_ids"], face_rows[unique_pair_faces])
        predicted_records = _exact_surface_records(predicted_world, local_pair_faces)
        record_by_face = {int(face): predicted_records[index]
                          for index, face in enumerate(unique_pair_faces)}
        for first_face, second_face in sample["pairs"]:
            if ci.triangle_intersection_points(record_by_face[int(first_face)][0],
                                               record_by_face[int(second_face)][0]):
                raise human.ImportError(f"Float32-rounded first-order pose prediction retains an exact supplied self-pair at pose={pose_index}")

    return {
        "status": "bounded_local_linearized_proposal",
        "proposal_source_positions_m_f32": proposed,
        "movable_vertex_ids": movable.copy(),
        "source_increment_mm_f32_applied": actual_mm,
        "constraint_rows_by_pose": rows_by_pose,
        "pair_pose_receipts": pair_receipts,
        "rounding_reserve_max_component_mm": float(rounding_reserve_mm.max()),
        "requested_reserved_linearized_minimum_sat_slack_mm": float(requested_reserved_slack.min()),
        "requested_max_vertex_l2_increment_mm": float(requested_l2.max(initial=0.0)),
        "float32_max_vertex_l2_increment_mm": float(actual_l2.max(initial=0.0)),
        "float32_component_bound_rounding_tolerance_mm_max": float(component_rounding_tolerance_mm.max(initial=0.0)),
        "post_rounding_linearized_minimum_sat_slack_mm": float(rounded_slack.min()),
        "post_rounding_predicted_supplied_pair_exact_count": 0,
        "protected_vertex_ids": protected.copy(),
        "exact_forward_or_full_mesh_acceptance_performed": False,
    }


def _signed_surface_integral(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def _generalized_winding_number(point: np.ndarray, triangles: np.ndarray) -> float:
    vectors = triangles.astype(np.float64, copy=False) - np.asarray(point, dtype=np.float64)[None, None, :]
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    lengths = np.linalg.norm(vectors, axis=2)
    la, lb, lc = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    ab = np.einsum("ij,ij->i", a, b)
    bc = np.einsum("ij,ij->i", b, c)
    ca = np.einsum("ij,ij->i", c, a)
    denominator = la * lb * lc + ab * lc + bc * la + ca * lb
    return float(np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * np.pi))


def _candidate_outward_winding(centroid: np.ndarray, normal: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    """Classify a changed face against the current complete Float32 skin shell."""
    point = np.asarray(centroid, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64)
    shell = np.asarray(triangles, dtype=np.float64)
    if (point.shape != (3,) or direction.shape != (3,) or shell.ndim != 3
            or shell.shape[1:] != (3, 3) or len(shell) == 0
            or not np.isfinite(point).all() or not np.isfinite(direction).all()
            or not np.isfinite(shell).all()):
        raise human.ImportError("candidate winding received malformed or non-finite geometry")
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-15:
        raise human.ImportError("candidate winding received a zero face normal")
    direction = direction / length
    offset = _CANDIDATE_WINDING_OFFSET_M
    winding_minus = _generalized_winding_number(point - offset * direction, shell)
    winding_plus = _generalized_winding_number(point + offset * direction, shell)
    contrast = abs(winding_minus) - abs(winding_plus)
    if not np.isfinite([winding_minus, winding_plus, contrast]).all() or contrast <= _MIN_CANDIDATE_WINDING_CONTRAST:
        raise human.ImportError(
            "candidate current-normal winding is ambiguous or inward: "
            f"winding_minus={winding_minus:.12g}, winding_plus={winding_plus:.12g}, "
            f"absolute_inside_minus_outside_contrast={contrast:.12g}, required_gt={_MIN_CANDIDATE_WINDING_CONTRAST:.12g}"
        )
    return {
        "winding_minus": float(winding_minus),
        "winding_plus": float(winding_plus),
        "absolute_inside_minus_outside_contrast": float(contrast),
        "sample_offset_mm": float(offset * 1000.0),
    }


def _bounded_direction_projection(direction_vectors: np.ndarray, current_normal: np.ndarray) -> np.ndarray:
    """Require at most 2x conversion when resolving outward normal demand."""
    directions = np.asarray(direction_vectors, dtype=np.float64)
    normal = np.asarray(current_normal, dtype=np.float64)
    if (directions.ndim != 2 or directions.shape[1] != 3 or normal.shape != (3,)
            or not np.isfinite(directions).all() or not np.isfinite(normal).all()):
        raise human.ImportError("candidate direction projection received malformed or non-finite vectors")
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-15:
        raise human.ImportError("candidate direction projection received a zero current normal")
    projections = directions @ (normal / length)
    minimum = float(projections.min()) if len(projections) else float("nan")
    if not np.isfinite(minimum) or minimum < _MIN_CANDIDATE_DIRECTION_PROJECTION:
        raise human.ImportError(
            "candidate displacement direction is not positively and boundedly conditioned against its current normal: "
            f"minimum_dot={minimum:.12g}, required_ge={_MIN_CANDIDATE_DIRECTION_PROJECTION:.12g}, "
            f"maximum_normal_conversion={1.0 / _MIN_CANDIDATE_DIRECTION_PROJECTION:.12g}"
        )
    return projections



def _shared_source_directions_from_pose_normals(
    jacobians: np.ndarray,
    outward_world_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Find one common source-space direction per vertex for several accepted poses.

    The accepted NHSKIN ABI 5 source coordinates are metres (also confirmed by its
    canonical binding rest-validation record). Each per-vertex Jacobian therefore
    maps source metres to first-order world metres. Pull
    each pose's unit outward direction back through that pose's Jacobian, normalize
    before averaging so no pose wins because of scale, then push the common unit
    source direction forward through every pose map. Admission against active
    face normals is performed by the caller, not inferred from this vertex metric.
    """
    maps = np.asarray(jacobians, dtype=np.float64)
    normals = np.asarray(outward_world_directions, dtype=np.float64)
    if (maps.ndim != 4 or maps.shape[0] < 1 or maps.shape[1] < 1
            or maps.shape[2:] != (3, 3)
            or normals.shape != maps.shape[:2] + (3,)
            or not np.isfinite(maps).all() or not np.isfinite(normals).all()):
        raise human.ImportError("multi-pose source direction received malformed or non-finite pose maps/normals")
    normal_lengths = np.linalg.norm(normals, axis=2)
    if np.any(normal_lengths <= 1.0e-12):
        raise human.ImportError("multi-pose source direction received a zero outward normal")
    normals = normals / normal_lengths[:, :, None]

    singular = np.linalg.svd(maps, compute_uv=False)
    smallest = singular[..., -1]
    condition = singular[..., 0] / smallest
    if (not np.isfinite(condition).all() or float(smallest.min()) < 1.0e-8
            or float(condition.max()) > 2.0):
        raise human.ImportError(
            "multi-pose skin affine Jacobian is singular or ill-conditioned: "
            f"minimum_singular={float(smallest.min()):.12g}, maximum_condition={float(condition.max()):.12g}"
        )

    pulled = np.linalg.solve(maps, normals[..., None])[..., 0]
    pulled_lengths = np.linalg.norm(pulled, axis=2)
    if np.any(pulled_lengths <= 1.0e-12) or not np.isfinite(pulled_lengths).all():
        raise human.ImportError("multi-pose normal pullback is singular or non-finite")
    pulled_unit = pulled / pulled_lengths[:, :, None]
    resultant = pulled_unit.sum(axis=0)
    resultant_lengths = np.linalg.norm(resultant, axis=1)
    coherence = resultant_lengths / float(maps.shape[0])
    if np.any(coherence <= 1.0e-8) or not np.isfinite(coherence).all():
        vertex = int(np.flatnonzero(coherence <= 1.0e-8)[0])
        raise human.ImportError(
            f"accepted-pose outward directions cancel in common source space at vertex {vertex}: "
            f"normalized_resultant={float(coherence[vertex]):.12g}"
        )
    source_directions = resultant / resultant_lengths[:, None]
    mapped = np.einsum("pnij,nj->pni", maps, source_directions)
    mapped_lengths = np.linalg.norm(mapped, axis=2)
    if np.any(mapped_lengths <= 1.0e-12) or not np.isfinite(mapped_lengths).all():
        raise human.ImportError("common source direction maps to a zero or non-finite accepted-pose displacement")
    mapped_unit = mapped / mapped_lengths[:, :, None]
    alignment = np.einsum("pni,pni->pn", mapped_unit, normals)
    if not np.isfinite(alignment).all():
        raise human.ImportError("common source direction has non-finite accepted-pose outward alignment")
    return source_directions, mapped_unit, alignment, {
        "minimum_pullback_resultant_coherence": float(coherence.min()),
        "minimum_pose_outward_alignment": float(alignment.min()),
        "maximum_pose_outward_alignment": float(alignment.max()),
        "maximum_jacobian_condition_number": float(condition.max()),
        "minimum_jacobian_singular_value_m_per_source_m": float(smallest.min()),
    }



def _condition_shared_source_directions(
    jacobians: np.ndarray,
    source_directions: np.ndarray,
    face_vertex_ids: np.ndarray,
    outward_face_normals: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep smoothed directions when admissible; project others into all face cones.

    A smoothed vertex normal need not satisfy an incident face's angle bound.
    For each offending vertex, use SLSQP to project its preferred source
    direction onto the intersection n dot Jd >= c * norm(Jd), over every
    active face and pose. These are convex cones; scale is removed only after
    the optimizer result passes an independent normalized feasibility check.
    If SLSQP reports failure but returns a feasible nonzero point, that point is
    admitted as feasible only; it does not certify the exact closest-point
    projection. The original admission bound is independently rechecked by the
    seed-demand owner. This is offline registration, never a runtime force or
    state update.
    """
    from scipy.optimize import minimize

    maps = np.asarray(jacobians, dtype=np.float64)
    source = np.asarray(source_directions, dtype=np.float64)
    faces = np.asarray(face_vertex_ids, dtype=np.int64)
    normals = np.asarray(outward_face_normals, dtype=np.float64)
    if (maps.ndim != 4 or maps.shape[0] < 1 or maps.shape[1] < 1
            or maps.shape[2:] != (3, 3) or source.shape != maps.shape[1:2] + (3,)
            or faces.ndim != 2 or faces.shape[1] != 3
            or normals.shape != (maps.shape[0], len(faces), 3)
            or not np.isfinite(maps).all() or not np.isfinite(source).all()
            or not np.isfinite(normals).all()
            or (faces.size and (faces.min() < 0 or faces.max() >= len(source)))):
        raise human.ImportError("active-face direction conditioning received malformed or non-finite input")
    source_lengths = np.linalg.norm(source, axis=1)
    normal_lengths = np.linalg.norm(normals, axis=2)
    if np.any(source_lengths <= 1.0e-12) or np.any(normal_lengths <= 1.0e-12):
        raise human.ImportError("active-face direction conditioning received a zero direction or normal")
    unit_source = source / source_lengths[:, None]
    unit_normals = normals / normal_lengths[:, :, None]
    output = source.copy()
    minimum = _MIN_CANDIDATE_DIRECTION_PROJECTION
    # An inward numerical margin avoids admitting an optimizer's roundoff at
    # the boundary. It tightens selection, not the independent admission gate.
    selection_bound = minimum + 1.0e-6
    before_minimum = None
    corrected = []
    nonconverged_feasible_projections = []
    maximum_angle = 0.0
    if len(faces):
        mapped = np.einsum("pfcij,fcj->pfci", maps[:, faces], unit_source[faces])
        lengths = np.linalg.norm(mapped, axis=3)
        if np.any(lengths <= 1.0e-12) or not np.isfinite(lengths).all():
            raise human.ImportError("active-face direction maps to zero or non-finite world motion")
        projections = np.einsum("pfci,pfi->pfc", mapped / lengths[..., None], unit_normals)
        before_minimum = float(projections.min())
        offending = np.unique(faces[np.any(projections < minimum, axis=0)])
        for vertex in offending:
            incident = np.flatnonzero(np.any(faces == vertex, axis=1))
            vertex_maps = np.repeat(maps[:, vertex, None, :, :], len(incident), axis=1).reshape(-1, 3, 3)
            vertex_normals = unit_normals[:, incident].reshape(-1, 3)
            preferred = unit_source[vertex]

            def constraints(direction):
                world = np.einsum("kij,j->ki", vertex_maps, direction)
                return np.einsum("ki,ki->k", vertex_normals, world) - selection_bound * np.linalg.norm(world, axis=1)

            def constraint_jacobian(direction):
                world = np.einsum("kij,j->ki", vertex_maps, direction)
                world_length = np.maximum(np.linalg.norm(world, axis=1), 1.0e-15)
                gradient_world = vertex_normals - selection_bound * world / world_length[:, None]
                return np.einsum("ki,kij->kj", gradient_world, vertex_maps)

            result = minimize(
                lambda direction: 0.5 * float(np.dot(direction - preferred, direction - preferred)),
                preferred,
                jac=lambda direction: direction - preferred,
                constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                method="SLSQP",
                options={"maxiter": 64, "ftol": 1.0e-12},
            )
            projected = np.asarray(result.x, dtype=np.float64)
            if projected.shape != (3,) or not np.isfinite(projected).all():
                raise human.ImportError(
                    f"active-face direction conditioning found no finite nonzero direction at vertex={int(vertex)}: "
                    f"{result.message}"
                )
            length = float(np.linalg.norm(projected))
            if not np.isfinite(length) or length <= 1.0e-8:
                raise human.ImportError(
                    f"active-face direction conditioning found no finite nonzero direction at vertex={int(vertex)}: "
                    f"{result.message}"
                )
            direction = projected / length
            world = np.einsum("kij,j->ki", vertex_maps, direction)
            world_lengths = np.linalg.norm(world, axis=1)
            if np.any(world_lengths <= 1.0e-12):
                raise human.ImportError(f"conditioned direction maps to zero world motion at vertex={int(vertex)}")
            alignment = np.einsum("ki,ki->k", world / world_lengths[:, None], vertex_normals)
            if not np.isfinite(alignment).all() or float(alignment.min()) < minimum:
                raise human.ImportError(
                    f"conditioned direction still fails unchanged active-face bound at vertex={int(vertex)}: "
                    f"projection={float(alignment.min()):.12g}, required_ge={minimum:.12g}"
                )
            # Optimizer status alone is not an admission predicate. A line-search
            # status can occur at a feasible point; accept it only after the
            # normalized direction passes every unchanged face/pose projection
            # above. The caller's seed-demand gate still independently rechecks
            # the resulting field before use.
            if not bool(result.success):
                nonconverged_feasible_projections.append({
                    "compact_vertex_id": int(vertex),
                    "optimizer_success": False,
                    "optimizer_status": int(getattr(result, "status", -1)),
                    "optimizer_message": str(getattr(result, "message", "")),
                    "optimizer_iterations": int(getattr(result, "nit", 0)),
                    "result_norm_before_normalization": length,
                    "minimum_all_constraint_alignment": float(alignment.min()),
                })
            output[vertex] = direction
            corrected.append(int(vertex))
            maximum_angle = max(maximum_angle, float(np.arccos(np.clip(np.dot(direction, preferred), -1.0, 1.0))))

    _, verified = _shared_source_seed_demands(
        maps, output, faces, normals, np.zeros((maps.shape[0], len(faces))),
    )
    return output, {
        "conditioned_vertex_count": len(corrected),
        "conditioned_compact_vertex_ids": corrected,
        "minimum_active_projection_before": before_minimum,
        "minimum_active_projection_after": verified["minimum_active_projection"],
        "unchanged_admission_projection": minimum,
        "selection_projection_with_numerical_margin": selection_bound,
        "maximum_source_direction_change_radians": maximum_angle,
        "nonconverged_feasible_projections": nonconverged_feasible_projections,
    }

def _source_scalar_for_normal_demand(
    jacobian: np.ndarray,
    source_direction: np.ndarray,
    outward_normal: np.ndarray,
    required_normal_distance_m: float,
    *,
    minimum_projection: float = _MIN_CANDIDATE_DIRECTION_PROJECTION,
) -> tuple[float, dict[str, float]]:
    """Convert an outward world-normal gap into one NHSKIN source displacement in metres."""
    mapping = np.asarray(jacobian, dtype=np.float64)
    source = np.asarray(source_direction, dtype=np.float64)
    normal = np.asarray(outward_normal, dtype=np.float64)
    if (mapping.shape != (3, 3) or source.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(mapping).all() or not np.isfinite(source).all()
            or not np.isfinite(normal).all() or not np.isfinite(required_normal_distance_m)
            or required_normal_distance_m < 0.0 or not np.isfinite(minimum_projection)
            or minimum_projection <= 0.0 or minimum_projection > 1.0):
        raise human.ImportError("source scalar demand received malformed or non-finite input")
    source_length = float(np.linalg.norm(source))
    normal_length = float(np.linalg.norm(normal))
    if source_length <= 1.0e-12 or normal_length <= 1.0e-12:
        raise human.ImportError("source scalar demand received a zero direction")
    source = source / source_length
    normal = normal / normal_length
    mapped = mapping @ source
    mapped_length = float(np.linalg.norm(mapped))
    if not np.isfinite(mapped_length) or mapped_length <= 1.0e-12:
        raise human.ImportError("source scalar demand maps to zero or non-finite world motion")
    projection = float(np.dot(mapped / mapped_length, normal))
    if not np.isfinite(projection) or projection < minimum_projection:
        raise human.ImportError(
            "shared source direction is not positively and boundedly conditioned against an active face: "
            f"projection={projection:.12g}, required_ge={minimum_projection:.12g}"
        )
    world_normal_motion_per_source_m = float(np.dot(mapped, normal))
    if not np.isfinite(world_normal_motion_per_source_m) or world_normal_motion_per_source_m <= 0.0:
        raise human.ImportError("shared source direction has no positive active-face normal motion")
    source_distance_m = required_normal_distance_m / world_normal_motion_per_source_m
    if not np.isfinite(source_distance_m) or source_distance_m < 0.0:
        raise human.ImportError("source scalar demand is non-finite or negative")
    return source_distance_m, {
        "world_direction_normal_projection": projection,
        "world_normal_motion_per_source_m": world_normal_motion_per_source_m,
        "source_distance_m": float(source_distance_m),
    }


def _shared_source_seed_demands(
    jacobians: np.ndarray,
    source_directions: np.ndarray,
    face_vertex_ids: np.ndarray,
    outward_face_normals: np.ndarray,
    required_normal_distances_m: np.ndarray,
    *,
    minimum_projection: float = _MIN_CANDIDATE_DIRECTION_PROJECTION,
) -> tuple[np.ndarray, dict[str, float]]:
    """Take the maximum source scalar needed by every active face in every pose.

    Arrays are indexed [pose, vertex] or [pose, active face]. The per-face normal
    demand is projected through the actual per-vertex Jacobian, including scale;
    the returned seed field is therefore in common-atlas source metres.
    """
    maps = np.asarray(jacobians, dtype=np.float64)
    source = np.asarray(source_directions, dtype=np.float64)
    faces = np.asarray(face_vertex_ids, dtype=np.int64)
    normals = np.asarray(outward_face_normals, dtype=np.float64)
    demands = np.asarray(required_normal_distances_m, dtype=np.float64)
    if (maps.ndim != 4 or maps.shape[0] < 1 or maps.shape[2:] != (3, 3)
            or source.shape != maps.shape[1:2] + (3,)
            or faces.ndim != 2 or faces.shape[1] != 3
            or normals.shape != (maps.shape[0], len(faces), 3)
            or demands.shape != (maps.shape[0], len(faces))
            or not np.isfinite(maps).all() or not np.isfinite(source).all()
            or not np.isfinite(normals).all() or not np.isfinite(demands).all()
            or np.any(demands < 0.0)
            or (faces.size and (int(faces.min()) < 0 or int(faces.max()) >= maps.shape[1]))):
        raise human.ImportError("multi-pose active source demand received malformed or non-finite input")
    source_lengths = np.linalg.norm(source, axis=1)
    if np.any(source_lengths <= 1.0e-12):
        raise human.ImportError("multi-pose active source demand received a zero shared direction")
    source = source / source_lengths[:, None]
    seed_required = np.zeros(maps.shape[1], dtype=np.float64)
    if len(faces) == 0:
        return seed_required, {
            "active_face_count": 0,
            "minimum_active_projection": None,
            "maximum_source_seed_demand_m": 0.0,
        }

    face_maps = maps[:, faces]
    face_source = source[faces]
    mapped = np.einsum("pfcij,fcj->pfci", face_maps, face_source)
    mapped_lengths = np.linalg.norm(mapped, axis=3)
    if np.any(mapped_lengths <= 1.0e-12) or not np.isfinite(mapped_lengths).all():
        raise human.ImportError("multi-pose active source direction maps to zero or non-finite world motion")
    normal_lengths = np.linalg.norm(normals, axis=2)
    if np.any(normal_lengths <= 1.0e-12) or not np.isfinite(normal_lengths).all():
        raise human.ImportError("multi-pose active face has a zero or non-finite normal")
    normals = normals / normal_lengths[:, :, None]
    normal_projection = np.sum(mapped / mapped_lengths[:, :, :, None] * normals[:, :, None, :], axis=3)
    if not np.isfinite(normal_projection).all() or float(normal_projection.min()) < minimum_projection:
        pose, face, corner = np.unravel_index(int(np.argmin(normal_projection)), normal_projection.shape)
        raise human.ImportError(
            "shared source direction is not positively and boundedly conditioned at an active face: "
            f"pose={pose}, face={face}, corner={corner}, projection={float(normal_projection[pose, face, corner]):.12g}, "
            f"required_ge={minimum_projection:.12g}"
        )
    normal_motion = np.sum(mapped * normals[:, :, None, :], axis=3)
    if not np.isfinite(normal_motion).all() or np.any(normal_motion <= 0.0):
        raise human.ImportError("shared source direction has no positive active-face normal motion")
    scalar_demands = demands[:, :, None] / normal_motion
    if not np.isfinite(scalar_demands).all() or np.any(scalar_demands < 0.0):
        raise human.ImportError("multi-pose source scalar demand is non-finite or negative")
    worst_pose_per_face_corner = scalar_demands.max(axis=0)
    for corner in range(3):
        np.maximum.at(seed_required, faces[:, corner], worst_pose_per_face_corner[:, corner])
    return seed_required, {
        "active_face_count": int(len(faces)),
        "minimum_active_projection": float(normal_projection.min()),
        "maximum_source_seed_demand_m": float(seed_required.max(initial=0.0)),
    }


def _maximum_pose_edge_lengths(world_positions_by_pose: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Use the maximum accepted-pose world length for each shared source edge."""
    positions = np.asarray(world_positions_by_pose, dtype=np.float64)
    edge_rows = np.asarray(edges, dtype=np.int64)
    if (positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[2] != 3
            or edge_rows.ndim != 2 or edge_rows.shape[1] != 2
            or not np.isfinite(positions).all()
            or (edge_rows.size and (int(edge_rows.min()) < 0 or int(edge_rows.max()) >= positions.shape[1]))):
        raise human.ImportError("multi-pose geodesic metric received malformed or non-finite positions/edges")
    if len(edge_rows) == 0:
        return np.empty(0, dtype=np.float64)
    delta = positions[:, edge_rows[:, 0]] - positions[:, edge_rows[:, 1]]
    lengths = np.linalg.norm(delta, axis=2)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise human.ImportError("multi-pose geodesic metric has zero or non-finite source-edge length")
    return lengths.max(axis=0)

def _smooth_collateral_displacement_near_faces(
    displacement: np.ndarray,
    graph,
    compact_faces: np.ndarray,
    folded_face_rows: np.ndarray,
    required_seed_vertices: np.ndarray,
    fixed_vertices: np.ndarray,
    *,
    graph_hops: int = 3,
    jacobi_sweeps: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Taper a candidate increment to zero on collateral folded faces.

    The exact fold-face vertices are Dirichlet zero increments. The outer
    graph ring retains the proposed displacement, and harmonic interpolation
    transitions across the intervening source-surface rings. Required
    correction seeds and fixed anchors retain their proposed values exactly.
    A fold incident to a required seed or fixed anchor is rejected: this
    helper may regularize collateral support motion, never alter a demanded
    active-source displacement or an exact support/contact pin.
    """
    values = np.asarray(displacement, dtype=np.float64)
    graph_faces = np.asarray(compact_faces, dtype=np.int64)
    folded = np.asarray(folded_face_rows, dtype=np.int64)
    seeds = np.asarray(required_seed_vertices, dtype=np.int64)
    fixed = np.asarray(fixed_vertices, dtype=np.int64)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or graph_faces.ndim != 2 or graph_faces.shape[1] != 3
            or folded.ndim != 1 or not len(folded)
            or seeds.ndim != 1 or fixed.ndim != 1
            or graph.shape != (len(values), len(values))
            or not np.isfinite(graph.data).all() or np.any(graph.data <= 0.0)
            or graph_hops < 2 or jacobi_sweeps < 1
            or int(folded.min()) < 0 or int(folded.max()) >= len(graph_faces)):
        raise human.ImportError("collateral-fold taper received malformed geometry or graph")
    if ((seeds.size and (int(seeds.min()) < 0 or int(seeds.max()) >= len(values)))
            or (fixed.size and (int(fixed.min()) < 0 or int(fixed.max()) >= len(values)))):
        raise human.ImportError("collateral-fold taper received an out-of-range protected vertex")
    folded = np.unique(folded)
    fold_vertices = np.unique(graph_faces[folded].reshape(-1))
    if np.intersect1d(fold_vertices, seeds).size:
        raise human.ImportError("collateral fold touches a required active correction seed")
    if np.intersect1d(fold_vertices, fixed).size:
        raise human.ImportError("collateral fold touches an exact fixed support/contact anchor")

    # Distances use the existing shared-source surface graph. The three-ring
    # domain keeps the edit local while allowing a smooth transition.
    from collections import deque

    distance = np.full(len(values), -1, dtype=np.int32)
    queue = deque()
    for vertex in fold_vertices:
        distance[int(vertex)] = 0
        queue.append(int(vertex))
    indptr = np.asarray(graph.indptr, dtype=np.int64)
    indices = np.asarray(graph.indices, dtype=np.int64)
    while queue:
        vertex = queue.popleft()
        if distance[vertex] >= graph_hops:
            continue
        for neighbor_value in indices[indptr[vertex]:indptr[vertex + 1]]:
            neighbor = int(neighbor_value)
            if distance[neighbor] < 0:
                distance[neighbor] = distance[vertex] + 1
                queue.append(neighbor)
    patch = np.flatnonzero((distance >= 0) & (distance <= graph_hops))
    core = np.zeros(len(values), dtype=bool)
    core[fold_vertices] = True
    protected = np.zeros(len(values), dtype=bool)
    if len(seeds):
        protected[seeds] = True
    if len(fixed):
        protected[fixed] = True
    interior = patch[(distance[patch] > 0) & (distance[patch] < graph_hops)
                     & ~protected[patch] & ~core[patch]]
    smoothed = values.copy()
    smoothed[fold_vertices] = 0.0
    outer_ring = patch[distance[patch] == graph_hops]
    for _ in range(jacobi_sweeps):
        updated = smoothed.copy()
        for vertex_value in interior:
            vertex = int(vertex_value)
            start, end = int(indptr[vertex]), int(indptr[vertex + 1])
            neighbors = indices[start:end]
            lengths = np.asarray(graph.data[start:end], dtype=np.float64)
            if len(neighbors) == 0 or np.any(lengths <= 0.0):
                raise human.ImportError("collateral-fold taper encountered an isolated or invalid source vertex")
            weights = 1.0 / lengths
            updated[vertex] = np.einsum("i,ij->j", weights, smoothed[neighbors]) / float(weights.sum())
        smoothed = updated
        smoothed[fold_vertices] = 0.0
        smoothed[protected] = values[protected]
    if not np.isfinite(smoothed).all() or not np.array_equal(smoothed[protected], values[protected]):
        raise human.ImportError("collateral-fold taper changed a protected seed/anchor or produced non-finite motion")
    if not np.array_equal(smoothed[fold_vertices], np.zeros_like(smoothed[fold_vertices])):
        raise human.ImportError("collateral-fold taper did not zero its non-seed fold-core increments")
    return smoothed, {
        "method": "three-ring source-graph harmonic transition from zero fold core to original proposal",
        "folded_face_rows": [int(value) for value in folded],
        "fold_vertex_count": int(len(fold_vertices)),
        "patch_vertex_count": int(len(patch)),
        "interior_vertex_count": int(len(interior)),
        "outer_ring_vertex_count": int(len(outer_ring)),
        "zero_increment_fold_core_vertex_count": int(len(fold_vertices)),
        "protected_seed_count": int(len(seeds)),
        "fixed_vertex_count": int(len(fixed)),
        "graph_hops": int(graph_hops),
        "jacobi_sweeps": int(jacobi_sweeps),
        "maximum_increment_change_m": float(np.linalg.norm(smoothed - values, axis=1).max(initial=0.0)),
        "maximum_increment_before_m": float(np.linalg.norm(values, axis=1).max(initial=0.0)),
        "maximum_increment_after_m": float(np.linalg.norm(smoothed, axis=1).max(initial=0.0)),
        "required_seed_values_bitwise_preserved": True,
        "fixed_values_bitwise_preserved": True,
        "fold_core_increment_exactly_zero": True,
    }

def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # Recompute area-weighted normals from the current candidate geometry.
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.isfinite(points).all() or not np.isfinite(triangles).all()
            or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(points)))):
        raise human.ImportError("current skin normal reconstruction received malformed geometry")
    tri = points[triangles]
    area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], area_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12) or not np.isfinite(normals).all():
        raise human.ImportError("current skin area-weighted vertex normals are degenerate")
    return normals / lengths[:, None]


def derive_step0_inferred_clearance(
    *,
    source_positions: np.ndarray,
    full_weights: np.ndarray,
    bindings: np.ndarray,
    binding_owner_ids: np.ndarray,
    faces: np.ndarray,
    source_payload_sha256: str,
    accepted_pack_path: Path,
    accepted_receipt_path: Path,
    bone_artifact_path: Path,
    bone_manifest_path: Path,
    witness_path: Path,
    orientation_report_path: Path,
    surface_inventory_path: Path,
    selected_margin_mm: float = 0.25,
    support_radius_edge_multiple: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer a compact common-atlas skin clearance field from exact step-0 witnesses.

    This changes only referenced common-atlas NHSKIN source positions. It preserves
    the canonical shared-atlas bindings, all influence records, full 86-column weights,
    and topology. The clearance margins are engineering separation parameters, not
    measured skin thickness or measured bone-to-skin spacing.
    """
    if selected_margin_mm != 0.25:
        raise human.ImportError("this bounded candidate currently qualifies only the reviewed 0.25 mm primary margin")
    if not np.isfinite(support_radius_edge_multiple) or support_radius_edge_multiple <= 0.0:
        raise human.ImportError("clearance support radius must be a finite positive multiple of the local edge length")
    inputs = [Path(path).resolve() for path in (
        accepted_pack_path, accepted_receipt_path, bone_artifact_path, bone_manifest_path, witness_path, orientation_report_path, surface_inventory_path,
    )]
    if any(not path.is_file() for path in inputs):
        raise human.ImportError("step-0 skin clearance requires all seven pinned evidence/source inputs")
    pack_path, receipt_path, bone_path, bone_manifest, witness_path, orientation_path, inventory_path = inputs
    sha = {str(path): _sha(path) for path in inputs}
    receipt = json.loads(receipt_path.read_bytes())
    all_target_keys, target_keys, expected_inventory_counts, inventory_populations = _load_target_inventory(inventory_path)
    witnesses_doc = json.loads(witness_path.read_bytes())
    orientation_doc = json.loads(orientation_path.read_bytes())
    pack_sha = sha[str(pack_path)]
    if (receipt.get("accepted_step") != 0 or receipt.get("accepted_pack_path") != str(pack_path)
            or Path(witnesses_doc.get("pack_path", "")).resolve() != pack_path
            or witnesses_doc.get("pack_sha256") != pack_sha
            or witnesses_doc.get("witness_count") != _EXPECTED_ALL_WITNESS_COUNT
            or orientation_doc.get("inputs", {}).get("accepted_step0", {}).get("pack_sha256") != pack_sha
            or orientation_doc.get("inputs", {}).get("accepted_step0", {}).get("receipt_sha256") != sha[str(receipt_path)]
            or orientation_doc.get("inputs", {}).get("full_witnesses", {}).get("sha256") != sha[str(witness_path)]
            or orientation_doc.get("inputs", {}).get("nhskin", {}).get("sha256") != source_payload_sha256):
        raise human.ImportError("clearance pack, receipt, full witness file, nonocular orientation report, and base NHSKIN are not hash-linked")
    orientation_selection = orientation_doc.get("selection", {})
    orientation_full_witnesses = orientation_doc.get("inputs", {}).get("full_witnesses", {})
    if (orientation_full_witnesses.get("total_file_count") != _EXPECTED_ALL_WITNESS_COUNT
            or orientation_full_witnesses.get("selected_nonocular_count") != _EXPECTED_CLEARANCE_WITNESS_COUNT
            or orientation_selection.get("excluded_semantic") != 51010
            or set(map(int, orientation_selection.get("excluded_stable_ids", []))) != {key[1] for key in _OCULAR_MONITOR_KEYS}
            or orientation_selection.get("excluded_witnesses") != _EXPECTED_OCULAR_WITNESS_COUNT):
        raise human.ImportError("clearance orientation selection does not match the retained nonocular/ocular partition")
    orientation_rows = orientation_doc.get("per_face", [])
    if len(orientation_rows) != _EXPECTED_ORIENTATION_FACE_COUNT:
        raise human.ImportError("clearance local orientation report does not contain all 980 nonocular witness faces")
    all_witnesses = witnesses_doc.get("witnesses")
    if not isinstance(all_witnesses, list) or len(all_witnesses) != _EXPECTED_ALL_WITNESS_COUNT:
        raise human.ImportError("clearance must cover all 8,863 exact accepted step-0 witnesses")
    all_grouped_counts: dict[tuple[int, int], int] = {}
    for item in all_witnesses:
        key = tuple(map(int, item["other_primitive"]))
        all_grouped_counts[key] = all_grouped_counts.get(key, 0) + 1
    expected_all_witness_counts = {key: value for key, value in expected_inventory_counts.items() if value}
    if all_grouped_counts != expected_all_witness_counts or not set(all_grouped_counts).issubset(all_target_keys):
        raise human.ImportError("full witness partition differs from the pinned 859-surface inventory")
    ocular_witnesses = [item for item in all_witnesses if tuple(map(int, item["other_primitive"])) in _OCULAR_MONITOR_KEYS]
    witnesses = [item for item in all_witnesses if tuple(map(int, item["other_primitive"])) in target_keys]
    grouped_counts: dict[tuple[int, int], int] = {}
    for item in witnesses:
        key = tuple(map(int, item["other_primitive"]))
        grouped_counts[key] = grouped_counts.get(key, 0) + 1
    if len(witnesses) != _EXPECTED_CLEARANCE_WITNESS_COUNT or len(ocular_witnesses) != _EXPECTED_OCULAR_WITNESS_COUNT:
        raise human.ImportError("full witness file does not match the 4,448-clearance/4,415-ocular split")
    expected_witness_counts = {key: value for key, value in expected_inventory_counts.items() if key in target_keys and value}
    if grouped_counts != expected_witness_counts:
        raise human.ImportError("nonocular clearance witness partition differs from the pinned all-surface inventory")
    witness_face_ids = {int(item["skin_source_face_row"]) for item in witnesses}
    if len(witness_face_ids) != _EXPECTED_ORIENTATION_FACE_COUNT:
        raise human.ImportError("nonocular clearance witnesses no longer cover exactly 980 unique skin faces")
    orientation_face_ids = {int(row["face"]) for row in orientation_rows}
    if orientation_face_ids != witness_face_ids:
        raise human.ImportError("clearance orientation report face set differs from the selected nonocular witnesses")
    if not np.isfinite(source_positions).all() or full_weights.shape != (len(source_positions), 86):
        raise human.ImportError("clearance input source positions or full 86-owner weight matrix is malformed")
    if bindings.shape != (86, 9) or binding_owner_ids.shape != (86,) or len(faces) == 0 or faces.ndim != 2 or faces.shape[1] != 3:
        raise human.ImportError("clearance input bindings or source topology is malformed")

    pack_positions, surfaces, pack_counts = _pack_surfaces(pack_path, all_target_keys)
    if not receipt.get("captured_vertex_buffer_sha256"):
        raise human.ImportError("accepted step-0 receipt lacks captured vertex-buffer identity")
    skin_surface = surfaces[(51007, 1)]
    global_skin_faces = skin_surface["faces"]
    skin_base = int(global_skin_faces.min())
    if (int(global_skin_faces.min()) != skin_base or int(global_skin_faces.max()) >= skin_base + len(source_positions)
            or not np.array_equal(global_skin_faces - skin_base, faces)):
        raise human.ImportError("accepted step-0 skin topology/order differs from the registered NHSKIN source")
    referenced = np.unique(faces)
    compact_faces = np.searchsorted(referenced, faces)
    captured = pack_positions[skin_base + referenced].astype(np.float64)
    captured_skin_records = _exact_surface_records(captured, compact_faces)
    poses, stable_to_body, pose_fit, bone_abi = _fit_registered_bone_poses(pack_positions, surfaces, bone_path, bone_manifest)
    owner_ids = binding_owner_ids.astype(int)
    if set(owner_ids) != set(poses):
        raise human.ImportError("clearance full skin binding owners do not match recovered accepted owners")
    predicted_base = np.zeros_like(captured)
    jacobian = np.zeros((len(referenced), 3, 3), dtype=np.float64)
    for bind_index, body_id_value in enumerate(owner_ids):
        body_id = int(body_id_value)
        body_rotation, body_translation = poses[body_id]
        binding = bindings[bind_index]
        binding_rotation = _rotation(binding[4:8])
        binding_scale = float(binding[8])
        local = binding[1:4].astype(np.float64) + binding_scale * (source_positions[referenced] @ binding_rotation.T)
        mapped = local @ body_rotation.T + body_translation
        influence = full_weights[referenced, bind_index].astype(np.float64)
        predicted_base += influence[:, None] * mapped
        jacobian += influence[:, None, None] * (body_rotation @ binding_rotation * binding_scale)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = singular[:, 0] / singular[:, -1]
    if not np.isfinite(condition).all() or float(condition.max()) > 2.0 or float(singular[:, -1].min()) < 1.0e-8:
        raise human.ImportError("clearance 86-influence affine Jacobian is singular or ill-conditioned")
    base_residual = np.linalg.norm(predicted_base - captured, axis=1)
    if float(base_residual.max()) > 1.0e-6:
        raise human.ImportError(f"base NHSKIN full-influence replay misses accepted step-0 geometry by {base_residual.max() * 1e6:.6g} um")

    orientation_faces = {int(row["face"]): row for row in orientation_doc.get("per_face", [])}
    if not orientation_faces:
        raise human.ImportError("clearance orientation report lacks face-level winding witnesses")
    face_constraints = {}
    pair_plane_demands = []
    pair_sat_demands = []
    for item in witnesses:
        face_id = int(item["skin_source_face_row"])
        source_face = faces[face_id]
        if not np.array_equal(source_face, np.asarray(item["skin_source_vertex_ids"], dtype=np.int64)):
            raise human.ImportError(f"clearance witness source face identity differs at face {face_id}")
        ori = orientation_faces.get(face_id)
        if (ori is None or ori.get("class") != "stored_normal_from_locally_more_enclosed_to_less_enclosed"
                or not np.array_equal(source_face, np.asarray(ori.get("source_vertex_ids"), dtype=np.int64))):
            raise human.ImportError(f"clearance outward orientation is not established for skin face {face_id}")
        expected_normal = np.asarray(ori["normal_world"], dtype=np.float64)
        captured_triangle = captured[compact_faces[face_id]]
        normal = np.cross(captured_triangle[1] - captured_triangle[0], captured_triangle[2] - captured_triangle[0])
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1.0e-15:
            raise human.ImportError(f"clearance skin face {face_id} is degenerate in accepted pose")
        normal /= normal_length
        if float(np.dot(normal, expected_normal)) < 0.999999:
            raise human.ImportError(f"clearance orientation report does not match captured winding at face {face_id}")
        target = tuple(map(int, item["other_primitive"]))
        if target[0] == 51004:
            body_id = stable_to_body[target[1]]
            body_center = poses[body_id][1]
            radial_dot = float(np.dot(normal, captured_triangle.mean(axis=0) - body_center))
            if radial_dot <= 0.05:
                raise human.ImportError(f"clearance bone face {face_id} lacks an outward owning-body context")
        surface_faces = surfaces[target]["faces"]
        other_face = int(item["other_primitive_face_row"])
        actual_ids = surface_faces[other_face]
        if not np.array_equal(actual_ids, np.asarray(item["other_mrv_global_vertex_ids"], dtype=np.int64)):
            raise human.ImportError(f"clearance target face identity differs for {target} face {other_face}")
        actual = pack_positions[actual_ids].astype(np.float64)
        opponent = np.asarray(item["other_captured_world_xyz_m_by_vertex"], dtype=np.float64)
        if not np.array_equal(actual.astype("<f4"), opponent.astype("<f4")):
            raise human.ImportError(f"clearance target face coordinates differ from accepted pack for {target} face {other_face}")
        projection = max(0.0, float(np.max((actual - captured_triangle[0]) @ normal)))
        separation = _triangle_normal_translation_to_separate(captured_triangle, actual, normal)
        pair_plane_demands.append(projection)
        pair_sat_demands.append(separation)
        current = face_constraints.get(face_id)
        if current is None:
            face_constraints[face_id] = {
                "normal": normal, "minimum_separating_translation_m": separation,
                "maximum_plane_projection_m": projection,
                "target_surfaces": set(), "target_pair_count": 0,
            }
        else:
            current["minimum_separating_translation_m"] = max(current["minimum_separating_translation_m"], separation)
            current["maximum_plane_projection_m"] = max(current["maximum_plane_projection_m"], projection)
        row = face_constraints[face_id]
        row["target_surfaces"].add(f"{target[0]}:{target[1]}")
        row["target_pair_count"] += 1
    if len(face_constraints) != _EXPECTED_ORIENTATION_FACE_COUNT:
        raise human.ImportError(f"clearance expected 980 unique oriented nonocular witness faces, got {len(face_constraints)}")

    witness_target_faces_by_skin_face: dict[int, set[str]] = {}
    for item in witnesses:
        skin_face = int(item["skin_source_face_row"])
        target_semantic, target_stable_id = map(int, item["other_primitive"])
        witness_target_faces_by_skin_face.setdefault(skin_face, set()).add(
            f"{target_semantic}:{target_stable_id}/face/{int(item['other_primitive_face_row'])}"
        )
    source_edges = np.concatenate((compact_faces[:, [0, 1]], compact_faces[:, [1, 2]], compact_faces[:, [2, 0]]), axis=0)
    source_edges.sort(axis=1)
    edges = np.unique(source_edges, axis=0)
    edge_lengths = np.linalg.norm(captured[edges[:, 0]] - captured[edges[:, 1]], axis=1)
    seed_mask = np.zeros(len(referenced), dtype=bool)
    for face_id in face_constraints:
        seed_mask[compact_faces[face_id]] = True
    touching = np.isin(edges[:, 0], np.flatnonzero(seed_mask)) | np.isin(edges[:, 1], np.flatnonzero(seed_mask))
    median_edge = float(np.median(edge_lengths[touching]))
    radius = float(support_radius_edge_multiple) * median_edge
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    graph_rows = np.concatenate((edges[:, 0], edges[:, 1]))
    graph_cols = np.concatenate((edges[:, 1], edges[:, 0]))

    aabb_audit_by_margin = {}
    ocular_monitor_by_margin = {}
    output_source_by_margin = {}
    support_by_margin = {}
    orientation_checks_added = {}
    all_target_faces = {key: surfaces[key]["faces"] for key in all_target_keys}
    target_faces = {key: surfaces[key]["faces"] for key in target_keys}
    ocular_monitor_faces = {key: surfaces[key]["faces"] for key in _OCULAR_MONITOR_KEYS}
    target_geometry_validation_cache: set[str] = set()
    baseline_all_audit = _target_intersection_audit(
        captured_skin_records, all_target_faces, pack_positions,
        _validated_target_geometry_cache=target_geometry_validation_cache,
    )
    baseline_audit = {f"{key[0]}:{key[1]}": baseline_all_audit[f"{key[0]}:{key[1]}"] for key in target_keys}
    baseline_ocular_audit = {f"{key[0]}:{key[1]}": baseline_all_audit[f"{key[0]}:{key[1]}"] for key in _OCULAR_MONITOR_KEYS}
    baseline_count = _sum_target_pairs(baseline_audit)
    baseline_ocular_count = _sum_target_pairs(baseline_ocular_audit)
    baseline_all_count = _sum_target_pairs(baseline_all_audit)
    if (baseline_count != len(witnesses) or baseline_ocular_count != len(ocular_witnesses)
            or baseline_all_count != len(all_witnesses)):
        raise human.ImportError(
            f"clearance exact scan split differs from pinned witnesses: nonocular={baseline_count}, "
            f"ocular={baseline_ocular_count}, all={baseline_all_count}"
        )
    witness_pair_sets: dict[str, set[tuple[int, int]]] = {}
    for item in all_witnesses:
        key = ":".join(map(str, map(int, item["other_primitive"])))
        witness_pair_sets.setdefault(key, set()).add((int(item["skin_source_face_row"]), int(item["other_primitive_face_row"])))
    for key_text, row in baseline_all_audit.items():
        if _triangle_pair_set(row) != witness_pair_sets.get(key_text, set()):
            raise human.ImportError(f"clearance exact baseline triangle identities differ from the retained witness file for {key_text}")
    for key, expected in expected_inventory_counts.items():
        if int(baseline_all_audit[f"{key[0]}:{key[1]}"]["count"]) != expected:
            raise human.ImportError(f"clearance exact all-surface scan changed the pinned inventory count for {key}")
    baseline_skin_self_audit = _audit_pair(captured_skin_records, captured_skin_records, same_surface=True)
    if int(baseline_skin_self_audit["count"]) != 0:
        raise human.ImportError("accepted step-0 NHSKIN differs from its retained zero-self-intersection audit")
    full_skin_world = pack_positions[skin_base:skin_base + len(source_positions)].astype(np.float64)
    if full_skin_world.shape != source_positions.shape:
        raise human.ImportError("clearance full captured skin vertex range is incomplete")
    full_skin_triangles = full_skin_world[faces]
    base_area_vectors = np.cross(full_skin_triangles[:, 1] - full_skin_triangles[:, 0], full_skin_triangles[:, 2] - full_skin_triangles[:, 0])
    base_double_areas = np.linalg.norm(base_area_vectors, axis=1)
    if np.any(base_double_areas == 0.0) or not np.isfinite(base_double_areas).all():
        raise human.ImportError("accepted step-0 skin has exact zero-area or non-finite source triangles")
    base_surface_integral = _signed_surface_integral(full_skin_world, faces)
    geometry_quality_by_margin = {}
    for margin_mm in (0.125, 0.25, 0.5):
        source_base = source_positions[referenced].astype("<f4").astype(np.float64)
        world_delta_total = np.zeros_like(captured)
        candidate_world = captured.copy()
        source_corrected = source_base.copy()
        source_delta = np.zeros_like(source_base)
        maximum_roundtrip_error_um = 0.0
        candidate_records = captured_skin_records
        margin_audit = baseline_audit
        margin_ocular_audit = baseline_ocular_audit
        candidate_winding_checks: dict[str, dict[str, Any]] = {}
        active_faces_seen: set[int] = set()
        minimum_active_direction_projection = float("inf")
        final_iteration = None
        final_seed_count = 0
        final_field = np.zeros(len(referenced), dtype=np.float64)
        final_quality: dict[str, Any] | None = None
        iteration_receipts: list[dict[str, Any]] = []

        for iteration in range(6):
            remaining_before = _sum_target_pairs(margin_audit)
            if remaining_before == 0:
                final_iteration = iteration
                break

            current_normals = _area_weighted_vertex_normals(candidate_world, compact_faces)
            current_directions = _smooth_vertex_directions(current_normals, compact_faces)
            seed_required = np.zeros(len(referenced), dtype=np.float64)

            for surface_key, row in margin_audit.items():
                target_key = tuple(int(part) for part in surface_key.split(":"))
                target_triangles = target_faces[target_key]
                for skin_face, target_face in row["triangle_pairs"]:
                    skin_face = int(skin_face)
                    target_face = int(target_face)
                    active_faces_seen.add(skin_face)
                    if skin_face not in face_constraints and skin_face not in orientation_checks_added:
                        base_triangle = captured[compact_faces[skin_face]]
                        base_normal = np.cross(base_triangle[1] - base_triangle[0], base_triangle[2] - base_triangle[0])
                        base_length = float(np.linalg.norm(base_normal))
                        if base_length <= 1.0e-15:
                            raise human.ImportError(f"clearance new active source face {skin_face} is degenerate")
                        base_normal /= base_length
                        centroid = base_triangle.mean(axis=0)
                        offset = _CANDIDATE_WINDING_OFFSET_M * base_normal
                        winding_minus = _generalized_winding_number(centroid - offset, full_skin_triangles)
                        winding_plus = _generalized_winding_number(centroid + offset, full_skin_triangles)
                        contrast = abs(winding_minus) - abs(winding_plus)
                        if not np.isfinite([winding_minus, winding_plus, contrast]).all() or contrast <= _MIN_CANDIDATE_WINDING_CONTRAST:
                            raise human.ImportError(
                                f"clearance cannot establish accepted-pose outward direction for new active face {skin_face}: "
                                f"winding_minus={winding_minus:.12g}, winding_plus={winding_plus:.12g}, contrast={contrast:.12g}"
                            )
                        orientation_checks_added[skin_face] = {
                            "source_vertex_ids": [int(value) for value in faces[skin_face]],
                            "winding_minus": float(winding_minus),
                            "winding_plus": float(winding_plus),
                            "absolute_winding_difference": float(contrast),
                            "basis": "accepted step-0 full-shell generalized winding at 0.5 mm on either side of the original source-winding face normal",
                        }

                    ids = compact_faces[skin_face]
                    current_triangle = candidate_world[ids]
                    current_normal = np.cross(
                        current_triangle[1] - current_triangle[0],
                        current_triangle[2] - current_triangle[0],
                    )
                    current_normal_length = float(np.linalg.norm(current_normal))
                    if current_normal_length <= 1.0e-15:
                        raise human.ImportError(
                            f"clearance incremental demand has collapsed face {skin_face}, target={surface_key}/{target_face}, iteration={iteration + 1}"
                        )
                    current_normal /= current_normal_length
                    base_normal = base_area_vectors[skin_face] / base_double_areas[skin_face]
                    base_current_dot = float(np.dot(base_normal, current_normal))
                    if base_current_dot <= 0.0:
                        raise human.ImportError(
                            f"clearance incremental demand lost positive face orientation at face {skin_face}, "
                            f"target={surface_key}/{target_face}, iteration={iteration + 1}, dot={base_current_dot:.12g}"
                        )
                    try:
                        direction_dots = _bounded_direction_projection(current_directions[ids], current_normal)
                    except human.ImportError as error:
                        raise human.ImportError(
                            f"clearance incremental direction projection failed: margin_mm={margin_mm}, "
                            f"iteration={iteration + 1}, skin_face={skin_face}, target={surface_key}, "
                            f"target_face={target_face}, base_current_normal_dot={base_current_dot:.9g}, "
                            f"current_normal={current_normal.tolist()}, current_triangle_m={current_triangle.tolist()}, "
                            f"smoothed_current_directions={current_directions[ids].tolist()}, {error}"
                        ) from error
                    minimum_active_direction_projection = min(
                        minimum_active_direction_projection, float(direction_dots.min()),
                    )
                    target_triangle = pack_positions[target_triangles[target_face]].astype(np.float64)
                    first_exit = _triangle_normal_translation_to_separate(current_triangle, target_triangle, current_normal)
                    demand = first_exit + margin_mm / 1000.0
                    np.maximum.at(seed_required, ids, demand / direction_dots)

            seed = np.flatnonzero(seed_required > 0.0)
            if len(seed) == 0:
                raise human.ImportError(
                    f"clearance margin {margin_mm} mm has exact intersections but no incremental correction seed at iteration {iteration + 1}"
                )
            current_edge_lengths = np.linalg.norm(
                candidate_world[edges[:, 0]] - candidate_world[edges[:, 1]], axis=1,
            )
            if np.any(current_edge_lengths <= 0.0) or not np.isfinite(current_edge_lengths).all():
                raise human.ImportError(
                    f"clearance current geodesic graph has zero or non-finite edges at iteration {iteration + 1}"
                )
            current_graph_weights = np.concatenate((current_edge_lengths, current_edge_lengths))
            current_graph = csr_matrix(
                (current_graph_weights, (graph_rows, graph_cols)),
                shape=(len(referenced), len(referenced)),
            )
            seed_count = int(len(seed))
            increment_field = np.zeros(len(referenced), dtype=np.float64)
            for batch_start in range(0, seed_count, _GEODESIC_SEED_BATCH_SIZE):
                batch_seed = seed[batch_start:batch_start + _GEODESIC_SEED_BATCH_SIZE]
                distance_batch = dijkstra(current_graph, directed=False, indices=batch_seed, limit=radius)
                unit_batch = np.clip(distance_batch / radius, 0.0, 1.0)
                envelope_batch = 1.0 - 3.0 * unit_batch * unit_batch + 2.0 * unit_batch * unit_batch * unit_batch
                envelope_batch[distance_batch >= radius] = 0.0
                batch_field = np.max(seed_required[batch_seed, None] * envelope_batch, axis=0)
                np.maximum(increment_field, batch_field, out=increment_field)
                del distance_batch, unit_batch, envelope_batch, batch_field, batch_seed
            del current_graph, current_graph_weights, current_edge_lengths, seed_required, seed, current_normals
            if not np.isfinite(increment_field).all() or not np.any(increment_field > 0.0):
                raise human.ImportError(f"clearance incremental compact field is invalid at iteration {iteration + 1}")

            accepted = False
            last_rejection = "no candidate attempted"
            correction_scale_accepted = None
            for backtrack in range(7):
                correction_scale = 0.5 ** backtrack
                proposed_world_delta = world_delta_total + correction_scale * increment_field[:, None] * current_directions
                proposed_source_delta = np.linalg.solve(jacobian, proposed_world_delta[:, :, None])[:, :, 0]
                trial_source = (source_base + proposed_source_delta).astype("<f4").astype(np.float64)
                packed_source_delta = trial_source - source_base
                mapped_world_delta = np.einsum("nij,nj->ni", jacobian, packed_source_delta)
                roundtrip_error = np.linalg.norm(mapped_world_delta - proposed_world_delta, axis=1)
                trial_world = _apply_captured_world_delta(captured, mapped_world_delta)
                if np.array_equal(trial_world, candidate_world):
                    last_rejection = "packed Float32 update produced no new candidate coordinates"
                    continue

                trial_triangles = trial_world[compact_faces]
                trial_area_vectors = np.cross(
                    trial_triangles[:, 1] - trial_triangles[:, 0],
                    trial_triangles[:, 2] - trial_triangles[:, 0],
                )
                trial_double_areas = np.linalg.norm(trial_area_vectors, axis=1)
                if np.any(trial_double_areas == 0.0) or not np.isfinite(trial_double_areas).all():
                    bad_face = int(np.argmin(trial_double_areas))
                    last_rejection = f"exact zero-area or non-finite trial triangle face={bad_face}"
                    continue
                trial_normal_alignment = np.einsum("ij,ij->i", base_area_vectors, trial_area_vectors) / (
                    base_double_areas * trial_double_areas
                )
                if not np.isfinite(trial_normal_alignment).all() or float(trial_normal_alignment.min()) <= 0.0:
                    bad_face = int(np.argmin(trial_normal_alignment))
                    last_rejection = (
                        f"nonpositive accepted-to-trial face normal dot at face={bad_face}, "
                        f"dot={float(trial_normal_alignment[bad_face]):.12g}"
                    )
                    continue

                trial_records = _exact_surface_records(trial_world, compact_faces)
                trial_self_audit = _audit_pair(trial_records, trial_records, same_surface=True)
                if int(trial_self_audit["count"]) != 0:
                    last_rejection = f"trial has {trial_self_audit['count']} exact nonadjacent skin self-intersections"
                    continue

                trial_full_world = full_skin_world.copy()
                trial_full_world[referenced] = trial_world
                trial_full_triangles = trial_full_world[faces]
                trial_all_audit = _target_intersection_audit(
                    trial_records, all_target_faces, pack_positions,
                    _validated_target_geometry_cache=target_geometry_validation_cache,
                )
                trial_margin_audit = {
                    f"{key[0]}:{key[1]}": trial_all_audit[f"{key[0]}:{key[1]}"]
                    for key in target_keys
                }
                trial_ocular_audit = {
                    f"{key[0]}:{key[1]}": trial_all_audit[f"{key[0]}:{key[1]}"]
                    for key in _OCULAR_MONITOR_KEYS
                }
                ocular_changed = []
                for key_text, baseline_row in baseline_ocular_audit.items():
                    before = _triangle_pair_set(baseline_row)
                    after = _triangle_pair_set(trial_ocular_audit[key_text])
                    if before != after:
                        ocular_changed.append({
                            "surface": key_text,
                            "added": sorted(after - before)[:16],
                            "removed": sorted(before - after)[:16],
                        })
                if ocular_changed:
                    last_rejection = f"trial changed monitored ocular pair identities: {ocular_changed[:4]}"
                    continue

                trial_active_faces = set(active_faces_seen)
                for row in trial_margin_audit.values():
                    trial_active_faces.update(int(pair[0]) for pair in row["triangle_pairs"])
                trial_winding_checks = {}
                orientation_failure = None
                for skin_face in sorted(trial_active_faces):
                    face_normal = trial_area_vectors[skin_face] / trial_double_areas[skin_face]
                    base_normal = base_area_vectors[skin_face] / base_double_areas[skin_face]
                    base_current_dot = float(np.dot(base_normal, face_normal))
                    if base_current_dot <= 0.0:
                        orientation_failure = (
                            f"trial face {skin_face} lost positive accepted-to-candidate normal dot {base_current_dot:.12g}"
                        )
                        break
                    if base_current_dot < _ROTATED_ACTIVE_FACE_BASE_DOT:
                        face_ids = compact_faces[skin_face]
                        center = trial_world[face_ids].mean(axis=0)
                        try:
                            winding = _candidate_outward_winding(center, face_normal, trial_full_triangles)
                        except human.ImportError as error:
                            orientation_failure = f"trial rotated face {skin_face} current-shell winding rejected: {error}"
                            break
                        check_key = f"iteration-{iteration + 1}:backtrack-{backtrack}:face-{skin_face}"
                        trial_winding_checks[check_key] = {
                            "skin_face": int(skin_face),
                            "iteration": int(iteration + 1),
                            "backtrack": int(backtrack),
                            "base_current_normal_dot": base_current_dot,
                            **winding,
                        }
                if orientation_failure is not None:
                    last_rejection = orientation_failure
                    continue

                trial_margin_count = _sum_target_pairs(trial_margin_audit)
                trial_area_ratio = trial_double_areas / base_double_areas
                base_edge_lengths = np.stack((
                    np.linalg.norm(captured[compact_faces[:, 1]] - captured[compact_faces[:, 0]], axis=1),
                    np.linalg.norm(captured[compact_faces[:, 2]] - captured[compact_faces[:, 1]], axis=1),
                    np.linalg.norm(captured[compact_faces[:, 0]] - captured[compact_faces[:, 2]], axis=1),
                ), axis=1)
                trial_edge_lengths = np.stack((
                    np.linalg.norm(trial_triangles[:, 1] - trial_triangles[:, 0], axis=1),
                    np.linalg.norm(trial_triangles[:, 2] - trial_triangles[:, 1], axis=1),
                    np.linalg.norm(trial_triangles[:, 0] - trial_triangles[:, 2], axis=1),
                ), axis=1)
                if np.any(base_edge_lengths <= 0.0) or not np.isfinite(trial_edge_lengths).all():
                    last_rejection = "trial has zero or non-finite source/candidate skin edges"
                    continue
                trial_edge_stretch = trial_edge_lengths / base_edge_lengths
                trial_surface_integral = _signed_surface_integral(trial_full_world, faces)

                candidate_world = trial_world
                candidate_records = trial_records
                world_delta_total = trial_world - captured
                source_corrected = trial_source
                source_delta = source_corrected - source_base
                maximum_roundtrip_error_um = max(
                    maximum_roundtrip_error_um, float(roundtrip_error.max() * 1.0e6),
                )
                margin_audit = trial_margin_audit
                margin_ocular_audit = trial_ocular_audit
                active_faces_seen = trial_active_faces
                candidate_winding_checks.update(trial_winding_checks)
                correction_scale_accepted = correction_scale
                final_seed_count = seed_count
                final_field = increment_field * correction_scale
                final_quality = {
                    "exact_nonadjacent_skin_self_intersection_pairs": int(trial_self_audit["count"]),
                    "exact_zero_area_triangles": int(np.count_nonzero(trial_double_areas == 0.0)),
                    "minimum_triangle_area_ratio": float(trial_area_ratio.min()),
                    "maximum_triangle_area_ratio": float(trial_area_ratio.max()),
                    "minimum_triangle_normal_dot": float(trial_normal_alignment.min()),
                    "maximum_edge_length_stretch_ratio": float(trial_edge_stretch.max()),
                    "minimum_edge_length_stretch_ratio": float(trial_edge_stretch.min()),
                    "minimum_applied_demand_direction_current_normal_dot": (
                        minimum_active_direction_projection if np.isfinite(minimum_active_direction_projection) else None
                    ),
                    "minimum_candidate_winding_contrast": (
                        min((row["absolute_inside_minus_outside_contrast"] for row in candidate_winding_checks.values()), default=None)
                    ),
                    "rotated_active_face_candidate_winding_checks": candidate_winding_checks,
                    "rotated_active_face_candidate_winding_check_count": len(candidate_winding_checks),
                    "signed_surface_integral_m3_proxy": trial_surface_integral,
                    "signed_surface_integral_delta_m3_proxy": trial_surface_integral - base_surface_integral,
                    "surface_integral_limitation": "The NHSKIN has 115 boundary edges at ocular openings, so this oriented surface integral is a sensitivity proxy, not a closed-shell volume.",
                }
                iteration_receipts.append({
                    "iteration": int(iteration + 1),
                    "accepted_backtrack_factor": float(correction_scale),
                    "seed_vertex_count": seed_count,
                    "maximum_increment_mm": float(final_field.max() * 1000.0),
                    "maximum_total_world_displacement_mm": float(np.linalg.norm(world_delta_total, axis=1).max() * 1000.0),
                    "nonocular_pairs_before": int(remaining_before),
                    "nonocular_pairs_after": int(trial_margin_count),
                    "rotated_active_face_winding_check_count": len(trial_winding_checks),
                    "minimum_direction_projection_for_applied_demand": (
                        float(minimum_active_direction_projection)
                        if np.isfinite(minimum_active_direction_projection) else None
                    ),
                    "packed_inverse_roundtrip_error_um": float(roundtrip_error.max() * 1.0e6),
                })
                accepted = True
                break

            if not accepted:
                raise human.ImportError(
                    f"clearance could not accept a geometry-valid incremental correction: margin_mm={margin_mm}, "
                    f"iteration={iteration + 1}, backtrack_attempts=7, last_rejection={last_rejection}"
                )
            if _sum_target_pairs(margin_audit) == 0:
                final_iteration = iteration + 1
                break

        if final_iteration is None or margin_audit is None or final_quality is None:
            raise human.ImportError(
                f"incremental clearance did not resolve all exact nearby target intersections at margin {margin_mm} mm within 6 iterations"
            )
        pair_count = _sum_target_pairs(margin_audit)
        geometry_quality_by_margin[str(margin_mm)] = final_quality
        aabb_audit_by_margin[str(margin_mm)] = {
            "remaining_exact_skin_target_triangle_pairs": pair_count,
            "active_set_iterations": final_iteration,
            "by_target_surface": margin_audit,
        }
        ocular_monitor_by_margin[str(margin_mm)] = {
            "exact_pair_sets_unchanged": True,
            "pair_count": _sum_target_pairs(margin_ocular_audit),
            "per_surface_pair_counts": {key: int(row["count"]) for key, row in margin_ocular_audit.items()},
        }
        output_source_by_margin[margin_mm] = source_corrected
        support_by_margin[str(margin_mm)] = {
            "margin_mm": margin_mm,
            "initial_witness_faces": len(face_constraints),
            "additional_active_faces": len(orientation_checks_added),
            "witness_triangle_pairs": len(witnesses),
            "seed_vertices_last_increment": final_seed_count,
            "nonzero_displaced_vertices": int(np.count_nonzero(np.linalg.norm(world_delta_total, axis=1) > 0.0)),
            "support_radius_mm": radius * 1000.0,
            "support_radius_edge_multiple": float(support_radius_edge_multiple),
            "median_local_edge_mm": median_edge * 1000.0,
            "maximum_world_displacement_mm": float(np.linalg.norm(world_delta_total, axis=1).max() * 1000.0),
            "maximum_increment_mm_last_iteration": float(final_field.max() * 1000.0),
            "maximum_common_atlas_source_displacement_mm": float(np.linalg.norm(source_delta, axis=1).max() * 1000.0),
            "maximum_packed_inverse_roundtrip_error_um": maximum_roundtrip_error_um,
            "maximum_jacobian_condition_number": float(condition.max()),
            "maximum_accepted_pose_fit_residual_um": float(base_residual.max() * 1.0e6),
            "maximum_accepted_pose_fit_residual_on_packed_displaced_vertices_um": float(
                base_residual[np.any(source_delta != 0.0, axis=1)].max() * 1.0e6
            ) if np.any(source_delta != 0.0) else 0.0,
            "extension_direction_smoothing": "recompute current candidate area-weighted vertex normals and apply existing Human skin visual-normal smoother for eight iterations before each incremental correction",
            "extension_direction_smoothing_iterations": _DIRECTION_SMOOTHING_ITERATIONS,
            "iterations": iteration_receipts,
            "geodesic_seed_batch_size": _GEODESIC_SEED_BATCH_SIZE,
        }
    main_source = source_positions.copy()
    main_source[referenced] = output_source_by_margin[0.25]
    report = {
        "schema": "numi.human.step0-witnessed-common-atlas-skin-clearance.v1",
        "status": "inferred_engineering_clearance_candidate_pending_native_replay",
        "method": {
            "direction": "each iteration recomputes current candidate area-weighted vertex normals and applies the existing eight-pass Human visual-normal smoother; only unresolved exact pairs generate a new correction demand. Source winding remains the independent orientation/provenance reference, and rotated active faces are checked against current-candidate full-shell winding.",
            "extension_direction_smoothing": "for each bounded increment, recompute area-weighted normals from the current candidate triangles and smooth them with the existing Human skin visual-normal owner for eight iterations; require each demanded vertex direction to project at least 0.5 onto its current active face normal, bounding normal-demand conversion to at most 2x. Previously resolved faces are not rechecked against this demand-conditioning bound.",
            "finite_pair_demand": "for each remaining exact pair, compute the minimum translation along its current skin-face normal that separates the current finite skin and fixed target triangles under their separating-axis intervals, then add the selected engineering offset; convert that current-normal demand through the current smoothed vertex directions.",
            "field": "at most six incremental corrections per margin. Each iteration builds a compact geodesic smootherstep field around currently unresolved pair vertices using current candidate edge lengths and the original median-edge support radius; try scales 1, 1/2, through 1/64 and accept only a candidate that passes geometry/orientation checks after an exact all-target rescan and preserves ocular pair identities; any remaining nonocular pairs become demands for the next increment.",
            "inverse_map": "accumulate each bounded world-space increment, solve the full 3x3 affine Jacobian built from all 86 canonical skin bindings, unchanged source weights, and recovered accepted body rotations, pack the resulting common-atlas source delta to Float32, then apply the mapped delta to the exact captured accepted-world baseline; zero-delta vertices retain their captured Float32 coordinates",
            "active_face_orientation_admission": "every trial keeps finite nonzero face areas and strictly positive accepted-to-candidate face-normal dot; when an active face rotates below the 0.75 audit-trigger dot, classify its current normal using generalized winding on the full candidate Float32 shell at +/-0.5 mm and require the minus-normal side to be more enclosed by >0.5 absolute winding units. The 0.75 dot triggers this stronger winding check and is not an angle pass/fail ceiling.",
            "local_shape_diagnostics": "per margin report full-mesh minimum/maximum face-area ratio and edge-length stretch ratio; no unvalidated shape-change envelope is inferred from these geometric diagnostics",
            "pose_fit_uncertainty": "candidate displacement is linearized through the recovered 86-owner Jacobian; the full captured-vs-rigid-fit residual and the maximum residual on packed-displaced vertices are reported per margin and remain a sub-micrometer uncertainty requiring native confirmation",
            "interpretation": "inferred finite-triangle separation translation plus a normal-translation engineering offset; not measured skin thickness, penetration depth, or measured physiologic bone-to-skin spacing",
            "topology_or_bindings_changed": False,
        },
        "inputs": {
            "accepted_step0_pack": {"path": str(pack_path), "sha256": pack_sha},
            "accepted_step0_receipt": {"path": str(receipt_path), "sha256": sha[str(receipt_path)]},
            "NHBONES": {"path": str(bone_path), "sha256": sha[str(bone_path)], "manifest_path": str(bone_manifest), "manifest_sha256": sha[str(bone_manifest)]},
            "all_8863_hit_witnesses": {"path": str(witness_path), "sha256": sha[str(witness_path)]},
            "all_859_surface_inventory": {"path": str(inventory_path), "sha256": sha[str(inventory_path)]},
            "nonocular_orientation_report": {"path": str(orientation_path), "sha256": sha[str(orientation_path)]},
            "base_NHSKIN_sha256": source_payload_sha256,
        },
        "finite_triangle_demand_comparison": {
            "pair_count": len(witnesses),
            "plane_max_opponent_vertex_projection_mm": {
                "median": float(np.median(pair_plane_demands) * 1000.0),
                "p95": float(np.percentile(pair_plane_demands, 95) * 1000.0),
                "maximum": float(np.max(pair_plane_demands) * 1000.0),
            },
            "first_separating_axis_translation_mm": {
                "median": float(np.median(pair_sat_demands) * 1000.0),
                "p95": float(np.percentile(pair_sat_demands, 95) * 1000.0),
                "maximum": float(np.max(pair_sat_demands) * 1000.0),
            },
            "pairwise_plane_minus_SAT_reduction_mm": {
                "median": float(np.median(np.asarray(pair_plane_demands) - np.asarray(pair_sat_demands)) * 1000.0),
                "p95": float(np.percentile(np.asarray(pair_plane_demands) - np.asarray(pair_sat_demands), 95) * 1000.0),
                "maximum": float(np.max(np.asarray(pair_plane_demands) - np.asarray(pair_sat_demands)) * 1000.0),
            },
            "interpretation": "finite triangle separating-axis translation is a directional displacement parameter for each exact pair; the plane quantity is retained as a conservative vertex-projection comparison, and neither is reported as measured tissue thickness or physical penetration depth",
        },
        "pose_reconstruction": {
            "source": "Kabsch fit of every source-registered NHBONES member's exact local vertices/topology to its accepted MRVPACK world vertices",
            "registered_owner_count": len(poses), "fit_residuals_by_body": pose_fit,
            "maximum_pose_fit_residual_um": max(row["maximum_rigid_fit_residual_um"] for row in pose_fit.values()),
            "bone_payload": bone_abi,
        },
        "accepted_skin_prediction": {
            "source": "full 86-owner affine map using recovered captured owner poses",
            "maximum_base_geometry_residual_mm": float(base_residual.max() * 1000.0),
            "p95_base_geometry_residual_mm": float(np.percentile(base_residual * 1000.0, 95)),
        },
        "scope": {
            "nonocular_clearance_witness_pairs": len(witnesses),
            "ocular_monitor_witness_pairs": len(ocular_witnesses),
            "all_inventory_witness_pairs": len(all_witnesses),
            "unique_nonocular_skin_faces": len(face_constraints),
            "witness_pairs_by_clearance_target": {f"{k[0]}:{k[1]}": v for k, v in grouped_counts.items()},
            "target_surfaces_by_semantic": {str(k): v for k, v in inventory_populations.items()},
            "nonocular_clearance_target_surface_count": len(target_keys),
            "ocular_monitor_surface_count": len(_OCULAR_MONITOR_KEYS),
            "all_surfaces_scanned": len(all_target_keys),
            "full_target_surface_scan": (
                "baseline and every candidate margin are scanned exactly against all faces of all 859 inventory surfaces; "
                "842 nonocular surfaces are clearance targets, while the 17 eye subsurfaces retain their exact baseline "
                "triangle-pair identities as separate monitors and are not treated as corrected or cleared"
            ),
        },
        "active_set_orientation_checks": orientation_checks_added,
        "support_variants": support_by_margin,
        "exact_intersection_audits": {
            "baseline_before_clearance": {
                "nonocular_target_pair_count": baseline_count,
                "ocular_monitor_pair_count": baseline_ocular_count,
                "all_inventory_pair_count": baseline_all_count,
                "by_nonocular_target_surface": baseline_audit,
                "by_ocular_monitor_surface": {key: int(row["count"]) for key, row in baseline_ocular_audit.items()},
            },
            "baseline_skin_self_intersection_pairs": int(baseline_skin_self_audit["count"]),
            "margins": aabb_audit_by_margin,
            "ocular_monitor_margins": ocular_monitor_by_margin,
        },
        "skin_geometry_quality_by_margin": geometry_quality_by_margin,
        "selected_margin_mm": 0.25,
        "support_radius_edge_multiple": float(support_radius_edge_multiple),
        "qualification": {
            "accepted_pose0_all_4448_nonocular_pairs_and_all_842_clearance_targets": (
                "passed" if int(aabb_audit_by_margin["0.25"]["remaining_exact_skin_target_triangle_pairs"]) == 0 else "failed"
            ),
            "accepted_pose0_17_ocular_interfaces": "exact_pair_sets_unchanged_and_monitored_separately",
            "native_pose_clearance": "pending_native_replay",
            "all_pose_skin_self_and_bone_clearance": "pending",
            "physical_or_collision_use": "not_admitted",
        },
    }
    return main_source, report

def _propagate_source_face_orientation(
    faces: np.ndarray,
    outward_basis_face_ids: np.ndarray,
) -> np.ndarray:
    """Propagate a known outward face ordering across a connected triangle surface."""
    triangles = np.asarray(faces, dtype=np.int64)
    seeds = np.asarray(outward_basis_face_ids, dtype=np.int64)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0
            or seeds.ndim != 1 or len(seeds) == 0
            or np.any(triangles < 0) or np.any(seeds < 0) or np.any(seeds >= len(triangles))):
        raise human.ImportError("source winding propagation requires valid faces and at least one outward basis face")
    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, face in enumerate(triangles):
        if len(set(map(int, face))) != 3:
            raise human.ImportError(f"source winding face {face_id} repeats a vertex")
        for corner in range(3):
            start, end = int(face[corner]), int(face[(corner + 1) % 3])
            edge = (min(start, end), max(start, end))
            direction = 1 if start < end else -1
            edge_owners.setdefault(edge, []).append((face_id, direction))
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(triangles))]
    for edge, owners in edge_owners.items():
        if len(owners) > 2:
            raise human.ImportError(f"source winding edge {edge} has {len(owners)} incident faces")
        if len(owners) == 2:
            (first, first_direction), (second, second_direction) = owners
            relation = -first_direction * second_direction
            adjacency[first].append((second, relation))
            adjacency[second].append((first, relation))
    signs = np.zeros(len(triangles), dtype=np.int8)
    stack = []
    for face_id in seeds:
        face_id = int(face_id)
        if signs[face_id] not in (0, 1):
            raise human.ImportError("outward source winding basis contradicts itself")
        signs[face_id] = 1
        stack.append(face_id)
    while stack:
        face_id = stack.pop()
        for neighbor, relation in adjacency[face_id]:
            expected = int(signs[face_id]) * relation
            if signs[neighbor] == 0:
                signs[neighbor] = expected
                stack.append(neighbor)
            elif int(signs[neighbor]) != expected:
                raise human.ImportError("source face adjacency is not consistently orientable from its outward basis")
    if np.any(signs == 0):
        missing = int(np.flatnonzero(signs == 0)[0])
        raise human.ImportError(f"source skin has a disconnected face component without an outward basis at face {missing}")
    return signs


def _verify_source_winding_in_accepted_poses(
    source_positions: np.ndarray,
    compact_faces: np.ndarray,
    source_outward_face_signs: np.ndarray,
    jacobians_by_pose: np.ndarray,
    accepted_area_vectors_by_pose: np.ndarray,
) -> dict[str, float]:
    """Check accepted face winding against topology-propagated source outward normals.

    The per-face average affine Jacobian transports the source oriented area. This
    is a sign check only; native accepted geometry remains the position authority.
    """
    source = np.asarray(source_positions, dtype=np.float64)
    faces = np.asarray(compact_faces, dtype=np.int64)
    signs = np.asarray(source_outward_face_signs, dtype=np.int8)
    maps = np.asarray(jacobians_by_pose, dtype=np.float64)
    actual = np.asarray(accepted_area_vectors_by_pose, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or signs.shape != (len(faces),) or not np.isin(signs, (-1, 1)).all()
            or maps.ndim != 4 or maps.shape[2:] != (3, 3)
            or actual.shape != (maps.shape[0], len(faces), 3)
            or not np.isfinite(source).all() or not np.isfinite(maps).all() or not np.isfinite(actual).all()):
        raise human.ImportError("source-to-pose winding verification received malformed data")
    source_triangles = source[faces]
    source_areas = np.cross(source_triangles[:, 1] - source_triangles[:, 0],
                            source_triangles[:, 2] - source_triangles[:, 0])
    source_areas *= signs[:, None]
    face_maps = maps[:, faces].mean(axis=2)
    determinants = np.linalg.det(face_maps)
    if not np.isfinite(determinants).all() or np.any(determinants <= 0.0):
        raise human.ImportError("accepted skin face map is reflected, singular, or non-finite")
    rhs = np.broadcast_to(source_areas, (maps.shape[0], *source_areas.shape))
    transported = determinants[:, :, None] * np.linalg.solve(
        np.swapaxes(face_maps, -1, -2), rhs[..., None],
    )[..., 0]
    transported_lengths = np.linalg.norm(transported, axis=2)
    actual_lengths = np.linalg.norm(actual, axis=2)
    if (np.any(transported_lengths <= 1.0e-15) or np.any(actual_lengths <= 1.0e-15)
            or not np.isfinite(transported_lengths).all() or not np.isfinite(actual_lengths).all()):
        raise human.ImportError("accepted skin source/pose winding has a degenerate face")
    alignment = np.einsum("pfi,pfi->pf", transported, actual) / (transported_lengths * actual_lengths)
    if not np.isfinite(alignment).all() or float(alignment.min()) <= 0.0:
        pose, face = np.unravel_index(int(np.argmin(alignment)), alignment.shape)
        raise human.ImportError(
            f"accepted pose face winding disagrees with topology-propagated outward source basis: "
            f"pose={pose}, face={face}, dot={float(alignment[pose, face]):.12g}"
        )
    return {
        "minimum_source_to_accepted_pose_normal_alignment": float(alignment.min()),
        "maximum_source_to_accepted_pose_normal_alignment": float(alignment.max()),
        "minimum_face_map_determinant": float(determinants.min()),
        "maximum_face_map_determinant": float(determinants.max()),
    }


def derive_shared_multipose_inferred_clearance(
    *,
    source_positions: np.ndarray,
    faces: np.ndarray,
    jacobians_by_pose: np.ndarray | None,
    accepted_skin_world_by_pose: np.ndarray,
    baseline_target_audits_by_pose: list[dict[str, dict[str, Any]]],
    baseline_skin_self_pairs_by_pose: list[int],
    source_outward_face_signs: np.ndarray,
    scan_candidate_targets,
    target_triangle_by_row,
    all_target_keys: set[str],
    ocular_monitor_keys: set[str],
    fixed_source_vertex_ids: np.ndarray,
    bed_plane_origins_by_pose: np.ndarray,
    bed_plane_normals_by_pose: np.ndarray,
    selected_margin_mm: float = 0.25,
    support_radius_edge_multiple: float = 4.0,
    max_iterations: int = 6,
    backtrack_count: int = 7,
    progress_callback=None,
    candidate_forward=None,
    baseline_replay_tolerance_m: float = 1.0e-6,
    preserved_source_anchor_vertex_ids: np.ndarray | None = None,
    scan_progress_callback=None,
    resume_source_positions: np.ndarray | None = None,
    resume_target_audits_by_pose: list[dict[str, dict[str, Any]]] | None = None,
    resume_self_audits_by_pose: list[dict[str, Any]] | None = None,
    resume_provenance: dict[str, Any] | None = None,
    closed_target_surfaces_by_pose: list[dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    closed_target_face_counts: dict[str, int] | None = None,
    closed_target_outer_envelope_keys: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer one source-space correction constrained by all supplied accepted poses.

    The caller must hash-validate packs/receipts and exact baseline audits. The scan
    callback must cover every all_target_key and return _target_intersection_audit
    rows. The triangle callback returns exact captured Float32 world coordinates.
    An optional candidate_forward(full_source_f32) callback returns a mapping with
    world_positions_by_pose[pose, sorted_referenced_vertex, xyz] and
    jacobians_by_pose[pose, sorted_referenced_vertex, world_xyz, source_xyz].
    It is reevaluated for each candidate so source-dependent maps, including
    respiratory basis/weight changes, are not treated as fixed affine transforms.
    Optional resume source/audits must be hash-validated by the caller and are
    checked against immutable baseline orientation, bed, anchor, and ocular gates.
    An optional accepted self-audit proof can skip only the redundant full initial
    same-surface scans when its per-pose world, source, face, pair-table, owner,
    and predicate identities match; absent proof keeps the original scans, and
    changed candidate geometry still receives the exact incremental self audit.
    Optional closed_target_surfaces_by_pose supplies complete captured meshes for
    an explicitly selected nonocular subset. Exact coordinate quotient, manifold,
    and self-intersection gates must pass before inside vertices can add demands.
    closed_target_face_counts must bind the complete captured face inventory.
    Explicit closed_target_outer_envelope_keys may use a uniquely proven outer
    component for external-skin enclosure; every component remains in all scans.
    Every triangle must match the existing target lookup exactly. These direct vertex
    demands supplement the boundary-triangle demands and preserve all audits.
    NHSKIN source and mapped world positions are metres; no runtime state changes.
    """
    source = np.asarray(source_positions, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    maps = None if jacobians_by_pose is None else np.asarray(jacobians_by_pose, dtype=np.float64)
    captured = np.asarray(accepted_skin_world_by_pose, dtype=np.float64).astype("<f4").astype(np.float64)
    if (source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all()
            or source_faces.ndim != 2 or source_faces.shape[1] != 3 or source_faces.size == 0
            or source_faces.min() < 0 or source_faces.max() >= len(source)
            or captured.ndim != 3 or captured.shape[0] < 2 or captured.shape[2] != 3
            or not np.isfinite(captured).all()
            or (maps is None and candidate_forward is None)
            or (candidate_forward is not None and not callable(candidate_forward))
            or (scan_progress_callback is not None and not callable(scan_progress_callback))
            or ((resume_source_positions is None) != (resume_target_audits_by_pose is None))
            or (resume_self_audits_by_pose is not None and resume_source_positions is None)
            or (resume_source_positions is not None and not isinstance(resume_provenance, dict))
            or (maps is not None and (maps.ndim != 4 or maps.shape[0] < 2 or maps.shape[2:] != (3, 3)
                                      or maps.shape[:2] != captured.shape[:2] or not np.isfinite(maps).all()))
            or not np.isfinite(baseline_replay_tolerance_m) or baseline_replay_tolerance_m <= 0.0):
        raise human.ImportError("multi-pose clearance inputs have malformed/non-finite source or pose geometry")
    pose_count, referenced_count = captured.shape[:2]
    if (len(baseline_target_audits_by_pose) != pose_count
            or len(baseline_skin_self_pairs_by_pose) != pose_count
            or any(int(value) != 0 for value in baseline_skin_self_pairs_by_pose)):
        raise human.ImportError("multi-pose baseline audits or zero-self-intersection proof are incomplete")
    if (selected_margin_mm != 0.25 or not np.isfinite(selected_margin_mm)
            or not np.isfinite(support_radius_edge_multiple) or support_radius_edge_multiple <= 0.0
            or max_iterations < 1 or backtrack_count < 1):
        raise human.ImportError("multi-pose clearance requires the reviewed 0.25 mm margin and bounded iteration settings")
    targets, ocular = set(all_target_keys), set(ocular_monitor_keys)
    if not targets or not ocular or not ocular.issubset(targets) or targets == ocular:
        raise human.ImportError("multi-pose clearance target inventory has an invalid ocular partition")
    fixed_ids = np.asarray(fixed_source_vertex_ids, dtype=np.int64)
    anchor_ids = np.asarray(
        np.empty(0, dtype=np.int64) if preserved_source_anchor_vertex_ids is None
        else preserved_source_anchor_vertex_ids,
        dtype=np.int64,
    )
    for label, rows in (("fixed bed/support witness", fixed_ids), ("preserved source anchor", anchor_ids)):
        if (rows.ndim != 1 or len(set(map(int, rows))) != len(rows)
                or (len(rows) and (rows.min() < 0 or rows.max() >= len(source)))):
            raise human.ImportError(f"{label} source vertices are malformed")
    if np.intersect1d(fixed_ids, anchor_ids).size:
        raise human.ImportError("fixed support witnesses and preserved source anchors must have distinct roles")
    all_fixed_ids = np.unique(np.concatenate((fixed_ids, anchor_ids)))
    origins = np.asarray(bed_plane_origins_by_pose, dtype=np.float64)
    bed_normals = np.asarray(bed_plane_normals_by_pose, dtype=np.float64)
    if (origins.shape != (pose_count, 3) or bed_normals.shape != (pose_count, 3)
            or not np.isfinite(origins).all() or not np.isfinite(bed_normals).all()):
        raise human.ImportError("multi-pose bed-plane data is malformed")
    bed_lengths = np.linalg.norm(bed_normals, axis=1)
    if np.any(bed_lengths <= 1.0e-12):
        raise human.ImportError("multi-pose bed-plane normal is zero")
    bed_normals /= bed_lengths[:, None]

    referenced = np.unique(source_faces)
    if len(referenced) != referenced_count:
        raise human.ImportError("pose Jacobian rows do not exactly cover the sorted referenced skin vertices")
    lookup = np.full(len(source), -1, dtype=np.int64)
    lookup[referenced] = np.arange(referenced_count, dtype=np.int64)
    compact_faces = lookup[source_faces]
    fixed = lookup[all_fixed_ids] if len(all_fixed_ids) else np.empty(0, dtype=np.int64)
    if np.any(compact_faces < 0) or (len(fixed) and np.any(fixed < 0)):
        raise human.ImportError("skin topology or fixed support witness is absent from the pose map")
    source_base = source[referenced].astype("<f4").astype(np.float64)
    current_source = source_base.copy()

    def evaluate_forward(source_rows):
        if candidate_forward is None:
            world = (captured + np.einsum("pnij,nj->pni", maps, source_rows - source_base)).astype("<f4").astype(np.float64)
            return world, maps, {}
        full_source = source.copy()
        full_source[referenced] = np.asarray(source_rows, dtype="<f4").astype(np.float64)
        result = candidate_forward(full_source.astype("<f4"))
        if not isinstance(result, dict):
            raise human.ImportError("candidate forward callback must return a mapping")
        if not {"world_positions_by_pose", "jacobians_by_pose"}.issubset(result):
            raise human.ImportError("candidate forward callback omitted world positions or source Jacobians")
        world = np.asarray(result["world_positions_by_pose"], dtype="<f4").astype(np.float64)
        forward_maps = np.asarray(result["jacobians_by_pose"], dtype=np.float64)
        diagnostics = result.get("diagnostics", {})
        if (world.shape != captured.shape or forward_maps.shape != (pose_count, referenced_count, 3, 3)
                or not np.isfinite(world).all() or not np.isfinite(forward_maps).all()
                or not isinstance(diagnostics, dict)):
            raise human.ImportError("candidate forward callback returned malformed/non-finite geometry or Jacobians")
        return world, forward_maps, diagnostics

    base_world, current_maps, initial_forward_diagnostics = evaluate_forward(source_base)
    baseline_replay_error = np.linalg.norm(base_world - captured, axis=2)
    baseline_replay_max = float(baseline_replay_error.max(initial=0.0))
    if candidate_forward is not None and initial_forward_diagnostics.get("admissible", True) is not True:
        raise human.ImportError(
            "candidate forward baseline violates a declared source-map invariant: "
            + str(initial_forward_diagnostics.get("rejection_reason", "unspecified invariant"))
        )
    if baseline_replay_max > baseline_replay_tolerance_m:
        raise human.ImportError(
            "candidate forward model does not reproduce accepted captured skin within its explicit replay bound: "
            f"max_vertex_error_m={baseline_replay_max:.12g}, bound_m={baseline_replay_tolerance_m:.12g}"
        )
    current_forward_diagnostics = initial_forward_diagnostics
    base_triangles = base_world[:, compact_faces]
    base_area = np.cross(
        base_triangles[:, :, 1] - base_triangles[:, :, 0],
        base_triangles[:, :, 2] - base_triangles[:, :, 0],
    )
    base_area_norm = np.linalg.norm(base_area, axis=2)
    if np.any(base_area_norm == 0.0) or not np.isfinite(base_area_norm).all():
        raise human.ImportError("accepted skin has a zero-area/non-finite triangle")
    source_winding_report = _verify_source_winding_in_accepted_poses(
        source[referenced], compact_faces, source_outward_face_signs, current_maps, base_area,
    )
    base_bed_gap = np.einsum("pni,pi->pn", captured - origins[:, None, :], bed_normals)
    bed_floor = np.minimum(base_bed_gap, 0.0)

    def pair_set(row):
        return _triangle_pair_set(row)

    # Cache only complete exact audits of the accepted current geometry.
    # A failed trial must never become the baseline for a later backtrack.
    current_self_audits: list[dict[str, Any] | None] = [None] * pose_count
    compact_face_sha = _face_index_sha256(compact_faces)
    if candidate_forward is not None and not np.array_equal(base_world, captured):
        for pose in range(pose_count):
            records = _exact_surface_records(base_world[pose], compact_faces)
            self_audit = _audit_pair(records, records, same_surface=True)
            current_self_audits[pose] = self_audit
            if int(self_audit["count"]) != 0:
                raise human.ImportError(f"candidate forward baseline has {self_audit['count']} exact self-pairs at pose={pose}")
            replay_audit = scan_candidate_targets(pose, base_world[pose])
            if not isinstance(replay_audit, dict) or set(replay_audit) != targets:
                raise human.ImportError(f"candidate forward baseline scan omitted target coverage at pose={pose}")
            for key in targets:
                if (pair_set(replay_audit[key]) != pair_set(baseline_target_audits_by_pose[pose][key])
                        or int(replay_audit[key].get("count", -1)) != int(baseline_target_audits_by_pose[pose][key].get("count", -2))):
                    raise human.ImportError(
                        f"candidate forward baseline changes the exact captured target-pair set at pose={pose}, surface={key}"
                    )

    def audit_count(audit, keys):
        return sum(int(audit[key]["count"]) for key in keys)

    current_audits = []
    baseline_ocular_sets = []
    for pose, audit in enumerate(baseline_target_audits_by_pose):
        if set(audit) != targets:
            raise human.ImportError(f"baseline pose {pose} does not cover the exact complete target inventory")
        ocular_sets = {}
        for key, row in audit.items():
            if (int(row.get("count", -1)) != len(pair_set(row))
                    or row.get("degenerate_face_rows")):
                raise human.ImportError(f"baseline pose {pose} has inconsistent pairs or degenerate faces for {key}")
            if key in ocular:
                ocular_sets[key] = pair_set(row)
        current_audits.append(audit)
        baseline_ocular_sets.append(ocular_sets)
    current_world = base_world.copy()
    current_counts = [audit_count(row, targets - ocular) for row in current_audits]
    initial_counts = current_counts.copy()
    initial_ocular_counts = [sum(len(pairs) for pairs in row.values()) for row in baseline_ocular_sets]
    resume_start_report = None
    if resume_source_positions is not None:
        resume_full = np.asarray(resume_source_positions, dtype=np.float64)
        if resume_full.shape != source.shape or not np.isfinite(resume_full).all():
            raise human.ImportError("resume source positions are malformed or non-finite")
        unreferenced = np.setdiff1d(np.arange(len(source), dtype=np.int64), referenced, assume_unique=True)
        if len(unreferenced) and not np.array_equal(
                resume_full[unreferenced].astype("<f4"), source[unreferenced].astype("<f4")):
            raise human.ImportError("resume source changed unreferenced NHSKIN positions")
        resume_rows = resume_full[referenced].astype("<f4").astype(np.float64)
        expected_source_sha = resume_provenance.get("source_positions_f32_sha256")
        if (not isinstance(expected_source_sha, str) or len(expected_source_sha) != 64
                or hashlib.sha256(np.asarray(resume_rows, dtype="<f4").tobytes()).hexdigest() != expected_source_sha):
            raise human.ImportError("resume source positions do not match caller hash provenance")
        for key in ("source_positions_path", "target_audits_path", "target_audits_sha256"):
            if not isinstance(resume_provenance.get(key), str) or not resume_provenance[key]:
                raise human.ImportError(f"resume source provenance omits {key}")
        if len(fixed) and not np.array_equal(resume_rows[fixed], source_base[fixed]):
            raise human.ImportError("resume candidate moved an exact bed/support witness or preserved anchor")
        resume_world, resume_maps, resume_diagnostics = evaluate_forward(resume_rows)
        if resume_diagnostics.get("admissible", True) is not True:
            raise human.ImportError(
                "resume candidate violates a declared source-map invariant: "
                + str(resume_diagnostics.get("rejection_reason", "unspecified invariant"))
            )
        resume_self_pair_counts = []
        self_cache_used = resume_self_audits_by_pose is not None
        self_cache_sha256 = None
        if self_cache_used:
            if (not isinstance(resume_provenance.get("self_audits_path"), str)
                    or not resume_provenance["self_audits_path"]
                    or not isinstance(resume_provenance.get("self_audits_sha256"), str)
                    or len(resume_provenance["self_audits_sha256"]) != 64):
                raise human.ImportError("resume provenance omits the accepted self-audit artifact identity")
            if any(ch not in "0123456789abcdef" for ch in resume_provenance["self_audits_sha256"]):
                raise human.ImportError("resume self-audit artifact SHA-256 is malformed")
            validated_self_audits, self_cache_sha256 = _validate_resumed_skin_self_audits(
                resume_self_audits_by_pose, pose_count=pose_count, resume_world=resume_world,
                source_sha256=expected_source_sha, faces=compact_faces,
            )
            if self_cache_sha256 != resume_provenance["self_audits_sha256"]:
                raise human.ImportError("resume self-audit artifact SHA-256 does not match its proof rows")
            for pose, self_audit in enumerate(validated_self_audits):
                current_self_audits[pose] = self_audit
                resume_self_pair_counts.append(0)
        else:
            for pose in range(pose_count):
                resume_records = _exact_surface_records(resume_world[pose], compact_faces)
                self_audit = _audit_pair(resume_records, resume_records, same_surface=True)
                current_self_audits[pose] = self_audit
                count = int(self_audit["count"])
                resume_self_pair_counts.append(count)
                if count:
                    raise human.ImportError(
                        f"resume candidate has {count} exact skin self-intersections at pose={pose}"
                    )
        resume_world_f32_sha256 = hashlib.sha256(
            np.asarray(resume_world, dtype="<f4").tobytes()
        ).hexdigest()
        resume_triangles = resume_world[:, compact_faces]
        resume_area = np.cross(resume_triangles[:, :, 1] - resume_triangles[:, :, 0],
                               resume_triangles[:, :, 2] - resume_triangles[:, :, 0])
        resume_area_norm = np.linalg.norm(resume_area, axis=2)
        resume_orientation = np.einsum("pfi,pfi->pf", base_area, resume_area) / (
            base_area_norm * np.maximum(resume_area_norm, np.finfo(np.float64).tiny)
        )
        if (not np.isfinite(resume_area_norm).all() or np.any(resume_area_norm == 0.0)
                or not np.isfinite(resume_orientation).all() or float(resume_orientation.min()) <= 0.0):
            raise human.ImportError("resume candidate violates immutable baseline nonzero-area/source-winding gates")
        resume_bed_gap = np.einsum("pni,pi->pn", resume_world - origins[:, None, :], bed_normals)
        if not np.isfinite(resume_bed_gap).all() or np.any(resume_bed_gap < bed_floor):
            raise human.ImportError("resume candidate worsens an immutable baseline bed-plane gap")
        if len(resume_target_audits_by_pose) != pose_count:
            raise human.ImportError("resume exact target audits do not cover every accepted pose")
        resume_audits = []
        for pose, audit in enumerate(resume_target_audits_by_pose):
            if not isinstance(audit, dict) or set(audit) != targets:
                raise human.ImportError(f"resume exact target inventory is incomplete at pose={pose}")
            ocular_sets = {}
            for key, row in audit.items():
                if (int(row.get("count", -1)) != len(pair_set(row))
                        or row.get("degenerate_face_rows")):
                    raise human.ImportError(f"resume exact target audit is malformed at pose={pose}, surface={key}")
                if key in ocular:
                    ocular_sets[key] = pair_set(row)
            if any(ocular_sets[key] != baseline_ocular_sets[pose][key] for key in ocular):
                raise human.ImportError(f"resume candidate changed an exact ocular pair set at pose={pose}")
            resume_audits.append(audit)
        current_source = resume_rows
        current_world = resume_world
        current_maps = resume_maps
        current_forward_diagnostics = resume_diagnostics
        current_audits = resume_audits
        current_counts = [audit_count(row, targets - ocular) for row in current_audits]
        resume_start_report = {
            **resume_provenance,
            "source_positions_f32_sha256": expected_source_sha,
            "nonocular_pair_count_by_pose": [int(value) for value in current_counts],
            "baseline_orientation_minimum_dot": float(resume_orientation.min()),
            "baseline_minimum_triangle_area_ratio": float((resume_area_norm / base_area_norm).min()),
            "baseline_minimum_bed_gap_m_by_pose": [float(row.min()) for row in resume_bed_gap],
            "exact_skin_self_pair_count_by_pose": resume_self_pair_counts,
            "resume_world_positions_f32_sha256": resume_world_f32_sha256,
            "resume_self_audit_predicate": "_audit_pair(same_surface=True) over each complete resumed accepted-pose skin",
            "resume_self_audit_cache": {
                "used": bool(self_cache_used),
                "proof_sha256": self_cache_sha256,
                "full_initial_pose_scans_skipped": int(pose_count if self_cache_used else 0),
                "clearance_owner_source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                "exact_predicate_source_sha256": _module_source_sha256(ci, "exact self-intersection predicate"),
            },
            "status": "caller-hash-bound-accepted-candidate-revalidated-against-immutable-baseline-gates",
        }
    closed_targets = [dict() for _ in range(pose_count)]
    if (closed_target_outer_envelope_keys is not None
            and (not isinstance(closed_target_outer_envelope_keys, (set, frozenset))
                 or any(not isinstance(key, str) for key in closed_target_outer_envelope_keys))):
        raise human.ImportError("closed-target outer-envelope keys must be an explicit set of surface keys")
    outer_envelope_keys = set(closed_target_outer_envelope_keys or ())
    if outer_envelope_keys and closed_target_surfaces_by_pose is None:
        raise human.ImportError("closed-target outer-envelope keys require captured meshes")
    if closed_target_surfaces_by_pose is None and closed_target_face_counts is not None:
        raise human.ImportError("closed-target face counts require captured meshes")
    if closed_target_surfaces_by_pose is not None:
        if (not isinstance(closed_target_surfaces_by_pose, list)
                or len(closed_target_surfaces_by_pose) != pose_count
                or any(not isinstance(row, dict) for row in closed_target_surfaces_by_pose)):
            raise human.ImportError("closed-target meshes must cover every supplied pose")
        closed_keys = set(closed_target_surfaces_by_pose[0])
        if not outer_envelope_keys.issubset(closed_keys):
            raise human.ImportError("closed-target outer-envelope keys must be selected captured targets")
        if (not closed_keys or not closed_keys.issubset(targets - ocular)
                or any(set(row) != closed_keys for row in closed_target_surfaces_by_pose)):
            raise human.ImportError("closed-target meshes must name the same nonocular subset in every pose")
        if (not isinstance(closed_target_face_counts, dict)
                or set(closed_target_face_counts) != closed_keys
                or any(type(count) is not int or count <= 0
                       for count in closed_target_face_counts.values())):
            raise human.ImportError("closed-target meshes require complete captured face counts")
        for pose, row in enumerate(closed_target_surfaces_by_pose):
            for key, mesh in sorted(row.items()):
                if not isinstance(mesh, (tuple, list)) or len(mesh) != 2:
                    raise human.ImportError(f"closed-target mesh is malformed at pose={pose}, surface={key}")
                vertices, target_faces = mesh
                if len(target_faces) != closed_target_face_counts[key]:
                    raise human.ImportError(
                        f"closed-target mesh has incomplete captured face coverage at pose={pose}, surface={key}")
                prepared = _prepare_closed_clearance_target(
                    vertices, target_faces, allow_nested_enclosure=key in outer_envelope_keys)
                triangles = prepared["vertices"][prepared["faces"]]
                for face_id, triangle in enumerate(triangles):
                    authoritative = np.asarray(target_triangle_by_row(pose, key, face_id), dtype=np.float64)
                    if (authoritative.shape != (3, 3) or not np.isfinite(authoritative).all()
                            or not np.array_equal(authoritative, authoritative.astype("<f4").astype(np.float64))
                            or not np.array_equal(triangle, authoritative)):
                        raise human.ImportError(
                            f"closed-target mesh differs from captured target at pose={pose}, "
                            f"surface={key}, face={face_id}")
                closed_targets[pose][key] = prepared

    def classify_closed_targets(world):
        return [
            {key: _closed_target_inside_vertices(world[pose], target)
             for key, target in sorted(closed_targets[pose].items())}
            for pose in range(pose_count)
        ]

    def interior_counts(rows):
        return [sum(len(ids) for ids in row.values()) for row in rows]

    initial_interiors = classify_closed_targets(captured)
    current_interiors = (initial_interiors if np.array_equal(current_world, captured)
                         else classify_closed_targets(current_world))
    initial_interior_counts = interior_counts(initial_interiors)
    current_interior_counts = interior_counts(current_interiors)
    edge_rows = np.concatenate((compact_faces[:, [0, 1]], compact_faces[:, [1, 2]], compact_faces[:, [2, 0]]))
    edge_rows.sort(axis=1)
    edges = np.unique(edge_rows, axis=0)
    graph_rows, graph_cols = np.concatenate((edges[:, 0], edges[:, 1])), np.concatenate((edges[:, 1], edges[:, 0]))
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    iteration_rows = []
    winding_receipt = {}
    nonocular = targets - ocular
    attempt_index = 0

    def emit_trial(iteration, backtrack, scale, status, reason, trial_source, counts=None, audits=None,
                   diagnostics=None, self_audits=None, self_audit_world_by_pose=None):
        nonlocal attempt_index
        if progress_callback is None:
            return
        attempt_index += 1
        packed_source = np.asarray(trial_source, dtype="<f4").copy()
        audit_rows = None
        if audits is not None:
            audit_rows = [
                {
                    key: {
                        "triangle_pairs": [[int(face), int(target)] for face, target in row.get("triangle_pairs", [])],
                        "count": int(row.get("count", -1)),
                        "aabb_candidate_pairs": int(row.get("aabb_candidate_pairs", 0)),
                        "degenerate_face_rows": [int(face) for face in row.get("degenerate_face_rows", [])],
                    }
                    for key, row in audit.items()
                }
                for audit in audits
            ]
        accepted_self_rows = None
        accepted_self_rows_sha256 = None
        if status == "accepted":
            if self_audits is None or self_audit_world_by_pose is None:
                raise human.ImportError("accepted trial event omitted its exact self-audit rows or pose worlds")
            accepted_self_rows = _serialize_accepted_skin_self_audits(
                self_audits, self_audit_world_by_pose,
                hashlib.sha256(packed_source.tobytes(order="C")).hexdigest(), compact_faces,
            )
            accepted_self_rows_sha256 = _accepted_skin_self_audits_sha256(accepted_self_rows)
        progress_callback({
            "attempt": attempt_index,
            "iteration": iteration + 1,
            "backtrack": backtrack,
            "scale": float(scale),
            "status": status,
            "reason": reason,
            "nonocular_pair_counts_before_by_pose": [int(value) for value in current_counts],
            "nonocular_pair_counts_candidate_by_pose": None if counts is None else [int(value) for value in counts],
            "nonocular_pair_counts_by_pose": None if counts is None else [int(value) for value in counts],
            "target_audits_by_pose": audit_rows,
            "self_audits_by_pose": accepted_self_rows,
            "self_audits_sha256": accepted_self_rows_sha256,
            "candidate_diagnostics": diagnostics,
            "direction_metrics": direction_report,
            "closed_target_interior_counts_before_by_pose": current_interior_counts,
            "closed_target_interior_counts_candidate_by_pose": trial_interior_counts,
            "source_positions_f32_sha256": hashlib.sha256(packed_source.tobytes()).hexdigest(),
            "source_positions_f32": packed_source,
        })

    def audit_candidate_self(candidate_world, *, reuse_world=None, reuse_audits=None):
        if (reuse_world is None) != (reuse_audits is None):
            raise human.ImportError("self audit reuse requires both an exact world and its completed audit rows")
        if reuse_world is not None and (
                np.asarray(reuse_world).shape != current_world.shape or len(reuse_audits) != pose_count):
            raise human.ImportError("self audit reuse rows do not cover the accepted poses")
        audits = []
        failures = []
        for pose in range(pose_count):
            baseline_self = current_self_audits[pose]
            if baseline_self is None:
                records = _exact_surface_records(current_world[pose], compact_faces)
                baseline_self = _audit_pair(records, records, same_surface=True)
                if int(baseline_self["count"]) != 0:
                    raise human.ImportError(
                        f"current accepted skin has {baseline_self['count']} exact self-pairs at pose={pose}"
                    )
                current_self_audits[pose] = baseline_self
            if reuse_world is not None and np.array_equal(candidate_world[pose], reuse_world[pose]):
                self_audit = reuse_audits[pose]
            elif np.array_equal(candidate_world[pose], current_world[pose]):
                self_audit = baseline_self
            else:
                self_audit = audit_incremental_skin_self_intersections(
                    baseline_world_positions=current_world[pose],
                    candidate_world_positions=candidate_world[pose],
                    baseline_faces=compact_faces,
                    candidate_faces=compact_faces,
                    baseline_self_audit=baseline_self,
                    expected_baseline_world_f32_sha256=_float32_xyz_sha256(current_world[pose]),
                    expected_face_index_sha256=compact_face_sha,
                    expected_baseline_self_pair_table_sha256=_baseline_self_pair_table_sha256(
                        baseline_self, len(compact_faces),
                    ),
                )
            audits.append(self_audit)
            if int(self_audit.get("count", -1)) != len(self_audit.get("triangle_pairs", [])):
                raise human.ImportError(f"candidate exact self audit is malformed at pose={pose}")
            if self_audit.get("degenerate_face_rows"):
                raise human.ImportError(f"candidate exact self audit found degenerate faces at pose={pose}")
            if int(self_audit["count"]):
                failures.append({
                    "pose_index": pose,
                    "triangle_pairs": [tuple(map(int, pair)) for pair in self_audit["triangle_pairs"]],
                })
        return audits, failures

    for iteration in range(max_iterations):
        before_by_pose = current_counts.copy()
        before_total = sum(before_by_pose)
        before_inside = sum(current_interior_counts)
        if before_total == 0 and before_inside == 0:
            break
        pose_normals = np.stack([_area_weighted_vertex_normals(current_world[p], compact_faces)
                                 for p in range(pose_count)])
        smooth_normals = np.stack([_smooth_vertex_directions(pose_normals[p], compact_faces)
                                   for p in range(pose_count)])
        source_directions, _, _, direction_report = _shared_source_directions_from_pose_normals(
            current_maps, smooth_normals,
        )
        active_face_set = {
            int(face) for audit in current_audits for key in nonocular
            for face, _ in audit[key]["triangle_pairs"]
        }
        interior_lists = [ids for row in current_interiors for ids in row.values() if len(ids)]
        interior_vertices = (np.unique(np.concatenate(interior_lists))
                             if interior_lists else np.empty(0, dtype=np.int64))
        if len(interior_vertices):
            active_face_set.update(map(int, np.flatnonzero(
                np.any(np.isin(compact_faces, interior_vertices), axis=1))))
        active_faces = sorted(active_face_set)
        if not active_faces:
            raise human.ImportError("nonocular boundary/interior defects have no active skin faces")
        active = np.asarray(active_faces, dtype=np.int64)
        active_compact = compact_faces[active]
        active_triangles = current_world[:, active_compact]
        active_areas = np.cross(active_triangles[:, :, 1] - active_triangles[:, :, 0],
                                active_triangles[:, :, 2] - active_triangles[:, :, 0])
        active_area_lengths = np.linalg.norm(active_areas, axis=2)
        if np.any(active_area_lengths <= 0.0) or not np.isfinite(active_area_lengths).all():
            raise human.ImportError(f"active source triangle degenerated during iteration {iteration + 1}")
        active_normals = active_areas / active_area_lengths[:, :, None]
        face_column = {face: column for column, face in enumerate(active_faces)}
        required = np.zeros((pose_count, len(active)), dtype=np.float64)
        for pose in range(pose_count):
            for key in nonocular:
                for face_value, target_face_value in current_audits[pose][key]["triangle_pairs"]:
                    face, target_face = int(face_value), int(target_face_value)
                    target_triangle = np.asarray(target_triangle_by_row(pose, key, target_face), dtype=np.float64)
                    if target_triangle.shape != (3, 3) or not np.isfinite(target_triangle).all():
                        raise human.ImportError(f"target face lookup is invalid at pose={pose}, surface={key}, face={target_face}")
                    skin_triangle = current_world[pose, compact_faces[face]]
                    column = face_column[face]
                    distance = _triangle_normal_translation_to_separate(
                        skin_triangle, target_triangle, active_normals[pose, column],
                    ) + selected_margin_mm / 1000.0
                    required[pose, column] = max(required[pose, column], distance)
        source_directions, conditioning_report = _condition_shared_source_directions(
            current_maps, source_directions, active_compact, active_normals,
        )
        direction_report["active_face_conditioning"] = conditioning_report
        seed_required, demand_report = _shared_source_seed_demands(
            current_maps, source_directions, active_compact, active_normals, required,
        )
        interior_demand_rows = []
        for pose, row in enumerate(current_interiors):
            for key, ids in sorted(row.items()):
                demands = []
                for vertex in ids:
                    direction = current_maps[pose, vertex] @ source_directions[vertex]
                    demand = _closed_target_exit_source_scalar(
                        current_world[pose, vertex], direction, closed_targets[pose][key],
                        margin_m=selected_margin_mm / 1000.0)
                    seed_required[vertex] = max(seed_required[vertex], demand)
                    demands.append(demand)
                interior_demand_rows.append({
                    "pose": pose, "target": key, "inside_vertex_count": len(ids),
                    "maximum_source_scalar_m": max(demands, default=0.0),
                })
        direction_report["closed_target_interior_demands"] = interior_demand_rows
        direction_report["interior_seed_source_vertex_ids"] = referenced[interior_vertices].tolist()
        seeds = np.flatnonzero(seed_required > 0.0)
        if not len(seeds):
            raise human.ImportError(f"iteration {iteration + 1} produced no source-space correction seeds")
        edge_lengths = _maximum_pose_edge_lengths(current_world, edges)
        touching = np.isin(edges[:, 0], seeds) | np.isin(edges[:, 1], seeds)
        if not np.any(touching) or np.any(edge_lengths <= 0.0):
            raise human.ImportError(f"iteration {iteration + 1} has no valid multi-pose support edges")
        radius = support_radius_edge_multiple * float(np.median(edge_lengths[touching]))
        graph = csr_matrix(
            (np.concatenate((edge_lengths, edge_lengths)), (graph_rows, graph_cols)),
            shape=(referenced_count, referenced_count),
        )
        field = np.zeros(referenced_count, dtype=np.float64)
        for start in range(0, len(seeds), _GEODESIC_SEED_BATCH_SIZE):
            batch = seeds[start:start + _GEODESIC_SEED_BATCH_SIZE]
            distance = dijkstra(graph, directed=False, indices=batch, limit=radius)
            unit = np.clip(distance / radius, 0.0, 1.0)
            envelope = 1.0 - 3.0 * unit * unit + 2.0 * unit * unit * unit
            envelope[distance >= radius] = 0.0
            np.maximum(field, np.max(seed_required[batch, None] * envelope, axis=0), out=field)
        if len(fixed):
            field[fixed] = 0.0
        if not np.isfinite(field).all() or not np.any(field > 0.0):
            raise human.ImportError(f"iteration {iteration + 1} has no movable correction field after support pins")

        accepted = False
        rejection = "no candidate trial"
        for backtrack in range(backtrack_count):
            scale = 0.5 ** backtrack
            trial_regularization = None
            trial_interior_counts = None
            raw_trial_source = (current_source + scale * field[:, None] * source_directions).astype("<f4").astype(np.float64)
            trial_source = raw_trial_source
            if len(fixed) and not np.array_equal(trial_source[fixed], source_base[fixed]):
                rejection = "trial changed an exact bed/support witness source vertex"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source)
                continue
            if candidate_forward is None:
                trial_delta = trial_source - source_base
                trial_world = (captured + np.einsum("pnij,nj->pni", current_maps, trial_delta)).astype("<f4").astype(np.float64)
                trial_maps, trial_forward_diagnostics = current_maps, {}
            else:
                trial_world, trial_maps, trial_forward_diagnostics = evaluate_forward(trial_source)
                if trial_forward_diagnostics.get("admissible", True) is not True:
                    rejection = ("candidate forward model violates a declared source-map invariant: "
                                 + str(trial_forward_diagnostics.get("rejection_reason", "unspecified invariant")))
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source)
                    continue
            if np.array_equal(trial_world, current_world):
                rejection = "Float32 source packing produced no new accepted-pose coordinates"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source)
                continue
            tri = trial_world[:, compact_faces]
            area = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
            area_norm = np.linalg.norm(area, axis=2)
            if not np.isfinite(area_norm).all():
                rejection = "trial contains non-finite skin triangle areas"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source)
                continue
            orientation_numerator = np.einsum("pfi,pfi->pf", base_area, area)
            orientation_denominator = base_area_norm * area_norm
            orientation = np.zeros_like(area_norm)
            np.divide(orientation_numerator, orientation_denominator, out=orientation,
                      where=orientation_denominator > 0.0)
            folded_faces = np.flatnonzero(np.any((area_norm == 0.0) | (orientation <= 0.0), axis=0))
            if len(folded_faces):
                fold_vertices = np.unique(compact_faces[folded_faces].reshape(-1))
                seed_touch = np.intersect1d(fold_vertices, seeds)
                active_face_touch = np.intersect1d(folded_faces, active)
                if len(seed_touch) or len(active_face_touch):
                    rejection = (
                        "collateral fold touches required active geometry: "
                        f"faces={folded_faces[:16].tolist()}, seed_vertices={seed_touch[:16].tolist()}, "
                        f"active_faces={active_face_touch[:16].tolist()}"
                    )
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                               diagnostics={"folded_face_count": int(len(folded_faces)),
                                            "required_seed_vertices_touched": seed_touch.tolist(),
                                            "active_faces_touched": active_face_touch.tolist()})
                    continue
                try:
                    regularized_increment, trial_regularization = _smooth_collateral_displacement_near_faces(
                        trial_source - current_source,
                        graph,
                        compact_faces,
                        folded_faces,
                        seeds,
                        fixed,
                    )
                    if len(seeds) and not np.array_equal(
                            regularized_increment[seeds], (raw_trial_source - current_source)[seeds]):
                        raise human.ImportError("local smoothing changed a required source-seed increment")
                    trial_source = (current_source + regularized_increment).astype("<f4").astype(np.float64)
                    if len(fixed) and not np.array_equal(trial_source[fixed], source_base[fixed]):
                        raise human.ImportError("local smoothing moved a fixed bed/support or preserved anchor")
                    if np.array_equal(trial_source, current_source):
                        raise human.ImportError("local smoothing produced no source-space correction")
                    if candidate_forward is None:
                        trial_delta = trial_source - source_base
                        trial_world = (captured + np.einsum("pnij,nj->pni", current_maps, trial_delta)).astype("<f4").astype(np.float64)
                        trial_maps, trial_forward_diagnostics = current_maps, {}
                    else:
                        trial_world, trial_maps, trial_forward_diagnostics = evaluate_forward(trial_source)
                        if trial_forward_diagnostics.get("admissible", True) is not True:
                            raise human.ImportError(
                                "regularized candidate violates a source-map invariant: "
                                + str(trial_forward_diagnostics.get("rejection_reason", "unspecified invariant"))
                            )
                    tri = trial_world[:, compact_faces]
                    area = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
                    area_norm = np.linalg.norm(area, axis=2)
                    if not np.isfinite(area_norm).all():
                        raise human.ImportError("regularized candidate contains non-finite skin triangle areas")
                    orientation_numerator = np.einsum("pfi,pfi->pf", base_area, area)
                    orientation_denominator = base_area_norm * area_norm
                    orientation = np.zeros_like(area_norm)
                    np.divide(orientation_numerator, orientation_denominator, out=orientation,
                              where=orientation_denominator > 0.0)
                    remaining_folded = np.flatnonzero(np.any((area_norm == 0.0) | (orientation <= 0.0), axis=0))
                    if len(remaining_folded):
                        trial_regularization["remaining_folded_face_rows"] = [int(value) for value in remaining_folded]
                        raise human.ImportError(
                            f"local taper did not restore strict all-pose area/orientation for faces {remaining_folded[:16].tolist()}"
                        )
                    trial_regularization["remaining_folded_face_rows"] = []
                    trial_regularization["regularized_candidate_orientation_minimum_dot"] = float(orientation.min())
                    trial_regularization["regularized_candidate_minimum_area_ratio"] = float((area_norm / base_area_norm).min())
                except human.ImportError as error:
                    rejection = f"collateral-fold local taper rejected: {error}"
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                               diagnostics=trial_regularization)
                    continue
            if np.any(area_norm == 0.0):
                rejection = "trial contains exact zero-area skin triangles after local regularity check"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                           diagnostics=trial_regularization)
                continue
            if not np.isfinite(orientation).all() or float(orientation.min()) <= 0.0:
                rejection = "trial reversed an accepted-pose source-winding face after local regularity check"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                           diagnostics=trial_regularization)
                continue
            trial_bed_gap = np.einsum("pni,pi->pn", trial_world - origins[:, None, :], bed_normals)
            if np.any(trial_bed_gap < bed_floor):
                pose, vertex = np.unravel_index(int(np.argmin(trial_bed_gap - bed_floor)), trial_bed_gap.shape)
                rejection = (f"trial introduced/worsened bed penetration at pose={pose}, "
                             f"source_vertex={int(referenced[vertex])}")
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, diagnostics=trial_regularization)
                continue
            trial_self_audits, self_failures = audit_candidate_self(trial_world)
            local_self_separation = None
            if self_failures:
                failed_pairs = [
                    (int(failure["pose_index"]), np.asarray(failure["triangle_pairs"], dtype=np.int64).reshape(-1, 2))
                    for failure in self_failures
                ]
                required_target_seed_ids = referenced[seeds]
                active_target_ids = referenced[np.unique(active_compact).reshape(-1)]
                protected_self_ids = np.unique(np.concatenate((
                    required_target_seed_ids, active_target_ids, all_fixed_ids,
                )))
                full_trial_source = source.astype("<f4").copy()
                full_trial_source[referenced] = trial_source.astype("<f4")
                coordinate_tied_ids = _exact_coordinate_tied_source_vertex_ids(full_trial_source)
                seed_blocked_ids = np.union1d(protected_self_ids, coordinate_tied_ids)
                selected_seed_vertices_by_face = {}
                protected_pair_side_faces = set()
                pair_selection_receipts = []
                ineligible_pair = None
                source_displacement = trial_source - current_source
                for pose, pairs in failed_pairs:
                    for face_a_value, face_b_value in pairs:
                        face_a, face_b = int(face_a_value), int(face_b_value)
                        sides = (face_a, face_b)
                        free_seed_vertices_by_face = {
                            face: _free_self_pair_seed_vertex_ids(
                                source_faces[face], seed_blocked_ids,
                            )
                            for face in sides
                        }
                        eligible = [face for face in sides if len(free_seed_vertices_by_face[face])]
                        if not eligible:
                            ineligible_pair = {
                                "pose_index": pose,
                                "triangle_pair": [face_a, face_b],
                                "protected_side_faces": [face_a, face_b],
                                "coordinate_tied_source_vertex_ids": coordinate_tied_ids.tolist(),
                            }
                            break
                        # Compare only vertices that can actually move. Protected
                        # target-core, support, anchor, and coordinate-tied rows stay fixed.
                        movement_cost = {}
                        for face in eligible:
                            compact_seed_vertices = lookup[free_seed_vertices_by_face[face]]
                            if np.any(compact_seed_vertices < 0):
                                raise human.ImportError(
                                    "self-pair seed vertex is absent from the sorted compact source map"
                                )
                            movement_cost[face] = float(np.sum(
                                source_displacement[compact_seed_vertices] ** 2
                            ))
                        selected_face = min(eligible, key=lambda face: (movement_cost[face], face))
                        selected_seed_vertices_by_face.setdefault(selected_face, set()).update(
                            map(int, free_seed_vertices_by_face[selected_face])
                        )
                        # Preserve the prior boundary: only a fully ineligible side
                        # is blocked. An eligible but unselected side may still enter
                        # the local patch if it is adjacent to a selected seed.
                        protected_sides = [face for face in sides if face not in eligible]
                        protected_pair_side_faces.update(protected_sides)
                        pair_selection_receipts.append({
                            "pose_index": int(pose),
                            "triangle_pair": [face_a, face_b],
                            "eligible_seed_faces": eligible,
                            "eligible_seed_vertex_ids_by_face": [
                                {"face_row": int(face),
                                 "vertex_ids": free_seed_vertices_by_face[face].tolist()}
                                for face in eligible
                            ],
                            "selected_seed_face": int(selected_face),
                            "selected_seed_vertex_ids": sorted(
                                map(int, free_seed_vertices_by_face[selected_face])
                            ),
                            "selected_face_source_displacement_squared_m2": movement_cost[selected_face],
                            "protected_pair_side_faces": protected_sides,
                        })
                    if ineligible_pair is not None:
                        break
                if ineligible_pair is not None:
                    rejection = (
                        "local self-separation rejected: neither face in an exact self-pair "
                        "has a free vertex for a movable seed"
                    )
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                               diagnostics={
                                   "self_pair_failures": [
                                       {"pose_index": pose, "triangle_pairs": pairs.tolist()}
                                       for pose, pairs in failed_pairs
                                   ],
                                   "ineligible_pair": ineligible_pair,
                               })
                    continue
                self_seed_ids = np.asarray(sorted({
                    vertex for rows in selected_seed_vertices_by_face.values() for vertex in rows
                }), dtype=np.int64)
                protected_pair_side_ids = np.unique(np.concatenate([
                    source_faces[face] for face in sorted(protected_pair_side_faces)
                ])) if protected_pair_side_faces else np.empty(0, dtype=np.int64)
                shared_protected_side_ids = np.intersect1d(self_seed_ids, protected_pair_side_ids)
                if len(shared_protected_side_ids):
                    rejection = (
                        "local self-separation rejected: selected seed faces overlap vertices of a "
                        "protected pair-side face"
                    )
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                               diagnostics={
                                   "shared_protected_pair_side_vertex_ids": shared_protected_side_ids.tolist(),
                                   "pair_seed_face_selection": pair_selection_receipts,
                                   "protected_pair_side_face_rows": sorted(protected_pair_side_faces),
                               })
                    continue
                patch_blocked_ids = np.union1d(seed_blocked_ids, protected_pair_side_ids)
                try:
                    source_edge_rows = referenced[edges]
                    adjacency = {}
                    for first, second in source_edge_rows:
                        first, second = int(first), int(second)
                        adjacency.setdefault(first, set()).add(second)
                        adjacency.setdefault(second, set()).add(first)
                    movable_set = set(map(int, self_seed_ids))
                    frontier = set(movable_set)
                    patch_blocked_set = set(map(int, patch_blocked_ids))
                    for _ring in range(2):
                        next_frontier = {
                            neighbor for vertex in frontier for neighbor in adjacency.get(vertex, ())
                            if neighbor not in movable_set and neighbor not in patch_blocked_set
                        }
                        movable_set.update(next_frontier)
                        frontier = next_frontier
                        if len(movable_set) > _LOCAL_SELF_SEPARATION_MAX_MOVABLE_VERTICES:
                            raise human.ImportError(
                                "local self-separation two-ring patch exceeds the helper's 256-vertex bound"
                            )
                    movable_ids = np.asarray(sorted(movable_set), dtype=np.int64)
                    if len(movable_ids) > _LOCAL_SELF_SEPARATION_MAX_MOVABLE_VERTICES:
                        raise human.ImportError(
                            "local self-separation patch exceeds the helper's 256-vertex bound"
                        )
                    pose_samples = []
                    for pose, pair_rows in failed_pairs:
                        pose_samples.append({
                            "accepted_pose_index": int(pose),
                            "vertex_ids": referenced,
                            "baseline_world_positions": current_world[pose].astype("<f4"),
                            "rejected_world_positions": trial_world[pose].astype("<f4"),
                            "jacobians": trial_maps[pose],
                            "self_pair_rows": pair_rows,
                        })
                    preproposal_world = trial_world.copy()
                    preproposal_self_audits = trial_self_audits
                    proposal = _propose_local_self_separation_increment(
                        source_positions_m=full_trial_source,
                        faces=source_faces,
                        movable_vertex_ids=movable_ids,
                        self_patch_seed_vertex_ids=self_seed_ids,
                        required_target_seed_vertex_ids=required_target_seed_ids,
                        active_target_face_vertex_ids=active_target_ids,
                        fixed_support_vertex_ids=fixed_ids,
                        preserved_anchor_vertex_ids=anchor_ids,
                        pose_samples=pose_samples,
                    )
                    if (not isinstance(proposal, dict)
                            or proposal.get("status") != "bounded_local_linearized_proposal"):
                        raise human.ImportError("local self-separation returned no recognized proposal record")
                    raw_constraint_rows = proposal.get("constraint_rows_by_pose")
                    if (not isinstance(raw_constraint_rows, (list, tuple))
                            or len(raw_constraint_rows) != len(failed_pairs)):
                        raise human.ImportError(
                            "local self-separation constraint counts do not match the failing-pose samples"
                        )
                    constraint_rows_by_pose = [0] * pose_count
                    for sample_index, row_count in enumerate(raw_constraint_rows):
                        if (isinstance(row_count, (bool, np.bool_))
                                or not isinstance(row_count, (int, np.integer)) or int(row_count) < 0):
                            raise human.ImportError("local self-separation returned malformed per-pose constraint counts")
                        accepted_pose_index = failed_pairs[sample_index][0]
                        constraint_rows_by_pose[accepted_pose_index] = int(row_count)
                    expected_pairs_by_sample = [
                        {tuple(map(int, pair)) for pair in pairs.tolist()}
                        for _, pairs in failed_pairs
                    ]
                    seen_pairs_by_sample = [set() for _ in failed_pairs]
                    remapped_pair_receipts = []
                    raw_pair_receipts = proposal.get("pair_pose_receipts", [])
                    if (not isinstance(raw_pair_receipts, (list, tuple))
                            or len(raw_pair_receipts) != sum(map(len, expected_pairs_by_sample))):
                        raise human.ImportError(
                            "local self-separation pair receipts do not cover the supplied pose/pair rows"
                        )
                    for receipt in raw_pair_receipts:
                        if not isinstance(receipt, dict):
                            raise human.ImportError("local self-separation returned a malformed pair receipt")
                        sample_index = receipt.get("pose_index")
                        if (isinstance(sample_index, (bool, np.bool_))
                                or not isinstance(sample_index, (int, np.integer))
                                or int(sample_index) < 0 or int(sample_index) >= len(failed_pairs)):
                            raise human.ImportError("local self-separation pair receipt has an invalid sample index")
                        pair_raw = receipt.get("face_pair")
                        if (not isinstance(pair_raw, (list, tuple)) or len(pair_raw) != 2
                                or any(isinstance(value, (bool, np.bool_))
                                       or not isinstance(value, (int, np.integer)) for value in pair_raw)):
                            raise human.ImportError("local self-separation pair receipt has a malformed face pair")
                        face_pair = tuple(map(int, pair_raw))
                        sample_index = int(sample_index)
                        if (face_pair not in expected_pairs_by_sample[sample_index]
                                or face_pair in seen_pairs_by_sample[sample_index]):
                            raise human.ImportError(
                                "local self-separation pair receipt does not match a unique supplied pose/pair row"
                            )
                        seen_pairs_by_sample[sample_index].add(face_pair)
                        remapped_pair_receipts.append({
                            **receipt,
                            "sample_index": sample_index,
                            "pose_index": int(failed_pairs[sample_index][0]),
                        })
                    if seen_pairs_by_sample != expected_pairs_by_sample:
                        raise human.ImportError(
                            "local self-separation pair receipts omit supplied pose/pair rows"
                        )
                    raw_proposed_source = np.asarray(proposal.get("proposal_source_positions_m_f32"))
                    if (raw_proposed_source.dtype.kind != "f" or raw_proposed_source.dtype.itemsize != 4
                            or raw_proposed_source.shape != full_trial_source.shape):
                        raise human.ImportError("local self-separation did not return matching Float32 source positions")
                    proposed_full_source = np.ascontiguousarray(raw_proposed_source, dtype="<f4")
                    if not np.isfinite(proposed_full_source).all():
                        raise human.ImportError("local self-separation returned non-finite source positions")
                    if not np.array_equal(np.asarray(proposal.get("movable_vertex_ids"), dtype=np.int64), movable_ids):
                        raise human.ImportError("local self-separation proposal changed its declared movable patch")
                    if not np.array_equal(np.asarray(proposal.get("protected_vertex_ids"), dtype=np.int64),
                                          protected_self_ids):
                        raise human.ImportError("local self-separation proposal changed its protected vertex set")
                    outside_patch = np.setdiff1d(
                        np.arange(len(source), dtype=np.int64), movable_ids, assume_unique=True,
                    )
                    if not np.array_equal(proposed_full_source[outside_patch], full_trial_source[outside_patch]):
                        raise human.ImportError("local self-separation changed source vertices outside its patch")
                    trial_source = proposed_full_source[referenced].astype(np.float64)
                    if len(fixed) and not np.array_equal(trial_source[fixed], source_base[fixed]):
                        raise human.ImportError("local self-separation moved a fixed support or preserved anchor")
                    if candidate_forward is None:
                        trial_delta = trial_source - source_base
                        trial_world = (captured + np.einsum(
                            "pnij,nj->pni", current_maps, trial_delta,
                        )).astype("<f4").astype(np.float64)
                        trial_maps, trial_forward_diagnostics = current_maps, {}
                    else:
                        trial_world, trial_maps, trial_forward_diagnostics = evaluate_forward(trial_source)
                        if trial_forward_diagnostics.get("admissible", True) is not True:
                            raise human.ImportError(
                                "local self-separation forward model violates a source-map invariant: "
                                + str(trial_forward_diagnostics.get("rejection_reason", "unspecified invariant"))
                            )
                    if np.array_equal(trial_world, current_world):
                        raise human.ImportError("local self-separation produced no changed accepted-pose coordinates")
                    tri = trial_world[:, compact_faces]
                    area = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
                    area_norm = np.linalg.norm(area, axis=2)
                    if not np.isfinite(area_norm).all() or np.any(area_norm == 0.0):
                        raise human.ImportError("local self-separation produced zero-area/non-finite skin triangles")
                    orientation_numerator = np.einsum("pfi,pfi->pf", base_area, area)
                    orientation_denominator = base_area_norm * area_norm
                    orientation = np.zeros_like(area_norm)
                    np.divide(orientation_numerator, orientation_denominator, out=orientation,
                              where=orientation_denominator > 0.0)
                    if not np.isfinite(orientation).all() or float(orientation.min()) <= 0.0:
                        raise human.ImportError("local self-separation violated accepted-pose area/orientation gates")
                    source_winding = _verify_source_winding_in_accepted_poses(
                        trial_source, compact_faces, source_outward_face_signs, trial_maps, area,
                    )
                    trial_bed_gap = np.einsum("pni,pi->pn", trial_world - origins[:, None, :], bed_normals)
                    if not np.isfinite(trial_bed_gap).all() or np.any(trial_bed_gap < bed_floor):
                        raise human.ImportError("local self-separation worsened an immutable baseline bed-plane gap")
                    trial_self_audits, remaining_self_failures = audit_candidate_self(
                        trial_world, reuse_world=preproposal_world, reuse_audits=preproposal_self_audits,
                    )
                    if remaining_self_failures:
                        first = remaining_self_failures[0]
                        raise human.ImportError(
                            "nonlinear exact self recheck retained "
                            f"{len(remaining_self_failures)} failing pose(s); first pose={first['pose_index']} "
                            f"pairs={first['triangle_pairs'][:8]}"
                        )
                    local_self_separation = {
                        "status": "bounded_local_proposal_revalidated_before_target_scans",
                        "failure_pose_indices": [int(pose) for pose, _ in failed_pairs],
                        "failure_pair_count_by_pose": [int(len(pairs)) for _, pairs in failed_pairs],
                        "failure_triangle_pairs_by_pose": [
                            {"pose_index": int(pose), "triangle_pairs": pairs.tolist()}
                            for pose, pairs in failed_pairs
                        ],
                        "self_pair_seed_source_vertex_ids": self_seed_ids.tolist(),
                        "coordinate_tied_source_vertex_ids_blocked": coordinate_tied_ids.tolist(),
                        "movable_source_vertex_ids": movable_ids.tolist(),
                        "movable_patch_expansion_rings": 2,
                        "pair_pose_receipts": remapped_pair_receipts,
                        "constraint_rows_by_pose": constraint_rows_by_pose,
                        "constraint_rows_by_sample": [int(value) for value in raw_constraint_rows],
                        "pair_seed_face_selection": pair_selection_receipts,
                        "protected_pair_side_face_rows": sorted(protected_pair_side_faces),
                        "rounding_reserve_max_component_mm": proposal.get("rounding_reserve_max_component_mm"),
                        "requested_reserved_linearized_minimum_sat_slack_mm": proposal.get(
                            "requested_reserved_linearized_minimum_sat_slack_mm"),
                        "requested_max_vertex_l2_increment_mm": proposal["requested_max_vertex_l2_increment_mm"],
                        "float32_max_vertex_l2_increment_mm": proposal["float32_max_vertex_l2_increment_mm"],
                        "post_rounding_linearized_minimum_sat_slack_mm": proposal[
                            "post_rounding_linearized_minimum_sat_slack_mm"],
                        "post_rounding_predicted_supplied_pair_exact_count": proposal[
                            "post_rounding_predicted_supplied_pair_exact_count"],
                        "nonlinear_exact_self_pair_count_by_pose": [int(row["count"]) for row in trial_self_audits],
                        "source_winding": source_winding,
                        "nonlinear_forward_revalidated": True,
                        "full_mesh_self_revalidated": True,
                        "full_target_admission_performed": False,
                    }
                    if trial_regularization is None:
                        trial_regularization = {}
                    trial_regularization["local_self_separation"] = local_self_separation
                except human.ImportError as error:
                    rejection = f"local self-separation proposal rejected: {error}"
                    failure_diagnostics = dict(trial_regularization or {"self_pair_failures": [
                        {"pose_index": pose, "triangle_pairs": pairs.tolist()}
                        for pose, pairs in failed_pairs
                    ]})
                    proposal_failure = getattr(error, "local_self_separation_diagnostics", None)
                    if proposal_failure is not None:
                        for stage in ("requested", "rounded"):
                            row = proposal_failure.get(stage, {}).get("minimum_sat_constraint")
                            if row is not None:
                                row["pose_index"] = int(failed_pairs[row["sample_index"]][0])
                        failure_diagnostics["local_self_separation_failure"] = proposal_failure
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                               diagnostics=failure_diagnostics)
                    continue
            if any(int(row["count"]) != 0 for row in trial_self_audits):
                rejection = "trial retains an exact skin self-intersection before target scans"
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source,
                           diagnostics=trial_regularization)
                continue
            trial_audits = []
            valid = True
            trial_key = f"iteration-{iteration + 1:02d}-backtrack-{backtrack:02d}"
            if scan_progress_callback is not None:
                scan_progress_callback({
                    "event": "trial_started",
                    "trial_key": trial_key,
                    "iteration": iteration + 1,
                    "backtrack": backtrack,
                    "scale": float(scale),
                    "source_positions_f32_sha256": hashlib.sha256(np.asarray(trial_source, dtype="<f4").tobytes()).hexdigest(),
                    "source_positions_f32": np.asarray(trial_source, dtype="<f4").copy(),
                })
            for pose in range(pose_count):
                self_audit = trial_self_audits[pose]
                if int(self_audit["count"]) != 0:
                    rejection = f"trial has {self_audit['count']} exact self-pairs at pose={pose}"
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, diagnostics=trial_regularization)
                    valid = False
                    break
                audit = scan_candidate_targets(pose, trial_world[pose])
                if not isinstance(audit, dict) or set(audit) != targets:
                    raise human.ImportError(f"candidate exact scan omitted target coverage at pose={pose}")
                if scan_progress_callback is not None:
                    scan_progress_callback({
                        "event": "pose_audit_complete",
                        "trial_key": trial_key,
                        "iteration": iteration + 1,
                        "backtrack": backtrack,
                        "pose_index": pose,
                        "target_audit": audit,
                    })
                for key, row in audit.items():
                    if (int(row.get("count", -1)) != len(pair_set(row))
                            or row.get("degenerate_face_rows")):
                        rejection = f"candidate exact scan found malformed/degenerate target={key} at pose={pose}"
                        emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, diagnostics=trial_regularization)
                        valid = False
                        break
                if not valid:
                    break
                if any(pair_set(audit[key]) != baseline_ocular_sets[pose][key] for key in ocular):
                    rejection = f"trial changed an exact ocular pair set at pose={pose}"
                    emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, diagnostics=trial_regularization)
                    valid = False
                    break
                trial_audits.append(audit)
            if not valid:
                continue
            trial_counts = [audit_count(audit, nonocular) for audit in trial_audits]
            trial_interiors = classify_closed_targets(trial_world)
            trial_interior_counts = interior_counts(trial_interiors)
            progress_before = (before_total, before_inside)
            progress_after = (sum(trial_counts), sum(trial_interior_counts))
            if (any(after > before for after, before in zip(progress_after, progress_before))
                    or progress_after == progress_before):
                rejection = (
                    "trial must reduce a defect count without increasing boundary pairs or interior vertices "
                    f"({before_total}, {before_inside})->"
                    f"({sum(trial_counts)}, {sum(trial_interior_counts)})")
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, trial_counts, trial_audits, diagnostics=trial_regularization)
                continue
            faces_to_check = sorted({
                int(face) for audit in trial_audits for key in nonocular
                for face, _ in audit[key]["triangle_pairs"]
            } | set(active_faces))
            winding_checks = {}
            winding_failure = None
            for pose in range(pose_count):
                for face in faces_to_check:
                    dot = float(orientation[pose, face])
                    if dot < _ROTATED_ACTIVE_FACE_BASE_DOT:
                        try:
                            result = _candidate_outward_winding(
                                trial_world[pose, compact_faces[face]].mean(axis=0),
                                area[pose, face] / area_norm[pose, face],
                                tri[pose],
                            )
                        except human.ImportError as error:
                            winding_failure = f"outward winding rejected pose={pose}, face={face}: {error}"
                            break
                        winding_checks[f"{pose}:{face}"] = {"pose": pose, "face": face, "normal_dot": dot, **result}
                if winding_failure:
                    break
            if winding_failure:
                rejection = winding_failure
                emit_trial(iteration, backtrack, scale, "rejected", rejection, trial_source, trial_counts, trial_audits, diagnostics=trial_regularization)
                continue
            emit_trial(iteration, backtrack, scale, "accepted", "all pose gates and exact scans passed",
                       trial_source, trial_counts, trial_audits, diagnostics=trial_regularization,
                       self_audits=trial_self_audits, self_audit_world_by_pose=trial_world)
            applied_source_increment = float(np.linalg.norm(trial_source - current_source, axis=1).max())
            current_source, current_world, current_audits, current_counts = trial_source, trial_world, trial_audits, trial_counts
            current_self_audits = trial_self_audits
            current_interiors, current_interior_counts = trial_interiors, trial_interior_counts
            current_maps = trial_maps
            current_forward_diagnostics = trial_forward_diagnostics
            winding_receipt.update(winding_checks)
            iteration_rows.append({
                "iteration": iteration + 1, "backtrack_scale": scale,
                "nonocular_pairs_before_by_pose": before_by_pose,
                "nonocular_pairs_after_by_pose": trial_counts,
                "closed_target_interior_vertices_after_by_pose": trial_interior_counts,
                "active_skin_faces": len(active_faces), "seed_vertices": int(len(seeds)),
                "fixed_seed_vertices": int(np.intersect1d(seeds, fixed).size),
                "maximum_source_increment_m": float(field.max() * scale),
                "maximum_applied_source_increment_m": applied_source_increment,
                "local_collateral_fold_regularization": trial_regularization,
                "local_self_separation": local_self_separation,
                "maximum_source_displacement_m": float(np.linalg.norm(current_source - source_base, axis=1).max()),
                "maximum_world_displacement_m_by_pose": [
                    float(np.linalg.norm(current_world[p] - captured[p], axis=1).max()) for p in range(pose_count)
                ],
                "minimum_active_projection": demand_report["minimum_active_projection"],
                "minimum_face_normal_alignment": float(orientation.min()),
                "minimum_triangle_area_ratio": float((area_norm / base_area_norm).min()),
                "maximum_triangle_area_ratio": float((area_norm / base_area_norm).max()),
                "minimum_bed_gap_m_by_pose": [float(row.min()) for row in trial_bed_gap],
                "direction_metrics": direction_report, "outward_winding_checks": winding_checks,
                "forward_model_diagnostics": trial_forward_diagnostics,
            })
            accepted = True
            break
        if not accepted:
            raise human.ImportError(
                f"multi-pose clearance could not admit iteration={iteration + 1} after {backtrack_count} backtracks: {rejection}"
            )

    final_counts = [audit_count(audit, nonocular) for audit in current_audits]
    if any(final_counts):
        raise human.ImportError(f"multi-pose clearance left nonocular exact pairs: {final_counts}")
    if any(current_interior_counts):
        raise human.ImportError(
            f"multi-pose clearance left skin vertices inside closed targets: {current_interior_counts}")
    final_triangles = current_world[:, compact_faces]
    final_area = np.cross(final_triangles[:, :, 1] - final_triangles[:, :, 0],
                          final_triangles[:, :, 2] - final_triangles[:, :, 0])
    final_area_norm = np.linalg.norm(final_area, axis=2)
    final_area_ratio = final_area_norm / base_area_norm
    base_edge_lengths = _maximum_pose_edge_lengths(captured, edges)
    final_edge_lengths = _maximum_pose_edge_lengths(current_world, edges)
    edge_stretch = final_edge_lengths / base_edge_lengths
    output = source.copy()
    output[referenced] = current_source
    final_bed_gap = np.einsum("pni,pi->pn", current_world - origins[:, None, :], bed_normals)
    report = {
        "schema": "numi.human.accepted-multipose-common-atlas-skin-clearance.v1",
        "status": "inferred_engineering_clearance_candidate_pending_native_replay",
        "coordinate_units": {"NHSKIN_source": "metres", "world": "metres",
                             "jacobian": "world metres per source metre"},
        "selected_margin_mm": selected_margin_mm,
        "support_radius_edge_multiple": support_radius_edge_multiple,
        "fixed_support_source_vertices": [int(value) for value in fixed_ids],
        "preserved_source_anchor_vertices": [int(value) for value in anchor_ids],
        "fixed_source_vertex_ids": [int(value) for value in all_fixed_ids],
        "source_orientation": {
            "basis": "outward face basis from the retained full-shell local-winding report, propagated by exact shared-edge adjacency over the complete NHSKIN source topology",
            "all_source_faces_connected_and_oriented": True,
            "accepted_pose_affine_transport": source_winding_report,
        },
        "accepted_pose_count": pose_count,
        "self_intersection_audit": {
            "baseline": "complete exact same-surface owner audit of each accepted current pose",
            "trials": "exact changed-face versus full candidate scan plus untouched accepted pair rows",
            "cache_update": "only after every geometry, target, containment, and progress gate accepts the trial",
        },
        "candidate_forward_model": {
            "mode": "recomputed_candidate_callback" if candidate_forward is not None else "fixed_affine_jacobian",
            "baseline_replay_max_vertex_error_m": baseline_replay_max,
            "baseline_replay_tolerance_m": float(baseline_replay_tolerance_m),
            "baseline_diagnostics": initial_forward_diagnostics,
            "final_diagnostics": current_forward_diagnostics,
        },
        "initial_nonocular_pair_count_by_pose": initial_counts,
        "initial_ocular_pair_count_by_pose": initial_ocular_counts,
        "selected_closed_target_interior": {
            "target_keys": sorted(closed_targets[0]),
            "explicit_outer_envelope_keys": sorted(outer_envelope_keys),
            "initial_inside_vertex_counts_by_pose": initial_interior_counts,
            "final_inside_vertex_counts_by_pose": current_interior_counts,
            "target_validation_by_pose": [
                {key: target["report"] for key, target in sorted(row.items())}
                for row in closed_targets],
            "interpretation": "additional exact inside checks for the explicitly selected closed targets; not whole-body solid-containment qualification",
        },
        "resume_start": resume_start_report,
        "starting_nonocular_pair_count_by_pose": (
            initial_counts if resume_start_report is None else resume_start_report["nonocular_pair_count_by_pose"]
        ),
        "final_nonocular_pair_count_by_pose": final_counts,
        "final_ocular_pair_count_by_pose": [audit_count(audit, ocular) for audit in current_audits],
        "iterations": iteration_rows,
        "outward_winding_checks": winding_receipt,
        "maximum_source_displacement_m": float(np.linalg.norm(current_source - source_base, axis=1).max()),
        "maximum_world_displacement_m_by_pose": [
            float(np.linalg.norm(current_world[p] - captured[p], axis=1).max()) for p in range(pose_count)
        ],
        "baseline_minimum_bed_gap_m_by_pose": [float(row.min()) for row in base_bed_gap],
        "candidate_minimum_bed_gap_m_by_pose": [float(row.min()) for row in final_bed_gap],
        "baseline_negative_bed_vertex_count_by_pose": [int(np.count_nonzero(row < 0.0)) for row in base_bed_gap],
        "candidate_negative_bed_vertex_count_by_pose": [int(np.count_nonzero(row < 0.0)) for row in final_bed_gap],
        "fixed_support_vertex_count": int(len(fixed_ids)),
        "preserved_source_anchor_vertex_count": int(len(anchor_ids)),
        "fixed_source_vertex_count": int(len(all_fixed_ids)),
        "active_source_vertex_count": int(np.count_nonzero(np.linalg.norm(current_source - source_base, axis=1) > 0.0)),
        "surface_shape_diagnostics": {
            "minimum_face_area_ratio_across_poses": float(final_area_ratio.min()),
            "maximum_face_area_ratio_across_poses": float(final_area_ratio.max()),
            "minimum_face_normal_dot_to_accepted_across_poses": float(
                (np.einsum("pfi,pfi->pf", base_area, final_area) /
                 (base_area_norm * final_area_norm)).min()
            ),
            "minimum_edge_stretch_ratio_using_maximum_pose_edge_metric": float(edge_stretch.min()),
            "maximum_edge_stretch_ratio_using_maximum_pose_edge_metric": float(edge_stretch.max()),
            "interpretation": "diagnostics only; positive orientation and exact nonzero area are enforced, but no tissue-thickness or shape-tolerance claim is inferred",
        },
        "qualification": {
            "all_supplied_accepted_pose_nonocular_exact_pairs_zero": "passed",
            "all_supplied_accepted_pose_ocular_pair_sets_unchanged": "passed",
            "native_replay": "pending",
            "physical_or_collision_use": "not_admitted",
        },
        "interpretation": "bounded engineering clearance candidate; not measured tissue thickness, physical penetration, or clinical clearance",
    }
    return output, report

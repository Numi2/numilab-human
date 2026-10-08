from __future__ import annotations

import csv
import hashlib
import json
import mmap
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import model as human
from .cardiac_cavity_intersections import (
    _audit_pair,
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


def _target_intersection_audit(skin_records, target_faces: dict[tuple[int, int], np.ndarray], pack_positions: np.ndarray):
    per_surface = {}
    for key in sorted(target_faces):
        faces = target_faces[key]
        unique = np.unique(faces)
        compact_faces = np.searchsorted(unique, faces)
        target_records = _exact_surface_records(pack_positions[unique], compact_faces)
        audit = _audit_pair(skin_records, target_records, same_surface=False)
        per_surface[f"{key[0]}:{key[1]}"] = {
            "triangle_pairs": audit["triangle_pairs"], "count": audit["count"],
            "aabb_candidate_pairs": audit["aabb_candidate_pairs"],
        }
    return per_surface


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
    baseline_all_audit = _target_intersection_audit(captured_skin_records, all_target_faces, pack_positions)
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
                trial_all_audit = _target_intersection_audit(trial_records, all_target_faces, pack_positions)
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

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
    """Use the existing Human skin-normal smoother for extension directions.

    The source-winding normals remain the orientation authority. This limited
    visual-normal operation only chooses a smoother displacement direction;
    callers retain independent captured-face winding and dot-product checks.
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
    vertex_normals = np.zeros_like(captured)
    area_vectors = np.cross(captured[compact_faces[:, 1]] - captured[compact_faces[:, 0]],
                            captured[compact_faces[:, 2]] - captured[compact_faces[:, 0]])
    for corner in range(3):
        np.add.at(vertex_normals, compact_faces[:, corner], area_vectors)
    normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(normal_lengths <= 1.0e-12) or not np.isfinite(vertex_normals).all():
        raise human.ImportError("clearance captured skin has invalid referenced vertex normals")
    vertex_normals /= normal_lengths[:, None]

    source_edges = np.concatenate((compact_faces[:, [0, 1]], compact_faces[:, [1, 2]], compact_faces[:, [2, 0]]), axis=0)
    source_edges.sort(axis=1)
    edges = np.unique(source_edges, axis=0)
    edge_lengths = np.linalg.norm(captured[edges[:, 0]] - captured[edges[:, 1]], axis=1)
    vertex_directions = _smooth_vertex_directions(vertex_normals, compact_faces)
    seed_demand_by_margin: dict[float, np.ndarray] = {}
    for margin_mm in (0.125, 0.25, 0.5):
        required = np.zeros(len(referenced), dtype=np.float64)
        for face_id, row in face_constraints.items():
            compact_ids = compact_faces[face_id]
            projections = vertex_directions[compact_ids] @ row["normal"]
            if float(projections.min()) < 0.8:
                raise human.ImportError(f"smoothed clearance directions are not aligned with outward source-winding face {face_id}")
            scalar = (row["minimum_separating_translation_m"] + margin_mm / 1000.0) / projections
            np.maximum.at(required, compact_ids, scalar)
        seed_demand_by_margin[margin_mm] = required
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
    graph_weights = np.concatenate((edge_lengths, edge_lengths))
    graph = csr_matrix((graph_weights, (graph_rows, graph_cols)), shape=(len(referenced), len(referenced)))

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
        active_vertex_required = np.zeros(len(referenced), dtype=np.float64)
        seed_required = seed_demand_by_margin[margin_mm]
        margin_audit = None
        field = None
        source_corrected = None
        source_delta = None
        world_delta_from_packed_source = None
        source_roundtrip = None
        final_iteration = None
        for iteration in range(6):
            total_required = np.maximum(seed_required, active_vertex_required)
            seed = np.flatnonzero(total_required > 0.0)
            if len(seed) == 0:
                raise human.ImportError("clearance compact field has no constrained vertices")
            distance = dijkstra(graph, directed=False, indices=seed, limit=radius)
            unit = np.clip(distance / radius, 0.0, 1.0)
            envelope = 1.0 - 3.0 * unit * unit + 2.0 * unit * unit * unit
            envelope[distance >= radius] = 0.0
            field = np.max(total_required[seed, None] * envelope, axis=0)
            for face_id, row in face_constraints.items():
                compact_ids = compact_faces[face_id]
                achieved = field[compact_ids] * (vertex_directions[compact_ids] @ row["normal"])
                needed = row["minimum_separating_translation_m"] + margin_mm / 1000.0
                if float(achieved.min()) + 1.0e-12 < needed:
                    raise human.ImportError(f"clearance compact field misses an original witness face constraint at face {face_id}")
            world_delta = field[:, None] * vertex_directions
            source_delta = np.linalg.solve(jacobian, world_delta[:, :, None])[:, :, 0]
            source_candidate = source_positions[referenced].astype("<f4").astype(np.float64)
            source_corrected = (source_candidate + source_delta).astype("<f4").astype(np.float64)
            packed_source_delta = source_corrected - source_candidate
            world_delta_from_packed_source = np.einsum("nij,nj->ni", jacobian, packed_source_delta)
            source_roundtrip = np.linalg.norm(world_delta_from_packed_source - world_delta, axis=1)
            candidate_world = _apply_captured_world_delta(captured, world_delta_from_packed_source)
            candidate_records = _exact_surface_records(candidate_world, compact_faces)
            candidate_triangles = candidate_world[compact_faces]
            candidate_area_vectors = np.cross(candidate_triangles[:, 1] - candidate_triangles[:, 0], candidate_triangles[:, 2] - candidate_triangles[:, 0])
            candidate_double_areas = np.linalg.norm(candidate_area_vectors, axis=1)
            if np.any(candidate_double_areas == 0.0) or not np.isfinite(candidate_double_areas).all():
                raise human.ImportError(f"clearance margin {margin_mm} mm created an exact zero-area or non-finite skin triangle")
            normal_alignment = np.einsum("ij,ij->i", base_area_vectors, candidate_area_vectors) / (base_double_areas * candidate_double_areas)
            if float(normal_alignment.min()) <= 0.0:
                bad_face = int(np.argmin(normal_alignment))
                bad_ids = compact_faces[bad_face]
                bad_source_ids = faces[bad_face]
                bad_base = captured[bad_ids]
                bad_candidate = candidate_world[bad_ids]
                edge_pairs = ((0, 1), (1, 2), (2, 0))
                edge_gradients = [
                    float((field[bad_ids[i]] - field[bad_ids[j]]) / max(np.linalg.norm(bad_base[i] - bad_base[j]), 1.0e-15))
                    for i, j in edge_pairs
                ]
                bad_base_area_mm2 = float(base_double_areas[bad_face] * 0.5e6)
                bad_candidate_area_mm2 = float(candidate_double_areas[bad_face] * 0.5e6)
                face_constraint = face_constraints.get(bad_face)
                dominant_seed_sources = []
                for compact_vertex in bad_ids:
                    scores = total_required[seed] * envelope[:, int(compact_vertex)]
                    winner = int(np.argmax(scores))
                    seed_vertex = int(seed[winner])
                    seed_faces = [
                        int(source_face) for source_face, source_row in face_constraints.items()
                        if seed_vertex in compact_faces[source_face]
                    ]
                    dominant_seed_sources.append({
                        "corrected_skin_source_vertex": int(referenced[int(compact_vertex)]),
                        "dominant_seed_skin_source_vertex": int(referenced[seed_vertex]),
                        "seed_distance_mm": float(distance[winner, int(compact_vertex)] * 1000.0),
                        "seed_demand_mm": float(total_required[seed_vertex] * 1000.0),
                        "seed_envelope_value": float(envelope[winner, int(compact_vertex)]),
                        "contributes_to_field_mm": float(scores[winner] * 1000.0),
                        "prior_active_set_seed": bool(active_vertex_required[seed_vertex] > 0.0),
                        "initial_witness_sources": [
                            {
                                "skin_face": source_face,
                                "targets": sorted(witness_target_faces_by_skin_face.get(source_face, set())),
                            }
                            for source_face in seed_faces
                        ],
                    })
                raise human.ImportError(
                    f"clearance margin {margin_mm} mm reversed skin triangle: face={bad_face}, "
                    f"iteration={iteration + 1}, prior_active_vertex_count={int(np.count_nonzero(active_vertex_required))}, "
                    f"normal_dot={float(normal_alignment[bad_face]):.12g}, "
                    f"base_area_mm2={bad_base_area_mm2:.12g}, candidate_area_mm2={bad_candidate_area_mm2:.12g}, "
                    f"area_ratio={float(candidate_double_areas[bad_face] / base_double_areas[bad_face]):.12g}, "
                    f"source_vertex_ids={bad_source_ids.tolist()}, compact_vertex_ids={bad_ids.tolist()}, "
                    f"base_world_m={bad_base.tolist()}, candidate_world_m={bad_candidate.tolist()}, "
                    f"vertex_field_mm={(field[bad_ids] * 1000.0).tolist()}, "
                    f"vertex_world_displacement_mm={(np.linalg.norm(bad_candidate - bad_base, axis=1) * 1000.0).tolist()}, "
                    f"edge_field_difference_mm={[float((field[bad_ids[i]] - field[bad_ids[j]]) * 1000.0) for i, j in edge_pairs]}, "
                    f"edge_field_gradient={[round(value, 12) for value in edge_gradients]}, "
                    f"common_atlas_source_delta_mm={(source_delta[bad_ids] * 1000.0).tolist()}, "
                    f"initial_witness_targets={sorted(face_constraint['target_surfaces']) if face_constraint else []}, "
                    f"initial_witness_pair_count={face_constraint['target_pair_count'] if face_constraint else 0}, "
                    f"dominant_seed_sources={dominant_seed_sources}, "
                    f"support_radius_mm={radius * 1000.0:.12g}, median_edge_mm={median_edge * 1000.0:.12g}"
                )
            skin_self_audit = _audit_pair(candidate_records, candidate_records, same_surface=True)
            if int(skin_self_audit["count"]) != 0:
                raise human.ImportError(f"clearance margin {margin_mm} mm created {skin_self_audit['count']} exact non-adjacent skin self-intersections")
            candidate_full_world = full_skin_world.copy()
            candidate_full_world[referenced] = candidate_world
            candidate_surface_integral = _signed_surface_integral(candidate_full_world, faces)
            geometry_quality_by_margin[str(margin_mm)] = {
                "exact_nonadjacent_skin_self_intersection_pairs": int(skin_self_audit["count"]),
                "exact_zero_area_triangles": int(np.count_nonzero(candidate_double_areas == 0.0)),
                "minimum_triangle_area_ratio": float((candidate_double_areas / base_double_areas).min()),
                "minimum_triangle_normal_dot": float(normal_alignment.min()),
                "signed_surface_integral_m3_proxy": candidate_surface_integral,
                "signed_surface_integral_delta_m3_proxy": candidate_surface_integral - base_surface_integral,
                "surface_integral_limitation": "The NHSKIN has 115 boundary edges at ocular openings, so this oriented surface integral is a sensitivity proxy, not a closed-shell volume.",
            }
            all_margin_audit = _target_intersection_audit(candidate_records, all_target_faces, pack_positions)
            margin_audit = {f"{key[0]}:{key[1]}": all_margin_audit[f"{key[0]}:{key[1]}"] for key in target_keys}
            margin_ocular_audit = {f"{key[0]}:{key[1]}": all_margin_audit[f"{key[0]}:{key[1]}"] for key in _OCULAR_MONITOR_KEYS}
            for key_text, baseline_row in baseline_ocular_audit.items():
                baseline_pairs = _triangle_pair_set(baseline_row)
                candidate_pairs = _triangle_pair_set(margin_ocular_audit[key_text])
                if candidate_pairs != baseline_pairs:
                    added = sorted(candidate_pairs - baseline_pairs)
                    removed = sorted(baseline_pairs - candidate_pairs)
                    raise human.ImportError(
                        f"clearance margin {margin_mm} mm changes exact pair identities at monitored ocular interface {key_text}: "
                        f"baseline_pair_count={len(baseline_pairs)}, candidate_pair_count={len(candidate_pairs)}, "
                        f"introduced_pairs_first32={added[:32]}, resolved_pairs_first32={removed[:32]}"
                    )
            ocular_monitor_by_margin[str(margin_mm)] = {
                "exact_pair_sets_unchanged": True,
                "pair_count": _sum_target_pairs(margin_ocular_audit),
                "per_surface_pair_counts": {key: int(row["count"]) for key, row in margin_ocular_audit.items()},
            }
            remaining = _sum_target_pairs(margin_audit)
            if remaining == 0:
                final_iteration = iteration + 1
                break
            old_required = active_vertex_required.copy()
            for surface_key, row in margin_audit.items():
                target = tuple(int(part) for part in surface_key.split(":"))
                target_triangles = target_faces[target]
                for skin_face, target_face in row["triangle_pairs"]:
                    skin_face = int(skin_face)
                    if skin_face not in face_constraints and skin_face not in orientation_checks_added:
                        base_tri = captured[compact_faces[skin_face]]
                        base_normal = np.cross(base_tri[1] - base_tri[0], base_tri[2] - base_tri[0])
                        base_normal /= np.linalg.norm(base_normal)
                        centroid = base_tri.mean(axis=0)
                        offset = 0.0005 * base_normal
                        winding_minus = _generalized_winding_number(centroid - offset, full_skin_triangles)
                        winding_plus = _generalized_winding_number(centroid + offset, full_skin_triangles)
                        if abs(winding_minus) - abs(winding_plus) <= 0.5:
                            raise human.ImportError(f"clearance cannot establish local outward direction for newly intersecting skin face {skin_face}")
                        orientation_checks_added[skin_face] = {
                            "source_vertex_ids": [int(value) for value in faces[skin_face]],
                            "winding_minus": winding_minus, "winding_plus": winding_plus,
                            "absolute_winding_difference": abs(winding_minus) - abs(winding_plus),
                            "basis": "captured full-shell generalized winding at 0.5 mm on either side of the source-winding face normal",
                        }
                    ids = compact_faces[skin_face]
                    current_triangle = candidate_world[ids]
                    target_ids = target_triangles[int(target_face)]
                    target_tri = pack_positions[target_ids].astype(np.float64)
                    current_normal = np.cross(current_triangle[1] - current_triangle[0], current_triangle[2] - current_triangle[0])
                    current_normal_length = float(np.linalg.norm(current_normal))
                    if current_normal_length <= 1.0e-15:
                        raise human.ImportError(f"clearance active-set face {skin_face} collapsed")
                    current_normal /= current_normal_length
                    base_normal = face_constraints.get(skin_face, {}).get("normal")
                    if base_normal is None:
                        base_tri = captured[ids]
                        base_normal = np.cross(base_tri[1] - base_tri[0], base_tri[2] - base_tri[0])
                        base_normal /= np.linalg.norm(base_normal)
                    alignment = float(np.dot(current_normal, base_normal))
                    current_gap = float(np.max((target_tri - current_triangle[0]) @ current_normal))
                    if alignment < 0.75:
                        face_world_delta = candidate_world[ids] - captured[ids]
                        raise human.ImportError(
                            f"clearance active-set face {skin_face} rotated too far from its outward source winding: "
                            f"margin_mm={margin_mm}, iteration={iteration + 1}, target={surface_key}, "
                            f"target_face={int(target_face)}, normal_dot={alignment:.9g}, "
                            f"vertex_field_mm={(field[ids] * 1000.0).tolist()}, "
                            f"face_vertex_displacement_mm={(np.linalg.norm(face_world_delta, axis=1) * 1000.0).tolist()}, "
                            f"normal_displacement_mm={(face_world_delta @ base_normal * 1000.0).tolist()}, "
                            f"tangential_displacement_mm={np.linalg.norm(face_world_delta - (face_world_delta @ base_normal)[:, None] * base_normal, axis=1).tolist()}, "
                            f"field_max_mm={field.max() * 1000.0:.9g}, radius_mm={radius * 1000.0:.9g}, "
                            f"median_edge_mm={median_edge * 1000.0:.9g}, current_gap_mm={current_gap * 1000.0:.9g}, "
                            f"skin_source_vertex_ids={faces[skin_face].tolist()}, target_vertex_ids={target_ids.tolist()}, "
                            f"base_skin_triangle_world_m={captured[ids].tolist()}, "
                            f"candidate_skin_triangle_world_m={current_triangle.tolist()}, target_triangle_world_m={target_tri.tolist()}"
                        )
                    dots = vertex_directions[ids] @ current_normal
                    source_normal_dots = vertex_normals[ids] @ current_normal
                    if float(dots.min()) < 0.75 or float(source_normal_dots.min()) < 0.75:
                        raise human.ImportError(f"clearance active-set extension or source normals disagree with outward face {skin_face}")
                    exact_pair_separation = _triangle_normal_translation_to_separate(current_triangle, target_tri, current_normal)
                    extra = exact_pair_separation + margin_mm / 1000.0
                    required_total = field[ids] + extra / dots
                    np.maximum.at(active_vertex_required, ids, required_total)
            if not np.any(active_vertex_required > old_required + 1.0e-12):
                raise human.ImportError(f"clearance active set did not strengthen any vertex despite {remaining} exact intersections")
        if final_iteration is None or margin_audit is None or source_corrected is None or source_delta is None:
            raise human.ImportError(f"clearance did not remove all nearby target intersections at margin {margin_mm} mm")
        pair_count = _sum_target_pairs(margin_audit)
        aabb_audit_by_margin[str(margin_mm)] = {
            "remaining_exact_skin_target_triangle_pairs": pair_count,
            "active_set_iterations": final_iteration,
            "by_target_surface": margin_audit,
        }
        output_source_by_margin[margin_mm] = source_corrected
        support_by_margin[str(margin_mm)] = {
            "margin_mm": margin_mm, "initial_witness_faces": len(face_constraints),
            "additional_active_faces": len(orientation_checks_added),
            "witness_triangle_pairs": len(witnesses), "seed_vertices": int(len(seed)),
            "nonzero_displaced_vertices": int(np.count_nonzero(field)),
            "support_radius_mm": radius * 1000.0, "support_radius_edge_multiple": float(support_radius_edge_multiple),
            "median_local_edge_mm": median_edge * 1000.0,
            "maximum_world_displacement_mm": float(field.max() * 1000.0),
            "maximum_common_atlas_source_displacement_mm": float(np.linalg.norm(source_delta, axis=1).max() * 1000.0),
            "maximum_packed_inverse_roundtrip_error_um": float(source_roundtrip.max() * 1.0e6),
            "maximum_jacobian_condition_number": float(condition.max()),
            "maximum_accepted_pose_fit_residual_um": float(base_residual.max() * 1.0e6),
            "maximum_accepted_pose_fit_residual_on_packed_displaced_vertices_um": float(base_residual[np.any(packed_source_delta != 0.0, axis=1)].max() * 1.0e6) if np.any(packed_source_delta != 0.0) else 0.0,
            "extension_direction_smoothing": "existing Human skin visual-normal smoother, eight triangle-neighbor iterations",
            "extension_direction_smoothing_iterations": _DIRECTION_SMOOTHING_ITERATIONS,
        }
    main_source = source_positions.copy()
    main_source[referenced] = output_source_by_margin[0.25]
    report = {
        "schema": "numi.human.step0-witnessed-common-atlas-skin-clearance.v1",
        "status": "inferred_engineering_clearance_candidate_pending_native_replay",
        "method": {
            "direction": "captured skin source winding, independently classified locally toward lower full-shell winding at every witness face; bone witness normals also point away from the fitted registered owning-body origin",
            "extension_direction_smoothing": "eight iterations of the existing Human skin visual-normal smoother over original triangle adjacency; it only selects the inferred extension direction, while original face winding remains the anatomical outward classifier and every seed face retains at least 0.8 extension-direction dot. The eight-ring orientation influence is broader than a single edge and is an explicit candidate limitation.",
            "finite_pair_demand": "minimum outward translation along the captured skin face normal that makes the finite skin and target triangles separate in at least one separating-axis interval; this avoids treating all three opponent vertices as if they overlapped the skin-face footprint",
            "field": "same compact geodesic smootherstep field for all margins: each witnessed finite triangle pair receives its first outward normal translation that separates the pair under the triangle separating-axis intervals, plus the selected engineering translation parameter; max seed envelope over geodesic distance on the original skin mesh; the manifest records the selected multiple of the median local edge length as the support radius",
            "inverse_map": "per referenced vertex solve the full 3x3 affine Jacobian built from all 86 canonical skin bindings, all unchanged source weights, and 86 accepted body rotations recovered independently from all 185 exact-topology registered bone surfaces; apply the resulting packed source delta to the exact captured accepted-world baseline so zero-delta vertices retain their exact float32 coordinates",
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

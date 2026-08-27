"""Narrow, provenance-carrying Z-Anatomy calf visual supplement importer.

This module deliberately produces the existing NHTISS3 interchange consumed by
the native Core renderer.  It does not introduce a second physics model: each
Z-Anatomy vertex inherits a named existing MyoSim articulated-body weight from
the matching BodyParts3D calf surface.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

from .model import (
    ImportError,
    _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_ABI,
    _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_MAGIC,
    _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE,
    _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON,
    _bodyparts_obj_member,
    _bodyparts_obj_triangles,
    _bodyparts_secondary_attachment_weight_lock,
    _bodyparts_visual_local_pose,
    read_json,
    sha256,
    write_json,
)


_EXPORT_SCHEMA = "numi.human.zanatomy-calf-blender-export.v1"
_SUPPLEMENT_SCHEMA = "numi.human.zanatomy-calf-visual-supplement.v1"
_PAYLOAD_HEADER = struct.Struct("<8s5I32s")
_PAYLOAD_RECORD = struct.Struct("<10I24f")
_PAYLOAD_VERTEX = struct.Struct("<9f")
_INVALID = 0xFFFFFFFF
_ZANATOMY_OVERLAY_BONE_LAYER = 3


def _finite_vector(value: Any, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ImportError(f"{context} is not a three-vector")
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ImportError(f"{context} is non-numeric") from error
    if not all(math.isfinite(component) for component in result):
        raise ImportError(f"{context} is non-finite")
    return result


def _vector_mean(points: list[list[float]], context: str) -> list[float]:
    if not points:
        raise ImportError(f"{context} has no points")
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def _distance_squared(left: list[float], right: list[float]) -> float:
    return sum((left[axis] - right[axis]) ** 2 for axis in range(3))


def _read_export(path: Path, configuration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    if value.get("schema") != _EXPORT_SCHEMA:
        raise ImportError("Z-Anatomy calf export schema is unsupported")
    source = value.get("source")
    if not isinstance(source, dict) or source.get("blend_file") != configuration["source"]["blend_file"]:
        raise ImportError("Z-Anatomy calf export has a different source blend identity")
    source_hash = source.get("blend_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ImportError("Z-Anatomy calf export has no source blend SHA-256")
    entries = value.get("objects")
    if not isinstance(entries, list):
        raise ImportError("Z-Anatomy calf export has no mesh objects")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ImportError("Z-Anatomy calf export has an invalid mesh record")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or identifier in result:
            raise ImportError("Z-Anatomy calf export duplicates or omits a mesh identity")
        layer = entry.get("layer")
        vertices = entry.get("vertices_world_m")
        normals = entry.get("normals_world")
        triangles = entry.get("triangles")
        if layer not in {"muscle", "tendon", "bone"} or \
                not isinstance(vertices, list) or not isinstance(normals, list) or \
                not isinstance(triangles, list) or len(vertices) != len(normals) or not vertices:
            raise ImportError(f"Z-Anatomy calf export {identifier} has invalid geometry arrays")
        parsed_vertices = [_finite_vector(point, f"Z-Anatomy {identifier} vertex") for point in vertices]
        parsed_normals = []
        for normal in normals:
            parsed = _finite_vector(normal, f"Z-Anatomy {identifier} normal")
            length = math.sqrt(sum(component * component for component in parsed))
            if length <= 1.0e-8:
                raise ImportError(f"Z-Anatomy calf export {identifier} has a zero normal")
            parsed_normals.append([component / length for component in parsed])
        parsed_triangles: list[tuple[int, int, int]] = []
        for triangle in triangles:
            if not isinstance(triangle, list) or len(triangle) != 3 or any(
                not isinstance(index, int) or not 0 <= index < len(parsed_vertices) for index in triangle
            ):
                raise ImportError(f"Z-Anatomy calf export {identifier} has an invalid triangle")
            if len(set(triangle)) != 3:
                raise ImportError(f"Z-Anatomy calf export {identifier} has a degenerate triangle")
            parsed_triangles.append((triangle[0], triangle[1], triangle[2]))
        if not parsed_triangles:
            raise ImportError(f"Z-Anatomy calf export {identifier} has no triangles")
        result[identifier] = {
            "layer": layer,
            "vertices": parsed_vertices,
            "normals": parsed_normals,
            "triangles": parsed_triangles,
        }
    configured = configuration.get("objects")
    if not isinstance(configured, list) or {entry.get("id") for entry in configured} != set(result):
        raise ImportError("Z-Anatomy calf export object coverage differs from the configured narrow supplement")
    return result


def _read_base_payload(path: Path) -> tuple[tuple[Any, ...], dict[int, tuple[Any, ...]], list[tuple[float, ...]]]:
    data = path.read_bytes()
    if len(data) < _PAYLOAD_HEADER.size:
        raise ImportError("base BodyParts3D tissue payload is truncated")
    header = _PAYLOAD_HEADER.unpack_from(data)
    magic, abi, tissue_count, vertex_count, index_count, _, _ = header
    if magic != _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_MAGIC or \
            abi != _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_ABI:
        raise ImportError("Z-Anatomy supplement requires an NHTISS3 BodyParts3D base payload")
    expected = _PAYLOAD_HEADER.size + tissue_count * _PAYLOAD_RECORD.size + \
        vertex_count * _PAYLOAD_VERTEX.size + index_count * 4
    if len(data) != expected:
        raise ImportError("base BodyParts3D tissue payload byte count is inconsistent")
    offset = _PAYLOAD_HEADER.size
    records: dict[int, tuple[Any, ...]] = {}
    for _ in range(tissue_count):
        record = _PAYLOAD_RECORD.unpack_from(data, offset)
        offset += _PAYLOAD_RECORD.size
        stable_id = record[7]
        first_vertex, count, first_index, triangle_indices = record[3:7]
        if stable_id in records or first_vertex + count > vertex_count or \
                first_index + triangle_indices > index_count or triangle_indices % 3:
            raise ImportError("base BodyParts3D tissue payload record is malformed")
        records[stable_id] = record
    vertices = [
        _PAYLOAD_VERTEX.unpack_from(data, offset + index * _PAYLOAD_VERTEX.size)
        for index in range(vertex_count)
    ]
    return header, records, vertices


def _registration_world_transform(registration_path: Path) -> tuple[list[list[float]], list[float], list[list[float]], float]:
    registration = read_json(registration_path)
    coordinates = registration.get("coordinate_system")
    matrix = coordinates.get("global_source_mm_to_myosim_world_m") if isinstance(coordinates, dict) else None
    _, _, source_scale = _bodyparts_visual_local_pose(matrix, "Z-Anatomy supplement BodyParts3D global transform")
    if not isinstance(matrix, list):
        raise ImportError("Z-Anatomy supplement has no BodyParts3D global registration")
    # The `source_scale` returned by the existing visual validator maps stored
    # source metres (rather than OBJ millimetres) into world metres.
    rotation = [[matrix[row][column] / (0.001 * source_scale) for column in range(3)] for row in range(3)]
    if any(abs(sum(rotation[row][axis] * rotation[column][axis] for axis in range(3)) - (1.0 if row == column else 0.0)) > 1.0e-5
           for row in range(3) for column in range(3)):
        raise ImportError("Z-Anatomy supplement BodyParts3D registration is not orthonormal")
    return matrix, [matrix[row][3] for row in range(3)], rotation, source_scale


def _world_from_stored_source(vertex: list[float], translation: list[float], rotation: list[list[float]], scale: float) -> list[float]:
    return [translation[row] + scale * sum(rotation[row][column] * vertex[column] for column in range(3)) for row in range(3)]


def _stored_source_from_world(vertex: list[float], translation: list[float], rotation: list[list[float]], scale: float) -> list[float]:
    delta = [vertex[row] - translation[row] for row in range(3)]
    return [sum(rotation[column][row] * delta[column] for column in range(3)) / scale for row in range(3)]


def _stored_normal_from_world(normal: list[float], rotation: list[list[float]]) -> list[float]:
    result = [sum(rotation[column][row] * normal[column] for column in range(3)) for row in range(3)]
    length = math.sqrt(sum(value * value for value in result))
    if length <= 1.0e-8:
        raise ImportError("Z-Anatomy world normal cannot be converted into BodyParts3D source coordinates")
    return [value / length for value in result]


def _nearest_weights(
    points: list[list[float]], base_points: list[list[float]], base_weights: list[tuple[float, float, float]], context: str,
) -> tuple[list[tuple[float, float, float]], dict[str, float]]:
    if not base_points or len(base_points) != len(base_weights):
        raise ImportError(f"{context} has no compatible base BodyParts3D weights")
    weights: list[tuple[float, float, float]] = []
    squared_distances: list[float] = []
    # A 4-surface calf payload is small enough for this exact direct nearest
    # query, which avoids a new interpolation or inferred attachment model.
    for point in points:
        nearest_index = min(range(len(base_points)), key=lambda index: _distance_squared(point, base_points[index]))
        weights.append(base_weights[nearest_index])
        squared_distances.append(_distance_squared(point, base_points[nearest_index]))
    return weights, {
        "nearest_bodyparts_vertex_rms_m": math.sqrt(sum(squared_distances) / len(squared_distances)),
        "nearest_bodyparts_vertex_max_m": math.sqrt(max(squared_distances)),
    }


def _closest_point_on_triangle(
    point: list[float], first: list[float], second: list[float], third: list[float],
) -> tuple[list[float], list[float]]:
    """Return the exact closest point and its unit face normal.

    This is deliberately evaluated only for the small attachment band of the
    one supplemental tendon.  It gives the visual registration a real named
    calcaneus surface target rather than a body centre, a box, or a synthetic
    connector mesh.
    """
    subtract = lambda left, right: [left[axis] - right[axis] for axis in range(3)]
    dot = lambda left, right: sum(left[axis] * right[axis] for axis in range(3))
    add_scaled = lambda origin, direction, scale: [
        origin[axis] + direction[axis] * scale for axis in range(3)
    ]
    ab, ac, ap = subtract(second, first), subtract(third, first), subtract(point, first)
    normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    normal_length = math.sqrt(dot(normal, normal))
    if normal_length <= 1.0e-12:
        raise ImportError("Z-Anatomy calcaneus has a degenerate source triangle")
    normal = [value / normal_length for value in normal]
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return first, normal
    bp = subtract(point, second)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return second, normal
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return add_scaled(first, ab, d1 / (d1 - d3)), normal
    cp = subtract(point, third)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return third, normal
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= d2:
        return add_scaled(first, ac, d2 / (d2 - d6)), normal
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        bc = subtract(third, second)
        return add_scaled(second, bc, (d4 - d3) / ((d4 - d3) + (d5 - d6))), normal
    denominator = 1.0 / (va + vb + vc)
    return [
        first[axis] + (vb * ab[axis] + vc * ac[axis]) * denominator
        for axis in range(3)
    ], normal


def _project_tendon_attachment_band(
    vertices: list[list[float]], attenuation: list[float],
    bone_vertices: list[list[float]], bone_triangles: list[tuple[int, int, int]],
) -> tuple[list[list[float]], dict[str, float | int | str]]:
    """Carry the named tendon end just inside its matching bone surface.

    The imported Z-Anatomy tendon and the BodyParts3D calcaneus are separate
    licensed meshes.  Matching the calcaneus centroids alone leaves their
    closest contact vertices a few millimetres apart.  We therefore use the
    same exact named calcaneus triangle surface as the existing distal
    articulated-body lock. The prior visual treatment placed every locked
    vertex *on* that surface with a positive offset. That makes a closed
    tendon cap read as a detached beige plate even though its pose is tied to
    the correct bone. Instead, carry the source end through the cortex by a
    tiny, fixed visual-only inset. Depth testing then occludes the artificial
    closed cap while the unmodified proximal surface visibly enters the named
    calcaneus. This is a rendering registration, not a tendon weld: it changes
    no MyoSim path, tendon parameter, or force law.
    """
    if len(vertices) != len(attenuation) or not bone_triangles:
        raise ImportError("Z-Anatomy tendon attachment-band input is incomplete")
    result: list[list[float]] = []
    projected = fully_locked = feathered = 0
    corrections: list[float] = []
    for vertex, proximal_weight in zip(vertices, attenuation, strict=True):
        blend = 1.0 - proximal_weight
        if blend <= 1.0e-8:
            result.append(vertex)
            continue
        closest: list[float] | None = None
        normal: list[float] | None = None
        nearest_squared = math.inf
        for triangle in bone_triangles:
            first, second, third = (bone_vertices[index] for index in triangle)
            candidate, candidate_normal = _closest_point_on_triangle(vertex, first, second, third)
            squared = _distance_squared(vertex, candidate)
            if squared < nearest_squared:
                closest, normal, nearest_squared = candidate, candidate_normal, squared
        if closest is None or normal is None:
            raise ImportError("Z-Anatomy tendon attachment band has no calcaneal target")
        difference = [vertex[axis] - closest[axis] for axis in range(3)]
        side = 1.0 if sum(difference[axis] * normal[axis] for axis in range(3)) >= 0.0 else -1.0
        # An opaque bone should hide the terminal cap instead of displaying a
        # second, coplanar surface over it. 1.5 mm is deliberately smaller
        # than the existing 3--15 mm feather band, so this is a local
        # enthesis presentation correction rather than a replacement tendon.
        visual_enthesis_inset_m = 0.0015
        target = [closest[axis] - side * normal[axis] * visual_enthesis_inset_m for axis in range(3)]
        corrected = [vertex[axis] * (1.0 - blend) + target[axis] * blend for axis in range(3)]
        correction = math.sqrt(_distance_squared(vertex, corrected))
        result.append(corrected)
        corrections.append(correction)
        projected += 1
        if proximal_weight <= 1.0e-8:
            fully_locked += 1
        else:
            feathered += 1
    return result, {
        "method": "exact named calcaneus triangle projection with a visual-only interior enthesis inset",
        "visual_enthesis_inset_m": 0.0015,
        "projected_vertex_count": projected,
        "fully_locked_vertex_count": fully_locked,
        "feathered_vertex_count": feathered,
        "rms_correction_m": math.sqrt(sum(value * value for value in corrections) / len(corrections)) if corrections else 0.0,
        "max_correction_m": max(corrections, default=0.0),
        "boundary": "visual rest-surface registration only; not a tendon weld, force-transfer law, continuum, or clinical attachment certificate",
    }


def build_zanatomy_calf_visual_supplement_payload(
    sources: Path, registration_path: Path, base_payload_path: Path, export_path: Path, output: Path,
) -> dict[str, Any]:
    """Build an NHTISS3 visual-only Z-Anatomy right-calf supplement.

    The source geometry is more detailed than the BodyParts3D calf slice. The
    force-route and body-binding authority remains the existing source
    payload: muscles and tendon inherit named BodyParts3D body weights, while
    the matching Z-Anatomy calcaneus is rigidly bound to the existing MyoSim
    calcn_r body for this narrow five-surface inspection. This preserves the
    source-authored tendon-to-bone visual connection without changing a force
    path or the whole-body skeletal source.
    """
    configuration_path = Path(__file__).resolve().parents[2] / "config/zanatomy-calf-visual-supplement.v1.json"
    configuration = read_json(configuration_path)
    if configuration.get("schema") != _SUPPLEMENT_SCHEMA:
        raise ImportError("Z-Anatomy calf supplement configuration schema is unsupported")
    source = configuration.get("source")
    entries = configuration.get("objects")
    if not isinstance(source, dict) or source.get("license") != "CC-BY-SA-4.0" or \
            not isinstance(entries, list) or len(entries) != 5:
        raise ImportError("Z-Anatomy calf supplement configuration has invalid source terms or scope")
    exported = _read_export(export_path.resolve(), configuration)
    header, base_records, base_vertices = _read_base_payload(base_payload_path.resolve())
    _, _, _, _, _, registration_fingerprint, source_sha = header
    _, world_translation, world_rotation, world_scale = _registration_world_transform(registration_path.resolve())

    overlay_entry = next((entry for entry in entries if entry.get("layer") == "bone"), None)
    if not isinstance(overlay_entry, dict):
        raise ImportError("Z-Anatomy calf supplement has no calcaneus overlay")
    overlay_id = overlay_entry.get("id")
    if not isinstance(overlay_id, str) or overlay_id not in exported:
        raise ImportError("Z-Anatomy calf supplement calcaneus overlay is absent from the export")
    member_id, hierarchy = overlay_entry.get("bodyparts_member_id"), overlay_entry.get("bodyparts_hierarchy")
    if not isinstance(member_id, str) or not isinstance(hierarchy, str):
        raise ImportError("Z-Anatomy calf supplement calcaneus target is incomplete")
    _, bone_member, bone_obj = _bodyparts_obj_member(sources.resolve(), hierarchy, member_id)
    bone_vertices_mm, bone_triangles = _bodyparts_obj_triangles(bone_obj, bone_member)
    bone_world = [
        [
            sum(world_rotation[row][column] * (coordinate * 0.001 * world_scale) for column, coordinate in enumerate(vertex)) + world_translation[row]
            for row in range(3)
        ]
        for vertex in bone_vertices_mm
    ]
    overlay_world = exported[overlay_id]["vertices"]
    registration_translation = [
        target - origin for target, origin in zip(_vector_mean(bone_world, "BodyParts3D calcaneus"), _vector_mean(overlay_world, "Z-Anatomy calcaneus"), strict=True)
    ]
    overlay_world_registered = [
        [point[axis] + registration_translation[axis] for axis in range(3)]
        for point in overlay_world
    ]
    overlay_triangles = exported[overlay_id]["triangles"]

    records: list[bytes] = []
    stored_vertices: list[tuple[float, ...]] = []
    stored_indices: list[int] = []
    surfaces: list[dict[str, Any]] = []
    surface_entries = [entry for entry in entries if isinstance(entry, dict)]
    for output_stable_id, entry in enumerate(surface_entries, start=1):
        identifier, base_stable_id, layer_name = entry.get("id"), entry.get("base_stable_id"), entry.get("layer")
        if not isinstance(identifier, str) or not isinstance(base_stable_id, int) or identifier not in exported or base_stable_id not in base_records:
            raise ImportError("Z-Anatomy calf supplement source/body binding is unresolved")
        zmesh = exported[identifier]
        base = base_records[base_stable_id]
        expected_layer = _BODYPARTS_MYOSIM_VISUAL_LAYER_MUSCLE if layer_name == "muscle" else \
            _BODYPARTS_MYOSIM_VISUAL_LAYER_TENDON if layer_name == "tendon" else \
            _ZANATOMY_OVERLAY_BONE_LAYER if layer_name == "bone" else None
        if expected_layer is None or zmesh["layer"] != layer_name:
            raise ImportError(f"Z-Anatomy calf supplement {identifier} layer conflicts with its BodyParts3D binding")
        if layer_name == "bone":
            expected_bodies = [entry.get("body_index")]
            if expected_bodies != [138] or base[2] != 138:
                raise ImportError("Z-Anatomy calcaneus overlay must use the named MyoSim calcn_r binding")
        else:
            expected_bodies = entry.get("body_indices")
            if base[8] != expected_layer:
                raise ImportError(f"Z-Anatomy calf supplement {identifier} layer conflicts with its BodyParts3D binding")
        if not isinstance(expected_bodies, list) or not 1 <= len(expected_bodies) <= 3 or \
                any(not isinstance(body, int) for body in expected_bodies) or \
                (layer_name != "bone" and tuple(expected_bodies) != tuple(body for body in base[:3] if body != _INVALID)):
            raise ImportError(f"Z-Anatomy calf supplement {identifier} has a different named MyoSim body binding")
        first_vertex, base_count = base[3], base[4]
        base_points = [
            _world_from_stored_source(list(base_vertices[index][:3]), world_translation, world_rotation, world_scale)
            for index in range(first_vertex, first_vertex + base_count)
        ]
        base_weights = [tuple(base_vertices[index][6:9]) for index in range(first_vertex, first_vertex + base_count)]
        world_vertices = [
            [point[axis] + registration_translation[axis] for axis in range(3)]
            for point in zmesh["vertices"]
        ]
        if layer_name == "bone":
            weights = [(1.0, 0.0, 0.0)] * len(world_vertices)
            correspondence = {"method": "rigid existing_MyoSim_calcn_r_body_binding"}
            record_bodies = (138, _INVALID, _INVALID)
            identity_binding = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
            record_bindings = (*base[26:34], *identity_binding, *identity_binding)
        else:
            weights, correspondence = _nearest_weights(world_vertices, base_points, base_weights, identifier)
            record_bodies = base[:3]
            record_bindings = base[10:]
        attachment_lock: dict[str, Any] | None = None
        if layer_name == "tendon":
            attenuation, attachment_lock = _bodyparts_secondary_attachment_weight_lock(
                world_vertices, [1.0] * len(world_vertices), overlay_world_registered, overlay_triangles,
            )
            proximal_base = [
                (point, weight) for point, weight in zip(base_points, base_weights, strict=True)
                if weight[0] + weight[1] > 1.0e-8
            ]
            if not proximal_base:
                raise ImportError("Z-Anatomy tendon base payload has no proximal MyoSim body weights")
            proximal_fallback, fallback_correspondence = _nearest_weights(
                world_vertices, [point for point, _ in proximal_base],
                [weight for _, weight in proximal_base], identifier + " proximal fallback",
            )
            adjusted_weights: list[tuple[float, float, float]] = []
            for inherited, fallback, value in zip(weights, proximal_fallback, attenuation, strict=True):
                proximal = inherited[0] + inherited[1]
                if value <= 1.0e-8:
                    adjusted_weights.append((0.0, 0.0, 1.0))
                    continue
                if proximal <= 1.0e-8:
                    inherited = fallback
                    proximal = inherited[0] + inherited[1]
                adjusted_weights.append((
                    inherited[0] / proximal * value,
                    inherited[1] / proximal * value,
                    1.0 - value,
                ))
            weights = adjusted_weights
            attachment_lock["proximal_weight_fallback"] = {
                "method": "nearest named BodyParts3D calcaneal-tendon vertex retaining a nonzero proximal MyoSim weight",
                **fallback_correspondence,
            }
            attachment_lock["method"] = (
                "Z-Anatomy tendon vertices with named matching Z-Anatomy calcaneus source-triangle lock; "
                "the calcaneus is rigidly bound to existing MyoSim calcn_r while proximal MyoSim body "
                "proportions transfer from the nearest named BodyParts3D tendon vertex"
            )
            attachment_lock["paired_calcaneus_overlay"] = overlay_id
        first_output_vertex, first_output_index = len(stored_vertices), len(stored_indices)
        for world_vertex, normal, weight in zip(world_vertices, zmesh["normals"], weights, strict=True):
            if abs(sum(weight) - 1.0) > 1.0e-6 or any(value < 0.0 or not math.isfinite(value) for value in weight):
                raise ImportError(f"Z-Anatomy calf supplement {identifier} has invalid transferred body weights")
            stored_vertices.append((*_stored_source_from_world(world_vertex, world_translation, world_rotation, world_scale),
                                    *_stored_normal_from_world(normal, world_rotation), *weight))
        stored_indices.extend(first_output_vertex + index for triangle in zmesh["triangles"] for index in triangle)
        records.append(_PAYLOAD_RECORD.pack(
            *record_bodies, first_output_vertex, len(zmesh["vertices"]), first_output_index,
            len(zmesh["triangles"]) * 3, output_stable_id, expected_layer, 0, *record_bindings,
        ))
        surface = {
            "id": identifier,
            "stable_id": output_stable_id,
            "base_bodyparts_stable_id": base_stable_id,
            "layer": layer_name,
            "vertex_count": len(zmesh["vertices"]),
            "triangle_count": len(zmesh["triangles"]),
            "body_bindings": list(record_bodies),
            "body_weight_transfer": correspondence.get(
                "method", "nearest named matching BodyParts3D source vertex"
            ),
            **correspondence,
        }
        if attachment_lock is not None:
            surface["named_calcaneus_attachment_lock"] = attachment_lock
        surfaces.append(surface)
    if len(records) != 5 or len(stored_vertices) > 0xFFFFFFFF or len(stored_indices) > 0xFFFFFFFF:
        raise ImportError("Z-Anatomy calf supplement payload capacity or scope is invalid")
    payload = b"".join((
        _PAYLOAD_HEADER.pack(_BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_MAGIC,
                             _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_ABI, len(records),
                             len(stored_vertices), len(stored_indices), registration_fingerprint, source_sha),
        *records,
        b"".join(_PAYLOAD_VERTEX.pack(*vertex) for vertex in stored_vertices),
        struct.pack(f"<{len(stored_indices)}I", *stored_indices),
    ))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "zanatomy-calf-myosim-tissues.nhtissue"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.human.zanatomy-calf-myosim-visual-supplement-payload.v1",
        "payload": {
            "file": payload_path.name, "sha256": sha256(payload_path), "bytes": len(payload),
            "magic": "NHTISS3", "payload_abi": _BODYPARTS_MYOSIM_MULTI_BODY_SOFT_TISSUE_VISUAL_ABI,
            "registration_fingerprint32": f"{registration_fingerprint:08x}",
            "surface_count": len(records), "vertex_count": len(stored_vertices), "index_count": len(stored_indices),
        },
        "source": {
            "zanatomy": source,
            "zanatomy_export": {"file": export_path.name, "sha256": sha256(export_path)},
            "bodyparts3d_registration": {"file": registration_path.name, "sha256": sha256(registration_path)},
            "bodyparts3d_calcaneus": {"member": bone_member, "member_id": member_id, "sha256": hashlib.sha256(bone_obj).hexdigest()},
            "base_bodyparts_tissue_payload": {"file": base_payload_path.name, "sha256": sha256(base_payload_path)},
        },
        "registration": {
            "method": configuration["registration"]["method"],
            "translation_world_m": registration_translation,
            "zanatomy_calcaneus_centroid_world_m": _vector_mean(overlay_world, "Z-Anatomy calcaneus"),
            "bodyparts_calcaneus_centroid_world_m": _vector_mean(bone_world, "BodyParts3D calcaneus"),
            "rotation": "identity",
        },
        "surfaces": surfaces,
        "runtime_binding": "Z-Anatomy right-calf visual geometry with copied named BodyParts3D/MyoSim muscle and tendon weights plus one matching Z-Anatomy calcaneus rigidly bound to the existing MyoSim calcn_r body; the paired tendon remains on its authored source attachment without a generated cross-source projection",
        "status": "visual_supplement_input_not_a_force_path_or_continuum",
        "evidence_boundary": "The Z-Anatomy mesh is a CC-BY-SA visual supplement for the selected right-calf slice. The detailed calcaneus replaces the BodyParts3D calcaneus only in that visual inspection and remains rigidly attached to the existing MyoSim calcn_r body. BodyParts3D and MyoSim remain the whole-body geometry registration, articulated-body, source route, tendon parameter, and force authority. This does not create a deformable muscle/tendon continuum, change a MyoSim attachment, or establish medical registration or collision.",
    }
    write_json(output / "zanatomy-calf-myosim-tissues.manifest.json", manifest)
    return manifest

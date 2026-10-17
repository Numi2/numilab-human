"""Derive a passive external visceral-pleura proxy from registered lung lobes.

The proxy replaces the generic source surface at stable ID 310 in the existing
NHANAT1 ABI5 payload.  It is the external union boundary of the five closed
registered lobes.  It does not claim to include parietal pleura, pleural fluid,
or the lining of interlobar fissures.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
from fractions import Fraction

import numpy as np

from .resting_anatomy_interface_patch import (
    HEADER,
    RECORD,
    normals,
    signed_volume,
    topology_report,
)

LOBE_IDS = (305, 306, 307, 308, 309)
PLEURA_ID = 310
SOURCE_MICROFRAGMENT_ANCHORS_M = np.asarray([
    [0.009118776768445969, 0.036569900810718536, -0.056037694215774536],
    [0.008264528587460518, 0.037101924419403076, -0.05697183683514595],
    [0.00812491588294506, 0.03752134367823601, -0.056306567043066025],
    [0.00816801656037569, 0.03760587051510811, -0.05599856749176979],
], dtype=np.float64)


def _coord_key(point):
    point = tuple(float(x) for x in point)
    if len(point) != 3 or not all(np.isfinite(x) for x in point):
        raise ValueError("lobe source contains a non-finite coordinate")
    return point


def _face_key(coords):
    return tuple(sorted(coords))


def _same_winding(a, b):
    return any(tuple(a[(i + shift) % 3] for i in range(3)) == tuple(b) for shift in range(3))


def _opposite_winding(a, b):
    return any(tuple(a[(shift - i) % 3] for i in range(3)) == tuple(b) for shift in range(3))


def _edge_components(faces):
    """Return face components joined across exact indexed edges."""
    edge_faces = collections.defaultdict(list)
    for face_id, tri in enumerate(faces):
        a, b, c = map(int, tri)
        for x, y in ((a, b), (b, c), (c, a)):
            edge_faces[(min(x, y), max(x, y))].append(face_id)
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for incident in edge_faces.values():
        if len(incident) > 2:
            raise ValueError("derived lung union has a nonmanifold edge")
        if len(incident) == 2:
            a, b = map(find, incident)
            if a != b:
                parent[b] = a
    grouped = collections.defaultdict(list)
    for face_id in range(len(faces)):
        grouped[find(face_id)].append(face_id)
    return list(grouped.values()), edge_faces


def _component_detail(vertices, faces, owners, face_ids):
    vertex_ids = sorted({int(v) for fi in face_ids for v in faces[fi]})
    remap = {old: new for new, old in enumerate(vertex_ids)}
    local_faces = np.asarray([[remap[int(v)] for v in faces[fi]] for fi in face_ids], dtype=np.int64)
    local_vertices = vertices[np.asarray(vertex_ids, dtype=np.int64)]
    volume = signed_volume(local_vertices.astype(np.float64), local_faces)
    area = 0.0
    tri = local_vertices[local_faces].astype(np.float64)
    area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() * 0.5)
    return {
        "_face_ids": list(map(int, face_ids)),
        "_vertex_ids": vertex_ids,
        "_source_faces": [list(map(int, owners[fi])) for fi in face_ids],
        "face_count": int(len(face_ids)),
        "vertex_count": int(len(vertex_ids)),
        "source_triangle_count_by_stable_id": {
            str(sid): int(count) for sid, count in sorted(collections.Counter(owners[fi][0] for fi in face_ids).items())
        },
        "signed_volume_m3": float(volume),
        "surface_area_m2": area,
        "bounds_min_m": local_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max_m": local_vertices.max(axis=0).astype(float).tolist(),
    }


def _is_known_source_microfragment(vertices, detail, coordinate_quantization_m=None):
    """Recognize only the source-anchored 305/308 tetrahedral seam fragment."""
    source_faces = detail["_source_faces"]
    owner_counts = collections.Counter(int(row[0]) for row in source_faces)
    if set(owner_counts) != {305, 308} or owner_counts[305] != owner_counts[308]:
        return False
    if detail["face_count"] < 4 or detail["face_count"] > 64 or detail["vertex_count"] > 40:
        return False
    if abs(detail["signed_volume_m3"]) > 1e-12 or not (1.4e-6 <= detail["surface_area_m2"] <= 1.7e-6):
        return False
    bounds_min = np.asarray(detail["bounds_min_m"], dtype=np.float64)
    bounds_max = np.asarray(detail["bounds_max_m"], dtype=np.float64)
    expected_anchors = SOURCE_MICROFRAGMENT_ANCHORS_M.copy()
    if coordinate_quantization_m is not None:
        quantum = float(coordinate_quantization_m)
        if not np.isfinite(quantum) or quantum <= 0 or quantum > 1e-4:
            return False
        expected_anchors = np.rint(expected_anchors / quantum) * quantum
    anchor_min = expected_anchors.min(axis=0)
    anchor_max = expected_anchors.max(axis=0)
    anchor_tolerance = (max(2e-8, float(coordinate_quantization_m) * 0.025)
                        if coordinate_quantization_m is not None else 2e-7)
    if (np.max(np.abs(bounds_min - anchor_min)) > anchor_tolerance
            or np.max(np.abs(bounds_max - anchor_max)) > anchor_tolerance):
        return False
    component_vertices = vertices[np.asarray(detail["_vertex_ids"], dtype=np.int64)].astype(np.float64)
    nearest = np.linalg.norm(
        expected_anchors[:, None, :] - component_vertices[None, :, :], axis=2,
    ).min(axis=1)
    return bool(np.all(nearest <= anchor_tolerance))


def _fraction_cross2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _fraction_polygon_area2(points):
    if len(points) < 3:
        return Fraction(0)
    return sum(points[i][0] * points[(i + 1) % len(points)][1]
               - points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points)))


def _fraction_clip_convex(subject, clip):
    output = list(subject)
    for index, a in enumerate(clip):
        b = clip[(index + 1) % len(clip)]
        source = output
        output = []
        if not source:
            break
        previous = source[-1]
        previous_side = _fraction_cross2(a, b, previous)
        for current in source:
            current_side = _fraction_cross2(a, b, current)
            previous_inside = previous_side >= 0
            current_inside = current_side >= 0
            if previous_inside != current_inside:
                denominator = previous_side - current_side
                if denominator:
                    t = previous_side / denominator
                    output.append((previous[0] + t * (current[0] - previous[0]),
                                   previous[1] + t * (current[1] - previous[1])))
            if current_inside:
                output.append(current)
            previous, previous_side = current, current_side
    return output


def _fraction_point_on_segment(point, start, end):
    direction = tuple(end[i] - start[i] for i in range(3))
    offset = tuple(point[i] - start[i] for i in range(3))
    cross = (direction[1] * offset[2] - direction[2] * offset[1],
             direction[2] * offset[0] - direction[0] * offset[2],
             direction[0] * offset[1] - direction[1] * offset[0])
    if any(cross):
        return False
    axis = next((i for i, value in enumerate(direction) if value), None)
    if axis is None:
        return point == start
    parameter = offset[axis] / direction[axis]
    return 0 <= parameter <= 1


def _oriented_patch_boundary(triangles, split_points=()):
    universe = {point for triangle in triangles for point in triangle}
    universe.update(split_points)
    boundary = {}
    for triangle in triangles:
        for start, end in ((triangle[0], triangle[1]),
                           (triangle[1], triangle[2]),
                           (triangle[2], triangle[0])):
            axis = next(i for i, value in enumerate(tuple(end[j] - start[j] for j in range(3))) if value)
            direction = end[axis] - start[axis]
            points = [point for point in universe
                      if _fraction_point_on_segment(point, start, end)]
            points.sort(key=lambda point: (point[axis] - start[axis]) / direction)
            for left, right in zip(points, points[1:]):
                if left == right:
                    continue
                key = (left, right) if left < right else (right, left)
                sign = 1 if left < right else -1
                boundary[key] = boundary.get(key, 0) + sign
    return {key: value for key, value in boundary.items() if value}


def _exact_parent_barycentric(point, triangle):
    a, b, c = triangle
    edge_a = tuple(b[i] - a[i] for i in range(3))
    edge_b = tuple(c[i] - a[i] for i in range(3))
    offset = tuple(point[i] - a[i] for i in range(3))
    axes = ((0, 1), (0, 2), (1, 2))
    selected = max(axes, key=lambda pair: abs(edge_a[pair[0]] * edge_b[pair[1]]
                                              - edge_a[pair[1]] * edge_b[pair[0]]))
    i, j = selected
    determinant = edge_a[i] * edge_b[j] - edge_a[j] * edge_b[i]
    if not determinant:
        return None
    weight_b = (offset[i] * edge_b[j] - offset[j] * edge_b[i]) / determinant
    weight_c = (edge_a[i] * offset[j] - edge_a[j] * offset[i]) / determinant
    weight_a = Fraction(1) - weight_b - weight_c
    if any(weight_a * a[k] + weight_b * b[k] + weight_c * c[k] != point[k]
           for k in range(3)):
        return None
    return (weight_a, weight_b, weight_c)


def _source_face_subdivision_patch_report(candidate_vertices6, candidate_faces,
                                           source_vertices6, source_faces, face_map,
                                           evidence, repeated_source_face_ids):
    required = {"support_source_face_ids", "raw_vertex_lattice_numerators",
                "raw_vertex_lattice_denominators", "lattice_scale"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("source-face subdivision requires exact local patch geometry evidence")
    raw_support = list(evidence["support_source_face_ids"])
    if (not raw_support or any(isinstance(value, (bool, np.bool_))
                               or not isinstance(value, (int, np.integer)) for value in raw_support)):
        raise ValueError("source-face subdivision patch support requires integer source face IDs")
    support = tuple(map(int, raw_support))
    if (len(set(support)) != len(support)
            or any(value < 0 or value >= len(source_faces) for value in support)):
        raise ValueError("source-face subdivision patch support contains duplicate or out-of-range IDs")
    repeated = set(map(int, repeated_source_face_ids))
    if not repeated or not repeated.issubset(support):
        raise ValueError("source-face subdivision repeated parents are outside the declared patch support")

    scale = evidence["lattice_scale"]
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, np.integer)) or int(scale) <= 0:
        raise ValueError("source-face subdivision exact lattice scale is invalid")
    scale = int(scale)
    numerators = np.asarray(evidence["raw_vertex_lattice_numerators"])
    denominators = np.asarray(evidence["raw_vertex_lattice_denominators"])
    if (numerators.shape != (len(candidate_vertices6), 3)
            or denominators.shape != numerators.shape
            or numerators.dtype.kind not in "iuUS"
            or denominators.dtype.kind not in "iuUS"):
        raise ValueError("source-face subdivision exact lattice arrays have invalid shape or dtype")

    raw_points = []
    emitted_points = []
    for vertex_id in range(len(candidate_vertices6)):
        point = []
        emitted = []
        for axis in range(3):
            try:
                numerator = int(numerators[vertex_id, axis])
                denominator = int(denominators[vertex_id, axis])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("source-face subdivision exact lattice value is not an integer") from exc
            if denominator <= 0:
                raise ValueError("source-face subdivision exact lattice denominator must be positive")
            coordinate = Fraction(numerator, denominator) / scale
            f32 = np.float32(float(coordinate))
            if not np.isfinite(f32) or f32 != candidate_vertices6[vertex_id, axis]:
                raise ValueError("source-face subdivision raw geometry does not roundtrip to candidate Float32")
            point.append(Fraction(numerator, denominator))
            emitted.append(Fraction.from_float(float(f32)) * scale)
        raw_points.append(tuple(point))
        emitted_points.append(tuple(emitted))

    support_set = set(support)
    child_ids = []
    child_parent_counts = collections.Counter()
    for face_id, parents in enumerate(face_map):
        if any(parent in support_set for parent in parents):
            if len(parents) != 1 or parents[0] not in support_set:
                raise ValueError("source-face subdivision patch has a child with ambiguous parent support")
            child_ids.append(face_id)
            child_parent_counts[parents[0]] += 1
    if set(child_parent_counts) != support_set:
        raise ValueError("source-face subdivision patch omits a declared source parent")
    actual_repeated = {face_id for face_id, count in child_parent_counts.items() if count > 1}
    if actual_repeated != repeated:
        raise ValueError("source-face subdivision patch repeated-parent inventory differs")

    source_points_by_id = {}
    for face_id in support:
        for vertex_id in source_faces[face_id]:
            vertex_id = int(vertex_id)
            if vertex_id not in source_points_by_id:
                source_points_by_id[vertex_id] = tuple(
                    Fraction.from_float(float(np.float32(value))) * scale
                    for value in source_vertices6[vertex_id, :3]
                )
    source_triangles = [tuple(source_points_by_id[int(vertex)]
                              for vertex in source_faces[face_id])
                        for face_id in support]
    child_triangles_raw = []
    child_triangles_emitted = []
    coverage_by_parent = {face_id: Fraction(0) for face_id in support}
    child_polygons_by_parent = collections.defaultdict(list)
    for face_id in child_ids:
        parent_id = int(face_map[face_id][0])
        source_triangle = tuple(source_points_by_id[int(vertex)]
                                for vertex in source_faces[parent_id])
        child_face = tuple(map(int, candidate_faces[face_id]))
        raw_triangle = tuple(raw_points[vertex] for vertex in child_face)
        barycentric = [_exact_parent_barycentric(point, source_triangle)
                       for point in raw_triangle]
        if any(weights is None or min(weights) < 0 for weights in barycentric):
            raise ValueError(f"source-face subdivision child {face_id} is outside its exact parent triangle")
        parent_normal = tuple(source_triangle[1][i] - source_triangle[0][i] for i in range(3))
        parent_edge = tuple(source_triangle[2][i] - source_triangle[0][i] for i in range(3))
        parent_cross = (parent_normal[1] * parent_edge[2] - parent_normal[2] * parent_edge[1],
                        parent_normal[2] * parent_edge[0] - parent_normal[0] * parent_edge[2],
                        parent_normal[0] * parent_edge[1] - parent_normal[1] * parent_edge[0])
        child_edge_a = tuple(raw_triangle[1][i] - raw_triangle[0][i] for i in range(3))
        child_edge_b = tuple(raw_triangle[2][i] - raw_triangle[0][i] for i in range(3))
        child_cross = (child_edge_a[1] * child_edge_b[2] - child_edge_a[2] * child_edge_b[1],
                       child_edge_a[2] * child_edge_b[0] - child_edge_a[0] * child_edge_b[2],
                       child_edge_a[0] * child_edge_b[1] - child_edge_a[1] * child_edge_b[0])
        signed_area = sum(parent_cross[i] * child_cross[i] for i in range(3))
        parent_area = sum(value * value for value in parent_cross)
        if signed_area <= 0 or parent_area <= 0:
            raise ValueError(f"source-face subdivision child {face_id} has degenerate or reversed parent orientation")
        coverage_by_parent[parent_id] += signed_area / parent_area
        polygon = [(weights[1], weights[2]) for weights in barycentric]
        if _fraction_polygon_area2(polygon) <= 0:
            raise ValueError(f"source-face subdivision child {face_id} has degenerate projected area")
        child_polygons_by_parent[parent_id].append((face_id, polygon))
        child_triangles_raw.append(raw_triangle)
        child_triangles_emitted.append(tuple(emitted_points[vertex] for vertex in child_face))

    for parent_id, children in child_polygons_by_parent.items():
        for index, (left_id, left) in enumerate(children):
            for right_id, right in children[index + 1:]:
                overlap = _fraction_clip_convex(left, right)
                if abs(_fraction_polygon_area2(overlap)) > 0:
                    raise ValueError(f"source-face subdivision children {left_id} and {right_id} overlap in parent {parent_id}")

    raw_split_points = {point for triangle in source_triangles + child_triangles_raw for point in triangle}
    raw_source_boundary = _oriented_patch_boundary(source_triangles, raw_split_points)
    raw_child_boundary = _oriented_patch_boundary(child_triangles_raw, raw_split_points)
    if raw_source_boundary != raw_child_boundary:
        raise ValueError("source-face subdivision exact patch boundary differs from source support")
    emitted_source_triangles = [
        tuple(tuple(Fraction.from_float(float(np.float32(source_vertices6[int(vertex), axis]))) * scale
                    for axis in range(3)) for vertex in source_faces[face_id])
        for face_id in support
    ]
    emitted_split_points = {point for triangle in emitted_source_triangles + child_triangles_emitted for point in triangle}
    emitted_child_boundary = _oriented_patch_boundary(child_triangles_emitted, emitted_split_points)
    emitted_source_boundary = _oriented_patch_boundary(emitted_source_triangles, emitted_split_points)
    if emitted_source_boundary != emitted_child_boundary:
        raise ValueError("source-face subdivision Float32 patch boundary differs from source support")

    return {
        "passed": True,
        "support_source_face_ids": list(support),
        "repeated_source_face_ids": sorted(repeated),
        "output_child_face_count": int(len(child_ids)),
        "output_child_count_by_source_face": {str(face_id): int(child_parent_counts[face_id])
                                               for face_id in support},
        "exact_child_orientation_and_containment_checked": int(len(child_ids)),
        "exact_child_nonoverlap_parent_groups_checked": int(len(child_polygons_by_parent)),
        "source_parent_area_coverage_fraction_by_face": {
            str(face_id): float(coverage_by_parent[face_id]) for face_id in support
        },
        "exact_raw_patch_boundary_segment_count": int(len(raw_source_boundary)),
        "float32_patch_boundary_segment_count": int(len(emitted_source_boundary)),
        "exact_raw_and_float32_patch_boundary_conserved": True,
    }


def _normalize_source_face_ancestry(raw_face_map, source_face_count, stable_id, *,
                                    allow_two_parent_flip,
                                    allowed_subdivision_sources=()):
    """Validate unchanged faces, local flips, and explicitly named 1-to-N remesh ancestry."""
    allowed_values = list(allowed_subdivision_sources)
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) for value in allowed_values):
        raise ValueError("source-face subdivision opt-in requires integer source face IDs")
    allowed = set(map(int, allowed_values))
    if (len(allowed) != len(allowed_values)
            or any(value < 0 or value >= source_face_count for value in allowed)):
        raise ValueError("source-face subdivision opt-in contains duplicate or out-of-range source face IDs")
    face_map = []
    single_parent_sources = set()
    single_parent_uses = collections.Counter()
    paired_flip_uses = collections.Counter()
    for output_face, parents in enumerate(raw_face_map):
        values = np.asarray(parents)
        if values.dtype.kind not in "iu":
            raise ValueError(f"source face ancestry must use integer source IDs in row {stable_id}")
        values = values.astype(np.int64, copy=False).reshape(-1)
        if (len(values) == 0 or len(values) > (2 if allow_two_parent_flip else 1)
                or np.any(values < 0) or np.any(values >= source_face_count)
                or len(np.unique(values)) != len(values)):
            if not allow_two_parent_flip and len(values) != 1:
                raise ValueError(f"source face ancestry must map each output face to one unique source face in row {stable_id}")
            raise ValueError(f"source face ancestry is invalid for output face {output_face} in row {stable_id}")
        parent_ids = tuple(sorted(map(int, values)))
        if len(parent_ids) == 1:
            source_id = parent_ids[0]
            single_parent_uses[source_id] += 1
            single_parent_sources.add(source_id)
            if single_parent_uses[source_id] > 1 and source_id not in allowed:
                if allowed:
                    raise ValueError(f"source-face subdivision declaration differs from repeated ancestry in row {stable_id}")
                raise ValueError(f"source face ancestry must map each output face to one unique source face in row {stable_id}")
        else:
            if not allow_two_parent_flip:
                raise ValueError(f"source face ancestry must map each output face to one unique source face in row {stable_id}")
            paired_flip_uses[parent_ids] += 1
        face_map.append(parent_ids)
    repeated = {source_id for source_id, count in single_parent_uses.items() if count > 1}
    if repeated != allowed:
        raise ValueError(f"source-face subdivision declaration differs from repeated ancestry in row {stable_id}")
    if any(count != 2 for count in paired_flip_uses.values()):
        raise ValueError(f"a local 2-to-2 flip must map exactly two output faces to the same source-face pair in row {stable_id}")
    paired_sources = [source for pair in paired_flip_uses for source in pair]
    if (len(set(paired_sources)) != len(paired_sources)
            or set(paired_sources) & single_parent_sources):
        raise ValueError(f"a flipped source-face pair must be retired only by its two mapped output faces in row {stable_id}")
    return tuple(face_map)


def _normalize_pinned_parent_map(raw_map, row_count, parent_count, label, *, max_support):
    try:
        rows = list(raw_map)
    except TypeError as exc:
        raise ValueError(f"{label} must contain one support row per element") from exc
    if len(rows) != row_count:
        raise ValueError(f"{label} length differs from its parent geometry")
    normalized = []
    for output_id, raw_support in enumerate(rows):
        values = np.asarray(raw_support)
        if values.dtype.kind not in "iu":
            raise ValueError(f"{label} requires integer parent IDs")
        values = values.astype(np.int64, copy=False).reshape(-1)
        if (len(values) == 0 or len(values) > max_support
                or np.any(values < 0) or np.any(values >= parent_count)
                or len(np.unique(values)) != len(values)):
            raise ValueError(f"{label} has invalid support for output {output_id}")
        normalized.append(tuple(sorted(map(int, values))))
    return tuple(normalized)


def _lineage_map_sha256(stable_id, rows, *, kind):
    digest = hashlib.sha256()
    digest.update(b"numi.human.pleura.parent-lineage-map.v1\0")
    encoded_kind = kind.encode("ascii")
    digest.update(len(encoded_kind).to_bytes(2, "little", signed=False))
    digest.update(encoded_kind)
    digest.update(int(stable_id).to_bytes(4, "little", signed=False))
    offsets = [0]
    values = []
    for row in rows:
        values.extend(sorted(map(int, row)))
        offsets.append(len(values))
    digest.update(np.asarray(offsets, dtype="<u8").tobytes())
    digest.update(np.asarray(values, dtype="<u8").tobytes())
    return digest.hexdigest()


def _parent_surface_sha256(stable_id, vertices6, faces):
    """Hash canonical ABI5 row geometry separately from its whole-payload pin."""
    vertices = np.asarray(vertices6)
    triangles = np.asarray(faces)
    if (vertices.dtype.kind != "f" or vertices.dtype.itemsize != 4
            or triangles.dtype.kind not in "iu"
            or vertices.ndim != 2 or vertices.shape[1] != 6
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.isfinite(vertices).all()
            or (triangles.size and (np.any(triangles < 0)
                                    or int(triangles.max()) >= len(vertices)))):
        raise ValueError("immediate subdivision parent must be finite ABI5 Float32 geometry with integer local faces")
    canonical_vertices = np.ascontiguousarray(vertices, dtype="<f4")
    canonical_faces = np.ascontiguousarray(triangles, dtype="<u8")
    digest = hashlib.sha256()
    digest.update(b"numi.human.pleura.immediate-parent-surface.v1\0")
    digest.update(int(stable_id).to_bytes(4, "little", signed=False))
    digest.update(np.asarray(canonical_vertices.shape, dtype="<u8").tobytes())
    digest.update(canonical_vertices.tobytes())
    digest.update(np.asarray(canonical_faces.shape, dtype="<u8").tobytes())
    digest.update(canonical_faces.tobytes())
    return digest.hexdigest()


def _validate_source_face_subdivision_parent_chain(
        stable_id, candidate_faces, candidate_vertices6, ultimate_face_map,
        ultimate_vertex_map, source_face_count, source_vertex_count,
        evidence, ultimate_subdivision_ids, ultimate_patch_support_ids):
    """Authenticate candidate -> immediate parent -> source ancestry composition."""
    required = {
        "parent_vertices6", "parent_faces", "parent_payload_bytes", "parent_payload_sha256",
        "parent_surface_sha256", "candidate_face_ancestry",
        "candidate_vertex_ancestry", "parent_face_ancestry",
        "parent_vertex_ancestry", "immediate_subdivision_face_ids",
        "candidate_face_ancestry_sha256", "candidate_vertex_ancestry_sha256",
        "parent_face_ancestry_sha256", "parent_vertex_ancestry_sha256",
    }
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("immediate subdivision parent requires independently pinned geometry and complete ancestry evidence")
    for name in ("parent_payload_sha256", "parent_surface_sha256",
                 "candidate_face_ancestry_sha256", "candidate_vertex_ancestry_sha256",
                 "parent_face_ancestry_sha256", "parent_vertex_ancestry_sha256"):
        value = evidence[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"immediate subdivision parent {name} must be a lowercase SHA-256 pin")

    parent_vertices6 = np.asarray(evidence["parent_vertices6"])
    parent_faces_raw = np.asarray(evidence["parent_faces"])
    parent_payload_bytes = evidence["parent_payload_bytes"]
    if not isinstance(parent_payload_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("immediate subdivision parent requires the authenticated parent payload bytes")
    parent_payload_bytes = bytes(parent_payload_bytes)
    actual_payload_sha = hashlib.sha256(parent_payload_bytes).hexdigest()
    if actual_payload_sha != evidence["parent_payload_sha256"]:
        raise ValueError("immediate subdivision parent payload bytes differ from their independent SHA-256 pin")
    try:
        _, _, parent_rows = _parse_payload(parent_payload_bytes)
    except (ValueError, struct.error) as exc:
        raise ValueError("immediate subdivision parent payload is not a valid ABI5 anatomy payload") from exc
    if stable_id not in parent_rows:
        raise ValueError("immediate subdivision parent payload lacks the selected stable lobe row")
    payload_vertices6 = parent_rows[stable_id]["vertices6"]
    payload_faces = parent_rows[stable_id]["faces"]
    if (parent_faces_raw.dtype.kind not in "iu"
            or parent_faces_raw.ndim != 2 or parent_faces_raw.shape[1] != 3
            or not np.array_equal(parent_vertices6, payload_vertices6)
            or not np.array_equal(parent_faces_raw, payload_faces)):
        raise ValueError("immediate subdivision parent arrays differ from the independently pinned payload row")
    parent_faces = np.ascontiguousarray(parent_faces_raw, dtype="<u8")
    actual_surface_sha = _parent_surface_sha256(stable_id, parent_vertices6, parent_faces_raw)
    if actual_surface_sha != evidence["parent_surface_sha256"]:
        raise ValueError("immediate subdivision parent row geometry differs from its independent SHA-256 pin")

    candidate_face_to_parent = _normalize_pinned_parent_map(
        evidence["candidate_face_ancestry"], len(candidate_faces), len(parent_faces),
        f"candidate-to-immediate-parent face ancestry row {stable_id}", max_support=1)
    candidate_vertex_to_parent = _normalize_pinned_parent_map(
        evidence["candidate_vertex_ancestry"], len(candidate_vertices6), len(parent_vertices6),
        f"candidate-to-immediate-parent vertex ancestry row {stable_id}",
        max_support=len(parent_vertices6))
    parent_face_to_source = _normalize_pinned_parent_map(
        evidence["parent_face_ancestry"], len(parent_faces), source_face_count,
        f"immediate-parent-to-source face ancestry row {stable_id}", max_support=2)
    parent_vertex_to_source = _normalize_pinned_parent_map(
        evidence["parent_vertex_ancestry"], len(parent_vertices6), source_vertex_count,
        f"immediate-parent-to-source vertex ancestry row {stable_id}",
        max_support=source_vertex_count)

    actual_map_hashes = {
        "candidate_face_ancestry_sha256": _lineage_map_sha256(
            stable_id, candidate_face_to_parent, kind="candidate-face-to-parent-face"),
        "candidate_vertex_ancestry_sha256": _lineage_map_sha256(
            stable_id, candidate_vertex_to_parent, kind="candidate-vertex-to-parent-vertex"),
        "parent_face_ancestry_sha256": _lineage_map_sha256(
            stable_id, parent_face_to_source, kind="parent-face-to-source-face"),
        "parent_vertex_ancestry_sha256": _lineage_map_sha256(
            stable_id, parent_vertex_to_source, kind="parent-vertex-to-source-vertex"),
    }
    for name, actual in actual_map_hashes.items():
        if actual != evidence[name]:
            raise ValueError(f"immediate subdivision parent {name} differs from its independent SHA-256 pin")

    composed_faces = []
    for parent_ids in candidate_face_to_parent:
        composed_faces.append(tuple(sorted({source_id for parent_id in parent_ids
                                            for source_id in parent_face_to_source[parent_id]})))
    composed_vertices = []
    for parent_ids in candidate_vertex_to_parent:
        composed_vertices.append(tuple(sorted({source_id for parent_id in parent_ids
                                               for source_id in parent_vertex_to_source[parent_id]})))
    if tuple(composed_faces) != tuple(tuple(sorted(row)) for row in ultimate_face_map):
        raise ValueError("candidate-to-immediate-to-source face ancestry differs from the complete source face ancestry")
    if tuple(composed_vertices) != tuple(tuple(sorted(row)) for row in ultimate_vertex_map):
        raise ValueError("candidate-to-immediate-to-source vertex ancestry differs from the complete source vertex ancestry")

    parent_use_counts = collections.Counter(row[0] for row in candidate_face_to_parent)
    repeated_parent_ids = tuple(sorted(face_id for face_id, count in parent_use_counts.items() if count > 1))
    declared_repeated = evidence["immediate_subdivision_face_ids"]
    if (not isinstance(declared_repeated, (list, tuple))
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer)) for value in declared_repeated)):
        raise ValueError("immediate subdivision parent repeated-face inventory must contain integer IDs")
    declared_repeated = tuple(sorted(map(int, declared_repeated)))
    if len(set(declared_repeated)) != len(declared_repeated) or repeated_parent_ids != declared_repeated:
        raise ValueError("immediate subdivision parent repeated-face inventory differs from candidate ancestry")
    repeated_source_ids = tuple(sorted({parent_face_to_source[parent_id][0]
                                        for parent_id in repeated_parent_ids
                                        if len(parent_face_to_source[parent_id]) == 1}))
    if len(repeated_source_ids) != len(repeated_parent_ids):
        raise ValueError("repeated immediate subdivision faces do not each have one original source face")
    if repeated_source_ids != tuple(sorted(map(int, ultimate_subdivision_ids))):
        raise ValueError("immediate and original-source subdivision declarations do not compose")

    raw_patch_support = list(ultimate_patch_support_ids)
    if (not raw_patch_support
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer)) for value in raw_patch_support)):
        raise ValueError("immediate subdivision parent requires the complete ultimate patch-support face inventory")
    ultimate_patch_support = tuple(sorted(map(int, raw_patch_support)))
    if (len(set(ultimate_patch_support)) != len(ultimate_patch_support)
            or any(face_id < 0 or face_id >= source_face_count
                   for face_id in ultimate_patch_support)):
        raise ValueError("immediate subdivision parent ultimate patch support is duplicate or out of range")
    used_parent_ids = {row[0] for row in candidate_face_to_parent}
    ultimate_support_set = set(ultimate_patch_support)
    immediate_patch_support = []
    composed_patch_support = set()
    for parent_id, source_ids in enumerate(parent_face_to_source):
        source_support = set(source_ids) & ultimate_support_set
        if not source_support:
            continue
        if not set(source_ids).issubset(ultimate_support_set):
            raise ValueError("immediate parent face ancestry crosses the declared subdivision patch boundary")
        if parent_id not in used_parent_ids:
            raise ValueError("immediate parent subdivision support face has no candidate child")
        immediate_patch_support.append(parent_id)
        composed_patch_support.update(source_ids)
    if composed_patch_support != ultimate_support_set:
        raise ValueError("immediate parent patch support does not compose to the complete original-source patch support")
    if not set(repeated_parent_ids).issubset(immediate_patch_support):
        raise ValueError("repeated immediate subdivision faces are outside the authenticated patch support")

    return {
        "passed": True,
        "parent_payload_sha256": evidence["parent_payload_sha256"],
        "parent_surface_sha256": actual_surface_sha,
        "candidate_face_ancestry_sha256": actual_map_hashes["candidate_face_ancestry_sha256"],
        "candidate_vertex_ancestry_sha256": actual_map_hashes["candidate_vertex_ancestry_sha256"],
        "parent_face_ancestry_sha256": actual_map_hashes["parent_face_ancestry_sha256"],
        "parent_vertex_ancestry_sha256": actual_map_hashes["parent_vertex_ancestry_sha256"],
        "immediate_subdivision_face_ids": list(repeated_parent_ids),
        "original_source_subdivision_face_ids": list(repeated_source_ids),
        "immediate_patch_support_face_ids": list(immediate_patch_support),
        "ultimate_patch_support_face_ids": list(ultimate_patch_support),
        "face_ancestry_composition_checked": len(composed_faces),
        "vertex_ancestry_composition_checked": len(composed_vertices),
        "parent_vertices6": parent_vertices6,
        "parent_faces": parent_faces.astype(np.int64, copy=False),
        "candidate_face_to_parent": candidate_face_to_parent,
    }

def _closed_oriented_collapse_surface_summary(vertices6, faces, label):
    vertices, triangles = np.asarray(vertices6), np.asarray(faces)
    if (vertices.dtype.kind != "f" or vertices.dtype.itemsize != 4
            or vertices.ndim != 2 or vertices.shape[1] != 6
            or triangles.dtype.kind not in "iu" or triangles.ndim != 2
            or triangles.shape[1] != 3 or not len(triangles) or not np.isfinite(vertices).all()
            or np.any(triangles < 0) or int(triangles.max()) >= len(vertices)):
        raise ValueError(f"{label} edge-collapse surface arrays are malformed")
    triangles = np.asarray(triangles, dtype=np.int64)
    if (len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles)
            or not np.array_equal(np.unique(triangles.reshape(-1)), np.arange(len(vertices)))):
        raise ValueError(f"{label} edge-collapse surface has duplicate faces or unused vertex slots")
    topology = topology_report(triangles)
    components, _ = _edge_components(triangles)
    if topology["nonmanifold_edge_count"] or topology["orientation_error_edge_count"]:
        raise ValueError(f"{label} edge-collapse surface has nonmanifold or inconsistently oriented edges")
    component_by_face = {
        int(face_id): component_id
        for component_id, component in enumerate(components)
        for face_id in component
    }
    edge_owners = collections.defaultdict(list)
    for face_id, triangle in enumerate(triangles):
        a, b, c = map(int, triangle)
        for start, end in ((a, b), (b, c), (c, a)):
            edge_owners[(min(start, end), max(start, end))].append((face_id, start, end))
    boundary_edges_directed = []
    boundary_edges_by_component = collections.Counter()
    for edge, owners in edge_owners.items():
        if len(owners) == 1:
            face_id, start, end = owners[0]
            boundary_edges_directed.append((start, end))
            boundary_edges_by_component[component_by_face[face_id]] += 1
    boundary_edges_directed.sort()
    xyz = vertices[:, :3].astype(np.float64)
    points = xyz[triangles]
    areas2 = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                     points[:, 2] - points[:, 0]), axis=1)
    if not np.isfinite(areas2).all() or np.any(areas2 <= 0):
        raise ValueError(f"{label} edge-collapse surface has degenerate faces")
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]), axis=0)
    edge_count = len(np.unique(np.sort(edges, axis=1), axis=0))
    component_rows = []
    for component in components:
        component_face_ids = np.asarray(component, dtype=np.int64)
        component_triangles = triangles[component_face_ids]
        component_vertex_ids = np.unique(component_triangles.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[component_vertex_ids] = np.arange(len(component_vertex_ids), dtype=np.int64)
        local_faces = remap[component_triangles]
        local_edges = np.concatenate((local_faces[:, [0, 1]], local_faces[:, [1, 2]],
                                      local_faces[:, [2, 0]]), axis=0)
        local_edge_count = len(np.unique(np.sort(local_edges, axis=1), axis=0))
        local_volume = float(signed_volume(xyz[component_vertex_ids], local_faces))
        if not np.isfinite(local_volume):
            raise ValueError(f"{label} edge-collapse component has non-finite signed volume")
        component_rows.append({
            "face_count": int(len(component_face_ids)),
            "vertex_count": int(len(component_vertex_ids)),
            "edge_count": int(local_edge_count),
            "euler_characteristic": int(len(component_vertex_ids) - local_edge_count + len(component_face_ids)),
            "boundary_edge_count": int(boundary_edges_by_component[len(component_rows)]),
            "signed_volume_m3": local_volume,
        })
    volume = float(sum(row["signed_volume_m3"] for row in component_rows))
    return {
        "vertex_count": int(len(vertices)), "face_count": int(len(triangles)),
        "edge_count": int(edge_count), "euler_characteristic": int(len(vertices) - edge_count + len(triangles)),
        "connected_face_components": int(len(components)),
        "component_topology_and_volume": component_rows,
        "boundary_edge_count": int(topology["boundary_edge_count"]),
        "boundary_edges_directed": [list(edge) for edge in boundary_edges_directed],
        "nonmanifold_edge_count": int(topology["nonmanifold_edge_count"]),
        "orientation_error_edge_count": int(topology["orientation_error_edge_count"]),
        "minimum_triangle_area_m2": float(areas2.min() * 0.5), "signed_volume_m3": volume,
    }


def _edge_collapse_repeated_source_face_ids(stable_id, evidence, source_face_count):
    if not isinstance(evidence, dict) or "parent_face_ancestry" not in evidence:
        raise ValueError("source-face edge-collapse evidence lacks parent face ancestry")
    parent_faces = np.asarray(evidence.get("parent_faces"))
    if parent_faces.dtype.kind not in "iu" or parent_faces.ndim != 2 or parent_faces.shape[1] != 3:
        raise ValueError("source-face edge-collapse evidence lacks integer immediate-parent faces")
    parent_map = _normalize_pinned_parent_map(
        evidence["parent_face_ancestry"], len(parent_faces), source_face_count,
        f"collapse parent-to-source face ancestry row {stable_id}", max_support=2,
    )
    counts = collections.Counter(source_id for row in parent_map if len(row) == 1 for source_id in row)
    return tuple(sorted(source_id for source_id, count in counts.items() if count > 1))


def _validate_source_face_edge_collapse_parent_chain(
        stable_id, candidate_faces, candidate_vertices6, ultimate_face_map,
        ultimate_vertex_map, source_face_count, source_vertex_count, evidence):
    # This opt-in proves operation identity and ancestry, never anatomical admission.
    required = {
        "parent_vertices6", "parent_faces", "parent_payload_bytes", "parent_payload_sha256",
        "parent_surface_sha256", "candidate_face_ancestry", "candidate_vertex_ancestry",
        "parent_face_ancestry", "parent_vertex_ancestry",
        "candidate_face_ancestry_sha256", "candidate_vertex_ancestry_sha256",
        "parent_face_ancestry_sha256", "parent_vertex_ancestry_sha256",
        "edge_endpoint_parent_vertex_ids", "retired_parent_face_ids",
        "retained_position_policy", "collapse_position_f32_m",
        "maximum_endpoint_displacement_bound_m", "maximum_metric_deviation_bound_m",
    }
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("source-face edge collapse requires pinned parent geometry, full ancestry, and operation evidence")
    hash_fields = ("parent_payload_sha256", "parent_surface_sha256",
                   "candidate_face_ancestry_sha256", "candidate_vertex_ancestry_sha256",
                   "parent_face_ancestry_sha256", "parent_vertex_ancestry_sha256")
    for name in hash_fields:
        value = evidence[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError(f"source-face edge collapse {name} must be a lowercase SHA-256 pin")
    parent_vertices6, parent_faces_raw = np.asarray(evidence["parent_vertices6"]), np.asarray(evidence["parent_faces"])
    raw_payload = evidence["parent_payload_bytes"]
    if not isinstance(raw_payload, (bytes, bytearray, memoryview)):
        raise ValueError("source-face edge collapse requires exact immediate-parent payload bytes")
    raw_payload = bytes(raw_payload)
    if hashlib.sha256(raw_payload).hexdigest() != evidence["parent_payload_sha256"]:
        raise ValueError("source-face edge-collapse parent payload differs from its independent SHA-256 pin")
    try:
        _, _, parent_rows = _parse_payload(raw_payload)
    except (ValueError, struct.error) as exc:
        raise ValueError("source-face edge-collapse parent is not a valid ABI5 payload") from exc
    if stable_id not in parent_rows:
        raise ValueError("source-face edge-collapse parent payload lacks the selected lobe row")
    payload_row = parent_rows[stable_id]
    if (parent_vertices6.dtype.kind != "f" or parent_vertices6.dtype.itemsize != 4
            or parent_faces_raw.dtype.kind not in "iu" or parent_faces_raw.ndim != 2
            or parent_faces_raw.shape[1] != 3
            or not np.array_equal(parent_vertices6, payload_row["vertices6"])
            or not np.array_equal(parent_faces_raw, payload_row["faces"])):
        raise ValueError("source-face edge-collapse parent arrays differ from their pinned ABI5 row")
    parent_faces = np.asarray(parent_faces_raw, dtype=np.int64)
    parent_surface_sha = _parent_surface_sha256(stable_id, parent_vertices6, parent_faces_raw)
    if parent_surface_sha != evidence["parent_surface_sha256"]:
        raise ValueError("source-face edge-collapse parent row differs from its independent geometry pin")

    candidate_faces, candidate_vertices6 = np.asarray(candidate_faces), np.asarray(candidate_vertices6)
    if (candidate_faces.dtype.kind not in "iu" or candidate_faces.ndim != 2
            or candidate_faces.shape[1] != 3 or candidate_vertices6.dtype.kind != "f"
            or candidate_vertices6.dtype.itemsize != 4):
        raise ValueError("source-face edge-collapse candidate must use Float32 vertices and integer faces")
    candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
    cfp = _normalize_pinned_parent_map(
        evidence["candidate_face_ancestry"], len(candidate_faces), len(parent_faces),
        f"collapse candidate-to-parent face ancestry row {stable_id}", max_support=1)
    cvp = _normalize_pinned_parent_map(
        evidence["candidate_vertex_ancestry"], len(candidate_vertices6), len(parent_vertices6),
        f"collapse candidate-to-parent vertex ancestry row {stable_id}", max_support=2)
    pfs = _normalize_pinned_parent_map(
        evidence["parent_face_ancestry"], len(parent_faces), source_face_count,
        f"collapse parent-to-source face ancestry row {stable_id}", max_support=2)
    pvs = _normalize_pinned_parent_map(
        evidence["parent_vertex_ancestry"], len(parent_vertices6), source_vertex_count,
        f"collapse parent-to-source vertex ancestry row {stable_id}", max_support=source_vertex_count)
    map_hashes = {
        "candidate_face_ancestry_sha256": _lineage_map_sha256(stable_id, cfp, kind="candidate-face-to-parent-face"),
        "candidate_vertex_ancestry_sha256": _lineage_map_sha256(stable_id, cvp, kind="candidate-vertex-to-parent-vertex"),
        "parent_face_ancestry_sha256": _lineage_map_sha256(stable_id, pfs, kind="parent-face-to-source-face"),
        "parent_vertex_ancestry_sha256": _lineage_map_sha256(stable_id, pvs, kind="parent-vertex-to-source-vertex"),
    }
    for name, actual in map_hashes.items():
        if actual != evidence[name]:
            raise ValueError(f"source-face edge-collapse {name} differs from its independent pin")
    repeated_source_ids = _edge_collapse_repeated_source_face_ids(stable_id, evidence, source_face_count)
    parent_subdivision_report = None
    nested_subdivision = evidence.get("parent_source_subdivision_evidence")
    if repeated_source_ids:
        required_nested = {
            "subdivided_source_face_ids", "source_patch_support_face_ids",
            "patch_geometry", "subdivision_parent_evidence",
        }
        if not isinstance(nested_subdivision, dict) or not required_nested.issubset(nested_subdivision):
            raise ValueError("edge-collapse parent has repeated source faces without its independently checked exact subdivision chain")
        declared_repeated = nested_subdivision["subdivided_source_face_ids"]
        if (not isinstance(declared_repeated, (list, tuple))
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer)) for value in declared_repeated)
                or tuple(sorted(map(int, declared_repeated))) != repeated_source_ids
                or len(set(map(int, declared_repeated))) != len(declared_repeated)):
            raise ValueError("edge-collapse parent exact subdivision declaration differs from repeated source ancestry")
        subdivision_parent = _validate_source_face_subdivision_parent_chain(
            stable_id, parent_faces, parent_vertices6, pfs, pvs,
            source_face_count, source_vertex_count,
            nested_subdivision["subdivision_parent_evidence"], repeated_source_ids,
            nested_subdivision["source_patch_support_face_ids"],
        )
        parent_patch_report = _source_face_subdivision_patch_report(
            parent_vertices6, parent_faces,
            subdivision_parent["parent_vertices6"], subdivision_parent["parent_faces"],
            subdivision_parent["candidate_face_to_parent"],
            nested_subdivision["patch_geometry"],
            subdivision_parent["immediate_subdivision_face_ids"],
        )
        parent_subdivision_report = {
            "parent_exact_subdivision_chain": {
                key: value for key, value in subdivision_parent.items()
                if key not in ("parent_vertices6", "parent_faces", "candidate_face_to_parent")
            },
            "parent_exact_subdivision_patch_geometry": parent_patch_report,
        }
    elif nested_subdivision is not None:
        raise ValueError("edge-collapse parent declares an exact subdivision chain that has no repeated source face")

    composed_faces = tuple(tuple(sorted({source for p in row for source in pfs[p]})) for row in cfp)
    composed_vertices = tuple(tuple(sorted({source for p in row for source in pvs[p]})) for row in cvp)
    if composed_faces != tuple(tuple(sorted(map(int, row))) for row in ultimate_face_map):
        raise ValueError("edge-collapse candidate-to-parent-to-source face ancestry differs from the complete source map")
    if composed_vertices != tuple(tuple(sorted(map(int, row))) for row in ultimate_vertex_map):
        raise ValueError("edge-collapse candidate-to-parent-to-source vertex ancestry differs from the complete source map")

    endpoints = evidence["edge_endpoint_parent_vertex_ids"]
    if (not isinstance(endpoints, (tuple, list)) or len(endpoints) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in endpoints)):
        raise ValueError("edge-collapse endpoint evidence requires two integer immediate-parent vertex IDs")
    removed, retained = map(int, endpoints)
    if removed == retained or min(removed, retained) < 0 or max(removed, retained) >= len(parent_vertices6):
        raise ValueError("edge-collapse endpoint evidence is duplicate or out of range")
    raw_retired = evidence["retired_parent_face_ids"]
    if (not isinstance(raw_retired, (tuple, list))
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in raw_retired)):
        raise ValueError("edge-collapse retired-face evidence requires integer immediate-parent face IDs")
    retired = tuple(sorted(map(int, raw_retired)))
    if len(set(retired)) != len(retired):
        raise ValueError("edge-collapse retired-face evidence contains duplicate IDs")
    parent_topology = _closed_oriented_collapse_surface_summary(parent_vertices6, parent_faces, "immediate parent")
    candidate_topology = _closed_oriented_collapse_surface_summary(candidate_vertices6, candidate_faces, "candidate")
    edge_faces = np.flatnonzero(np.sum(np.isin(parent_faces, (removed, retained)), axis=1) == 2)
    if len(edge_faces) != 2 or retired != tuple(map(int, edge_faces)):
        raise ValueError("edge-collapse retired faces are not exactly the two incident immediate-parent faces")

    def neighbors(vertex):
        incident = parent_faces[np.any(parent_faces == vertex, axis=1)]
        return set(map(int, incident.reshape(-1))) - {vertex}
    common = neighbors(removed) & neighbors(retained)
    opposite = set()
    for face_id in edge_faces:
        opposite.update(map(int, parent_faces[face_id][~np.isin(parent_faces[face_id], (removed, retained))]))
    if len(common) != 2 or common != opposite:
        raise ValueError("edge-collapse immediate-parent edge fails the closed two-manifold link condition")

    collapsed_ids = [i for i, parents in enumerate(cvp) if parents == tuple(sorted((removed, retained)))]
    if len(collapsed_ids) != 1:
        raise ValueError("edge-collapse candidate requires exactly one output vertex supported by both endpoints")
    collapsed = collapsed_ids[0]
    singleton_ids = [row[0] for row in cvp if len(row) == 1]
    expected_singletons = set(range(len(parent_vertices6))) - {removed, retained}
    if (len(singleton_ids) != len(expected_singletons) or set(singleton_ids) != expected_singletons
            or len(candidate_vertices6) != len(parent_vertices6) - 1):
        raise ValueError("edge-collapse candidate vertex support omits or duplicates an uncollapsed parent vertex")
    parent_to_candidate = {parent: out for out, row in enumerate(cvp) if len(row) == 1 for parent in row}
    parent_to_candidate[removed] = collapsed
    parent_to_candidate[retained] = collapsed
    parent_boundary = parent_topology["boundary_edges_directed"]
    candidate_boundary = candidate_topology["boundary_edges_directed"]
    parent_boundary_vertices = {vertex for edge in parent_boundary for vertex in edge}
    if {removed, retained} & parent_boundary_vertices:
        raise ValueError("edge-collapse operation touches a registered open source-boundary feature")
    remapped_parent_boundary = sorted(
        (parent_to_candidate[start], parent_to_candidate[end]) for start, end in parent_boundary
    )
    if remapped_parent_boundary != [tuple(edge) for edge in candidate_boundary]:
        raise ValueError("edge-collapse changes an immediate-parent open-boundary identity or orientation")

    policy = evidence["retained_position_policy"]
    if policy not in ("midpoint", "keep"):
        raise ValueError("edge-collapse retained-position policy must be midpoint or keep")
    if policy == "midpoint":
        position = ((parent_vertices6[removed, :3].astype(np.float64)
                     + parent_vertices6[retained, :3].astype(np.float64)) * 0.5).astype(np.float32)
    else:
        position = parent_vertices6[retained, :3].copy()
    raw_position = np.asarray(evidence["collapse_position_f32_m"])
    if (raw_position.shape != (3,) or raw_position.dtype.kind != "f"
            or not np.isfinite(raw_position).all()
            or not np.array_equal(np.asarray(raw_position, dtype=np.float32), position)
            or not np.array_equal(candidate_vertices6[collapsed, :3], position)):
        raise ValueError("edge-collapse position differs from the exact declared Float32 operation")
    for out, row in enumerate(cvp):
        if len(row) == 1 and not np.array_equal(candidate_vertices6[out, :3], parent_vertices6[row[0], :3]):
            raise ValueError("edge-collapse changed vertex geometry outside its two-endpoint support")

    parent_to_face = {}
    for out, row in enumerate(cfp):
        if row[0] in parent_to_face:
            raise ValueError("edge-collapse maps multiple output faces to one immediate-parent face")
        parent_to_face[row[0]] = out
    if set(parent_to_face) != set(range(len(parent_faces))) - set(retired):
        raise ValueError("edge-collapse face ancestry omits or invents non-retired parent faces")
    for parent_face, out_face in parent_to_face.items():
        expected = tuple(parent_to_candidate[int(v)] for v in parent_faces[parent_face])
        if tuple(map(int, candidate_faces[out_face])) != expected:
            raise ValueError("edge-collapse candidate face is not the exact oriented remap of its parent face")

    edge_length = float(np.linalg.norm(parent_vertices6[removed, :3].astype(float)
                                       - parent_vertices6[retained, :3].astype(float)))
    endpoint_delta = max(float(np.linalg.norm(position.astype(float)
                                               - parent_vertices6[v, :3].astype(float)))
                         for v in (removed, retained))
    endpoint_bound, metric_bound = (evidence["maximum_endpoint_displacement_bound_m"],
                                    evidence["maximum_metric_deviation_bound_m"])
    for name, value in (("maximum_endpoint_displacement_bound_m", endpoint_bound),
                        ("maximum_metric_deviation_bound_m", metric_bound)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"edge-collapse {name} must be finite and non-negative")
    if endpoint_bound > edge_length or endpoint_delta > endpoint_bound:
        raise ValueError("edge-collapse endpoint displacement exceeds its local edge-length bound")

    star_faces = np.flatnonzero(np.any(np.isin(parent_faces, (removed, retained)), axis=1))
    max_metric_delta = 0.0
    for parent_face in star_faces:
        if int(parent_face) in retired:
            continue
        out_face = parent_to_face[int(parent_face)]
        old_points = parent_vertices6[parent_faces[parent_face], :3].astype(np.float64)
        new_points = candidate_vertices6[candidate_faces[out_face], :3].astype(np.float64)
        for edge in range(3):
            old_len = float(np.linalg.norm(old_points[(edge + 1) % 3] - old_points[edge]))
            new_len = float(np.linalg.norm(new_points[(edge + 1) % 3] - new_points[edge]))
            max_metric_delta = max(max_metric_delta, abs(new_len - old_len))
    star_ids = np.unique(parent_faces[star_faces].reshape(-1))
    star_xyz = parent_vertices6[star_ids, :3].astype(np.float64)
    star_diameter = float(np.linalg.norm(star_xyz[:, None, :] - star_xyz[None, :, :], axis=2).max())
    if metric_bound > star_diameter or max_metric_delta > metric_bound:
        raise ValueError("edge-collapse surviving local metric deviation exceeds its support bound")
    parent_components, _ = _edge_components(parent_faces)
    candidate_components, _ = _edge_components(candidate_faces)
    if len(parent_components) != len(candidate_components):
        raise ValueError("edge-collapse changed the closed-surface component count")
    parent_component_by_face = {
        int(face_id): component_id
        for component_id, component in enumerate(parent_components)
        for face_id in component
    }
    component_map = {}
    for candidate_component_id, component in enumerate(candidate_components):
        parent_component_ids = {
            parent_component_by_face[cfp[int(candidate_face_id)][0]]
            for candidate_face_id in component
        }
        if len(parent_component_ids) != 1:
            raise ValueError("edge-collapse merges distinct closed-surface components")
        parent_component_id = next(iter(parent_component_ids))
        if parent_component_id in component_map:
            raise ValueError("edge-collapse splits an immediate-parent closed-surface component")
        component_map[parent_component_id] = candidate_component_id
    if set(component_map) != set(range(len(parent_components))):
        raise ValueError("edge-collapse loses an immediate-parent closed-surface component")
    for parent_component_id, candidate_component_id in component_map.items():
        parent_metrics = parent_topology["component_topology_and_volume"][parent_component_id]
        candidate_metrics = candidate_topology["component_topology_and_volume"][candidate_component_id]
        if parent_metrics["euler_characteristic"] != candidate_metrics["euler_characteristic"]:
            raise ValueError("edge-collapse changed a component Euler characteristic")
        parent_component_vertices = set(map(
            int, parent_faces[np.asarray(parent_components[parent_component_id], dtype=np.int64)].reshape(-1)
        ))
        changed_component = bool(parent_component_vertices & {removed, retained})
        if changed_component and (parent_metrics["boundary_edge_count"]
                                  or candidate_metrics["boundary_edge_count"]):
            raise ValueError("edge-collapse operation is not confined to a closed parent component")
        if changed_component and (parent_metrics["signed_volume_m3"] * candidate_metrics["signed_volume_m3"] <= 0):
            raise ValueError("edge-collapse reversed or collapsed the signed volume of its source component")
    for parent_face in star_faces:
        if int(parent_face) in retired:
            continue
        out_face = parent_to_face[int(parent_face)]
        old = parent_vertices6[parent_faces[parent_face], :3].astype(np.float64)
        new = candidate_vertices6[candidate_faces[out_face], :3].astype(np.float64)
        if float(np.dot(np.cross(old[1] - old[0], old[2] - old[0]),
                        np.cross(new[1] - new[0], new[2] - new[0]))) <= 0:
            raise ValueError("edge-collapse reverses a surviving face in the changed local star")
    return {
        "passed": True, "operation": "one authenticated local manifold edge collapse",
        "parent_payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "parent_surface_sha256": parent_surface_sha,
        "candidate_surface_sha256": _parent_surface_sha256(stable_id, candidate_vertices6, candidate_faces),
        **map_hashes,
        "edge_endpoint_parent_vertex_ids": [removed, retained],
        "retired_parent_face_ids": list(retired),
        "collapsed_candidate_vertex_id": int(collapsed),
        "collapse_position_policy": policy, "collapse_position_f32_m": position.astype(float).tolist(),
        "maximum_endpoint_displacement_m": endpoint_delta,
        "maximum_endpoint_displacement_bound_m": float(endpoint_bound),
        "maximum_surviving_edge_length_deviation_m": max_metric_delta,
        "maximum_metric_deviation_bound_m": float(metric_bound),
        "parent_topology": parent_topology, "candidate_topology": candidate_topology,
        "signed_volume_delta_m3": candidate_topology["signed_volume_m3"] - parent_topology["signed_volume_m3"],
        "source_face_ancestry_composition_checked": len(composed_faces),
        "source_vertex_ancestry_composition_checked": len(composed_vertices),
        "parent_source_subdivision_evidence": parent_subdivision_report,
        "exact_source_and_world_self_intersection_rechecks_required": True,
    }


def derive_lung_union_exterior(lobes, *, allow_source_microfragment=True,
                               coordinate_quantization_m=None, source_lineage_lobes=None,
                               source_face_ancestry=None, source_vertex_ancestry=None,
                               allow_multi_parent_source_face_ancestry=(),
                               allow_source_face_subdivisions=None,
                               source_face_subdivision_patch_geometry=None,
                               source_face_subdivision_parent_evidence=None,
                               source_face_edge_collapse_parent_evidence=None):
    """Build the external union surface from five registered lobe meshes.

    `lobes` maps stable IDs 305..309 to `(vertices6, local_faces)` arrays.
    Exact reciprocal triangles are canceled from the exterior. Explicit
    multi-parent face ancestry is disabled by default; an authenticated caller
    may opt named rows into one exact local 2-to-2 flip pair. A separate
    source-face subdivision opt-in names repeated parents and requires exact
    child containment/orientation, non-overlap, and whole-support boundary
    conservation evidence; per-face counts alone never authorize a subdivision. The
    optional `source_face_subdivision_parent_evidence` separates the immediate
    registered parent geometry from the ultimate source-lineage rows. It requires
    independent whole-payload and row-geometry SHA-256 pins plus pinned candidate-to-
    parent and parent-to-source ancestry maps; the composed maps must equal the
    existing candidate-to-source maps. Subdivision geometry is checked against the
    immediate parent, while omitted-fragment identity remains checked against
    `source_lineage_lobes`. By default, any detached component is a hard error. A source-specific microfragment
    may be omitted only when it matches the exact registered 305/308 source
    anchor coordinates, sub-millimeter-scale bounds, owner identities, and
    bounded area/volume recorded below. With explicit ancestry from an
    authenticated caller, unrelated candidate inventory changes are allowed
    while the omitted component must map to the same source faces, ordered
    source connectivity, and source vertex inventory. Any other detached
    component fails. The separate `source_face_edge_collapse_parent_evidence`
    opt-in authenticates one local closed-surface edge collapse against its
    exact immediate ABI5 parent and composes that parent lineage to the
    original source maps. It does not relax the external-union, source/world
    self-intersection, or anatomical-neighbor gates.
    """
    # The respiratory preparation receipt explicitly uses zero for the
    # unquantized alternative. It has the same source anchors as an omitted
    # quantization field; it must not be interpreted as a zero-sized grid.
    if coordinate_quantization_m == 0:
        coordinate_quantization_m = None
    if set(map(int, lobes)) != set(LOBE_IDS):
        raise ValueError("the pleura proxy requires exactly the five stable lung lobe IDs")

    if (source_face_ancestry is None) != (source_vertex_ancestry is None):
        raise ValueError("source pleura face and vertex ancestry must be supplied together")
    has_explicit_ancestry = source_face_ancestry is not None
    try:
        raw_multi_parent_rows = list(allow_multi_parent_source_face_ancestry)
    except TypeError as exc:
        raise ValueError("multi-parent source-face ancestry opt-in must be an iterable of stable lobe IDs") from exc
    multi_parent_rows = set()
    for stable_id in raw_multi_parent_rows:
        if isinstance(stable_id, (bool, np.bool_)):
            raise ValueError("multi-parent source-face ancestry opt-in rejects boolean stable lobe IDs")
        if isinstance(stable_id, (int, np.integer)):
            normalized_id = int(stable_id)
        elif isinstance(stable_id, str) and stable_id.isdecimal() and str(int(stable_id)) == stable_id:
            normalized_id = int(stable_id)
        else:
            raise ValueError("multi-parent source-face ancestry opt-in requires integer or canonical decimal stable lobe IDs")
        if normalized_id not in LOBE_IDS or normalized_id in multi_parent_rows:
            raise ValueError("multi-parent source-face ancestry opt-in contains an invalid or duplicate stable lobe ID")
        multi_parent_rows.add(normalized_id)
    if multi_parent_rows and not has_explicit_ancestry:
        raise ValueError("multi-parent source-face ancestry opt-in requires explicit source ancestry")
    raw_subdivisions = {} if allow_source_face_subdivisions is None else allow_source_face_subdivisions
    try:
        subdivision_items = list(raw_subdivisions.items())
    except (AttributeError, TypeError) as exc:
        raise ValueError("source-face subdivision opt-in must map stable lobe IDs to source face IDs") from exc
    subdivision_sources_by_row = {}
    for stable_id, raw_source_ids in subdivision_items:
        if isinstance(stable_id, (bool, np.bool_)):
            raise ValueError("source-face subdivision opt-in rejects boolean stable lobe IDs")
        if isinstance(stable_id, (int, np.integer)):
            normalized_id = int(stable_id)
        elif isinstance(stable_id, str) and stable_id.isdecimal() and str(int(stable_id)) == stable_id:
            normalized_id = int(stable_id)
        else:
            raise ValueError("source-face subdivision opt-in requires integer or canonical decimal stable lobe IDs")
        if normalized_id not in LOBE_IDS or normalized_id in subdivision_sources_by_row:
            raise ValueError("source-face subdivision opt-in contains an invalid or duplicate stable lobe ID")
        try:
            source_ids = list(raw_source_ids)
        except TypeError as exc:
            raise ValueError("source-face subdivision opt-in requires iterable source face IDs") from exc
        if not source_ids:
            raise ValueError("source-face subdivision opt-in cannot declare an empty source face set")
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) for value in source_ids):
            raise ValueError("source-face subdivision opt-in requires integer source face IDs")
        source_values = tuple(map(int, source_ids))
        if len(set(source_values)) != len(source_values):
            raise ValueError("source-face subdivision opt-in contains duplicate source face IDs")
        subdivision_sources_by_row[normalized_id] = source_values
    if subdivision_sources_by_row and not has_explicit_ancestry:
        raise ValueError("source-face subdivision opt-in requires explicit source ancestry")
    raw_patch_geometry = {} if source_face_subdivision_patch_geometry is None else source_face_subdivision_patch_geometry
    try:
        patch_geometry_items = list(raw_patch_geometry.items())
    except (AttributeError, TypeError) as exc:
        raise ValueError("source-face subdivision patch geometry must map stable lobe IDs to evidence") from exc
    patch_geometry_by_row = {}
    for stable_id, evidence in patch_geometry_items:
        if isinstance(stable_id, (bool, np.bool_)):
            raise ValueError("source-face subdivision patch geometry rejects boolean stable lobe IDs")
        if isinstance(stable_id, (int, np.integer)):
            normalized_id = int(stable_id)
        elif isinstance(stable_id, str) and stable_id.isdecimal() and str(int(stable_id)) == stable_id:
            normalized_id = int(stable_id)
        else:
            raise ValueError("source-face subdivision patch geometry requires integer or canonical decimal stable lobe IDs")
        if normalized_id not in LOBE_IDS or normalized_id in patch_geometry_by_row:
            raise ValueError("source-face subdivision patch geometry contains invalid or duplicate stable lobe IDs")
        patch_geometry_by_row[normalized_id] = evidence
    raw_parent_evidence = ({} if source_face_subdivision_parent_evidence is None
                           else source_face_subdivision_parent_evidence)
    try:
        parent_evidence_items = list(raw_parent_evidence.items())
    except (AttributeError, TypeError) as exc:
        raise ValueError("immediate subdivision parent evidence must map stable lobe IDs to pinned lineage") from exc
    parent_evidence_by_row = {}
    for stable_id, evidence in parent_evidence_items:
        if isinstance(stable_id, (bool, np.bool_)):
            raise ValueError("immediate subdivision parent evidence rejects boolean stable lobe IDs")
        if isinstance(stable_id, (int, np.integer)):
            normalized_id = int(stable_id)
        elif isinstance(stable_id, str) and stable_id.isdecimal() and str(int(stable_id)) == stable_id:
            normalized_id = int(stable_id)
        else:
            raise ValueError("immediate subdivision parent evidence requires integer or canonical decimal stable lobe IDs")
        if normalized_id not in LOBE_IDS or normalized_id in parent_evidence_by_row:
            raise ValueError("immediate subdivision parent evidence contains invalid or duplicate stable lobe IDs")
        parent_evidence_by_row[normalized_id] = evidence
    if parent_evidence_by_row and set(parent_evidence_by_row) != set(subdivision_sources_by_row):
        raise ValueError("immediate subdivision parent evidence must cover exactly the subdivided lobe rows")
    if parent_evidence_by_row and not has_explicit_ancestry:
        raise ValueError("immediate subdivision parent evidence requires explicit source lineage ancestry")
    raw_edge_collapse_evidence = ({} if source_face_edge_collapse_parent_evidence is None
                                  else source_face_edge_collapse_parent_evidence)
    try:
        edge_collapse_items = list(raw_edge_collapse_evidence.items())
    except (AttributeError, TypeError) as exc:
        raise ValueError("source-face edge-collapse evidence must map stable lobe IDs to pinned parent lineage") from exc
    edge_collapse_evidence_by_row = {}
    for stable_id, evidence in edge_collapse_items:
        if isinstance(stable_id, (bool, np.bool_)):
            raise ValueError("source-face edge-collapse evidence rejects boolean stable lobe IDs")
        if isinstance(stable_id, (int, np.integer)):
            normalized_id = int(stable_id)
        elif isinstance(stable_id, str) and stable_id.isdecimal() and str(int(stable_id)) == stable_id:
            normalized_id = int(stable_id)
        else:
            raise ValueError("source-face edge-collapse evidence requires integer or canonical decimal stable lobe IDs")
        if normalized_id not in LOBE_IDS or normalized_id in edge_collapse_evidence_by_row:
            raise ValueError("source-face edge-collapse evidence contains invalid or duplicate stable lobe IDs")
        edge_collapse_evidence_by_row[normalized_id] = evidence
    if edge_collapse_evidence_by_row and not has_explicit_ancestry:
        raise ValueError("source-face edge-collapse evidence requires explicit source lineage ancestry")
    if edge_collapse_evidence_by_row and source_lineage_lobes is None:
        raise ValueError("source-face edge-collapse evidence requires authenticated source lineage rows")
    if set(edge_collapse_evidence_by_row) & set(parent_evidence_by_row):
        raise ValueError("one candidate row cannot reuse subdivision and edge-collapse immediate-parent evidence")
    if source_lineage_lobes is not None:
        for stable_id, evidence in edge_collapse_evidence_by_row.items():
            source_faces = np.asarray(source_lineage_lobes[stable_id][1])
            if source_faces.dtype.kind not in "iu" or source_faces.ndim != 2 or source_faces.shape[1] != 3:
                raise ValueError(f"edge-collapse source lineage faces are malformed for row {stable_id}")
            repeated = _edge_collapse_repeated_source_face_ids(
                stable_id, evidence, len(source_faces),
            )
            if repeated:
                declared = subdivision_sources_by_row.get(stable_id)
                if declared is not None and tuple(sorted(declared)) != repeated:
                    raise ValueError("edge-collapse parent repeated source faces differ from the declared source subdivision")
                subdivision_sources_by_row[stable_id] = repeated
    if set(patch_geometry_by_row) != set(subdivision_sources_by_row):
        raise ValueError("every source-face subdivision opt-in requires matching exact patch geometry evidence")
    if has_explicit_ancestry and source_lineage_lobes is None:
        raise ValueError("explicit source pleura ancestry requires authenticated source lineage rows")
    if has_explicit_ancestry:
        normalized_maps = []
        for name, mapping in (("face", source_face_ancestry), ("vertex", source_vertex_ancestry)):
            try:
                items = list(mapping.items())
            except (AttributeError, TypeError) as exc:
                raise ValueError(f"explicit source pleura {name} ancestry must be keyed by stable lobe ID") from exc
            normalized = {}
            for stable_id, value in items:
                if isinstance(stable_id, (bool, np.bool_)):
                    raise ValueError(f"explicit source pleura {name} ancestry rejects boolean stable lobe IDs")
                if isinstance(stable_id, (int, np.integer)):
                    normalized_id = int(stable_id)
                elif isinstance(stable_id, str) and stable_id.isdecimal():
                    normalized_id = int(stable_id)
                    if str(normalized_id) != stable_id:
                        raise ValueError(f"explicit source pleura {name} ancestry requires canonical decimal stable lobe IDs")
                else:
                    raise ValueError(f"explicit source pleura {name} ancestry requires integer or canonical decimal stable lobe IDs")
                if normalized_id in normalized:
                    raise ValueError(f"explicit source pleura {name} ancestry contains duplicate normalized stable lobe IDs")
                normalized[normalized_id] = value
            if set(normalized) != set(LOBE_IDS):
                raise ValueError(f"explicit source pleura {name} ancestry requires all five unique stable lobe IDs")
            normalized_maps.append(normalized)
        source_face_ancestry, source_vertex_ancestry = normalized_maps
    lineage = None
    if source_lineage_lobes is not None:
        if set(map(int, source_lineage_lobes)) != set(LOBE_IDS):
            raise ValueError("source pleura lineage requires exactly the five stable lung lobe IDs")
        reference = {}
        candidate_face_ancestry = {}
        candidate_vertex_ancestry = {}
        face_ancestry_digest = hashlib.sha256()
        vertex_ancestry_digest = hashlib.sha256()
        multi_parent_face_counts = {}
        subdivision_patch_reports = {}
        edge_collapse_reports = {}
        for stable_id in LOBE_IDS:
            candidate_vertices6, candidate_faces = lobes[stable_id]
            source_vertices6, source_faces = source_lineage_lobes[stable_id]
            candidate_faces = np.asarray(candidate_faces)
            source_faces = np.asarray(source_faces)
            if has_explicit_ancestry and (
                    candidate_faces.dtype.kind not in "iu"
                    or source_faces.dtype.kind not in "iu"):
                raise ValueError(f"explicit source pleura lineage requires integer candidate and source face indices for row {stable_id}")
            if stable_id in edge_collapse_evidence_by_row:
                raw_candidate_vertices6 = np.asarray(candidate_vertices6)
                raw_source_vertices6 = np.asarray(source_vertices6)
                if (raw_candidate_vertices6.dtype.kind != "f" or raw_candidate_vertices6.dtype.itemsize != 4
                        or raw_source_vertices6.dtype.kind != "f" or raw_source_vertices6.dtype.itemsize != 4):
                    raise ValueError(f"edge-collapse candidate and source rows must preserve ABI5 Float32 vertex arrays for row {stable_id}")
            candidate_vertices6 = np.asarray(candidate_vertices6, dtype=np.float32)
            candidate_faces = candidate_faces.astype(np.int64, copy=False)
            source_vertices6 = np.asarray(source_vertices6, dtype=np.float32)
            source_faces = source_faces.astype(np.int64, copy=False)
            if (candidate_vertices6.ndim != 2 or candidate_vertices6.shape[1] != 6
                    or source_vertices6.ndim != 2 or source_vertices6.shape[1] != 6
                    or candidate_faces.ndim != 2 or candidate_faces.shape[1] != 3
                    or source_faces.ndim != 2 or source_faces.shape[1] != 3
                    or (len(candidate_faces) and (candidate_faces.min() < 0
                                                   or candidate_faces.max() >= len(candidate_vertices6)))
                    or (len(source_faces) and (source_faces.min() < 0
                                               or source_faces.max() >= len(source_vertices6)))):
                raise ValueError(f"malformed candidate or source pleura lineage arrays for row {stable_id}")
            if has_explicit_ancestry:
                raw_face_map = source_face_ancestry[stable_id]
                raw_vertex_map = source_vertex_ancestry[stable_id]
                try:
                    face_map_length = len(raw_face_map)
                    vertex_map_length = len(raw_vertex_map)
                except TypeError as exc:
                    raise ValueError(f"explicit source pleura ancestry must contain per-element rows for row {stable_id}") from exc
                if face_map_length != len(candidate_faces) or vertex_map_length != len(candidate_vertices6):
                    raise ValueError(f"explicit source pleura ancestry length differs for row {stable_id}")
                face_map = _normalize_source_face_ancestry(
                    raw_face_map, len(source_faces), stable_id,
                    allow_two_parent_flip=stable_id in multi_parent_rows,
                    allowed_subdivision_sources=subdivision_sources_by_row.get(stable_id, ()),
                )
                multi_parent_face_counts[stable_id] = sum(len(parents) > 1 for parents in face_map)
                vertex_map = []
                for output_vertex, parents in enumerate(raw_vertex_map):
                    values = np.asarray(parents)
                    if values.dtype.kind not in "iu":
                        raise ValueError(f"source vertex ancestry must use integer source IDs in row {stable_id}")
                    values = values.astype(np.int64, copy=False).reshape(-1)
                    if (len(values) == 0 or np.any(values < 0)
                            or np.any(values >= len(source_vertices6))
                            or len(np.unique(values)) != len(values)):
                        raise ValueError(f"source vertex ancestry is invalid for output vertex {output_vertex} in row {stable_id}")
                    vertex_map.append(tuple(map(int, values)))
                if stable_id in edge_collapse_evidence_by_row:
                    edge_collapse_reports[stable_id] = _validate_source_face_edge_collapse_parent_chain(
                        stable_id, candidate_faces, candidate_vertices6,
                        face_map, vertex_map, len(source_faces), len(source_vertices6),
                        edge_collapse_evidence_by_row[stable_id],
                    )
                if stable_id in subdivision_sources_by_row:
                    if stable_id in parent_evidence_by_row:
                        parent_chain = _validate_source_face_subdivision_parent_chain(
                            stable_id, candidate_faces, candidate_vertices6,
                            face_map, vertex_map, len(source_faces),
                            len(source_vertices6), parent_evidence_by_row[stable_id],
                            subdivision_sources_by_row[stable_id],
                            patch_geometry_by_row[stable_id].get("support_source_face_ids", ()),
                        )
                        immediate_patch_geometry = dict(patch_geometry_by_row[stable_id])
                        immediate_patch_geometry["support_source_face_ids"] = list(
                            parent_chain["immediate_patch_support_face_ids"])
                        patch_report = _source_face_subdivision_patch_report(
                            candidate_vertices6, candidate_faces,
                            parent_chain["parent_vertices6"], parent_chain["parent_faces"],
                            parent_chain["candidate_face_to_parent"], immediate_patch_geometry,
                            parent_chain["immediate_subdivision_face_ids"],
                        )
                        subdivision_patch_reports[stable_id] = {
                            "immediate_parent_chain": {
                                key: value for key, value in parent_chain.items()
                                if key not in ("parent_vertices6", "parent_faces", "candidate_face_to_parent")
                            },
                            "patch_geometry": patch_report,
                        }
                    else:
                        subdivision_patch_reports[stable_id] = _source_face_subdivision_patch_report(
                            candidate_vertices6, candidate_faces, source_vertices6, source_faces,
                            face_map, patch_geometry_by_row[stable_id],
                            subdivision_sources_by_row[stable_id],
                        )
                face_offsets = [0]
                face_values = []
                for parents in face_map:
                    face_values.extend(parents)
                    face_offsets.append(len(face_values))
                vertex_offsets = [0]
                vertex_values = []
                for parents in vertex_map:
                    vertex_values.extend(parents)
                    vertex_offsets.append(len(vertex_values))
                face_ancestry_digest.update(int(stable_id).to_bytes(4, "little", signed=False))
                face_ancestry_digest.update(np.asarray(face_offsets, dtype="<u8").tobytes())
                face_ancestry_digest.update(np.asarray(face_values, dtype="<u8").tobytes())
                vertex_ancestry_digest.update(int(stable_id).to_bytes(4, "little", signed=False))
                vertex_ancestry_digest.update(np.asarray(vertex_offsets, dtype="<u8").tobytes())
                vertex_ancestry_digest.update(np.asarray(vertex_values, dtype="<u8").tobytes())
            else:
                if (candidate_vertices6.shape != source_vertices6.shape
                        or candidate_faces.shape != source_faces.shape
                        or not np.array_equal(candidate_faces, source_faces)):
                    raise ValueError(
                        f"candidate lobe face/index inventory differs from source lineage for row {stable_id}"
                    )
                face_map = [(int(index),) for index in range(len(candidate_faces))]
                vertex_map = [(int(index),) for index in range(len(candidate_vertices6))]
            candidate_face_ancestry[stable_id] = tuple(face_map)
            candidate_vertex_ancestry[stable_id] = tuple(vertex_map)
            reference[stable_id] = (source_vertices6, source_faces)
        _, _, reference_report = derive_lung_union_exterior(
            reference, allow_source_microfragment=True,
            coordinate_quantization_m=coordinate_quantization_m,
        )
        reference_fragments = reference_report["excluded_source_microfragments"]
        if len(reference_fragments) != 1:
            raise ValueError("source lineage does not contain one unique declared microfragment")
        source_fragment = reference_fragments[0]
        source_pairs = tuple(sorted((int(owner[0]), int(owner[1]))
                                    for owner in source_fragment["source_faces"]))
        source_vertex_inventory = {}
        for stable_id in LOBE_IDS:
            face_ids = sorted(face_id for owner, face_id in source_pairs if owner == stable_id)
            if face_ids:
                source_vertex_inventory[stable_id] = sorted(set(map(
                    int, reference[stable_id][1][np.asarray(face_ids, dtype=np.int64)].reshape(-1)
                )))
        lineage = {
            "reference_lobes": reference,
            "candidate_face_ancestry": candidate_face_ancestry,
            "candidate_vertex_ancestry": candidate_vertex_ancestry,
            "multi_parent_face_counts": multi_parent_face_counts,
            "subdivided_source_face_ids": {
                stable_id: tuple(source_ids)
                for stable_id, source_ids in subdivision_sources_by_row.items()
            },
            "source_face_subdivision_patch_reports": subdivision_patch_reports,
            "source_face_edge_collapse_reports": edge_collapse_reports,
            "explicit_ancestry": has_explicit_ancestry,
            "face_ancestry_sha256": face_ancestry_digest.hexdigest() if has_explicit_ancestry else None,
            "vertex_ancestry_sha256": vertex_ancestry_digest.hexdigest() if has_explicit_ancestry else None,
            "source_pairs": source_pairs,
            "source_vertex_inventory": source_vertex_inventory,
            "source_face_count": int(source_fragment["face_count"]),
            "source_vertex_count": int(source_fragment["vertex_count"]),
            "source_surface_area_m2": float(source_fragment["surface_area_m2"]),
            "reference_report": reference_report,
        }

    face_groups = collections.defaultdict(list)
    lobe_positions = {}
    for stable_id in LOBE_IDS:
        vertices6, faces = lobes[stable_id]
        vertices6 = np.asarray(vertices6, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if (vertices6.ndim != 2 or vertices6.shape[1] != 6 or faces.ndim != 2
                or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0
                or faces.max() >= len(vertices6)):
            raise ValueError(f"malformed lobe geometry for stable ID {stable_id}")
        xyz = vertices6[:, :3]
        if not np.isfinite(vertices6).all():
            raise ValueError(f"non-finite lobe geometry for stable ID {stable_id}")
        lobe_positions[stable_id] = (vertices6, faces)
        for face_id, tri in enumerate(faces):
            coords = tuple(_coord_key(xyz[int(vertex)]) for vertex in tri)
            if len(set(coords)) != 3:
                raise ValueError(f"degenerate source face {(stable_id, face_id)}")
            face_groups[_face_key(coords)].append((stable_id, face_id, coords))

    shared_groups = []
    excluded = set()
    for key, group in face_groups.items():
        if len(group) == 1:
            continue
        if len(group) != 2 or group[0][0] == group[1][0]:
            raise ValueError("lung source has duplicate or >2-owner exact triangles")
        a, b = group
        if not _opposite_winding(a[2], b[2]):
            if _same_winding(a[2], b[2]):
                raise ValueError("reciprocal lobe interface has same-wound faces")
            raise ValueError("lobe interface triangles do not have reciprocal winding")
        pair = ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
        shared_groups.append((key, pair))
        excluded.update(pair)

    exterior_triangles = []
    exterior_owners = []
    for stable_id in LOBE_IDS:
        vertices6, faces = lobe_positions[stable_id]
        for face_id, tri in enumerate(faces):
            owner = (stable_id, face_id)
            if owner in excluded:
                continue
            exterior_triangles.append(tuple(_coord_key(vertices6[int(v), :3]) for v in tri))
            exterior_owners.append(owner)

    coordinate_to_vertex = {}
    vertex_rows = []
    exterior_faces = []
    for tri in exterior_triangles:
        mapped = []
        for xyz in tri:
            if xyz not in coordinate_to_vertex:
                coordinate_to_vertex[xyz] = len(vertex_rows)
                vertex_rows.append(xyz)
            mapped.append(coordinate_to_vertex[xyz])
        exterior_faces.append(mapped)
    vertices = np.asarray(vertex_rows, dtype=np.float32)
    faces = np.asarray(exterior_faces, dtype=np.int64)

    components, _ = _edge_components(faces)
    component_rows = [_component_detail(vertices, faces, exterior_owners, comp) for comp in components]
    omitted = []
    if len(components) > 2:
        candidates = []
        for comp, detail in zip(components, component_rows, strict=True):
            if lineage is None:
                matches = _is_known_source_microfragment(vertices, detail, coordinate_quantization_m)
            else:
                candidate_pairs = []
                candidate_face_by_source = {}
                unique_ancestry = True
                for owner, candidate_face_id in detail["_source_faces"]:
                    owner = int(owner)
                    candidate_face_id = int(candidate_face_id)
                    parents = lineage["candidate_face_ancestry"][owner][candidate_face_id]
                    if len(parents) != 1:
                        unique_ancestry = False
                        continue
                    source_pair = (owner, int(parents[0]))
                    if source_pair in candidate_face_by_source:
                        unique_ancestry = False
                    candidate_face_by_source[source_pair] = candidate_face_id
                    candidate_pairs.append(source_pair)
                candidate_pairs = tuple(sorted(candidate_pairs))
                candidate_inventory = {}
                unique_vertex_ancestry = True
                source_connectivity_matches = True
                for stable_id in LOBE_IDS:
                    component_face_ids = sorted(
                        int(face_id) for owner, face_id in detail["_source_faces"]
                        if int(owner) == stable_id
                    )
                    if not component_face_ids:
                        continue
                    candidate_faces_for_owner = lobe_positions[stable_id][1]
                    candidate_vertex_ids = sorted(set(map(
                        int, candidate_faces_for_owner[np.asarray(component_face_ids, dtype=np.int64)].reshape(-1)
                    )))
                    source_vertex_ids = []
                    for candidate_vertex_id in candidate_vertex_ids:
                        parents = lineage["candidate_vertex_ancestry"][stable_id][candidate_vertex_id]
                        if not parents:
                            unique_vertex_ancestry = False
                        source_vertex_ids.extend(map(int, parents))
                    candidate_inventory[stable_id] = sorted(set(source_vertex_ids))
                # Face IDs alone are insufficient if a stale ancestry table
                # is supplied after rewiring a triangle. Check the ordered
                # source vertices for every face in the omitted component;
                # cyclic rotations preserve winding, reversed order does not.
                for owner, candidate_face_id in detail["_source_faces"]:
                    owner = int(owner)
                    candidate_face_id = int(candidate_face_id)
                    parents = lineage["candidate_face_ancestry"][owner][candidate_face_id]
                    candidate_face = lobe_positions[owner][1][candidate_face_id]
                    candidate_vertex_parents = [
                        lineage["candidate_vertex_ancestry"][owner][int(vertex)]
                        for vertex in candidate_face
                    ]
                    if len(parents) != 1 or any(len(vertex_parents) != 1
                                                 for vertex_parents in candidate_vertex_parents):
                        source_connectivity_matches = False
                        continue
                    source_face = tuple(map(
                        int, lineage["reference_lobes"][owner][1][parents[0]]
                    ))
                    mapped_face = tuple(int(vertex_parents[0])
                                        for vertex_parents in candidate_vertex_parents)
                    source_cycles = (source_face, source_face[1:] + source_face[:1],
                                     source_face[2:] + source_face[:2])
                    if mapped_face not in source_cycles:
                        source_connectivity_matches = False
                matches = (unique_ancestry and unique_vertex_ancestry
                           and source_connectivity_matches
                           and candidate_pairs == lineage["source_pairs"]
                           and detail["face_count"] == lineage["source_face_count"]
                           and detail["vertex_count"] == lineage["source_vertex_count"]
                           and candidate_inventory == lineage["source_vertex_inventory"])
            if matches:
                candidates.append((comp, detail))
        if not allow_source_microfragment or len(components) != 3 or len(candidates) != 1:
            raise ValueError("detached lung-union component lacks the exact declared source lineage")
        orphan_component, orphan = candidates[0]
        orphan_topology = topology_report(faces[np.asarray(orphan_component, dtype=np.int64)])
        if orphan_topology["boundary_edge_count"] or orphan_topology["nonmanifold_edge_count"] or orphan_topology["orientation_error_edge_count"]:
            raise ValueError("source microfragment is not a closed oriented shell")
        lineage_report = None
        if lineage is not None:
            if not np.isfinite(orphan["signed_volume_m3"]) or abs(orphan["signed_volume_m3"]) > 1e-12:
                raise ValueError("source-lineage microfragment exceeds the candidate signed-volume bound")
            if not np.isfinite(orphan["surface_area_m2"]) or orphan["surface_area_m2"] <= 0:
                raise ValueError("source-lineage microfragment has zero or non-finite candidate area")
            face_area_ratios = []
            for stable_id, face_id in lineage["source_pairs"]:
                source_vertices6, source_faces = lineage["reference_lobes"][stable_id]
                candidate_vertices6, candidate_faces = lobe_positions[stable_id]
                candidate_face_id = next(
                    int(candidate_id) for candidate_owner, candidate_id in orphan["_source_faces"]
                    if int(candidate_owner) == stable_id
                    and lineage["candidate_face_ancestry"][stable_id][int(candidate_id)] == (face_id,)
                )
                source_tri = source_vertices6[source_faces[face_id], :3].astype(np.float64)
                candidate_tri = candidate_vertices6[candidate_faces[candidate_face_id], :3].astype(np.float64)
                source_cross = np.cross(source_tri[1] - source_tri[0], source_tri[2] - source_tri[0])
                candidate_cross = np.cross(candidate_tri[1] - candidate_tri[0], candidate_tri[2] - candidate_tri[0])
                source_area2 = float(np.linalg.norm(source_cross))
                candidate_area2 = float(np.linalg.norm(candidate_cross))
                if not np.isfinite(source_area2) or not np.isfinite(candidate_area2) or source_area2 <= 0 or candidate_area2 <= 0:
                    raise ValueError("source-lineage microfragment contains a zero-area candidate face")
                face_area_ratios.append(candidate_area2 / source_area2)
            if not face_area_ratios or min(face_area_ratios) <= 0.05:
                raise ValueError("source-lineage microfragment candidate face area falls below the 5 percent source bound")
            reference_row_hashes = {}
            for stable_id in LOBE_IDS:
                reference_vertices6, reference_faces = lineage["reference_lobes"][stable_id]
                normalized_vertices6 = np.ascontiguousarray(reference_vertices6, dtype="<f4")
                normalized_faces = np.ascontiguousarray(reference_faces, dtype="<i8")
                reference_row_hashes[str(stable_id)] = {
                    "vertex_count": int(len(normalized_vertices6)),
                    "face_count": int(len(normalized_faces)),
                    "vertices6_f32_sha256": hashlib.sha256(normalized_vertices6.tobytes()).hexdigest(),
                    "faces_i64_sha256": hashlib.sha256(normalized_faces.tobytes()).hexdigest(),
                }
            lineage_report = {
                "recognition_mode": ("exact_source_face_and_vertex_ancestry" if lineage["explicit_ancestry"]
                                     else "exact_source_face_and_vertex_inventory"),
                "explicit_candidate_face_ancestry": lineage["explicit_ancestry"],
                "candidate_face_ancestry_sha256": lineage["face_ancestry_sha256"],
                "candidate_vertex_ancestry_sha256": lineage["vertex_ancestry_sha256"],
                "multi_parent_candidate_face_count_by_stable_id": {
                    str(sid): int(count) for sid, count in lineage["multi_parent_face_counts"].items()
                },
                "subdivided_source_face_ids_by_stable_id": {
                    str(sid): list(map(int, ids))
                    for sid, ids in lineage["subdivided_source_face_ids"].items()
                },
                "source_face_subdivision_patch_reports_by_stable_id": {
                    str(sid): report for sid, report in lineage["source_face_subdivision_patch_reports"].items()
                },
                "source_face_connectivity_verified": True,
                "source_face_connectivity_mode": (
                    "explicit_ordered_source_vertex_ancestry" if lineage["explicit_ancestry"]
                    else "identical_candidate_and_source_face_arrays"
                ),
                "reference_lobe_row_arrays": reference_row_hashes,
                "source_faces": [list(pair) for pair in lineage["source_pairs"]],
                "source_vertex_indices_by_stable_id": {
                    str(sid): values for sid, values in lineage["source_vertex_inventory"].items()
                },
                "reference_surface_area_m2": lineage["source_surface_area_m2"],
                "candidate_surface_area_m2": orphan["surface_area_m2"],
                "candidate_to_reference_surface_area_ratio": (
                    orphan["surface_area_m2"] / lineage["source_surface_area_m2"]
                ),
                "minimum_candidate_to_reference_face_area_ratio": float(min(face_area_ratios)),
                "maximum_candidate_to_reference_face_area_ratio": float(max(face_area_ratios)),
                "candidate_signed_volume_m3": orphan["signed_volume_m3"],
                "candidate_topology": {
                    "boundary_edge_count": orphan_topology["boundary_edge_count"],
                    "nonmanifold_edge_count": orphan_topology["nonmanifold_edge_count"],
                    "orientation_error_edge_count": orphan_topology["orientation_error_edge_count"],
                },
            }
        omit_ids = set(orphan["_face_ids"])
        omitted_row = {
            "source_faces": orphan["_source_faces"],
            "face_count": orphan["face_count"],
            "vertex_count": orphan["vertex_count"],
            "signed_volume_m3": orphan["signed_volume_m3"],
            "surface_area_m2": orphan["surface_area_m2"],
            "bounds_min_m": orphan["bounds_min_m"],
            "bounds_max_m": orphan["bounds_max_m"],
        }
        if lineage_report is not None:
            omitted_row["source_lineage"] = lineage_report
        omitted = [omitted_row]
        keep = [i for i in range(len(faces)) if i not in omit_ids]
        faces = faces[np.asarray(keep, dtype=np.int64)]
        exterior_owners = [exterior_owners[i] for i in keep]
        # Compact exact source-coordinate vertices after dropping the closed
        # source fragment; no new boundary is introduced by this operation.
        used = sorted(set(map(int, faces.reshape(-1))))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[np.asarray(used, dtype=np.int64)] = np.arange(len(used), dtype=np.int64)
        vertices = vertices[np.asarray(used, dtype=np.int64)]
        faces = remap[faces]
        components, _ = _edge_components(faces)
    if len(components) != 2:
        raise ValueError(f"external lung union must have two connected components, found {len(components)}")

    component_details = []
    for component in components:
        detail = _component_detail(vertices, faces, exterior_owners, component)
        local_face_ids = np.asarray(component, dtype=np.int64)
        component_faces = faces[local_face_ids]
        topo = topology_report(component_faces)
        if topo["boundary_edge_count"] or topo["nonmanifold_edge_count"] or topo["orientation_error_edge_count"]:
            raise ValueError("external lung union component is not closed and oriented")
        if detail["signed_volume_m3"] <= 0:
            raise ValueError("external lung union source winding is not outward")
        detail["topology"] = {
            "boundary_edge_count": topo["boundary_edge_count"],
            "nonmanifold_edge_count": topo["nonmanifold_edge_count"],
            "orientation_error_edge_count": topo["orientation_error_edge_count"],
        }
        component_details.append({key: value for key, value in detail.items() if not key.startswith("_")})

    face_areas = np.linalg.norm(
        np.cross(vertices[faces[:, 1]].astype(np.float64) - vertices[faces[:, 0]],
                 vertices[faces[:, 2]].astype(np.float64) - vertices[faces[:, 0]]),
        axis=1,
    ) * 0.5
    if not np.isfinite(face_areas).all() or np.any(face_areas <= 0):
        raise ValueError("external lung union contains zero-area triangles")
    vertex_normals = normals(vertices.astype(np.float64), faces).astype(np.float32)
    vertices6 = np.column_stack((vertices, vertex_normals)).astype(np.float32)
    detail = {
        "stable_source_lobe_ids": list(LOBE_IDS),
        "exact_reciprocal_face_group_count": int(len(shared_groups)),
        "cancelled_opposite_face_count": int(2 * len(shared_groups)),
        "retained_external_triangle_count": int(len(faces)),
        "retained_external_vertex_count": int(len(vertices)),
        "retained_external_surface_area_m2": float(face_areas.sum()),
        "retained_external_signed_volume_m3": float(sum(x["signed_volume_m3"] for x in component_details)),
        "components": component_details,
        "excluded_source_microfragments": omitted,
        "source_coordinate_quantization_m": coordinate_quantization_m,
        "source_microfragment_anchor_rule": (
            "exact source-face and source-vertex ancestry inherited from caller-authenticated ancestral lobe rows; unrelated face/index inventory changes require explicit ancestry maps"
            if lineage is not None and lineage["explicit_ancestry"] else
            "exact per-lobe face/index arrays and source-face/vertex inventory inherited from caller-authenticated ancestral lobe rows"
            if lineage is not None else
            "original source anchors rounded to the receipt-declared coordinate quantum"
            if coordinate_quantization_m is not None else "exact registered float32 source anchors"
        ),
        "position_coordinates_are_exact_source_float32": True,
        "normal_field": "recomputed area-weighted exterior normals from retained oriented source triangles",
        "interlobar_fissure_lining_included": False,
        "parietal_pleura_included": False,
        "pleural_fluid_or_mechanics": False,
    }
    if lineage is not None and lineage["source_face_edge_collapse_reports"]:
        detail["source_face_edge_collapse_reports_by_stable_id"] = {
            str(stable_id): report
            for stable_id, report in sorted(lineage["source_face_edge_collapse_reports"].items())
        }
    return vertices6, faces, detail


def _parse_payload(raw: bytes):
    if len(raw) < HEADER.size:
        raise ValueError("truncated NHANAT payload")
    header = HEADER.unpack_from(raw)
    magic, abi, surface_count, vertex_count, index_count, registration_fp, source_sha = header
    if magic != b"NHANAT1\0" or abi != 5:
        raise ValueError("expected the existing NHANAT1 ABI5 payload")
    expected_length = HEADER.size + surface_count * RECORD.size + vertex_count * 24 + index_count * 4
    if len(raw) != expected_length:
        raise ValueError("NHANAT byte length does not match its header")
    records = [RECORD.unpack_from(raw, HEADER.size + i * RECORD.size) for i in range(surface_count)]
    vertex_offset = HEADER.size + surface_count * RECORD.size
    index_offset = vertex_offset + vertex_count * 24
    all_vertices = np.frombuffer(raw, dtype="<f4", count=vertex_count * 6, offset=vertex_offset).reshape(-1, 6)
    all_indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=index_offset)
    rows = {}
    for record in records:
        body, vertex_start, vertex_count_i, index_start, index_count_i, stable_id, layer, flags = map(int, record)
        if stable_id in rows:
            raise ValueError("duplicate anatomy stable ID")
        if vertex_start + vertex_count_i > vertex_count or index_start + index_count_i > index_count or index_count_i % 3:
            raise ValueError(f"anatomy record {stable_id} is out of bounds")
        global_indices = all_indices[index_start:index_start + index_count_i].reshape(-1, 3)
        if global_indices.size and (int(global_indices.min()) < vertex_start or int(global_indices.max()) >= vertex_start + vertex_count_i):
            raise ValueError(f"anatomy record {stable_id} has a nonlocal triangle index")
        rows[stable_id] = {
            "record": tuple(map(int, record)),
            "body_index": body,
            "layer": layer,
            "flags": flags,
            "vertices6": all_vertices[vertex_start:vertex_start + vertex_count_i].copy(),
            "faces": (global_indices.astype(np.int64) - vertex_start),
        }
    if len(rows) != surface_count:
        raise ValueError("anatomy record count mismatch")
    return header, records, rows


def _record_content_bytes(row):
    return (
        np.asarray(row["vertices6"], dtype="<f4").tobytes()
        + np.asarray(row["faces"], dtype="<i8").tobytes()
        + struct.pack("<3I", int(row["body_index"]), int(row["layer"]), int(row["flags"]))
    )


def _replace_geometry(raw: bytes, header, records, rows, vertices6, faces):
    surface_count = int(header[2])
    original_order = [int(record[5]) for record in records]
    if PLEURA_ID not in rows or any(sid not in rows for sid in LOBE_IDS):
        raise ValueError("NHANAT payload lacks a required lobe or pleura identity")
    previous = rows[PLEURA_ID]
    if int(previous["body_index"]) != 20 or int(previous["layer"]) != 8:
        raise ValueError("stable ID 310 is not the registered body-20 pleura layer")
    updated = {sid: dict(row) for sid, row in rows.items()}
    updated[PLEURA_ID]["vertices6"] = np.asarray(vertices6, dtype=np.float32)
    updated[PLEURA_ID]["faces"] = np.asarray(faces, dtype=np.int64)
    payload_records = []
    vertex_parts = []
    index_parts = []
    nv = 0
    ni = 0
    for sid in original_order:
        row = updated[sid]
        v = np.ascontiguousarray(row["vertices6"], dtype="<f4")
        f = np.asarray(row["faces"], dtype=np.int64)
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
            raise ValueError(f"invalid surface geometry for stable ID {sid}")
        rec = row["record"]
        payload_records.append(RECORD.pack(
            int(row["body_index"]), nv, len(v), ni, f.size,
            sid, int(row["layer"]), int(row["flags"]),
        ))
        vertex_parts.append(v.tobytes())
        index_parts.append((f.reshape(-1) + nv).astype("<u4").tobytes())
        row["vertices6"] = v
        row["faces"] = f
        nv += len(v)
        ni += f.size
    new_header = HEADER.pack(header[0], header[1], surface_count, nv, ni, header[5], header[6])
    out = new_header + b"".join(payload_records) + b"".join(vertex_parts) + b"".join(index_parts)
    _, _, out_rows = _parse_payload(out)
    for sid in original_order:
        if sid == PLEURA_ID:
            continue
        if _record_content_bytes(rows[sid]) != _record_content_bytes(out_rows[sid]):
            raise ValueError(f"replacement changed source geometry bytes outside stable ID {PLEURA_ID}: {sid}")
    return out, out_rows


def build_candidate(base_payload: Path, base_receipt: Path, output_dir: Path):
    base_raw = base_payload.read_bytes()
    receipt_raw = base_receipt.read_bytes()
    base_sha = hashlib.sha256(base_raw).hexdigest()
    receipt = json.loads(receipt_raw)
    if receipt.get("payload", {}).get("sha256") != base_sha:
        raise ValueError("source receipt does not bind the exact input NHANAT payload")
    if receipt.get("schema") != "numi.human.resting-anatomy-receipt.v1":
        raise ValueError("input is not the existing resting anatomy receipt")
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    pleura_source = source_map.get(str(PLEURA_ID), {})
    if (pleura_source.get("name") != "Pleura" or pleura_source.get("source_member") != "Pleura"
            or pleura_source.get("source_sha256") != receipt.get("provenance", {}).get("zanatomy_export_sha256")):
        raise ValueError("stable ID 310 does not preserve the expected generic Z-Anatomy pleura provenance")
    if receipt.get("provenance", {}).get("zanatomy_license") != "CC-BY-SA-4.0":
        raise ValueError("generic pleura source license/provenance is missing")

    header, records, rows = _parse_payload(base_raw)
    for sid in LOBE_IDS:
        row = rows.get(sid)
        if row is None or int(row["body_index"]) != 20:
            raise ValueError(f"registered lobe {sid} is missing or not in torso body frame")
    lobes = {sid: (rows[sid]["vertices6"], rows[sid]["faces"]) for sid in LOBE_IDS}
    cell_registration = receipt.get("provenance", {}).get("conforming_respiratory_cells", {})
    coordinate_quantization_m = cell_registration.get("coordinate_resolution_m")
    proxy_v, proxy_f, derivation = derive_lung_union_exterior(
        lobes, coordinate_quantization_m=coordinate_quantization_m,
    )
    out_raw, out_rows = _replace_geometry(base_raw, header, records, rows, proxy_v, proxy_f)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "resting-thorax.nhanatomy"
    receipt_path = output_dir / "resting-anatomy-receipt.json"
    payload_path.write_bytes(out_raw)
    output_sha = hashlib.sha256(out_raw).hexdigest()
    output_receipt = json.loads(json.dumps(receipt))
    output_receipt["payload"].update({
        "path": str(payload_path),
        "input_payload_sha256": base_sha,
        "sha256": output_sha,
        "vertex_count": int(HEADER.unpack_from(out_raw)[3]),
        "index_count": int(HEADER.unpack_from(out_raw)[4]),
    })
    output_receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    output_receipt["functional_bindings"]["pleura_representation"] = {
        "stable_id": PLEURA_ID,
        "name": "inferred passive visceral pleura proxy",
        "scope": "external lung-lobe union surface only",
        "interlobar_fissure_lining_included": False,
        "parietal_pleura_included": False,
        "pleural_fluid_or_mechanics": False,
        "surface_is_coincident_with_registered_lung_envelope": True,
        "parameter_status": "derived source geometry; not a measured subject-specific pleural surface",
    }
    # These identities are content-addressed results of the cardiac geometry
    # work. The replacement has already byte-verified every stable ID other
    # than 310; update only fields explicitly describing its output identity.
    def update_output_identity(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "output_anatomy_payload_sha256" and child == base_sha:
                    value[key] = output_sha
                else:
                    update_output_identity(child)
        elif isinstance(value, list):
            for child in value:
                update_output_identity(child)
    update_output_identity(output_receipt.get("provenance", {}))
    output_receipt["provenance"]["derived_visceral_pleura_proxy"] = {
        "algorithm": "exact_float32_registered_lobe_union_boundary_v1",
        "source_payload_path": str(base_payload),
        "source_payload_sha256": base_sha,
        "source_receipt_path": str(base_receipt),
        "source_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "output_payload_path": str(payload_path),
        "output_payload_sha256": output_sha,
        "source_generic_pleura": {
            "stable_id": PLEURA_ID,
            "provider": pleura_source.get("provider"),
            "source_member": pleura_source.get("source_member"),
            "source_member_sha256": pleura_source.get("source_sha256"),
            "source_export_sha256": receipt.get("provenance", {}).get("zanatomy_export_sha256"),
            "source_license": receipt.get("provenance", {}).get("zanatomy_license"),
            "original_native_record_and_geometry_sha256": hashlib.sha256(_record_content_bytes(rows[PLEURA_ID])).hexdigest(),
            "original_native_vertex_count": int(len(rows[PLEURA_ID]["vertices6"])),
            "original_native_triangle_count": int(len(rows[PLEURA_ID]["faces"])),
        },
        "derived_source_lobes": [
            {"stable_id": sid, "name": source_map[str(sid)]["name"],
             "source_member": source_map[str(sid)]["source_member"],
             "source_sha256": source_map[str(sid)]["source_sha256"]}
            for sid in LOBE_IDS
        ],
        "geometry_derivation": derivation,
        "source_coordinate_quantization_m": coordinate_quantization_m,
        "exact_native_float32_topology_checked": True,
        "self_intersection_audit": "not_assessed_by_this_source-preparation_step",
        "mechanics_or_physiology": False,
        "source_lobe_geometry_changed": False,
    }
    output_receipt["qualification"]["pleura_proxy"] = (
        "derived external lung union surface replaces generic Pleura geometry at stable ID 310; "
        "no parietal layer, interlobar fissure lining, pleural fluid, or pleural mechanics; "
        "self-intersection and native rendering qualification are separate gates"
    )
    receipt_path.write_text(json.dumps(output_receipt, indent=2, sort_keys=True) + "\n")
    return {
        "input_payload_sha256": base_sha,
        "input_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "output_payload_path": str(payload_path),
        "output_payload_sha256": output_sha,
        "output_receipt_path": str(receipt_path),
        "output_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "stable_id_310_vertex_count": int(len(out_rows[PLEURA_ID]["vertices6"])),
        "stable_id_310_triangle_count": int(len(out_rows[PLEURA_ID]["faces"])),
        "changed_stable_id": PLEURA_ID,
        "other_surface_geometry_byte_identity": True,
        "derivation": derivation,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-payload", required=True, type=Path)
    parser.add_argument("--base-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_candidate(args.base_payload, args.base_receipt, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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


def _normalize_source_face_ancestry(raw_face_map, source_face_count, stable_id, *, allow_two_parent_flip):
    """Validate exact source-face ancestry for unchanged faces and one local edge flip."""
    face_map = []
    single_parent_sources = set()
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
            if parent_ids[0] in single_parent_sources:
                raise ValueError(f"source face ancestry must map each output face to one unique source face in row {stable_id}")
            single_parent_sources.add(parent_ids[0])
        else:
            if not allow_two_parent_flip:
                raise ValueError(f"source face ancestry must map each output face to one unique source face in row {stable_id}")
            paired_flip_uses[parent_ids] += 1
        face_map.append(parent_ids)
    if any(count != 2 for count in paired_flip_uses.values()):
        raise ValueError(f"a local 2-to-2 flip must map exactly two output faces to the same source-face pair in row {stable_id}")
    paired_sources = [source for pair in paired_flip_uses for source in pair]
    if (len(set(paired_sources)) != len(paired_sources)
            or set(paired_sources) & single_parent_sources):
        raise ValueError(f"a flipped source-face pair must be retired only by its two mapped output faces in row {stable_id}")
    return tuple(face_map)


def derive_lung_union_exterior(lobes, *, allow_source_microfragment=True,
                               coordinate_quantization_m=None, source_lineage_lobes=None,
                               source_face_ancestry=None, source_vertex_ancestry=None,
                               allow_multi_parent_source_face_ancestry=()):
    """Build the external union surface from five registered lobe meshes.

    `lobes` maps stable IDs 305..309 to `(vertices6, local_faces)` arrays.
    Exact reciprocal triangles are canceled from the exterior. Explicit
    multi-parent face ancestry is disabled by default; an authenticated caller
    may opt named rows into one exact local 2-to-2 flip pair. By default,
    any detached component is a hard error.  A source-specific microfragment
    may be omitted only when it matches the exact registered 305/308 source
    anchor coordinates, sub-millimeter-scale bounds, owner identities, and
    bounded area/volume recorded below. With explicit ancestry from an
    authenticated caller, unrelated candidate inventory changes are allowed
    while the omitted component must map to the same source faces, ordered
    source connectivity, and source vertex inventory. Any other detached
    component fails.
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
        for stable_id in LOBE_IDS:
            candidate_vertices6, candidate_faces = lobes[stable_id]
            source_vertices6, source_faces = source_lineage_lobes[stable_id]
            candidate_faces = np.asarray(candidate_faces)
            source_faces = np.asarray(source_faces)
            if has_explicit_ancestry and (
                    candidate_faces.dtype.kind not in "iu"
                    or source_faces.dtype.kind not in "iu"):
                raise ValueError(f"explicit source pleura lineage requires integer candidate and source face indices for row {stable_id}")
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

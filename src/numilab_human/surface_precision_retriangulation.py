"""Local, position-preserving 2-to-2 flips for source precision preparation.

A caller should run :func:`validate_closed_oriented_surface` once before a
batch, build :class:`SurfaceAdjacency`, choose non-overlapping eligible edges,
apply the returned face pairs, and validate the completed batch once more.
This module does not choose anatomy, move vertices, set runtime tolerances, or
claim that a changed PL triangulation remains conforming to an unrelated field.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

Edge = tuple[int, int]


@dataclass(frozen=True)
class SurfaceAdjacency:
    """Incidence built from one exact face array; rebuild after overlapping edits."""

    edge_faces: Mapping[Edge, tuple[int, ...]]
    vertex_neighbors: Mapping[int, frozenset[int]]


def _face_array(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("faces must be a non-empty Mx3 array")
    if not np.issubdtype(f.dtype, np.integer):
        raise ValueError("face indices must have an integer dtype")
    if np.issubdtype(f.dtype, np.signedinteger) and bool((f < 0).any()):
        raise ValueError("face indices must be non-negative")
    if int(f.max()) > np.iinfo(np.int64).max:
        raise ValueError("face index exceeds signed 64-bit range")
    f = f.astype(np.int64, copy=False)
    if bool(((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])).any()):
        raise ValueError("a triangle repeats a vertex index")
    return f


def _positions(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v):
        raise ValueError("vertices must be a non-empty Nx3 position array")
    if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.complexfloating):
        raise ValueError("vertex positions must be real numbers")
    if not np.isfinite(v).all():
        raise ValueError("vertex positions must be finite")
    return v.astype(np.float64, copy=False)


def _check_face_bounds(vertices: np.ndarray, faces: np.ndarray) -> None:
    if int(faces.max()) >= len(vertices):
        raise ValueError("face index is outside the vertex array")


def _edges(triangle: Sequence[int]) -> tuple[tuple[Edge, int], ...]:
    a, b, c = map(int, triangle)
    out = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = (min(start, end), max(start, end))
        out.append((edge, 1 if (start, end) == edge else -1))
    return tuple(out)


def build_surface_adjacency(faces: np.ndarray) -> SurfaceAdjacency:
    """Build edge-face and vertex-neighbor maps in one pass over ``faces``."""
    f = _face_array(faces)
    edge_rows: dict[Edge, list[int]] = defaultdict(list)
    neighbors: dict[int, set[int]] = defaultdict(set)
    seen_faces: set[tuple[int, int, int]] = set()
    for face_index, triangle in enumerate(f):
        canonical = tuple(sorted(map(int, triangle)))
        if canonical in seen_faces:
            raise ValueError("faces contain a duplicate unoriented triangle")
        seen_faces.add(canonical)
        a, b, c = map(int, triangle)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
        for edge, _ in _edges(triangle):
            edge_rows[edge].append(face_index)
    return SurfaceAdjacency(
        edge_faces={edge: tuple(rows) for edge, rows in edge_rows.items()},
        vertex_neighbors={vertex: frozenset(row) for vertex, row in neighbors.items()},
    )


def _triangle_metrics(points: np.ndarray) -> tuple[float, float, np.ndarray]:
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    cross_norm = float(np.linalg.norm(cross))
    edge_lengths = (
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[0] - points[2])),
    )
    longest = max(edge_lengths)
    if not math.isfinite(cross_norm) or cross_norm <= 0.0 or not math.isfinite(longest) or longest <= 0.0:
        return 0.0, 0.0, cross
    return 0.5 * cross_norm, cross_norm / longest, cross


def _directed_edge_and_opposite(triangle: Sequence[int], edge: Edge) -> tuple[int, int, int]:
    a, b, c = map(int, triangle)
    for start, end, opposite in ((a, b, c), (b, c, a), (c, a, b)):
        if {start, end} == set(edge):
            return start, end, opposite
    raise ValueError("selected face does not contain the selected edge")


def _face_origin_values(face_origins: Sequence[int] | np.ndarray | None,
                        face_indices: tuple[int, int]) -> list[int] | None:
    if face_origins is None:
        return None
    origins = np.asarray(face_origins)
    if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("face_origins must be a one-dimensional integer array")
    if int(origins.max(initial=-1)) > np.iinfo(np.int64).max:
        raise ValueError("face origin exceeds signed 64-bit range")
    if max(face_indices) >= len(origins):
        raise ValueError("face_origins is shorter than the selected face array")
    return [int(origins[index]) for index in face_indices]


def flip_interior_edge(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    edge: Sequence[int],
    face_indices: Sequence[int],
    minimum_altitude_m: float,
    adjacency: SurfaceAdjacency,
    face_origins: Sequence[int] | np.ndarray | None = None,
    in_place: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace one interior diagonal while retaining all source vertices.

    ``adjacency`` must describe ``faces`` at the selected local neighborhood.
    For a batch, use vertex-disjoint flips against one adjacency snapshot or
    rebuild/update the adjacency after applying an overlapping flip. Set
    ``in_place=True`` only for a caller-owned candidate face array; all checks
    finish before its two selected rows are written. The default returns a
    copied face array. Validate the full source and final surface once per
    batch with :func:`validate_closed_oriented_surface`.

    Both output faces retain their input row slots. Each is explicitly mapped
    to the exact two source parent face indices in the report; no one-to-one
    source face origin is implied for the retriangulated cells.
    """
    v = _positions(vertices)
    f = _face_array(faces)
    _check_face_bounds(v, f)
    if not isinstance(adjacency, SurfaceAdjacency):
        raise ValueError("adjacency must come from build_surface_adjacency")
    if len(edge) != 2:
        raise ValueError("edge must contain exactly two endpoint indices")
    raw_u, raw_v = int(edge[0]), int(edge[1])
    if raw_u == raw_v or min(raw_u, raw_v) < 0 or max(raw_u, raw_v) >= len(v):
        raise ValueError("edge endpoints are invalid")
    selected_edge = (min(raw_u, raw_v), max(raw_u, raw_v))
    if len(face_indices) != 2:
        raise ValueError("an interior flip must select exactly two source face indices")
    i0, i1 = sorted(map(int, face_indices))
    if i0 == i1 or min(i0, i1) < 0 or i1 >= len(f):
        raise ValueError("selected face indices are invalid")
    if not math.isfinite(float(minimum_altitude_m)) or float(minimum_altitude_m) <= 0.0:
        raise ValueError("minimum_altitude_m must be finite and positive")

    if adjacency.edge_faces.get(selected_edge) != (i0, i1):
        raise ValueError("selected edge must have exactly the two declared incident faces")
    first = f[i0]
    second = f[i1]
    edge_start, edge_end, opposite_a = _directed_edge_and_opposite(first, selected_edge)
    second_start, second_end, opposite_b = _directed_edge_and_opposite(second, selected_edge)
    if (second_start, second_end) != (edge_end, edge_start):
        raise ValueError("selected faces do not orient the shared edge oppositely")
    if opposite_a == opposite_b or len({edge_start, edge_end, opposite_a, opposite_b}) != 4:
        raise ValueError("selected edge neighborhood is not a four-vertex quadrilateral")
    if adjacency.edge_faces.get(tuple(sorted((opposite_a, opposite_b)))) is not None:
        raise ValueError("proposed replacement diagonal already exists")
    common_neighbors = (
        adjacency.vertex_neighbors.get(edge_start, frozenset())
        & adjacency.vertex_neighbors.get(edge_end, frozenset())
    )
    if not frozenset((opposite_a, opposite_b)).issubset(common_neighbors):
        raise ValueError("adjacency omits an opposite vertex from the selected edge link")
    # Extra common neighbors can be non-facial three-cycles in a valid
    # triangulated surface. They are an edge-collapse concern, not a reason to
    # reject a 2-to-2 flip when the replacement diagonal is absent.
    extra_common_neighbors = sorted(common_neighbors - {opposite_a, opposite_b})

    old_pair = (first.copy(), second.copy())
    new_first = np.asarray((opposite_a, edge_start, opposite_b), dtype=f.dtype)
    new_second = np.asarray((opposite_b, edge_end, opposite_a), dtype=f.dtype)
    old_metrics = [_triangle_metrics(v[triangle]) for triangle in old_pair]
    new_metrics = [_triangle_metrics(v[triangle]) for triangle in (new_first, new_second)]
    if any(area <= 0.0 for area, _, _ in old_metrics):
        raise ValueError("selected source faces must have positive exact Float64 area")
    if any(area <= 0.0 for area, _, _ in new_metrics):
        raise ValueError("replacement diagonal creates a zero-area face")

    reference_normal = old_metrics[0][2] + old_metrics[1][2]
    reference_norm = float(np.linalg.norm(reference_normal))
    if not math.isfinite(reference_norm) or reference_norm <= 0.0:
        raise ValueError("source face pair has no stable oriented area reference")
    alignment_cosines = []
    for _, _, normal in (*old_metrics, *new_metrics):
        cosine = float(np.dot(normal, reference_normal) / (float(np.linalg.norm(normal)) * reference_norm))
        alignment_cosines.append(cosine)
    if any(not math.isfinite(value) or value <= 0.0 for value in alignment_cosines):
        raise ValueError("source or replacement faces fold across the local oriented reference")

    old_min_altitude = min(metric[1] for metric in old_metrics)
    new_min_altitude = min(metric[1] for metric in new_metrics)
    if new_min_altitude <= old_min_altitude:
        raise ValueError("replacement does not improve the selected pair minimum altitude")
    if new_min_altitude < float(minimum_altitude_m):
        raise ValueError("replacement pair is below the caller's minimum altitude bound")

    if in_place:
        if not f.flags.writeable:
            raise ValueError("in_place=True requires a writable candidate face array")
        output = f
    else:
        output = f.copy()
    old_origins = _face_origin_values(face_origins, (i0, i1))
    parent_pair: dict[str, object] = {"source_parent_face_indices": [i0, i1]}
    if old_origins is not None:
        parent_pair["source_parent_face_origins"] = old_origins

    changed_edges = set()
    for triangle in (*old_pair, new_first, new_second):
        changed_edges.update(edge_key for edge_key, _ in _edges(triangle))
    edge_incidence_updates = []
    for changed_edge in sorted(changed_edges):
        before = tuple(adjacency.edge_faces.get(changed_edge, ()))
        after_set = set(before) - {i0, i1}
        for face_index, triangle in ((i0, new_first), (i1, new_second)):
            if changed_edge in {edge_key for edge_key, _ in _edges(triangle)}:
                after_set.add(face_index)
        edge_incidence_updates.append({
            "edge": list(changed_edge),
            "before_face_indices": list(before),
            "after_face_indices": sorted(after_set),
        })

    output[i0] = new_first
    output[i1] = new_second
    old_area_vector = 0.5 * (old_metrics[0][2] + old_metrics[1][2])
    new_area_vector = 0.5 * (new_metrics[0][2] + new_metrics[1][2])
    old_volume = math.fsum(
        float(np.dot(v[t[0]], np.cross(v[t[1]], v[t[2]]))) / 6.0 for t in old_pair
    )
    new_volume = math.fsum(
        float(np.dot(v[t[0]], np.cross(v[t[1]], v[t[2]]))) / 6.0 for t in (new_first, new_second)
    )
    report: dict[str, object] = {
        "operation": "position-preserving interior 2-to-2 edge flip",
        "replacement_edge": [opposite_a, opposite_b],
        "selected_edge_directed_in_first_parent": [edge_start, edge_end],
        "extra_common_neighbors_not_used_by_selected_faces": extra_common_neighbors,
        "output_face_lineage": [
            {"output_face_index": i0, **parent_pair},
            {"output_face_index": i1, **parent_pair},
        ],
        "edge_incidence_updates": edge_incidence_updates,
        "vertex_neighbor_updates": {
            "remove": [[edge_start, edge_end], [edge_end, edge_start]],
            "add": [[opposite_a, opposite_b], [opposite_b, opposite_a]],
        },
        "source_pair_area_m2": [metric[0] for metric in old_metrics],
        "candidate_pair_area_m2": [metric[0] for metric in new_metrics],
        "source_pair_minimum_altitude_m": old_min_altitude,
        "candidate_pair_minimum_altitude_m": new_min_altitude,
        "required_minimum_altitude_m": float(minimum_altitude_m),
        "source_pair_oriented_area_vector_m2": old_area_vector.tolist(),
        "candidate_pair_oriented_area_vector_m2": new_area_vector.tolist(),
        "oriented_area_vector_delta_norm_m2": float(np.linalg.norm(new_area_vector - old_area_vector)),
        "local_orientation_alignment_cosines": {
            "source_faces": alignment_cosines[:2],
            "candidate_faces": alignment_cosines[2:],
            "reference": "sum of the two source oriented face area vectors",
        },
        "source_pair_signed_volume_contribution_m3": old_volume,
        "candidate_pair_signed_volume_contribution_m3": new_volume,
        "signed_volume_delta_m3": new_volume - old_volume,
        "vertices_modified": False,
        "face_count_preserved": True,
    }
    return output, report


def validate_closed_oriented_surface(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int | bool]:
    """Validate one full triangle array after a batch of local flips.

    This checks finite positions, nondegenerate Float64 source triangles,
    duplicate faces, two-face edge incidence with opposing winding, and
    connected components. It reports Euler characteristic without assuming a
    particular genus or silently requiring all payload vertices to be used.
    """
    v = _positions(vertices)
    f = _face_array(faces)
    _check_face_bounds(v, f)
    adjacency = build_surface_adjacency(f)
    canonical_faces = np.sort(f, axis=1)
    if len(np.unique(canonical_faces, axis=0)) != len(f):
        raise ValueError("surface contains duplicate unoriented faces")
    triangles = v[f]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    cross_norms = np.linalg.norm(crosses, axis=1)
    if not np.isfinite(cross_norms).all() or bool((cross_norms <= 0.0).any()):
        raise ValueError("surface contains a zero-area or non-finite source face")

    directions: dict[Edge, list[tuple[int, int]]] = defaultdict(list)
    vertex_link_neighbors: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    vertex_link_edges: dict[int, set[Edge]] = defaultdict(set)
    for face_index, triangle in enumerate(f):
        for edge, direction in _edges(triangle):
            directions[edge].append((face_index, direction))
        a, b, c = map(int, triangle)
        for center, left, right in ((a, b, c), (b, c, a), (c, a, b)):
            link_edge = (min(left, right), max(left, right))
            if link_edge in vertex_link_edges[center]:
                raise ValueError(f"vertex {center} has a duplicate edge in its link")
            vertex_link_edges[center].add(link_edge)
            vertex_link_neighbors[center][left].add(right)
            vertex_link_neighbors[center][right].add(left)
    for edge, rows in directions.items():
        if len(rows) != 2:
            raise ValueError(f"surface edge {edge} has {len(rows)} incident faces, expected two")
        if rows[0][1] + rows[1][1] != 0:
            raise ValueError(f"surface edge {edge} has inconsistent oriented face winding")
    for vertex, link in vertex_link_neighbors.items():
        expected_neighbors = adjacency.vertex_neighbors[vertex]
        if set(link) != set(expected_neighbors):
            raise ValueError(f"vertex {vertex} link does not cover its surface neighbors")
        if any(len(neighbors) != 2 for neighbors in link.values()):
            raise ValueError(f"vertex {vertex} link is not a cycle")
        unseen_neighbors = set(link)
        component_count = 0
        while unseen_neighbors:
            component_count += 1
            queue = deque((unseen_neighbors.pop(),))
            while queue:
                for neighbor in link[queue.popleft()] & unseen_neighbors:
                    unseen_neighbors.remove(neighbor)
                    queue.append(neighbor)
        if component_count != 1:
            raise ValueError(f"vertex {vertex} link has {component_count} disconnected cycles")

    referenced = set(map(int, f.reshape(-1)))
    unseen = set(referenced)
    components = 0
    while unseen:
        components += 1
        queue = deque((unseen.pop(),))
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency.vertex_neighbors[vertex] & unseen:
                unseen.remove(neighbor)
                queue.append(neighbor)
    vertex_count = len(referenced)
    edge_count = len(adjacency.edge_faces)
    return {
        "closed_oriented": True,
        "vertex_manifold_checked": True,
        "vertex_count": vertex_count,
        "unused_vertex_count": len(v) - vertex_count,
        "edge_count": edge_count,
        "face_count": len(f),
        "connected_component_count": components,
        "euler_characteristic": vertex_count - edge_count + len(f),
        "zero_area_face_count": 0,
    }

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def inferred_boundary_projection_field(
    vertices_world_m: np.ndarray,
    faces: np.ndarray,
    rings: list[dict[str, Any]],
    *,
    support_radius_edge_multiple: float = 4.0,
    neighbor_count: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a compact, tapered field for source-bound skin aperture vertices.

    Each ring supplies exact boundary vertex IDs, an eye-envelope center, and
    nonnegative radial target distances in metres. The boundary targets are
    applied exactly; nearby skin vertices receive an inverse-distance blend
    that tapers to zero at a graph-geodesic support radius. This is an inferred
    registration field, not a measured eyelid or skin-thickness model.
    """
    points = np.asarray(vertices_world_m, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("skin positions must be a finite N x 3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("skin topology must be a nonempty M x 3 array")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(points):
        raise ValueError("skin topology index is out of range")
    if not np.isfinite(support_radius_edge_multiple) or support_radius_edge_multiple <= 0.0:
        raise ValueError("support radius multiplier must be finite and positive")
    if neighbor_count < 1:
        raise ValueError("neighbor count must be positive")

    edge_rows = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]]), axis=0)
    edges = np.unique(np.sort(edge_rows, axis=1), axis=0)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    if not np.isfinite(edge_lengths).all():
        raise ValueError("skin edge lengths contain non-finite values")
    graph = csr_matrix(
        (np.concatenate((edge_lengths, edge_lengths)),
         (np.concatenate((edges[:, 0], edges[:, 1])),
          np.concatenate((edges[:, 1], edges[:, 0])))),
        shape=(len(points), len(points)),
    )

    total_delta = np.zeros_like(points)
    result: dict[str, Any] = {
        "support_radius_edge_multiple": float(support_radius_edge_multiple),
        "interpolation_neighbor_count": int(neighbor_count),
        "rings": {},
    }
    occupied = np.zeros(len(points), dtype=np.uint8)

    for ring in rings:
        name = str(ring["name"])
        vertex_ids = np.asarray(ring["vertex_ids"], dtype=np.int64)
        center = np.asarray(ring["center_world_m"], dtype=np.float64)
        target_distance = np.asarray(ring["radial_displacement_m"], dtype=np.float64)
        if (vertex_ids.ndim != 1 or len(vertex_ids) < 3
                or len(np.unique(vertex_ids)) != len(vertex_ids)
                or int(vertex_ids.min()) < 0 or int(vertex_ids.max()) >= len(points)
                or center.shape != (3,) or not np.isfinite(center).all()
                or target_distance.shape != vertex_ids.shape
                or not np.isfinite(target_distance).all()
                or bool((target_distance < 0.0).any())):
            raise ValueError(f"ring {name} has invalid boundary targets")
        if bool((occupied[vertex_ids] != 0).any()):
            raise ValueError("multiple rings reuse a boundary vertex")
        occupied[vertex_ids] = 1

        incident = np.isin(edges[:, 0], vertex_ids) | np.isin(edges[:, 1], vertex_ids)
        local_lengths = edge_lengths[incident & (edge_lengths > 0.0)]
        if local_lengths.size == 0:
            raise ValueError(f"ring {name} has no positive incident skin edges")
        median_edge = float(np.median(local_lengths))
        radius = support_radius_edge_multiple * median_edge
        radial = points[vertex_ids] - center[None, :]
        radial_length = np.linalg.norm(radial, axis=1)
        if bool((radial_length <= 1.0e-12).any()):
            raise ValueError(f"ring {name} has a boundary point at its envelope center")
        prescribed = radial / radial_length[:, None] * target_distance[:, None]

        distances = np.asarray(dijkstra(
            graph, directed=False, indices=vertex_ids, limit=radius,
        ), dtype=np.float64)
        nearest = distances.min(axis=0)
        active = np.flatnonzero(nearest < radius)
        field = np.zeros_like(points)
        if active.size:
            available = distances[:, active]
            k = min(neighbor_count, len(vertex_ids))
            nearest_rows = np.argpartition(available, kth=k - 1, axis=0)[:k]
            selected_distance = np.take_along_axis(available, nearest_rows, axis=0)
            epsilon = max(median_edge * 0.25, 1.0e-12)
            weights = 1.0 / np.square(selected_distance + epsilon)
            selected_displacement = prescribed[nearest_rows]
            blended = np.sum(weights[:, :, None] * selected_displacement, axis=0)
            blended /= np.sum(weights, axis=0)[:, None]
            unit = nearest[active] / radius
            fade = 1.0 - 3.0 * unit * unit + 2.0 * unit * unit * unit
            field[active] = blended * fade[:, None]

        # Ring vertices are the source-bound projection targets and anchors.
        # Keep them exact instead of allowing neighbor interpolation to move
        # already-outside end-arc anchors or overshoot low-demand inward points.
        field[vertex_ids] = prescribed
        overlap = np.linalg.norm(total_delta, axis=1) > 0.0
        local_nonzero = np.linalg.norm(field, axis=1) > 0.0
        if bool((overlap & local_nonzero).any()):
            raise ValueError("eye registration support fields overlap")
        total_delta += field
        result["rings"][name] = {
            "boundary_vertex_count": int(len(vertex_ids)),
            "positive_projection_count": int(np.count_nonzero(target_distance > 0.0)),
            "preserved_anchor_count": int(np.count_nonzero(target_distance == 0.0)),
            "median_incident_edge_mm": median_edge * 1000.0,
            "support_radius_mm": radius * 1000.0,
            "support_vertex_count": int(np.count_nonzero(np.linalg.norm(field, axis=1) > 0.0)),
            "maximum_boundary_projection_mm": float(target_distance.max(initial=0.0) * 1000.0),
            "maximum_field_displacement_mm": float(np.linalg.norm(field, axis=1).max(initial=0.0) * 1000.0),
            "field": field,
        }

    result["rings"] = {
        name: {key: value for key, value in detail.items() if key != "field"}
        for name, detail in result["rings"].items()
    }
    result["nonzero_vertex_count"] = int(np.count_nonzero(np.linalg.norm(total_delta, axis=1) > 0.0))
    result["maximum_displacement_mm"] = float(np.linalg.norm(total_delta, axis=1).max(initial=0.0) * 1000.0)
    if not np.isfinite(total_delta).all():
        raise ValueError("skin eye-registration field is non-finite")
    return total_delta, result

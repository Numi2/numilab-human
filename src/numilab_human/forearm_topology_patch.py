"""Strict local edge-collapse construction for emitted forearm surfaces.

This owner performs only a topology change. It never moves or merges source
points, canonical identities, or full21 material terms.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def collapse_exact_duplicate_edge(
    faces_local_u32: np.ndarray,
    positions_world_f32: np.ndarray,
    canonical_material_gids_u64: np.ndarray,
    keep_vertex: int,
    retire_vertex: int,
    *,
    expected_incident_faces: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Collapse a byte-identical emitted edge after checking the simplicial link.

    The kept vertex's complete material owner remains authoritative. The retired
    vertex and its source/parent records remain allocated and unchanged; only
    face indices are rewritten, and the two incident degenerate faces are
    removed. This is a local inferred topology repair, not an anatomical gate.
    """
    faces = np.asarray(faces_local_u32)
    points = np.asarray(positions_world_f32)
    gids = np.asarray(canonical_material_gids_u64)
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be ordered UInt32 triangles")
    if points.dtype != np.dtype("<f4") or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("emitted points must be Float32 metre triples")
    if gids.dtype != np.dtype("<u8") or gids.shape != (len(points),):
        raise ValueError("canonical material GIDs must align with emitted points")
    keep, retire = int(keep_vertex), int(retire_vertex)
    if keep == retire or min(keep, retire) < 0 or max(keep, retire) >= len(points):
        raise ValueError("edge endpoints must be distinct valid vertices")
    if len(np.unique(gids)) != len(gids) or gids[keep] == gids[retire]:
        raise ValueError("edge endpoints require distinct unique canonical GIDs")
    if points[keep].tobytes() != points[retire].tobytes():
        raise ValueError("edge endpoints are not byte-identical emitted points")
    if int(faces.max(initial=0)) >= len(points):
        raise ValueError("face index exceeds emitted vertex count")
    if np.any((faces[:, 0] == faces[:, 1]) |
              (faces[:, 1] == faces[:, 2]) |
              (faces[:, 2] == faces[:, 0])):
        raise ValueError("input contains a topological degenerate face")

    keep_faces = np.any(faces == keep, axis=1)
    retire_faces = np.any(faces == retire, axis=1)
    edge_faces = np.flatnonzero(keep_faces & retire_faces)
    if len(edge_faces) != 2:
        raise ValueError(f"edge must have exactly two incident faces, got {len(edge_faces)}")
    if expected_incident_faces is not None and tuple(map(int, edge_faces)) != tuple(
        sorted(map(int, expected_incident_faces))
    ):
        raise ValueError("incident face indices differ from pinned local repair witness")

    keep_neighbors = set(map(int, faces[keep_faces].ravel())) - {keep}
    retire_neighbors = set(map(int, faces[retire_faces].ravel())) - {retire}
    common_link = keep_neighbors & retire_neighbors
    opposites = []
    for face_index in edge_faces:
        opposite = [int(v) for v in faces[int(face_index)] if int(v) not in (keep, retire)]
        if len(opposite) != 1:
            raise ValueError("edge incident triangle has no unique opposite vertex")
        opposites.append(opposite[0])
    if len(set(opposites)) != 2 or common_link != set(opposites):
        raise ValueError("edge fails the strict simplicial link condition")

    rewritten = faces.copy()
    rewritten[rewritten == retire] = np.uint32(keep)
    degenerate = np.flatnonzero(
        (rewritten[:, 0] == rewritten[:, 1]) |
        (rewritten[:, 1] == rewritten[:, 2]) |
        (rewritten[:, 2] == rewritten[:, 0])
    )
    if tuple(map(int, degenerate)) != tuple(map(int, edge_faces)):
        raise ValueError("collapse would remove faces beyond the two edge incident triangles")
    repaired = np.delete(rewritten, degenerate, axis=0)

    canonical_faces = [tuple(sorted(map(int, face))) for face in repaired]
    if len(set(canonical_faces)) != len(canonical_faces):
        raise ValueError("collapse would create duplicate unordered faces")
    edge_counts = Counter()
    for face in repaired:
        a, b, c = map(int, face)
        edge_counts[tuple(sorted((a, b)))] += 1
        edge_counts[tuple(sorted((b, c)))] += 1
        edge_counts[tuple(sorted((c, a)))] += 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    non_two_incidence = [edge for edge, count in edge_counts.items() if count != 2]
    if boundary or non_two_incidence:
        raise ValueError(
            f"collapse output fails closed edge incidence: boundary={len(boundary)}, "
            f"non_two_incidence={len(non_two_incidence)}"
        )
    return repaired.astype("<u4", copy=False), {
        "method": "strict_link_condition_collapse_of_byte_identical_emitted_edge",
        "kept_vertex_local_index": keep,
        "retired_vertex_local_index_still_allocated": retire,
        "kept_canonical_material_gid": int(gids[keep]),
        "retired_canonical_material_gid": int(gids[retire]),
        "endpoint_world_positions_byte_identical": True,
        "incident_face_indices_removed": list(map(int, edge_faces)),
        "opposite_vertices": sorted(opposites),
        "neighbor_link_intersection": sorted(common_link),
        "strict_link_condition_passed": True,
        "remaining_face_count": int(len(repaired)),
        "closed_edge_incidence_two_after_collapse": True,
        "vertex_link_orientation_and_full_manifold_gate_required": True,
        "no_vertex_or_material_arrays_are_modified_by_this_face_only_operation": True,
        "anatomical_admission": False,
    }


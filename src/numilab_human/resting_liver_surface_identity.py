"""Receipt contract for the inferred eight-patch aggregate liver display."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

LIVER_PATCH_IDS = tuple(range(14, 22))
LIVER_REPRESENTATION = "single_closed_aggregate_partitioned_into_eight_passive_exterior_patches_v1"
LIVER_PATCH_ROLE = "one open surface patch of a single closed aggregate shell; no independent segment volume"
LIVER_PATCH_INTERPRETATION = "inferred display identity; no internal segment partition or segment volume"


def liver_surface_identity_metadata() -> dict[str, Any]:
    """Return the only supported receipt declaration for the eight patch form."""
    return {
        "representation": LIVER_REPRESENTATION,
        "stable_ids": list(LIVER_PATCH_IDS),
        "retired_source_aliases": {"22": 21},
        "internal_segment_boundaries": False,
        "segment_volumes_defined": False,
    }


def validate_liver_surface_identity_metadata(value: Any) -> None:
    """Fail closed unless a receipt names exactly the supported patch form."""
    if not isinstance(value, Mapping) or set(value) != {
        "representation", "stable_ids", "retired_source_aliases",
        "internal_segment_boundaries", "segment_volumes_defined",
    }:
        raise ValueError("liver surface identity must contain exactly the supported representation fields")
    stable_ids = value["stable_ids"]
    alias = value["retired_source_aliases"]
    if (
        value["representation"] != LIVER_REPRESENTATION
        or not isinstance(stable_ids, list)
        or any(type(item) is not int for item in stable_ids)
        or stable_ids != list(LIVER_PATCH_IDS)
        or not isinstance(alias, Mapping)
        or set(alias) != {"22"}
        or type(alias["22"]) is not int
        or alias["22"] != 21
        or value["internal_segment_boundaries"] is not False
        or value["segment_volumes_defined"] is not False
    ):
        raise ValueError("liver surface identity does not match the supported eight-patch aggregate representation")


def validate_liver_patch_coverage(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    stable_id_per_face: np.ndarray,
    patch_rows: Mapping[int, Mapping[str, Any]],
    source_id_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify exact, single-owner coverage of a closed aggregate source mesh.

    `patch_rows` are the eight serialized display surfaces before any later
    common-field refinement. The source face guide owns every aggregate face
    once; coordinate triangles must exactly reproduce their assigned faces.
    This proves display coverage, not internal Couinaud planes or volumes.
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int64)
    owner = np.asarray(stable_id_per_face, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or not np.isfinite(xyz[:, :3]).all():
        raise ValueError("aggregate liver source vertices are invalid")
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise ValueError("aggregate liver source faces are invalid")
    if tri.min() < 0 or tri.max() >= len(xyz):
        raise ValueError("aggregate liver face index is out of range")
    if owner.shape != (len(tri),) or set(map(int, np.unique(owner))) != set(LIVER_PATCH_IDS):
        raise ValueError("aggregate liver source-face guide must assign all eight IDs exactly")
    if set(map(int, patch_rows)) != set(LIVER_PATCH_IDS):
        raise ValueError("serialized liver patch IDs differ from the declared eight-patch representation")
    if "22" in source_id_map:
        raise ValueError("retired liver alias 22 must not be a rendered source-map entry")

    def triangle_key(points: np.ndarray) -> tuple[tuple[float, float, float], ...]:
        return tuple(sorted(tuple(float(x) for x in row[:3]) for row in points))

    expected_counts: dict[str, int] = {}
    covered_source_faces: list[int] = []
    aggregate_edge_orientations: dict[tuple[tuple[float, ...], tuple[float, ...]], list[int]] = {}
    source_face_keys: set[tuple[tuple[float, ...], ...]] = set()
    for triangle in tri:
        points = xyz[triangle, :3]
        if float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))) == 0.0:
            raise ValueError("aggregate liver source contains an exactly degenerate face")
        key = triangle_key(points)
        if key in source_face_keys:
            raise ValueError("aggregate liver source contains duplicate triangle geometry")
        source_face_keys.add(key)
        vertices3 = [tuple(float(x) for x in point) for point in points]
        for a, b in ((vertices3[0], vertices3[1]), (vertices3[1], vertices3[2]), (vertices3[2], vertices3[0])):
            edge = (min(a, b), max(a, b))
            aggregate_edge_orientations.setdefault(edge, []).append(1 if a < b else -1)
    if any(len(rows) != 2 or sum(rows) != 0 for rows in aggregate_edge_orientations.values()):
        raise ValueError("aggregate liver source is not a closed, consistently oriented two-manifold")

    def same_winding(a: np.ndarray, b: np.ndarray) -> bool:
        aa = tuple(tuple(float(x) for x in point[:3]) for point in a)
        bb = tuple(tuple(float(x) for x in point[:3]) for point in b)
        return any(aa == bb[i:] + bb[:i] for i in range(3))

    for stable_id in LIVER_PATCH_IDS:
        key = str(stable_id)
        entry = source_id_map.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"liver source map lacks stable ID {stable_id}")
        owner_metadata = entry.get("source_owner_metadata")
        repair = entry.get("repair")
        if (
            entry.get("body_index") != 20
            or entry.get("layer") != 1
            or not isinstance(owner_metadata, Mapping)
            or owner_metadata.get("source_atlas") != "Z-Anatomy"
            or owner_metadata.get("geometry_role") != LIVER_PATCH_ROLE
            or not isinstance(repair, Mapping)
            or repair.get("interpretation") != LIVER_PATCH_INTERPRETATION
        ):
            raise ValueError(f"liver stable ID {stable_id} is not source-bound as a passive aggregate-shell patch")

        source_face_ids = np.flatnonzero(owner == stable_id)
        row = patch_rows[stable_id]
        row_xyz = np.asarray(row["vertices6"], dtype=np.float32)
        row_faces = np.asarray(row["faces"], dtype=np.int64)
        if row_xyz.ndim != 2 or row_xyz.shape[1] < 3 or row_faces.ndim != 2 or row_faces.shape[1] != 3:
            raise ValueError(f"serialized liver patch {stable_id} is malformed")
        if len(source_face_ids) != len(row_faces):
            raise ValueError(f"liver patch {stable_id} face count differs from source attribution")
        recorded_count = repair.get("patch_face_count")
        if isinstance(recorded_count, bool) or recorded_count != len(source_face_ids):
            raise ValueError(f"liver patch {stable_id} source receipt face count differs")
        source_triangles = [xyz[tri[int(i)]] for i in source_face_ids]
        patch_triangles = [row_xyz[row_faces[i]] for i in range(len(row_faces))]
        source_keys = [triangle_key(t) for t in source_triangles]
        patch_keys = [triangle_key(t) for t in patch_triangles]
        if len(set(source_keys)) != len(source_keys) or len(set(patch_keys)) != len(patch_keys):
            raise ValueError(f"liver patch {stable_id} contains duplicate triangle geometry")
        if set(source_keys) != set(patch_keys):
            raise ValueError(f"liver patch {stable_id} does not reproduce its assigned aggregate faces")
        source_by_key = {triangle_key(t): t for t in source_triangles}
        if any(not same_winding(source_by_key[triangle_key(t)], t) for t in patch_triangles):
            raise ValueError(f"liver patch {stable_id} reverses an aggregate source face")
        expected_counts[key] = len(source_face_ids)
        covered_source_faces.extend(map(int, source_face_ids))

    if sorted(covered_source_faces) != list(range(len(tri))):
        raise ValueError("liver display patches do not cover every aggregate source face exactly once")
    return {
        "aggregate_source_face_count": len(tri),
        "aggregate_source_vertex_count": len(xyz),
        "compiled_source_face_count_by_stable_id": expected_counts,
        "all_aggregate_faces_assigned_exactly_once": True,
        "internal_segment_boundaries": False,
        "segment_volumes_defined": False,
    }

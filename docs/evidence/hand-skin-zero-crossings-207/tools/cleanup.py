"""Exact Float32 cleanup and provenance helpers for the offline Manifold CSG prototype.

The cleanup performs only exact-coordinate Float32 welding, snap-back of the
known source head boundary, removal of repeated-index or exactly zero-area
triangles, and optional restoration of the source winding convention. It keeps
all nonzero sliver triangles.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any
import importlib.util
import sys

import numpy as np

EXACT_OWNER = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/"
    "metal-optimization-080/visual-verification-20261022/"
    "wrist-connected-repair-001/collision-free-seam-records-owner-315/src/"
    "numilab_human/cardiac_cavity_intersections.py"
)


def _f32_keys(points: np.ndarray) -> np.ndarray:
    """Canonical exact-coordinate keys; signed zero is treated as one coordinate."""
    q = np.array(points, dtype=np.float32, order="C", copy=True)
    q[q == 0] = np.float32(0.0)
    return q.view(np.uint32).reshape((-1, 3))


def _source_key_rows(points: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    keys = _f32_keys(points)
    out: dict[tuple[int, int, int], list[int]] = {}
    for row, key in enumerate(keys):
        out.setdefault(tuple(int(v) for v in key), []).append(int(row))
    return out


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    tri = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _boundary_edges(faces: np.ndarray) -> tuple[np.ndarray, int]:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be Fx3")
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    e = np.sort(e, axis=1)
    u, n = np.unique(e, axis=0, return_counts=True)
    return u[n == 1], int(np.count_nonzero(n > 2))


def _edge_coordinate_counter(points: np.ndarray, edges: np.ndarray) -> Counter:
    keys = _f32_keys(points)
    out: Counter = Counter()
    for a, b in np.asarray(edges, dtype=np.int64):
        ka = tuple(int(v) for v in keys[int(a)])
        kb = tuple(int(v) for v in keys[int(b)])
        out[tuple(sorted((ka, kb)))] += 1
    return out


def extract_mesh64_face_provenance(mesh64: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read XYZ, triangle indices, and per-face (run_original_id, face_id).

    Manifold3D Mesh64.run_index offsets are in flattened triangle-index units.
    """
    points = np.asarray(mesh64.vert_properties, dtype=np.float64)
    faces = np.asarray(mesh64.tri_verts, dtype=np.uint64)
    runs = np.asarray(mesh64.run_index, dtype=np.int64)
    original_ids = np.asarray(mesh64.run_original_id, dtype=np.uint32)
    face_ids = np.asarray(mesh64.face_id, dtype=np.uint64)
    if points.ndim != 2 or points.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("invalid Mesh64 position or triangle layout")
    if len(runs) != len(original_ids) + 1 or not len(runs) or runs[0] != 0:
        raise ValueError("Mesh64 run_index and run_original_id are not aligned")
    if runs[-1] != 3 * len(faces) or np.any(runs % 3):
        raise ValueError("Mesh64 run_index does not cover triangle-aligned indices")
    if len(face_ids) != len(faces):
        raise ValueError("Mesh64 face_id does not match triangle count")
    per_face_original = np.empty(len(faces), dtype=np.uint32)
    face_offsets = runs // 3
    for i, original_id in enumerate(original_ids):
        per_face_original[face_offsets[i]:face_offsets[i + 1]] = original_id
    return points[:, :3].copy(), faces.astype(np.int64), per_face_original, face_ids


def _weld_exact_f32(
    points: np.ndarray, faces: np.ndarray, source_rows: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.array(points, dtype=np.float32, order="C", copy=True)
    p[p == 0] = np.float32(0.0)
    keybits = p.view(np.uint32).reshape((-1, 3))
    unique, first, inverse = np.unique(keybits, axis=0, return_index=True, return_inverse=True)
    out = p[first].copy()
    out_rows = source_rows[first].copy()
    by_key = _source_key_rows(source)
    for i, key in enumerate(unique):
        matches = by_key.get(tuple(int(v) for v in key), ())
        if matches:
            out_rows[i] = min(matches)
            out[i] = source[min(matches)]
    remapped_faces = inverse[np.asarray(faces, dtype=np.int64)].astype(np.int64, copy=False)
    return out, remapped_faces, out_rows, np.asarray(first, dtype=np.int64)


def _compact(points: np.ndarray, faces: np.ndarray, source_rows: np.ndarray):
    used = np.unique(np.asarray(faces, dtype=np.int64).reshape(-1))
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return points[used], remap[faces], source_rows[used]


def _restore_head_boundary(
    points: np.ndarray,
    faces: np.ndarray,
    source: np.ndarray,
    source_boundary_edges: np.ndarray,
    source_rows: np.ndarray,
    *,
    snap_ulps: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Snap output boundary vertices back only to unique source head vertices."""
    p = np.array(points, dtype=np.float32, order="C", copy=True)
    rows = np.asarray(source_rows, dtype=np.int64).copy()
    target_rows = np.unique(np.asarray(source_boundary_edges, dtype=np.int64).reshape(-1))
    candidate_edges, _ = _boundary_edges(faces)
    candidate_rows = np.unique(candidate_edges.reshape(-1)) if len(candidate_edges) else np.empty(0, dtype=np.int64)
    target_set = set(map(int, target_rows))
    row_owner: dict[int, int] = {}
    source_by_key = _source_key_rows(source)
    for v in candidate_rows:
        key = tuple(int(x) for x in _f32_keys(p[[v]])[0])
        exact = [r for r in source_by_key.get(key, ()) if r in target_set]
        if len(exact) == 1:
            row_owner[int(v)] = int(exact[0])
    used_targets = set(row_owner.values())
    unmatched = [int(v) for v in candidate_rows if int(v) not in row_owner]
    snapped = []
    source_boundary = np.asarray(source[target_rows], dtype=np.float32)
    for v in unmatched:
        if not len(target_rows):
            break
        d = np.linalg.norm(source_boundary.astype(np.float64) - p[int(v)].astype(np.float64), axis=1)
        order = np.argsort(d, kind="stable")
        target_index = int(order[0])
        row = int(target_rows[target_index])
        spacing = np.spacing(np.abs(source[row]).astype(np.float32)).astype(np.float64)
        tol = float(snap_ulps * max(np.linalg.norm(spacing), np.finfo(np.float32).tiny))
        second = float(d[int(order[1])]) if len(order) > 1 else np.inf
        if float(d[target_index]) <= tol and second - float(d[target_index]) > tol and row not in used_targets:
            p[int(v)] = source[row]
            rows[int(v)] = row
            row_owner[int(v)] = row
            used_targets.add(row)
            snapped.append({"output_vertex": int(v), "source_row": row, "distance_m": float(d[target_index]), "tolerance_m": tol})
    return p, rows, {
        "source_head_boundary_vertex_count": int(len(target_rows)),
        "candidate_head_boundary_vertex_count": int(len(candidate_rows)),
        "exact_or_snapped_boundary_vertex_matches": int(len(row_owner)),
        "boundary_vertices_snapped_to_source": snapped,
        "unmatched_candidate_boundary_vertices": int(len(candidate_rows) - len(row_owner)),
        "unrepresented_source_boundary_vertices": int(len(set(map(int, target_rows)) - set(row_owner.values()))),
    }


def cleanup_boolean_mesh(
    points64: np.ndarray,
    faces: np.ndarray,
    face_original_ids: np.ndarray,
    face_ids: np.ndarray,
    source_points_f32: np.ndarray,
    source_faces: np.ndarray,
    *,
    cap_original_id: int | None = None,
    source_original_id: int | None = None,
    cap_face_id_start: int | None = None,
    expected_original_ids: tuple[int, ...] | None = None,
    preserve_source_rows: np.ndarray | None = None,
    restore_source_winding: bool = True,
    boundary_snap_ulps: float = 2.0,
) -> dict[str, Any]:
    """Clean CSG output after removing temporary caps by explicit face provenance.

    Caps may use a dedicated run_original_id, or share the source run and use
    face_id >= cap_face_id_start. A cap selector is mandatory. No fuzzy
    triangle deletion occurs: only repeated welded vertices and exact-zero
    cross products are removed.
    """
    p64 = np.asarray(points64, dtype=np.float64)
    f0 = np.asarray(faces, dtype=np.int64)
    fo0 = np.asarray(face_original_ids, dtype=np.uint32)
    fi0 = np.asarray(face_ids, dtype=np.uint64)
    src = np.asarray(source_points_f32, dtype=np.float32)
    sf = np.asarray(source_faces, dtype=np.int64)
    if p64.ndim != 2 or p64.shape[1] != 3 or not np.isfinite(p64).all():
        raise ValueError("points64 must be finite Nx3")
    if f0.ndim != 2 or f0.shape[1] != 3 or f0.min(initial=0) < 0 or f0.max(initial=0) >= len(p64):
        raise ValueError("invalid boolean triangle indices")
    if fo0.shape != (len(f0),) or fi0.shape != (len(f0),):
        raise ValueError("face provenance arrays do not align")
    if src.ndim != 2 or src.shape[1] != 3 or not np.isfinite(src).all():
        raise ValueError("source points must be finite Nx3 Float32")
    if sf.ndim != 2 or sf.shape[1] != 3 or sf.min(initial=0) < 0 or sf.max(initial=0) >= len(src):
        raise ValueError("invalid source faces")
    if cap_original_id is None and (source_original_id is None or cap_face_id_start is None):
        raise ValueError("provide cap_original_id or source_original_id plus cap_face_id_start")

    cap = np.zeros(len(f0), dtype=bool)
    if cap_original_id is not None:
        cap |= fo0 == np.uint32(cap_original_id)
    if source_original_id is not None and cap_face_id_start is not None:
        cap |= (fo0 == np.uint32(source_original_id)) & (fi0 >= np.uint64(cap_face_id_start))
    f = f0[~cap].copy()
    fo = fo0[~cap].copy()
    fi = fi0[~cap].copy()

    # F64->F32 is the emitted geometry. Exact source-coordinate matches are
    # rewritten from the source buffer so their original packed bits survive.
    p32 = p64.astype(np.float32)
    p32[p32 == 0] = np.float32(0.0)
    source_by_key = _source_key_rows(src)
    point_keys = _f32_keys(p32)
    source_row = np.full(len(p32), -1, dtype=np.int64)
    ambiguous_source_match = 0
    for v, key in enumerate(point_keys):
        matches = source_by_key.get(tuple(int(x) for x in key), ())
        if matches:
            source_row[v] = min(matches)
            p32[v] = src[min(matches)]
            if len(matches) > 1:
                ambiguous_source_match += 1

    p32, f, source_row, _ = _weld_exact_f32(p32, f, source_row, src)
    source_boundary, source_nonmanifold = _boundary_edges(sf)
    p32, source_row, boundary_restore = _restore_head_boundary(
        p32, f, src, source_boundary, source_row, snap_ulps=boundary_snap_ulps
    )
    # Snap-back can merge vertices; re-weld before exact degeneracy checks.
    p32, f, source_row, _ = _weld_exact_f32(p32, f, source_row, src)

    repeated_vertex = (
        (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    )
    tri = p32[f].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    exact_zero_area = np.all(cross == 0.0, axis=1)
    degenerate = repeated_vertex | exact_zero_area
    fo, fi = fo[~degenerate], fi[~degenerate]
    f = f[~degenerate]
    p32, f, source_row = _compact(p32, f, source_row)

    baseline_volume = _signed_volume(src, sf)
    candidate_volume_before = _signed_volume(p32, f)
    reversed_for_source = False
    if restore_source_winding and baseline_volume * candidate_volume_before < 0.0:
        f = f[:, [0, 2, 1]].copy()
        reversed_for_source = True

    candidate_boundary, candidate_nonmanifold = _boundary_edges(f)
    source_boundary_keys = _edge_coordinate_counter(src, source_boundary)
    candidate_boundary_keys = _edge_coordinate_counter(p32, candidate_boundary)
    missing_boundary = source_boundary_keys - candidate_boundary_keys
    extra_boundary = candidate_boundary_keys - source_boundary_keys
    if expected_original_ids is None:
        unknown_original_ids = {}
    else:
        expected = set(map(int, expected_original_ids))
        ids, counts = np.unique(fo, return_counts=True)
        unknown_original_ids = {int(k): int(v) for k, v in zip(ids, counts) if int(k) not in expected}

    required = np.arange(len(src), dtype=np.int64) if preserve_source_rows is None else np.unique(np.asarray(preserve_source_rows, dtype=np.int64))
    if np.any(required < 0) or np.any(required >= len(src)):
        raise ValueError("preserve_source_rows outside source point domain")
    present = set(map(int, source_row[source_row >= 0]))
    missing_source_rows = sorted(set(map(int, required)) - present)
    areas2 = np.linalg.norm(np.cross(
        p32[f[:, 1]].astype(np.float64) - p32[f[:, 0]].astype(np.float64),
        p32[f[:, 2]].astype(np.float64) - p32[f[:, 0]].astype(np.float64),
    ), axis=1)

    report = {
        "input_mesh64_vertices": int(len(p64)),
        "input_mesh64_faces": int(len(f0)),
        "temporary_cap_faces_removed": int(np.count_nonzero(cap)),
        "vertices_after_exact_f32_weld_and_compaction": int(len(p32)),
        "exact_f32_vertices_welded": int(len(p64) - len(p32)),
        "degenerate_faces_removed_exact_only": int(np.count_nonzero(degenerate)),
        "degenerate_breakdown": {
            "repeated_welded_vertex": int(np.count_nonzero(repeated_vertex)),
            "exact_zero_cross": int(np.count_nonzero(exact_zero_area)),
        },
        "retained_face_count": int(len(f)),
        "source_vertex_coordinate_matches": int(np.count_nonzero(source_row >= 0)),
        "ambiguous_source_coordinate_matches": int(ambiguous_source_match),
        "source_signed_volume_m3": float(baseline_volume),
        "candidate_signed_volume_m3_before_source_winding_restore": float(candidate_volume_before),
        "reversed_to_source_winding": bool(reversed_for_source),
        "head_boundary_restore": boundary_restore,
        "source_open_boundary_edge_count": int(len(source_boundary)),
        "candidate_open_boundary_edge_count": int(len(candidate_boundary)),
        "source_nonmanifold_edge_count": int(source_nonmanifold),
        "candidate_nonmanifold_edge_count": int(candidate_nonmanifold),
        "head_boundary_exact_match": not missing_boundary and not extra_boundary,
        "missing_source_boundary_edge_count": int(sum(missing_boundary.values())),
        "extra_candidate_boundary_edge_count": int(sum(extra_boundary.values())),
        "required_source_vertex_count": int(len(required)),
        "required_source_vertices_missing": int(len(missing_source_rows)),
        "missing_source_vertex_rows_first_32": missing_source_rows[:32],
        "unknown_face_original_id_counts": unknown_original_ids,
        "minimum_retained_triangle_cross_norm_m2": float(areas2.min()) if len(areas2) else None,
        "geometry_or_native_admission": False,
    }
    return {
        "points_world_m": p32,
        "source_O_faces": f,
        "face_original_id": fo,
        "face_id": fi,
        "vertex_source_row": source_row,
        "report": report,
    }


def exact_self_intersection_report(points_f32: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Run the pinned exact-rational same-surface triangle-pair predicate."""
    if not EXACT_OWNER.is_file():
        raise FileNotFoundError(EXACT_OWNER)
    import importlib.util
    package_src = str(EXACT_OWNER.parents[3] / "incremental-exact-audit-owner-253" / "src")
    if package_src not in sys.path:
        sys.path.insert(0, package_src)
    module_name = "numilab_human.connected_skin_bone_exact_159"
    spec = importlib.util.spec_from_file_location(module_name, EXACT_OWNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load exact intersection owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    p = np.asarray(points_f32, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    vertices = tuple(tuple(Fraction.from_float(float(x)) for x in point) for point in p)
    records = module._records(vertices, tuple(tuple(int(v) for v in face) for face in f))
    result = module._audit_pair(records, records, same_surface=True)
    return {
        "predicate_owner": str(EXACT_OWNER),
        "triangle_count": int(len(f)),
        "exact_triangle_pair_count": int(result["count"]),
        "aabb_candidate_pairs": int(result["aabb_candidate_pairs"]),
        "allowed_shared_vertex_or_edge_pairs": int(result["allowed_shared_vertex_or_edge_pairs"]),
        "first_pairs": result["triangle_pairs"][:32],
        "complete": True,
        "geometry_or_native_admission": False,
    }


def write_minimal_candidate(path: str | Path, points_f32: np.ndarray, faces: np.ndarray) -> None:
    """Write the exact-audit-compatible NPZ containing only XYZ and faces."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, points_world_m=np.asarray(points_f32, np.float32),
                            source_O_faces=np.asarray(faces, np.int64))


def write_provenance(path: str | Path, result: dict[str, Any]) -> None:
    """Persist per-face Manifold tags and the exact source-row vertex map."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            vertex_source_row=np.asarray(result["vertex_source_row"], dtype=np.int64),
            face_original_id=np.asarray(result["face_original_id"], dtype=np.uint32),
            face_id=np.asarray(result["face_id"], dtype=np.uint64),
        )


if __name__ == "__main__":
    raise SystemExit(
        "Import cleanup_boolean_mesh, extract_mesh64_face_provenance, or "
        "exact_self_intersection_report from the CSG runner."
    )

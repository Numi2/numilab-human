"""Bounded source-based simplification for one reconstructed subscapularis row.

This wrapper applies the existing closed-surface edge-collapse owner to a
preselected local edge set at a declared geometric resolution. It preserves
face-parent lineage and interpolates existing canonical skinning maps at each
midpoint. It does not register the source, certify an attachment footprint, or
admit a runtime payload.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .resting_lung_edge_repair import collapse_midpoint_edge


def collapse_subscapularis_resolution_edges(
    vertices6: np.ndarray,
    faces: np.ndarray,
    face_origins: np.ndarray,
    canonical_weights: np.ndarray,
    selected_edges: Iterable[tuple[int, int]],
    *,
    resolution_m: float,
    locked_source_vertex_ids: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Collapse declared sub-resolution source edges and retain exact lineage.

    selected_edges refer to vertex IDs in the supplied initial mesh. The
    helper rejects an edge if prior collapse made either endpoint ambiguous,
    if it touches a locked source vertex, if it exceeds the declared resolution,
    or if the existing owner rejects the closed-manifold link/volume/orientation
    checks. Canonical map columns are inherited body owners; the midpoint gets
    their arithmetic mean, with no weight renormalization.
    """
    v = np.ascontiguousarray(vertices6, dtype="<f4")
    # Source identifiers must not be silently rounded or truncated by coercion.
    face_ids = np.asarray(faces)
    origin_ids = np.asarray(face_origins)
    if face_ids.dtype.kind not in "iu" or origin_ids.dtype.kind not in "iu":
        raise ValueError("faces and face-parent origins must use integer source identifiers")
    f = np.ascontiguousarray(face_ids, dtype=np.int64)
    origins = np.ascontiguousarray(origin_ids, dtype=np.int64)
    weights = np.ascontiguousarray(canonical_weights, dtype="<f4")
    if (v.ndim != 2 or v.shape[1] != 6 or not len(v) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or origins.shape != (len(f),) or np.any(origins < 0)
            or weights.ndim != 2 or weights.shape[0] != len(v)
            or weights.shape[1] < 1 or not np.isfinite(weights).all()):
        raise ValueError("malformed reconstructed subscapularis mesh or map table")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("reconstructed subscapularis face index outside vertex table")
    if not np.allclose(weights.astype(np.float64).sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("canonical body weights do not sum to one before simplification")
    resolution = float(resolution_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("declared source resolution must be finite and positive")

    locked_ids = list(locked_source_vertex_ids)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in locked_ids):
        raise ValueError("locked vertices must use integer source identifiers")
    locked = {int(i) for i in locked_ids}
    if any(i < 0 or i >= len(v) for i in locked):
        raise ValueError("locked source vertex is outside reconstructed mesh")
    initial_vertex_count = len(v)
    lineage: list[frozenset[int]] = [frozenset((i,)) for i in range(len(v))]
    rows: list[dict] = []
    touched: set[int] = set()
    for pair in selected_edges:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in pair)):
            raise ValueError("selected edge must name two initial integer source vertex IDs")
        source_u, source_w = map(int, pair)
        if source_u == source_w or min(source_u, source_w) < 0 or max(source_u, source_w) >= initial_vertex_count:
            raise ValueError("selected source edge endpoint is invalid")
        if source_u in touched or source_w in touched:
            raise ValueError("selected resolution edges must be disjoint")
        touched.update((source_u, source_w))
        if {source_u, source_w} & locked:
            raise ValueError("resolution edge touches a locked attachment/core source row")
        matches_u = [i for i, ids in enumerate(lineage) if source_u in ids]
        matches_w = [i for i, ids in enumerate(lineage) if source_w in ids]
        if len(matches_u) != 1 or len(matches_w) != 1 or matches_u[0] == matches_w[0]:
            raise ValueError("selected original endpoint no longer maps uniquely after collapse")
        u, w = matches_u[0], matches_w[0]
        edge_length = float(np.linalg.norm(v[u, :3].astype(np.float64) - v[w, :3].astype(np.float64)))
        if edge_length > resolution:
            raise ValueError("selected edge exceeds the declared reconstruction resolution")
        # The source resolution is a maximum edge length; half that value is
        # the per-endpoint displacement cap for a midpoint collapse.
        endpoint_rounding = 2.0 * np.finfo(np.float32).eps * float(np.max(np.abs(v[[u, w], :3])))
        endpoint_cap = 0.5 * resolution + endpoint_rounding
        replaced = f.copy()
        replaced[replaced == u] = w
        keep = ((replaced[:, 0] != replaced[:, 1]) &
                (replaced[:, 1] != replaced[:, 2]) &
                (replaced[:, 2] != replaced[:, 0]))
        used = np.unique(replaced[keep].reshape(-1))
        old_to_new = np.full(len(v), -1, dtype=np.int64)
        old_to_new[used] = np.arange(len(used), dtype=np.int64)
        expected_survivor = int(old_to_new[w])
        if expected_survivor < 0 or old_to_new[u] >= 0:
            raise ValueError("selected collapse does not remove exactly the declared endpoint")
        old_v, old_f, old_origins, old_weights, old_lineage = v, f, origins, weights, lineage
        v, f, origins, detail = collapse_midpoint_edge(
            old_v, old_f, old_origins, remove_vertex=u, keep_vertex=w,
            position_policy="midpoint", max_endpoint_displacement_m=endpoint_cap,
            max_abs_volume_delta_m3=resolution ** 3,
        )
        if detail["retained_vertex_after_compaction"] != expected_survivor:
            raise ValueError("edge-collapse owner returned an unexpected compacted survivor")
        weights = old_weights[used].copy()
        weights[expected_survivor] = (
            old_weights[u].astype(np.float64) + old_weights[w].astype(np.float64)
        ).astype("<f4") * np.float32(0.5)
        lineage = [old_lineage[int(i)] for i in used]
        lineage[expected_survivor] = old_lineage[u] | old_lineage[w]
        if not np.allclose(weights.astype(np.float64).sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError("midpoint map interpolation changed canonical weight sums")
        rows.append({
            **detail,
            "selected_initial_source_edge": [source_u, source_w],
            "declared_resolution_m": resolution,
            "endpoint_displacement_cap_m": endpoint_cap,
            "canonical_weights_interpolated_at_midpoint": True,
            "weight_renormalization_performed": False,
            "merged_source_vertex_ids": sorted(lineage[expected_survivor]),
        })
    return v, f, origins, weights, rows

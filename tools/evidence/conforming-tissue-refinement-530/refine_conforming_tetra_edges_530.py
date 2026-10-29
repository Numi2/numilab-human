#!/usr/bin/env python3
"""Offline conforming tetrahedral edge refinement with exact source ancestry."""
from dataclasses import dataclass
import numpy as np

@dataclass
class EdgeRefinement:
    points_reference_f64: np.ndarray
    tetrahedra: np.ndarray
    child_source_tet_rows: np.ndarray
    child_source_measure_fractions: np.ndarray
    node_source_offsets: np.ndarray
    node_source_rows: np.ndarray
    node_source_weights: np.ndarray
    inserted_node_rows: np.ndarray
    inserted_edge_endpoints: np.ndarray

def signed_volumes(points, tets):
    p = np.asarray(points, dtype=np.float64)[np.asarray(tets, dtype=np.int64)]
    return np.einsum("ij,ij->i", p[:,1]-p[:,0],
                     np.cross(p[:,2]-p[:,0], p[:,3]-p[:,0])) / 6.0

def refine_edges(points, tets, edges):
    """Bisect complete incident tet stars; never move original coordinates.

    Source coordinates stay Float64 here to preserve the source refinement.
    The caller must separately emit and validate Float32 physical coordinates,
    projected targets, material labels, surfaces and anatomical admission.
    A new node's CSR weights refer to input nodes; child fractions refer to
    input cells. Original node IDs remain fixed; no runtime constraint is added.
    """
    x = np.asarray(points, dtype=np.float64).copy()
    raw = np.asarray(tets)
    ee = np.asarray(edges)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite 3D reference points required")
    if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("typed tetrahedra required")
    t = raw.astype(np.int64, copy=True)
    if t.min() < 0 or t.max() >= len(x) or np.any(np.diff(np.sort(t, axis=1), axis=1) == 0):
        raise ValueError("invalid tetrahedron node domain")
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("duplicate input tetrahedron")
    if not np.all(signed_volumes(x, t) > 0):
        raise ValueError("input reference cells must be positively oriented")
    if ee.size == 0:
        ee = np.empty((0,2), dtype=np.int64)
    if ee.ndim != 2 or ee.shape[1] != 2 or not np.issubdtype(ee.dtype, np.integer):
        raise ValueError("typed edge pairs required")
    # Normalize duplicates deterministically, preserving first occurrence order.
    marked = []; seen = set()
    for row in ee:
        a,b = sorted(map(int,row))
        if a < 0 or a == b:
            raise ValueError("invalid marked edge")
        if (a,b) not in seen:
            marked.append((a,b)); seen.add((a,b))
    parent = np.arange(len(t), dtype=np.int64)
    fractions = np.ones(len(t), dtype=np.float64)
    ancestry = [{i:1.0} for i in range(len(x))]
    inserted = []; processed = []
    for a,b in marked:
        if b >= len(x):
            raise ValueError("marked edge references an unavailable node")
        active = np.any(t == a, axis=1) & np.any(t == b, axis=1)
        if not np.any(active):
            raise ValueError("marked edge has no incident tetrahedra")
        node = len(x)
        midpoint = x[a]*0.5 + x[b]*0.5
        if not np.isfinite(midpoint).all():
            raise ValueError("nonfinite source midpoint")
        x = np.vstack((x, midpoint))
        weights = {}
        for endpoint in (a,b):
            for old, weight in ancestry[endpoint].items():
                weights[old] = weights.get(old,0.0) + weight*0.5
        ancestry.append(weights)
        left = t[active].copy(); right = t[active].copy()
        left[left == b] = node; right[right == a] = node
        t = np.concatenate((t[~active], left, right))
        parent = np.concatenate((parent[~active], parent[active], parent[active]))
        fractions = np.concatenate((fractions[~active], fractions[active]*0.5, fractions[active]*0.5))
        inserted.append(node); processed.append((a,b))
    if not np.all(signed_volumes(x,t) > 0):
        raise ValueError("source refinement produced a nonpositive reference cell")
    totals = np.bincount(parent, weights=fractions, minlength=len(raw))
    if not np.array_equal(totals, np.ones(len(raw))):
        raise ValueError("source-cell refinement measure does not close")
    offsets = [0]; rows = []; weights = []
    for mapping in ancestry:
        for old, weight in sorted(mapping.items()):
            rows.append(old); weights.append(weight)
        offsets.append(len(rows))
    return EdgeRefinement(x,t,parent,fractions,
        np.asarray(offsets,dtype=np.int64),np.asarray(rows,dtype=np.int64),
        np.asarray(weights,dtype=np.float64),np.asarray(inserted,dtype=np.int64),
        np.asarray(processed,dtype=np.int64).reshape(-1,2))

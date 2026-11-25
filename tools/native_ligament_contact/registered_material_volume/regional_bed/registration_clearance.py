"""Linear clearance constraints for two separated triangles during material registration."""

import numpy as np


def clearance_constraints(first, second, first_motion, second_motion):
    """Return nine inequalities on per-vertex motion fractions for each triangle pair.

    The coordinates and proposed displacements have shape (pairs, 3, 3). For vertex
    i of the first triangle and j of the second, the two returned coefficients
    multiply the respective motion fractions. Their sum must stay below the
    returned opening gap. Convexity then keeps both complete triangles on opposite
    sides along their selected axis, including intermediate linear motion.
    """
    p, q, u, v = [np.asarray(a, dtype=float) for a in
                  (first, second, first_motion, second_motion)]
    if p.ndim != 3 or p.shape[1:] != (3, 3) or any(
        a.shape != p.shape or not np.isfinite(a).all() for a in (p, q, u, v)
    ):
        raise ValueError("Expected four finite arrays of matching triangle triples")
    batch = len(p)
    if not batch:
        return np.empty((0, 3, 3, 2)), np.empty((0, 3, 3))
    p, q = p - p[:, :1], q - p[:, :1]
    e = np.roll(p, -1, axis=1) - p
    f = np.roll(q, -1, axis=1) - q
    edge_scale = np.maximum(np.linalg.norm(e, axis=2).max(1), np.linalg.norm(f, axis=2).max(1))
    if np.any(edge_scale == 0):
        raise ValueError("Degenerate triangle pair")
    e, f = e / edge_scale[:, None, None], f / edge_scale[:, None, None]
    n, m = np.cross(e[:, 0], e[:, 1]), np.cross(f[:, 0], f[:, 1])
    if np.any(np.linalg.norm(n, axis=1) == 0) or np.any(np.linalg.norm(m, axis=1) == 0):
        raise ValueError("Degenerate triangle")
    # The in-plane edge normals also separate disjoint coplanar triangles.
    directions = np.concatenate((
        n[:, None], m[:, None],
        np.cross(e[:, :, None], f[:, None, :]).reshape(batch, 9, 3),
        np.cross(n[:, None], e), np.cross(m[:, None], f),
    ), axis=1)
    directions = np.concatenate((directions, -directions), axis=1)
    length = np.linalg.norm(directions, axis=2)
    directions /= np.where(length > 0, length, 1)[:, :, None]
    def projected(a):
        return np.einsum("nvc,nkc->nkv", a, directions)
    ap, bp, au, bu = map(projected, (p, q, u, v))
    gaps = bp[:, :, None, :] - ap[:, :, :, None]
    # Require a strictly separating axis at the initial configuration. Coincident
    # interfaces and pre-existing crossings need an explicit anatomical decision.
    separation = gaps.min(axis=(2, 3))
    valid = (length > 0) & (separation > 0)
    if not valid.any(1).all():
        raise ValueError("No strictly separating axis in the opening pose")
    approach = np.maximum(au, 0)[:, :, :, None] + np.maximum(-bu, 0)[:, :, None, :]
    fractions = np.full(gaps.shape, np.inf)
    np.divide(gaps, approach, out=fractions, where=approach > 0)
    margin = fractions.min(axis=(2, 3))
    best = np.argmax(np.where(valid, margin, -np.inf), axis=1)
    rows = np.arange(batch)
    left = np.broadcast_to(au[rows, best, :, None], (batch, 3, 3))
    right = np.broadcast_to(-bu[rows, best, None, :], (batch, 3, 3))
    return np.stack((left, right), axis=-1), gaps[rows, best]

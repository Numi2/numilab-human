"""Finite triangle-footprint preparation, shared by regional anatomy repairs.

These binary64 constructions carry source ancestry through clipping. They do
not replace the exact emitted-Float32 intersection and containment verifier.
"""
from __future__ import annotations

import numpy as np


def _triangle(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be three finite XYZ points in metres")
    return result


def projected_barycentric(points, tri):
    """Orthogonal projection coordinates in one finite target triangle."""
    points = _triangle(points, "source triangle")
    tri = _triangle(tri, "target triangle")
    v0, v1 = tri[1] - tri[0], tri[2] - tri[0]
    d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
    denominator = d00 * d11 - d01 * d01
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("degenerate finite target triangle")
    relative = points - tri[0]
    d20, d21 = relative @ v0, relative @ v1
    u = (d11 * d20 - d01 * d21) / denominator
    v = (d00 * d21 - d01 * d20) / denominator
    return np.column_stack((1 - u - v, u, v))


def clip_projected_triangle_to_target_bary(points_bary):
    """Return source and target barycentrics at the finite overlap vertices.

    Preserve distinct source points with equal projections, including their
    different normal depths. Each source weight row sums to one.
    """
    projected = np.asarray(points_bary, dtype=np.float64)
    if (projected.shape != (3, 3) or not np.isfinite(projected).all()
            or not np.allclose(projected.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("invalid projected barycentric triangle")
    polygon = [(np.eye(3)[i], projected[i].copy()) for i in range(3)]

    def equivalent(a, b):
        return (np.linalg.norm(a[0] - b[0]) <= 2e-14
                and np.linalg.norm(a[1] - b[1]) <= 2e-14)

    def deduplicate(items):
        result = []
        for item in items:
            if not result or not equivalent(result[-1], item):
                result.append(item)
        if len(result) > 1 and equivalent(result[0], result[-1]):
            result.pop()
        return result

    for axis in range(3):
        if not polygon:
            break
        clipped = []
        previous = polygon[-1]
        previous_value = float(previous[1][axis])
        previous_inside = previous_value >= -1e-12
        for current in polygon:
            value = float(current[1][axis])
            inside = value >= -1e-12
            if previous_inside != inside:
                denominator = previous_value - value
                if denominator == 0:
                    raise ValueError("parallel edge changed clip classification")
                fraction = min(1., max(0., previous_value / denominator))
                clipped.append(tuple(a + fraction * (b - a)
                                     for a, b in zip(previous, current)))
            if inside:
                clipped.append(current)
            previous, previous_value, previous_inside = current, value, inside
        polygon = deduplicate(clipped)
    return polygon


def finite_projected_overlap(source, target):
    """Return source weights, projected target weights and signed metre depths.

    Empty arrays mean no finite projected overlap. Negative depth alone does
    not establish solid penetration; exact geometry remains authoritative.
    """
    source, target = _triangle(source, "source triangle"), _triangle(target, "target triangle")
    polygon = clip_projected_triangle_to_target_bary(projected_barycentric(source, target))
    if not polygon:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    source_weights = np.asarray([item[0] for item in polygon])
    target_weights = np.asarray([item[1] for item in polygon])
    normal = np.cross(target[1] - target[0], target[2] - target[0])
    normal /= np.linalg.norm(normal)
    depths = (source_weights @ source - target[0]) @ normal
    return source_weights, target_weights, depths

"""A trust-region starting direction with an affine normal displacement of the material.

This is an offline search basis. The native tissue, bed and articulated degrees of freedom
remain independent during motion. Reactions on the search bounds do not support the body.
"""

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.spatial import ConvexHull


def near_shape_limits(positions, cells, active, gradient_limit):
    """Keep edges that can cross the existing normalized determinant gate."""
    directions, ratios = [], []
    keep_ratio = (1 + gradient_limit) ** 3 / (1 - gradient_limit)
    for start in range(0, len(cells), 32768):
        batch = cells[start : start + 32768]
        members = active[batch]
        if np.any(members != members[:, :1]):
            raise ValueError("Partial cell in the projected material components")
        points = positions[batch[members[:, 0]]]
        edges = points[:, [1, 2, 3, 2, 3, 3]] - points[:, [0, 0, 0, 1, 1, 2]]
        lengths = np.linalg.norm(edges, axis=2)
        determinant = np.linalg.det(edges[:, :3])
        if np.any(determinant <= 0):
            raise ValueError("Invalid opening volume")
        ratio = determinant[:, None] / (1e-5 * lengths**3)
        valid = (ratio.min(1) >= 1.0)[:, None]
        ids, ends = np.nonzero(valid & (ratio <= keep_ratio * (1 + 1e-12)))
        directions.append(edges[ids, ends] / lengths[ids, ends, None])
        ratios.append(ratio[ids, ends])
    return np.concatenate(directions), np.concatenate(ratios)


def shape_limits(q, edges, ratios, normal, scale):
    # det(I - n g^T) = 1 - n.g. Each normalized edge must obey
    # |(I - n g^T)e|^3 <= det(current) * (1 - n.g) / (h_min |e_source|^3).
    grad = q[1:4] * scale
    moved = edges - (edges @ grad)[:, None] * normal
    lengths = np.linalg.norm(moved, axis=1)
    slack = ratios * (1.0 - grad @ normal) - lengths**3
    slack -= np.minimum(1e-9, np.maximum(0.0, ratios - 1.0) * 0.5)
    derivative = (
        3.0 * (lengths * (moved @ normal))[:, None] * edges - ratios[:, None] * normal
    ) * scale
    jac = np.zeros((len(edges), 5))
    jac[:, 1:4] = derivative
    return slack, jac


def affine_contact_step(
    hessian,
    gradient,
    positions,
    upper,
    radius,
    *,
    normal=None,
    shape=None,
    gradient_limit=0.04,
):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Nonpositive trust radius")
    n = len(positions)
    outer = np.flatnonzero(np.isfinite(upper))
    if not len(outer) or np.any(upper[outer] < 0) or upper.shape != (n,):
        raise ValueError("Missing or infeasible unilateral bed clearances")
    center = positions.mean(axis=0)
    span = np.ptp(positions, axis=0)
    if np.any(span <= 0):
        raise ValueError("Material reference does not span three dimensions")
    P = np.zeros((n + 1, 5))
    P[:n, 0] = 1.0
    P[:n, 1:4] = (positions - center) / span
    P[-1, 4] = 1.0
    H = P.T @ (hessian @ P)
    H = (H + H.T) * 0.5
    g = P.T @ gradient
    force_scale = max(np.max(abs(g)), 1.0)
    Q = H * radius / force_scale
    if np.linalg.eigvalsh(Q).min() < -1e-8 * max(1.0, np.linalg.norm(Q)):
        raise ValueError("Indefinite projected tangent")
    c = g / force_scale
    # Extremes of an affine displacement are attained at hull vertices.
    hull = np.r_[ConvexHull(positions).vertices, n]
    bounds = np.vstack((P[hull], -P[hull]))
    scale = radius / span
    grad_constraints = np.zeros((3, 5))
    grad_constraints[:, 1:4] = np.diag(scale * np.sqrt(3.0) / gradient_limit)
    bounds = np.vstack((bounds, grad_constraints, -grad_constraints))
    shape_edges, shape_ratio = (
        shape if shape is not None else (np.empty((0, 3)), np.empty(0))
    )
    chosen_edges = set()
    chosen = {int(outer[np.argmin(upper[outer])])}
    for iteration in range(60):
        ids = np.array(sorted(chosen))
        A = np.vstack((P[ids], bounds))
        b = np.r_[upper[ids] / radius, np.ones(len(bounds))]
        selected = np.array(sorted(chosen_edges), dtype=int)
        constraints = [dict(type="ineq", fun=lambda q: b - A @ q, jac=lambda q: -A)]
        if len(selected):

            def material(q):
                return shape_limits(
                    q, shape_edges[selected], shape_ratio[selected], normal, scale
                )

            constraints.append(
                dict(
                    type="ineq",
                    fun=lambda q: material(q)[0],
                    jac=lambda q: material(q)[1],
                )
            )
        sol = minimize(
            lambda q: (0.5 * q @ Q @ q + c @ q, Q @ q + c),
            np.zeros(5),
            jac=True,
            constraints=constraints,
            method="SLSQP",
            options=dict(ftol=1e-13, maxiter=120),
        )
        if not sol.success:
            raise ValueError(f"Projected contact search failed: {sol.message}")
        q = sol.x
        trial = radius * (P @ q)
        gap = trial[outer] - upper[outer]
        previous = len(chosen) + len(chosen_edges)
        if gap.max() > radius * 5e-9:
            chosen.add(int(outer[np.argmax(gap)]))
        if len(shape_edges):
            slack, _ = shape_limits(q, shape_edges, shape_ratio, normal, scale)
            bad = np.flatnonzero(slack < -2e-12)
            if len(bad):
                chosen_edges.update(bad[np.argsort(slack[bad])[:16]].tolist())
        if previous == len(chosen) + len(chosen_edges):
            if gap.max() > radius * 5e-9 or (len(shape_edges) and slack.min() < -2e-12):
                raise ValueError("Contact or shape constraints failed to close")
            break
    else:
        raise ValueError("Unfinished projected contact search")
    if np.max(abs(trial)) > radius * (1 + 5e-8):
        raise ValueError("Affine material displacement escaped trust region")
    physical_and_trust = len(b)
    if len(selected):
        material_slack, jac = material(q)
        b = np.r_[b, material_slack - jac @ q]
        A = np.vstack((A, -jac))
    tight = (b - A @ q) < 2e-7
    multipliers, residual = nnls(A[tight].T, -(Q @ q + c), maxiter=10000)
    if residual > 2e-6:
        raise ValueError(f"Unclosed projected stationarity {residual}")
    dual = np.zeros(len(b))
    dual[tight] = multipliers * force_scale
    bed_reaction = np.zeros(n + 1)
    bed_reaction[ids] = dual[: len(ids)]
    history = [
        dict(
            contact_constraint_exchanges=iteration,
            affine_tissue_coordinates=4,
            trust_radius_m=radius,
            trust_reaction_active=bool(
                np.any(dual[len(ids) : physical_and_trust] > 1e-5)
            ),
            peak_search_bound_multiplier_N=float(
                dual[len(ids) : physical_and_trust].max(initial=0.0)
            ),
            active_shape_search_constraints=len(chosen_edges),
            screened_edges=len(shape_edges),
            bed_rows_with_positive_multiplier=int((bed_reaction > 1e-6).sum()),
            projected_stationarity_error_N=float(residual * force_scale),
        )
    ]
    return trial, bed_reaction, history

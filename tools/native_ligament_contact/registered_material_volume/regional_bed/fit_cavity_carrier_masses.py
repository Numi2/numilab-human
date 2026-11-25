"""Fit authoring-time cell masses to an authenticated native carrier graph in two material frames."""

from math import fsum

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def fit_carrier_masses(
    supply,
    allowed,
    preferred,
    volume,
    density_bounds,
    old_mass,
    old_centers,
    new_centers,
):
    """Return feasible child masses, or None if the fixed footprint cannot carry them."""
    supply, preferred, volume, old_mass = (
        np.asarray(v, dtype=float) for v in (supply, preferred, volume, old_mass)
    )
    old_centers = np.asarray(old_centers, dtype=float)
    new_centers = np.asarray(new_centers, dtype=float)
    if any(a.ndim != 1 for a in (supply, preferred, volume, old_mass)):
        raise ValueError("Expected one-dimensional material masses and volumes")
    n, m = len(supply), len(preferred)
    # Source quadrature is authoritative for flow. The independently rounded
    # cell total can differ by several ulps within the native moment tolerance.
    total = fsum(supply)
    lo, hi = density_bounds
    if (
        not n
        or not m
        or len(allowed) != n
        or volume.shape != (m,)
        or old_centers.shape != (2, len(old_mass), 3)
        or new_centers.shape != (2, m, 3)
        or any(
            not np.isfinite(v).all()
            for v in (supply, preferred, volume, old_mass, old_centers, new_centers)
        )
        or any(np.any(v <= 0) for v in (supply, preferred, volume, old_mass))
        or not np.isfinite([lo, hi]).all()
        or not 0 < lo <= hi
        or abs(fsum(old_mass) - total) > 1e-10 * total
    ):
        raise ValueError("Inconsistent source mass, frames or density bounds")
    rows, cols = [], []
    for k, receivers in enumerate(allowed):
        if not len(receivers):
            return None
        if len(set(receivers)) != len(receivers) or any(
            not isinstance(j, (int, np.integer)) or j < 0 or j >= m for j in receivers
        ):
            raise ValueError("Invalid carrier receiver")
        rows.extend([k] * len(receivers))
        cols.extend(receivers)
    rows = np.asarray(rows)
    count = len(rows)
    edge = np.arange(count)
    supply_eq = sparse.coo_matrix((np.ones(count), (rows, edge)), shape=(n, count))
    child_eq = sparse.coo_matrix((supply[rows] / total, (cols, edge)), shape=(m, count))
    # Center and scale the moments so a world-origin translation cannot affect feasibility.
    moment = [np.ones(m)]
    required = [1.0]
    for old, new in zip(old_centers, new_centers, strict=True):
        a, b = old - old[0], new - old[0]
        scale = max(np.abs(a).max(), np.abs(b).max())
        if scale > 0:
            moment.extend(b.T / scale)
            required.extend((old_mass / total) @ (a / scale))
    moment, required = np.asarray(moment), np.asarray(required)
    U, s, V = np.linalg.svd(moment, full_matrices=False)
    active = s > s[0] * 1e-12
    basis = U[:, active]
    if np.max(np.abs(required - basis @ (basis.T @ required))) > 1e-10:
        return None
    I = sparse.eye(m)
    # Unknowns are the fraction of each carrier sent on each edge, child mass fractions,
    # and absolute deviations from the requested child masses.
    eq = sparse.bmat(
        [[supply_eq, None], [child_eq, -I], [None, sparse.csr_matrix(V[active])]],
        format="csr",
    )
    eq = sparse.hstack((eq, sparse.csr_matrix((eq.shape[0], m))), format="csr")
    rhs = np.r_[np.ones(n), np.zeros(m), (basis.T @ required) / s[active]]
    limits = sparse.vstack((sparse.hstack((I, -I)), sparse.hstack((-I, -I))))
    limits = sparse.hstack((sparse.csr_matrix((2 * m, count)), limits), format="csr")
    want = preferred / total
    result = linprog(
        np.r_[np.zeros(count + m), np.ones(m)],
        A_eq=eq,
        b_eq=rhs,
        A_ub=limits,
        b_ub=np.r_[want, -want],
        bounds=[(0, 1)] * count
        + list(zip(lo * volume / total, hi * volume / total))
        + [(0, None)] * m,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    shipment = result.x[:count]
    if np.any(shipment < 0) or np.max(abs(supply_eq @ shipment - 1)) > 1e-10:
        raise ValueError("Carrier marginals failed to close")
    # Recover receiver demand from the shipments, using compensated summation.
    # Close each source before gathering children, keeping every registered edge.
    from_rows = np.r_[0, np.cumsum([len(receivers) for receivers in allowed])]
    by_child = [[] for _ in range(m)]
    for i, receivers in enumerate(allowed):
        share = shipment[from_rows[i] : from_rows[i + 1]]
        amounts = supply[i] * share
        largest = int(np.argmax(amounts))
        amounts[largest] = supply[i] - fsum(np.delete(amounts, largest))
        if np.any(amounts < 0):
            return None
        for receiver, amount in zip(receivers, amounts, strict=True):
            by_child[receiver].append(amount)
    delivered = np.array([fsum(contributions) for contributions in by_child])
    tolerance = 64 * np.finfo(float).eps * hi * volume
    if (
        np.any(delivered < lo * volume - tolerance)
        or np.any(delivered > hi * volume + tolerance)
        or np.max(abs(moment @ (delivered / total) - required)) > 1e-10
    ):
        return None
    return delivered

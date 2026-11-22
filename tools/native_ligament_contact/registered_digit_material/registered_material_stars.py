"""Fit interior points and redistribute existing donor mass on the installed Float32 mesh."""

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from torso_cell_geometry import geom

EDGES = list(itertools.combinations(range(4), 2))


def gradient(points, cells, axis, variables, quantum):
    edges = points[cells[:, 1:]] - points[cells[:, :1]]
    a, b, c = edges.transpose(1, 0, 2)
    grad = (
        np.stack(
            [np.cross(b, c)[:, axis], np.cross(c, a)[:, axis], np.cross(a, b)[:, axis]],
            axis=1,
        )
        / 6
    )
    grad = np.column_stack((-grad.sum(1), grad))
    out = np.zeros((len(cells), len(variables)))
    for i, n in enumerate(variables):
        rr, cc = np.where(cells == n)
        out[rr, i] = grad[rr, cc] * quantum[i]
    return out


def project_mass(states, changed, t, mass, groups, density_limits=(850.0, 950.0)):
    # Preserve each original source allocation group and the first mass
    # moment at all three recorded placements. Scale coordinates about a local
    # origin so conversion from metres to millimetres cannot change admission.
    group_ids, inv = np.unique(groups, return_inverse=True)
    identity = np.eye(len(group_ids))[inv].T
    old = [identity]
    new = [identity]
    for q, p in zip(states, changed):
        u = q[t].mean(axis=1)
        v = p[t].mean(axis=1)
        origin = u.mean(0)
        scale = np.linalg.norm(u - origin, axis=1).max()
        old.append(((u - origin) / scale).T)
        new.append(((v - origin) / scale).T)
    total = mass.sum()
    A = np.vstack(new)
    b = np.vstack(old) @ (mass / total)
    vol = geom(changed[0], t)[0]
    lo, hi = (bound * vol / total for bound in density_limits)
    # Parameterize each tet by its fractional density allowance. A small cell
    # then receives the same numerical bound precision as a large one.
    allowed = hi - lo
    M = A * allowed
    rhs = b - A @ lo
    count = len(group_ids)
    # Allocate each source group exactly. Account for the measured coordinate
    # moments in the original dimensionless frame, then certify them there.
    # Identical loaded and reference rows do not require independent pivots.
    scale = np.max(np.abs(M), axis=1)
    if np.any(scale == 0):
        return None
    B = M / scale[:, None]
    v = rhs / scale
    error = 2e-11 / scale[count:]
    prior = (mass / total - lo) / allowed
    n = len(t)
    eye = np.eye(n)
    C = B[count:]
    zeros = np.zeros_like(C)
    Aub = np.block([[eye, -eye], [-eye, -eye], [C, zeros], [-C, zeros]])
    bub = np.r_[prior, -prior, v[count:] + error, -v[count:] + error]
    sol = linprog(
        np.r_[np.zeros(n), allowed / allowed.max()],
        A_ub=Aub,
        b_ub=bub,
        A_eq=np.c_[B[:count], np.zeros_like(B[:count])],
        b_eq=v[:count],
        bounds=[*[(0, 1)] * n, *[(0, None)] * n],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not sol.success:
        return None
    m = (lo + allowed * sol.x[:n]) * total
    if (
        np.max(abs(A @ (m / total) - b)) > 2e-10
        or np.any(m / vol < density_limits[0] - 1e-6)
        or np.any(m / vol > density_limits[1] + 1e-6)
    ):
        return None
    return m


def response(x, y, t, mu, lam):
    # Displacement gradient gives exactly zero strain for a translation.
    R = x[t[:, 1:]] - x[t[:, :1]]
    u = y - x
    U = u[t[:, 1:]] - u[t[:, :1]]
    D = np.linalg.solve(R, U)
    F = D + np.eye(3)
    vol = np.linalg.det(R) / 6
    lj = np.linalg.slogdet(F)[1]
    FinvT = np.linalg.inv(F).transpose(0, 2, 1)
    # F - F^-T = (D + D^T + D D^T) F^-T.
    stress = (
        mu * (D + D.transpose(0, 2, 1) + D @ D.transpose(0, 2, 1)) @ FinvT
        + lam * lj[:, None, None] * FinvT
    )
    gradient = np.linalg.solve(R.transpose(0, 2, 1), stress) * vol[:, None, None]
    force = np.zeros_like(x)
    np.add.at(force, t[:, 0], gradient.sum(1))
    for i in range(3):
        np.add.at(force, t[:, i + 1], -gradient[:, i])
    energy = vol @ (
        0.5
        * mu
        * (np.sum(D * D, axis=(1, 2)) + 2 * (np.trace(D, axis1=1, axis2=2) - lj))
        + 0.5 * lam * lj**2
    )
    return energy, force


def compare(states, changes, t, properties):
    measured = []
    for j in (1, 2):
        E, F = response(states[0], states[j], t, *properties)
        e, f = response(changes[0], changes[j], t, *properties)
        measured.append(
            {
                "baseline_force_N": float(np.linalg.norm(F)),
                "baseline_energy_J": E,
                "force_error_N": float(np.linalg.norm(F - f)),
                "energy_error_J": abs(E - e),
            }
        )
    return measured


def run_patch(
    states,
    t,
    wrow,
    free,
    mass,
    groups,
    axis,
    reference,
    reserve,
    mu,
    K,
    density_limits=(850.0, 950.0),
):
    # Unknowns are nodal coordinate quanta and fractions of original element mass.
    # solve with unchanged carrier/source allocations, retaining both density limits.
    n = len(free)
    nc = len(t)
    N = n + nc
    quant = np.maximum(
        abs(np.spacing(states[0][free, axis].astype("f4"))),
        abs(np.spacing(states[1][free, axis].astype("f4"))),
    ).astype(float)
    vol = [geom(s, t)[0] for s in states]
    G = [
        gradient(s, t, axis, free, quant) if reference or j else np.zeros((nc, n))
        for j, s in enumerate(states)
    ]
    identity = np.eye(nc)
    zero = np.zeros_like(identity)
    laws = []
    limits = []

    def add(a, b):
        a = np.atleast_2d(a)
        b = np.broadcast_to(b, (len(a),))
        scale = np.max(abs(a), axis=1)
        keep = scale > 0
        if np.any((~keep) & (b < -1e-19)):
            raise ValueError("Fixed condition is invalid")
        laws.append(a[keep] / scale[keep, None])
        limits.extend(b[keep] / scale[keep])

    def ge(a, b):
        add(np.c_[-a, zero], b)

    low, high = density_limits
    add(np.c_[low * G[0], -identity * mass], mass - low * vol[0])
    add(np.c_[-high * G[0], identity * mass], high * vol[0] - mass)
    for phase in (1, 2):
        lo = 0.4 if phase == 1 else 0.35
        ge(G[phase] - lo * G[0], vol[phase] - lo * vol[0])
        ge(2.5 * G[0] - G[phase], 2.5 * vol[0] - vol[phase])
    gid, inv = np.unique(groups, return_inverse=True)
    gs = np.eye(len(gid))[inv].T * mass
    eq = [np.c_[np.zeros((len(gid), n)), gs]]
    eq_rhs = [np.zeros(len(gid))]
    nodeWeights = (t[:, :, None] == free).sum(1) * 0.25
    fixed_moment = mass @ nodeWeights * quant
    # Preserve first moments around the patch. The quadratic term is measured
    # after emission and the fully nonlinear mass equations are re-solved.
    for j, s in enumerate(states):
        c = s[t].mean(1)
        c -= c.mean(0)
        for k in range(3):
            disp = fixed_moment if k == axis and (reference or j) else np.zeros(n)
            row = np.r_[disp, c[:, k] * mass]
            add(row, 3e-14)
            add(-row, 3e-14)
    for j, s in enumerate(states):
        if not reference and j == 0:
            continue
        ge(G[j], vol[j] - 1.000001e-18)
        q = np.full(nc, 1.0000001e-5)
        q[wrow] = reserve
        for a, b in EDGES:
            edge = s[t[:, b]] - s[t[:, a]]
            length = np.linalg.norm(edge, axis=1)
            diff = (
                (t[:, b, None] == free).astype(float) - (t[:, a, None] == free)
            ) * quant
            ge(
                G[j] - 0.5 * (q * length * edge[:, axis])[:, None] * diff,
                vol[j] - q * length**3 / 6,
            )
    Ae = np.vstack(eq)
    Be = np.concatenate(eq_rhs)
    Ae /= np.max(np.abs(Ae), axis=1, keepdims=True)
    I = np.eye(n)
    ub = np.vstack(
        (
            np.c_[np.vstack(laws), np.zeros((len(limits), n))],
            np.c_[I, np.zeros((n, nc)), -I],
            np.c_[-I, np.zeros((n, nc)), -I],
        )
    )
    rhs = np.r_[limits, np.zeros(2 * n)]
    eq = np.c_[Ae, np.zeros((len(Ae), n))]
    cap = np.minimum(32000, np.floor(0.0004 / quant))
    bound = np.r_[cap, np.full(nc, 0.49), cap]
    lb = -bound.copy()
    lb[N:] = 0
    cost = np.r_[np.zeros(N), quant / quant.min()]
    sol = linprog(
        cost,
        A_ub=ub,
        b_ub=rhs,
        A_eq=eq,
        b_eq=Be,
        bounds=np.c_[lb, bound],
        method="highs",
    )
    if not sol.success:
        return {"status": "infeasible"}
    center = sol.x[:n]
    candidates = [np.rint(center)]
    for radius in (1, 4):
        l = lb.copy()
        u = bound.copy()
        l[:n] = np.maximum(l[:n], np.floor(center) - radius)
        u[:n] = np.minimum(u[:n], np.ceil(center) + radius)
        opt = milp(
            cost,
            integrality=np.r_[np.ones(n), np.zeros(nc + n)],
            bounds=Bounds(l, u),
            constraints=[
                LinearConstraint(ub, -np.inf, rhs),
                LinearConstraint(eq, Be, Be),
            ],
            options={"time_limit": 2.0},
        )
        if opt.x is not None:
            candidates.append(np.rint(opt.x[:n]))
            break
    errors = []
    for bins in candidates:
        changes = [s.copy() for s in states]
        move = bins * quant
        for j, s in enumerate(changes):
            if j or reference:
                s[free, axis] += move
            if j < 2:
                s[free, axis] = s[free, axis].astype("f4")
        vq = [geom(s, t) for s in changes]
        qs = [float(q[wrow]) for v, q in vq]
        if min(q.min() for v, q in vq) < 1e-5 or min(v.min() for v, q in vq) <= 1e-18:
            errors.append("shape")
            continue
        if (
            min(vq[1][0] / vq[0][0]) < 0.4
            or min(vq[2][0] / vq[0][0]) < 0.35
            or max(max(v / vq[0][0]) for v, q in vq[1:]) > 2.5
        ):
            errors.append("material")
            continue
        # This second exact projection uses the actual emitted coordinates.
        m = project_mass(states, changes, t, mass, groups, density_limits)
        if m is None:
            errors.append("emitted mass")
            continue
        error = compare(states, changes, t, (mu, K))
        ok = all(
            r["force_error_N"] <= 0.01 * r["baseline_force_N"]
            and r["energy_error_J"] <= 0.05 * abs(r["baseline_energy_J"])
            for r in error
        )
        return {
            "status": "passed" if ok else "force or energy",
            "axis": axis,
            "reference_shift": reference,
            "shift_m": move.tolist(),
            "target_shapes": qs,
            "mechanical_checks": error,
            "new_mass": m.tolist(),
            "peak_m": float(max(abs(move))),
        }
    return {"status": "quantized constraints", "failures": errors}

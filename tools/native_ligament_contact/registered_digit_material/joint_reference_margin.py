"""Fit three coordinate components together in a retained dermal cavity."""

import json
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from registered_density import density_bounds
from registered_material_stars import EDGES, compare, gradient, project_mass
from scipy.optimize import Bounds, LinearConstraint, milp
from torso_cell_geometry import geom


def trial(states, t, free, i, mass, groups, props, limits, goal):
    # Share all three displacement components and all carrier masses in one
    # linearized problem. Recompute volumes and mass on emitted coordinates.
    n = len(free) * 3
    nc = len(t)
    2 * n + nc
    quantum = np.maximum(
        abs(np.spacing(states[0][free].astype("f4"))),
        abs(np.spacing(states[1][free].astype("f4"))),
    ).astype(float)
    mask = t[:, :, None] == free
    lump = mask.sum(1) * 0.25
    v = [geom(q, t)[0] for q in states]
    G = [
        np.stack(
            [gradient(q, t, axis, free, quantum[:, axis]) for axis in range(3)], axis=2
        ).reshape(nc, n)
        for q in states
    ]
    rows = []
    rhs = []

    def le(A, b):
        A = np.atleast_2d(A)
        b = np.broadcast_to(b, (len(A),))
        norm = np.max(abs(A), axis=1)
        active = norm > 0
        if np.any(b[~active] < -1e-19):
            raise ValueError("Infeasible fixed boundary")
        rows.extend(A[active] / norm[active, None])
        rhs.extend(b[active] / norm[active])

    zero = np.zeros((nc, n))
    dm = np.diag(mass)
    empty = np.zeros_like(dm)
    lo, hi = limits
    le(np.c_[lo * G[0], -dm, zero], mass - lo * v[0])
    le(np.c_[-hi * G[0], dm, zero], hi * v[0] - mass)
    I = np.eye(n)
    zc = np.zeros((n, nc))
    le(np.c_[I, zc, -I], 0)
    le(np.c_[-I, zc, -I], 0)

    def ge(D, b):
        le(np.c_[-D, empty, zero], b)

    for k, q in enumerate(states):
        ge(G[k], v[k] - 1.000001e-18)
        rq = np.full(nc, 1.0000001e-5)
        rq[i] = goal
        for a, b in EDGES:
            edge = q[t[:, b]] - q[t[:, a]]
            length = np.linalg.norm(edge, axis=1)
            d = (mask[:, b].astype(float) - mask[:, a].astype(float))[
                :, :, None
            ] * quantum
            e = (d * edge[:, None, :]).reshape(nc, n)
            ge(G[k] - 0.5 * (rq * length)[:, None] * e, v[k] - rq * length**3 / 6)
    gids = np.unique(groups)
    gs = (groups == gids[:, None]) * mass
    eq = np.c_[np.zeros((len(gs), n)), gs, np.zeros((len(gs), n))]
    eq /= np.max(abs(eq), axis=1)[:, None]
    weights = lump.T @ mass
    for q in states:
        centers = q[t].mean(1)
        centers -= centers.mean(0)
        for a in range(3):
            coeff = np.zeros_like(quantum)
            coeff[:, a] = weights * quantum[:, a]
            co = np.r_[coeff.ravel(), centers[:, a] * mass, np.zeros(n)]
            le(co, 1.0e-14)
            le(-co, 1.0e-14)
    cap = np.minimum(2500, np.floor(50e-6 / quantum.ravel()))
    lb = np.r_[-cap, -0.45 * np.ones(nc), np.zeros(n)]
    ub = np.r_[cap, 0.45 * np.ones(nc), cap]
    obj = np.r_[np.zeros(n + nc), quantum.ravel() / quantum.max()]
    solution = milp(
        obj,
        integrality=np.r_[np.ones(n), np.zeros(n + nc)],
        bounds=Bounds(lb, ub),
        constraints=[
            LinearConstraint(np.asarray(rows), -np.inf, np.asarray(rhs)),
            LinearConstraint(eq, 0, 0),
        ],
        options={"time_limit": 8, "mip_rel_gap": 0.005},
    )
    if solution.x is None:
        return {"status": "no integer solution", "reason": solution.message}
    step = np.rint(solution.x[:n]).reshape(-1, 3) * quantum
    new = [q.copy() for q in states]
    for k, q in enumerate(new):
        q[free] += step
        if k < 2:
            q[free] = q[free].astype("f4")
    if not np.array_equal(new[0] - states[0], new[1] - states[1]):
        return {"status": "reference and loaded coordinate changes differ"}
    quality = [geom(q, t) for q in new]
    if any(
        np.min(v) <= 1e-18 or np.min(s) < 1e-5 or s[i] < goal * 0.999
        for v, s in quality
    ):
        return {"status": "emitted shape"}
    m = project_mass(states, new, t, mass, groups, limits)
    if m is None:
        return {"status": "emitted source mass"}
    err = compare(states, new, t, props)
    if any(
        r["force_error_N"] > 0.01 * r["baseline_force_N"]
        or r["energy_error_J"] > 0.05 * abs(r["baseline_energy_J"])
        for r in err
    ):
        return {"status": "regional response", "checks": err}
    return {
        "status": "passed",
        "moves_m": step.tolist(),
        "mass_kg": m.tolist(),
        "region_checks": err,
        "quality": [s[i] for _, s in quality],
        "density_min": float((m / quality[0][0]).min()),
        "density_max": float((m / quality[0][0]).max()),
    }


def run(case, inventory, material, owner, out):
    d = json.loads(inventory.read_text())
    x, y, t, _role, pins = inputs(case)
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if d["source"] != pins or d["mass"] != sha(flow):
        raise ValueError("Different material input")
    with np.load(flow) as f:
        m = f["element_mass_kg"]
        g = f["group_for_cell"]
    z = np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    props, mat = materials(owner)
    bounds = density_bounds(owner, 15)
    results = []
    for e in d["results"]:
        if any(
            r["status"] == "passed"
            for attempt in e["attempts"]
            for r in attempt["trials"]
        ):
            continue
        row = e["source_row"]
        free = np.array(e["free"])
        attempts = []
        for a in e["attempts"]:
            cells = np.array(a["cells"])
            nodes = np.unique(t[cells])
            tt = np.searchsorted(nodes, t[cells])
            n = np.searchsorted(nodes, free)
            i = int(np.flatnonzero(cells == row)[0])
            states = [s[nodes].astype(float) for s in (x, y, z)]
            for goal in (1.01e-5, 1.003e-5):
                f = trial(states, tt, n, i, m[cells], g[cells], props[0], bounds, goal)
                f.update(goal=goal, cells=cells.tolist())
                attempts.append(f)
                print(row, len(cells), goal, f["status"], flush=True)
                if f["status"] == "passed":
                    break
            if f["status"] == "passed":
                break
        results.append({"row": row, "free": free.tolist(), "attempts": attempts})
    out.write_text(
        json.dumps(
            {
                "source": pins,
                "control": sha(inventory),
                "materials": mat,
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))

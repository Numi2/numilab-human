"""Repair the signed determinant and scaled quality of a registered tetrahedral volume."""

import json, sys, time
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix, hstack, eye
from scipy.optimize import linprog
from integrate_reference import cofactors
from transfer_connected_volume import geometry


def constraints(y, cells, orig):
    p = y[cells]
    g, d = cofactors(p)
    u, v = np.array(list(combinations(range(4), 2))).T
    delta = p[:, u] - p[:, v]
    sq = np.einsum("nij,nij->ni", delta, delta)
    k = sq.argmax(1)
    ix = np.arange(len(cells))
    h = np.sqrt(sq[ix, k])
    quality_target = 1.03e-5 * h**3
    base = np.maximum(6.24e-18, 0.12 * orig)
    bound = np.maximum(base, quality_target)
    active = quality_target > base
    dg = 3 * 1.03e-5 * h[:, None] * delta[ix, k] * active[:, None]
    g[ix, u[k]] -= dg
    g[ix, v[k]] += dg
    return bound - d, g, h


def metric(y, t, orig):
    v, q = geometry(y, t)
    bad = (v <= 1e-18) | (q <= 1e-5)
    deficit = np.maximum.reduce(
        (
            np.zeros(len(t)),
            0.12 - v * 6 / orig,
            (1.03e-5 - q) / np.maximum(np.abs(q), 1e-5),
            (1.04e-18 - v) / np.maximum(abs(v), 1e-18),
        )
    )
    return (int(bad.sum()), int((v <= 0).sum()), float(np.clip(deficit, 0, 10).sum()))


def restore(x, y, t, hist, improve, steps):
    ov, oq = geometry(x, t)
    if np.any(ov <= 1e-18) or np.any(oq <= 1e-5):
        raise ValueError("Invalid source tissue reference")
    base = ov * 6
    y = y.astype("f4").astype(float)
    best = metric(y, t, base)
    start = time.monotonic()
    for it in range(steps):
        v, q = geometry(y, t)
        bad = (v <= 1.015e-18) | (q <= 1.015e-5) | (v * 6 < 0.1 * base)
        if not bad.any():
            break
        nodes = np.unique(t[bad])
        lookup = np.full(len(y), -1, dtype=np.int32)
        lookup[nodes] = np.arange(len(nodes))
        mask = np.any(lookup[t] >= 0, axis=1)
        touched = t[mask]
        ids = lookup[touched]
        gap, g, h = constraints(y, touched, base[mask])
        # The update is emitted as world-space Float32. Half an ulp in each
        # coordinate bounds its first-order signed determinant error.
        half_ulp = np.abs(np.spacing(y[touched].astype("f4")).astype(float)) * 0.5
        rounding = np.sum(np.abs(g) * half_ulp, axis=(1, 2))
        movable = ids >= 0
        moving_error = np.sum(np.abs(g) * half_ulp * movable[:, :, None], axis=(1, 2))
        required = gap >= -0.3 * rounding
        gap[required] += 1.1 * moving_error[required]
        # Scale adjustments by the smallest incident material span. The
        # linear programs are independent across disjoint incident stencils.
        reach = np.full(len(nodes) + 1, np.inf)
        np.minimum.at(reach, ids.ravel(), np.repeat(h, 4))
        span = np.clip(0.35 * reach[:-1], 1.5e-6, 5e-3)
        radius = np.r_[span, 0.0][ids]
        A = g * radius[:, :, None]
        norm = np.sum(abs(A), axis=(1, 2))
        keep = (norm > 1e-40) & (gap > -norm * 1.01)
        A = A[keep] / norm[keep, None, None]
        b = gap[keep] / norm[keep]
        ids = ids[keep]
        valid = ids >= 0
        nr, nv = len(b), 3 * len(nodes)
        rr = np.broadcast_to(np.arange(nr)[:, None, None], A.shape)
        cc = 3 * ids[:, :, None] + np.arange(3)
        mat = coo_matrix(
            (A[valid, :].ravel(), (rr[valid, :].ravel(), cc[valid, :].ravel())),
            shape=(nr, nv),
        ).tocsr()
        # A positive slack can be used if linearized rows conflict. The actual
        # geometry decides acceptance after emission at native precision.
        G = hstack((-mat, mat, -eye(nr, format="csr")), format="csr")
        c = np.r_[np.full(2 * nv, 0.0005), np.ones(nr)]
        solution = linprog(
            c,
            A_ub=G,
            b_ub=-b,
            bounds=[(0, 1)] * (2 * nv) + [(0, None)] * nr,
            method="highs",
            options={
                "presolve": True,
                "dual_feasibility_tolerance": 1e-8,
                "primal_feasibility_tolerance": 1e-8,
            },
        )
        if not solution.success:
            raise ArithmeticError(solution.message)
        move = span[:, None] * (solution.x[:nv] - solution.x[nv : 2 * nv]).reshape(
            -1, 3
        )
        # Evaluate one global quality criterion. A support shared with an
        # unaffected tetrahedron has already pulled that cell into touched.
        trialbest = None
        for f in (1.0, 0.75, 0.5, 0.25, 0.125):
            candidate = y.copy()
            candidate[nodes] = (y[nodes] + f * move).astype("f4")
            score = metric(candidate, touched, base[mask])
            rank = (score[0], score[1], score[2])
            if trialbest is None or rank < trialbest[0]:
                trialbest = (rank, f, candidate)
        score, f, candidate = trialbest
        old = metric(y, touched, base[mask])
        if score >= old:
            hist.append([it, "stalled", old, score])
            break
        y = candidate
        best = metric(y, t, base)
        hist.append(
            [
                it,
                len(nodes),
                nr,
                float(f),
                best,
                float(np.max(abs(move))),
                float(solution.x[2 * nv :].sum()),
            ]
        )
        print(
            it,
            best,
            "nodes",
            len(nodes),
            "constraints",
            nr,
            "fraction",
            f,
            "seconds",
            round(time.monotonic() - start, 1),
            flush=True,
        )
    np.save(improve, y.astype("f4"))
    return y


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    p = Path(sys.argv[3])
    history = []
    y = restore(
        z["x"].astype(float),
        np.load(sys.argv[2]).astype(float),
        z["tets"],
        history,
        p.with_suffix(".npy"),
        int(sys.argv[4]),
    )
    p.write_text(json.dumps(history, indent=2) + "\n")

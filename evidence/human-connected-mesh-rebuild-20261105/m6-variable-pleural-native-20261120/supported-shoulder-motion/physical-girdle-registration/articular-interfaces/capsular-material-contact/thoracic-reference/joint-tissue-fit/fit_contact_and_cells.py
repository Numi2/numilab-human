"""One sparse material-coordinate correction for bone contact, layer contact, cell quality and receiver volume."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, diags, eye, hstack, vstack
from scipy.spatial.transform import Rotation
from integrate_reference import cofactors
from restore_emitted import constraints
from transfer_connected_volume import geometry


def project_axes(a, b, ar, br):
    def edges(x):
        return np.roll(x, -1, axis=1) - x

    axes = []
    for u, v in ((ar, br), (a, b)):
        e, f = edges(u), edges(v)
        for el in (e, f):
            axes.append(np.cross(el[:, 0], el[:, 1])[:, None])
        axes.append(np.cross(e[:, :, None, :], f[:, None, :, :]).reshape(-1, 9, 3))
    raw = np.concatenate(axes, axis=1)
    n = raw / np.maximum(np.linalg.norm(raw, axis=2, keepdims=True), 1e-70)
    n = np.concatenate((n, -n), axis=1)

    def separation(p, q, n):
        pa = np.einsum("nvc,nac->nav", p - q[:, :1], n)
        pb = np.einsum("nvc,nac->nav", q - q[:, :1], n)
        return pa.min(2) - pb.max(2)

    src = separation(ar, br, n)
    qualified = (src >= -2e-14) & (np.linalg.norm(n, axis=2) > 0.99)
    sums = (n * qualified[:, :, None]).sum(1)
    norm = np.linalg.norm(sums, axis=1, keepdims=True)
    n = np.concatenate((n, (sums / np.maximum(norm, 1e-70))[:, None]), axis=1)
    src = separation(ar, br, n)
    cur = separation(a, b, n)
    qualified = (src >= -2e-14) & (np.linalg.norm(n, axis=2) > 0.99)
    # Among reference-separating axes choose one that can resolve the current
    # crossing with the least motion. A source gap caps the requested buffer.
    gap = np.minimum(np.maximum(src, 0) * 0.3, 0.00008)
    k = np.argmax(np.where(qualified, cur - gap, -np.inf), axis=1)
    ii = np.arange(len(k))
    return n[ii, k], qualified.any(1)


def contact_rows(z, p, x, y, arms, banks, selfs):
    faces = [p[k] for k in ("l1", "l4294966545", "l4294966546")]
    types = {1: 0, 4294966545: 1, 4294966546: 2}
    blocks = []
    defs = {
        s: (
            Rotation.from_quat(g["rotation_xyzw"]),
            np.asarray(g["pivot_before_m"]),
            np.asarray(g["pivot_after_m"]),
        )
        for g in arms["moved"]
        for s in g["banks"]
    }
    for bank, layer, pairs in banks:
        bb, fa = np.asarray(pairs).T
        ia = faces[types[layer]][fa]
        a = y[ia]
        g = p["f" + str(bank)][bb]
        b = p["b" + str(bank)][g].astype(float)
        r, c, d = defs[bank]
        ar = d + r.apply((x[ia] - c).reshape(-1, 3)).reshape(a.shape)
        n, use = project_axes(a, b, ar, b)
        cur = (
            np.einsum("nvc,nc->nv", a - b[:, :1], n)
            - np.einsum("nvc,nc->nv", b - b[:, :1], n).max(1)[:, None]
        )
        src = (
            np.einsum("nvc,nc->nv", ar - b[:, :1], n)
            - np.einsum("nvc,nc->nv", b - b[:, :1], n).max(1)[:, None]
        )
        delta = np.minimum(np.maximum(src, 0) * 0.25, 0.00008) - cur
        take = use[:, None] & (delta > -0.001)
        blocks.append(
            (
                ia[take],
                np.full(int(take.sum()), -1),
                np.broadcast_to(n[:, None], a.shape)[take],
                delta[take],
            )
        )
    for key, pairs in selfs["witnesses"].items():
        i, j = map(int, key.split("-"))
        f, h = np.asarray(pairs).T
        ia, ib = faces[i][f], faces[j][h]
        a, b = y[ia], y[ib]
        old = np.concatenate((x[ia], x[ib]), axis=1)
        new = np.concatenate((a, b), axis=1)
        old -= old.mean(1, keepdims=True)
        new -= new.mean(1, keepdims=True)
        U, sv, V = np.linalg.svd(old.transpose(0, 2, 1) @ new)
        inv = np.linalg.det(U @ V) < 0
        U[inv, :, -1] *= -1
        ref = old @ (U @ V)
        ar, br = ref[:, :3], ref[:, 3:]
        n, use = project_axes(a, b, ar, br)
        cur = np.einsum("nijc,nc->nij", a[:, :, None] - b[:, None], n)
        sr = np.einsum("nijc,nc->nij", ar[:, :, None] - br[:, None], n)
        delta = np.minimum(np.maximum(sr, 0) * 0.25, 0.00008) - cur
        take = (
            (ia[:, :, None] != ib[:, None, :]) & use[:, None, None] & (delta > -0.001)
        )
        blocks.append(
            (
                np.broadcast_to(ia[:, :, None], take.shape)[take],
                np.broadcast_to(ib[:, None, :], take.shape)[take],
                np.broadcast_to(n[:, None, None], (*take.shape, 3))[take],
                delta[take],
            )
        )
    return [np.concatenate(cols) for cols in zip(*blocks)]


def run(paths):
    material, pack, base, arms, banks, selfs, flow, cap, out = paths
    z = np.load(material)
    p = np.load(pack)
    x = z["x"].astype(float)
    y = np.load(base).astype(float)
    t = z["tets"]
    A, B, N, d = contact_rows(
        z, p, x, y, *[json.loads(f.read_text()) for f in (arms, banks, selfs)]
    )
    selected = np.unique(np.r_[A, B[B >= 0]])
    # Add every original neighboring free material vertex once. Restrict
    # movement to this physical stencil even if another limb is close in space.
    incident = (np.isin(t, selected)).any(1)
    nodes = np.unique(t[incident])
    ids = np.full(len(y) + 1, -1, dtype="i4")
    ids[nodes] = np.arange(len(nodes))
    mov = (ids[t] >= 0).any(1)
    ct = t[mov]
    iv = ids[ct]
    nv = 3 * len(nodes)
    maxstep = 0.0015
    row = np.repeat(np.arange(len(d)), 3)
    axes = np.tile(np.arange(3), len(d))
    valid = B >= 0
    rr = np.r_[row, np.repeat(np.flatnonzero(valid), 3)]
    cc = np.r_[
        np.repeat(3 * ids[A], 3) + axes,
        np.repeat(3 * ids[B[valid]], 3) + np.tile(np.arange(3), valid.sum()),
    ]
    coeff = np.r_[N.ravel(), -N[valid].ravel()]
    C = coo_matrix((coeff, (rr, cc)), shape=(len(d), nv)).tocsr()
    target = d / maxstep
    # Native per-cell quality inequalities include the emitted coordinate
    # rounding bound. Matrix columns use dimensionless trust-region motions.
    v0, _ = geometry(x, t)
    vol, q = geometry(y, t)
    gap, grad, h = constraints(y, ct, 6 * v0[mov])
    ok = iv >= 0
    codec = 3 * iv[:, :, None] + np.arange(3)

    def assemble(block, rows, n):
        rr = np.broadcast_to(rows[:, None, None], block.shape)
        return coo_matrix(
            (block[ok].ravel(), (rr[ok].ravel(), codec[ok].ravel())), shape=(n, nv)
        ).tocsr()

    M = assemble(grad * maxstep, np.arange(len(ct)), len(ct))
    scale = np.asarray(abs(M).sum(1)).ravel()
    keep = (scale > 1e-32) & (gap > -scale)
    half = np.abs(np.spacing(y[ct].astype("f4")).astype(float)) * 0.5
    quant = (abs(grad) * half * ok[:, :, None]).sum((1, 2))
    qb = (gap[keep] + np.where(gap[keep] > -0.3 * quant[keep], quant[keep], 0)) / scale[
        keep
    ]
    G = diags(1 / scale[keep]) @ M[keep]
    # Reuse the current conserved carrier assignment. The final decision
    # always reruns the original native capacity solver.
    ownership = np.load(flow)
    groups = ownership["group_for_cell"]
    mass = np.load(cap)["group_mass"]
    tags = z["tags"]
    low = np.where(tags == 15, 900.0, 850.0)
    high = np.where(tags == 15, 1100.0, 950.0)
    deltaL = np.bincount(groups, weights=low * vol) - mass
    deltaU = mass - np.bincount(groups, weights=high * vol)
    derivative, _ = cofactors(y[ct])
    derivative *= maxstep / 6
    L = assemble(-derivative * low[mov, None, None], groups[mov], len(mass))
    H = assemble(derivative * high[mov, None, None], groups[mov], len(mass))
    V = vstack((L, H), format="csr")
    vs = np.asarray(abs(V).sum(1)).ravel()
    vb = np.r_[deltaL, deltaU]
    inc = (vs > 1e-25) & (vb > -vs)
    V = diags(1 / vs[inc]) @ V[inc]
    vb = vb[inc] / vs[inc]
    rows = vstack([C, G, V], format="csr")
    rhs = np.r_[target, qb, vb]
    cost = np.r_[
        np.full(2 * nv, 0.01),
        np.ones(len(d)),
        np.full(len(qb), 300.0),
        np.full(len(vb), 4.0),
    ]
    mat = hstack([-rows, rows, -eye(len(rhs), format="csr")], format="csr")
    print("nodes", len(nodes), "row counts", len(d), len(qb), len(vb), flush=True)
    sol = linprog(
        cost,
        A_ub=mat,
        b_ub=-rhs,
        bounds=[(0, 1)] * (2 * nv) + [(0, None)] * len(rhs),
        method="highs",
        options=dict(
            time_limit=90.0,
            dual_feasibility_tolerance=1e-8,
            primal_feasibility_tolerance=1e-8,
        ),
    )
    print(sol.message, flush=True)
    if not sol.success:
        return
    move = (sol.x[:nv] - sol.x[nv : 2 * nv]).reshape(-1, 3) * maxstep
    candidates = []
    for frac in (1.0, 0.75, 0.5, 0.25, 0.125):
        yy = y.copy()
        yy[nodes] = (y[nodes] + frac * move).astype("f4")
        v, q = geometry(yy, ct)
        bad = ((v <= 1e-18) | (q <= 1e-5)).sum()
        print(
            "fraction",
            frac,
            "invalid",
            bad,
            "max-mm",
            1000 * np.max(np.linalg.norm(move, axis=1)) * frac,
            flush=True,
        )
        candidates.append((int(bad), -frac, yy.astype("f4")))
    # Save a full-rate proposal to feed exact local restoration, plus the
    # best directly admissible fraction if one exists.
    np.save(out, candidates[0][2])
    best = min(candidates, key=lambda x: x[:2])
    if best[0] == 0:
        np.save(out.with_name(out.stem + "-valid.npy"), best[2])


if __name__ == "__main__":
    run(list(map(Path, sys.argv[1:])))

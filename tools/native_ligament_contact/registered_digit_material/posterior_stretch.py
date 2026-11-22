"""Exact stretch inequalities for affected tetrahedra, with a conservative screening bound."""

import numpy as np


def stretch_envelope(x, y, cells, free, limit_um):
    # F = sum(y_i outer grad N_i). Only the normal coordinate may vary.
    node_to_free = np.full(len(x), -1, dtype=int)
    node_to_free[free] = np.arange(len(free))
    ids = node_to_free[cells]
    D = x[cells[:, 1:]] - x[cells[:, :1]]
    G = np.linalg.inv(D).transpose(0, 2, 1)
    G = np.concatenate((-G.sum(axis=1)[:, None, :], G), axis=1)
    F = np.einsum("nki,nkj->nij", y[cells] - y[cells[:, :1]], G)
    s = np.linalg.svd(F, compute_uv=False)
    radius = np.sum(np.linalg.norm(G, axis=2) * (ids >= 0), axis=1) * limit_um * 1e-6
    at_risk = (radius > 0) & (
        (s[:, 0] + radius >= 1.5 - 1.0e-6) | (s[:, -1] - radius <= 2 / 3 + 1.0e-6)
    )
    F = F[at_risk]
    ids = ids[at_risk]
    G = G[at_risk] * 1.0e-6
    n = len(free)
    max_z = np.max(np.abs(y[cells[at_risk], 2]) + limit_um * 1.0e-6, axis=1)
    gap = (
        2.0e-8
        + np.spacing(max_z.astype(np.float32)).astype(float)
        * np.sum(np.linalg.norm(G, axis=2) * (ids >= 0), axis=1)
        * 1.0e6
    )
    H = np.zeros((len(F), n, 3))
    a, b = np.where(ids >= 0)
    H[a, ids[a, b]] = G[a, b]

    def constraint(v):
        current = F.copy()
        current[:, 2, :] += np.einsum("nkj,k->nj", H, v)
        U, s, V = np.linalg.svd(current)
        deriv = U[:, 2, :, None] * np.einsum("nij,nkj->nik", V, H)
        return (
            np.r_[1.5 - s[:, 0] - gap, s[:, -1] - 2 / 3 - gap],
            np.vstack((-deriv[:, 0], deriv[:, -1])),
        )

    return constraint, np.flatnonzero(at_risk)


def normal_stiffness(x, y, tags, cells, properties, free, rows):
    """Extract only the normal-normal block of the registered material tangent."""
    D = x[cells[:, 1:]] - x[cells[:, :1]]
    Y = y[cells[:, 1:]] - y[cells[:, :1]]
    inverse_D = np.linalg.inv(D)
    F = inverse_D @ Y
    J = np.linalg.det(F)
    if np.any(J <= 0) or not np.isfinite(J).all():
        raise ValueError("Invalid material gradient")
    tail = inverse_D.transpose(0, 2, 1)
    grad = np.concatenate((-tail.sum(axis=1)[:, None, :], tail), axis=1)
    p = np.einsum("nki,ni->nk", grad, np.linalg.inv(F)[:, 2, :])
    mu, bulk = properties[tags - 15].T
    # Normal block of d(-f_internal)/dz for both registered compressible materials.
    coeff = bulk - (bulk * np.log(J) - mu)
    blocks = (
        (
            mu[:, None, None] * (grad @ grad.transpose(0, 2, 1))
            + coeff[:, None, None] * p[:, :, None] * p[:, None, :]
        )
        * np.linalg.det(D)[:, None, None]
        / 6
    )
    rid = np.full(len(x), -1, int)
    rid[rows] = np.arange(len(rows))
    cid = np.full(len(x), -1, int)
    cid[free] = np.arange(len(free))
    ii = np.broadcast_to(rid[cells][:, :, None], blocks.shape)
    jj = np.broadcast_to(cid[cells][:, None, :], blocks.shape)
    valid = (ii >= 0) & (jj >= 0)
    result = np.zeros((len(rows), len(free)))
    np.add.at(result, (ii[valid], jj[valid]), blocks[valid])
    return result

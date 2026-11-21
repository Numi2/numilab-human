"""Offline integration of rigid reference gradients over the existing paired tissue mesh."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from transfer_connected_volume import geometry


def cofactors(points):
    a, b, c = (points[:, 1:] - points[:, :1]).transpose(1, 0, 2)
    ga, gb, gc = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    return np.stack((-ga - gb - gc, ga, gb, gc), axis=1), np.einsum("ij,ij->i", a, ga)


def project(x, tets, tags, weights, group, rot, tr, length):
    n = len(x)
    target = np.zeros_like(x)
    for i, (r, t) in enumerate(zip(rot, tr)):
        target += weights[i, :, None] * (x @ (r - np.eye(3)).T + t)
    original_volume, original_quality = geometry(x, tets)
    if min(original_volume) <= 1e-18 or min(original_quality) <= 1e-5:
        raise ValueError("Reference geometry fails")
    moving = weights.max(0) < 1 - 1e-10
    rounded = (x + target).astype("f4")
    v, q = geometry(rounded, tets)
    # Include any nearly degenerate reference cell upset by Float32 rounding
    # of an otherwise exact rigid motion, and its incident stencil.
    bad = (v <= 1e-18) | (q <= 1e-5)
    moving[tets[bad].ravel()] = True
    influence = moving[tets].any(1)
    tet = tets[influence]
    vol = original_volume[influence]
    idx = -np.ones(n, dtype=np.int64)
    idx[moving] = np.arange(moving.sum())
    nfree = moving.sum()
    rr = []
    cc = []
    dd = []
    rhs = np.zeros((n, 3))
    screen = np.zeros(n)
    for i in range(0, len(tet), 20000):
        cells = tet[i : i + 20000]
        V = vol[i : i + len(cells)]
        g, d = cofactors(x[cells])
        g /= d[:, None, None]
        stiffness = V[:, None, None] * (g @ g.transpose(0, 2, 1))
        # Integrate the rotational part of the prescribed owner motion and
        # let the mesh resolve its incompatible translation across the seam.
        expected = np.einsum(
            "nm,mjk->njk", weights[:, cells].mean(2).T, rot - np.eye(3)
        )
        current = target[cells].transpose(0, 2, 1) @ g
        local = V[:, None, None] * (g @ (expected - current).transpose(0, 2, 1))
        for j in range(3):
            np.add.at(rhs[:, j], cells.ravel(), local[:, :, j].ravel())
        np.add.at(screen, cells.ravel(), np.repeat(V / (4 * length**2), 4))
        packed = idx[cells]
        row, col = np.broadcast_arrays(packed[:, :, None], packed[:, None, :])
        keep = (row >= 0) & (col >= 0)
        rr.append(row[keep].astype("i4"))
        cc.append(col[keep].astype("i4"))
        dd.append(stiffness[keep])
    k = coo_matrix(
        (np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))),
        shape=(nfree, nfree),
    ).tocsc() + diags(screen[moving], format="csc")
    del rr, cc, dd
    normal = 1 / np.sqrt(k.diagonal())
    diag = diags(normal, format="csc")
    factor = splu(
        (diag @ k @ diag).tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0,
        options={"SymmetricMode": True},
    )

    def solve(b):
        return normal[:, None] * factor.solve(normal[:, None] * b)

    correction = solve(rhs[moving])
    y = x + target
    y[moving] += correction
    raw = y.copy()
    print(
        "free",
        nfree,
        "stencil",
        len(tet),
        "operator",
        k.nnz,
        "factor",
        factor.nnz,
        flush=True,
    )
    # Restore the two actual material volumes independently using the same
    # minimum-energy response as the position solve.
    region = tags[influence]
    tag_values = np.unique(tags)
    refs = np.array([original_volume[tags == s].sum() for s in tag_values])
    const = np.array(
        [original_volume[(~influence) & (tags == s)].sum() for s in tag_values]
    )
    # Rigidly transported Dirichlet cells conserve exact volume analytically.
    history = []
    for iteration in range(7):
        gradients = np.zeros((n, 3, len(tag_values)))
        totals = const.copy()
        for i in range(0, len(tet), 24000):
            cells = tet[i : i + 24000]
            g, d = cofactors(y[cells])
            g /= 6
            for tagj, s in enumerate(tag_values):
                mask = region[i : i + len(cells)] == s
                totals[tagj] += d[mask].sum() / 6
                for j in range(3):
                    np.add.at(
                        gradients[:, j, tagj],
                        cells[mask].ravel(),
                        g[mask, :, j].ravel(),
                    )
        gradients = gradients[moving]
        history.append((totals - refs).tolist())
        if max(abs((totals - refs) / refs)) < 1e-10:
            break
        z = solve(gradients.reshape(nfree, -1)).reshape(gradients.shape)
        schur = np.einsum("nja,njb->ab", gradients, z)
        y[moving] += np.einsum("njs,s->nj", z, np.linalg.solve(schur, refs - totals))
    return (
        raw,
        y,
        dict(
            free_nodes=int(nfree),
            changed_cell_stencil=len(tet),
            volume_iterations=history,
            displacement_correction_max_m=np.linalg.norm(
                y - (x + target), axis=1
            ).max(),
            volume_reference_m3=refs.tolist(),
        ),
    )


def measure(y, t, tags, ref):
    v, q = geometry(y.astype("f4"), t)
    return [
        dict(
            material=int(tag),
            inverted=int(np.count_nonzero(v[tags == tag] <= 0)),
            low_volume=int(np.count_nonzero(v[tags == tag] <= 1e-18)),
            low_quality=int(np.count_nonzero(q[tags == tag] <= 1e-5)),
            minimum_volume_m3=float(v[tags == tag].min()),
            min_quality=float(q[tags == tag].min()),
            volume_relative_error=v[tags == tag].sum() / ref[j] - 1,
        )
        for j, tag in enumerate(np.unique(tags))
    ]


def main(inp, length, out):
    z = np.load(inp)
    x = z["x"].astype(float)
    t = z["tets"]
    tags = z["tags"]
    start = time.monotonic()
    raw, y, result = project(x, t, tags, z["w"], z["group"], z["rot"], z["tr"], length)
    refs = result["volume_reference_m3"]
    result.update(
        uncorrected_volumes=measure(raw, t, tags, refs),
        candidate=measure(y, t, tags, refs),
        length_m=length,
        elapsed_authoring_s=time.monotonic() - start,
    )
    print(result["candidate"], flush=True)
    np.save(out.with_suffix(".npy"), y.astype("f4"))
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]))

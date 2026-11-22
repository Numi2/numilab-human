"""Enforce the installed material's shape and seated-stretch gates together over a connected patch."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu
from prepare_connected_pose import read, sha, validate
from fit_material_gradients import unilateral_z
from fit_dorsal_material import inputs
from torso_cell_geometry import geom


def metric(ref, y, tet):
    E = ref[tet[:, 1:]] - ref[tet[:, 0, None]]
    e = y[tet[:, 1:]] - y[tet[:, 0, None]]
    F = np.linalg.solve(E, e)
    s = np.linalg.svd(F, compute_uv=False)
    v, shape = geom(y, tet)
    J = np.linalg.det(F)
    grade = np.minimum.reduce(
        [
            s[:, -1] / (2 / 3),
            1.5 / np.maximum(s[:, 0], 1e-20),
            shape / 1e-5,
            v / 1e-18,
            J / 0.4,
            2.5 / np.maximum(J, 1e-20),
        ]
    )
    return grade


def all_metrics(ref, y, cells):
    return np.concatenate(
        [metric(ref, y, cells[i : i + 49152]) for i in range(0, len(cells), 49152)]
    )


def proposal(ref, initial, y, tet, score, fixed, outer):
    bad = score < 1.0
    move = np.setdiff1d(np.unique(tet[bad]), fixed)
    local = np.full(len(y), -1, dtype=int)
    local[move] = np.arange(len(move))
    t = tet[(local[tet] >= 0).any(axis=1)]
    # Only the implicated star contributes. A fixed exterior preserves the
    # previous fit exactly. Slack inside the original gates protects against
    # Float32 rounding on small-altitude elements.
    K = csr_matrix((len(move), len(move)))
    rhs = np.zeros((len(move), 3))
    near = np.zeros(len(move))
    for i in range(0, len(t), 24576):
        c = t[i : i + 24576]
        l = local[c]
        witness = metric(ref, y, c)
        E = initial[c[:, 1:]] - initial[c[:, 0, None]]
        e = y[c[:, 1:]] - y[c[:, 0, None]]
        g = np.empty((len(c), 4, 3))
        g[:, 1:] = np.linalg.inv(E).transpose(0, 2, 1)
        g[:, 0] = -g[:, 1:].sum(axis=1)
        cur = np.linalg.solve(E, e)
        u, s, v = np.linalg.svd(cur)
        u[:, :, -1] *= np.sign(np.linalg.det(u @ v))[:, None]
        target = u @ v
        # The closest proper rotation of the good installed cell is feasible
        # independent of aspect ratio. Only cells near a physical gate pull
        # to that target. All other star cells resist new differential strain.
        beta = np.clip((1.10 - witness) / 0.10, 0, 1)
        delta = beta[:, None, None] * (target - cur)
        maxedge = np.maximum((E * E).sum(2).max(1), 1e-16) ** 1.5
        volume = np.linalg.det(E) / 6
        w = np.maximum(volume, 0.02 * maxedge) * (1 + 8 * beta)
        normal = (g @ g.transpose(0, 2, 1)) * w[:, None, None]
        ii = np.broadcast_to(l[:, :, None], normal.shape)
        jj = np.broadcast_to(l[:, None, :], normal.shape)
        ok = (ii >= 0) & (jj >= 0)
        K += csr_matrix((normal[ok], (ii[ok], jj[ok])), shape=K.shape)
        load = w[:, None, None] * (g @ delta)
        for j in range(4):
            active = l[:, j] >= 0
            idx = l[active, j]
            np.add.at(rhs, idx, load[active, j])
            np.add.at(near, idx, maxedge[active])
    # Small dimensional screen removes rigid null spaces if a patch encloses
    # an entire disconnected component. Its length is 3 cm as in the seed.
    K += diags(near / 0.03**2 * 0.001)
    f = splu(
        K.tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    trial = y.copy()
    trial[move] += f.solve(rhs)
    trial[:, 2], _ = unilateral_z(f, move, trial[:, 2], outer)
    return trial, len(move), len(t)


def fit(fine, bed, layers, seed, out, budget):
    out.mkdir(exist_ok=False)
    ref, start, tet, roles, pins = inputs(fine)
    ref = ref.astype(float)
    initial = start.astype(float)
    identity = read(seed / "quality.json")
    if (
        sha(seed / "positions.npy") != identity["candidate_sha256"]
        or identity["source"] != pins
    ):
        raise ValueError("Stale seed")
    anchors = np.array([r["volume_row"] for r in read(bed)["attachment_rows"]])
    outer = np.load(layers)["Orows"]
    current = np.load(seed / "positions.npy").astype(float)
    hard = current[anchors].copy()
    score = all_metrics(ref, current, tet)
    history = []
    for iteration in range(budget):
        old = int((score < 1).sum())
        if not old:
            break
        trial, free, count = proposal(ref, initial, current, tet, score, anchors, outer)
        success = False
        for factor in (1.0, 0.5, 0.25, 0.125):
            y = (current + factor * (trial - current)).astype("<f4").astype(float)
            if np.any(y[outer, 2] < 0) or not np.array_equal(y[anchors], hard):
                raise ValueError("Broken bed/attachment")
            s = all_metrics(ref, y, tet)
            fail = int((s < 1).sum())
            if fail < old:
                history.append(
                    dict(
                        iteration=iteration + 1,
                        factor=factor,
                        movable=free,
                        affected=count,
                        before=old,
                        after=fail,
                    )
                )
                print(history[-1], flush=True)
                current = y
                score = s
                success = True
                break
        if not success:
            history.append(dict(iteration=iteration + 1, stalled=old))
            break
    result = dict(
        source=pins,
        seed_sha256=identity["candidate_sha256"],
        iterations=history,
        owner_all_gate_failures=int((score < 1).sum()),
        owner_failures_by_role={
            str(i): int(np.sum((score < 1) & (roles == i))) for i in (15, 16)
        },
        bed=dict(below=int(np.sum(current[outer, 2] < 0)), outer_nodes=len(outer)),
        hard_attachment_count=len(anchors),
        max_change_at_attachment_m=float(np.max(abs(current[anchors] - hard))),
        cells=validate(ref, current, tet, roles),
        scope="Offline all-tetrahedron Float32 reconstruction at the full registered attachment target. Native acceptance and anatomical crossings require independent checks.",
    )
    np.save(out / "positions.npy", current.astype("<f4"))
    result["candidate_sha256"] = sha(out / "positions.npy")
    (out / "quality.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    fit(*(Path(s).resolve() for s in sys.argv[1:6]), int(sys.argv[6]))

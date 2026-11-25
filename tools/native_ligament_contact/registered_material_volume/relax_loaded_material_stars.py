"""Relax connected registration errors directly on the native coordinate lattice.

Every trial moves one source node and evaluates all attached elements. The
reference lattice, material tables, all cortical binding facets, and protected
attachment nodes are fixed. The resulting field initializes the physical model.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


EDGE = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def errors(s):
    return (
        np.maximum(np.log(s[..., 0] / 1.5), 0) ** 2
        + np.maximum(np.log((2 / 3) / np.maximum(s[..., -1], 1e-20)), 0) ** 2
    )


def relax(opt):
    scene = json.loads(opt.volume_launch.read_bytes())
    registration = json.loads(opt.material_registration.read_bytes())
    if digest(opt.volume_launch) != registration["source_launch"]:
        raise ValueError("Unmatched source volume launch")
    cfg = scene["environment"]
    pre = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def pinned(key):
        path = Path(cfg[key])
        proof = [scene["pins"].get(str(path))]
        if key.endswith("_CSV"):
            proof.append(cfg.get(key.removesuffix("_CSV") + "_SHA256"))
        expected = {value for value in proof if value}
        if expected != {digest(path)}:
            raise ValueError(f"Unmatched material asset {key}")
        return path

    tet = pinned(pre + "TETGEN_ELEMENTS")
    ref = pinned(pre + "TETGEN_NODE")
    current = pinned("NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE")
    walls = tet.parent / "material-walls.npz"
    if (
        digest(walls) != registration["material_wall_sha256"]
        or digest(current) != registration["loaded_nodes_sha256"]
    ):
        raise ValueError("Different loaded material registration")

    def positions(path):
        data = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2, 3))
        if (
            not np.array_equal(data[:, 0], np.arange(1, len(data) + 1))
            or not np.isfinite(data).all()
        ):
            raise ValueError("Invalid registered node rows")
        return np.ascontiguousarray(data[:, 1:])

    ysrc = positions(current)
    X = positions(ref).astype("f4").astype(float)
    elem = np.loadtxt(tet, dtype=np.int64, skiprows=1, usecols=(0, 1, 2, 3, 4))
    if not np.array_equal(elem[:, 0], np.arange(1, len(elem) + 1)):
        raise ValueError("Invalid registered tetrahedron rows")
    t = elem[:, 1:] - 1
    if ysrc.shape != X.shape or t.min() < 0 or t.max() >= len(X):
        raise ValueError("Inconsistent material topology")
    y0 = ysrc.astype("f4").astype(float)
    y = y0.copy()
    failed = []
    for start in range(0, len(t), 32768):
        cells = t[start : start + 32768]
        xlocal = X[cells]
        ylocal = y[cells]
        source = xlocal[:, 1:] - xlocal[:, :1]
        if (np.linalg.det(source) <= 0).any():
            raise ValueError("Invalid stress-free material tetrahedron")
        response = np.linalg.solve(source, ylocal[:, 1:] - ylocal[:, :1])
        if (np.linalg.det(response) <= 0).any():
            raise ValueError("The loaded material contains an inverted tetrahedron")
        s = np.linalg.svd(response, compute_uv=False)
        failed.extend(start + np.flatnonzero((s[:, 0] > 1.5) | (s[:, -1] < 2 / 3)))
    bad_source = np.array(failed, dtype=int)
    # Preserve the actual cortical facets and mapped deep-attachment identities.
    blocked = np.zeros(len(X), bool)
    blocked[np.unique(np.load(walls)["C"])] = True
    with pinned(pre + "DEEP_ATTACHMENT_REQUESTS_CSV").open() as f:
        source_ids = {int(a["stable_source_vertex"]) for a in csv.DictReader(f)}
    with pinned(pre + "COMBINED_NODE_MAP_CSV").open() as f:
        for rec in csv.DictReader(f):
            raw = rec["source_vertex_row"]
            if raw and int(raw) in source_ids:
                blocked[int(rec["joint_node_id"]) - 1] = True
    protected = np.load(opt.protected_nodes)
    if (
        protected.ndim != 1
        or not np.issubdtype(protected.dtype, np.integer)
        or np.any((protected < 0) | (protected >= len(y)))
    ):
        raise ValueError("Invalid protected material row")
    blocked[protected] = True
    eligible = np.zeros(len(y), bool)
    eligible[np.unique(t[bad_source])] = True
    eligible &= ~blocked
    rows = np.flatnonzero(eligible[t].any(1))
    c = t[rows]
    cc = len(c)
    ii = np.repeat(np.arange(cc), 4)
    corner = np.tile(np.arange(4), cc)
    stars = csr_matrix((ii * 4 + corner + 1, (c.ravel(), ii)), shape=(len(X), cc))
    sx = X[c]
    E = sx[:, 1:] - sx[:, :1]
    inv = np.linalg.inv(E)
    vol0 = np.linalg.det(E) / 6
    grads = np.concatenate((-inv.sum(axis=2)[:, :, None], inv), axis=2)

    def initial_state():
        verts = y[c]
        return inv @ (verts[:, 1:] - verts[:, :1])

    F = initial_state()
    start_error = errors(np.linalg.svd(F, compute_uv=False))
    bad = start_error > 0
    before = int(bad.sum())
    history = []
    charts = {}
    for side in [-1, 1]:
        ds = (y0 - X)[
            (np.sign(X[:, 0]) == side) & (np.linalg.norm(y0 - X, axis=1) > 0.01)
        ]
        mode = np.median(ds, axis=0) if len(ds) else np.zeros(3)
        charts[side] = np.round(
            mode / np.array([2.0**-24, 2.0**-22, 2.0**-24])
        ) * np.array([2.0**-24, 2.0**-22, 2.0**-24])
    for it in range(opt.iterations):
        s = np.linalg.svd(F, compute_uv=False)
        global_err = errors(s)
        if not np.any(global_err > 0):
            break
        ranks = np.argsort(-global_err, kind="stable")
        nodes = c[ranks[global_err[ranks] > 0]].ravel()
        _, order = np.unique(nodes, return_index=True)
        nodes = nodes[np.sort(order)]
        nodes = nodes[eligible[nodes]]
        accepted = 0
        for node in nodes:
            indices = stars.data[stars.indptr[node] : stars.indptr[node + 1]] - 1
            cells = indices // 4
            corners = indices % 4
            local = F[cells]
            u, ss, vh = np.linalg.svd(local)
            loss = errors(ss)
            needs = loss > 0
            if not np.any(needs):
                continue
            gv = np.take_along_axis(grads[cells], corners[:, None, None], axis=2)[
                :, :, 0
            ]
            norm = np.einsum("ij,ij->i", gv, gv)
            # Project a violated deformation gradient toward the existing
            # principal stretch interval, then solve its rank-one nodal update.
            target = (u * np.clip(ss, 0.73, 1.38)[:, None, :]) @ vh
            proposal = np.einsum("ij,ijk->ik", gv, target - local) / norm[:, None]
            sel = np.flatnonzero(needs)
            sel = sel[np.argsort(-loss[sel])[:8]]
            trial = [
                *proposal[sel],
                np.average(proposal[sel], axis=0, weights=loss[sel]),
            ]
            side = 1 if X[node, 0] > 0 else -1
            chart = X[node] + charts[side] - y[node]
            if np.linalg.norm(chart) < 0.001:
                trial.append(chart)
            trial = np.asarray(trial)
            ds = (
                trial[:, None, :] * np.array([0.125, 0.25, 0.5, 0.75, 1, 1.35])[:, None]
            ).reshape(-1, 3)
            # Include nearest grid neighbors when a continuous optimum rounds to zero.
            axis = []
            for k in range(3):
                for where in (-np.inf, np.inf):
                    h = np.zeros(3)
                    h[k] = (
                        float(np.nextafter(np.float32(y[node, k]), np.float32(where)))
                        - y[node, k]
                    )
                    axis.append(h)
            offsets = np.asarray(axis).reshape(3, 2, 3)
            offsets = np.concatenate((np.zeros((3, 1, 3)), offsets), axis=1)
            grid = (
                offsets[0, :, None, None]
                + offsets[1, None, :, None]
                + offsets[2, None, None, :]
            ).reshape(-1, 3)
            ds = np.vstack((ds, grid))
            loc = (y[node] + ds).astype("f4").astype(float)
            loc = np.unique(loc, axis=0)
            loc = loc[np.linalg.norm(loc - y0[node], axis=1) < opt.max_mm * 0.001]
            if not len(loc):
                continue
            ds = loc - y[node]
            new = local[None] + gv[None, :, :, None] * ds[:, None, None, :]
            sig = np.linalg.svd(new, compute_uv=False)
            next_loss = errors(sig)
            J = np.linalg.det(new)
            vol = J * vol0[cells]
            q = y[c[cells]]
            edge = q[:, EDGE[:, 1]] - q[:, EDGE[:, 0]]
            coef = (EDGE[None, :, 1] == corners[:, None]).astype(float) - (
                EDGE[None, :, 0] == corners[:, None]
            )
            dist = np.linalg.norm(
                edge[None] + coef[None, :, :, None] * ds[:, None, None, :], axis=3
            ).max(2)
            shape = 6 * vol / (dist**3)
            ok = (
                (J >= 0.4)
                & (J <= 2.5)
                & (vol > 1e-18)
                & (shape >= 1.000000001e-5)
                & ((next_loss == 0) | needs)
            ).all(1)
            score = next_loss.sum(1)
            score[~ok] = np.inf
            k = int(score.argmin())
            if score[k] >= loss.sum() * (1 - 1e-7):
                continue
            y[node] = loc[k]
            F[cells] = new[k]
            accepted += 1
        F = initial_state()
        sv = np.linalg.svd(F, compute_uv=False)
        err = errors(sv)
        record = {
            "sweep": it,
            "accepted": accepted,
            "violating": int((err > 0).sum()),
            "max_stretch": float(sv[:, 0].max()),
            "min_stretch": float(sv[:, -1].min()),
            "loss": float(err.sum()),
        }
        print(record, flush=True)
        history.append(record)
        if accepted == 0 or not (err > 0).any():
            break
    ids = np.flatnonzero(np.any(y != y0, axis=1))
    if blocked[ids].any():
        raise ValueError("Changed a protected anatomical node")
    np.savez_compressed(
        opt.out.with_suffix(".npz"), node_rows=ids, delta_m=y[ids] - ysrc[ids]
    )
    data = {
        "source_launch_sha256": digest(opt.volume_launch),
        "material_registration_sha256": digest(opt.material_registration),
        "original_stretch_failures": len(bad_source),
        "local_before": before,
        "eligible_nodes": int(eligible.sum()),
        "checked_cells": len(c),
        "locked": len(np.flatnonzero(blocked)),
        "moved": len(ids),
        "max_displacement_mm": float(
            np.linalg.norm(y[ids] - y0[ids], axis=1).max() * 1000
        )
        if len(ids)
        else 0,
        "reference": digest(ref),
        "nodes": digest(current),
        "cells": digest(tet),
        "protected_nodes": digest(opt.protected_nodes),
        "sweeps": history,
    }
    opt.out.with_suffix(".json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume-launch", required=True, type=Path)
    p.add_argument("--material-registration", required=True, type=Path)
    p.add_argument(
        "--protected-nodes",
        required=True,
        type=Path,
        help="Protected source-node rows for this registration",
    )
    p.add_argument("--iterations", type=int, default=28)
    p.add_argument("--max-mm", type=float, default=3.0)
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Base filename for the sparse material proposal and JSON observations",
    )
    o = p.parse_args()
    if o.iterations < 1 or not np.isfinite(o.max_mm) or o.max_mm <= 0:
        p.error("Iterations and displacement limit must be positive")
    relax(o)

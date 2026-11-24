"""Fit a source muscle against all of its registered native neighbors using exact intersection-volume derivatives."""

import json
import sys
from pathlib import Path

import numpy as np
from assemble_material import frames
from compile_connected_joint import digest
from manifold3d import Error, Manifold, Mesh64
from registration_stages import parts
from repair_registered_surface import per_pose, signed_volume_gradient, state, volume
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


def solid(x, f, oid=None):
    extra = (
        {}
        if oid is None
        else dict(
            run_index=np.array([0, 3 * len(f)], "u8"),
            run_original_id=np.array([oid], "u4"),
            face_id=np.arange(len(f), dtype="u8"),
        )
    )
    m = Mesh64(np.ascontiguousarray(x), np.ascontiguousarray(f, dtype="u8"), **extra)
    s = Manifold(m)
    if s.status() != Error.NoError:
        m.merge()
        s = Manifold(m)
    if s.status() != Error.NoError:
        raise ValueError(s.status())
    return s


def overlap(x, f, body, oid, obstacles):
    """Apply the boundary integral to each clipped original triangle. All coordinates are mm."""
    volumes = []
    gradients = []
    for pieces in obstacles:
        vol = 0.0
        g = np.zeros_like(x)
        for s in pieces:
            cut = body ^ s
            if cut.status() != Error.NoError:
                raise ValueError(cut.status())
            vol += cut.volume()
            m = cut.to_mesh64()
            for j, id_ in enumerate(m.run_original_id):
                if id_ != oid:
                    continue
                first, end = np.array(m.run_index[j : j + 2]) // 3
                idx = np.asarray(m.face_id, dtype=np.int64)[first:end]
                sub = np.asarray(m.vert_properties)[
                    np.asarray(m.tri_verts)[first:end], :3
                ]
                face = f[idx]
                tri = x[face]
                a = tri[:, 1] - tri[:, 0]
                b = tri[:, 2] - tri[:, 0]
                n = np.cross(a, b)
                squared = (n * n).sum(1)
                q = sub.mean(1) - tri[:, 0]
                bary1 = (np.cross(q, b) * n).sum(1) / squared
                bary2 = (np.cross(a, q) * n).sum(1) / squared
                bary = np.column_stack((1 - bary1 - bary2, bary1, bary2))
                av = np.cross(sub[:, 1] - sub[:, 0], sub[:, 2] - sub[:, 0]) * 0.5
                for k in range(3):
                    np.add.at(g, face[:, k], av * bary[:, k, None])
        volumes.append(vol)
        gradients.append(g)
    return np.array(volumes), np.array(gradients)


def fit(w, case, folder, reference):
    z = np.load(folder / "geometry.npz")
    meta = json.loads((folder / "geometry.json").read_bytes())
    if digest(folder / "geometry.npz") != meta["geometry"]:
        raise ValueError("Changed native obstacle set")
    x = z["xyz"]
    f = np.ascontiguousarray(z["faces"])
    pins = z["pins"]
    origin = z["origin"]
    names = list(meta["solids"])
    obs = [
        [
            solid(z[f"{name}_{i}_x"], z[f"{name}_{i}_f"])
            for i in range(meta["solids"][name])
        ]
        for name in names
    ]
    oid = Manifold.reserve_ids(1)

    def evaluate(y):
        return overlap(y, f, solid(y, f, oid), oid, obs)

    bed = json.loads((case / "support-check.json").read_bytes())
    normal = np.asarray(bed["bed_normal"])
    zero = (bed["source_plane_offset_m"] - origin @ normal) * 1000

    def clearance(y):
        return y @ normal - zero

    original, g = evaluate(x)
    controls = json.loads(reference.read_bytes())["changes"]
    by_name = {"-".join(map(str, v["target"])): v["old_mm3"] for v in controls}
    if max(abs(v - by_name.get(k, 0.0)) for k, v in zip(names, original)) > 1e-3:
        raise ValueError("Cached solids disagree with full-scene audit")
    cfg = json.loads((case / "launch.json").read_bytes())
    args = cfg["argv"]
    src = Path(args[args.index("--soft-tissue-payload") + 1])
    v, wt, b, tri, aff = parts(src, 83)
    source, ii, inv = np.unique(v, axis=0, return_index=True, return_inverse=True)
    poses = frames(case)
    weights = wt[ii]
    if not np.array_equal(source, z["source"]) or not np.array_equal(inv[tri], f):
        raise ValueError("Wrong muscle donor")
    matrix = np.einsum(
        "vb,bij->vij",
        weights,
        [
            Rotation.from_quat(poses[-1][k]["quaternion_xyzw"]).as_matrix() @ M
            for k, (M, t) in zip(b, aff)
        ],
    )

    def physical(y):
        v = (
            (source + np.linalg.solve(matrix, (y - x)[:, :, None] * 0.001)[:, :, 0])
            .astype("f4")
            .astype(float)
        )
        return v, x + np.einsum("vij,vj->vi", matrix, v - source) * 1000

    baseline = [source] + [state(source, weights, aff, b, p) for p in poses]
    edge = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0
    )
    old_length = np.linalg.norm(x[edge[:, 0]] - x[edge[:, 1]], axis=1)
    # Compact smooth coordinates. The cardinal correction enforces exact tendon interpolation.
    dist = cdist(x, x)
    selected = [int(np.argmax(np.linalg.norm(x - x.mean(0), axis=1)))]
    while dist[:, selected].min(1).max() > 12:
        selected.append(int(np.argmax(dist[:, selected].min(1))))
    K = np.exp(-0.5 * (dist / 16) ** 2)
    B = K[:, selected] - K[:, pins] @ np.linalg.solve(
        K[np.ix_(pins, pins)], K[np.ix_(pins, selected)]
    )
    B[pins] = 0
    Q, sigma, _ = np.linalg.svd(B, full_matrices=False)
    B = Q[:, sigma > 1e-5] * np.sqrt(len(x))
    K = B.shape[1]

    def projected(G):
        return np.einsum("nk,cnj->ckj", B, G).reshape(len(G), -1)

    def objective(V):
        return np.sum(V)

    # Derivative parity at the actual measured body configuration.
    rng = np.random.default_rng(83153)
    direction = B @ rng.normal(size=(K, 3))
    direction /= np.linalg.norm(direction, axis=1).max()
    ep = 0.001
    vp, _ = evaluate(x + ep * direction)
    vm, _ = evaluate(x - ep * direction)
    fd = (vp - vm) / (2 * ep)
    analytic = (g * direction).sum((1, 2))
    if np.max(abs(fd - analytic)) / max(1, np.max(abs(fd))) > 2e-3:
        raise ValueError(("Contact derivative mismatch", np.max(abs(fd - analytic))))
    hist = []
    y = x.copy()
    V = original.copy()
    src_out = source.copy()
    for iteration in range(40):
        A = projected(g)
        vol_grad = projected(signed_volume_gradient(y, f)[None])[0]
        active = np.flatnonzero(V > 1e-7)
        scale = np.maximum(np.linalg.norm(A[active], axis=1), 1)
        # Keep every interacting neighbor below its independently measured control,
        # progressively reduce the largest overlaps, and preserve posed volume to first order.
        limits = 0.88 * (original[active] - V[active]) / scale
        lin = A[active] / scale[:, None]
        w = np.maximum(V[active], 10.0) ** -0.35
        drive = (A[active] * w[:, None]).sum(0)
        drive /= max(np.linalg.norm(drive), 1e-12)
        Vg = vol_grad / np.linalg.norm(vol_grad)
        # Linearize nearly active stretch constraints so search can redistribute deformation.
        edge_vec = y[edge[:, 0]] - y[edge[:, 1]]
        edge_len = np.linalg.norm(edge_vec, axis=1)
        ratio = edge_len / old_length
        near = (ratio < 0.85) | (ratio > 1.16)
        E = (
            (
                (B[edge[near, 0]] - B[edge[near, 1]])[:, :, None]
                * (edge_vec[near] / edge_len[near, None])[:, None, :]
            ).reshape(near.sum(), 3 * K)
            if near.any()
            else np.empty((0, 3 * K))
        )
        shift = y - x
        length = np.linalg.norm(shift, axis=1)
        hot = length > 7.0
        D = (
            (B[hot, :, None] * (shift[hot] / length[hot, None])[:, None, :]).reshape(
                hot.sum(), 3 * K
            )
            if hot.any()
            else np.empty((0, 3 * K))
        )
        lin = np.vstack([lin, -E, E, D])
        limits = np.r_[
            limits,
            edge_len[near] - 0.7505 * old_length[near],
            1.2495 * old_length[near] - edge_len[near],
            8.99 - length[hot],
        ]
        gap = clearance(y)
        on = gap < 4.0
        bed_rows = -(B[on, :, None] * normal[None, None, :]).reshape(on.sum(), 3 * K)
        lin = np.vstack([lin, bed_rows])
        limits = np.r_[limits, gap[on] - 0.001]
        opt = minimize(
            lambda c: 0.5 * (c @ c) + drive @ c,
            np.zeros(3 * K),
            jac=lambda c: c + drive,
            constraints=[
                dict(type="ineq", fun=lambda c: limits - lin @ c, jac=lambda c: -lin),
                dict(type="eq", fun=lambda c: Vg @ c, jac=lambda c: Vg),
            ],
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 100},
        )
        if not opt.success or np.linalg.norm(opt.x) < 1e-8:
            break
        move = B @ opt.x.reshape(K, 3)
        move *= min(1.5 / max(np.linalg.norm(move, axis=1).max(), 1e-9), 1)
        accepted = False
        for h in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            candidate, Y = physical(y + h * move)
            d = np.linalg.norm(Y[edge[:, 0]] - Y[edge[:, 1]], axis=1) / old_length
            if (
                d.min() < 0.75
                or d.max() > 1.25
                or np.linalg.norm(Y - x, axis=1).max() > 9
            ):
                continue
            if clearance(Y).min() < min(clearance(y).min() + 0.000005, 0.0):
                continue
            if not np.array_equal(candidate[pins], source[pins]):
                raise ValueError("Shifted tendon sample")
            predicted = [candidate] + [
                state(candidate, weights, aff, b, p) for p in poses
            ]
            rel = [
                abs(volume(a, f) / volume(b0, f) - 1)
                for a, b0 in zip(predicted, baseline)
            ]
            if max(rel) > 0.0095:
                continue
            folds = [len(v) for v in per_pose(candidate, weights, f, aff, b, poses)]
            if any(folds):
                continue
            nv, ng = evaluate(Y)
            if np.any(nv > original + 1e-5) or objective(nv) >= objective(V) - 0.01:
                continue
            y = Y
            src_out = candidate
            V = nv
            g = ng
            accepted = True
            hist.append(
                dict(
                    iteration=iteration,
                    fraction=h,
                    overlap_sum_mm3=float(V.sum()),
                    maximum_shift_mm=float(np.linalg.norm(y - x, axis=1).max()),
                    worst_volume_error=max(rel),
                    bed_clearance_mm=float(clearance(y).min()),
                    edge_ratio=[d.min(), d.max()],
                )
            )
            np.savez_compressed(
                folder / "candidate.npz", source=src_out, source_index=inv
            )
            print(hist[-1], flush=True)
            break
        if not accepted:
            break
    report = dict(
        source_sha256=meta["bank"],
        geometry=meta["geometry"],
        original_mm3=dict(zip(names, original)),
        candidate_mm3=dict(zip(names, V)),
        history=hist,
        max_derivative_difference_mm2=float(np.max(abs(analytic - fd))),
        checks="Control-envelope on each neighboring closed solid. Four source/native pose fold checks. Fixed tendon points. Static authoring only.",
    )
    (folder / "fit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    fit(*(Path(s).resolve() for s in sys.argv[1:]))

"""Fit existing muscle registration frames to two measured contact states, preserving the force route."""

import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from extensor_registration import sha, tissue, apply, affines, landmarks
from compare_hand_interfaces import query


def snapshot(case, reader, initial):
    import hashlib

    receipt = json.loads((case / "terminal.receipt.json").read_text())
    raw = subprocess.check_output(
        [
            "zstd",
            "-qdc",
            case / ("initial.mrvpack.zst" if initial else "accepted.mrvpack.zst"),
        ]
    )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt["base_pack_file_sha256" if initial else "pack_file_sha256"]:
        raise ValueError("Changed native frame")
    pose = {
        r["body_index"]: r
        for r in (
            receipt["initial_anatomical_registration"]["body_poses"]
            if initial
            else receipt["accepted_body_poses"]
        )
    }
    sys.path.insert(0, str(reader.parent / "src"))
    pr = runpy.run_path(str(reader))["primitives"]
    targets = {
        key[1]: v
        for key, _, v, f in pr(raw, pose)
        if key in ((51005, 137), (51005, 138))
    }
    banks = {sid: [] for sid in targets}
    for (kind, tid), _, v, f in pr(raw, pose):
        tri = v[f]
        lo = tri.min(1)
        hi = tri.max(1)
        for sid, y in targets.items():
            if (kind, tid) == (51005, sid):
                continue
            a = y.min(0) - 0.025
            b = y.max(0) + 0.025
            mask = ((hi >= a) & (lo <= b)).all(1)
            if not mask.any():
                continue
            # Native closed bank orientation comes from its full surface, before cropping.
            t = tri.astype(float) - v[0].astype(float)
            sign = (
                1.0
                if np.einsum("ij,ij->", t[:, 0], np.cross(t[:, 1], t[:, 2])) >= 0
                else -1.0
            )
            if kind == 51007:
                sign = -sign
            ids, remap = np.unique(f[mask], return_inverse=True)
            banks[sid].append((kind, tid, v[ids], remap.reshape(-1, 3), sign))
    return pose, targets, banks, digest


class Neighbors:
    def __init__(self, banks, faces, source, exe):
        xyz = []
        tris = []
        kind = []
        sgn = []
        n = 0
        for k, s, p, f, sign in banks:
            xyz.append(p)
            tris.append(f + n)
            n += len(p)
            kind.extend([k] * len(f))
            sgn.extend([sign] * len(f))
        self.fixed = np.concatenate(xyz)
        self.f = np.concatenate([*tris, faces + n])
        self.start = sum(len(f) for f in tris)
        self.kind = np.array(kind + [0] * len(faces))
        self.sign = np.array(sgn + [0] * len(faces))
        self.mask = np.arange(len(self.f)) >= self.start
        self.source = source
        self.faces = faces
        self.exe = exe
        self.lower = source.min(0) - 0.025
        self.upper = source.max(0) + 0.025

    def evaluate(self, y):
        if np.any(y < self.lower) or np.any(y > self.upper):
            raise ValueError("Candidate escaped prepared broad phase")
        all_points = np.concatenate((self.fixed, y))
        hit = np.array(
            sorted(query(self.exe, all_points, self.f, self.mask)), dtype=int
        ).reshape(-1, 2)
        kinds = self.kind[hit[:, 0]]
        metrics = [int((kinds == k).sum()) for k in (0, 51007, 51004, 51005)]
        metrics.append(int(len(hit) - sum(metrics)))
        return hit, np.array(metrics)


def bind_dofs(bind, src, which):
    spec = []
    for j, q in enumerate(bind):
        subset = src[which == q[0]]
        if not len(subset):
            continue
        local = Rotation.from_quat(q[4:8]).apply(subset) * q[8] + q[1:4]
        pivot = local.mean(0)
        h = local - pivot
        rank = np.linalg.matrix_rank(h, tol=1e-8)
        if rank == 0:
            basis = np.eye(3)
        elif rank == 1:
            basis = np.linalg.svd(h, full_matrices=False)[2][:1]
        else:
            continue
        spec.append((j, pivot, basis))
    return spec


def step_rows(rows, spec, dx):
    result = list(rows)
    start = 0
    for j, pivot, basis in spec:
        q = rows[j]
        delta = Rotation.from_rotvec(dx[start : start + len(basis)] @ basis)
        start += len(basis)
        t = pivot + delta.apply(np.array(q[1:4]) - pivot)
        quat = (delta * Rotation.from_quat(q[4:8])).as_quat()
        result[j] = struct.unpack("<I8f", struct.pack("<I8f", q[0], *t, *quat, q[8]))
    return result


def jacobian(x, slot, w, rows, pose, spec):
    n = sum(len(b) for _, _, b in spec)
    J = np.zeros((len(x), 3, n))
    offset = 0
    for j, pivot, basis in spec:
        q = rows[j]
        part = np.sum(w * (slot == j), axis=1)
        relative = Rotation.from_quat(q[4:8]).apply(x) * q[8] + q[1:4] - pivot
        rb = Rotation.from_quat(pose[q[0]]["quaternion_xyzw"])
        for axis in basis:
            J[:, :, offset] = part[:, None] * rb.apply(np.cross(axis, relative))
            offset += 1
    return J


def constraints(scene, y, J, hit):
    # Each existing interpenetration contributes its most restrictive corner.
    # For self collision choose the shortest face/face separating displacement.
    f = scene.faces
    L = []
    rhs = []
    kinds = []
    for self_contact in (False, True):
        hh = hit[(hit[:, 0] >= scene.start) == self_contact]
        if not len(hh):
            continue
        t2 = f[hh[:, 1] - scene.start]
        B = y[t2].astype(float)
        if self_contact:
            t1 = f[hh[:, 0] - scene.start]
            A = y[t1].astype(float)
            e1 = np.roll(A, -1, axis=1) - A
            e2 = np.roll(B, -1, axis=1) - B
            directions = np.concatenate(
                (
                    np.cross(e1[:, 0], e1[:, 1])[:, None, :],
                    np.cross(e2[:, 0], e2[:, 1])[:, None, :],
                    np.cross(e1[:, :, None, :], e2[:, None, :, :]).reshape(-1, 9, 3),
                ),
                axis=1,
            )
            length = np.linalg.norm(directions, axis=2)
            directions /= np.maximum(length[:, :, None], 1e-30)
            ap = np.einsum("nvi,nki->nkv", A - A[:, :1], directions)
            bp = np.einsum("nvi,nki->nkv", B - A[:, :1], directions)
            forward = ap.max(2) - bp.min(2)
            reverse = bp.max(2) - ap.min(2)
            costs = np.minimum(forward, reverse)
            costs[length < 1e-16] = np.inf
            axis = costs.argmin(1)
            rows = np.arange(len(hh))
            sign = np.where(forward[rows, axis] <= reverse[rows, axis], 1.0, -1.0)
            normal = directions[rows, axis] * sign[:, None]
            ia = t1[rows, (ap[rows, axis] * sign[:, None]).argmax(1)]
            ib = t2[rows, (bp[rows, axis] * sign[:, None]).argmin(1)]
            gap = np.einsum("ni,ni->n", normal, y[ib] - y[ia])
            mat = np.einsum("ni,nip->np", normal, J[ib] - J[ia])
        else:
            A = scene.fixed[scene.f[hh[:, 0]]].astype(float)
            normal = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
            normal *= scene.sign[hh[:, 0], None] / np.maximum(
                np.linalg.norm(normal, axis=1)[:, None], 1e-30
            )
            distances = np.einsum("nij,nj->ni", B - A[:, :1], normal)
            ix = distances.argmin(1)
            ib = t2[np.arange(len(hh)), ix]
            gap = distances.min(1)
            mat = np.einsum("ni,nip->np", normal, J[ib])
        L.append(mat)
        rhs.append(np.maximum(0.000015 - gap, 0.0))
        kinds.append(scene.kind[hh[:, 0]])
    return np.concatenate(L), np.concatenate(rhs), np.concatenate(kinds)


def measured_descent(rows, spec, evaluate, ys, hits, stats, baseline, angular_step):
    # The active facets may change at a fraction of a degree. Use physical
    # frame perturbations to measure their exact count response in each pose.
    from scipy.optimize import linprog

    n = sum(len(b) for _, _, b in spec)
    basis = np.eye(n) * angular_step
    den = np.maximum(baseline[:, :4], 1.0)
    reference = (
        float((stats[:, :4] / den).sum()) if np.all(stats <= baseline) else np.inf
    )
    best = None

    def try_step(dx):
        nonlocal best
        q = step_rows(rows, spec, dx)
        points, active, counts = evaluate(q)
        score = float((counts[:, :4] / den).sum())
        if (
            score < reference
            and np.all(counts <= baseline)
            and (best is None or score < best[0])
        ):
            best = score, q, points, active, counts, float(np.linalg.norm(dx))
        return (counts[:, :4] / den).ravel()

    columns = []
    for axis in basis:
        columns.append((try_step(axis) - try_step(-axis)) * 0.5)
    D = np.stack(columns, axis=1)
    # A minimax step rewards improvements shared by both poses and all tissue
    # classes. Its trust region is in radians about the named bone landmarks.
    lp = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[D, np.ones(len(D))],
        b_ub=np.zeros(len(D)),
        bounds=[(-1, 1)] * n + [(None, None)],
        method="highs",
    )
    print(
        "measured direction",
        float(np.rad2deg(angular_step)),
        lp.x[-1] if lp.success else None,
        flush=True,
    )
    if lp.success:
        for scale in (1.0, 0.5, 0.25):
            try_step(lp.x[:-1] * angular_step * scale)
    return best


def mirror_seed(bind, other, target, spec):
    # Reflect the source-frame change, then project it onto the free rotational
    # axes of the contralateral donor. Its own named sites stay fixed.
    reflection = np.diag([-1.0, 1.0, 1.0])
    dx = []
    if len(bind) != len(other) or len(other) != len(target):
        raise ValueError("Different bilateral binding chain")
    for j, _, basis in spec:
        ro = Rotation.from_quat(other[j][4:8])
        new = Rotation.from_quat(target[j][4:8])
        ds = (ro.inv() * new).as_matrix()
        local = Rotation.from_quat(bind[j][4:8])
        delta = local * Rotation.from_matrix(reflection @ ds @ reflection) * local.inv()
        dx.extend(basis @ delta.as_rotvec())
    return np.asarray(dx)


def run(start, end, reader, route_path, exe, out, side="both", right_seed=None):
    out.mkdir(exist_ok=False)
    frames = [snapshot(start, reader, True), snapshot(end, reader, False)]
    launch = json.loads((end / "launch.json").read_text())
    argv = launch["argv"]
    asset = Path(argv[argv.index("--soft-tissue-payload") + 1])
    data = asset.read_bytes()
    route = json.loads(route_path.read_text())
    if (
        sha(asset) != launch["pins"][str(asset)]
        or sha(Path(argv[2])) != route["identities"]["myosim"]
    ):
        raise ValueError("Mismatched physical payload")
    report = dict(
        source_sha256=sha(asset),
        frames=[s[3] for s in frames],
        routes_sha256=sha(route_path),
        banks={},
    )
    sides = {
        "right": ((137, ""),),
        "left": ((138, "_l"),),
        "both": ((137, ""), (138, "_l")),
    }
    for sid, suffix in sides[side]:
        x, f, slots, w, bind, ids, vo = tissue(data, sid)
        src, dst, which = landmarks(
            bind,
            frames[0][0],
            [route["digits"][f"EDC{d}{suffix}"]["route"] for d in (2, 3, 4, 5)],
        )
        spec = bind_dofs(bind, src, which)
        scenes = [Neighbors(fr[2][sid], f, fr[1][sid], exe) for fr in frames]

        def evaluate(rows):
            ys = []
            hits = []
            stats = []
            for fr, scene in zip(frames, scenes):
                A, t = affines(slots, w, rows, fr[0])
                y = (apply(A, x) + t).astype("<f4")
                h, c = scene.evaluate(y)
                ys.append(y)
                hits.append(h)
                stats.append(c)
            return ys, hits, np.stack(stats)

        rows = bind
        ys, hits, baseline = evaluate(rows)
        if (
            max(
                np.linalg.norm(a - fr[1][sid], axis=1).max()
                for a, fr in zip(ys, frames)
            )
            > 2e-6
        ):
            raise ValueError("Pose replay mismatch")
        stats = baseline.copy()
        log = []
        den = np.maximum(baseline[:, :4], 1.0)
        print("baseline", sid, baseline.tolist(), flush=True)
        if sid == 138 and (right_seed is not None or side == "both"):
            other = tissue(data, 137)[4]
            with np.load(
                right_seed if right_seed is not None else out / "137-fitted.npz"
            ) as src_right:
                delta = mirror_seed(bind, other, src_right["binding"], spec)
            for scale in np.arange(0.32, 1.031, 0.035):
                q = step_rows(bind, spec, scale * delta)
                pts, hs, cs = evaluate(q)
                print("mirror", scale, cs.tolist(), flush=True)

                def seed_cost(counts):
                    u = counts[:, :4] / den
                    return np.sum(u) + 10 * np.maximum(0, u - 1).sum()

                if not cs[:, 4].any() and seed_cost(cs) < seed_cost(stats):
                    rows, ys, hits, stats = q, pts, hs, cs
            log.append(dict(kind="contralateral seed", counts=stats.tolist()))
        if np.any(stats > baseline):
            # An intermediate authoring proposal may be infeasible. Seek a
            # feasible output around it before using it in the simulator.
            best = measured_descent(
                rows, spec, evaluate, ys, hits, stats, baseline, np.deg2rad(2.5)
            )
            if best is not None:
                _, rows, ys, hits, stats, _ = best
        for it in range(4 if sid == 137 else 5):
            M = []
            b = []
            jac = []
            for fr, scene, y, h in zip(frames, scenes, ys, hits):
                J = jacobian(x, slots, w, rows, fr[0], spec)
                jac.append(J)
                L, r, kind = constraints(scene, y, J, h)
                for k in (0, 51007, 51004, 51005):
                    keep = kind == k
                    if keep.any():
                        mass = 1.0 / np.sqrt(keep.sum())
                        M.append(L[keep] * mass)
                        b.append(r[keep] * mass)
            A = np.vstack(M)
            r = np.concatenate(b)
            reg = 0.001
            H = A.T @ A + reg**2 * np.eye(A.shape[1])
            rhs = A.T @ r
            dx = np.linalg.solve(H, rhs)
            predicted = max(
                np.linalg.norm(np.einsum("vij,j->vi", J, dx), axis=1).max() for J in jac
            )
            dx *= min(
                1.0,
                0.003 / max(predicted, 1e-20),
                np.deg2rad(8) / max(abs(dx).max(), 1e-20),
            )
            choices = []
            for frac in (1.0, 0.5, 0.25, -0.5):
                candidate = step_rows(rows, spec, frac * dx)
                points, hnew, counts = evaluate(candidate)
                if np.any(counts[:, 4] > baseline[:, 4]) or np.any(
                    counts[:, :4] > baseline[:, :4]
                ):
                    continue
                score = float((counts[:, :4] / den).sum())
                if score < float((stats[:, :4] / den).sum()):
                    choices.append((score, candidate, points, hnew, counts, frac))
            if not choices:
                measured = None
                for angle in (2.5, 6.0, 0.7):
                    measured = measured_descent(
                        rows,
                        spec,
                        evaluate,
                        ys,
                        hits,
                        stats,
                        baseline,
                        np.deg2rad(angle),
                    )
                    if measured is not None:
                        break
                if measured is None:
                    print("no joint descent", sid, it, flush=True)
                    break
                choices.append(measured)
            _, rows, ys, hits, stats, frac = min(choices, key=lambda q: q[0])
            log.append(dict(iteration=it + 1, fraction=frac, counts=stats.tolist()))
            print("keep", sid, it + 1, stats.tolist(), flush=True)
        if np.any(stats > baseline) or np.all(stats == baseline):
            raise ValueError(f"No admissible regional improvement for {sid}")
        record = np.frombuffer(
            data, "<u4", offset=64, count=struct.unpack_from("<I", data, 12)[0] * 8
        ).reshape(-1, 8)
        (bank,) = record[record[:, 6] == sid]
        by = {q[0]: q for q in rows}
        maxerr = 0.0
        for q in bind:
            s = src[which == q[0]]
            if not len(s):
                continue
            fn = lambda p: Rotation.from_quat(p[4:8]).apply(s) * p[8] + p[1:4]
            maxerr = max(maxerr, np.linalg.norm(fn(q) - fn(by[q[0]]), axis=1).max())
        if maxerr > 1e-6:
            raise ValueError("A registered route landmark moved in its bone")
        np.savez_compressed(
            out / f"{sid}-fitted.npz",
            ids=ids,
            binding=np.array(rows),
            frame0=ys[0],
            frame1=ys[1],
            first_binding=bank[0],
        )
        report["banks"][sid] = dict(
            candidate_sha256=sha(out / f"{sid}-fitted.npz"),
            baseline=baseline.tolist(),
            final=stats.tolist(),
            iterations=log,
            max_site_error_m=float(maxerr),
            max_change_mm=[
                float(1000 * np.linalg.norm(y - fr[1][sid], axis=1).max())
                for y, fr in zip(ys, frames)
            ],
        )
        (out / "site-fit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("start", "end", "reader", "route_path", "exe", "out"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--right-seed", type=Path)
    run(**vars(parser.parse_args()))

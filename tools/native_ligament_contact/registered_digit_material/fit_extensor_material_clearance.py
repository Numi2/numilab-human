"""Bound the existing donor-site correction using measured inner, outer and deep material interfaces."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog

from extensor_registration import affines, apply, landmarks, sha, tissue
from solve_extensor_sites import bind_dofs, Neighbors, snapshot, step_rows
from verify_extensor_material_layers import boundaries
from verify_native_extensor_pair import loaded


def fit(start, end, dense, reader, predicate, registry, prior, out):
    out.mkdir(exist_ok=False)
    frames = [snapshot(start, reader, True), snapshot(end, reader, False)]
    routes = json.loads(registry.read_text())
    launch = json.loads((dense / "launch.json").read_text())
    argv = launch["argv"]
    source = Path(argv[argv.index("--soft-tissue-payload") + 1])
    report = json.loads((prior / "site-fit.json").read_text())
    if (
        sha(source) != report["source_sha256"]
        or sha(registry) != report["routes_sha256"]
        or report["frames"] != [fr[3] for fr in frames]
    ):
        raise ValueError("Different material or source registration")
    sid = 138
    x, faces, slots, weights, old, ids, _ = tissue(source.read_bytes(), sid)
    route = [routes["digits"][f"EDC{k}_l"]["route"] for k in range(2, 6)]
    sites, _, owners = landmarks(old, frames[0][0], route)
    spec = bind_dofs(old, sites, owners)
    case = prior / "138-fitted.npz"
    if sha(case) != report["banks"]["138"]["candidate_sha256"]:
        raise ValueError("Different saved left donor rows")
    with np.load(case) as inp:
        prior_rows = inp["binding"]
        idx = inp["first_binding"].copy()
    receipt, dense_pose, _ = loaded(dense, initial=False, dense=True)
    xyz, layers, _ = boundaries(dense)

    def posed(rows, pose):
        a, t = affines(slots, weights, rows, pose)
        return (apply(a, x) + t).astype("<f4")

    initial = posed(old, dense_pose)
    lo, hi = initial.min(0) - 0.025, initial.max(0) + 0.025
    neighbor_layers = []
    for i, f in enumerate(layers):
        tri = xyz[f]
        mask = ((tri.max(1) >= lo) & (tri.min(1) <= hi)).all(1)
        indices, remap = np.unique(f[mask], return_inverse=True)
        neighbor_layers.append((i + 1, i, xyz[indices], remap.reshape(-1, 3), 1.0))
    dense_scene = Neighbors(neighbor_layers, faces, initial, predicate)
    compact = [Neighbors(fr[2][sid], faces, fr[1][sid], predicate) for fr in frames]

    def eval_rows(rows):
        pt = posed(rows, dense_pose)
        pairs, _ = dense_scene.evaluate(pt)
        kinds = dense_scene.kind[pairs[:, 0]]
        counts = [int(np.sum(kinds == k)) for k in (1, 2, 3)]
        for fr, scene in zip(frames, compact):
            _, count = scene.evaluate(posed(rows, fr[0]))
            counts.extend(count.tolist())
        return np.array(counts)

    base = eval_rows(old)
    current = eval_rows(prior_rows)
    print("old, first repair", base.tolist(), current.tolist(), flush=True)
    nonzero = base > 0
    # Retain a numerical margin on every class for binary32 native replay.
    # For any previously clear class the only admissible result is still zero.
    margin = np.where(nonzero, np.minimum(8, base // 25), 0)
    target = base - margin

    def score(count):
        err = (count - target) / np.maximum(1, base)
        return (
            np.max(np.maximum(err, 0)),
            np.sum(np.maximum(err, 0)),
            np.sum(count[nonzero] / base[nonzero]),
        )

    rows = [tuple([int(q[0]), *q[1:]]) for q in prior_rows]
    counts = current
    n = sum(len(basis) for _, _, basis in spec)
    evaluations = []
    origin = [posed(old, fr[0]) for fr in frames]
    # Each local perturbation is about the calibrated named material sites.
    # Evaluate all mandatory interfaces in the same proposal, including the
    # three installed deformable boundaries in the measured dense pose.
    for iteration in range(10):
        best = (score(counts), rows, counts)
        previous = best[0]
        for angle in (3.0, 0.8, 7.0, 12.0):
            step = np.deg2rad(angle)

            def trial(dx):
                nonlocal best
                cand = step_rows(rows, spec, dx)
                if any(
                    np.linalg.norm(posed(cand, fr[0]) - src, axis=1).max() > 0.009
                    for fr, src in zip(frames, origin)
                ):
                    return counts
                c = eval_rows(cand)
                if np.any(c[~nonzero] > 0):
                    return c
                key = score(c)
                if key < best[0]:
                    best = (key, cand, c)
                return c

            dg = np.stack(
                [(trial(v) - trial(-v)) * 0.5 for v in np.eye(n) * step], axis=1
            )
            J = dg[nonzero] / base[nonzero, None]
            slack = (target[nonzero] - counts[nonzero]) / base[nonzero]
            solution = linprog(
                np.r_[np.zeros(n), 1.0],
                A_ub=np.c_[J, -np.ones(len(slack))],
                b_ub=slack,
                bounds=[(-1.0, 1.0)] * n + [(None, None)],
                method="highs",
            )
            if solution.success:
                for scale in (0.125, 0.25, 0.5, 0.8, 1.0):
                    trial(step * scale * solution.x[:-1])
            if best[0] < previous:
                break
        if best[0] >= previous:
            break
        _, rows, counts = best
        evaluations.append(counts.tolist())
        print("angular sweep", iteration, angle, counts.tolist(), flush=True)
        if np.all(counts <= target):
            break
    if np.any(counts > target):
        raise ValueError(f"No clearance-preserving update: {counts} vs {target}")
    dst = out / "138-fitted.npz"
    np.savez_compressed(
        dst,
        ids=ids,
        binding=np.array(rows),
        first_binding=idx,
        frame0=posed(rows, frames[0][0]),
        frame1=posed(rows, frames[1][0]),
    )
    other = prior / "137-fitted.npz"
    if sha(other) != report["banks"]["137"]["candidate_sha256"]:
        raise ValueError("Changed right result")
    shutil.copyfile(other, out / other.name)
    info = report["banks"]["138"]
    info["parent_candidate_sha256"] = info["candidate_sha256"]
    info["candidate_sha256"] = sha(dst)
    info["iterations"].append(
        dict(
            kind="physical boundary clearance",
            control=base[:3].tolist(),
            first_repair=current[:3].tolist(),
            final=counts[:3].tolist(),
            angular_sweeps=evaluations,
            dense_pack=receipt["pack_file_sha256"],
            physical_nodes=sha(dense / "physical-accepted.node"),
        )
    )
    info["final"] = counts[3:].reshape(2, 5).tolist()
    info["max_change_mm"] = [
        float(1000 * np.linalg.norm(posed(rows, fr[0]) - fr[1][sid], axis=1).max())
        for fr in frames
    ]

    # Installer independently measures serialized-site errors for both banks.
    def mapped(q, pts):
        return Rotation.from_quat(q[4:8]).apply(pts) * q[8] + q[1:4]

    info["max_site_error_m"] = float(
        max(
            np.linalg.norm(
                mapped(q, sites[owners == q[0]]) - mapped(p, sites[owners == q[0]]),
                axis=1,
            ).max()
            for p, q in zip(old, rows)
            if np.any(owners == q[0])
        )
    )
    (out / "site-fit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "start",
        "end",
        "dense",
        "reader",
        "predicate",
        "registry",
        "prior",
        "out",
    ):
        parser.add_argument(key, type=lambda s: Path(s).resolve())
    fit(**vars(parser.parse_args()))

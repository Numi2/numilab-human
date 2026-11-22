"""Register four connected source branches to their corresponding compiled digit donor chains."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
from extensor_transfer import transfer
from compare_hand_interfaces import query
from material_context import sha


def ramp(u):
    s = np.clip(u, 0, 1)
    return s * s * (3 - 2 * s)


def fit(root, run, exe):
    meta = json.loads((root / "geometry.json").read_text())
    q = json.loads((run / "admission.json").read_text())
    rc = json.loads(
        (
            run / f"native/accepted-geometry/step-{q['accepted_steps']}.receipt.json"
        ).read_text()
    )
    poses = {r["body_index"]: r for r in rc["accepted_body_poses"]}
    if rc["pack_file_sha256"] != meta["identities"]["native_pack"]:
        raise ValueError("Unmatched accepted pose")
    report = {}
    for sid, suffix in ((137, ""), (138, "_l")):
        data = np.load(root / f"extensor-{sid}.npz")
        x = data["x"]
        f = data["f"]
        w = data["weights"]
        body = data["body"].tolist()
        rot = Rotation.from_quat(data["rot"])
        trans = data["shift"]
        scale = data["scale"]

        def inverse(b, world):
            i = body.index(b)
            return rot[i].inv().apply(np.asarray(world) - trans[i]) / scale[i]

        joint = np.stack([inverse(b, poses[b]["position_m"]) for b in body])
        routes = [meta["digits"][f"EDC{j}{suffix}"]["route"] for j in (2, 3, 4, 5)]
        paths = [[inverse(r["body"], r["world_m"]) for r in rows] for rows in routes]
        axis = np.mean([p[-1] - p[0] for p in paths], axis=0)
        axis /= np.linalg.norm(axis)
        coord = x @ axis
        edge = f.flatten()
        end = f[:, [1, 2, 0]].flatten()
        graph = coo_matrix(
            (np.ones(len(edge)), (edge, end)), shape=(len(x), len(x))
        ).tocsr()
        seeds = cKDTree(x).query([p[-3] for p in paths])[1]
        # Trace proximally until the four distal source branches first merge.
        if len(set(seeds)) != 4:
            raise ValueError(
                "Compiled middle-phalangeal sites do not identify separate source branches"
            )
        selected = None
        for split in np.arange(coord[seeds].min() - 0.002, coord.min(), -0.001):
            ids = np.flatnonzero(coord > split)
            count, groups = connected_components(graph[ids][:, ids], directed=False)
            lookup = groups[np.searchsorted(ids, seeds)]
            if len(np.unique(lookup)) != 4:
                break
            if count == 4:
                selected = ids, groups, lookup, split
        if selected is None:
            raise ValueError("Cannot resolve four connected source branches")
        ids, groups, lookup, split = selected
        report[sid] = dict(
            branch_nodes=[int(np.sum(groups == k)) for k in lookup],
            source_seed_rows=seeds.tolist(),
            trials=[],
        )
        for width in (0.006, 0.012, 0.024):
            weights = w.copy()
            for d, rows in enumerate(routes):
                nodes = ids[groups == lookup[d]]
                distal = list(dict.fromkeys(r["body"] for r in rows[3:]))
                order = [body.index(b) for b in distal]
                if len(order) != 4:
                    raise ValueError("Incomplete donor chain")
                carpal = 2
                # The wrist-to-metacarpal transition starts in the proximal palm. Every
                # subsequent joint transition is centred at that named phalanx's frame.
                # A displaced distal centre beyond the end of the source shell creates no extension.
                sites = (joint[order] @ axis).copy()
                back = np.flatnonzero(np.diff(sites) <= 0)
                if len(back):
                    cut = int(back[0])
                    if min(sites[cut:]) <= coord[nodes].max():
                        raise ValueError("Ambiguous joint order within source branch")
                    # Both joint centres lie beyond this branch tip. There is no distal source surface to bind.
                    order = order[:cut]
                    sites = sites[:cut]
                target = np.zeros((len(nodes), len(body)))
                target[:, carpal] = 1.0
                for bone, s in zip(order, sites):
                    a = ramp((coord[nodes] - s + width) / (2 * width))
                    target *= 1 - a[:, None]
                    target[:, bone] += a
                blend = ramp((coord[nodes] - split) / 0.025)
                weights[nodes] = (1 - blend[:, None]) * weights[nodes] + blend[
                    :, None
                ] * target
            # NHTISS4 owns four influences per vertex. Preserve deterministic ordering and unit sum.
            ii = np.argsort(-weights, axis=1, kind="stable")[:, :4]
            out = np.zeros_like(weights)
            ww = np.take_along_axis(weights, ii, axis=1)
            ww = (ww / ww.sum(1)[:, None]).astype("f4")
            np.put_along_axis(out, ii, ww, axis=1)
            world = transfer(x, out, rot, trans, scale).astype("f4")
            self_pairs = query(exe, world, f, np.ones(len(f), bool))
            arg = f"{sid}-{1000 * width:g}"
            np.savez_compressed(root / f"branch-{arg}.npz", x=world, weights=out, tri=f)
            report[sid]["trials"].append(
                dict(
                    width_m=width,
                    self_pairs=len(self_pairs),
                    moved_nodes=int(np.sum(np.any(out != w, axis=1))),
                    maximum_change_m=float(
                        np.linalg.norm(world - data["y"], axis=1).max()
                    ),
                    file_sha256=sha(root / f"branch-{arg}.npz"),
                )
            )
        print(sid, report[sid], flush=True)
    (root / "branch-transfer.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    fit(*(Path(v).resolve() for v in sys.argv[1:]))

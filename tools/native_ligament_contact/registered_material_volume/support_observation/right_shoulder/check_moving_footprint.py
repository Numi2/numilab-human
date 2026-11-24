"""Test physical traction-site anatomy at every saved body-motion sample and exact geometry receipt."""

import csv
import sys
from pathlib import Path

import igl
import numpy as np
from compile_connected_joint import digest
from igl.copyleft import cgal
from rebind_ligament_material import banks
from registration_stages import parts
from repair_registered_surface import state

from transfer_cortical_origin import load, local, save


def check(case, registered, out, preflight=False):
    c = load(case / "launch.json")
    reg = load(registered / "native-owner/registration.json")
    r = load(registered / "candidates.json")
    if not preflight and c["registered_shoulder"] != digest(
        registered / "native-owner/registration.json"
    ):
        raise ValueError("Different physical route")
    controls = c["argv"]
    if preflight and (
        digest(Path(controls[3])) != r["bone"]
        or digest(Path(controls[controls.index("--soft-tissue-payload") + 1]))
        != r["tissue"]
    ):
        raise ValueError("Different body or tissue for trajectory preflight")
    e = r["proposals"][reg["candidate"]]
    a = c["argv"]
    t = Path(a[a.index("--soft-tissue-payload") + 1])
    f, y, _ = banks(Path(a[3]))[34, 10]
    f = f.astype("i4")
    sites = np.array([e["point"]] + e["active_nodes_m"])
    d, idx, centers = igl.point_mesh_squared_distance(sites, y, f)
    normal = np.cross(y[f[idx, 1]] - y[f[idx, 0]], y[f[idx, 2]] - y[f[idx, 0]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    normal[np.abs(igl.winding_number(y, f, centers + normal * 0.00005)) > 0.5] *= -1
    depth = np.array([0.05, 0.25, 0.5, 1, 2, 3, 5, 8]) * 1e-3
    rays = np.ascontiguousarray(
        (centers[:, None] + normal[:, None] * depth[None, :, None]).reshape(-1, 3)
    )
    outside = np.abs(igl.winding_number(y, f, rays)).reshape(5, -1) < 0.5
    raw = [parts(t, j) for j in (83, 85)]
    geom = []
    for x, w, b, tri, aff in raw:
        v, i, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
        if np.max(abs(w[i][inv] - w)) > 1e-6:
            raise ValueError("Seam weights differ")
        geom.append((v, w[i], b, np.ascontiguousarray(inv[tri], dtype="i4"), aff))

    def measure(step, pose):
        inside = []
        folds = []
        near = []
        for j, (x, w, b, tri, aff) in enumerate(geom):
            xyz = state(x, w, aff, b, pose).astype("f4").astype(float)
            used = local(xyz, pose[34])
            inside.append(
                abs(igl.winding_number(np.ascontiguousarray(used), tri, rays)).reshape(
                    5, -1
                )
                > 0.5
            )
            if j == 0:
                folds = cgal.remesh_self_intersections(
                    np.ascontiguousarray(xyz), tri, detect_only=True
                )[2]
                near = np.sqrt(
                    igl.point_mesh_squared_distance(
                        sites, np.ascontiguousarray(used), tri
                    )[0]
                ).tolist()
        valid = (inside[0] & ~inside[1] & outside).any(1)
        return dict(
            step=step,
            self_pairs=len(folds),
            proven_sites=valid.tolist(),
            point_muscle_gap_m=near,
        )

    out_frames = {}
    rec_files = {}
    base = case / "native-traces"
    for path in sorted((base / "accepted-geometry").glob("*.receipt.json")):
        g = load(path)
        n = g["accepted_step"]
        pose = {q["body_index"]: q for q in g["accepted_body_poses"]}
        out_frames[n] = pose
        rec_files[n] = digest(path)
    exact = [measure(k, v) for k, v in out_frames.items()]
    frames = {}
    diff = 0.0
    for q in csv.DictReader((base / "resting-body-motion-partition.csv").open()):
        b = int(q["body_index"])
        k = int(q["accepted_step"])
        if b not in (34, 41, 25):
            continue
        p = np.array(
            [
                float(q[f"root_{a}_m"]) + float(q[f"com_relative_root_{a}_m"])
                for a in "xyz"
            ]
        )
        rot = np.array([float(q[f"orientation_{a}"]) for a in "xyzw"])
        if k in out_frames:
            v = out_frames[k][b]
            diff = max(
                diff,
                np.linalg.norm(p - v["position_m"]),
                np.minimum(
                    np.linalg.norm(rot - v["quaternion_xyzw"]),
                    np.linalg.norm(rot + v["quaternion_xyzw"]),
                ),
            )
        frames.setdefault(k, {})[b] = dict(position_m=p, quaternion_xyzw=rot)
    if diff > 1e-6 or set(frames) != set(range(32, 8001, 32)):
        raise ValueError("Motion series is unmatched")
    sampled = [measure(k, p) for k, p in sorted(frames.items())]
    bad = [x for x in sampled if x["self_pairs"] or not all(x["proven_sites"])]
    result = dict(
        launch=digest(case / "launch.json"),
        candidate=digest(registered / "native-owner/registration.json"),
        candidate_ran_in_this_motion=not preflight,
        receipts=rec_files,
        exact_receipts=exact,
        sampled_count=len(sampled),
        bad_sampled_count=len(bad),
        maximum_sampled_self_pairs=max(z["self_pairs"] for z in sampled),
        minimum_supported_sites=min(sum(z["proven_sites"]) for z in sampled),
        maximum_source_gap_at_five_sites_m=np.max(
            [z["point_muscle_gap_m"] for z in sampled], axis=0
        ).tolist(),
        sampled_witnesses=bad[:12],
        maximum_native_terminal_position_or_quaternion_component_norm_error=diff,
        scope="Two authoritative native capture poses and 250 sampled CPU kinematic reconstructions from the same accepted motion. The ray test checks facing at five traction sites. The intersection test checks the right infraspinatus surface only.",
    )
    save(out, result)
    print(
        len(sampled),
        len(bad),
        result["maximum_sampled_self_pairs"],
        result["minimum_supported_sites"],
        flush=True,
    )


if __name__ == "__main__":
    check(*(Path(v).resolve() for v in sys.argv[1:4]), "--preflight" in sys.argv[4:])

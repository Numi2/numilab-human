"""Transport the bonded tissue together through its existing material-parent rows."""

import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from live_pose_frames import current_donors
from supported_material_admission import geom


def transfer(case, volume, skeleton, support, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    e = cfg["environment"]
    rig = json.loads(skeleton.read_bytes())
    bed = json.loads(support.read_bytes())
    if (
        rig["current_launch_sha256"] != digest(case / "launch.json")
        or bed["launch_sha256"] != rig["current_launch_sha256"]
        or rig["final_joint_limit_violations"]
    ):
        raise ValueError("Mismatched or inadmissible skeleton")
    dst = np.asarray(rig["target_poses"])
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    frames_file = Path(e[prefix + "DONOR_FRAMES_CSV"])
    map_file = Path(e[prefix + "COMBINED_NODE_MAP_CSV"])
    for k, p in [("DONOR_FRAMES", frames_file), ("COMBINED_NODE_MAP", map_file)]:
        if digest(p) != e[prefix + k + "_SHA256"]:
            raise ValueError("Stale tissue registration")
    frames = {int(r["donor"]): r for r in csv.DictReader(frames_file.open())}
    live, donors = current_donors(case)
    current = {
        p["body_index"]: p["current"]
        for p in donors
    }
    R = np.tile(np.eye(3), (len(dst), 1, 1))
    base = np.zeros((len(dst), 3))
    for owner, p in current.items():
        R[owner] = Rotation.from_quat(p["orientation_world_xyzw"]).as_matrix()
        base[owner] = p["position_world_m"]
    D = Rotation.from_quat(dst[:, 3:]).as_matrix() @ R.transpose(0, 2, 1)
    source = Path(e["NUMI_H706_FRACTIONAL_SOURCE330_NODES"])
    topology = Path(e["NUMI_H706_FRACTIONAL_SOURCE330_TETS"])
    xyz = np.loadtxt(source, skiprows=1, delimiter=",", usecols=(2, 3, 4, 9))
    parents = np.loadtxt(
        topology, skiprows=1, delimiter=",", usecols=(3, 4, 5, 6), dtype="i4"
    )
    maps = np.loadtxt(
        map_file, skiprows=1, delimiter=",", usecols=(0, 29, 30, 31, 32, 33)
    )
    joint = maps[:, 0].astype(int) - 1
    parent = maps[:, 1].astype(int)
    weights = maps[:, 2:]
    if abs(weights.sum(1) - 1).max() > 1.0e-6 or weights.min() < -1.0e-7:
        raise ValueError("Invalid source ancestry")
    corner_owner = xyz[:, 3].astype(int)
    if not set(corner_owner).issubset(current) or not set(corner_owner).issubset(
        frames
    ):
        raise ValueError("Unregistered ancestor")
    local = np.zeros_like(xyz[:, :3])
    for owner in set(corner_owner):
        mask = corner_owner == owner
        f = frames[owner]
        origin = np.array([float(f["current_" + s]) for s in "xyz"])
        F = np.array(
            [[float(f[f"current_R{a}{b}"]) for b in range(3)] for a in range(3)]
        )
        local[mask] = np.linalg.solve(F, (xyz[mask, :3] - origin).T).T
    anchor0 = np.einsum("nij,nj->ni", R[corner_owner], local) + base[corner_owner]
    # Move each registered material point with the weighted rigid motions of
    # its source corners. The residual from that simplex rotates with it.
    source_x = np.loadtxt(volume / "mesh.node", skiprows=1, usecols=(1, 2, 3))
    original = np.loadtxt(volume / "current.node", skiprows=1, usecols=(1, 2, 3))
    if (
        np.max(joint) >= len(original)
        or np.min(joint) != 0
        or len(np.unique(joint)) != len(original)
    ):
        raise ValueError("Missing skin/fat coordinates")
    parents = parents[parent]
    owners = corner_owner[parents]
    count = np.bincount(joint)
    a0 = np.einsum("ni,nij->nj", weights, anchor0[parents])
    residual = original[joint] - a0
    fitted = np.zeros_like(original[joint])
    for k in range(4):
        oi = owners[:, k]
        cur = anchor0[parents[:, k]] + residual
        nxt = np.einsum("nij,nj->ni", D[oi], cur - base[oi]) + dst[oi, :3]
        fitted += weights[:, k, None] * (nxt - cur)
    inc = np.zeros_like(original)
    np.add.at(inc, joint, fitted)
    inc /= count[:, None]
    source_disagreement = np.linalg.norm(fitted - inc[joint], axis=1)
    tet = (
        np.loadtxt(volume / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype="i4")
        - 1
    )
    outer = np.unique(np.load(volume / "material-walls.npz")["O"])
    normal = np.asarray(bed["bed_normal"])
    off = bed["source_plane_offset_m"]
    ref = np.concatenate(
        [geom(source_x, chunk)[0] for chunk in np.array_split(tet, 20)]
    )
    trials = []
    for alpha in (0.0, 0.001, 0.01, 0.05, 0.25, 1.0):
        displacement = (original + alpha * inc) - source_x
        high = displacement.astype("f4").astype("f8")
        low = (displacement - high).astype("f4").astype("f8")
        z = source_x + (high + low)
        if alpha == 0.0 and not np.array_equal(z, original):
            raise ValueError("Baseline differs from emitted coordinates")
        vol = []
        shape = []
        for chunk in np.array_split(tet, 20):
            v, h = geom(z, chunk)
            vol.append(v)
            shape.append(h)
        v = np.concatenate(vol)
        h = np.concatenate(shape)
        J = v / ref
        bad = (v <= 1.0e-18) | (J < 0.4) | (J > 2.5) | (h < 1.0e-5)
        trials.append(
            dict(
                fraction=alpha,
                inverted_cells=int((v <= 0).sum()),
                failed_quality=int(bad.sum()),
                failed_shape=int((h < 1.0e-5).sum()),
                min_J=float(J.min()),
                max_J=float(J.max()),
                min_surface_gap_m=float((z[outer] @ normal - off).min()),
                outer_nodes_below_bed=int((z[outer] @ normal < off).sum()),
            )
        )
    record = dict(
        live_body_pose=digest(live),
        source_parent=digest(source),
        source_tets=digest(topology),
        node_map=digest(map_file),
        reference=digest(volume / "mesh.node"),
        registered=digest(volume / "current.node"),
        skeleton=digest(skeleton),
        bed=digest(support),
        grid_points=len(original),
        cells=len(tet),
        max_disagreement_of_shared_node_predictions_m=float(source_disagreement.max()),
        repeated_source_rows=len(joint) - len(original),
        trials=trials,
        scope="Initialization proposals only. A fraction between the two poses is a geometric diagnostic; articulated coordinates and tissue require separate joint and contact validation.",
    )
    out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print([(v["fraction"], v["failed_quality"], v["inverted_cells"]) for v in trials])


if __name__ == "__main__":
    transfer(*(Path(s).resolve() for s in sys.argv[1:]))

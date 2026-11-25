"""Enumerate wall/cortex triangle pairs in one accepted native configuration, split by wall role."""

import argparse
import json
from pathlib import Path
import numpy as np
import igl.copyleft.cgal as cgal
from audit_registered_cortical_occupancy import banks, digest
from scipy.spatial.transform import Rotation


def straddles(faces, vertices):
    """Identify triangles with vertices safely on both sides of an opposing plane."""
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    relative = vertices - faces[:, :1]
    side = np.einsum("ij,ikj->ik", normal, relative)
    uncertainty = (
        64
        * np.finfo(float).eps
        * np.linalg.norm(normal, axis=1)
        * np.linalg.norm(relative, axis=2).max(axis=1)
    )
    return (side.min(axis=1) < -uncertainty) & (side.max(axis=1) > uncertainty)


def audit(case, volume, verification, out, expected_walls):
    if out.suffix != ".json" or out.resolve() == verification.resolve():
        raise ValueError("Use a separate JSON path for wall intersections")

    cfg = json.loads((case / "launch.json").read_bytes())
    verified = json.loads(verification.read_bytes())
    o = verified["accepted_observation"]
    step = o["step"]
    rp = case / "native-traces/accepted-geometry" / f"step-{step}.receipt.json"
    r = json.loads(rp.read_bytes())
    if (
        o["launch_sha256"] != digest(case / "launch.json")
        or o["receipt_sha256"] != digest(rp)
        or r["accepted_time_s"] != o["time_s"]
    ):
        raise ValueError("Unmatched native surface and rigid frame")
    source = Path(cfg["argv"][3])
    if cfg["pins"].get(str(source)) != digest(source):
        raise ValueError("The registered cortical input changed")
    settings = cfg["environment"]
    pre = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    if digest(volume / "mesh.ele") != settings[pre + "TETGEN_ELEMENTS_SHA256"]:
        raise ValueError("Mismatched wall connectivity")
    wall_file = volume / "material-walls.npz"
    if digest(wall_file) != expected_walls:
        raise ValueError("The registered skin and fat wall file changed")
    p = case / "physical-accepted.node"
    if digest(p) != o["accepted_node_sha256"]:
        raise ValueError("The accepted material coordinates changed")
    x = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    roles = ("O", "I", "D", "C")
    with np.load(wall_file) as walls:
        wall_faces = np.vstack([walls[role] for role in roles])
        offsets = np.r_[0, np.cumsum([len(walls[role]) for role in roles])]
    lower = x[wall_faces].min(axis=1)
    upper = x[wall_faces].max(axis=1)
    poses = {pose["body_index"]: pose for pose in r["accepted_registered_body_poses"]}
    cortical_banks = banks(source)
    found = []
    stats = {}
    for (body_id, stable_id), (faces, local_points, _) in cortical_banks.items():
        pose = poses[body_id]
        unique, seam = np.unique(local_points, axis=0, return_inverse=True)
        bone_points = (
            Rotation.from_quat(pose["quaternion_xyzw"]).apply(unique)
            + pose["position_m"]
        )
        bone_faces = seam[faces]
        selected = np.flatnonzero(
            np.all(
                (upper >= bone_points.min(0)) & (lower <= bone_points.max(0)), axis=1
            )
        )
        counts = np.zeros((len(roles), 2), dtype=int)
        if len(selected):
            nodes, compact_faces = np.unique(wall_faces[selected], return_inverse=True)
            hits = cgal.intersect_other(
                x[nodes],
                compact_faces.reshape(-1, 3),
                bone_points,
                bone_faces,
                detect_only=True,
                first_only=False,
            )[0]
            if len(hits):
                wall_rows = selected[hits[:, 0]]
                bone_rows = hits[:, 1]
                wall_role = np.searchsorted(offsets[1:], wall_rows, side="right")
                tissue = x[wall_faces[wall_rows]]
                cortex = bone_points[bone_faces[bone_rows]]
                transverse = straddles(tissue, cortex) & straddles(cortex, tissue)
                counts[:, 0] = np.bincount(wall_role, minlength=len(roles))
                counts[:, 1] = np.bincount(wall_role[transverse], minlength=len(roles))
                found.extend(
                    np.column_stack(
                        (
                            np.full(len(hits), body_id),
                            np.full(len(hits), stable_id),
                            wall_role,
                            wall_rows - offsets[wall_role],
                            bone_rows,
                            transverse.astype(int),
                        )
                    ).tolist()
                )
        stats[f"{body_id}/{stable_id}"] = dict(zip(roles, counts.tolist(), strict=True))
    pairs = np.array(sorted(found), dtype="i4").reshape(-1, 6)
    np.savez_compressed(out.with_suffix(".npz"), pairs=pairs)
    totals = {
        k: np.array([v[k] for v in stats.values()]).sum(axis=0).tolist() for k in roles
    }
    rep = dict(
        launch_sha256=digest(case / "launch.json"),
        receipt_sha256=digest(rp),
        wall_sha256=digest(volume / "material-walls.npz"),
        pair_sha256=digest(out.with_suffix(".npz")),
        tested_banks=len(cortical_banks),
        roles=roles,
        wall_triangles=len(wall_faces),
        total=totals,
        by_bank=stats,
        scope="All CGAL wall/cortex intersections and the subset whose triangles each strictly straddle the opposing plane. Roles retain intended cortical-interface identity. This endpoint calculation does not measure continuous trajectories or tissue/tissue self-intersection.",
    )
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(totals, flush=True)
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-run", type=Path, required=True)
    parser.add_argument("--material-report", type=Path, required=True)
    parser.add_argument("--registered-walls-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = args.accepted_run.resolve()
    cfg = json.loads((case / "launch.json").read_bytes())
    key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS"
    volume = Path(cfg["environment"][key]).parent
    audit(case, volume, args.material_report, args.out, args.registered_walls_sha256)

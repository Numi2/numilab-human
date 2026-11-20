"""Choose a unilateral flat-bed pose from the emitted skin footprint."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from build_hand_trial import digest
from check_coarse_bed_footprint import report
from fit_hand_cavity import table


def select_plane(outer, center, root_rotation, position):
    """Find a lower supporting face with a positive gravity load split."""
    hull = ConvexHull(outer)
    vertical = np.array([0., 0., 1.])
    best = None
    for corners, equation in zip(hull.simplices, hull.equations, strict=True):
        inward = -equation[:3]
        if inward[2] < np.cos(np.deg2rad(4.)):
            continue
        cross = np.cross(inward, vertical)
        length = np.linalg.norm(cross)
        angle = np.arctan2(length, inward[2])
        rotation = Rotation.from_rotvec(cross * (angle / length) if length else cross)
        rotated = rotation.apply(outer[corners]-position)+position
        rotated_center = rotation.apply(center-position)+position
        face = rotated[:,:2]
        matrix = np.vstack((face.T, np.ones(3)))
        try:
            weights = np.linalg.solve(matrix, np.r_[rotated_center[:2], 1.])
        except np.linalg.LinAlgError:
            continue
        if min(weights) <= .01:
            continue
        # Use the smallest common pose change. The actual force split is
        # subsequently evaluated by the native deformable contact solver.
        if best is not None and angle >= best["rotation_radians"]:
            continue
        minimum = float(np.min((outer-outer[corners[0]]) @ inward))
        assert minimum >= -5.e-9
        best = {
            "rotation_radians": float(angle),
            "contact_outer_rows": sorted(map(int,corners)),
            "face_outer_rows_in_solver_order": [int(x) for x in corners],
            "normal_only_barycentric_weights": weights.tolist(),
            "estimated_exact_clearance_m": minimum,
            "rotation_xyzw": (rotation*root_rotation).as_quat().tolist(),
            "unaligned_gravity_moment_direction": np.cross(
                rotated_center-rotated[0], -vertical).tolist(),
        }
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    trial = args.native_trial.resolve()
    audit = report(trial)
    points = table(trial/"initial-skin-fem.csv")
    outer = np.array([[float(r[f"{a}_m"]) for a in "xyz"] for r in points if r["layer"]=="0"])
    launch_path = trial/"launch.json"
    argv = json.loads(launch_path.read_text())["argv"]
    idx = argv.index("--root-pose")+1
    root = np.array(argv[idx:idx+7], dtype=float)
    candidate = select_plane(outer, np.array(audit["body_com_m"]), Rotation.from_quat(root[3:]), root[:3])
    assert candidate is not None, "No nearby stable flat-bed footprint"
    n = candidate["rotation_xyzw"]
    result = {
        **candidate,
        "launch_sha256": digest(launch_path),
        "source_outer_sha256": digest(trial/"initial-skin-fem.csv"),
        "native_weight_N": audit["weight_N"],
        "root_pose_trial": [*root[:3].tolist(), *n],
        "static_force_split_N_geometry_only": [
            p*audit["weight_N"] for p in candidate["normal_only_barycentric_weights"]],
        "scope": "One-time source stance preparation. No assigned reactions, boundary motion or state modification after initialization.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Test whether positive flat-bed normal forces can balance weight and moment."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from audit import digest, rows

GRAVITY_MPS2 = 9.81


def center(log, motion):
    mass_kg = 0.
    first_moment = np.zeros(3)
    node_count = 0
    plane = None
    band = None
    with log.open() as f:
        for line in f:
            if line.startswith("resting_FEM_node,"):
                props = dict(x.split("=", 1) for x in line.rstrip().split(",")[1:])
                mass = float(props["initial_mass_kg"])
                assert np.isfinite(mass) and mass >= 0.
                mass_kg += mass
                first_moment += mass * np.array([float(props[f"initial_{i}"]) for i in "xyz"])
                node_count += 1
            elif line.startswith("resting_fem_skin_bed_source="):
                assert " normal=0,0,1 " in line
                plane = float(re.search(r" plane_offset_m=([^ ]+)", line)[1])
            elif line.startswith("resting_fem_bed_barrier_release="):
                band = float(re.search(r" activation_width_m=([^ ]+)", line)[1])
    assert node_count and mass_kg and plane is not None and band is not None
    fem_mass = mass_kg
    with motion.open() as f:
        for line in csv.DictReader(f):
            if int(line["accepted_step"]) != 1:
                break
            mass = float(line["mass_kg"])
            mass_kg += mass
            first_moment += np.array([mass * float(line[f"root_{a}_m"]) +
                                      float(line[f"mass_weighted_relative_com_{a}_kg_m"]) for a in "xyz"])
    return first_moment / mass_kg, mass_kg, fem_mass, plane, band


def load_balance(points, target, eligible):
    if not len(eligible):
        return None
    locations = points[eligible] - target
    eq = np.vstack([np.ones(len(eligible)), locations.T])
    ans = linprog(np.linalg.norm(locations, axis=1), A_eq=eq, b_eq=[1., 0., 0.],
                  bounds=(0., None), method="highs")
    if ans.status == 2:
        return None
    if not ans.success:
        raise RuntimeError(ans.message)
    assert np.max(np.abs(eq @ ans.x - [1., 0., 0.])) < 1.e-8
    return [(int(eligible[j]), float(ans.x[j])) for j in np.flatnonzero(ans.x > 0.)]


def convex_distance(points, target):
    if not len(points):
        return None
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - target))
    if np.linalg.matrix_rank(points - points[0]) < 2:
        offset = points - points[0]
        d = offset[np.argmax((offset * offset).sum(axis=1))]
        if np.dot(d, d) == 0.:
            return float(np.linalg.norm(points[0] - target))
        j = (points - points[0]) @ d
        boundary = points[[j.argmin(), j.argmax()]]
    else:
        hull = ConvexHull(points)
        if np.all(hull.equations[:, :2] @ target + hull.equations[:, 2] <= 0.):
            return 0.
        boundary = points[hull.vertices]
    edges = np.roll(boundary, -1, axis=0) - boundary
    t = ((target - boundary) * edges).sum(axis=1) / (edges * edges).sum(axis=1)
    closest = boundary + np.clip(t, 0., 1.)[:, None] * edges
    return float(np.linalg.norm(closest - target, axis=1).min())


def main():
    parser = ArgumentParser(description=__doc__)
    for arg in ("run", "output"):
        parser.add_argument(arg, type=Path)
    cfg = parser.parse_args()
    log = cfg.run / "stdout.log"
    motion = cfg.run / "native/resting-body-motion-partition.csv"
    p = cfg.run / "shell-mass.csv.nodes.csv"
    node_rows = rows(p)
    n = len(node_rows) // 3
    com, mass, fem_mass, plane, band = center(log, motion)
    coords = np.array([[line[f"current_{d}_m"] for d in "xyz"] for line in node_rows[:n]],
                      dtype=np.float32).astype(float)
    plane = float(np.float32(plane))
    band = float(np.float32(band))
    gaps = coords[:, 2] - plane
    near = np.flatnonzero(gaps <= band + 1.e-9)
    assert gaps.min() > 0. and len(near)
    dist = convex_distance(coords[near, :2], com[:2])
    result = {
        "scope": ("Initial shell and articulated COM at accepted root 1. Necessary static support "
                  "condition on the present plane: nonnegative normals have a center of pressure in "
                  "the convex hull of active projections. Friction on that plane cannot cancel a "
                  "pitch or roll moment about a point on the plane. Geometry and native forces "
                  "are untouched. The height-band analysis permits arbitrary pressure distribution "
                  "in that band and is only a lower bound on required engagement."),
        "sha256": {f.name: digest(f) for f in (log, motion, p)},
        "mass_kg": mass, "fem_mass_kg": fem_mass, "weight_n": mass * GRAVITY_MPS2,
        "com_m": com.tolist(), "bed_offset_m": plane, "contact_width_m": band,
        "near_rows": near.tolist(), "com_projection_distance_to_support_hull_m": dist,
        "minimum_unbalanced_pitch_roll_torque_nm_at_full_weight": dist * mass * GRAVITY_MPS2,
    }
    xy = coords[:, :2]
    ordered = np.argsort(gaps, kind="stable")
    lo, hi = 0, len(ordered)
    assert load_balance(xy, com[:2], ordered) is not None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if load_balance(xy, com[:2], ordered[:mid]) is None:
            lo = mid
        else:
            hi = mid
    first_gap = float(gaps[ordered[hi - 1]])
    shares = load_balance(xy, com[:2], np.flatnonzero(gaps <= first_gap))
    assert shares
    result["minimum_clearance_band_with_moment_closure_m"] = first_gap
    result["minimum_additional_engagement_m"] = first_gap - band
    result["possible_normal_distribution_at_threshold"] = [
        {"outer_shell_row": i, "point_m": coords[i].tolist(), "fraction": share,
         "normal_force_n": share * mass * GRAVITY_MPS2} for i, share in shares]
    with cfg.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"scope", "sha256"}}))


if __name__ == "__main__":
    main()

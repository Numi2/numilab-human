"""Measure the native flat-bed contact footprint at the registered startup pose."""
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from prepare import HERE, REFERENCE, digest

PLAN = json.loads((REFERENCE / "launch.json").read_text())
ROLE_O = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_NODES_CSV"



def hull_distance(points, center):
    """Return horizontal distance to the convex set of unilateral normal loads."""
    one = len(points) == 1
    if one:
        return float(np.linalg.norm(center-points[0])), 0.0
    try:
        hull = ConvexHull(points) if len(points) >= 3 else None
    except QhullError:
        hull = None
    if hull is not None:
        if np.max(hull.equations[:, :2] @ center + hull.equations[:, 2]) <= 0.0:
            return 0.0, float(hull.volume)
        boundary = points[hull.vertices]
        area = float(hull.volume)
    else:
        axis = points[np.argmax(np.linalg.norm(points-points[0], axis=1))] - points[0]
        projection = points @ axis
        boundary = points[[projection.argmin(), projection.argmax()]]
        area = 0.0
    edges = np.roll(boundary, -1, axis=0) - boundary
    offset = center - boundary
    square = np.einsum("ij,ij->i", edges, edges)
    t = np.clip(np.einsum("ij,ij->i", offset, edges) /
                np.maximum(square, np.finfo(np.float64).tiny), 0.0, 1.0)
    return float(np.min(np.linalg.norm(offset - t[:, None]*edges, axis=1))), area


def main():
    plan_env = PLAN["environment"]
    role_path = Path(plan_env[ROLE_O])
    sha = digest(role_path)
    # The launcher keeps this hash in the role's registration receipt.
    register = json.loads((role_path.parent / "receipt.json").read_text())
    assert sha == register["outputs"][role_path.name]
    data = np.loadtxt(role_path, delimiter=",", skiprows=1, usecols=(0,1,2,3,4))
    assert np.array_equal(data[:, 0], np.arange(len(data)))
    points = data[:, 2:].astype("<f4").astype(np.float64)
    terminal_log = (REFERENCE / "stdout.log").read_text()
    rows = re.search(r"^resting_fem_skin_bed_compiled=validated pair_rows=(\d+)"
                     r" initial_outer_min_gap_m=(\S+) initial_outer_max_gap_m=(\S+)", terminal_log, re.M)
    assert rows and len(points) == int(rows[1])
    assert abs(points[:,2].min() - float(rows[2])) < 1.e-20
    assert abs(points[:,2].max() - float(rows[3])) < 1.e-15
    assert "normal=0,0,1 plane_offset_m=0" in terminal_log
    with (REFERENCE / "native/resting-surface-audit.csv").open() as stream:
        initial = next(csv.DictReader(stream))
    assert int(initial["step"]) == 0
    articulated_xy = np.array([float(initial[f"body_com_{axis}_m"]) for axis in "xy"])
    sorted_points = points[np.argsort(points[:,2], kind="stable")]
    z = sorted_points[:, 2]

    def bounds(count):
        return hull_distance(sorted_points[:count,:2], articulated_xy)

    # Hulls grow monotonically with the source gap cutoff. Find the first
    # cutoff that contains the projected articulated COM, to Float32 tie groups.
    lower, upper = 1, len(points)
    assert bounds(upper)[0] == 0.0
    while upper - lower > 1:
        mid = (lower+upper)//2
        if bounds(mid)[0] == 0.0:
            upper = mid
        else:
            lower = mid
    candidate_mm = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1., 2., 5., 10.]
    metrics = []
    for mm in candidate_mm:
        n = int(np.searchsorted(z, mm*1.e-3, side="right"))
        relative, area = bounds(n)
        metrics.append({"maximum_source_gap_mm": mm, "source_outer_nodes": n,
                        "normal_reaction_xy_hull_area_mm2": area*1.e6,
                        "articulated_com_to_hull_mm": relative*1.e3})
    i = int(points[:, 2].argmin())
    result = {"shape": "native Float32 flat-plane source footprint, normal-load geometry only",
              "outer_registration_sha256": sha, "outer_node_count": len(points),
              "nearest_source_o_row": i, "nearest_source_donor_body": int(data[i, 1]),
              "nearest_position_m": points[i].tolist(),
              "initial_articulated_com_xy_m": articulated_xy.tolist(),
              "articulated_mass_kg": float(initial["represented_body_mass_kg"]),
              "initial_com_min_source_gap_for_xy_support_mm": 1000.*float(z[upper-1]),
              "source_o_nodes_at_that_cutoff": int(np.searchsorted(z, z[upper-1], side="right")),
              "diagnostic": metrics,
              "limitation": "The COM here is the source articulated projection, exclusive of separately integrated FEM masses. A feasible geometric hull alone gives no pressure or equilibrium result."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

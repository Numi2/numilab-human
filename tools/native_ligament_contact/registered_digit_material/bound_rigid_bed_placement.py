"""Bound the support gap achievable by a single rigid reorientation of the detailed body."""

from pathlib import Path
from heapq import heappush, heappop
import csv, json, sys
import numpy as np
from scipy.spatial import ConvexHull


def solve(points, sets, limit, tol):
    # Height differences are independent of the body's origin. Horizontal
    # widths bound their one-sided slope changes, including switches of witness.
    span = np.ptp(points[:, :2], axis=0)
    points = points - points.mean(axis=0)

    def hull(v):
        return (
            v[ConvexHull(v).vertices]
            if len(v) > 4 and np.linalg.matrix_rank(v - v.mean(0)) == 3
            else v
        )

    allpts = hull(points)
    skins = [hull(points[idx]) for idx in sets]
    best = float("inf")
    best_ab = None
    stack = []
    counter = 0

    def add(a, b, r):
        nonlocal counter, best, best_ab
        axis = np.array([a, b, 1.0])
        floor = np.min(allpts @ axis)
        offsets = np.array([np.min(s @ axis) - floor for s in skins])
        value = offsets.max() / np.linalg.norm(axis)
        if value < best:
            best = float(value)
            best_ab = (a, b)
        largest_norm = np.sqrt(1 + (abs(a) + r) ** 2 + (abs(b) + r) ** 2)
        lower = max(0.0, offsets.max() - sum(span) * r) / largest_norm
        counter += 1
        heappush(stack, (lower, counter, a, b, r))

    add(0.0, 0.0, limit)
    while best - stack[0][0] > tol:
        bound, i, a, b, r = heappop(stack)
        r *= 0.5
        for dx, dy in ((-r, -r), (-r, r), (r, -r), (r, r)):
            add(a + dx, b + dy, r)
    a, b = best_ab
    axis = np.array([a, b, 1.0])
    axis /= np.linalg.norm(axis)
    floor = np.min(allpts @ axis)
    exact = [float(np.min(s @ axis) - floor) for s in skins]
    return dict(
        lower_bound_m=float(stack[0][0]),
        upper_bound_m=best,
        unit_plane_normal=axis.tolist(),
        individually_remaining_gaps_m=exact,
        boxes_tested=counter,
    )


def run(native, case, split, donors, out):
    from bind_posterior_equilibrium import coordinates
    from material_context import sha

    d = json.loads(donors.read_text())
    env = json.loads((native / "launch.json").read_text())["environment"]
    f = Path(
        env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if sha(f) != d["dense_map"] or sha(native / "launch.json") != d["fine"]:
        raise ValueError("Unmatched current donor mapping")
    outer = np.load(split / "physical.npz")["Orows"]
    idx = np.full(len(outer), -1, dtype=int)
    with f.open() as s:
        for row in csv.DictReader(s):
            if row["registered_surface_layer"] == "O":
                i = int(row["registered_surface_vertex_row"])
                if outer[i] != int(row["joint_node_id"]) - 1 or idx[i] != -1:
                    raise ValueError("Unmatched outer skin index")
                idx[i] = int(row["donor_body"])
    if np.min(idx) < 0:
        raise ValueError("Unmapped skin")
    points = coordinates(case / "current.node")[outer]
    heavy = sorted(
        int(k)
        for k, v in d["donors"].items()
        if v["compact_mean_N"] >= 0.02 * d["compact_58_to_60_s_mean_N"]
    )
    sets = [np.flatnonzero(idx == k) for k in heavy]
    opt = solve(points, sets, np.tan(np.deg2rad(25)), 0.00005)
    out.write_text(
        json.dumps(
            dict(
                donors=heavy,
                material=sha(case / "current.node"),
                matched_loads=sha(donors),
                slope_limit_per_axis_deg=25,
                eligibility="donors each carrying at least two percent of the late supported 32-section weight",
                result=opt,
                scope="Best possible maximum gap after a common rigid rotation and vertical translation, with no outer skin penetration. This bound deliberately allows unbalanced support moments.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(heavy, opt)


if __name__ == "__main__":
    run(*(Path(a).resolve() for a in sys.argv[1:]))

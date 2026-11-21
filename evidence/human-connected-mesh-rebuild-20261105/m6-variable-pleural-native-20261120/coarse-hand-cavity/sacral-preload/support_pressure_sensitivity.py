"""Keep the first-touch weighting from becoming an artificial point preload."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from audit import digest, rows
from mattress import projected_area


def solve(location, gap, area, weight_n, max_pressure):
    # Force fractions sum to one. Enforce both in-plane moment rows and
    # candidate pressure limits, then minimize force-weighted gap.
    cap = max_pressure * area / weight_n
    equality = np.r_[np.ones((1, len(area))), location.T]
    result = linprog(gap, A_eq=equality, b_eq=[1., 0., 0.],
                     bounds=np.c_[np.zeros(len(area)), cap], method="highs")
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    keep = np.flatnonzero(result.x > 1.e-10)
    assert len(keep) and np.max(np.abs(equality @ result.x - [1., 0., 0.])) < 1.e-8
    return dict(active_rows=len(keep),
                weighted_starting_gap_m=float(result.fun),
                furthest_required_gap_m=float(gap[keep].max()),
                entries=[dict(outer_row=int(j), gap_m=float(gap[j]),
                              force_n=float(result.x[j] * weight_n)) for j in keep])


def main():
    p = ArgumentParser(description=__doc__)
    for k in ("run", "reaction", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    report = json.loads(a.reaction.read_text())
    node_path = a.run / "shell-mass.csv.nodes.csv"
    tet_path = a.run / "shell-mass.csv.tets.csv"
    assert digest(node_path) == report["sha256"][node_path.name]
    x = np.array([[row[f"current_{j}_m"] for j in "xyz"] for row in rows(node_path)], dtype=np.float32).astype(float)
    cells = np.array([[int(row[f"parent_node{j}"]) for j in range(4)] for row in rows(tet_path)])
    outer = len(x) // 3
    gap = x[:outer, 2] - report["bed_offset_m"]
    center = np.array(report["com_m"])
    area = projected_area(x, cells, outer)
    point_source = report["possible_normal_distribution_at_threshold"]
    estimated_pressure = [dict(outer_row=rec["outer_shell_row"], normal_force_n=rec["normal_force_n"],
                               projected_area_m2=float(area[rec["outer_shell_row"]]),
                               nominal_stress_pa=rec["normal_force_n"] / float(area[rec["outer_shell_row"]]))
                          for rec in point_source]
    result = {
        "scope": ("Reaction distribution on fixed initial geometry. Area uses original downward-facing "
                  "skin triangles and counts overlapping projections. Pressure caps are illustrative "
                  "engineering bounds, with no claim about tissue tolerances, deformation or load rating. "
                  "This is a lower bound on pressure-weighted engagement with an unchanged center of mass."),
        "input_sha256": {f.name: digest(f) for f in (node_path, tet_path, a.reaction)},
        "three_touch_concentration": estimated_pressure,
        "moderate_pressure_sensitivity": [
            {"assumed_cap_pa": cap, "optimal": solve(x[:outer, :2] - center[:2], gap, area,
                                                    report["weight_n"], cap)}
            for cap in (10000., 25000., 50000., 100000.)],
    }
    data = json.dumps(result, indent=2, allow_nan=False)
    with a.output.open("x") as f:
        f.write(data + "\n")
    for row in result["moderate_pressure_sensitivity"]:
        v = row["optimal"]
        print(row["assumed_cap_pa"], {k: x for k, x in v.items() if k != "entries"} if v else None)


if __name__ == "__main__":
    main()

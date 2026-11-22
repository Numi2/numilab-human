"""Bound what the current unilateral bed footprint can support at zero net roll and pitch."""

import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from bind_posterior_equilibrium import coordinates
from material_context import sha


def bound(native, case, bed, donors, out):
    req = json.loads((bed / "load.json").read_text())
    target = json.loads(donors.read_text())
    env = json.loads((native / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    mp = Path(env[prefix + "COMBINED_NODE_MAP_CSV"])
    if sha(mp) != target["dense_map"] or sha(native / "launch.json") != target["fine"]:
        raise ValueError("Unmatched donor partition")
    nodes = np.array(req["nodes"], dtype=int)
    xyz = coordinates(case / "current.node")[nodes]
    active = {int(n): i for i, n in enumerate(nodes)}
    groups = np.full(len(nodes), -1, dtype=int)
    with mp.open() as stream:
        for row in csv.DictReader(stream):
            v = int(row["joint_node_id"]) - 1
            if v in active and row["registered_surface_layer"] == "O":
                i = active[v]
                if groups[i] != -1:
                    raise ValueError("Duplicate active skin node")
                groups[i] = int(row["donor_body"])
    if groups.min() < 0:
        raise ValueError("Missing active skin donor")
    weight = req["weight_N"]
    com = np.asarray(req["accounted_com_xy_m"])
    xy = xyz[:, :2] - com
    A = np.vstack([np.ones(len(nodes)), xy.T])
    b = np.array([weight, 0.0, 0.0])
    real = {int(k): v["compact_mean_N"] for k, v in target["donors"].items()}
    bounds = {}
    for group in sorted(real):
        part = (groups == group).astype(float)
        endpoints = []
        for sign in (1.0, -1.0):
            sol = linprog(sign * part, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
            if not sol.success or np.max(abs(A @ sol.x - b)) > 1e-6:
                raise ValueError("No balanced unilateral solution")
            endpoints.append(float(part @ sol.x))
        bounds[group] = dict(
            reference_N=real[group],
            reachable_min_N=endpoints[0],
            reachable_max_N=endpoints[1],
            present_rows=int(part.sum()),
            current_N=float(
                target["donors"][str(group)]["dense"]["reference"]["bed_N"]
            ),
        )
    # Minimum L1 discrepancy; donor extrema above do not generally coexist.
    graph = np.stack([(groups == k).astype(float) for k in sorted(real)])
    reference = np.array([real[k] for k in sorted(real)])
    n = len(real)
    eq = np.block([[A, np.zeros((3, 2 * n))], [graph, np.eye(n), -np.eye(n)]])
    rhs = np.r_[b, reference]
    joint = linprog(
        np.r_[np.zeros(len(nodes)), np.ones(2 * n)],
        A_eq=eq,
        b_eq=rhs,
        bounds=(0, None),
        method="highs",
    )
    if not joint.success or np.max(abs(eq @ joint.x - rhs)) > 1e-6:
        raise ValueError("No joint support bound")
    for key, value in zip(sorted(real), graph @ joint.x[: len(nodes)]):
        bounds[key]["one_joint_minimum_error_solution_N"] = float(value)
    res = dict(
        bed=sha(bed / "load.json"),
        donors=sha(donors),
        current_pose=sha(case / "current.node"),
        rows=len(nodes),
        nodes_hash=sha(mp),
        force_N=weight,
        com_xy_m=com.tolist(),
        joint_minimum_total_absolute_donor_discrepancy_N=float(joint.fun),
        donor_bounds=bounds,
        scope="Optimistic lower bound using all nonnegative normal tractions at existing native contact rows. No traction cap, tissue stiffness or scripted forces; this calculation does not change the scene.",
    )
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(
        joint.fun,
        [
            (
                k,
                round(bounds[k]["reachable_min_N"]),
                round(bounds[k]["reachable_max_N"]),
            )
            for k in sorted(real)
            if bounds[k]["present_rows"]
        ],
    )


if __name__ == "__main__":
    bound(*(Path(a).resolve() for a in sys.argv[1:]))

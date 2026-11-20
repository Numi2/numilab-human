"""Fit a small material-interface shift at the shared hand witness node.

The constraints cover both materials in the complete node star. The skin
and fat volumes, exterior and deep surfaces, topology and node IDs stay fixed.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from fit_hand_cavity import PARENT, ENDPOINT, ROWS, determinant, quality, table, digest

NODE = 4672
UM = 1.e-6


def coordinates(rows, prefix):
    return np.array([[float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"] for row in rows])


def gradient(points, tets):
    base = determinant(points, tets)
    derivative = []
    for axis in range(3):
        shifted = points.copy()
        shifted[NODE, axis] += UM
        derivative.append(determinant(shifted, tets) - base)
    return base, np.stack(derivative, axis=1)


def main():
    nodes_path = PARENT.with_name(PARENT.name + ".nodes.csv")
    cells_path = PARENT.with_name(PARENT.name + ".tets.csv")
    cells = table(cells_path)
    parent = table(nodes_path)
    endpoint = ENDPOINT / "accepted-parent.csv"
    assert digest(endpoint) == json.loads((ENDPOINT / "parent-endpoint.json").read_text())["node_table_sha256"]
    reference = coordinates(parent, "rest")
    world = coordinates(parent, "current")
    accepted = coordinates(table(endpoint), "current")
    topology = np.array([[int(c[f"parent_node{i}"]) for i in range(4)] for c in cells])
    star = np.flatnonzero(np.any(topology == NODE, axis=1))
    tets = topology[star]
    assert len(star) == 36 and set(ROWS).issubset(star)
    assert len(parent) == 3 * 2411 and NODE // 2411 == 1
    material = np.array([int(cells[i]["material_index"]) for i in star])
    target = np.isin(star, ROWS)
    jacobian, derivatives = gradient(reference, tets)
    initial, derivatives_initial = gradient(world, tets)
    latest, derivatives_latest = gradient(accepted, tets)
    assert np.all(jacobian > 0.) and np.all(initial > 0.) and np.all(latest > 0.)

    # Because one vertex moves, all determinant constraints are affine. Use
    # per-cell scaling to keep the sliver constraints numerically significant.
    keep = np.where(target, 1.5, .90)
    matrices, limits = [], []

    def inequality(a, b):
        matrices.append(a / jacobian[:, None])
        limits.append(b / jacobian)

    inequality(-derivatives, jacobian * (1. - keep))
    inequality(-derivatives_initial, initial * (1. - keep))
    inequality(derivatives_latest - 2.25 * derivatives_initial, 2.25 * initial - latest)
    inequality(.35 * derivatives_initial - derivatives_latest, latest - .35 * initial)
    skin = material == material.min()
    area_sum = derivatives[skin].sum(axis=0)
    assert np.linalg.norm(area_sum + derivatives[~skin].sum(axis=0)) < 1.e-15

    # Minimize the L1 coordinate move with a strict bound on the maximum
    # offset. The last three variables keep the solve a plain linear program.
    zero = np.zeros((len(jacobian), 3))
    constraints = np.concatenate((np.concatenate(matrices), np.tile(zero, (len(matrices), 1))), axis=1)
    axes = np.eye(3)
    search = linprog(
        [0., 0., 0., 1., 1., 1.],
        A_ub=np.concatenate((constraints, np.c_[axes, -axes], np.c_[-axes, -axes])),
        b_ub=np.concatenate((*limits, np.zeros(6))),
        A_eq=np.r_[area_sum / np.linalg.norm(area_sum), np.zeros(3)][None],
        b_eq=[0.],
        bounds=[(-40., 40.)] * 3 + [(0., 40.)] * 3,
        method="highs",
    )
    print("local_interface_fit", search.message)
    if not search.success:
        return
    delta = search.x[:3] * UM
    assert np.linalg.norm(delta) <= 40.e-6
    new_ref = reference.copy()
    new_ref[NODE] += delta
    new_world = world.copy()
    # Native Float32 upload is tested after adding the offset to the exact
    # Float64 rest coordinate, which is what the compiler actually rounds.
    new_world[NODE] = new_ref[NODE].astype(np.float32)
    new_accepted = accepted.copy()
    new_accepted[NODE] += new_world[NODE] - world[NODE]
    before_quality = quality(world, tets)
    after_quality = quality(new_world, tets)
    after_ref_jac = determinant(new_ref, tets)
    new_current_jac = determinant(new_world, tets)
    new_endpoint_j = determinant(new_accepted, tets) / new_current_jac
    layer_volume_changes = {
        str(mat): float(np.sum(after_ref_jac[material == mat] - jacobian[material == mat]) / 6.)
        for mat in sorted(set(material))
    }
    assert max(map(abs, layer_volume_changes.values())) <= 2.e-18
    assert np.all(after_ref_jac > .895 * jacobian)
    assert np.all(new_current_jac > .895 * initial)
    assert np.all(new_endpoint_j > .35) and np.all(new_endpoint_j < 2.26)
    result = {
        "parent_nodes_sha256": digest(nodes_path),
        "parent_tets_sha256": digest(cells_path),
        "native_endpoint_sha256": digest(endpoint),
        "local_node": NODE,
        "source_layer": 1,
        "reference_world_m": reference[NODE].tolist(),
        "initial_world_m": world[NODE].tolist(),
        "world_delta_m": delta.tolist(),
        "delta_norm_um": float(np.linalg.norm(delta) / UM),
        "incident_rows": star.tolist(),
        "minimum_quality_before": float(min(before_quality)),
        "minimum_quality_after": float(min(after_quality)),
        "witness_quality_before": before_quality[target].tolist(),
        "witness_quality_after": after_quality[target].tolist(),
        "witness_endpoint_j_before": (latest / initial)[target].tolist(),
        "witness_endpoint_j_after": new_endpoint_j[target].tolist(),
        "endpoint_j_range": [float(np.min(new_endpoint_j)), float(np.max(new_endpoint_j))],
        "layer_volume_change_m3": layer_volume_changes,
    }
    output = Path(__file__).with_name("hand-interface-fit.json")
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

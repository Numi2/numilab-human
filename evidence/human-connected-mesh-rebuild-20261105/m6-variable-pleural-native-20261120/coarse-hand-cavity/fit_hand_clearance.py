"""Screen one connected three-layer repair for the reproduced bone contacts."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fit_hand_cavity import EDGES, PARENT, determinant, quality, table, digest
from fit_hand_interface import coordinates

HERE = Path(__file__).resolve().parent
ROLE_COUNT = 2411
MM = .001


def main():
    node_path = PARENT.with_name(PARENT.name + ".nodes.csv")
    cell_path = PARENT.with_name(PARENT.name + ".tets.csv")
    witness_path = HERE / "connected-interface-replay-solid-witness/hand-solid.csv"
    ref = coordinates(table(node_path), "rest")
    cells = table(cell_path)
    topology = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells])
    connected = json.loads((HERE / "connected-interface-fit.json").read_text())
    starting = ref.copy()
    for row in connected["nodes"]:
        starting[row["local_node"]] += row["world_delta_m"]
    normal_by_node = {}
    with witness_path.open() as src:
        for row in csv.DictReader(src):
            if row["inside_rays"] != "3":
                continue
            local = int(row["layer"]) * ROLE_COUNT + int(row["row"])
            normal = np.array([float(row[f"force_{axis}_N"]) for axis in "xyz"])
            normal /= np.linalg.norm(normal)
            normal_by_node[local] = (normal, float(row["closest_distance_m"]) / MM + .05)
    assert len(normal_by_node) == 4
    seeds = np.array(sorted(normal_by_node))
    columns = np.unique(topology[np.isin(topology, seeds).any(axis=1)] % ROLE_COUNT)
    moved = np.sort(np.concatenate([columns + role * ROLE_COUNT for role in range(3)]))
    incident = np.flatnonzero(np.isin(topology, moved).any(axis=1))
    tets = topology[incident]
    assert 4 <= len(moved) <= 240 and 4 <= len(incident) <= 2400
    row, corner, moving_row = np.nonzero(tets[:, :, None] == moved)
    shell_role_rows = np.searchsorted(moved, np.vstack([columns + i * ROLE_COUNT for i in range(3)]))
    a = np.searchsorted(moved, seeds)
    directions, distances = map(np.array, zip(*(normal_by_node[int(x)] for x in seeds), strict=True))
    d0 = determinant(starting, tets)
    assert np.all(d0 > 0.)
    materials = np.array([int(cells[i]["material_index"]) for i in incident])
    weights = np.stack([np.where(materials == k, d0, 0.) for k in (15, 16)])
    layer_ref_volume_m3 = weights.sum(axis=1) / 6.
    weights /= weights.sum(axis=1, keepdims=True)
    span = len(moved) * 3
    ncols = len(columns)
    depth_limits = np.repeat([[.50001, .50001], [2.49999, 9.99999]], ncols, axis=1)
    clearance_jac = np.zeros((4, len(moved), 3))
    clearance_jac[np.arange(4), a] = directions
    clearance_jac = clearance_jac.reshape(4, span)
    changed_base = starting[tets]
    ea, eb = np.array(EDGES).T
    base_diameter = np.linalg.norm(changed_base[:, ea] - changed_base[:, eb], axis=2).max(axis=1)
    edge_incidence = np.eye(4)[ea] - np.eye(4)[eb]

    def shape(x, grad=False):
        p = changed_base.copy()
        p[row, corner] += x.reshape(-1, 3)[moving_row] * MM
        edges = p[:, ea] - p[:, eb]
        size = np.linalg.norm(edges, axis=2)
        ratio = size / base_diameter[:, None]
        if not grad:
            return (evaluate(x)[:, None] - .89 * ratio**3).ravel()
        slope = (3. * MM / base_diameter[:, None]**3) * size
        deriv = slope[:, :, None] * edges
        jac = np.zeros((len(tets), len(EDGES), len(moved), 3))
        jac[row, :, moving_row, :] = edge_incidence[:, corner].T[:, :, None] * deriv[row]
        return (evaluate(x, True)[:, None, :] - .89 * jac.reshape(len(tets), len(EDGES), span)).reshape(-1, span)

    def evaluate(x, grad=False):
        delta = x.reshape(-1, 3)
        p = changed_base.copy()
        p[row, corner] += delta[moving_row] * MM
        u, v, w = (p[:, 1:] - p[:, :1]).transpose(1, 0, 2)
        u_cross = np.cross(v, w)
        if not grad:
            return np.einsum("ij,ij->i", u, u_cross) / d0
        v_cross, w_cross = np.cross(w, u), np.cross(u, v)
        derivatives = np.stack((-u_cross-v_cross-w_cross, u_cross, v_cross, w_cross), axis=1)
        gradient = np.zeros((len(tets), len(moved), 3))
        gradient[row, moving_row] = derivatives[row, corner] * (MM / d0[row, None])
        return gradient.reshape(len(tets), span)

    def depth(x, grad=False):
        p = starting[moved] / MM + x.reshape(-1, 3)
        differences = (p[shell_role_rows[:-1]] - p[shell_role_rows[1:]]).reshape(-1, 3)
        lengths = np.linalg.norm(differences, axis=1)
        if not grad:
            return lengths
        direction = differences / np.maximum(lengths[:, None], 1.e-8)
        jac = np.zeros((2*ncols, len(moved), 3))
        r = np.arange(2*ncols)
        jac[r, shell_role_rows[:-1].ravel()] = direction
        jac[r, shell_role_rows[1:].ravel()] = -direction
        return jac.reshape(2*ncols, span)

    def inequalities(x, target, grad=False):
        jacs = evaluate(x, grad)
        thick = depth(x, grad)
        if grad:
            return np.concatenate((jacs, thick, -thick, clearance_jac, shape(x, True)))
        return np.concatenate((jacs - .92, thick - depth_limits[0],
                               depth_limits[1] - thick, clearance_jac @ x - distances * target, shape(x)))

    def volume(x, grad=False):
        if grad:
            return weights @ evaluate(x, True)
        return weights @ (evaluate(x) - 1.)

    x = np.zeros(span)
    steps = []
    for target in (.25, .5, .75, 1.):
        result = minimize(
            lambda x: .5 * (x @ x), x, jac=lambda x: x,
            constraints=[
                {"type": "ineq", "fun": lambda x: inequalities(x, target),
                 "jac": lambda x: inequalities(x, target, True)},
                {"type": "eq", "fun": volume, "jac": lambda x: volume(x, True)},
            ],
            bounds=[(-3., 3.)] * span, method="SLSQP",
            options={"maxiter": 60, "ftol": 1e-11},
        )
        minimum = float(inequalities(result.x, target).min())
        print(f"clearance={target} feasible={minimum>=-2e-7} minimum={minimum:.4g} {result.message}",
              flush=True)
        steps.append({"fraction":target,"minimum_constraint":minimum,"volume_rel_error":float(np.max(np.abs(volume(result.x)))),"message":result.message})
        if minimum < -2e-7 or np.any(np.abs(volume(result.x) * layer_ref_volume_m3) > 5.e-13):
            break
        x = result.x

    candidate = starting.copy()
    candidate[moved] += x.reshape(-1, 3) * MM
    changed = determinant(candidate, tets) - d0
    emitted = candidate.astype(np.float32).astype(np.float64)
    before_f32 = starting.astype(np.float32).astype(np.float64)
    exact_q = quality(emitted, tets) / quality(before_f32, tets)
    assert len(set(moved)) == len(moved)
    achieved = clearance_jac @ ((candidate[moved] - starting[moved]).ravel() / MM) / distances
    assert np.allclose(achieved, clearance_jac @ x / distances, atol=1.e-9, rtol=0.)
    certified = float(min(achieved)) >= .999999 and float(min(exact_q)) >= .88
    report = {
        "registered_parent_nodes_sha256": digest(node_path),
        "registered_parent_cells_sha256": digest(cell_path),
        "closed_solid_witness_sha256": digest(witness_path),
        "connected_reference_fit_sha256": digest(HERE / "connected-interface-fit.json"),
        "solver_stages": steps,
        "moved_columns": int(ncols),
        "complete_star_cells": len(incident),
        "min_emitted_relative_quality": float(min(exact_q)),
        "minimum_authored_det_ratio": float(min(determinant(candidate, tets) / d0)),
        "layer_reference_volume_change_m3": [float(changed[materials==k].sum()/6) for k in (15,16)],
        "modeled_clearance_fraction_each_seed": dict(zip(map(str,seeds),achieved.tolist(),strict=True)),
        "maximum_shift_mm": float(np.linalg.norm(x.reshape(-1,3),axis=1).max()),
        "tangent_plane_fit_pass": bool(certified),
        "scope": "A local tangent-plane contact fit. Native bank clearance, unchanged layer intersections, and registered provenance must be checked independently.",
    }
    if certified:
        report["candidate_nodes"] = [
            {"local_node":int(k),"source_world_m":ref[k].tolist(),
             "world_delta_m":(candidate[k]-ref[k]).tolist()} for k in moved
        ]
    out = HERE / "hand-clearance-fit.json"
    with out.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")

if __name__ == "__main__":
    main()

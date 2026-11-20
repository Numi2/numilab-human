"""Fit a connected cluster of skin/fat interface nodes around the hand witness."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fit_hand_cavity import PARENT, ENDPOINT, ROWS, determinant, quality, table, digest
from fit_hand_interface import NODE, UM, coordinates

HERE = Path(__file__).resolve().parent
COUNT = 2411
LIMIT = 90.
KEEP = .90
MIN_J = .35
MAX_J = 1.9


def depths(points, rows):
    outer = points[rows - COUNT]
    inner = points[rows]
    deep = points[rows + COUNT]
    return np.c_[np.linalg.norm(outer - inner, axis=1), np.linalg.norm(inner - deep, axis=1)]


def main():
    parent_path = PARENT.with_name(PARENT.name + ".nodes.csv")
    cell_path = PARENT.with_name(PARENT.name + ".tets.csv")
    endpoint_path = ENDPOINT / "accepted-parent.csv"
    assert digest(endpoint_path) == json.loads(
        (ENDPOINT / "parent-endpoint.json").read_text())["node_table_sha256"]
    parent, cells = table(parent_path), table(cell_path)
    ref = coordinates(parent, "rest")
    current = coordinates(parent, "current")
    endpoint = coordinates(table(endpoint_path), "current")
    topology = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells])
    incident = np.any(topology == NODE, axis=1)
    vertices = np.unique(topology[incident])
    moved = vertices[(vertices >= COUNT) & (vertices < 2 * COUNT)]
    stars = np.flatnonzero(np.any(np.isin(topology, moved), axis=1))
    tets = topology[stars]
    target = np.isin(stars, ROWS)
    materials = np.array([int(cells[cell]["material_index"]) for cell in stars])
    skin = materials == 15
    assert len(parent) == 3 * COUNT and set(materials) == {15, 16}
    assert set(ROWS).issubset(stars) and NODE in moved
    assert len(moved) <= 30 and len(stars) <= 600
    row, corner, node = np.nonzero(tets[:, :, None] == moved)
    d0 = determinant(ref, tets)
    di0 = determinant(current, tets)
    assert np.all(d0 > 0) and np.all(di0 > 0)
    normal_refs = (d0, di0, di0)
    points = (ref, current, endpoint)

    def evaluate(offset, with_gradients=False):
        offset = offset.reshape(-1, 3) * UM
        values, gradients = [], []
        for p, normal in zip(points, normal_refs, strict=True):
            vertices = p[tets]
            vertices[row, corner] += offset[node]
            edges = vertices[:, 1:] - vertices[:, :1]
            u, v, w = edges.transpose(1, 0, 2)
            a = np.cross(v, w)
            values.append(np.einsum("ij,ij->i", u, a) / normal)
            if with_gradients:
                b, c = np.cross(w, u), np.cross(u, v)
                deriv = np.stack((-a-b-c, a, b, c), axis=1)
                grad = np.zeros((len(stars), len(moved), 3))
                grad[row, node] = deriv[row, corner] * (UM / normal[row, None])
                gradients.append(grad.reshape(len(stars), -1))
        return gradients if with_gradients else values

    def inequality(offset, goal, grad=False):
        a, b, c = evaluate(offset, grad)
        if not grad:
            a = a - np.where(target, goal, KEEP)
            b = b - np.where(target, goal, KEEP)
            ref_ratio = np.where(target, goal, KEEP)
        else:
            ref_ratio = 0
        return np.concatenate((a, b, MAX_J * (b + ref_ratio) - c, c - MIN_J * (b + ref_ratio)))

    weights = d0[skin] / d0[skin].sum()

    def volume(offset, grad=False):
        entries = evaluate(offset, grad)[0][skin]
        return weights @ (entries if grad else entries - 1.)

    limited = np.zeros(3 * len(moved))
    selected = None
    for goal in (2.25, 2.75, 3.25):
        result = minimize(
            lambda x: .5 * (x @ x), limited, jac=lambda x: x,
            constraints=[
                {"type": "ineq", "fun": lambda x: inequality(x, goal),
                 "jac": lambda x: inequality(x, goal, True)},
                {"type": "eq", "fun": volume, "jac": lambda x: volume(x, True)},
            ],
            bounds=[(-LIMIT, LIMIT)] * len(limited), method="SLSQP",
            options={"maxiter": 120, "ftol": 1e-10},
        )
        print("connected_fit", goal, result.message, flush=True)
        if np.min(inequality(result.x, goal)) < -2e-7 or abs(volume(result.x)) > 1e-12:
            break
        shifts = result.x.reshape(-1, 3) * UM
        if np.max(np.linalg.norm(shifts, axis=1)) > LIMIT * UM:
            break
        next_ref = ref.copy()
        next_ref[moved] += shifts
        next_pos = current.copy()
        next_pos[moved] = next_ref[moved].astype(np.float32)
        next_endpoint = endpoint.copy()
        next_endpoint[moved] += next_pos[moved] - current[moved]
        exact_ref = determinant(next_ref, tets)
        exact_current = determinant(next_pos, tets)
        sizes = np.where(target, goal, KEEP) - .01
        if np.any(exact_ref < d0 * sizes) or np.any(exact_current < di0 * sizes):
            break
        endpoint_j = determinant(next_endpoint, tets) / exact_current
        if np.any(endpoint_j < MIN_J) or np.any(endpoint_j > MAX_J + .02):
            break
        volume_errors = {
            str(material): float(np.sum((exact_ref - d0)[materials == material]) / 6.)
            for material in sorted(set(materials))
        }
        assert max(map(abs, volume_errors.values())) < 2e-18
        assert np.array_equal(next_ref[:COUNT], ref[:COUNT])
        assert np.array_equal(next_ref[2*COUNT:], ref[2*COUNT:])
        q = quality(next_pos, tets)
        assert np.all(q >= .88 * quality(current, tets))
        # Check the straight displacement path as well as its endpoint.
        for fraction in np.linspace(0., 1., 11):
            path = ref.copy()
            path[moved] += shifts * fraction
            assert np.min(determinant(path, tets) / d0) >= KEEP - .01
        selected = {
            "parent_nodes_sha256": digest(parent_path),
            "parent_tets_sha256": digest(cell_path),
            "native_endpoint_sha256": digest(endpoint_path),
            "source_layer": 1,
            "target_minimum_determinant_multiple": goal,
            "unchanged_neighbor_minimum_multiple": KEEP,
            "max_displacement_um": float(np.max(np.linalg.norm(shifts, axis=1)) / UM),
            "incident_cells": list(map(int, stars)),
            "nodes": [
                {"local_node": int(index), "reference_world_m": ref[index].tolist(),
                 "world_delta_m": delta.tolist()}
                for index, delta in zip(moved, shifts, strict=True)
            ],
            "thickness_ranges_before_m": [depths(ref, moved).min(axis=0).tolist(),
                                          depths(ref, moved).max(axis=0).tolist()],
            "thickness_ranges_after_m": [depths(next_ref, moved).min(axis=0).tolist(),
                                         depths(next_ref, moved).max(axis=0).tolist()],
            "layer_volume_change_m3": volume_errors,
            "witness_quality_before": quality(current, tets)[target].tolist(),
            "witness_quality_after": q[target].tolist(),
            "witness_endpoint_j_before": (determinant(endpoint, tets) / di0)[target].tolist(),
            "witness_endpoint_j_after": endpoint_j[target].tolist(),
            "minimum_quality_before": float(quality(current, tets).min()),
            "minimum_quality_after": float(q.min()),
            "endpoint_j_range": [float(endpoint_j.min()), float(endpoint_j.max())],
        }
        limited = result.x
    assert selected is not None, "no connected candidate qualified for native replay"
    output = HERE / "connected-interface-fit.json"
    with output.open("x") as f:
        json.dump(selected, f, indent=2, allow_nan=False)
        f.write("\n")

if __name__ == "__main__":
    main()

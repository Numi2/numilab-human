"""Screen mass-conserving internal-node authoring against measured bed-load failures."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from mattress import Mattress, projected_area
from predict import ROOT, SeedEnergy, digest, read_model, read_table, solve


def determinants(points, cells):
    vertices = points[cells]
    return np.linalg.det(vertices[:, 1:] - vertices[:, :1])


def determinant_columns(points, cells, node):
    base = determinants(points, cells)
    perturbations = np.repeat(points[None], 3, axis=0)
    perturbations[np.arange(3), node, np.arange(3)] += .001
    # Determinants are affine when only one vertex of each tetrahedron moves.
    return base, np.stack([determinants(points, cells) - base for points in perturbations], axis=1)


def fit_node(rest, world, loaded, tets, rows, labels, node, witness, limits, expand):
    cells = tets[rows]
    index = np.flatnonzero(rows == witness).item()
    base, gradient = determinant_columns(rest, cells, node)
    pose, pose_gradient = determinant_columns(world, cells, node)
    target, target_gradient = determinant_columns(loaded, cells, node)
    minimum_j, maximum_j = limits

    # Keep separate reference volumes for each material and anatomical donor.
    banks = defaultdict(list)
    for local, row in enumerate(rows):
        label = labels[row]
        banks[tuple(label[key] for key in
                    ("material_index", "geometric_donor_body", "anatomical_region_id"))].append(local)
    equality = []
    for indices in banks.values():
        summed = gradient[indices].sum(axis=0)
        if np.linalg.norm(summed) > 1.e-14:
            equality.append(summed / np.linalg.norm(summed))

    lhs = np.vstack((-gradient, -pose_gradient,
                     minimum_j * gradient - target_gradient,
                     target_gradient - maximum_j * gradient))
    rhs = np.r_[.1 * base, .1 * pose, target - minimum_j * base, maximum_j * base - target]
    scale = np.tile(base, 4)
    # Moving the stretched cell toward larger reference volume reduces its
    # sensitivity to displacement. The compressed cell needs actual J margin.
    objective = gradient[index] if expand else target_gradient[index] - minimum_j * gradient[index]
    candidate = linprog(-objective / base[index], A_ub=lhs / scale[:, None], b_ub=rhs / scale,
                        A_eq=equality or None, b_eq=np.zeros(len(equality)) if equality else None,
                        bounds=[(-.5, .5)] * 3, method="highs")
    if not candidate.success:
        raise ValueError(candidate.message)

    delta = candidate.x * .001
    updated = []
    for points in (rest, world, loaded):
        moved = points.copy()
        moved[node] += delta
        updated.append(determinants(moved, cells))
    new_base, new_pose, new_target = updated
    np.testing.assert_allclose(new_base, base + gradient @ candidate.x, rtol=1.e-10, atol=1.e-20)
    np.testing.assert_allclose(new_pose, pose + pose_gradient @ candidate.x, rtol=1.e-10, atol=1.e-20)
    np.testing.assert_allclose(new_target, target + target_gradient @ candidate.x, rtol=1.e-10, atol=1.e-20)
    assert np.all(new_base >= .89999999 * base)
    assert np.all(new_pose >= .89999999 * pose)
    ratio = new_target / new_base
    assert min(ratio) >= minimum_j - 1.e-9 and max(ratio) <= maximum_j + 1.e-9
    for indices in banks.values():
        np.testing.assert_allclose(new_base[indices].sum(), base[indices].sum(), rtol=1.e-9, atol=1.e-20)
    return delta, base, new_base, ratio[index], list(banks.values())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-iterations", type=int, default=145)
    args = parser.parse_args()
    out = args.output
    out.mkdir()
    baseline, seed, world, rest, tets, pins, model_data = read_model()
    ref = ROOT / "thickness-rebuild-034/compiled-reference"
    node_rows = read_table(ref / "shell-mass.csv.nodes.csv")
    labels = read_table(ref / "shell-mass.csv.labels.csv")
    tet_mass = np.array([float(row["parent_mass_kg"]) for row in read_table(ref / "shell-mass.csv.tets.csv")])
    node_mass = np.array([float(row["compiled_lumped_mass_kg"]) for row in node_rows])
    new_tet_mass = tet_mass.copy()
    adjusted_rest, adjusted_world = rest.copy(), world.copy()
    fits = []
    for witness, node, run, expand in ((20754, 3703, "mattress-confirm-17", True),
                                       (22244, 4034, "rigid-confirm-16", False)):
        trace = Path(run) / "seed.npz"
        pins[str(trace)] = digest(trace)
        with np.load(trace, allow_pickle=False) as saved:
            loaded = adjusted_world + saved["displacement_m"]
            initial_ratio = float(saved["deformation_j"][witness])
        rows = np.flatnonzero(np.any(tets == node, axis=1))
        assert baseline.outer <= node < 2 * baseline.outer
        assert np.array_equal(adjusted_rest[tets[rows]], rest[tets[rows]])
        delta, original, moved, ratio, banks = fit_node(adjusted_rest, adjusted_world, loaded, tets, rows,
                                                       labels, node, witness, (.3500001, 2.4999999), expand)
        adjusted_rest[node] += delta
        adjusted_world[node] += delta
        masses = tet_mass[rows] * moved / original
        for indices in banks:
            quota = tet_mass[rows[indices]].sum()
            masses[indices] *= quota / masses[indices].sum()
            np.testing.assert_allclose(masses[indices].sum(), quota, rtol=5.e-15, atol=1.e-18)
        new_tet_mass[rows] = masses
        fits.append({"witness": witness, "node": node, "move_mm": (1.e3 * delta).tolist(),
                     "J_old_loaded_state": initial_ratio, "J_reauthored_at_same_displacement": ratio,
                     "reference_volume_ratio": float((moved / original)[np.flatnonzero(rows == witness).item()]),
                     "maximum_relative_density_change": float(np.max(abs(masses / tet_mass[rows] * original / moved - 1.))),
                     "incident_tetrahedra": len(rows)})

    change = np.bincount(tets.ravel(), weights=np.repeat((new_tet_mass - tet_mass) / 4., 4),
                         minlength=len(node_mass))
    mass = node_mass + change
    assert np.all(mass > 0.) and abs(mass.sum() - node_mass.sum()) < 1.e-12
    # The same Float32 emission policy is used for the new current coordinates.
    adjusted_world = adjusted_world.astype(np.float32).astype(np.float64)
    moved_com = ((mass @ adjusted_world - node_mass @ world) / model_data["total_compiled_mass_kg"]
                 + np.array(model_data["body_com_m"]))
    assert np.array_equal(adjusted_world[:baseline.outer], world[:baseline.outer])
    assert np.array_equal(adjusted_world[2 * baseline.outer:], world[2 * baseline.outer:])
    assert np.all(determinants(adjusted_world, tets) > 0.) and np.all(determinants(adjusted_rest, tets) > 0.)
    model = SeedEnergy(adjusted_world, adjusted_rest, tets, baseline.mu, baseline.bulk,
                       mass, baseline.outer, model_data["total_compiled_mass_kg"], moved_com,
                       baseline.width, baseline.dt)
    new_seed = seed.copy()
    old_seed_rows = np.argsort(baseline.positions(seed)[1])[:3]
    new_seed[model.q] = np.linalg.solve(model.a[old_seed_rows], baseline.positions(seed)[1][old_seed_rows]
                                        - model.base_gap[old_seed_rows])
    np.savez_compressed(out / "candidate.npz", rest=adjusted_rest, world=adjusted_world, tet_mass=new_tet_mass,
                        node_mass=mass)
    runs = {}
    for use_mattress, loads in ((False, (.05, .06, .075, .1)), (True, (.025, .05, .06, .075, .1))):
        mode = "mattress" if use_mattress else "rigid"
        start = new_seed.copy()
        if use_mattress:
            model.mattress = Mattress(projected_area(adjusted_world, tets, model.outer), 10000., .12)
            start[model.q[0]] -= .001
        terminal, traces = solve(model, start, args.max_iterations, weights=loads, guard=use_mattress)
        _, gap, jac = model.positions(terminal)
        (out / (mode + "-trace.json")).write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n")
        runs[mode] = {"last": traces[-1], "critical_minimum_cell": int(jac.argmin()),
                      "critical_maximum_cell": int(jac.argmax()), "minimum_nominal_gap_m": float(gap.min()),
                      "converged_loads": [row["load_fraction"] for row in traces if row.get("outcome") == "converged"]}
    report = {"candidate_sha256": digest(out / "candidate.npz"), "input_sha256": pins, "fits": fits,
              "unchanged_exterior_and_deep_nodes": True, "unchanged_tetrahedron_topology": True,
              "original_mass_kg": float(node_mass.sum()), "new_mass_kg": float(mass.sum()),
              "maximum_lumped_mass_change_kg": float(abs(change).max()), "offline_runs": runs}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()

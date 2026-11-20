"""Fit material layers with separate current and rest geometry validity."""
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity, vstack

from fit_hand_cavity import determinant, quality
from fit_refined_patch import determinant_gradient, layer_bounds, sparse_jacobian

EDGES = np.array(list(combinations(range(4), 2)))


def fit_layers(world, reference, cells, parent, regional, proposals, columns, *, all_edges=False):
    count = len(proposals)
    assert count > 0
    node_id = np.full(len(world), -1, dtype=np.int32)
    direction = np.zeros_like(world)
    target = np.array([np.clip(delta, -.01, .014) for _, delta, _, _ in proposals])
    importance = np.array([weight for _, _, _, weight in proposals])
    for i, (node, _, axis, _) in enumerate(proposals):
        assert node_id[node] == -1
        node_id[node] = i
        direction[node] = axis
    active = node_id >= 0
    incidence = np.any(active[cells], axis=1)
    local_rows = np.flatnonzero(incidence)
    local = cells[local_rows]
    initial_world_det = determinant(world, cells)
    initial_reference_det = determinant(reference, cells)
    assert np.min(initial_world_det) > 0 and np.min(initial_reference_det) > 0
    minimum_det = (initial_world_det.min(), initial_reference_det.min())
    minimum_quality = (quality(world, cells).min(), quality(reference, cells).min())

    groups = regional.group_for_parent[parent]
    constraints, label = np.unique(groups[incidence], return_inverse=True)
    aggregate = coo_matrix((np.ones(len(local)), (label, np.arange(len(local)))),
                           shape=(len(constraints), len(local))).tocsr()
    initial_pool = np.bincount(groups, weights=initial_reference_det)[constraints]
    absolute_pool_low, absolute_pool_high = regional.determinant_bounds()
    pool_low, pool_high = (bound[constraints] for bound in (absolute_pool_low, absolute_pool_high))
    assert np.all((initial_pool >= pool_low) & (initial_pool <= pool_high))
    # Keep an interior band against Float32 and LP residual roundoff.
    bound_pad = 1.e-7 * initial_pool
    pool_low += bound_pad
    pool_high -= bound_pad

    identity_step = identity(count, format="csr")
    absolute_error = vstack((hstack((identity_step, -identity_step)),
                             hstack((-identity_step, -identity_step))), format="csr")
    thickness, thickness_margin = layer_bounds(world, node_id, direction, count, columns)
    update = np.zeros(count)
    world_state = world.copy()
    reference_state = reference.copy()
    report = []
    objective = np.r_[np.zeros(count), importance]
    index = np.arange(len(local))

    def limiting_rows(measurements, thresholds):
        witnesses = {}
        for frame, values, floor in zip(("world", "rest"), measurements, thresholds, strict=True):
            below = np.flatnonzero(values < floor)
            if len(below):
                order = below[np.argsort(values[below], kind="stable")[:8]]
                witnesses[frame] = {
                    "below_floor": int(len(below)), "floor": float(floor),
                    "worst_children": [[int(local_rows[row]), int(parent[local_rows[row]]),
                                        float(values[row])] for row in order],
                }
        return witnesses

    def constraints_at(state, volume, jac, absolute_minimum, shape, radius):
        corners = state[local]
        edges = corners[:, EDGES[:, 1]] - corners[:, EDGES[:, 0]]
        length = np.einsum("ted,ted->te", edges, edges)
        longest = length.argmax(axis=1)
        gradient = determinant_gradient(corners)
        scale = 1. / np.maximum(volume, 1.e-11)
        rows = [-jac.multiply(scale[:, None]).tocsr()]
        bounds = [(volume - absolute_minimum) * scale]
        edge_indices = range(len(EDGES)) if all_edges else (longest,)
        for edge_index in edge_indices:
            selected = edges[index, edge_index]
            square = length[index, edge_index]
            diameter_gradient = np.zeros_like(gradient)
            diameter_gradient[index, EDGES[edge_index, 1]] = selected
            diameter_gradient[index, EDGES[edge_index, 0]] = -selected
            required = (1.002 * shape) * square**1.5
            offset = gradient - (3. * required / square)[:, None, None] * diameter_gradient
            shape_derivative = sparse_jacobian(offset, local, node_id, direction, count)
            rows.append(-shape_derivative.multiply(scale[:, None]).tocsr())
            bounds.append((volume - required) * scale)
        system = vstack(rows, format="csr")
        bounds = np.concatenate(bounds)
        total = len(bounds)
        if all_edges:
            # Only omit rows that cannot constrain the linearized trust box.
            excursion = radius * np.asarray(abs(system).sum(axis=1)).ravel()
            keep = bounds <= excursion + 1.e-12
            system, bounds = system[keep], bounds[keep]
        return system, bounds, total

    for radius in (.002, .002, .001, .001, .001, .0005, .0005, .00025):
        world_det = determinant(world_state, local)
        ref_det = determinant(reference_state, local)
        world_jac, ref_jac = (sparse_jacobian(determinant_gradient(state[local]), local, node_id, direction, count)
                              for state in (world_state, reference_state))
        world_system, world_bound, world_rows = constraints_at(world_state, world_det, world_jac,
                                                  minimum_det[0], minimum_quality[0], radius)
        ref_system, ref_bound, rest_rows = constraints_at(reference_state, ref_det, ref_jac,
                                              minimum_det[1], minimum_quality[1], radius)
        pool = initial_pool + aggregate @ (ref_det - initial_reference_det[incidence])
        pool_grad = aggregate @ ref_jac
        pool_rows = pool_grad.multiply((1. / initial_pool)[:, None])
        ineq = vstack((world_system, ref_system, pool_rows, -pool_rows, thickness), format="csr")
        rem = target - update
        ineq_bound = np.r_[world_bound, ref_bound, (pool_high - pool) / initial_pool,
                           (pool - pool_low) / initial_pool, thickness_margin - thickness @ update]
        system = vstack((absolute_error, 1.e-3 * hstack((ineq, csr_matrix((ineq.shape[0], count))))), format="csr")
        result = linprog(objective, A_ub=system, b_ub=np.r_[1.e3 * rem, -1.e3 * rem, ineq_bound],
                         bounds=[(-1.e3 * radius, 1.e3 * radius)] * count + [(0., None)] * count,
                         method="highs", options={"time_limit": 12.})
        failed = Counter()
        entry = {"radius_m": radius, "status": int(result.status),
                 "geometric_rows": int(world_rows + rest_rows),
                 "retained_geometric_rows": int(len(world_bound) + len(ref_bound)),
                 "trial_failures": failed, "geometric_witnesses": []}
        report.append(entry)
        if not result.success:
            break
        increment = 1.e-3 * result.x[:count]
        error_before = importance @ np.abs(rem)
        for fraction in (1., .5, .25, .125, .0625, .03125):
            trial = update + fraction * increment
            displacement = trial[node_id[active], None] * direction[active]
            current_world = world_state.copy()
            current_reference = reference_state.copy()
            current_world[active] = (world[active] + displacement).astype(np.float32)
            current_reference[active] = reference[active] + displacement
            world_v = determinant(current_world, local)
            ref_v = determinant(current_reference, local)
            actual_pool = initial_pool + aggregate @ (ref_v - initial_reference_det[incidence])
            error = importance @ np.abs(target - trial)
            if error >= error_before:
                failed["objective"] += 1
                continue
            if np.any(thickness @ trial > thickness_margin + 1.e-10):
                failed["world_thickness"] += 1
                continue
            determinant_failures = limiting_rows((world_v, ref_v), minimum_det)
            if determinant_failures:
                failed["determinant"] += 1
                entry["geometric_witnesses"].append({"fraction": fraction, "quantity": "determinant",
                                                      "frames": determinant_failures})
                continue
            shape = (quality(current_world, local), quality(current_reference, local))
            shape_failures = limiting_rows(shape, minimum_quality)
            if shape_failures:
                failed["shape"] += 1
                entry["geometric_witnesses"].append({"fraction": fraction, "quantity": "shape",
                                                      "frames": shape_failures})
                continue
            if np.any(actual_pool < pool_low) or np.any(actual_pool > pool_high):
                failed["mass_capacity"] += 1
                continue
            update, world_state, reference_state = trial, current_world, current_reference
            entry.update(fraction=fraction, weighted_error_before_m=float(error_before), weighted_error_m=float(error))
            break
        else:
            entry["fraction"] = 0.
    return world_state, reference_state, report

"""Fit a connected three-layer patch with simultaneous shape and volume bounds."""
from itertools import combinations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity, vstack

from fit_hand_cavity import determinant, quality

EDGES = np.array(list(combinations(range(4), 2)))


def determinant_gradient(corners):
    a, b, c = (corners[:, 1:] - corners[:, :1]).transpose(1, 0, 2)
    g = np.stack((np.cross(b, c), np.cross(c, a), np.cross(a, b)), axis=1)
    return np.concatenate((-g.sum(axis=1)[:, None], g), axis=1)


def sparse_jacobian(gradient, cells, node_id, direction, count):
    columns = node_id[cells]
    active = columns >= 0
    rows = np.broadcast_to(np.arange(len(cells))[:, None], cells.shape)
    derivative = np.einsum("tcd,tcd->tc", gradient, direction[cells])
    return coo_matrix((derivative[active], (rows[active], columns[active])),
                      shape=(len(cells), count)).tocsr()


def layer_bounds(reference, node_id, directions, count, columns):
    if len(columns) == 0:
        return csr_matrix((0, count)), np.empty(0)
    nodes = np.asarray(columns)
    upper = nodes[:, :-1].ravel()
    lower = nodes[:, 1:].ravel()
    direction = directions[upper]
    thickness = np.einsum("nd,nd->n", reference[upper] - reference[lower], direction)
    # The per-layer registration admits no thickness below 0.5 mm.
    # Any already thinner bisected projected column cannot lose more.
    minimum = np.minimum(thickness, 0.0005)
    rows = np.arange(len(upper))
    keys = (np.r_[rows, rows], np.r_[node_id[upper], node_id[lower]])
    return coo_matrix((np.r_[-np.ones(len(rows)), np.ones(len(rows))], keys),
                      shape=(len(rows), count)).tocsr(), thickness - minimum


def fit_patch(reference, cells, parents, proposals, layer_columns=()):
    if not proposals:
        return reference.copy(), []
    count = len(proposals)
    node_id = np.full(len(reference), -1, dtype=np.int32)
    node_direction = np.zeros_like(reference)
    target = np.array([np.clip(distance, -.01, .01) for _, distance, _, _ in proposals])
    for i, (nodes, _, direction, _) in enumerate(proposals):
        assert np.all(node_id[nodes] < 0)
        node_id[nodes] = i
        node_direction[nodes] = direction
    active = node_id >= 0
    incident = np.flatnonzero(np.any(active[cells], axis=1))
    active_parents, compact_parents = np.unique(parents[incident], return_inverse=True)
    local = cells[incident]
    row_reference_volume = determinant(reference, local)
    row_reference_shape = quality(reference, local)
    parent_reference_volume = np.bincount(parents, weights=determinant(reference, cells))[active_parents]
    aggregate = coo_matrix((np.ones(len(local)), (compact_parents, np.arange(len(local)))),
                           shape=(len(active_parents), len(local))).tocsr()
    inv_reference_volume = (1. / row_reference_volume)[:, None]
    step_identity = identity(count, format="csr")
    simple = vstack((hstack((step_identity, -step_identity)),
                     hstack((-step_identity, -step_identity))), format="csr")
    layer_constraint, layer_margin = layer_bounds(reference, node_id, node_direction, count, layer_columns)
    slacks = csr_matrix((2 * len(local) + 2 * len(active_parents) + len(layer_margin), count))
    current = reference.copy()
    update = np.zeros(count)
    report = []
    for limit in (0.002, 0.002, 0.001, 0.0005):
        corners = current[local]
        volume = determinant(current, local)
        gradient = determinant_gradient(corners)
        differences = corners[:, EDGES[:, 1]] - corners[:, EDGES[:, 0]]
        lengths_squared = np.einsum("ted,ted->te", differences, differences)
        longest = lengths_squared.argmax(axis=1)
        index = np.arange(len(local))
        side = differences[index, longest]
        diameter_squared = lengths_squared[index, longest]
        minimum_ratio = .88
        shape_threshold = (minimum_ratio + .008) * row_reference_shape * diameter_squared**1.5
        diameter_gradient = np.zeros_like(gradient)
        diameter_gradient[index, EDGES[longest, 1]] = side
        diameter_gradient[index, EDGES[longest, 0]] = -side
        margin_gradient = gradient - (3. * shape_threshold / diameter_squared)[:, None, None] * diameter_gradient
        volume_jacobian = sparse_jacobian(gradient, local, node_id, node_direction, count)
        shape_jacobian = sparse_jacobian(margin_gradient, local, node_id, node_direction, count)
        parent_jacobian = aggregate @ volume_jacobian
        parent_gain = np.asarray(aggregate @ (volume - row_reference_volume)).ravel()
        jacobian = vstack((-volume_jacobian.multiply(inv_reference_volume),
                           -shape_jacobian.multiply(inv_reference_volume),
                           parent_jacobian.multiply((1. / parent_reference_volume)[:, None]),
                           -parent_jacobian.multiply((1. / parent_reference_volume)[:, None]),
                           layer_constraint), format="csr")
        remaining = target - update
        bound = np.concatenate((remaining, -remaining,
                                (volume - .76 * row_reference_volume) / row_reference_volume,
                                (volume - shape_threshold) / row_reference_volume,
                                .175 - parent_gain / parent_reference_volume,
                                .175 + parent_gain / parent_reference_volume,
                                layer_margin - layer_constraint @ update))
        # HiGHS works in millimetres to keep feasibility residuals meaningful.
        system = vstack((simple, 1.e-3 * hstack((jacobian, slacks))), format="csr")
        bound[:2 * count] *= 1.e3
        result = linprog(np.r_[np.zeros(count), np.ones(count)], A_ub=system, b_ub=bound,
                         bounds=[(-1.e3 * limit, 1.e3 * limit)] * count + [(0, None)] * count,
                         method="highs", options={"time_limit": 10.})
        iteration = {"trust_m": limit, "status": int(result.status)}
        report.append(iteration)
        if not result.success:
            break
        increment = result.x[:count] * 1.e-3
        previous_error = float(np.abs(remaining).sum())
        for fraction in (1., .5, .25, .125, .0625):
            trial_update = update + fraction * increment
            trial = current.copy()
            trial[active] = (reference[active] + trial_update[node_id[active], None] *
                             node_direction[active]).astype(np.float32)
            trial_volume = determinant(trial, local)
            trial_parent_gain = aggregate @ (trial_volume - row_reference_volume)
            trial_shape = quality(trial, local)
            trial_error = float(np.abs(target - trial_update).sum())
            if np.all(trial_volume >= .75 * row_reference_volume) and np.all(
                    trial_shape >= minimum_ratio * row_reference_shape) and np.all(
                    np.abs(trial_parent_gain) <= .18 * parent_reference_volume) and np.all(
                    layer_constraint @ trial_update <= layer_margin + 1.e-10) and trial_error < previous_error:
                update, current = trial_update, trial
                iteration.update(fraction=fraction, sum_target_error_before_m=previous_error,
                                 sum_target_error_after_m=trial_error)
                break
        else:
            iteration["fraction"] = 0.
    return current, report

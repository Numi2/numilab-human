"""Check cap-aware residual paths against independently solved small networks."""
import json
import time

import numpy as np
from scipy.optimize import linprog

from paired_mass_density_reserve import close_with_density_reserve, distribute_mass
from reflow_paired_mass import csr

generator = np.random.default_rng(5643)
cases = 480
feasible_cases = 0
for case in range(cases):
    n_carrier = int(generator.integers(2, 10))
    n_group = int(generator.integers(2, 10))
    donor, group = np.where(generator.random((n_carrier, n_group)) < 0.55)
    if not len(donor):
        continue
    carrier_incidence = np.eye(n_carrier)[donor].T
    group_incidence = np.eye(n_group)[group].T
    initial = generator.integers(0, 9, len(donor)).astype(float)
    legal = generator.random(len(donor)) > .15
    initial[~legal] = 0.
    flow = initial.copy()
    pending = generator.integers(0, 10, n_carrier).astype(float)

    # Construct known-feasible controls from a second complete assignment.
    limits = group_incidence @ initial
    lower = np.floor(limits * generator.random(n_group))
    upper = limits + generator.integers(1, 13, n_group)
    if case % 2 == 0:
        desired = initial.copy()
        for carrier in range(n_carrier):
            choices = np.flatnonzero((donor == carrier) & legal)
            if len(choices):
                desired[generator.choice(choices)] += pending[carrier]
            else:
                pending[carrier] = 0.0
        expected = group_incidence @ desired
        lower = np.floor(expected * (0.2 + generator.random(n_group)*0.8))
        upper = np.maximum(upper, expected)
    demand = carrier_incidence @ initial + pending
    row_order, row_offsets = csr(donor, n_carrier)
    group_order, group_offsets = csr(group, n_group)
    solution = linprog(
        np.zeros(len(flow)), A_eq=carrier_incidence, b_eq=demand,
        A_ub=np.vstack((group_incidence, -group_incidence)), b_ub=np.r_[upper, -lower],
        bounds=[(0., None) if allowed else (0., 0.) for allowed in legal], method="highs",
    )
    remaining, assigned, detail = close_with_density_reserve(
        flow, donor, group, row_order, row_offsets, group_order, group_offsets,
        pending, limits, lower, upper, lambda edge: legal[edge], time.monotonic()+5.,
    )
    closed = bool(np.max(remaining) < 1e-12 and np.all(assigned >= lower - 1e-12)
                  and np.all(assigned <= upper + 1e-12))
    assert closed == solution.success, (case, closed, solution.message, detail)
    assert np.all(flow >= 0.) and np.all(flow[~legal] == 0.)
    assert np.allclose(carrier_incidence @ flow + remaining, demand, atol=1e-12, rtol=0.)
    assert np.allclose(group_incidence @ flow, assigned, atol=1e-12, rtol=0.)
    feasible_cases += int(closed)

cell_groups = np.array([0, 1, 0, 0, 1])
base = np.array([2., 10., 5., 3., 11.])
minimum = np.array([2., 8., 4., 2.5, 9.])
maximum = np.array([3., 12., 5.5, 3.3, 15.])
target = np.array([8.5, 27.])
result = distribute_mass(base, cell_groups, target, minimum, maximum)
assert np.all(result >= minimum) and np.all(result <= maximum)
assert np.array_equal(np.bincount(cell_groups, weights=result), target)
print(json.dumps({"matched_reference_networks": cases, "feasible_networks": feasible_cases,
                  "distribution_bounds_and_totals": "passed"}))

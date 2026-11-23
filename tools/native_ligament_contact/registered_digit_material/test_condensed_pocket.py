"""An affine patch admits exact interface tractions after eliminating its extra node."""

import json
import numpy as np
from audit_cavity_response import elastic
from screen_condensed_fat_pockets import candidates, interior_equilibrium


def test():
    x = np.array(
        [
            [0, 0, 0],
            [0.006, 0, 0],
            [0.001, 0.006, 0],
            [0.001, 0.001, 0.004],
            [0.002, 0.001, -0.003],
        ],
        dtype="f4",
    ).astype(float)
    old = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    deformation = np.array([[1.08, 0.08, 0.03], [0, 0.94, 0.14], [0.015, 0, 1.02]])
    shift = np.array([0.012, 0.001, 0.024])
    y = x @ deformation + shift
    mu, bulk = 11000.0, 55000.0
    energy, force, _ = elastic(x, y, old, np.arange(len(x)), mu, bulk)
    results = []
    for ref, fan, _, _ in candidates(x, old):
        start = np.vstack(
            (y, ref[-1] @ deformation + shift + np.array([2e-5, 8e-6, -3e-5]))
        )
        equilibrated, residual = interior_equilibrium(ref, start, fan, mu, bulk)
        e, f, _ = elastic(ref, equilibrated, fan, np.arange(len(ref)), mu, bulk)
        results.append(
            [
                abs(e - energy) / energy,
                np.linalg.norm(f[:-1] - force) / np.linalg.norm(force),
                residual,
            ]
        )
    if not results or max(map(max, results)) > 1e-7:
        raise ValueError(results)
    return dict(energy_force_and_internal_residual=results, unit_force="N")


if __name__ == "__main__":
    print(json.dumps(test(), indent=2))

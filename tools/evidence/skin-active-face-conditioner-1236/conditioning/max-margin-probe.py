"""Bounded numerical cone-feasibility probe, not a source-fit admission."""
import numpy as np
from scipy.optimize import minimize


def probe_cone_feasibility(maps, normals, preferred, selection_bound):
    maps = np.asarray(maps, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    preferred = np.asarray(preferred, dtype=np.float64)
    assert maps.ndim == 3 and maps.shape[1:] == (3, 3)
    assert normals.shape == (len(maps), 3) and preferred.shape == (3,)
    assert np.isfinite(maps).all() and np.isfinite(normals).all()
    assert np.isfinite(preferred).all() and np.linalg.norm(preferred) > 0
    assert np.all(np.linalg.norm(normals, axis=1) > 0)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    preferred = preferred / np.linalg.norm(preferred)
    pulled = np.linalg.solve(maps, normals[..., None])[..., 0]
    centre = (pulled / np.linalg.norm(pulled, axis=1)[:, None]).sum(axis=0)
    seeds = [("origin", np.zeros(3)), ("preferred", preferred)]
    if np.linalg.norm(centre) > 1e-12:
        seeds.append(("pulled-normal-centre", centre / np.linalg.norm(centre)))

    def cone_value(d):
        world = np.einsum("kij,j->ki", maps, d)
        return np.einsum("ki,ki->k", normals, world) - selection_bound * np.linalg.norm(world, axis=1)

    def constraint(x):
        return np.r_[cone_value(x[:3]) - x[3], 1.0 - np.dot(x[:3], x[:3])]

    def derivative(x):
        world = np.einsum("kij,j->ki", maps, x[:3])
        length = np.maximum(np.linalg.norm(world, axis=1), 1e-15)
        grad = np.einsum("ki,kij->kj", normals - selection_bound * world / length[:, None], maps)
        return np.vstack([np.column_stack([grad, -np.ones(len(maps))]), np.r_[-2.0*x[:3], 0.0]])

    results = []
    for name, seed in seeds:
        x0 = np.r_[seed, float(cone_value(seed).min()) - 1e-5]
        result = minimize(
            lambda x: -x[3], x0, jac=lambda x: np.array([0., 0., 0., -1.]),
            constraints={"type": "ineq", "fun": constraint, "jac": derivative},
            method="SLSQP", options={"maxiter": 256, "ftol": 1e-12},
        )
        length = float(np.linalg.norm(result.x[:3]))
        alignment = None
        direction = None
        if np.isfinite(result.x).all() and length > 1e-8:
            direction = result.x[:3] / length
            world = np.einsum("kij,j->ki", maps, direction)
            if np.all(np.linalg.norm(world, axis=1) > 1e-12):
                alignment = np.einsum("ki,ki->k", normals, world / np.linalg.norm(world, axis=1)[:, None])
        results.append({
            "seed": name, "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "optimizer_x": result.x.tolist(), "norm": length,
            "minimum_constraint_residual": float(constraint(result.x).min()),
            "normalized_direction": None if direction is None else direction.tolist(),
            "minimum_normalized_alignment": None if alignment is None else float(alignment.min()),
            "meets_unchanged_half_alignment": bool(alignment is not None and float(alignment.min()) >= 0.5),
            "interpretation": "A independently rechecked direction is a numerical feasibility witness; failed searches do not prove infeasibility.",
        })
    return results

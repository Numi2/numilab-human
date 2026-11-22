"""Preload the connected two-material stars at the GPU rejection sites."""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from material_context import sha
from scipy.linalg import null_space, solve
from scipy.optimize import minimize
from torso_cell_geometry import geom

from fit_support_strain import Patch


def constrained_direction(K, v, AJ, lower, upper, applicable):
    """Minimize the quadratic model on the feasible face of the volume bounds."""
    C = np.vstack(
        (AJ[np.isfinite(upper) & applicable], -AJ[np.isfinite(lower) & applicable])
    )
    bounds = np.r_[
        upper[np.isfinite(upper) & applicable], -lower[np.isfinite(lower) & applicable]
    ]
    inv = solve(K, C.T, assume_a="pos")
    length = np.sqrt(np.sum(C * inv.T, axis=1))
    C = C / length[:, None]
    bounds = bounds / length
    inv = inv / length
    gram = C @ inv
    if len(bounds) > 8:
        # Minimize the convex dual for larger sets of simultaneous cell limits.
        # Keep the same primal feasibility and complementarity certificates.
        excess = C @ v - bounds
        dual = minimize(
            lambda lam: (0.5 * lam @ gram @ lam - excess @ lam, gram @ lam - excess),
            np.zeros(len(bounds)),
            jac=True,
            bounds=[(0.0, None)] * len(bounds),
            method="L-BFGS-B",
            options={"ftol": 1.0e-15, "gtol": 1.0e-11, "maxiter": 300},
        )
        candidate = v - inv @ dual.x
        slack = bounds - C @ candidate
        if np.min(slack) < -1.0e-7 or np.max(np.abs(dual.x * slack)) > 1.0e-7:
            return None
        return candidate
    feasible = []
    for count in range(len(C) + 1):
        for subset in combinations(range(len(C)), count):
            sel = np.array(subset, dtype=int)
            v1 = v.copy()
            if count:
                G = gram[np.ix_(sel, sel)]
                if np.linalg.cond(G) > 1e12:
                    continue
                multipliers = solve(G, C[sel] @ v - bounds[sel], assume_a="pos")
                if np.min(multipliers) < -1e-8:
                    continue
                v1 -= inv[:, sel] @ multipliers
            if np.all(C @ v1 <= bounds + 1e-7):
                feasible.append((float((v1 - v) @ K @ (v1 - v)), v1))
    if not feasible:
        return None
    return min(feasible, key=lambda item: item[0])[1]


def run(prepared, out):
    src = json.loads((prepared / "source.json").read_text())
    if sha(prepared / "material.npz") != src["prepared_sha256"]:
        raise ValueError("Different material")
    with np.load(prepared / "material.npz") as a:
        p = Patch(a)
    start = p.y.copy()
    end = p.endpoint
    f = p.free.copy()
    support = p.critical_nodes
    target = p.guard_cells
    original = p.volume
    if not len(f) or not len(target):
        raise ValueError("Incomplete witnessed physical domain")
    V0 = geom(end, p.tet)[0]
    d = end - start

    def measure(y):
        r = p.residual(y)
        v, shape = geom(y + d, p.tet)
        J = v / original
        return {
            "physical_free_residual_N": float(np.linalg.norm(r[f])),
            "shared_residual_N": float(np.linalg.norm(r[support])),
            "bed_node_residual_N": float(np.linalg.norm(r[p.probes])),
            "neighbor_residual_N": float(np.linalg.norm(r)),
            "captured_J": J[target].tolist(),
            "minimum_J": float(J.min()),
            "maximum_J": float(J.max()),
            "minimum_shape": float(shape.min()),
        }

    original_results = measure(start)
    if len(p.invalid(start)):
        raise ValueError("Invalid loaded input")

    def valid(y):
        v, shape = geom(y + d, p.tet)
        J = v / original
        return (
            not len(p.invalid(y))
            and np.all(v > 1e-18)
            and np.all(shape >= 1e-5)
            and np.all(J >= np.minimum(0.4, V0 / original) - 1e-8)
            and np.all(J <= np.maximum(2.5, V0 / original) + 1e-8)
        )

    if not valid(start):
        raise ValueError("Existing shape/volume failure in the patch.")
    residual_scale = max(
        original_results["shared_residual_N"] / np.sqrt(len(support)), 0.001
    )

    def state(y):
        R = p.residual(y)
        R[support] *= 5.0
        J = geom(y + d, p.tet[target])[0] / original[target]
        return np.r_[R.ravel(), 8 * residual_scale * np.log(J)]

    history = []
    y = start.copy()
    active = f.copy()
    for it in range(28):
        W = p.W[:, np.searchsorted(f, active)]
        W = W[W.max(1) > 0]
        Q = (
            null_space(W / np.linalg.norm(W, axis=1)[:, None], rcond=1e-12)
            if len(W)
            else np.eye(len(active))
        )
        Z = np.kron(Q, np.eye(3))
        if not Z.shape[1]:
            break
        # -d(internal force)/dx is the native material tangent.
        H = p.tangent(y, active)
        H.reshape(len(y), 3, -1)[support] *= 5.0
        c = state(y)
        T = p.tet[target]
        current = y + d
        D = current[T[:, 1:]] - current[T[:, :1]]
        g = np.linalg.inv(D).transpose(0, 2, 1)
        g = np.concatenate((-g.sum(1)[:, None], g), axis=1)
        dv = np.zeros((len(T), len(y), 3))
        for k in range(4):
            dv[np.arange(len(T)), T[:, k]] = g[:, k]
        M = np.vstack([H, -8 * residual_scale * dv[:, active].reshape(len(T), -1)]) @ Z
        norm = np.maximum(np.linalg.norm(M, axis=0), 1)
        A = M / norm
        reg = np.kron(Q.T @ p.strain_metric(y, active) @ Q, np.eye(3))
        K = A.T @ A + reg / norm[:, None] / norm[None, :]
        K.flat[:: len(K) + 1] += 0.002**2
        rhs = A.T @ c
        v = solve(K, rhs, assume_a="pos")
        AJ = (dv[:, active].reshape(len(T), -1) @ Z) / norm
        previous = np.array(original_results["captured_J"])
        live = geom(current, p.tet[target])[0] / original[target]
        lower = np.full(len(T), -np.inf)
        upper = np.full(len(T), np.inf)
        threatened = previous < 0.8
        lower[threatened] = np.log((1.004 * previous[threatened]) / live[threatened])
        high = previous > 1.6
        upper[high] = np.log(
            np.maximum(previous[high] * 0.95, live[high] * 0.98) / live[high]
        )
        applicable = (np.isfinite(lower) | np.isfinite(upper)) & (
            np.linalg.norm(AJ, axis=1) > 1.0e-13
        )
        if np.any(applicable):
            v = constrained_direction(K, v, AJ, lower, upper, applicable)
            if v is None:
                break
        step = (Z @ (v / norm)).reshape(-1, 3)
        dr = np.zeros_like(y)
        dr[active] = step
        mag = np.linalg.norm(step, axis=1).max()
        if mag < 1e-11:
            break
        dr *= min(1.0, 0.0005 / mag)
        before = np.linalg.norm(c)
        ok = False
        for j in range(19):
            u = (y + 2.0**-j * dr).astype("f4").astype(float)
            if not valid(u) or np.max(abs(p.W @ (u[f] - start[f]))) > 8e-11:
                continue
            trial_J = geom((u + d), p.tet[target])[0] / original[target]
            if np.min(trial_J) < np.min(previous) - 1.0e-8:
                continue
            score = np.linalg.norm(state(u))
            if score < before * (1.0 - 1e-8):
                y = u
                ok = True
                history.append(
                    {
                        "iteration": it,
                        "fraction": 2.0**-j,
                        "score": float(score),
                        "metrics": measure(y),
                    }
                )
                break
        if not ok:
            print(
                "backtrack exhausted",
                it,
                "max_step",
                mag,
                "min score",
                before,
                "invalid@1%",
                len(p.invalid((y + 0.01 * dr).astype("f4").astype(float))),
                flush=True,
            )
            # Freeze only the free vertices of witnessed invalid tetrahedra,
            # retaining the original donor moments on the remaining material.
            bad = p.invalid((y + 0.01 * dr).astype("f4").astype(float))
            remove = np.intersect1d(active, np.unique(p.tet[bad]))
            if not len(remove) or len(active) - len(remove) < 3:
                break
            active = np.setdiff1d(active, remove)
    changed = np.flatnonzero(np.any(y != start, axis=1))
    m = measure(y)
    out.mkdir(exist_ok=False)
    np.savez_compressed(
        out / "fit.npz", rows=p.full_rows[changed], coords=y[changed].astype("f4")
    )
    gated = (
        len(changed) > 0
        and m["shared_residual_N"] < original_results["shared_residual_N"]
        and m["neighbor_residual_N"] <= original_results["neighbor_residual_N"]
        and np.max(m["captured_J"]) < np.max(original_results["captured_J"])
        and np.min(m["captured_J"]) >= np.min(original_results["captured_J"]) - 1e-6
    )
    result = {
        "source": src,
        "fit_sha256": sha(out / "fit.npz"),
        "initial": original_results,
        "emitted": m,
        "gated": bool(gated),
        "changed_nodes": len(changed),
        "largest_move_m": float(np.linalg.norm(y - start, axis=1).max()),
        "max_donor_mass_moment_kg_m": float(np.max(abs(p.W @ (y[f] - start[f])))),
        "history": history,
    }
    (out / "fit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "qualifies for native test:",
        bool(gated),
        len(changed),
        "initial",
        original_results,
        "end",
        m,
        flush=True,
    )


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))

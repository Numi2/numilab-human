"""Keep a registered starting pose admissible under measured release displacements."""

import json
import sys
from pathlib import Path

import numpy as np
from material_bed_release import balance_emitted_mass
from material_context import sha
from scipy.linalg import null_space
from scipy.optimize import minimize
from supported_material_admission import material_state
from torso_cell_geometry import geom

from fit_support_strain import Patch


def read(p):
    return json.loads(p.read_text())


from support_patch import motion


def solve(prepared, prior, control, trial, out, extra_trial=None):
    source = read(prepared / "source.json")
    if sha(prepared / "material.npz") != source["prepared_sha256"]:
        raise ValueError("Changed patch")
    with np.load(prepared / "material.npz") as arrays:
        p = Patch(arrays)
    data = read(prior / "fit.json")
    if (
        data["source"]["geometry"] != source["geometry"]
        or sha(prior / "fit.npz") != data["fit_sha256"]
    ):
        raise ValueError("Changed preload")
    delta = np.zeros_like(p.y)
    with np.load(prior / "fit.npz") as f:
        k = np.searchsorted(p.full_rows, f["rows"])
        if not np.array_equal(p.full_rows[k], f["rows"]):
            raise ValueError("Outside prepared mesh")
        delta[k] = f["coords"] - p.y[k]
    runs = [control, trial] + ([extra_trial] if extra_trial else [])
    cases = [motion(r, p.full_rows) for r in runs]
    if len({r["topology"] for _, r in cases}) != 1:
        raise ValueError("Different physical meshes")
    moves = [np.zeros_like(p.y)] + [d for d, _ in cases]

    # Preserve the original envelope deficiencies outside the edited tissue.
    def severity(y):
        J, s, stress = material_state(p.x[p.tet], y[p.tet], p.properties[p.roles - 15])
        return np.maximum.reduce(
            (
                s[:, 0] / 1.5,
                (2 / 3) / s[:, -1],
                stress / p.limits[p.roles - 15],
                0.4 / J,
                J / 2.5,
            )
        )

    support = np.isin(p.tet, np.flatnonzero(np.any(delta != 0, axis=1))).any(1)
    cap = [np.maximum(1.0, severity(p.y + d)) for d in moves]
    tested = []
    for ratio in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        y = (p.y + ratio * delta).astype("f4").astype(float)
        y[p.free] = balance_emitted_mass(
            p.y[p.free],
            y[p.free],
            p.W,
            np.full(len(p.W), 8e-11),
            np.zeros((len(p.free), 3), bool),
        )
        violations = [
            int(np.sum((severity(y + d) > a + 1e-7) & support))
            for d, a in zip(moves, cap)
        ]
        tested.append(
            {"fraction": ratio, "new_or_worse_stretch_stress_volume": violations}
        )
        if not any(violations):
            break
    else:
        raise ValueError(
            f"No force correction passed the registered strain margins: {tested}"
        )
    witness = p.guard_cells
    seed = np.unique(p.tet[witness])
    region = np.isin(p.tet, seed).any(1)
    free = np.intersect1d(p.free, np.unique(p.tet[region]))
    free = free[p.layer[free] == 0]
    if np.any(delta[free]) or not len(free):
        raise ValueError("Quality and preload regions overlap or have no interior")
    varied = np.isin(p.tet, free).any(1)
    idx = np.flatnonzero(varied)
    local = np.searchsorted(idx, witness)
    if not np.array_equal(idx[local], witness):
        raise ValueError("Witness has no eligible interior")
    t = p.tet[idx]
    W = p.W[:, np.searchsorted(p.free, free)]
    W = W[np.any(W != 0, axis=1)]
    N = np.kron(
        null_space(W / np.linalg.norm(W, axis=1)[:, None], rcond=1e-12), np.eye(3)
    )
    refs = np.array([geom(y + d, t)[1] for d in moves])
    targets = np.full(len(t), 1e-5)
    targets[local] += np.maximum(4 * np.ptp(refs[:, local], axis=0), 2e-9)
    construction_targets = targets.copy()

    def config(a):
        dy = np.zeros_like(y)
        dy[free] = (N @ a).reshape(-1, 3) * 1e-6
        return y + dy

    def constraints(a):
        c = config(a)
        return np.concatenate(
            [geom(c + d, t)[1] / construction_targets - 1 for d in moves]
        )

    # Minimum local change measured in deformation gradients of all touching cells.
    R = N.T @ np.kron(p.strain_metric(y, free), np.eye(3)) @ N
    R = R / max(1.0, np.linalg.norm(R, 2)) + 1e-6 * np.eye(R.shape[0])
    zero = np.zeros(N.shape[1])
    for _ in range(8):
        sol = minimize(
            lambda a: (0.5 * a @ R @ a, R @ a),
            zero,
            jac=True,
            constraints=[{"type": "ineq", "fun": constraints}],
            bounds=[(-25, 25)] * len(zero),
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 100},
        )
        if not sol.success or np.min(constraints(sol.x)) < -1e-7:
            raise ValueError(
                f"No conservative material registration {sol.message}, {np.min(constraints(sol.x))}"
            )
        keep0 = config(sol.x)
        final = keep0.astype("f4").astype(float)
        final[free] = balance_emitted_mass(
            p.y[free],
            final[free],
            W,
            np.full(len(W), 8e-11),
            np.zeros((len(free), 3), bool),
        )
        quality = np.array([geom(final + d, t)[1] for d in moves])
        shortage = np.maximum(0.0, targets - quality.min(axis=0))
        if not np.any(shortage):
            break
        construction_targets += 2 * shortage
    else:
        raise ValueError(
            "Quantized registration could not preserve the required margin"
        )
    if len(p.invalid(final)):
        raise ValueError("Registered material limit violated after lattice emission")
    residual = p.metrics(p.residual(final))
    base = p.metrics(p.residual(p.y))
    drift = float(np.abs(p.W @ (final[p.free] - p.y[p.free])).max())
    if drift > 8e-11 or residual["support_z_l2_N"] >= base["support_z_l2_N"]:
        raise ValueError("Force or mass regression")
    keep = np.flatnonzero(np.any(final != p.y, axis=1))
    out.mkdir(exist_ok=False)
    np.savez_compressed(
        out / "fit.npz", rows=p.full_rows[keep], coords=final[keep].astype("f4")
    )
    result = {
        "fit_sha256": sha(out / "fit.npz"),
        "source": source,
        "prior": sha(prior / "fit.json"),
        "captures": [r for _, r in cases],
        "support_fractions": tested,
        "chosen_fraction": ratio,
        "changed_nodes": len(keep),
        "guard_nodes": p.full_rows[free].tolist(),
        "guard_max_move_m": float(
            np.linalg.norm(final[free] - p.y[free], axis=1).max()
        ),
        "guard_source_cells": p.source_cells[witness].tolist(),
        "witness_before": refs[:, local].tolist(),
        "witness_after": quality[:, local].tolist(),
        "initial": base,
        "emitted": residual,
        "max_material_first_moment_change_kg_m": drift,
    }
    (out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["chosen_fraction"],
        result["guard_max_move_m"],
        result["witness_after"],
        result["emitted"],
    )


if __name__ == "__main__":
    solve(*(Path(v).resolve() for v in sys.argv[1:]))

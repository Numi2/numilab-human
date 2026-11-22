"""Compare fine-element strain energy and physical nodal forces on matched coordinates."""

import json, sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha


def elastic(x, y, cells, nodes, mu, K):
    X = x[cells[:, 1:]] - x[cells[:, :1]]
    Y = y[cells[:, 1:]] - y[cells[:, :1]]
    inv = np.linalg.inv(X)
    T = inv @ Y
    v = np.linalg.det(X) / 6.0
    det = np.linalg.det(T)
    if np.min(v) <= 0 or np.min(det) <= 0:
        raise ValueError("Invalid strain state")
    logj = np.log(det)
    energy = float(
        v
        @ (
            0.5 * mu * (np.sum(T * T, axis=(1, 2)) - 3.0 - 2.0 * logj)
            + 0.5 * K * logj**2
        )
    )
    dpsi = mu * T + (K * logj - mu)[:, None, None] * np.swapaxes(np.linalg.inv(T), 1, 2)
    g = np.swapaxes(inv, 1, 2) @ dpsi * v[:, None, None]
    f = np.zeros((len(nodes), 3))
    ix = np.searchsorted(nodes, cells)
    np.add.at(f, ix[:, 0], g.sum(axis=1))
    for c in range(3):
        np.add.at(f, ix[:, c + 1], -g[:, c])
    return energy, f, det


def audit(original, one, fit, build, out):
    x, y, t, mat, pins = inputs(original)
    x = x.astype(float)
    y = y.astype(float)
    entry = json.loads(fit.read_text())
    s = entry["proposal"]
    if pins != entry["source"]:
        raise ValueError("Geometry changed")
    old = t[np.array(s["retired_rows"])]
    new = np.array(s["replacement_tets"])
    if not np.all(mat[s["retired_rows"]] == 16):
        raise ValueError("Anatomical material changed")
    a = {tuple(e) for e in old}
    b = {tuple(e) for e in new}
    o = np.array(sorted(a - b))
    n = np.array(sorted(b - a))
    if len(o) != 3 or len(n) != 2:
        raise ValueError("Expected one edge-star replacement")
    props, mats = materials(build)
    mu, K = props[1]
    p = one / "physical-accepted.node"
    if sha(p) != entry["accepted_coordinates_sha256"]:
        raise ValueError("Unmatched endpoint")
    first = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    rows = []
    for name, Y in [("installed", y), ("one_old_accepted_step", first)]:
        for label, before, after in [
            ("altered_core", o, n),
            ("connected_mass_region", old, new),
        ]:
            nodes = np.unique(before)
            E, f, J = elastic(x, Y, before, nodes, mu, K)
            e, g, j = elastic(x, Y, after, nodes, mu, K)
            rows.append(
                dict(
                    pose=name,
                    scope=label,
                    original_J=J.tolist(),
                    replacement_J=j.tolist(),
                    original_energy_J=E,
                    replacement_energy_J=e,
                    relative_energy_error=abs(E - e) / E,
                    original_force_l2_N=float(np.linalg.norm(f)),
                    replacement_force_l2_N=float(np.linalg.norm(g)),
                    force_l2_difference_N=float(np.linalg.norm(f - g)),
                    relative_force_l2_error=float(
                        np.linalg.norm(f - g) / np.linalg.norm(f)
                    ),
                    resultant_force_error_N=np.max(abs(f.sum(0) - g.sum(0))),
                    nodal_force_delta_N=(g - f).tolist(),
                    node_rows=nodes.tolist(),
                )
            )
    out.write_text(
        json.dumps(
            dict(source=pins, native_material=mats, constitutive_comparisons=rows),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    audit(*(Path(i).resolve() for i in sys.argv[1:]))

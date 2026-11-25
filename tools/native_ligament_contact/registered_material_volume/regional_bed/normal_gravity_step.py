"""Constrained Newton search for a gravity-loaded starting pose. All coordinates point toward the bed."""

import numpy as np
from scipy import sparse


def contact_step(hessian, gradient, upper, active, solve, max_iterations=48):
    """Minimize a quadratic with one-sided exterior constraints and a free skeletal translation."""
    active = active.copy()
    if np.any(active & ~np.isfinite(upper)):
        raise ValueError("Non-exterior node in the bed set")
    history = []
    visited = set()
    for _ in range(max_iterations):
        key = np.packbits(active).tobytes()
        if key in visited:
            raise ValueError(f"Repeated contact set after {len(history)} solves")
        visited.add(key)
        trial = np.zeros(len(gradient))
        trial[active] = upper[active]
        free = ~active
        rhs = -gradient[free] - (hessian @ trial)[free]
        trial[free] = solve(hessian[free][:, free], rhs)
        reaction = -(gradient + hessian @ trial)
        violations = free & (trial > upper + 1e-9)
        unload = active & (reaction < -1e-7)
        history.append(
            dict(
                active=int(active.sum()),
                penetrated=int(violations.sum()),
                tensile=int(unload.sum()),
            )
        )
        print("Contact solve", len(history), history[-1], flush=True)
        if not np.any(violations | unload):
            if np.max(abs(reaction[free]), initial=0.0) > 1e-5:
                raise ValueError("Unbalanced free coordinates in normal contact solve")
            reaction[free] = 0
            return trial, reaction, history
        active |= violations
        active &= ~unload
        if not active.any():
            # Nonzero weight requires an external reaction. Retain the highest
            # multiplier until another contact takes over.
            candidates = np.flatnonzero(np.isfinite(upper))
            active[candidates[np.argmax(reaction[candidates])]] = True
    raise ValueError(f"Unfinished normal contact search {history[-1]}")


def constitutive_gradient(reference, xyz, t, normal, label, params):
    g = np.zeros(len(xyz))
    energy = 0.0
    for start in range(0, len(t), 65536):
        cell = t[start : start + 65536]
        mu, bulk = np.where(
            (label[start : start + len(cell)] == 15)[:, None], params[15], params[16]
        ).T
        dm = reference[cell[:, 1:]] - reference[cell[:, :1]]
        ds = xyz[cell[:, 1:]] - xyz[cell[:, :1]]
        volume = np.linalg.det(dm) / 6
        ft = np.linalg.solve(dm, ds)
        J = np.linalg.det(ft)
        if not np.all(J > 0):
            raise ValueError("Inverted trial cell")
        logJ = np.log(J)
        gradients = np.linalg.inv(dm).transpose(0, 2, 1)
        gradients = np.concatenate(
            (-gradients.sum(axis=1, keepdims=True), gradients), axis=1
        )
        normal_rhs = np.broadcast_to(normal, (len(cell), 3))[..., None]
        stress_n = (
            mu[:, None] * (ft @ normal)
            + (bulk * logJ - mu)[:, None]
            * np.linalg.solve(ft.transpose(0, 2, 1), normal_rhs)[..., 0]
        )
        local = -volume[:, None] * (gradients @ stress_n[..., None])[..., 0]
        g += np.bincount(cell.ravel(), weights=local.ravel(), minlength=len(g))
        I1 = np.einsum("tij,tij->t", ft, ft)
        energy += volume @ (0.5 * mu * (I1 - 3 - 2 * logJ) + 0.5 * bulk * logJ**2)
    return g, float(energy)


def certify_motion(reference, xyz, cells, direction, initial_alpha):
    """Choose the largest tested dyadic step adding no inversion or strain/volume violations."""

    def flags(dm, ds):
        f = np.linalg.solve(dm, ds)
        determinant = np.linalg.det(ds)
        volume = determinant / 6.0
        relative = np.linalg.det(f)
        stretch = np.linalg.svd(f, compute_uv=False)
        edge2 = np.sum(ds * ds, axis=2)
        others = ds[:, [1, 2, 2]] - ds[:, [0, 0, 1]]
        longest2 = np.maximum(edge2.max(1), np.sum(others * others, axis=2).max(1))
        shape = determinant / longest2**1.5
        if not np.isfinite(stretch).all() or not np.isfinite(shape).all():
            raise ValueError("Nonfinite trial cell")
        return np.column_stack(
            (
                volume <= 0,
                volume <= 1e-18,
                shape < 1e-5,
                (relative < 0.4) | (relative > 2.5),
                (stretch[:, 0] > 1.5) | (stretch[:, -1] < 2 / 3),
            )
        )

    initial = np.empty((len(cells), 5), dtype=bool)
    batch = 32768
    for start in range(0, len(cells), batch):
        c = cells[start : start + batch]
        dm = reference[c[:, 1:]] - reference[c[:, :1]]
        initial[start : start + len(c)] = flags(dm, xyz[c[:, 1:]] - xyz[c[:, :1]])
    alpha = initial_alpha
    witnesses = []
    for _ in range(32):
        cleared = np.zeros(5, dtype=int)
        for start in range(0, len(cells), batch):
            c = cells[start : start + batch]
            dm = reference[c[:, 1:]] - reference[c[:, :1]]
            ds = xyz[c[:, 1:]] - xyz[c[:, :1]]
            du = direction[c[:, 1:]] - direction[c[:, :1]]
            pre = initial[start : start + len(c)]
            post = flags(dm, ds + alpha * du)
            bad = np.argwhere(post & ~pre)
            if len(bad):
                cell, gate = bad[0]
                witnesses.append(
                    dict(fraction=alpha, cell=int(start + cell), gate=int(gate))
                )
                break
            cleared += np.sum(pre & ~post, axis=0)
        else:
            return alpha, cleared.tolist(), initial.sum(axis=0).tolist(), witnesses
        alpha *= 0.5
    raise ValueError("No deformation-valid normal starting increment")

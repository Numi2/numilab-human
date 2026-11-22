"""Energy descent for supported initialization with independent, cell-disjoint vertex updates."""

import numpy as np


def coloring(tets, free):
    adjacency = [set() for _ in range(len(free))]
    inverse = np.full(tets.max() + 1, -1, dtype=int)
    inverse[free] = np.arange(len(free))
    labels = inverse[tets]
    for tet in labels:
        vertices = tet[tet >= 0]
        for vertex in vertices:
            adjacency[vertex].update(vertices[vertices != vertex])
    colors = np.full(len(free), -1, dtype=int)
    for vertex in sorted(range(len(free)), key=lambda i: (-len(adjacency[i]), i)):
        taken = {colors[i] for i in adjacency[vertex]}
        colors[vertex] = next(i for i in range(len(taken) + 1) if i not in taken)
    groups = []
    for color in range(colors.max() + 1):
        members = np.flatnonzero(colors == color)
        lookup = np.full(len(inverse), -1, dtype=int)
        lookup[free[members]] = np.arange(len(members))
        stencil = lookup[tets]
        cells, corners = np.nonzero(stencil >= 0)
        if np.any(np.diff(cells) <= 0):
            raise ValueError("Two simultaneous updates share a tetrahedron")
        groups.append((members, cells, corners, stencil[cells, corners]))
    return groups


def relax(
    y,
    tets,
    free,
    gradients,
    volume,
    mu,
    bulk,
    mass,
    energy,
    force,
    certify,
    max_sweeps=40,
):
    y = y.copy()
    groups = coloring(tets, free)

    def value(F, J, idx):
        logj = np.log(J)
        return volume[idx] * (
            0.5 * mu[idx] * (np.sum(F * F, axis=(1, 2)) - 3 - 2 * logj)
            + 0.5 * bulk[idx] * logj**2
        )

    history = []
    for sweep in range(max_sweeps):
        before = energy(y)
        moved = 0
        for members, cells, corners, owners in (
            groups if sweep % 2 == 0 else reversed(groups)
        ):
            rows = free[members]
            xyz = y[tets[cells]]
            F = (xyz[:, 1:] - xyz[:, :1]).transpose(0, 2, 1) @ gradients[cells, 1:]
            J = np.linalg.det(F)
            base = value(F, J, cells)
            gN = gradients[cells, corners]
            v = np.linalg.solve(F.transpose(0, 2, 1), gN[:, :, None])[:, :, 0]
            elastic = mu[cells, None] * (F @ gN[:, :, None])[:, :, 0]
            elastic += (bulk[cells] * np.log(J) - mu[cells])[:, None] * v
            gradient = np.zeros((len(rows), 3))
            np.add.at(gradient, owners, volume[cells, None] * elastic)
            gradient[:, 2] += 9.81 * mass[members]
            k = volume[cells] * mu[cells] * np.sum(gN * gN, axis=1)
            c = volume[cells] * (bulk[cells] * (1 - np.log(J)) + mu[cells])
            blocks = np.zeros((len(rows), 3, 3))
            np.add.at(
                blocks,
                owners,
                k[:, None, None] * np.eye(3)
                + c[:, None, None] * v[:, :, None] * v[:, None, :],
            )
            step = np.linalg.solve(blocks, -gradient[:, :, None])[:, :, 0]
            pending = np.ones(len(rows), dtype=bool)
            for exponent in range(19):
                ev = np.flatnonzero(pending[owners])
                if not len(ev):
                    break
                p = owners[ev]
                idx = cells[ev]
                # Test exactly the positions the native mesh reader can represent.
                target = (
                    (y[rows] + step * 2.0**-exponent).astype(np.float32).astype(float)
                )
                delta = target - y[rows]
                change = delta[p]
                trial_F = F[ev] + change[:, :, None] * gN[ev, None, :]
                det = np.linalg.det(trial_F)
                singular = np.linalg.svd(trial_F, compute_uv=False)
                valid = (
                    (det >= 0.4)
                    & (det <= 2.5)
                    & (singular[:, 0] <= 1.5)
                    & (singular[:, -1] >= 2 / 3)
                )
                trial_X = xyz[ev].copy()
                trial_X[np.arange(len(ev)), corners[ev]] += change
                edges = trial_X[:, [0, 0, 0, 1, 1, 2]] - trial_X[:, [1, 2, 3, 2, 3, 3]]
                maxsq = np.max(np.sum(edges**2, axis=2), axis=1)
                vol = volume[idx] * det
                valid &= (vol > 1e-18) & (6 * vol >= 1e-5 * maxsq**1.5)
                bad = np.bincount(p, weights=~valid, minlength=len(rows)) > 0
                merit = value(trial_F, np.maximum(det, 1e-20), idx) - base[ev]
                decrease = (
                    np.bincount(p, weights=merit, minlength=len(rows))
                    + 9.81 * mass[members] * delta[:, 2]
                )
                slope = np.sum(gradient * delta, axis=1)
                accept = pending & ~bad & (slope < 0) & (decrease <= 1e-4 * slope)
                y[rows[accept]] = target[accept]
                moved += int(np.sum(accept))
                pending &= ~accept & np.any(delta != 0.0, axis=1)
        after = energy(y)
        if after > before + 1e-11 or not certify(y):
            raise ValueError(("Invalid constrained energy sweep", sweep, before, after))
        residual = force(y)
        history.append(
            dict(
                sweep=sweep,
                energy_J=after,
                g_l2_N=float(np.linalg.norm(residual)),
                changed=moved,
            )
        )
        if not moved or before - after < 1e-8 * max(1.0, abs(before)):
            break
    return y, history

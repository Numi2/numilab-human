"""Improve a fixed-boundary volume in the compiled Float32 coordinate domain."""
import numpy as np
from scipy.optimize import linprog

from extension import determinant, quality


def quality_at(x, cells):
    return quality(x, cells, determinant(x, cells))


def optimize(points, cells, fixed, floor, max_sweeps=5):
    x = points.astype(float)
    movable = np.ones(len(points), dtype=bool)
    movable[list(fixed)] = False
    incident = [[] for _ in points]
    for row, tet in enumerate(cells):
        for node in tet:
            incident[node].append(row)
    previous = quality_at(x, cells)
    count_path = [int((previous < floor).sum())]
    moved = set()
    for _ in range(max_sweeps):
        poor = np.flatnonzero(previous < floor)
        if not len(poor):
            break
        active = sorted({int(node) for node in cells[poor].ravel() if movable[node]})
        changed = False
        for node in active:
            ring = np.array(incident[node])
            local = cells[ring]
            before = previous[ring]
            if not np.any(before < floor):
                continue
            corners = x[local]
            edges = corners[:, :, None] - corners[:, None, :]
            max_edge = np.sqrt(np.max(np.sum(edges**2, axis=-1), axis=(1, 2)))
            travel = .15 * max_edge.min()
            u, v, w = np.moveaxis(corners[:, 1:] - corners[:, :1], 1, 0)
            derivatives = np.stack((np.cross(v, w), np.cross(w, u), np.cross(u, v)), axis=1)
            derivatives = np.concatenate((-derivatives.sum(axis=1, keepdims=True), derivatives), axis=1)
            g = derivatives[local == node] * travel / max_edge[:, None]**3
            fit = linprog([0., 0., 0., -1.], A_ub=np.c_[-g, np.ones(len(g))],
                          b_ub=before, bounds=[(-1., 1.)] * 3 + [(None, None)],
                          method="highs")
            if not fit.success or fit.x[3] <= before.min():
                continue
            current = x[node].copy()
            old_score = int((before < floor).sum()), -before.min()
            for fraction in (1., .5, .25, .125):
                x[node] = (current + fraction * travel * fit.x[:3]).astype(np.float32)
                after = quality_at(x, local)
                if np.any(after[before >= floor] < floor) or np.any(after[before > 0.] <= 0.):
                    continue
                if (int((after < floor).sum()), -after.min()) < old_score:
                    previous[ring] = after
                    moved.add(node)
                    changed = True
                    break
            else:
                x[node] = current
        count_path.append(int((previous < floor).sum()))
        if not changed:
            break
    assert np.array_equal(x[~movable], points[~movable])
    return x.astype(np.float32), {
        "under_quality_by_sweep": count_path,
        "moved_interior_nodes": len(moved),
        "maximum_movement_m": float(np.linalg.norm(x - points, axis=1).max()),
    }

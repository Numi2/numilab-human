"""Exact-witness driven x-only contact constraints on the projected triangle overlap."""

import numpy as np
from scipy.sparse import coo_matrix, hstack, csr_matrix
from compare_hand_interfaces import query


def barycentric(point, tri):
    coord = np.linalg.solve((tri[1:] - tri[:1]).T, point - tri[0])
    return np.r_[1 - coord.sum(), coord]


def overlap(A, B):
    # Work in a common millimetre-scale chart. The untouched y,z projection
    # makes the x-separation affine on this convex polygon for the entire edit.
    origin = A[0].copy()
    scale = max(np.ptp(A, axis=0).max(), np.ptp(B, axis=0).max())
    if scale <= 0:
        return []
    A = (A - origin) / scale
    B = (B - origin) / scale
    if max(np.linalg.cond((t[1:] - t[:1]).T) for t in (A, B)) > 1e9:
        return []
    pts = []
    for t, other in ((A, B), (B, A)):
        for p in t:
            if np.min(barycentric(p, other)) >= -2e-12:
                pts.append(p)
    for i in range(3):
        e = A[(i + 1) % 3] - A[i]
        for j in range(3):
            d = B[(j + 1) % 3] - B[j]
            M = np.column_stack((e, -d))
            if abs(np.linalg.det(M)) < 1e-14:
                continue
            u, v = np.linalg.solve(M, B[j] - A[i])
            if min(u, v) >= -2e-12 and max(u, v) <= 1 + 2e-12:
                pts.append(A[i] + u * e)
    return [(barycentric(p, A), barycentric(p, B)) for p in pts]


class NoNewCrossings:
    def __init__(self, path, exe, vars_for_nodes, N):
        dat = np.load(path)
        self.x = dat["x"]
        self.f = dat["f"]
        self.active = dat["changed"]
        owner = dat["material"]
        self.var = np.full(len(owner), -1, dtype=int)
        mapped = owner >= 0
        self.var[mapped] = vars_for_nodes[owner[mapped]]
        self.baseline = set(map(tuple, dat["baseline"].tolist()))
        self.exe = exe
        self.N = N
        self.tested = set()
        self.locked = set()
        self.log = []

    def evaluate(self, values, quantum):
        w = self.x.copy()
        free = self.var >= 0
        w[free, 0] += np.rint(values[self.var[free]] * 0.001 / quantum) * quantum
        current = query(self.exe, w, self.f, self.active)
        return current - self.baseline, self.baseline - current

    def constrain(self, pairs, quantum):
        rr = []
        cc = []
        data = []
        rhs = []
        fallback = 0
        repeat = 0

        def freeze(ids):
            nonlocal fallback
            fallback += 1
            for j in sorted(set(ids) - {-1} - self.locked):
                self.locked.add(j)
                for v in (-1.0, 1.0):
                    rr.append(len(rhs))
                    cc.append(j)
                    data.append(v)
                    rhs.append(0.0)

        for ia, ib in sorted(pairs):
            faces = np.r_[self.f[ia], self.f[ib]]
            p = self.x[faces]
            var = self.var[faces]
            if (ia, ib) in self.tested:
                repeat += 1
                freeze(var)
                continue
            self.tested.add((ia, ib))
            poly = overlap(p[:3, 1:], p[3:, 1:])
            if not poly:
                freeze(var)
                continue
            weights = np.stack([np.r_[a, -b] for a, b in poly])
            sep = 1000 * (weights @ (p[:, 0] - p[0, 0]))
            side = np.sign(sep[np.argmax(abs(sep))])
            if abs(sep).max() < 1e-10 or np.min(sep * side) < -1e-8:
                freeze(var)
                continue
            for w, v in zip(weights, side * sep):
                coeff = {}
                for j, c in zip(var, -side * w):
                    if j >= 0:
                        coeff[j] = coeff.get(j, 0.0) + c
                if not coeff:
                    continue
                # A strictly positive baseline gap may donate at most a quarter of its width.
                margin = min(
                    max(0, v) * 0.25,
                    0.501 * 1000 * quantum * sum(abs(k) for k in coeff.values()),
                )
                for j, c in coeff.items():
                    if c:
                        rr.append(len(rhs))
                        cc.append(j)
                        data.append(c)
                rhs.append(max(0.0, v - margin))
        M = coo_matrix((data, (rr, cc)), shape=(len(rhs), self.N)).tocsr()
        self.log.append(
            dict(
                new_pairs=len(pairs),
                repeated_pairs=repeat,
                degenerate_or_repeated_pairs=fallback,
                fixed_parameters=len(self.locked),
                linear_rows=len(rhs),
            )
        )
        return hstack([M, csr_matrix(M.shape)], format="csr"), np.asarray(rhs)

"""Recover the separating halfspaces of the measured repair in the physical source-face index space."""

import json
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.spatial import cKDTree
from digital_contact_constraints import overlap


def domain(root, points):
    report = json.loads((root / "constrained-1.5mm/interfaces.json").read_text())
    dat = np.load(root / "physical.npz")
    vertices = np.unique(
        np.concatenate(
            [
                dat[k + "v"][dat[k + "f"][f]]
                for k, f, _, _ in report["cleared_witnesses"]
            ]
        ),
        axis=0,
    )
    return cKDTree(vertices).query(points)[0] < 0.005


def targets(root, NoNew, quantum):
    data = np.load(root / "contact-domain.npz")
    xs = data["x"]
    faces = data["f"]
    m = data["material"]
    keys = data["keys"][data["gid"]]
    lookup = {
        (str(k), int(f)): i for i, (k, f) in enumerate(zip(keys, data["source_face"]))
    }
    reference = np.load(root / "constrained-1.5mm/field.npz")
    v = reference["nodes"]
    moved = dict(zip(v, 1000 * reference["delta_x_m"]))
    record = json.loads((root / "constrained-1.5mm/interfaces.json").read_text())
    row = []
    col = []
    value = []
    rhs = []
    for a, i, b, j in record["cleared_witnesses"]:
        tri = np.r_[faces[lookup[a, i]], faces[lookup[b, j]]]
        points = xs[tri]
        v0 = 1000 * (points[:, 0] - points[0, 0])
        shift = np.array([moved.get(n, 0.0) for n in m[tri]])
        poly = overlap(points[:3, 1:], points[3:, 1:])
        if not poly:
            raise ValueError("Unresolved source projection")
        coeff = np.stack([np.r_[u, -w] for u, w in poly])
        gap = coeff @ (v0 + shift)
        side = np.sign(gap[np.argmax(abs(gap))])
        if not np.all(gap * side > 0):
            raise ValueError("Candidate does not clear whole projected face pair")
        for w in coeff:
            c = {}
            for k, f in zip(NoNew.var[tri], -side * w):
                if k >= 0:
                    c[k] = c.get(k, 0.0) + f
            safety = 0.501 * quantum * 1000 * sum(abs(z) for z in c.values()) + 1.0e-7
            for k, f in c.items():
                row.append(len(rhs))
                col.append(k)
                value.append(f)
            rhs.append(side * (w @ v0) - safety)
    A = coo_matrix((value, (row, col)), shape=(len(rhs), NoNew.N)).tocsr()
    return hstack((A, csr_matrix(A.shape)), format="csr"), np.array(rhs)

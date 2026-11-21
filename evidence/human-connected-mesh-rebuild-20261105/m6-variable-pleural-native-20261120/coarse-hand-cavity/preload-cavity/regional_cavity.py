"""Find mesh-only replacements without changing any anatomical boundary face."""
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import csv
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from cavity import LABEL, LOWER, UPPER, cell_shape, determinant, oriented_boundary


def edge_stars(cells, owner, witness):
    same = np.array([label == owner[witness] for label in owner])
    stars = [frozenset(np.flatnonzero(same & (np.isin(cells, edge).sum(axis=1) == 2)).tolist())
             for edge in combinations(cells[witness], 2)]
    regions = set(stars) | {one | two for one, two in combinations(stars, 2)}
    return sorted((sorted(r) for r in regions if len(r) > 1), key=lambda r: (len(r), r))


def propose(cells, frames, outside, retired, witness):
    before = cells[retired]
    boundary = oriented_boundary(before)
    vertices = np.unique(before).tolist()
    result = {"retired": retired, "vertices": len(vertices)}
    if len(vertices) > 17:
        result["state"] = "oversize"
        return result
    proposed = np.array([cell for cell in combinations(vertices, 4)
                         if cell not in outside and cell != witness], dtype=np.intp)
    det = np.stack([determinant(x, proposed) for x in frames])
    inverted = det[0] < 0.
    proposed[inverted] = proposed[inverted][:, [0, 2, 1, 3]]
    det[:, inverted] *= -1
    valid = (det > 0.).all(axis=0)
    j = np.divide(det[-1], det[0], out=np.zeros(len(proposed)), where=valid)
    valid &= (j >= LOWER + .005) & (j <= UPPER - .005)
    proposed, det = proposed[valid], det[:, valid]
    quality_floor = np.array([cell_shape(x, before, determinant(x, before)).min() for x in frames])
    q = np.stack([cell_shape(x, proposed, d) for x, d in zip(frames, det, strict=True)])
    valid = (q >= quality_floor[:, None]).all(axis=0)
    proposed, det, q = proposed[valid], det[:, valid], q[:, valid]
    result["admissible_cells"] = len(proposed)
    face_index = {face: i for i, face in enumerate(boundary)}
    rows, cols, values = [], [], []
    for col, cell in enumerate(proposed):
        for face, sign in oriented_boundary([cell]).items():
            rows.append(face_index.setdefault(face, len(face_index)))
            cols.append(col)
            values.append(sign)
    n = len(proposed)
    signed = coo_matrix((values, (rows, cols)), shape=(len(face_index), n)).tocsr()
    rhs = np.zeros(len(face_index))
    rhs[:len(boundary)] = list(boundary.values())
    cap = np.full(len(face_index), 2.)
    cap[:len(boundary)] = 1.
    if n == 0 or np.any(np.diff(signed.indptr)[:len(boundary)] == 0):
        result["state"] = "missing_boundary_face"
        return result
    constraints = LinearConstraint(vstack((signed, abs(signed))), np.r_[rhs, np.zeros_like(rhs)],
                                   np.r_[rhs, cap])
    objective = 1. + np.mean(quality_floor[:, None] / q, axis=0)
    res = milp(objective, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
               constraints=constraints, options={"time_limit": 2.})
    result["state"] = str(res.message)
    if res.x is None:
        return result
    accepted = proposed[res.x > .5]
    assert oriented_boundary(accepted) == boundary
    checked = np.stack([determinant(x, accepted) for x in frames])
    expected = np.array([determinant(x, before).sum() for x in frames])
    np.testing.assert_allclose(checked.sum(axis=1), expected, rtol=1.e-9, atol=1.e-17)
    assert np.all(checked > 0.)
    j = checked[-1] / checked[0]
    assert min(j) >= LOWER + .005 and max(j) <= UPPER - .005
    result.update(replacement=accepted.tolist(), minimum_j=float(min(j)), maximum_j=float(max(j)))
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("fit", "trial", "labels", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with np.load(args.fit) as fits, np.load(args.trial) as trial:
        cells = fits["cells"]
        rest = fits["nodes_reference"].astype(float)
        world = fits["nodes_current"].astype(float)
        loaded = (world + trial["deformation_m"]).astype(np.float32).astype(float)
    with args.labels.open() as source:
        labels = [tuple(r[label] for label in LABEL) for r in csv.DictReader(source)]
    witness = 21848
    existing = {tuple(sorted(row)) for row in cells}
    output = []
    for retired in edge_stars(cells, labels, witness):
        outside = existing - {tuple(sorted(cells[row])) for row in retired}
        output.append(propose(cells, (rest, world, loaded), outside, retired,
                              tuple(sorted(cells[witness]))))
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for o in output:
        print(len(o["retired"]), o["vertices"], o.get("admissible_cells"), o["state"],
              o.get("minimum_j"), o.get("maximum_j"))


if __name__ == "__main__":
    main()

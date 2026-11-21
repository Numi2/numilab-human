"""Screen same-bank 2-to-3 and 3-to-2 flips on the exact preload witness."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import json

import numpy as np

FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
LABEL = ("material_index", "geometric_donor_body", "anatomical_region_id")
LOWER, UPPER = .350001, 2.499999


def determinant(points, cells):
    p = points[cells]
    return np.linalg.det(p[:, 1:] - p[:, :1])


def cell_shape(points, cells, det):
    p = points[cells]
    diameter_squared = np.max(
        np.sum((p[:, :, None] - p[:, None, :]) ** 2, axis=3), axis=(1, 2))
    return det / (diameter_squared * np.sqrt(diameter_squared))


def oriented_boundary(cells):
    cancelled = Counter()
    for cell in cells:
        for face in FACES:
            a, b, c = (int(cell[i]) for i in face)
            sign = 1 if (a > b) + (b > c) + (a > c) in (0, 2) else -1
            cancelled[tuple(sorted((a,b,c)))] += sign
    return {key: sign for key, sign in cancelled.items() if sign}


def index_incidence(cells, combinations_):
    lookup = defaultdict(list)
    for index, cell in enumerate(cells):
        for ids in combinations_:
            lookup[tuple(sorted(cell[list(ids)]))].append(index)
    return lookup


def candidates(cells, witness):
    faces = index_incidence(cells, FACES)
    edge_pairs = tuple(combinations(range(4), 2))
    edges = index_incidence(cells, edge_pairs)
    for local in FACES:
        key = tuple(sorted(cells[witness, list(local)]))
        roots = faces[key]
        if len(roots) == 2:
            ends = [int(v) for row in roots for v in cells[row] if v not in key]
            if len(set(ends)) == 2:
                yield "2-to-3", key, roots, [ends + list(pair) for pair in combinations(key, 2)]
    for local in edge_pairs:
        key = tuple(sorted(cells[witness, list(local)]))
        roots = edges[key]
        if len(roots) == 3:
            ring = {int(v) for row in roots for v in cells[row] if v not in key}
            if len(ring) == 3:
                yield "3-to-2", key, roots, [[end, *sorted(ring)] for end in key]


def evaluate(cells, frames, labels, witness):
    existing = {tuple(sorted(row)) for row in cells}
    result = []
    for kind, key, retired, proposed in candidates(cells, witness):
        record = {"kind": kind, "simplex": [int(vertex) for vertex in key], "retired": list(retired)}
        result.append(record)
        if len({labels[index] for index in retired}) != 1:
            record["rejected"] = "different_material_or_anatomical_bank"
            continue
        if any(tuple(sorted(row)) in existing for row in proposed):
            record["rejected"] = "existing_tetrahedron"
            continue
        replacement = np.asarray(proposed)
        negative = determinant(frames[0], replacement) < 0.
        replacement[negative] = replacement[negative][:, [0, 2, 1, 3]]
        record["replacement"] = replacement.tolist()
        current = cells[retired]
        if oriented_boundary(current) != oriented_boundary(replacement):
            record["rejected"] = "nonconvex_or_degenerate_cavity"
            continue
        old = np.stack([determinant(frame, current) for frame in frames])
        new = np.stack([determinant(frame, replacement) for frame in frames])
        if not np.allclose(old.sum(axis=1), new.sum(axis=1), rtol=1.e-9, atol=1.e-17):
            record["rejected"] = "cavity_volume_drift"
            continue
        if np.any(new <= 0.):
            record["rejected"] = "nonpositive_cell"
            continue
        q0 = np.array([cell_shape(frame, current, d).min() for frame, d in zip(frames, old)])
        q1 = np.array([cell_shape(frame, replacement, d).min() for frame, d in zip(frames, new)])
        ratio0, ratio1 = old[-1] / old[0], new[-1] / new[0]
        margin0 = np.minimum(ratio0 - LOWER, UPPER - ratio0).min()
        margin1 = np.minimum(ratio1 - LOWER, UPPER - ratio1).min()
        record.update(old_min_quality=q0.tolist(), new_min_quality=q1.tolist(),
                      old_j_bounds=[float(ratio0.min()), float(ratio0.max())],
                      new_j_bounds=[float(ratio1.min()), float(ratio1.max())],
                      margin_gain=float(margin1 - margin0))
        if np.any(q1 < q0):
            record["rejected"] = "worse_shape"
        elif margin1 <= max(0., margin0):
            record["rejected"] = "insufficient_volume_margin"
        else:
            record["accepted_geometry"] = True
    return result


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("fit", "trial", "labels", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--witness", type=int, default=21848)
    a = p.parse_args()
    with np.load(a.fit) as source, np.load(a.trial) as trial:
        initial = source["nodes_current"]
        loaded = (initial.astype(float) + trial["deformation_m"]).astype(np.float32)
        frames = tuple(frame.astype(float) for frame in (source["nodes_reference"], initial, loaded))
        cells = source["cells"]
    with a.labels.open() as stream:
        labels = [tuple(r[key] for key in LABEL) for r in csv.DictReader(stream)]
    assert len(labels) == len(cells)
    with a.out.open("x") as stream:
        report = evaluate(cells, frames, labels, a.witness)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print({key: sum(r.get("rejected", "accepted_geometry") == key for r in report)
           for key in sorted({r.get("rejected", "accepted_geometry") for r in report})})


if __name__ == "__main__":
    main()

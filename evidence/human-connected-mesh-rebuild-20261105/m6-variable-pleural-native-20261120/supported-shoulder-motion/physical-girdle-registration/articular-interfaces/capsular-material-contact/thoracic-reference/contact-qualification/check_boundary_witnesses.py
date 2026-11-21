"""Independently replay native material-boundary witnesses on the exact Float32 lattice."""

import json
from pathlib import Path
import sys
import numpy as np
from fast_skin_intersection_audit import owner, exact_integer_pairs


def verify(nodes, pack, report):
    x = np.load(nodes)
    p = np.load(pack)
    layers = [p[k] for k in ("l1", "l4294966545", "l4294966546")]
    rows = json.loads(report.read_text())
    total = 0
    for key, pairs in rows["witnesses"].items():
        a, b = map(int, key.split("-"))
        i, j = np.asarray(pairs).T
        faces = np.concatenate([layers[a][i], layers[b][j]])
        ids, local = np.unique(faces, return_inverse=True)
        rec, _ = owner.float32_surface_records(x[ids], local.reshape(-1, 3))
        half = len(i)
        rr = exact_integer_pairs(zip(rec[:half], rec[half:]), same_surface=True)
        if rr["count"] != half:
            raise ValueError(f"{key} native {half} but exact lattice {rr['count']}")
        total += half
    print("independently confirmed", total, flush=True)


if __name__ == "__main__":
    verify(*map(Path, sys.argv[1:]))

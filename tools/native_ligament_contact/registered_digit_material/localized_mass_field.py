"""Restrict the material solve to the measured connected neighborhoods of cleared interfaces."""

import json
import numpy as np
from scipy.spatial import cKDTree


def localize(root, loaded, field):
    record = json.loads((root / "constrained-1.5mm/interfaces.json").read_text())
    if record["new"] or not record["cleared"]:
        raise ValueError("Unverified antecedent")
    data = np.load(root / "physical.npz")
    source = np.load(root / "constrained-1.5mm/field.npz")
    requested = np.zeros(len(loaded))
    requested[source["nodes"]] = source["delta_x_m"]
    points = []
    for key, face, other, _ in record["cleared_witnesses"]:
        if key not in ("I", "D", "O"):
            raise ValueError("Unbound material interface")
        points.extend(data[key + "v"][data[key + "f"][face]])
    dist = cKDTree(np.unique(points, axis=0)).query(loaded)[0]
    t = np.clip((0.005 - dist) / 0.003, 0.0, 1.0)
    # Choose the displacement already qualified at these physical witnesses.
    return requested * 1000.0 * t * t * (3.0 - 2.0 * t)

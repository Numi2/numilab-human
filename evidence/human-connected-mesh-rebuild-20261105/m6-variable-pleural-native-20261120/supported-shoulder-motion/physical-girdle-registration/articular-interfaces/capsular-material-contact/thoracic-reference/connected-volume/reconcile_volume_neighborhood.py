"""Reconcile two physical reference maps at an observed triangle witness."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def splice(reference, candidate, donor, faces, seed_faces, inner_radius, taper):
    if inner_radius < 0 or taper <= 0:
        raise ValueError("Invalid material neighborhood")
    if candidate.shape != reference.shape or donor.shape != candidate.shape:
        raise ValueError("Different material domains")
    if candidate.dtype != np.float32 or donor.dtype != np.float32:
        raise ValueError("Use emitted Float32 coordinates")
    seeds = np.unique(faces[seed_faces])
    distances = cKDTree(reference[seeds]).query(reference)[0]
    t = np.clip((distances - inner_radius) / taper, 0, 1)
    weight = 1.0 - t**3 * (10 + t * (-15 + 6 * t))
    moved = weight > 0
    result = candidate.copy()
    result[moved] = candidate[moved].astype(float) + weight[moved, None] * (
        donor[moved].astype(float) - candidate[moved]
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("material", "contacts", "pairs", "candidate", "donor", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--bank", type=int, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--taper", type=float, required=True)
    a = parser.parse_args()
    witnesses = [
        face
        for bank, layer, pairs in json.loads(a.pairs.read_text())
        if (bank, layer) == (a.bank, a.layer)
        for _, face in pairs
    ]
    if not witnesses:
        raise ValueError("No witness for the requested physical interface")
    with np.load(a.material) as z, np.load(a.contacts) as p:
        y = splice(
            z["x"],
            np.load(a.candidate),
            np.load(a.donor),
            p[f"l{a.layer}"],
            witnesses,
            a.radius,
            a.taper,
        )
    np.save(a.output, y)

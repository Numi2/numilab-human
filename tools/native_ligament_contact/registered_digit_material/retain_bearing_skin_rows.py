"""Retain measured bearing-surface bindings while repairing adjacent free forearm skin."""

import csv
import numpy as np
from scipy.spatial import cKDTree
from registered_forearms import sha
from reweight_forearm_skin import ramp


def preserve(payload, neutral, rows, cases):
    verts = set()
    occupied = set()
    sources = {}
    for case in cases:
        for f, k in [
            ("resting-local-bed-footprint.csv", "vertex"),
            ("resting-support-selected-skin-witness.csv", "selected_skin_vertex_local"),
        ]:
            p = case / f
            sources[str(p)] = sha(p)
            with p.open() as stream:
                for sample in csv.DictReader(stream):
                    v = int(sample[k])
                    if not 0 <= v < len(neutral):
                        raise ValueError("Unregistered skin witness")
                    verts.add(v)
                    if (
                        f == "resting-local-bed-footprint.csv"
                        and float(sample["area_m2"]) > 0
                    ):
                        occupied.add(v)
    if len(occupied) < 1000 or len(verts) < len(occupied):
        raise ValueError("Incomplete source footprint")
    # Preserve the incident triangles as well as the quadrature points. Blend at
    # the boundary in material space. This edits source ancestry, never joint motion.
    faces = payload["inds"]
    seed = np.asarray(sorted(verts))
    ring = np.unique(np.r_[seed, faces[np.isin(faces, seed).any(1)].ravel()])
    dist, _ = cKDTree(neutral[ring]).query(neutral[rows])
    coeff = ramp((dist - 0.003) / 0.014)
    return coeff, dict(
        footprint_rows=len(occupied),
        all_witness_rows=len(verts),
        protected_rows=len(ring),
        data=sources,
    )

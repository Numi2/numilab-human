"""Cheap necessary volume-capacity bounds for existing native-domain-217.

These bounds reject impossible fixed-facet TetGen work before a volume build.
A passing bound neither certifies tetrahedralizability nor physical admission.
No geometric coincidence welding is performed.
"""
from __future__ import annotations
import numpy as np

NATIVE_MAX_NODES = 300_000
NATIVE_MAX_CELLS = 200_000

def closed_shell_bound(faces, *, label):
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or f.min() < 0:
        raise ValueError(f"{label}: invalid triangle table")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError(f"{label}: repeated triangle corner")
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError(f"{label}: duplicate triangle")
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"{label}: bound requires a closed edge-manifold shell")
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=signs, minlength=len(edges))
    if np.any(balance != 0):
        raise ValueError(f"{label}: inconsistent oriented edge")
    vertices = len(np.unique(f))
    chi_boundary = vertices - len(edges) + len(f)
    if chi_boundary % 2:
        raise ValueError(f"{label}: orientable closed boundary has odd Euler characteristic")
    # For a tetrahedral manifold with no removed boundary facets:
    # chi(boundary)=2 chi(volume), interior edges >= interior vertices.
    # Therefore T >= (F_boundary - chi_boundary)/2. Holes/cavities
    # are retained in the total Euler characteristic.
    minimum_cells = max(1, (len(f) - chi_boundary + 1) // 2)
    return {"label": label, "boundary_vertices": vertices,
            "boundary_edges": len(edges), "boundary_faces": len(f),
            "boundary_euler_characteristic": chi_boundary,
            "necessary_minimum_cells": minimum_cells}

def volume_capacity_report(skin_faces, fat_faces, surfaces,
                           *, max_nodes=NATIVE_MAX_NODES, max_cells=NATIVE_MAX_CELLS):
    parts = [closed_shell_bound(skin_faces, label="skin"),
             closed_shell_bound(fat_faces, label="fat")]
    # O/I/D input vertex identities remain distinct. I is shared once in
    # the combined mesh; each surface row is counted only when actually used.
    used = {role: len(np.unique(np.asarray(surfaces[role]["faces"], dtype=np.int64)))
            for role in ("O", "I", "D")}
    cells = sum(part["necessary_minimum_cells"] for part in parts)
    nodes = sum(used.values())
    failures = []
    if cells > max_cells: failures.append("necessary cell count exceeds native capacity")
    if nodes > max_nodes: failures.append("used boundary node count exceeds native capacity")
    return {"status": "PASS_NECESSARY_BOUNDS_ONLY" if not failures else "FAIL_NATIVE_CAPACITY",
            "physical_admission": False, "native_runtime": False,
            "materials": parts, "used_boundary_nodes_by_role": used,
            "necessary_combined_nodes": nodes, "necessary_combined_cells": cells,
            "maximum_nodes": max_nodes, "maximum_cells": max_cells,
            "failures": failures,
            "limitations": ["Steiner nodes, quality constraints and disconnected interiors may require more cells",
                           "No shape, intersection, material or mass admission is implied"]}

"""Select a material neighborhood by distance along connected tetrahedral edges."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def connected_nodes(points, cells, seeds, radius, *, eligible=None):
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells)
    seeds = np.unique(np.asarray(seeds))
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or cells.ndim != 2 or cells.shape[1] != 4
            or not np.issubdtype(cells.dtype, np.integer)
            or np.any((cells < 0) | (cells >= len(points)))
            or seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype, np.integer)
            or np.any((seeds < 0) | (seeds >= len(points)))
            or not np.isfinite(radius) or radius < 0):
        raise ValueError("Invalid material neighborhood")
    if not np.isin(seeds, cells).all():
        raise ValueError("A seed has no incident material cell")
    # A spatial broad phase may safely discard points farther than the path budget.
    within = cKDTree(points[seeds]).query(points)[0] <= radius
    if eligible is not None:
        eligible = np.asarray(eligible)
        if eligible.dtype != np.bool_ or eligible.shape != (len(points),) or not eligible[seeds].all():
            raise ValueError("Anatomical domain excludes a seed or uses different node identities")
        within &= eligible
    edges = []
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pair = cells[:, [a, b]]
        edges.append(np.sort(pair[within[pair].all(axis=1)], axis=1))
    edges = np.unique(np.concatenate(edges), axis=0)
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    if np.any(length == 0):
        raise ValueError("Coincident material edge")
    graph = sparse.coo_matrix((length, edges.T), shape=(len(points), len(points))).tocsr()
    distances = dijkstra(graph, directed=False, indices=seeds, limit=radius, min_only=True)
    return np.flatnonzero(np.isfinite(distances))

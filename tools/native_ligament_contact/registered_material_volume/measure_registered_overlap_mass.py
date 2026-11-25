"""Measure the mass inside a registered solid using the accepted physical tetrahedra."""

import igl
import manifold3d
import numpy as np
from igl.copyleft import cgal
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_TET_FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])


def overlap_mass(points, cells, cell_masses, cortex, faces, solid_volume):
    """Return occupied rows, volume fractions, masses and parent-barycentric centroids."""
    points, cortex = (np.asarray(x, dtype=float) for x in (points, cortex))
    cells, faces = (np.asarray(x, dtype=np.int64) for x in (cells, faces))
    cell_masses = np.asarray(cell_masses, dtype=float)
    if (
        cells.ndim != 2
        or cells.shape[1] != 4
        or not len(cells)
        or points.ndim != 2
        or points.shape[1] != 3
        or cortex.ndim != 2
        or cortex.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.isfinite(points).all()
        or not np.isfinite(cortex).all()
        or cells.min() < 0
        or cells.max() >= len(points)
        or faces.min() < 0
        or faces.max() >= len(cortex)
        or cell_masses.shape != (len(cells),)
        or not np.isfinite(cell_masses).all()
        or np.any(cell_masses <= 0)
        or not np.isfinite(solid_volume)
        or solid_volume < 0
    ):
        raise ValueError("Invalid registered material or solid")
    body = manifold3d.Manifold(manifold3d.Mesh64(cortex, faces.astype(np.uint64)))
    if body.status() != manifold3d.Error.NoError or body.volume() <= 0:
        raise ValueError("Registered cortex requires a positive oriented solid")
    tet = points[cells]
    volume = np.linalg.det(tet[:, 1:] - tet[:, :1]) / 6
    if np.any(volume <= 0):
        raise ValueError("The accepted material contains inverted cells")
    # Test unique material faces once. Treat enclosed components and a cortex
    # completely inside a tetrahedron separately from crossed tetrahedra.
    boundary, inverse = np.unique(
        np.sort(cells[:, _TET_FACES], axis=-1).reshape(-1, 3),
        axis=0,
        return_inverse=True,
    )
    compact, local = np.unique(boundary, return_inverse=True)
    pairs = cgal.intersect_other(
        points[compact],
        local.reshape(-1, 3),
        cortex,
        faces,
        detect_only=True,
        first_only=False,
    )[0]
    touched = np.isin(inverse.reshape(-1, 4), pairs[:, 0]).any(1)
    # A disconnected cortical component may fit in a single cell. A vertex
    # of every component is included in this conservative AABB query.
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)

    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(cortex), len(cortex)))
    _, components = connected_components(graph, directed=False)
    _, representatives = np.unique(components[faces.ravel()], return_index=True)
    lower, upper = tet.min(1), tet.max(1)
    for pos in cortex[faces.ravel()[representatives]]:
        touched |= np.all((lower <= pos) & (pos <= upper), axis=1)
    overlap = np.zeros(len(cells))
    barycentric = np.full((len(cells), 4), 0.25)
    if np.any(~touched):
        inside = abs(igl.winding_number(cortex, faces, tet[~touched].mean(1))) > 0.5
        overlap[~touched] = np.where(inside, volume[~touched], 0.0)
    face_u64 = _TET_FACES.astype(np.uint64)
    for i in np.flatnonzero(touched):
        cell = manifold3d.Manifold(manifold3d.Mesh64(tet[i], face_u64))
        cut = cell ^ body
        if (
            cell.status() != manifold3d.Error.NoError
            or cut.status() != manifold3d.Error.NoError
        ):
            raise ValueError(f"Unresolved solid clipping at local material cell {i}")
        overlap[i] = cut.volume()
        if overlap[i] > 0:
            mesh = cut.to_mesh64()
            vertices = mesh.vert_properties[:, :3]
            origin = vertices.mean(0)
            tri = vertices[mesh.tri_verts] - origin
            signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
            if signed.sum() <= 0:
                raise ValueError("Invalid oriented clipped moment")
            centroid = origin + signed @ tri.sum(1) / (4 * signed.sum())
            weights = np.linalg.solve((tet[i, 1:] - tet[i, 0]).T, centroid - tet[i, 0])
            barycentric[i] = np.r_[1 - weights.sum(), weights]
            if barycentric[i].min() < -1e-7 or barycentric[i].max() > 1 + 1e-7:
                raise ValueError(f"Clipped moment outside parent cell {i}")
    tol = 256 * np.finfo(float).eps * np.maximum(volume, solid_volume)
    if np.any(overlap < -tol) or np.any(overlap > volume + tol):
        raise ValueError("A clipped material volume lies outside its parent cell")
    overlap = np.clip(overlap, 0, volume)
    if not np.isclose(sum(overlap), solid_volume, rtol=1e-7, atol=1e-15):
        raise ValueError(
            f"Individual cell and aggregate solid volumes disagree: {sum(overlap)} / {solid_volume}"
        )
    rows = np.flatnonzero(overlap)
    share = overlap[rows] / volume[rows]
    return rows, share, share * cell_masses[rows], barycentric[rows]

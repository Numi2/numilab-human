"""Integrate the partitioned muscle on regular P1 coordinates. All integration weights come from clipped tissue."""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from manifold3d import Error, Manifold, Mesh64
from scipy import sparse


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def make_solid(x, faces, identity):
    # The registered partition preserves the orientation of nested internal cavities.
    vertices, indices = np.unique(x[faces.ravel()], axis=0, return_inverse=True)
    triangles = indices.reshape(-1, 3)
    source = Mesh64(
        np.ascontiguousarray(vertices, dtype=float),
        np.ascontiguousarray(triangles, dtype=np.uint64),
        run_index=np.array([0, 3 * len(triangles)], np.uint64),
        run_original_id=np.array([identity], np.uint32),
        face_id=np.arange(len(triangles), dtype=np.uint64),
    )
    solid = Manifold(source)
    if solid.status() != Error.NoError or solid.volume() <= 0:
        raise ValueError(
            "The registered muscle must be a closed, positively oriented material domain"
        )
    return solid


AXIS = np.eye(3)
ORDERS = list(itertools.permutations(range(3)))


def cubes(s, lo, hi):
    span = hi - lo
    if s.is_empty():
        return
    if span.max() == 1:
        yield lo, s
        return
    k = int(np.argmax(span))
    middle = (lo[k] + hi[k]) // 2
    above, below = s.split_by_plane(AXIS[k], float(middle))
    left = hi.copy()
    left[k] = middle
    right = lo.copy()
    right[k] = middle
    yield from cubes(below, lo, left)
    yield from cubes(above, right, hi)


def pieces(s, lower):
    for order in ORDERS:
        part = s
        for i, j in itertools.pairwise(order):
            normal = AXIS[i] - AXIS[j]
            distance = float(lower[i] - lower[j]) / np.sqrt(2.0)
            part = part.trim_by_plane(normal, distance)
            if part.is_empty():
                break
        if part.is_empty():
            continue
        corners = lower + np.vstack(
            (np.zeros(3, int), np.cumsum(AXIS[list(order)].astype(int), axis=0))
        )
        if np.linalg.det((corners[1:] - corners[0]).T) < 0:
            corners[[1, 2]] = corners[[2, 1]]
        yield corners, part


def build(parent, out, side, h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Coordinate spacing must be positive and finite")
    started = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    file = parent / (side + ".npz")
    z = np.load(file)
    x = z["positions"]
    f = z["faces"]
    origin = x.mean(0)
    u = (x - origin) / h
    identity = Manifold.reserve_ids(1)
    solid = make_solid(u, f, identity)
    vol = []
    local_mass = []
    corners = []
    cell, facet, basis, area = [], [], [], []
    reference_volume = solid.volume() * h**3
    for lower, cube in cubes(
        solid, np.floor(u.min(0)).astype(int), np.ceil(u.max(0)).astype(int)
    ):
        for corner, cut in pieces(cube, lower):
            if cut.status() != Error.NoError:
                raise ValueError("Invalid clipped material")
            mesh = cut.to_mesh64()
            p = np.asarray(mesh.vert_properties)[:, :3]
            g = np.asarray(mesh.tri_verts, dtype=int)
            tri = p[g]
            center = corner.mean(0)
            d = tri - center
            signed = (
                np.einsum("ij,ij->i", d[:, 0], np.cross(d[:, 1], d[:, 2])) * h**3 / 6
            )
            volume = signed.sum()
            if volume <= 0:
                if abs(volume) > 1e-16 * reference_volume:
                    raise ValueError(("Nonpositive cut volume", volume))
                continue
            inv = np.linalg.inv(corner[1:] - corner[0])
            q = (p - corner[0]) @ inv
            shape = np.column_stack((1 - q.sum(1), q))
            if shape.min() < -1e-8:
                raise ValueError(
                    (
                        "Support point outside basis tetrahedron",
                        shape.min(),
                        lower,
                        corner,
                    )
                )
            W = np.concatenate((np.full((len(g), 1, 4), 0.25), shape[g]), axis=1)
            sumW = W.sum(1)
            # Exact quadratic barycentric mass moments of each signed integration tetrahedron.
            M = np.einsum("f,fi,fj->ij", signed, sumW, sumW) + np.einsum(
                "f,fki,fkj->ij", signed, W, W
            )
            local_mass.append(M / 20)
            idx = len(corners)
            corners.append(corner)
            vol.append(volume)
            ids = np.repeat(
                np.asarray(mesh.run_original_id),
                np.diff(np.asarray(mesh.run_index, dtype=int)) // 3,
            )
            actual = np.flatnonzero(ids == identity)
            parents = np.asarray(mesh.face_id, dtype=int)[actual]
            keep = z["parent_bank"][parents] != 0
            rows = actual[keep]
            if len(rows):
                cell.extend([idx] * len(rows))
                facet.extend(parents[keep])
                basis.extend(shape[g[rows]].mean(1))
                faces_now = tri[rows]
                area.extend(
                    0.5
                    * h
                    * h
                    * np.cross(
                        faces_now[:, 1] - faces_now[:, 0],
                        faces_now[:, 2] - faces_now[:, 0],
                    )
                )
    grid, indices = np.unique(
        np.asarray(corners).reshape(-1, 3), axis=0, return_inverse=True
    )
    T = indices.reshape(-1, 4)
    X = grid * h + origin
    V = np.asarray(vol)
    M = np.asarray(local_mass)
    n = len(X)
    I = np.broadcast_to(T[:, :, None], M.shape).ravel()
    J = np.broadcast_to(T[:, None, :], M.shape).ravel()
    mass = sparse.coo_matrix((M.ravel(), (I, J)), shape=(n, n)).tocsr()
    shape_gradient = np.linalg.inv((X[T[:, 1:]] - X[T[:, :1]]).transpose(0, 2, 1))
    G = np.concatenate(
        (-shape_gradient.sum(1, keepdims=True), shape_gradient), axis=1
    ).transpose(0, 2, 1)
    ii = np.broadcast_to(np.arange(3 * len(T)).reshape(-1, 3, 1), G.shape)
    jj = np.broadcast_to(T[:, None, :], G.shape)
    gradient = sparse.coo_matrix(
        (G.ravel(), (ii.ravel(), jj.ravel())), shape=(3 * len(T), n)
    ).tocsr()
    cell, facet, basis, area = map(np.asarray, (cell, facet, basis, area))
    if not len(cell):
        raise ValueError("No registered interface quadrature")
    points = np.einsum("ij,ijk->ik", basis, X[T[cell]])
    tri = x[f[facet]]
    u0, v0 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    delta = points - tri[:, 0]
    normal = np.cross(u0, v0)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    if np.any(norm2 <= 0):
        raise ValueError("Degenerate source face")
    bu = np.einsum("ij,ij->i", np.cross(delta, v0), normal) / norm2
    bv = np.einsum("ij,ij->i", np.cross(u0, delta), normal) / norm2
    minimum_weight = float(np.minimum.reduce([bu, bv, 1 - bu - bv]).min())
    plane_distance = float(
        (abs(np.einsum("ij,ij->i", delta, normal)) / np.sqrt(norm2)).max()
    )
    if minimum_weight < -1e-6 or plane_distance > 2e-7:
        raise ValueError("Quadrature escaped its registered source face")
    ref = x[f]
    ref_area = 0.5 * np.cross(ref[:, 1] - ref[:, 0], ref[:, 2] - ref[:, 0])
    sums = np.zeros_like(ref_area)
    np.add.at(sums, facet, area)
    interface = z["parent_bank"] != 0
    total_area = np.linalg.norm(ref_area[interface], axis=1).sum()
    area_error = (
        np.linalg.norm(sums[interface] - ref_area[interface], axis=1).sum() / total_area
    )
    relvol = abs(V.sum() - reference_volume) / reference_volume
    moment0 = np.abs(mass.sum() - reference_volume) / reference_volume
    derivative = np.max(abs((gradient @ X).reshape(-1, 3, 3) - np.eye(3)))
    if max(area_error, relvol, moment0, derivative) > 1e-8:
        raise ValueError(
            ("Conservation error", area_error, relvol, moment0, derivative)
        )
    if np.min(np.linalg.eigvalsh(M)) < -1e-12 * V.max():
        raise ValueError("Negative consistent mass")
    root = out / side
    root.mkdir(exist_ok=True)
    np.savez_compressed(
        root / "parents.npz",
        reference=X,
        tetrahedra=T,
        volume=V,
        interface_cells=cell,
        interface_parent_faces=facet,
        interface_shape=basis,
        interface_area_vectors=area,
    )
    sparse.save_npz(root / "gradient.npz", gradient)
    sparse.save_npz(root / "unit-density-mass.npz", mass)
    report = {
        "side": side,
        "source": sha(file),
        "coordinate_spacing_m": h,
        "nodes": n,
        "integration_domains": len(V),
        "physical_volume_ml": V.sum() * 1e6,
        "minimum_fill_fraction": float(min(V) / (h**3 / 6)),
        "mass_row_sum_relative_error": moment0,
        "volume_relative_error": relvol,
        "interface_area_relative_error": area_error,
        "affine_gradient_error": derivative,
        "interface_quadrature_points": len(cell),
        "minimum_source_barycentric_weight": minimum_weight,
        "maximum_source_plane_distance_m": plane_distance,
        "construction_s": time.perf_counter() - started,
        "files": {p.name: sha(p) for p in root.glob("*.npz")},
        "scope": "Exact tissue integration for continuous piecewise-linear coordinates; detailed geometry and source interface ancestry are retained. Unit-density mass can be scaled or projected. This material candidate has not replaced a native mass owner.",
    }
    (root / "construction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("member", help="Atlas member stem in the partition directory")
    parser.add_argument("spacing_m", type=float)
    args = parser.parse_args()
    build(args.partition, args.output, args.member, args.spacing_m)

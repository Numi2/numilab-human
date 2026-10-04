from __future__ import annotations

import itertools
import math

from numilab_human.healthy_total_body_ct_voxel_tets import (
    FACE_CORNERS,
    _fill_interior,
    _point_index_map,
    _surface_boundary_cells,
    _tetrahedra_for_voxel,
    _voxel_boundary_triangles,
    _world_m,
    _cross,
)


def _surface_for_voxels(voxels: set[tuple[int, int, int]], matrix: list[list[float]]):
    point_ids: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for i, j, k in sorted(voxels):
        for axis, sign, corners in FACE_CORNERS:
            neighbor = [i, j, k]
            neighbor[axis] += sign
            if tuple(neighbor) in voxels:
                continue
            quad = [tuple(2 * (i, j, k)[d] + corner[d] for d in range(3)) for corner in corners]
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                ids = []
                for key in tri:
                    if key not in point_ids:
                        point_ids[key] = len(vertices)
                        ijk = tuple(value * 0.5 for value in key)
                        vertices.append(tuple(matrix[d][d] * ijk[d] + matrix[d][3] for d in range(3)))
                    ids.append(point_ids[key])
                face = tuple(ids)
                # A reflected affine reverses triangle orientation in RAS.
                determinant_sign = matrix[0][0] * matrix[1][1] * matrix[2][2]
                if determinant_sign < 0.0:
                    face = (face[0], face[2], face[1])
                faces.append(face)
    return vertices, faces


def _positive_tet_volumes(voxels: set[tuple[int, int, int]], matrix: list[list[float]]) -> list[float]:
    point_ids = _point_index_map(voxels)
    coordinates = {point_id: _world_m(key, matrix) for key, point_id in point_ids.items()}
    volumes = []
    for voxel in sorted(voxels):
        for tet in _tetrahedra_for_voxel(voxel, point_ids, matrix):
            p = [coordinates[index] for index in tet]
            a = tuple(p[1][d] - p[0][d] for d in range(3))
            b = tuple(p[2][d] - p[0][d] for d in range(3))
            c = tuple(p[3][d] - p[0][d] for d in range(3))
            volumes.append(sum(a[d] * _cross(b, c)[d] for d in range(3)) / 6.0)
    return volumes


def test_closed_surface_reconstructs_exact_voxel_volume_and_conforming_tets() -> None:
    voxels = {(i, j, k) for i in range(2) for j in range(3) for k in range(2)}
    # A reflected, anisotropic scan affine exercises orientation and physical units.
    matrix = [
        [-0.8, 0.0, 0.0, 20.0],
        [0.0, 1.1, 0.0, -30.0],
        [0.0, 0.0, 2.2, 40.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    vertices, faces = _surface_for_voxels(voxels, matrix)

    boundary, source_triangles = _surface_boundary_cells(vertices, faces, matrix)
    reconstructed, dimensions = _fill_interior(boundary)

    assert reconstructed == voxels
    assert all(size > 2 for size in dimensions)
    assert _voxel_boundary_triangles(reconstructed) == source_triangles

    volumes = _positive_tet_volumes(reconstructed, matrix)
    expected_each_tet_m3 = 0.8 * 1.1 * 2.2 * 1e-9 / 6.0
    assert len(volumes) == 6 * len(voxels)
    assert all(volume > 0.0 and math.isclose(volume, expected_each_tet_m3, rel_tol=2e-14)
               for volume in volumes)
    assert math.isclose(sum(volumes), len(voxels) * 0.8 * 1.1 * 2.2 * 1e-9,
                        rel_tol=2e-14)


def test_disconnected_voxel_solid_is_not_bridged_by_exterior_fill() -> None:
    voxels = {(0, 0, 0), (3, 0, 0)}
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    vertices, faces = _surface_for_voxels(voxels, matrix)
    boundary, _ = _surface_boundary_cells(vertices, faces, matrix)
    reconstructed, _ = _fill_interior(boundary)
    assert reconstructed == voxels

import numpy as np
import pytest

from numilab_human import cardiac_rodero04_lat_import as importer
from numilab_human.cardiac_rodero04_lat_import import map_cell_connectivity
from numilab_human.model import ImportError as HumanImportError


def test_maps_permuted_tetgen_rows_and_node_order_to_source_cells():
    vtk_tetrahedra = np.array(
        [
            [3, 2, 1, 0],
            [4, 5, 6, 7],
            [0, 8, 9, 10],
        ],
        dtype=np.uint32,
    )
    tetgen_tetrahedra = np.array(
        [
            [7, 6, 5, 4],
            [10, 9, 8, 0],
            [2, 0, 3, 1],
        ],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(
        map_cell_connectivity(vtk_tetrahedra, tetgen_tetrahedra),
        np.array([1, 2, 0], dtype=np.uint32),
    )


def test_rejects_tetgen_cell_missing_from_vtk_source():
    vtk_tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.uint32)
    tetgen_tetrahedra = np.array([[0, 1, 2, 4]], dtype=np.uint32)

    with pytest.raises(HumanImportError, match="cells absent from source VTK"):
        map_cell_connectivity(vtk_tetrahedra, tetgen_tetrahedra)


def test_rejects_duplicate_source_connectivity():
    vtk_tetrahedra = np.array(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ],
        dtype=np.uint32,
    )
    tetgen_tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.uint32)

    with pytest.raises(HumanImportError, match="duplicate tetrahedron connectivity"):
        map_cell_connectivity(vtk_tetrahedra, tetgen_tetrahedra)


def test_parses_packed_vtk_points_and_modern_offsets_header(monkeypatch):
    monkeypatch.setattr(
        importer,
        "VTK_EXPECTED_COUNTS",
        {
            "points": 4,
            "cells": 2,
            "tetrahedra": 1,
            "triangles": 1,
        },
    )
    vtk = b"""# vtk DataFile Version 5.1
Rodero cell map fixture
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0 1 0 0 0 1 0
0 0 1
CELLS 3 7
OFFSETS vtktypeint64
0 4 7
CONNECTIVITY vtktypeint64
0 1 2 3 0 2 1
CELL_TYPES 2
10 5
"""

    points, tetrahedron_cells, tetrahedra, cell_count = importer._parse_vtk(vtk)

    assert points.shape == (4, 3)
    assert cell_count == 2
    np.testing.assert_array_equal(tetrahedron_cells, np.array([0]))
    np.testing.assert_array_equal(tetrahedra, np.array([[0, 1, 2, 3]]))

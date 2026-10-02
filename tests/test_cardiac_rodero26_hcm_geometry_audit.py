import json

import numpy as np
import pytest

from numilab_human import cardiac_rodero26_hcm_geometry_audit as audit
from numilab_human.model import ImportError as HumanImportError


def _write_tiny_vtk(path, *, connectivity=None, cell_types=None):
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=">f4",
    )
    rows = np.asarray(
        connectivity or [[4, 0, 1, 2, 3], [4, 1, 2, 3, 4]], dtype=">i4"
    )
    types = np.asarray(cell_types or [10, 10], dtype=">i4")
    tags = np.asarray([1, 25], dtype=">i4")
    path.write_bytes(
        b"# vtk DataFile Version 3.0\nsource fixture\nBINARY\n"
        b"DATASET UNSTRUCTURED_GRID\n\nPOINTS 5 float\n"
        + points.tobytes()
        + b"\nCELL_TYPES 2\n"
        + types.tobytes()
        + b"\nCELLS 2 10\n"
        + rows.tobytes()
        + b"\nCELL_DATA 2\nSCALARS elemTag int 1\nLOOKUP_TABLE default\n"
        + tags.tobytes()
    )


def test_reads_source_order_tetra_mesh_and_registered_tags(tmp_path):
    path = tmp_path / "tiny.vtk"
    _write_tiny_vtk(path)

    mesh = audit.parse_vtk_tetra_mesh(path, expected_points=5)

    assert mesh["point_count"] == 5
    assert mesh["cell_count"] == 2
    assert mesh["cell_type_counts"] == {"vtk_tetra": 2}
    np.testing.assert_array_equal(mesh["connectivity"], [[0, 1, 2, 3], [1, 2, 3, 4]])
    np.testing.assert_array_equal(mesh["tags"], [1, 25])
    np.testing.assert_array_equal(mesh["points_um"][4], [1.0, 1.0, 1.0])


def test_rejects_non_tetra_cell_type(tmp_path):
    path = tmp_path / "wrong-cell-type.vtk"
    _write_tiny_vtk(path, cell_types=[10, 5])

    with pytest.raises(HumanImportError, match="non-tetrahedron"):
        audit.parse_vtk_tetra_mesh(path, expected_points=5)


def test_rejects_out_of_range_cell_connectivity(tmp_path):
    path = tmp_path / "bad-connectivity.vtk"
    _write_tiny_vtk(path, connectivity=[[4, 0, 1, 2, 3], [4, 1, 2, 3, 5]])

    with pytest.raises(HumanImportError, match="invalid point"):
        audit.parse_vtk_tetra_mesh(path, expected_points=5)


def test_rejects_activation_receipt_with_bad_report_digest(tmp_path):
    field = tmp_path / audit.ACTIVATION_FIELD
    receipt = tmp_path / "source-activation-import.json"
    field.write_bytes(b"placeholder")
    receipt.write_text(json.dumps({"schema": "untrusted"}), encoding="utf-8")

    with pytest.raises(HumanImportError, match="report digest is invalid"):
        audit._activation_field(field, receipt, expected_points=5)

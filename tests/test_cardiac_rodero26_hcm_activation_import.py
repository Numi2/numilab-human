import json

import numpy as np
import pytest

from numilab_human import cardiac_rodero26_hcm_activation_import as importer
from numilab_human.model import ImportError as HumanImportError


def test_reads_hcm1_legacy_binary_vtk_point_count(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "POINT_COUNT", 4)
    path = tmp_path / "mesh.vtk"
    path.write_bytes(
        b"# vtk DataFile Version 3.0\nvtk output\nbinary\n"
        b"DATASET UNSTRUCTURED_GRID\n\nPOINTS 4 float\n"
        + b"\x00\x00\x00\x00" * 12
    )

    assert importer._vtk_point_count(path) == 4


def test_rejects_point_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "POINT_COUNT", 5)
    path = tmp_path / "mesh.vtk"
    path.write_bytes(
        b"# vtk DataFile Version 3.0\nvtk output\nbinary\n"
        b"DATASET UNSTRUCTURED_GRID\n\nPOINTS 4 float\n"
    )

    with pytest.raises(HumanImportError, match="point count differs"):
        importer._vtk_point_count(path)


def test_reads_exactly_one_finite_time_or_publisher_inactive_sentinel_per_point(tmp_path):
    path = tmp_path / "activation.dat"
    path.write_text("0.25\n-1\n4.75\n", encoding="ascii")

    np.testing.assert_array_equal(importer._read_activation(path, 3), [0.25, -1.0, 4.75])

    with pytest.raises(HumanImportError, match="per mesh point"):
        importer._read_activation(path, 2)

    path.write_text("0.25\n-2.0\n4.75\n", encoding="ascii")
    with pytest.raises(HumanImportError, match="-1 inactive sentinel"):
        importer._read_activation(path, 3)


def test_requires_exact_publisher_sample_parameter_schema(tmp_path):
    path = tmp_path / "53.json"
    parameters = {
        "CV_f_v": 0.5,
        "ani_ratio_v": 0.3,
        "k_FEC": 6.8,
        "CV_f_a": 0.6,
        "ani_ratio_a": 0.2,
        "k_BB": 1.2,
    }
    path.write_text(json.dumps({"EP": parameters}), encoding="utf-8")

    assert importer._read_parameters(path) == parameters

    parameters["unreviewed"] = 1.0
    path.write_text(json.dumps({"EP": parameters}), encoding="utf-8")
    with pytest.raises(HumanImportError, match="reviewed field schema"):
        importer._read_parameters(path)

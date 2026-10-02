import pytest

from numilab_human.cardiac_cavity_intersections import _records
from numilab_human.muscle_surface_volume_disjointness import (
    MuscleDisjointnessError,
    _surface_pair_status,
    _write_immutable,
)


TETRA_FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


def _mesh(vertices):
    scaled = [tuple(round(value * 1000) for value in point) for point in vertices]
    return {"vertices": scaled, "records": _records(scaled, TETRA_FACES)}


def test_exact_pair_aabb_separation_proves_disjoint_domains():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(2, 0, 0), (3, 0, 0), (2, 1, 0), (2, 0, 1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "strictly_disjoint_aabbs"
    assert result["intersection_pairs"] == 0


def test_exact_pair_reports_crossing_surface_intersection():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0.2, 0, 0), (1.2, 0, 0), (0.2, 1, 0), (0.2, 0, 1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "surface_intersection"
    assert result["intersection_pairs"] > 0


def test_exact_pair_fails_closed_on_coplanar_boundary_contact():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "surface_intersection"
    assert result["intersection_pairs"] > 0


def test_exact_pair_distinguishes_containment_without_surface_crossing():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "nested_closed_domains"
    assert result["intersection_pairs"] == 0


def test_immutable_receipt_writer_rejects_symlink_output(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve", encoding="utf-8")
    output = tmp_path / "receipt.json"
    output.symlink_to(target)

    with pytest.raises(MuscleDisjointnessError, match="symlink"):
        _write_immutable(output, {"schema": "test"})

    assert target.read_text(encoding="utf-8") == "preserve"

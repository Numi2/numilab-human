from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as av
from numilab_human import cardiac_cavity_intersections as ci

FACES = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4")


def target(*, nested=False):
    outer = np.asarray([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype="<f4")
    if nested:
        inner = outer * np.float32(.125) + np.float32(.5)
        faces = np.concatenate((FACES, FACES[:, ::-1] + 4))
    else:
        inner = outer + np.asarray([8, 0, 0], dtype="<f4")
        faces = np.concatenate((FACES, FACES + 4))
    return av.prepare_surface((51010, 315), np.concatenate((outer, inner)), faces)


@pytest.mark.parametrize("nested,points", [
    (False, [(Fraction(1, 4),) * 3, (Fraction(33, 4), Fraction(1, 4), Fraction(1, 4)),
             (Fraction(6), Fraction(1), Fraction(1)), (Fraction(0),) * 3]),
    (True, [(Fraction(1, 4),) * 3, (Fraction(5, 8),) * 3,
            (Fraction(1, 2),) * 3, (Fraction(6),) * 3]),
])
def test_component_queries_match_complete_exact_index_including_nested_voids(nested, points):
    surface = target(nested=nested)
    assert surface.closed and surface.self_audit()["count"] == 0
    full = surface.meter_index()
    expected = [ci.point_location_prepared_indexed(p, full)["location"] for p in points]
    # Four faces is the fixture bound; the production predicate keeps its
    # bounded face guard and is never enlarged or monkeypatched.
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 4):
        actual = [surface.point_location_meters(p)["location"] for p in points]
    assert actual == expected
    assert len(surface._meter_component_indices) == 2
    assert all(len(part.records) <= 4 for part, _lo, _hi in surface._meter_component_indices)
    assert sum(len(part.records) for part, _lo, _hi in surface._meter_component_indices) == 8


@pytest.mark.parametrize("nested", [False, True])
def test_oversized_component_uses_complete_shared_ray_chunks(nested):
    surface = target(nested=nested)
    full = surface.meter_index()
    points = [(Fraction(1,4),)*3, (Fraction(5,8),)*3,
              (Fraction(-1),)*3, (Fraction(6),)*3, (Fraction(0),)*3]
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 3):
        for point in points:
            actual = surface.point_location_meters(point)
            expected = ci.point_location_prepared_indexed(point, full)
            assert {k:actual[k] for k in expected} == expected
            assert actual["method"] == "exact_shared_ray_face_chunks"
    assert surface._meter_component_indices is None
    assert sum(len(part.records) for part in surface._meter_face_chunks) == 8
    assert all(len(part.records) <= 3 for part in surface._meter_face_chunks)


def test_chunk_capacity_still_fails_without_truncation():
    surface = target()
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 1), patch.object(ci, "MAX_POINT_LOCATION_CHUNKS", 2):
        with pytest.raises(ValueError, match="chunk count"):
            surface.point_location_meters((Fraction(1,4),)*3)
    assert surface._meter_face_chunks is None


def test_component_containment_keeps_enclosed_solid_failure():
    enclosing = target()
    positions = np.asarray([[.2, .2, .2], [.3, .2, .2], [.2, .3, .2], [.2, .2, .3]], dtype="<f4")
    enclosed = av.prepare_surface((51004, 50), positions, FACES)
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 4):
        report = av.audit_surface_pair(enclosed, enclosing)
    assert not report["passed"]
    assert not report["unexpected_triangle_pairs"]
    assert any(row["inside_component_witnesses"] for row in report["containment"])


@pytest.mark.parametrize("face_limit", [3, 4, ci.MAX_PREPARED_POINT_LOCATION_FACES])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("direction,expected", [
    (0, "boundary"), (-1, "inside"), (1, "outside"),
])
def test_binary_meter_queries_preserve_boundary_neighbors(face_limit, dtype, direction, expected):
    surface = target()
    point = np.asarray([.25, .25, 3.5], dtype=dtype)
    if direction:
        point[2] = np.nextafter(point[2], dtype(np.inf if direction > 0 else -np.inf))
    exact = tuple(Fraction.from_float(float(value)) for value in point)
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", face_limit):
        result = surface.point_location_meters(point)
        assert result == surface.point_location_meters(exact)
    assert result["location"] == expected


@pytest.mark.parametrize("point", [
    [0., 0., float("nan")], [0., 0., float("inf")],
    [0., 0., -float("inf")], [0., 0.], [0., 0., 0., 0.],
    [True, 0, 0], ["0", 0, 0], None,
])
def test_invalid_meter_queries_fail_before_preparing_containment_index(point):
    surface = target()
    with pytest.raises(ValueError, match="query"):
        surface.point_location_meters(point)
    assert surface._meter_index is None
    assert surface._meter_component_indices is None


def test_surface_and_predicate_share_the_production_resource_bound():
    assert av._POINT_LOCATION_COMPONENT_FACE_LIMIT == ci.MAX_PREPARED_POINT_LOCATION_FACES

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
    # existing 200,000 face guard and is never enlarged or monkeypatched.
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 4):
        actual = [surface.point_location_meters(p)["location"] for p in points]
    assert actual == expected
    assert len(surface._meter_component_indices) == 2
    assert all(len(part.records) <= 4 for part, _lo, _hi in surface._meter_component_indices)
    assert sum(len(part.records) for part, _lo, _hi in surface._meter_component_indices) == 8


def test_individual_oversized_component_is_rejected_without_truncation():
    surface = target()
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 3):
        with pytest.raises(ValueError, match="individual closed component"):
            surface.point_location_meters((Fraction(1, 4),) * 3)
    assert surface._meter_component_indices is None


def test_component_containment_keeps_enclosed_solid_failure():
    enclosing = target()
    positions = np.asarray([[.2, .2, .2], [.3, .2, .2], [.2, .3, .2], [.2, .2, .3]], dtype="<f4")
    enclosed = av.prepare_surface((51004, 50), positions, FACES)
    with patch.object(av, "_POINT_LOCATION_COMPONENT_FACE_LIMIT", 4):
        report = av.audit_surface_pair(enclosed, enclosing)
    assert not report["passed"]
    assert not report["unexpected_triangle_pairs"]
    assert any(row["inside_component_witnesses"] for row in report["containment"])

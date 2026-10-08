import numpy as np
import pytest

from numilab_human.ocular_skin_registration_constrained import (
    _target_ring_deltas, radial_clearance_and_gradient_mm,
)


def _ring(movable=(10, 11)):
    return {
        "name": "right",
        "vertex_ids": [10, 11, 12],
        "movable_vertex_ids": list(movable),
        "center_world_m": [0.0, 0.0, 0.0],
        "radial_displacement_m": [0.00031, 0.00028, 0.0],
        "_captured_world_m": {
            "10": [0.01, 0.0, 0.0],
            "11": [0.0, 0.02, 0.0],
            "12": [0.0, 0.0, 0.03],
        },
    }


def test_movable_inward_ring_ids_are_free_and_outside_anchor_remains_exact():
    fixed, by_side = _target_ring_deltas([_ring()])

    assert by_side == {"right": [10, 11, 12]}
    assert set(fixed) == {12}
    np.testing.assert_array_equal(fixed[12], np.zeros(3))


def test_movable_ring_rejects_ids_outside_boundary():
    with pytest.raises(ValueError, match="movable boundary IDs are outside"):
        _target_ring_deltas([_ring(movable=(10, 99))])


def test_active_facet_radial_clearance_gradient_matches_finite_difference():
    shell = np.array([[[-100.0, -100.0, 10.0], [100.0, -100.0, 10.0], [0.0, 100.0, 10.0]]])
    point = np.array([3.0, 4.0, 20.0])
    center = np.zeros(3)
    gap, gradient = radial_clearance_and_gradient_mm(point, center, shell)
    finite = np.empty(3)
    step = 1.0e-5
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        plus = radial_clearance_and_gradient_mm(point + offset, center, shell)[0]
        minus = radial_clearance_and_gradient_mm(point - offset, center, shell)[0]
        finite[axis] = (plus - minus) / (2.0 * step)
    assert np.isfinite(gap)
    np.testing.assert_allclose(gradient, finite, atol=1.0e-8, rtol=1.0e-8)

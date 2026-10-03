from fractions import Fraction

import numpy as np

from numilab_human.patellofemoral_pose_intersection_audit import (
    _collision_gate,
    _exact_surface_pair_audit,
    _float_integer_lattice,
)


def test_binary64_points_are_represented_exactly_on_one_lattice():
    points = np.asarray([[0.1, -0.2, 2.0], [1.0 / 7.0, 3.5, -8.25]])
    integers, exponent = _float_integer_lattice(points)
    denominator = 1 << exponent
    for source, encoded in zip(points, integers, strict=True):
        assert [Fraction(value, denominator) for value in encoded] == [
            Fraction.from_float(float(value)) for value in source
        ]


def test_exact_surface_pair_audit_finds_crossing_and_rejects_separation():
    first_vertices = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ])
    second_vertices = np.asarray([
        [0.25, 0.25, -1.0], [0.25, 0.25, 1.0], [1.0, 0.25, 0.0],
    ])
    faces = np.asarray([[0, 1, 2]])

    crossing = _exact_surface_pair_audit(
        first_vertices, faces, second_vertices, faces
    )
    assert crossing["surface_intersection_triangle_pair_count"] == 1
    assert crossing["exact_unique_intersection_point_count"] >= 2
    assert crossing["intersection_world_bounds_m"] is not None

    separated = _exact_surface_pair_audit(
        first_vertices, faces, second_vertices + [0.0, 0.0, 2.0], faces
    )
    assert separated["surface_intersection_triangle_pair_count"] == 0
    assert separated["intersection_world_bounds_m"] is None


def test_out_of_range_intersection_is_diagnostic_but_valid_pose_intersection_fails():
    diagnostic_only = {
        "name": "outside-source-range",
        "projected_ranges": {"all_projected_coordinates_within_declared_ranges": False},
        "by_side": {"r": {"surface_intersection_triangle_pair_count": 7}},
    }
    passed = _collision_gate([diagnostic_only])
    assert passed["range_valid_pose_surface_intersections_absent"] is True
    assert passed["range_valid_intersection_pair_count"] == 0

    valid = {
        "name": "within-source-range",
        "projected_ranges": {"all_projected_coordinates_within_declared_ranges": True},
        "by_side": {"l": {"surface_intersection_triangle_pair_count": 4}},
    }
    failed = _collision_gate([diagnostic_only, valid])
    assert failed["range_valid_pose_surface_intersections_absent"] is False
    assert failed["range_valid_intersecting_pose_side_count"] == 1
    assert failed["range_valid_intersection_pair_count"] == 4

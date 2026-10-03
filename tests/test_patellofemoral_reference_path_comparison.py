import math

import pytest

from numilab_human.patellofemoral_reference_path_comparison import (
    _interpolate,
    _interpolate_reference_pose,
    _quaternion_angle_degrees,
    _quaternion_rotate,
    _relative_reference_rows,
)


def _rotation_z(degrees):
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _synthetic_archive():
    records = []
    for step in range(1, 141):
        flexion = 90.0 * (step - 1) / 139.0
        time = 0.05 + 1.95 * (step - 1) / 139.0
        records.extend([
            {
                "step": step,
                "continuation_time": time,
                "field": "center_of_mass",
                "units": "mm",
                "values": {"1": [1.0, 0.0, 0.0], "3": [0.0, 0.0, 0.0],
                           "4": [0.0, 0.0, 0.0]},
            },
            {
                "step": step,
                "continuation_time": time,
                "field": "rotation_quaternion",
                "units": "unit quaternion xyzw",
                "values": {"1": _rotation_z(flexion * 0.5),
                           "3": _rotation_z(flexion), "4": [0.0, 0.0, 0.0, 1.0]},
            },
        ])
    return {
        "status": "normal_termination_with_complete_observations",
        "observations_complete": True,
        "accepted_increment_count": 140,
        "time_semantics": "quasi-static continuation parameter, not elapsed physical seconds",
        "records": records,
    }


def test_quaternion_rotation_and_angle_preserve_rigid_transform_semantics():
    rotated = _quaternion_rotate(_rotation_z(90.0), [1.0, 0.0, 0.0])
    assert rotated == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert _quaternion_angle_degrees(_rotation_z(30.0)) == pytest.approx(30.0)


def test_reference_parser_joins_patella_tibia_and_femur_by_source_body_id():
    rigid_bodies = [
        {"id": "1", "name": "PTB"},
        {"id": "3", "name": "TBB"},
        {"id": "4", "name": "FMB"},
    ]
    rows = _relative_reference_rows(_synthetic_archive(), rigid_bodies)
    assert len(rows) == 140
    assert rows[0]["knee_flexion_deg"] == pytest.approx(0.0)
    assert rows[-1]["knee_flexion_deg"] == pytest.approx(90.0)
    assert rows[-1]["patella_position_relative_femur_com_mm"] == pytest.approx([1.0, 0.0, 0.0])
    assert rows[-1]["patella_rotation_relative_femur_xyzw"] == pytest.approx(
        _rotation_z(45.0)
    )


def test_reference_parser_rejects_changed_rigid_body_identity():
    with pytest.raises(RuntimeError, match="rigid-body identities changed"):
        _relative_reference_rows(
            _synthetic_archive(),
            [{"id": "1", "name": "PTB"}, {"id": "3", "name": "TBB"},
             {"id": "4", "name": "femur"}],
        )


def test_interpolation_requires_observed_flexion_range_and_uses_bracketing_states():
    rows = [
        {"knee_flexion_deg": 10.0, "displacement": 2.0},
        {"knee_flexion_deg": 30.0, "displacement": 8.0},
    ]
    assert _interpolate(rows, "displacement", 20.0) == pytest.approx(5.0)
    with pytest.raises(RuntimeError, match="requires extrapolation"):
        _interpolate(rows, "displacement", 0.0)


def test_pose_interpolation_uses_linear_origin_and_shortest_quaternion_arc():
    rows = [
        {
            "knee_flexion_deg": 10.0,
            "patella_position_relative_femur_com_mm": [0.0, 0.0, 0.0],
            "patella_rotation_relative_femur_xyzw": _rotation_z(0.0),
        },
        {
            "knee_flexion_deg": 30.0,
            "patella_position_relative_femur_com_mm": [20.0, 0.0, 0.0],
            "patella_rotation_relative_femur_xyzw": _rotation_z(40.0),
        },
    ]
    position, rotation = _interpolate_reference_pose(rows, 20.0)
    assert position == pytest.approx([10.0, 0.0, 0.0])
    assert _quaternion_angle_degrees(rotation) == pytest.approx(20.0)

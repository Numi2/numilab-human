import math

import numpy as np
import pytest

from numilab_human.patellofemoral_reference_transfer_audit import (
    PLAN_SCHEMA,
    _reference_pose_at,
    _relative_pose_from_reference,
    _validate_plan,
)


def _quat_z(degrees):
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def test_reference_delta_is_rotated_and_scaled_into_myosim_femur_frame():
    # Proper cyclic frame map: source x/y/z become target z/x/y.
    rotation = np.asarray([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    baseline_reference = {
        "patella_position_relative_femur_com_mm": [0.0, 0.0, 0.0],
        "patella_rotation_relative_femur_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    current_reference = {
        "patella_position_relative_femur_com_mm": [0.0, 10.0, 0.0],
        "patella_rotation_relative_femur_xyzw": _quat_z(30.0),
    }
    baseline_myosim = {
        "patella_inertial_com_relative_femur_inertial_com_m": [0.1, 0.2, 0.3],
        "patella_rotation_relative_femur_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    frame = {"rotation": rotation, "scale": 2.0}

    position, orientation = _relative_pose_from_reference(
        current_reference, baseline_reference, baseline_myosim, frame, np
    )

    assert position == pytest.approx([0.12, 0.2, 0.3])
    # Source z rotates into target y under this registration.
    expected = [0.0, math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0))]
    assert orientation == pytest.approx(expected)


def test_reference_pose_interpolates_position_and_shortest_quaternion_arc():
    rows = [
        {"step": 1, "knee_flexion_deg": 10.0,
         "patella_position_relative_femur_com_mm": [0.0, 0.0, 0.0],
         "patella_rotation_relative_femur_xyzw": _quat_z(0.0)},
        {"step": 2, "knee_flexion_deg": 20.0,
         "patella_position_relative_femur_com_mm": [10.0, 20.0, 30.0],
         "patella_rotation_relative_femur_xyzw": _quat_z(40.0)},
    ]
    result = _reference_pose_at(rows, 15.0)
    assert result["patella_position_relative_femur_com_mm"] == pytest.approx(
        [5.0, 10.0, 15.0]
    )
    assert result["patella_rotation_relative_femur_xyzw"] == pytest.approx(_quat_z(20.0))
    assert result["interpolation_steps"] == [1, 2]


def test_reference_pose_does_not_extrapolate():
    rows = [{
        "step": 1, "knee_flexion_deg": 10.0,
        "patella_position_relative_femur_com_mm": [0.0, 0.0, 0.0],
        "patella_rotation_relative_femur_xyzw": _quat_z(0.0),
    }]
    with pytest.raises(RuntimeError, match="does not cover"):
        _reference_pose_at(rows, 11.0)


def test_transfer_audit_requires_registered_plan():
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "registered_before_transfer_observation",
        "inputs": {"myosim_revision": "33c89c2bde282553dde3f526768eb3bdcfaa7649",
                   "myosim_archive_sha256": "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"},
    }
    _validate_plan(plan)
    plan["status"] = "edited_after_observation"
    with pytest.raises(RuntimeError, match="preregistered"):
        _validate_plan(plan)

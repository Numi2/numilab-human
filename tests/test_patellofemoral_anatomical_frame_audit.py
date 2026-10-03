import math

import numpy as np
import pytest

from numilab_human.patellofemoral_anatomical_frame_audit import (
    _anatomical_components,
    _myosim_pose_at,
    _registered_femoral_frame,
    _rotation_vector_xyzw,
)
from numilab_human.open_knee import EXPECTED_HASHES


def _manifest():
    source_axes = np.eye(3)
    target_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    return {
        "schema": "numi.human.open-knee-oks003-payload.v3",
        "status": "exact_source_payload_registered_to_live_left_knee_candidate",
        "source": {"files": {name: {"sha256": digest}
                                for name, digest in EXPECTED_HASHES.items()}},
        "registration": {
            "output_side": "left",
            "proper_rotation_source_to_femur_body": target_axes.tolist(),
            "uniform_scale": 1.0,
            "source_axes": {"Xf_axis": source_axes[:, 0].tolist(),
                            "Yf_axis": source_axes[:, 1].tolist(),
                            "Zf_axis": source_axes[:, 2].tolist()},
            "target_flexion_axis_femur_body": target_axes[:, 0].tolist(),
            "target_anterior_axis_femur_body": target_axes[:, 1].tolist(),
            "target_proximal_axis_femur_body": target_axes[:, 2].tolist(),
            "source_origin_mm": [0.0, 0.0, 0.0],
        },
    }


def test_registered_source_frame_preserves_handed_flexion_anterior_proximal_axes():
    frame = _registered_femoral_frame(_manifest(), np)
    assert np.linalg.det(frame["rotation"]) == pytest.approx(1.0)
    assert _anatomical_components([2.0, 3.0, 5.0], frame["target_axes"], np) == pytest.approx(
        [5.0, 2.0, 3.0]
    )


def test_registered_source_frame_rejects_inconsistent_axis_mapping():
    manifest = _manifest()
    manifest["registration"]["target_anterior_axis_femur_body"] = [-1.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="do not map"):
        _registered_femoral_frame(manifest, np)


def test_rotation_vector_retains_direction_and_shortest_signed_angle():
    half = math.radians(30.0) / 2.0
    vector = _rotation_vector_xyzw([0.0, 0.0, math.sin(half), math.cos(half)])
    assert vector == pytest.approx([0.0, 0.0, math.radians(30.0)])


def test_myo_pose_interpolation_refuses_to_bridge_a_source_invalid_interval():
    rows = [
        {"knee_flexion_deg": 0.0,
         "patella_inertial_com_relative_femur_inertial_com_m": [0.0, 0.0, 0.0],
         "patella_rotation_relative_femur_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"knee_flexion_deg": 0.5,
         "patella_inertial_com_relative_femur_inertial_com_m": [0.0, 0.0, 0.0],
         "patella_rotation_relative_femur_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"knee_flexion_deg": 2.0,
         "patella_inertial_com_relative_femur_inertial_com_m": [0.0, 0.0, 0.0],
         "patella_rotation_relative_femur_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ]
    with pytest.raises(RuntimeError, match="source-invalid path gap"):
        _myosim_pose_at(rows, 1.0)

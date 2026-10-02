from types import SimpleNamespace

import numpy as np

from numilab_human.upper_limb_pose_audit import _joint_equality_correction_by_unit
from numilab_human.lower_limb_pose_audit import _patellar_pose_status


class FakeMujoco:
    class mjtEq:
        mjEQ_JOINT = 1

    class mjtJoint:
        mjJNT_SLIDE = 2
        mjJNT_HINGE = 3

    class mjtObj:
        mjOBJ_JOINT = 0

    @staticmethod
    def mj_id2name(model, object_type, joint_id):
        assert object_type == FakeMujoco.mjtObj.mjOBJ_JOINT
        return model.joint_names[joint_id]


def test_joint_equality_projection_maxima_keep_slide_and_hinge_units_separate():
    model = SimpleNamespace(
        neq=2,
        eq_active0=[True, True],
        eq_type=[1, 1],
        eq_obj1id=[0, 1],
        eq_obj2id=[2, 3],
        jnt_type=[FakeMujoco.mjtJoint.mjJNT_SLIDE, FakeMujoco.mjtJoint.mjJNT_HINGE, 3, 2],
        jnt_qposadr=[0, 1, 2, 3],
        joint_names=["knee_slide", "knee_rotation", "knee_flexion", "hip_slide"],
    )
    before = np.zeros(4)
    after = np.array([0.004, -0.602, 0.9, 0.0])

    result = _joint_equality_correction_by_unit(model, before, after, FakeMujoco)

    assert result["m"] == {
        "maximum_absolute_correction": 0.004,
        "signed_correction": 0.004,
        "dependent_joint": "knee_slide",
        "driver_joint": "knee_flexion",
        "qpos_index": 0,
    }
    assert result["rad"] == {
        "maximum_absolute_correction": 0.602,
        "signed_correction": -0.602,
        "dependent_joint": "knee_rotation",
        "driver_joint": "hip_slide",
        "qpos_index": 1,
    }


def test_patellar_status_separates_raw_qpos0_from_projected_visual_poses():
    source_qpos0 = [
        {"side": "r", "passed": False},
        {"side": "l", "passed": False},
    ]
    projected = [{
        "patellar_anteriority": [
            {"side": "r", "passed": True, "minimum_signed_anterior_offset_m": 0.035},
            {"side": "l", "passed": True, "minimum_signed_anterior_offset_m": 0.034},
        ],
    }]

    result = _patellar_pose_status(source_qpos0, projected)

    assert result["literal_unprojected_source_qpos0"] == {
        "status": "posterior_or_intersecting_knee_anchor_plane",
        "runtime_acceptance_pose": False,
        "failed_sides": ["r", "l"],
    }
    assert result["source_equality_projected_pose_suite"] == {
        "status": "all_patella_vertices_anterior",
        "pose_count": 1,
        "evaluation_count": 2,
        "minimum_signed_anterior_offset_m": 0.034,
    }
    assert "Use a source-equality-projected neutral pose" in result["visual_guidance"]
    assert "not a native accepted state" in result["visual_guidance"]

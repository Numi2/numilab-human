"""A source-rest image must not silently stand in for projected anatomy."""

from types import SimpleNamespace

import numpy as np
import pytest

from numilab_human.myosim_visual import _patella_body_center_anteriority, _visual_qpos


def test_projected_neutral_is_default_and_raw_rest_is_explicit():
    source = SimpleNamespace(
        qpos0=np.array([0., 0.]), nq=2, njnt=2, neq=1,
        eq_active0=[True], eq_type=[1], eq_obj1id=[1], eq_obj2id=[0],
        jnt_qposadr=[0, 1], jnt_type=[2, 2],
        eq_data=np.array([[.052, 0., 0., 0., 0.]]),
    )
    mujoco = SimpleNamespace(
        mjtEq=SimpleNamespace(mjEQ_JOINT=1),
        mjtJoint=SimpleNamespace(mjJNT_SLIDE=2, mjJNT_HINGE=3),
        mjtObj=SimpleNamespace(mjOBJ_JOINT=3),
        mj_id2name=lambda _model, _kind, joint: ("driver", "dependent")[joint],
    )

    neutral, neutral_info = _visual_qpos(source, mujoco, False)
    raw, raw_info = _visual_qpos(source, mujoco, True)

    assert neutral.tolist() == pytest.approx([0., .052])
    assert neutral_info['pose_state'] == 'source_equality_projected_neutral'
    assert neutral_info['source_joint_equalities_projected'] == 1
    assert neutral_info['maximum_equality_coordinate_correction'] == pytest.approx(.052)
    assert raw.tolist() == [0., 0.]
    assert raw_info['pose_state'] == 'literal_unprojected_source_qpos0'
    assert source.qpos0.tolist() == [0., 0.]


def test_projected_visual_rejects_model_without_equality_projection():
    source = SimpleNamespace(qpos0=np.array([0.]), neq=0)
    mujoco = SimpleNamespace(mjtEq=SimpleNamespace(mjEQ_JOINT=1))
    with pytest.raises(RuntimeError, match='no active joint equalities'):
        _visual_qpos(source, mujoco, False)


def test_source_visual_rejects_posterior_patella_before_rendering():
    names = {'pelvis': 0, 'patella_r': 1, 'patella_l': 2,
             'knee_angle_r': 0, 'knee_angle_l': 1}
    mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_JOINT=2),
        mj_name2id=lambda _model, _kind, name: names.get(name, -1),
    )
    data = SimpleNamespace(
        xmat=np.eye(3)[None, :, :],
        xpos=np.array([[0., 0., 0.], [-.1, .11, .5], [.1, .11, .5]]),
        xanchor=np.array([[-.1, .15, .5], [.1, .15, .5]]),
    )
    measured = _patella_body_center_anteriority(object(), data, mujoco, False)
    assert [row['passed_display_gate'] for row in measured] == [True, True]
    assert measured[0]['source_body_center_anterior_offset_m'] == pytest.approx(.04)

    data.xpos[2, 1] = .16
    with pytest.raises(RuntimeError, match='patella_l behind/too near'):
        _patella_body_center_anteriority(object(), data, mujoco, False)
    raw = _patella_body_center_anteriority(object(), data, mujoco, True)
    assert raw[1]['passed_display_gate'] is False

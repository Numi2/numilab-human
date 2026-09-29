"""A source-rest image must not silently stand in for projected anatomy."""

from types import SimpleNamespace

import numpy as np
import pytest

from numilab_human.myosim_visual import _visual_qpos


def test_projected_neutral_is_default_and_raw_rest_is_explicit():
    source = SimpleNamespace(
        qpos0=np.array([0., 0.]), neq=1,
        eq_active0=[True], eq_type=[1], eq_obj1id=[1], eq_obj2id=[0],
        jnt_qposadr=[0, 1], eq_data=np.array([[.052, 0., 0., 0., 0.]]),
    )
    mujoco = SimpleNamespace(mjtEq=SimpleNamespace(mjEQ_JOINT=1))

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

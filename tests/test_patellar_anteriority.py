"""An anterior centroid cannot hide a posterior part of the patella."""

import numpy as np
import pytest

from numilab_human.lower_limb_pose_audit import _patellar_anteriority
from numilab_human.upper_limb_pose_audit import PoseAuditError, _finish_pose_audit


def test_full_patellar_support_and_source_axis_rotation():
    vertices = np.array([[0., -.03, 0.], [0., -.02, .01], [0., .001, 0.]])
    origin = np.zeros(3)
    axis = np.array([0., -1., 0.])
    result = _patellar_anteriority(vertices, origin, axis, side='r', member_id='FJ3381')
    assert result['centroid_signed_anterior_offset_m'] > 0
    assert result['minimum_signed_anterior_offset_m'] == pytest.approx(-.001)
    assert result['vertices_posterior_or_on_knee_anchor_plane'] == 1
    assert result['passed'] is False

    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = _patellar_anteriority(vertices @ rotation.T, origin @ rotation.T,
                                    axis @ rotation.T, side='r', member_id='FJ3381')
    assert rotated['minimum_signed_anterior_offset_m'] == pytest.approx(
        result['minimum_signed_anterior_offset_m'])
    assert rotated['passed'] is False


def test_all_points_must_be_strictly_anterior_and_inputs_finite():
    vertices = np.array([[0., -.03, 0.], [0., -.01, .01]])
    result = _patellar_anteriority(vertices, [0., 0., 0.], [0., -1., 0.],
                                   side='l', member_id='FJ3275')
    assert result['passed'] is True
    assert result['minimum_signed_anterior_offset_m'] == pytest.approx(.01)
    with pytest.raises(RuntimeError, match='invalid geometry'):
        _patellar_anteriority([[float('nan'), 0., 0.]], [0., 0., 0.], [0., -1., 0.],
                               side='l', member_id='FJ3275')


def test_pose_admission_refuses_posterior_point_even_with_positive_centroid():
    posterior = _patellar_anteriority(
        [[0., -.03, 0.], [0., -.02, 0.], [0., .001, 0.]],
        [0., 0., 0.], [0., -1., 0.], side='r', member_id='FJ3381')
    receipt = {
        'status': 'passed_source_owned_bilateral_lower_limb_multi_pose_interface_patches',
        'default_frame_maximum_centroid_residual_m': 0.,
        'default_frame_maximum_allowed_residual_m': 1e-9,
        'source_geometry_checks': [],
        'poses': [{'name': 'neutral', 'patellar_anteriority': [posterior],
                   'continuity': [], 'bilateral_gap_parity': []}],
        'inputs': {'registration': {'sha256': 'registered'},
                   'runtime_reference': {'rigid': {'sha256': 'rigid'}}},
    }
    with pytest.raises(PoseAuditError) as rejected:
        _finish_pose_audit(receipt, 'lower-limb')
    assert 'patella_r' in str(rejected.value)
    assert rejected.value.result['failures'][0].endswith('posterior_or_plane_vertices=1')

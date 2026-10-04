import numpy as np
from scipy.spatial.transform import Rotation

from numilab_human.skin_dual_quaternion_audit import blend_dual_quaternion_points


def test_dual_quaternion_single_influence_matches_rigid_transform():
    points = np.array([[0.2, -0.4, 1.3], [-0.1, 0.7, 0.2]])
    rotation = Rotation.from_rotvec([0.31, -0.22, 0.47])
    translation = np.array([0.4, -0.9, 0.15])
    actual = blend_dual_quaternion_points(
        points, np.ones((len(points), 1)), rotation.as_quat()[None, :],
        translation[None, :],
    )
    np.testing.assert_allclose(actual, rotation.apply(points)+translation, atol=1e-12)


def test_dual_quaternion_identity_preserves_points_with_many_influences():
    points = np.array([[0.0, 0.2, -0.8], [1.1, -0.3, 0.5]])
    weights = np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    translations = np.zeros((3, 3))
    actual = blend_dual_quaternion_points(points, weights, rotations, translations)
    np.testing.assert_allclose(actual, points, atol=1e-12)


def test_dual_quaternion_hemisphere_alignment_handles_antipodal_inputs():
    points = np.array([[0.3, 0.1, -0.7]])
    rotation = Rotation.from_rotvec([0.8, -0.2, 0.4])
    quaternion = rotation.as_quat()
    rotations = np.array([quaternion, -quaternion])
    translations = np.array([[0.2, -0.1, 0.6], [0.2, -0.1, 0.6]])
    actual = blend_dual_quaternion_points(
        points, np.array([[0.5, 0.5]]), rotations, translations,
    )
    np.testing.assert_allclose(actual, rotation.apply(points)+translations[0], atol=1e-12)

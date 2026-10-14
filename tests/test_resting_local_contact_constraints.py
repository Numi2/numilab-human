import unittest

import numpy as np

from numilab_human.resting_local_contact_constraints import finite_projected_overlap


class FiniteFootprintTests(unittest.TestCase):
    def setUp(self):
        self.target = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0]], dtype=np.float32)

    def test_outside_corner_does_not_create_remote_plane_constraint(self):
        source = np.array([[-.01, -.01, -.005], [.02, -.01, .010],
                           [-.01, .02, .010]], dtype=np.float32)
        weights, target_weights, depths = finite_projected_overlap(source, self.target)
        self.assertGreater(len(depths), 0)
        self.assertLess(source[0, 2], 0)
        self.assertGreater(float(depths.min()), .0049)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue(np.all(target_weights >= -1e-12))
        projected = weights @ source
        projected[:, 2] = 0
        np.testing.assert_allclose(projected, target_weights @ self.target, atol=1e-12)

    def test_actual_overlap_retains_both_sides_of_crossing(self):
        source = self.target.copy()
        source[:, 2] = [-.0001, .0001, .0001]
        weights, _, depths = finite_projected_overlap(source, self.target)
        self.assertLess(depths.min(), 0)
        self.assertGreater(depths.max(), 0)
        np.testing.assert_allclose(depths, (weights @ source)[:, 2])

    def test_separated_footprints_are_empty(self):
        weights, _, depths = finite_projected_overlap(self.target + [.01, .01, 0], self.target)
        self.assertEqual(weights.shape, (0, 3))
        self.assertEqual(depths.size, 0)

    def test_equal_projection_preserves_distinct_normal_depths(self):
        source = np.array([[.0001, .0001, -.001], [.0001, .0001, .001],
                           [.0002, .0002, .0005]])
        _, _, depths = finite_projected_overlap(source, self.target)
        self.assertAlmostEqual(float(depths.min()), -.001)
        self.assertAlmostEqual(float(depths.max()), .001)

    def test_rotation_translation_and_scale_preserve_depth_units(self):
        source = self.target.astype(float)
        source[:, 2] = [-.0001, .0001, .0002]
        weights, _, depths = finite_projected_overlap(source, self.target)
        rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        shift = np.array([.12, -.77, .05])
        rotated_source = (source * 3) @ rotation.T + shift
        rotated_target = (self.target.astype(float) * 3) @ rotation.T + shift
        next_weights, _, next_depths = finite_projected_overlap(rotated_source, rotated_target)
        np.testing.assert_allclose(next_weights, weights, atol=1e-10)
        np.testing.assert_allclose(next_depths, depths * 3, atol=1e-14)

    def test_invalid_and_degenerate_target_rejected(self):
        for target in (np.zeros((3, 3)), self.target * np.nan):
            with self.assertRaises(ValueError):
                finite_projected_overlap(self.target, target)

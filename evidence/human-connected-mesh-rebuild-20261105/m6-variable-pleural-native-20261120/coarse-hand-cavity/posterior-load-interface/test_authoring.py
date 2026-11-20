"""Regression tests for geometric witnesses and the affine height constraints."""
import unittest

import numpy as np

from audit import closed_surface, contacts
from extension import clearance_cut, determinant, fit_heights, signed_height_gradients


class PosteriorAuthoringTests(unittest.TestCase):
    def test_broadphase_never_drops_long_or_short_triangle(self):
        triangles = np.array([
            [[0., 0., 1.], [16., 0., 1.], [0., 16., 1.]],
            [[.1, .1, 2.], [.2, .1, 2.], [.1, .2, 2.]],
            [[.1, .1, 0.], [.1, .1, 1.], [.2, .1, 0.]],
        ])
        result = contacts(np.array([[.125, .125], [5., 6.], [11., 11.]]), triangles)
        self.assertEqual(result, [[(0, 1.), (1, 2.)], [(0, 1.)], []])

    def test_welded_closed_source_and_open_duplicate(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        corners = vertices[faces].reshape(-1, 3)
        split_ids = np.arange(12).reshape(-1, 3)
        self.assertTrue(closed_surface(corners, split_ids))
        self.assertFalse(closed_surface(corners, split_ids[:-1]))
        self.assertFalse(closed_surface(corners, np.r_[split_ids, split_ids[:1]]))

    def test_height_derivative_exact_for_large_changes(self):
        rng = np.random.default_rng(410)
        x = rng.normal(size=(28, 3))
        tet = np.array([rng.choice(len(x), 4, replace=False) for _ in range(20)])
        h = rng.normal(size=len(x))
        z = x.copy()
        z[:, 2] += h
        delta = np.einsum("ij,ij->i", signed_height_gradients(x, tet), h[tet])
        np.testing.assert_allclose(determinant(z, tet) - determinant(x, tet), delta, atol=2e-14)

    def test_cut_limits_crossing_at_interior_projected_point(self):
        shell = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        tissue = np.array([[.3, .3, .1], [.6, .3, .1], [.3, .6, .1]])
        coefficients, caps = clearance_cut(shell, tissue)
        self.assertTrue(np.all(coefficients @ np.full(3, .09) < caps))
        self.assertTrue(np.all(coefficients @ np.full(3, .11) > caps))
        np.testing.assert_allclose(coefficients.sum(axis=1), 1.)

    def test_fit_preserves_positive_orientation_in_both_frames(self):
        x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rest = x.copy()
        rest[3, 2] = .8
        moves = np.zeros_like(x)
        moves[0, 2] = 1.
        cells = np.array([[0, 1, 2, 3]])
        fit = fit_heights(x, rest, cells, np.array([True]), moves)
        self.assertAlmostEqual(fit[0, 2], .44, places=7)
        np.testing.assert_allclose(determinant(rest + fit, cells), [.36], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

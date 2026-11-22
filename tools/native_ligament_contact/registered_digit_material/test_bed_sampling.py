import unittest

import numpy as np
from fit_native_bed_barrier_load import force

from compare_bed_sampling import integrate


class QuadratureTests(unittest.TestCase):
    def test_uniform_and_empty(self):
        x = np.array([[0, 0, 0.8], [1, 0, 0.8], [0, 1, 0.8]])
        t = np.array([[0, 1, 2]])
        area = np.array([0.5])
        w = np.full(3, 4.0)
        nodal, first = integrate(x, t, area, w, 0.2, 1.0, 4)
        expected = 2 * force(0.8, 1.0, 0.2, 1.0)
        np.testing.assert_allclose(nodal, np.full(3, expected / 3))
        np.testing.assert_allclose(first, expected * x.mean(axis=0))
        x[:, 2] = 2.0
        np.testing.assert_array_equal(
            integrate(x, t, area, w, 0.2, 1.0, 4)[0], np.zeros(3)
        )

    def test_subdivided_material_face_preserves_generalized_reactions(self):
        x = np.array([[0.0, 0.0, 0.65], [1.0, 0.0, 1.24], [0.0, 1.0, 0.89]])
        w = np.array([2.0, 5.0, 3.0])
        embed = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0.5, 0.5, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
            ]
        )
        coarse = np.array([[0, 1, 2]])
        fine = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
        parent, moment = integrate(x, coarse, np.array([0.5]), w, 0.2, 1.0, 12)
        child, new_moment = integrate(
            embed @ x, fine, np.full(4, 0.125), embed @ w, 0.2, 1.0, 12
        )
        np.testing.assert_allclose(embed.T @ child, parent, rtol=5e-13, atol=1e-12)
        np.testing.assert_allclose(new_moment, moment, rtol=5e-13, atol=1e-12)

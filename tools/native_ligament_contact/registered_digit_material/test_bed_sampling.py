import unittest

import numpy as np
from compare_bed_sampling import integrate
from fit_native_bed_barrier_load import force


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

    def test_partial_clipping_and_tangent_symmetry(self):
        face = np.array([[0, 1, 2]])
        area = np.array([0.5])
        rho = np.array([0.8, 1.4, 1.1])
        a = np.array([0.2, -0.7, 0.4])
        b = np.array([-0.3, 0.2, 0.8])
        x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for gaps in (
            [0.72, 1.2, 1.1],
            [0.72, 0.91, 1.0],
            [0.72, 0.81, 0.98],
            [1.1, 1.2, 1.0],
        ):
            x[:, 2] = gaps
            _fa, _, ka = integrate(x, face, area, rho, 0.2, 1.0, 8, normal_direction=a)
            _, _, kb = integrate(x, face, area, rho, 0.2, 1.0, 8, normal_direction=b)
            self.assertAlmostEqual(float(a @ kb), float(b @ ka), places=9)
            self.assertGreaterEqual(float(a @ ka), 0.0)
            h = 1e-6
            up = x.copy()
            up[:, 2] += h * a
            down = x.copy()
            down[:, 2] -= h * a
            difference = (
                integrate(down, face, area, rho, 0.2, 1.0, 8)[0]
                - integrate(up, face, area, rho, 0.2, 1.0, 8)[0]
            ) / (2 * h)
            np.testing.assert_allclose(ka, difference, rtol=1e-7, atol=1e-7)

    def test_force_continuation_has_zero_curvature_below_floor(self):
        x = np.array([[0.0, 0.0, 2e-5], [1.0, 0.0, 3e-5], [0.0, 1.0, 4e-5]])
        f, _, k = integrate(
            x,
            np.array([[0, 1, 2]]),
            np.array([0.5]),
            np.ones(3),
            0.2,
            1.0,
            4,
            normal_direction=np.ones(3),
        )
        self.assertTrue(np.all(f > 0))
        np.testing.assert_array_equal(k, np.zeros(3))

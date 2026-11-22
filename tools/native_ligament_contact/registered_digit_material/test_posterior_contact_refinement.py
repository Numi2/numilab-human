import unittest

import numpy as np

from compare_posterior_support_load import solve_load
from posterior_stretch import normal_stiffness, stretch_envelope
from refine_posterior_contact import Stars, boundary, keys, mechanical, pullback


class RegisteredPosteriorContact(unittest.TestCase):
    def test_full_edge_star_and_adjoint(self):
        cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        x = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0.5, 0, 0]]
        )
        tet = Stars(cells)
        surface = Stars(boundary(cells))
        for s in [tet, surface]:
            s.split(0, 1, 5, s.users(0, 1))
        children, p = tet.arrays()
        self.assertEqual(keys(boundary(children)), keys(surface.arrays()[0]))
        v = np.linalg.det(x[children[:, 1:]] - x[children[:, :1]]) / 6
        np.testing.assert_allclose(np.bincount(p, weights=v), [1 / 6, 1 / 6])
        self.assertTrue(np.all(v > 0))
        load = np.arange(18.0).reshape(6, 3)
        self.assertAlmostEqual(
            np.sum(load * x), np.sum(pullback(load, [(0, 1)]) * x[:5])
        )
        self.assertRaisesRegex(
            ValueError, "orientation", boundary, cells[:, [0, 1, 2, 3]][[0, 0]]
        )

    def test_rigid_shift_and_support_tangent(self):
        x = np.array([[0.5, -1.0, 1.0e-5], [-0.2, 0.8, 1.01e-5], [0.2, 0.4, 0.02]])
        m = np.array([0.01, 0.005, 0.007])
        h, e, n, k = solve_load(x, m, 1.5625e-5, 1.0e-5, 100.0)
        new = x.copy()
        new[:, 2] += 0.00023
        h2, _e2, n2, k2 = solve_load(new, m, 1.5625e-5, 1.0e-5, 100.0)
        self.assertAlmostEqual(h - h2, 0.00023, places=13)
        self.assertAlmostEqual(n.sum(), 100.0, places=6)
        self.assertTrue(np.all(e >= 0) and np.min(x[:, 2] + h) > 0 and n[-1] == 0)
        np.testing.assert_allclose(n, n2, rtol=2e-9)
        np.testing.assert_allclose(k, k2, rtol=2e-9)
        self.assertRaises(
            ValueError,
            solve_load,
            x,
            np.array([0.01, np.inf, 0.007]),
            1.0e-5,
            1.0e-5,
            10.0,
        )

    def test_material_normal_tangent_matches_direct_forces(self):
        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [1, 0, 0],
                [0.12, 1.1, 0],
                [0.15, 0.1, 0.8],
                [0.2, -0.1, -1.3],
            ]
        )
        t = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        roles = np.array([15, 16])
        props = np.array([[1250.0, 15000.0], [910.0, 2800.0]])
        y = x @ np.array(
            [[1.03, 0.11, 0.04], [-0.03, 0.94, 0.12], [-0.02, -0.05, 1.12]]
        )
        rows = np.array([3, 0, 2])
        free = np.array([1, 0])
        v = np.array([0.3, -0.22])
        eps = 1.0e-7
        tangent = normal_stiffness(x, y, roles, t, props, free, rows)

        def force(step):
            q = y.copy()
            q[free, 2] += v * step
            return mechanical((x, q), t, roles, props)[1][rows, 2]

        np.testing.assert_allclose(
            tangent @ v,
            (force(-eps) - force(eps)) / (2 * eps),
            rtol=1.0e-7,
            atol=6.0e-6,
        )

    def test_stretch_derivative_and_roundoff(self):
        x = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1e-5], [2, 0, 1.0]])
        y = x.copy()
        y[1, 0] = 1.02
        y[3, 2] = 6.8e-6
        fun, rows = stretch_envelope(
            x, y, np.array([[0, 1, 2, 3], [4, 1, 0, 2]]), [3], 2.0
        )
        np.testing.assert_equal(rows, [0])
        v = np.array([0.12])
        val, J = fun(v)
        eps = 1.0e-5
        np.testing.assert_allclose(
            (fun(v + eps)[0] - fun(v - eps)[0]) / (2 * eps), J[:, 0], atol=2.0e-10
        )
        self.assertTrue(np.all(val > 0))
        self.assertLess(fun(np.array([-0.5]))[0].min(), 0.0)


if __name__ == "__main__":
    unittest.main()

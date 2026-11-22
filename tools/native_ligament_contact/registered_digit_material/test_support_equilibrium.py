"""Material-force checks for the registered loaded-skin initialization."""

import unittest

import numpy as np

from fit_support_strain import Patch


class Data(dict):
    @property
    def files(self):
        return self.keys()


class SupportEquilibriumTests(unittest.TestCase):
    def setUp(self):
        x = np.array(
            [
                [0.0, 0.0, 0.02],
                [0.01, 0.0, 0.02],
                [0.0, 0.018, 0.02],
                [0.0, 0.0, 0.0201],
            ]
        )
        y = x @ np.array(
            [[1.06, 0.02, 0.03], [-0.01, 0.91, 0.01], [0.01, 0.0, 1.02]]
        ) + [0.1, 0.3, 0.0]
        self.p = Patch(
            Data(
                x=x,
                y=y,
                tet=np.array([[0, 1, 2, 3]]),
                roles=np.array([15]),
                properties=np.array([[13255.0, 658333.0], [550.0, 5300.0]]),
                mass=np.ones(4) * 1e-5,
                free=np.arange(4),
                probes=np.array([0]),
                base_elastic=np.zeros_like(x),
                dt=np.array(0.001),
                width=np.array(0.00001),
            )
        )

    def test_force_and_moment_cancellation(self):
        p = self.p
        force = p.internal(p.y)
        np.testing.assert_allclose(force.sum(0), 0.0, atol=1e-11)
        np.testing.assert_allclose(np.cross(p.y, force).sum(0), 0.0, atol=1e-11)

    def test_current_tangent_random_directions(self):
        p = self.p
        d = np.random.default_rng(17).normal(size=(4, 3))
        h = 1e-9
        neg_df = (p.internal(p.y - h * d) - p.internal(p.y + h * d)) / (2 * h)
        np.testing.assert_allclose(
            p.tangent(p.y, p.free) @ d.ravel(), neg_df.ravel(), rtol=1e-6, atol=1e-4
        )

    def test_thin_direction_and_rigid_translation(self):
        p = self.p
        G = p.strain_metric(p.y, p.free)
        np.testing.assert_allclose(G.sum(0), 0.0, atol=1e-3)
        self.assertGreater(G[3, 3], 1000 * G[1, 1])
        np.testing.assert_allclose(G, G.T, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

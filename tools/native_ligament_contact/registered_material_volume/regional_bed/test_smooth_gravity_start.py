import unittest
import numpy as np
from scipy import sparse
from smooth_gravity_start import affine_contact_step, near_shape_limits, shape_limits


class AffineGravityTests(unittest.TestCase):
    def test_compression_cannot_draw_reaction_from_search_bound(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coupling = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        K = sparse.csr_matrix(np.outer(coupling, coupling) * 1000)
        gradient = np.array([0.0, 0.0, 0.0, 0.0, -2.0])
        upper = np.array([0.0, 0.0, 0.0, np.inf])
        for radius, expected in ((0.01, 2.0), (0.001, 1.0)):
            with self.subTest(radius=radius):
                trial, reaction, log = affine_contact_step(
                    K, gradient, xyz, upper, radius
                )
                self.assertLessEqual(np.max(trial[:3]), 1e-10)
                self.assertAlmostEqual(reaction.sum(), expected, places=6)
                self.assertAlmostEqual(trial[4], expected / 1000, places=6)
                self.assertEqual(log[0]["trust_reaction_active"], radius < 0.002)

    def test_material_limit_derivative_and_orientation(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3e-5]]
        )
        edges, ratios = near_shape_limits(
            xyz, np.array([[0, 1, 2, 3]]), np.ones(4, bool), 0.06
        )
        self.assertGreater(len(edges), 0)
        n = np.array([0.0, 0.0, 1.0])
        scale = np.array([0.11, 0.1, 0.2])
        q = np.array([0.02, 0.03, -0.02, 0.1, 0.25])
        val, jac = shape_limits(q, edges, ratios, n, scale)
        for i in range(5):
            eps = np.eye(5)[i] * 1e-6
            a = shape_limits(q + eps, edges, ratios, n, scale)[0]
            b = shape_limits(q - eps, edges, ratios, n, scale)[0]
            np.testing.assert_allclose((a - b) / 2e-6, jac[:, i], atol=2e-9)
        G = np.eye(3) - np.outer(n, scale * q[1:4])
        det = np.linalg.det(G)
        np.testing.assert_allclose(
            val, ratios * det - np.linalg.norm(edges @ G.T, axis=1) ** 3, atol=2e-9
        )

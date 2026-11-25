import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from normal_gravity_step import certify_motion, constitutive_gradient, contact_step
from solve_connected_normal_transfer import normal_tangent, transfer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support_observation"))
from assess_supported_release import mixed_normal_bound


class ConnectedTissue(unittest.TestCase):
    def test_motion_rejects_low_shape_before_a_cell_inverts(self):
        reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 4e-5]])
        direction = np.zeros_like(reference)
        direction[3, 2] = -1.6e-5
        step, cleared, opening, witnesses = certify_motion(
            reference, reference, np.array([[0, 1, 2, 3]]), direction, 1.
        )
        self.assertEqual(opening, [0] * 5)
        self.assertEqual(cleared, [0] * 5)
        self.assertEqual(step, 0.5)
        self.assertEqual(witnesses[0]["gate"], 2)

    def test_skew_tet_stored_energy_force_and_tangent(self):
        ref = np.array([[1, 0.7, 0.2], [1.3, 0.4, 0.8], [1.2, 1.8, 0.5], [1, 1.2, 1.3]])
        cells = np.array([[0, 1, 2, 3]])
        xyz = ref @ np.array([[0.96, 0.02, 0.08], [-0.18, 1.08, 0], [0.09, 0.06, 1.01]])
        n = np.array([2.0, 3.0, -4.0]) / np.sqrt(29)
        mu, bulk = 2.0, 21.0
        K = normal_tangent(ref, xyz, cells, n, np.array([mu]), np.array([bulk]))[0]
        material = {15: (mu, bulk), 16: (mu, bulk)}
        v = np.array([-0.1, 0.3, 0.25, 0.012])

        def evaluate(e):
            return constitutive_gradient(
                ref, xyz - e * v[:, None] * n, cells, n, np.array([15]), material
            )

        p, pe = evaluate(1e-5)
        m, me = evaluate(-1e-5)
        g, _ = evaluate(0)
        np.testing.assert_allclose((pe - me) / 2e-5, g @ v, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose((p - m) / 2e-5, K @ v, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(K @ np.ones(4), 0.0, atol=1e-12)

    def test_series_path_with_free_interior(self):
        # The sole physical path consists of three equal springs in series.
        K = sparse.csr_matrix([[4.0, -4.0, 0.0], [-4.0, 8.0, -4.0], [0.0, -4.0, 4.0]])
        u, result = transfer(K, [2], np.array([[4.0]]), [0])
        np.testing.assert_allclose(u, [0.0, 1 / 3, 2 / 3])
        self.assertAlmostEqual(result["bed_stiffness_N_per_m"], 4 / 3)
        self.assertAlmostEqual(result["work_closure_error_N_per_m"], 0.0)

    def test_contact_set_releases_and_recruits(self):
        # A body load on a free rigid coordinate pushes on the near surface only.
        H = sparse.csr_matrix([[2.0, 0.0, -2.0], [0.0, 2.0, -2.0], [-2.0, -2.0, 4.0]])
        g = np.array([0.0, 0.0, -1.0])
        q, reaction, history = contact_step(
            H,
            g,
            np.array([0.0, 2.0, np.inf]),
            np.array([False, True, False]),
            lambda A, b: np.linalg.solve(A.toarray(), b),
        )
        np.testing.assert_allclose(q, [0.0, 0.5, 0.5])
        np.testing.assert_allclose(reaction, [1.0, 0.0, 0.0])
        self.assertGreater(len(history), 1)

    def test_bilateral_attachment_can_turn_when_its_old_force_is_transverse(self):
        flags = np.array([3, 5, 5])
        area = np.array([1.0, 1.0, 1.0])
        direction = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 0, 1.0]])
        r = mixed_normal_bound(
            flags, area, direction, np.ones(3), np.ones(3), np.array([0.0, 0, 1.0]), 2.0
        )
        self.assertAlmostEqual(
            r["peak_attachment_extension_or_contact_closure_mm"], 1000.0
        )


if __name__ == "__main__":
    unittest.main()

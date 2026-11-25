import unittest

import numpy as np

from registration_clearance import clearance_constraints


class RegistrationClearanceTest(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])

    def test_sliding_and_approaching_on_both_sides(self):
        q = self.p + [0.1, 0.1, .001]
        u = np.zeros_like(self.p)
        v = np.zeros_like(q)
        u[:, 1, 2], v[:, 0, 2] = .003, -.002
        co, gap = clearance_constraints(self.p, q, u, v)
        self.assertTrue(np.all(gap > 0))
        self.assertTrue(np.any(co[..., 0] + co[..., 1] > gap))
        left = np.array([.7, .1, 1.])
        right = np.array([.1, 1., .4])
        self.assertTrue(np.all(co[..., 0] * left[:, None] + co[..., 1] * right <= gap))
        for t in np.linspace(0, 1, 11):
            self.assertLess((self.p + t * left[:, None] * u)[:, :, 2].max(),
                            (q + t * right[:, None] * v)[:, :, 2].min())

    def test_coplanar_triangles_require_in_plane_separation(self):
        q = self.p + [2, 0, 0]
        u = np.broadcast_to([3., 0, 0], self.p.shape)
        c, gap = clearance_constraints(self.p, q, u, -u)
        self.assertTrue(np.any(c.sum(axis=-1) > gap))
        self.assertTrue(np.all(.1 * c.sum(axis=-1) < gap))

    def test_reject_preexisting_crossings_shared_edges_and_invalid_input(self):
        for q in (self.p, self.p + [1, 0, 0], self.p * np.nan):
            with self.assertRaises(ValueError):
                clearance_constraints(self.p, q, self.p * 0, self.p * 0)

    def test_small_scale_and_empty_domains(self):
        p = self.p * 1e-9
        c, gap = clearance_constraints(p, p + [0, 0, 1e-12], p * 0, p * 0)
        self.assertTrue((gap > 0).all())
        self.assertFalse(c.any())
        c, gap = clearance_constraints(*[p[:0]] * 4)
        self.assertEqual(gap.shape, (0, 3, 3))
        self.assertEqual(c.shape, (0, 3, 3, 2))


if __name__ == "__main__":
    unittest.main()

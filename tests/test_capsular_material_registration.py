import unittest
import numpy as np

from numilab_human.capsular_material_registration import closest_material_attachment


class CapsularMaterialTest(unittest.TestCase):
    def setUp(self):
        self.tri = np.array([[[-2., -2., 0.], [3., -2., 0.], [-2., 3., 0.]]])

    def test_body_metric_and_atlas_footprint(self):
        local = self.tri * [2., 1., 1.]
        target = np.array([1.2, .8, .2])
        center = np.zeros(3)
        radius = .5
        err, f, w = closest_material_attachment(self.tri, local, center, radius, target)
        x, y, _ = w @ self.tri[f]
        self.assertAlmostEqual(np.hypot(x, y), radius, places=12)
        self.assertLess(abs(2*(2*x-target[0])*(-y) + (y-target[1])*x), 1e-12)
        self.assertTrue(np.all(w > 0))
        self.assertAlmostEqual(err, np.linalg.norm(w @ local[f] - target))

    def test_face_selection_and_edge(self):
        raw = np.concatenate([self.tri, self.tri + [0., 0., 2.]])
        local = raw.copy()
        distance, face, weights = closest_material_attachment(
            raw, local, np.array([.5,.5,2.]), .4, np.array([2.,2.,2.]))
        self.assertEqual(face, 1)
        self.assertAlmostEqual(weights.min(), 1e-5)
        self.assertGreater(distance, 0.)
        self.assertLess(np.linalg.norm(weights @ raw[face] - [.5,.5,2.]), .4)

    def test_impossible_and_degenerate(self):
        with self.assertRaisesRegex(ValueError, "No registered material face"):
            closest_material_attachment(self.tri, self.tri, [0,0,2], .1, [0,0,0])
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            closest_material_attachment(self.tri, np.zeros_like(self.tri), [0,0,0], .1, [0,0,0])

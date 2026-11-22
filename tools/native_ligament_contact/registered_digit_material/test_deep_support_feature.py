import unittest

import numpy as np

from measure_dense_support_paths import nearest_feature


class FeatureTests(unittest.TestCase):
    def test_interior_beats_nearer_vertex_and_respects_bound(self):
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [1, 1, 0.3]], dtype=float)
        faces = np.array([[3, 3, 3], [0, 1, 2]])
        point = np.array([1, 1, 0.1])
        distance, face, where = nearest_feature(points, faces, point, np.inf)
        self.assertEqual(face, 1)
        self.assertAlmostEqual(distance, 0.1)
        np.testing.assert_allclose(where, [1, 1, 0])
        self.assertIsNone(nearest_feature(points, faces, point, 0.09))

    def test_degenerate_segment(self):
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        tri = np.array([[0, 0, 1]])
        d, _, where = nearest_feature(points, tri, np.array([0.5, 1.0, 0.0]), 1.1)
        self.assertEqual(d, 1.0)
        np.testing.assert_allclose(where, [0.5, 0.0, 0.0])

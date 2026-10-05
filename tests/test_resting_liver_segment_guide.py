import unittest

import numpy as np

from numilab_human.resting_liver_segment_guide import point_triangle_distances


class PointToTriangleDistanceTests(unittest.TestCase):
    def test_projection_inside_triangle_uses_perpendicular_distance(self):
        points = np.asarray([[0.25, 0.25, 1.0]])
        triangle = np.asarray([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        self.assertAlmostEqual(float(point_triangle_distances(points, triangle)[0, 0]), 1.0)

    def test_projection_outside_triangle_uses_nearest_edge(self):
        points = np.asarray([[1.0, 1.0, 2.0]])
        triangle = np.asarray([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        self.assertAlmostEqual(float(point_triangle_distances(points, triangle)[0, 0]), np.sqrt(4.5))

    def test_degenerate_source_triangle_fails_closed(self):
        points = np.asarray([[0.25, 0.25, 1.0]])
        triangle = np.asarray([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
        with self.assertRaisesRegex(ValueError, "degenerate candidate triangle"):
            point_triangle_distances(points, triangle)


if __name__ == "__main__":
    unittest.main()

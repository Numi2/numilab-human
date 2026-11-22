"""Check the continuous support-height bound against two analytically solvable bodies."""

import unittest
import numpy as np
from bound_rigid_bed_placement import solve


class PlacementBound(unittest.TestCase):
    def test_two_nodes_on_a_vertical_line(self):
        slope = 0.5
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.04]])
        expected = 0.04 / np.sqrt(1 + 2 * slope**2)
        for shift in (np.zeros(3), np.array([-2.0, 3.0, 11.0])):
            fit = solve(points + shift, [np.array([1])], slope, 1e-5)
            self.assertLessEqual(fit["lower_bound_m"], expected + 1e-14)
            self.assertGreaterEqual(fit["upper_bound_m"], expected - 1e-14)
            self.assertLessEqual(fit["upper_bound_m"] - fit["lower_bound_m"], 1e-5)

    def test_coplanar_support(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 0.5, 0.5]])
        fit = solve(points, [np.array([0]), np.array([1])], 0.5, 1e-5)
        self.assertEqual(fit["upper_bound_m"], 0.0)
        self.assertEqual(fit["lower_bound_m"], 0.0)


if __name__ == "__main__":
    unittest.main()

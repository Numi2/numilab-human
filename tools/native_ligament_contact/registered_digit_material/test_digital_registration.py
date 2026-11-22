"""Numerical regression checks for the two geometric constructions used in the digital repair."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from digital_contact_constraints import overlap
from extensor_transfer import transfer


class DigitalRegistrationTests(unittest.TestCase):
    def test_overlap_hull_reconstructs_identical_points(self):
        a = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        b = np.array([[1.0, -1.0], [1.0, 3.0], [-1.0, 1.0]])
        for left, right in [
            (a, b),
            (a[::-1], b),
            (a, b[::-1]),
            (a * 1e-4 + 0.125, b * 1e-4 + 0.125),
        ]:
            poly = overlap(left, right)
            self.assertGreaterEqual(len(poly), 3)
            for wa, wb in poly:
                np.testing.assert_allclose(wa @ left, wb @ right, atol=1e-14)
                self.assertGreaterEqual(min(min(wa), min(wb)), -1e-11)
        self.assertEqual(overlap(a, b + 10), [])
        self.assertEqual(overlap(a, a[[0, 0, 2]]), [])

    def test_edge_crossing_without_contained_vertex(self):
        a = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
        b = np.array([[-2.0, 1.0], [2.0, 1.0], [0.0, -2.0]])
        points = {tuple(np.round(u @ a, 8)) for u, v in overlap(a, b)}
        self.assertEqual(len(points), 6)

    def test_rigid_single_owner_and_normalized_blend(self):
        x = np.array([[1.0, 2.0, 3.0], [0.1, -0.3, 0.5]])
        rotations = Rotation.from_euler("z", [[0], [90]], degrees=True)
        shifts = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]])
        scale = [1.0, 1.25]
        w = np.array([[1.0, 0.0], [0.25, 0.75]])
        y = transfer(x, w, rotations, shifts, scale)
        np.testing.assert_allclose(y[0], x[0], atol=2e-15)
        expected = 0.25 * x[1] + 0.75 * (np.array([0.3, 0.1, 0.5]) * 1.25 + shifts[1])
        np.testing.assert_allclose(y[1], expected, atol=2e-15)


if __name__ == "__main__":
    unittest.main()

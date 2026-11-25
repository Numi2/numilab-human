import unittest

import numpy as np

from measure_registered_overlap_mass import overlap_mass


class RegisteredOverlap(unittest.TestCase):
    def test_corner_cut_with_density_from_its_parent_cell(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # A translated unit tetrahedron cuts off one corner of the material.
        cortex = x + [0.5, 0, 0]
        faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
        rows, fraction, mass, barycentric = overlap_mass(
            x, [[0, 1, 2, 3]], [2.4], cortex, faces, 1 / 48
        )
        np.testing.assert_array_equal(rows, [0])
        np.testing.assert_allclose(fraction, [1 / 8])
        np.testing.assert_allclose(mass, [0.3])
        np.testing.assert_allclose(barycentric, [[0.125, 0.625, 0.125, 0.125]])
        with self.assertRaisesRegex(ValueError, "aggregate solid volumes disagree"):
            overlap_mass(x, [[0, 1, 2, 3]], [2.4], cortex, faces, 0.02)

    def test_enclosed_solid_without_face_intersection(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        body = 0.2 * x + 0.1
        faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
        _, fraction, _, _ = overlap_mass(
            x, [[0, 1, 2, 3]], [1], body, faces, 0.2**3 / 6
        )
        np.testing.assert_allclose(fraction, [0.2**3])

    def test_disjoint_and_inverted_material_are_distinct(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
        rows, _, masses, _ = overlap_mass(x, [[0, 1, 2, 3]], [2], x + 2, faces, 0)
        self.assertEqual(len(rows), 0)
        self.assertEqual(masses.sum(), 0.0)
        with self.assertRaisesRegex(ValueError, "inverted"):
            overlap_mass(x, [[0, 2, 1, 3]], [2], x + 2, faces, 0)

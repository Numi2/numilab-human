"""A positive cell volume can conceal a large shear in registered tissue."""

import unittest

import numpy as np
from audit_registered_cortical_occupancy import loaded_deformation


class LoadedMaterialTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.vstack((np.zeros(3), np.eye(3)))
        self.tet = np.array([[0, 1, 2, 3]])

    def measure(self, points, material=15):
        return loaded_deformation(
            self.reference, points, self.tet, np.array([material])
        )[material]

    def test_rigid_rotation_and_translation(self):
        basis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        stats = self.measure(self.reference @ basis + [1, -1, 2])
        self.assertEqual(stats["principal_stretch_failures"], 0)
        self.assertEqual(stats["minimum_stretch"], 1)

    def test_volume_preserving_shear(self):
        points = self.reference.copy()
        points[2, 0] = 2
        stats = self.measure(points, 16)
        self.assertEqual(stats["relative_volume_failures"], 0)
        self.assertEqual(stats["principal_stretch_failures"], 1)
        self.assertAlmostEqual(stats["maximum_stretch"], 1 + np.sqrt(2))
        self.assertEqual(stats["worst_stretch_cell"], 0)

    def test_inverted_stress_free_reference(self):
        self.reference[[1, 2]] = self.reference[[2, 1]]
        stats = self.measure(self.reference)
        self.assertEqual(stats["invalid_reference_cells"], 1)
        self.assertIsNone(stats["maximum_stretch"])

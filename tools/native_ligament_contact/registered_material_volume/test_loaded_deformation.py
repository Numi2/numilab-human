"""A positive cell volume can conceal a large shear in registered tissue."""

import unittest

import numpy as np
from audit_registered_cortical_occupancy import accepted_bed_load, loaded_deformation


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

    def test_compensated_displacements_survive_world_coordinate_rounding(self):
        observed = self.reference + 2**26
        stats = loaded_deformation(
            self.reference, observed, self.tet, np.array([15]), compensated=True
        )[15]
        self.assertEqual(stats["minimum_stretch"], 1)
        self.assertEqual(self.measure(observed)["relative_volume_failures"], 1)

    def test_dense_bed_force_matches_accepted_impulse(self):
        line = ("resting_fem_skin_bed_accepted_root_reaction normal_impulse_Ns=0.01 "
                "mean_normal_force_N=10 root_dt_s=0.001 valid_contact_sample_count=2 "
                "scope=last_accepted_root_existing_frame_reduction articulated_support_rows=0")
        self.assertEqual(accepted_bed_load([line], 0.001)["last_root_mean_normal_force_N"], 10)
        with self.assertRaises(ValueError):
            accepted_bed_load([line], 0.002)

    def test_inverted_stress_free_reference(self):
        self.reference[[1, 2]] = self.reference[[2, 1]]
        stats = self.measure(self.reference)
        self.assertEqual(stats["invalid_reference_cells"], 1)
        self.assertIsNone(stats["maximum_stretch"])

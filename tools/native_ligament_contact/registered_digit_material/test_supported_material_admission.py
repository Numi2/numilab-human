import unittest
import numpy as np
from supported_material_admission import certify, material_state


class MaterialAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.vstack((np.zeros(3), np.eye(3)))
        self.t = np.array([[0, 1, 2, 3]])
        self.p = np.array([[1.e5, 3.e5], [2.e3, 5.e3]])

    def test_rotated_and_translated_reference(self):
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        result = certify(self.x, self.x @ R + 5., self.t, np.array([16]), self.p)
        self.assertTrue(result['passed'])
        self.assertEqual(result['peak_stress_Pa'], 0.)

    def test_shear_preserves_volume_but_fails_admission(self):
        F = np.eye(3); F[0, 1] = 1.5
        result = certify(self.x, self.x @ F.T, self.t, np.array([15]), self.p)
        self.assertEqual(result['invalid_geometry'], 0)
        self.assertEqual(result['outside_stress_cap'], 0)
        self.assertFalse(result['passed'])
        self.assertEqual(result['outside_stretch_envelope'], 1)

    def test_component_stress_for_strong_shear(self):
        F = np.eye(3); F[0, 1] = 51.
        _, _, peak = material_state(self.x[self.t], (self.x @ F.T)[self.t], self.p[:1])
        self.assertAlmostEqual(peak[0], 51. * self.p[0, 0])
        self.assertEqual(certify(self.x, self.x @ F.T, self.t, np.array([15]), self.p)['outside_stress_cap'], 1)

    def test_reflection_and_nan_are_rejected(self):
        for scale in (-1., np.nan):
            y = self.x.copy(); y[:, 0] *= scale
            with self.assertRaises(ValueError):
                certify(self.x, y, self.t, np.array([15]), self.p)

    def test_global_mass_cancellation_cannot_hide_local_transfer(self):
        from transfer_registered_loading import donor_account
        cells = np.vstack((self.t, self.t))
        before = np.array([1., 1.]); after = np.array([.9, 1.1])
        for roles, donors in (([15, 16], [4, 4]), ([16, 16], [4, 5])):
            with self.assertRaises(ValueError):
                donor_account(cells, np.array(roles), np.array(donors), self.x, self.x, before, after)
        result = donor_account(cells, np.array([16, 16]), np.array([4, 4]), self.x, self.x, before, after)
        self.assertEqual(len(result), 1)
        self.assertLess(abs(result[0]['mass_delta_kg']), 1.e-12)

    def test_material_specific_stress_limit(self):
        F = np.eye(3); F[0, 1] = .1
        for tag, failed in ((15, 1), (16, 0)):
            r = certify(self.x, self.x @ F.T, self.t, np.array([tag]), self.p, [1.e3, 5.e5])
            self.assertEqual(r['outside_stress_cap'], failed)

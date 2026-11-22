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

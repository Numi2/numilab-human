import unittest
import numpy as np
from numilab_human import common_atlas_skin_geometry_registration as owner
from numilab_human import model as human


class SkinRegistrationRegularizationTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
            [0., 1., 0.], [.5, .5, .4],
        ])
        self.faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        self.free = np.array([2, 3, 4])

    def test_rigid_registration_is_a_fixed_point(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = self.points @ rotation.T + np.array([.3, -.7, 1.1])
        actual, report = owner.regularize_registered_skin_positions(
            self.points, target, self.faces, self.free)
        np.testing.assert_allclose(actual, target, atol=2e-14, rtol=0)
        self.assertLess(report["iterations"][-1]["energy"], 1e-25)
        self.assertFalse(report["anatomical_admission"])
        self.assertFalse(report["native_admission"])

    def test_nonrigid_targets_keep_fixed_boundary_and_reduce_energy(self):
        target = self.points.copy()
        target[2] += [.1, -.2, .3]
        target[4] += [0., 0., -.35]
        actual, report = owner.regularize_registered_skin_positions(
            self.points, target, self.faces, self.free, maximum_iterations=20)
        np.testing.assert_array_equal(actual[[0, 1]], target[[0, 1]])
        energy = [r["energy"] for r in report["iterations"]]
        self.assertLess(energy[-1], energy[0])
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.linalg.norm(actual[4] - target[4]), .01)
        self.assertTrue(report["source_topology_unchanged"])

    def test_rejects_invalid_geometry_and_controls(self):
        for kwargs in ({"target_fit_weight": 0}, {"target_fit_weight": float("nan")},
                       {"maximum_iterations": 0}, {"maximum_iterations": 97}):
            with self.subTest(kwargs=kwargs), self.assertRaises(human.ImportError):
                owner.regularize_registered_skin_positions(
                    self.points, self.points, self.faces, self.free, **kwargs)
        duplicate = self.points.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaises(human.ImportError):
            owner.regularize_registered_skin_positions(
                duplicate, duplicate, self.faces, self.free)
        with self.assertRaises(human.ImportError):
            owner.regularize_registered_skin_positions(
                self.points, self.points, self.faces, np.array([2, 2]))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from fit_cavity_mass import fit_mass, point_mass_features
from audit_cavity_response import elastic
from find_connected_mass_support import route
from torso_cell_geometry import geom


class MassRouteTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, 1.0, 0.0],
                [-0.5, -1.0, 0.0],
                [0.0, 0.0, -0.8],
                [0.0, 0.0, 1.3],
            ]
        )

        def orient(t):
            t = np.array(t)
            for r in t:
                if geom(self.x, r[None])[0][0] < 0:
                    r[1], r[2] = r[2], r[1]
            return t

        self.old = orient([[3, 4, 0, 1], [3, 4, 1, 2], [3, 4, 2, 0]])
        self.new = orient([[0, 1, 2, 3], [0, 1, 2, 4]])
        self.m = geom(self.x, self.old)[0] * 900.0

    def test_affine_mass_and_spatial_moment(self):
        poses = [self.x, self.x * np.array([1.1, 0.9, 1.2]) + [3.0, -2.0, 1.0]]
        mass = fit_mass(
            self.x, poses, self.old, self.new, self.m, (850.0, 950.0), False
        )
        self.assertIsNotNone(mass)
        np.testing.assert_allclose(
            point_mass_features(self.old, poses, None) @ self.m,
            point_mass_features(self.new, poses, None) @ mass,
            rtol=1e-12,
            atol=1e-11,
        )

    def test_matched_physical_force_is_the_negative_energy_gradient(self):
        y = self.x @ np.array(
            [[1.02, 0.03, 0.01], [0.07, 0.95, -0.04], [0.0, 0.05, 1.08]]
        )
        nodes = np.unique(self.old)
        dx = np.random.default_rng(24).standard_normal(y.shape)
        _, force, _ = elastic(self.x, y, self.old, nodes, 12.0e3, 85.0e3)
        eps = 1.0e-6
        plus = elastic(self.x, y + eps * dx, self.old, nodes, 12.0e3, 85.0e3)[0]
        minus = elastic(self.x, y - eps * dx, self.old, nodes, 12.0e3, 85.0e3)[0]
        self.assertAlmostEqual(
            (plus - minus) / (2 * eps), -np.sum(force * dx), delta=1.0e-4
        )
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)

    def test_affine_pose_constraints_survive_rank_reduction(self):
        shifted = self.x + [10.0, -20.0, 3.0]
        y = shifted @ np.array([[1.0, 0.1, 0.0], [0.0, 0.9, 0.05], [0.02, 0.03, 1.01]])
        m = route(shifted, y, self.old, np.arange(3), self.new, self.m)
        self.assertIsNotNone(m)
        for pose in (shifted, y):
            np.testing.assert_allclose(
                m @ pose[self.new].mean(1), self.m @ pose[self.old].mean(1), rtol=1e-12
            )

    def test_infeasible_density_and_old_nodal_masses(self):
        self.assertIsNone(
            fit_mass(
                self.x, [self.x], self.old, self.new, self.m, (400.0, 500.0), False
            )
        )
        self.assertIsNone(
            fit_mass(self.x, [self.x], self.old, self.new, self.m, (850.0, 950.0), True)
        )


if __name__ == "__main__":
    unittest.main()

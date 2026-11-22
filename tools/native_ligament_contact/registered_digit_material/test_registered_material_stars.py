"""Check conservative mass projection across source groups and density intervals."""

import unittest

import numpy as np
from registered_material_stars import project_mass
from torso_cell_geometry import geom


class MaterialStarTests(unittest.TestCase):
    def setUp(self):
        anchors = np.array(
            [[0, 0, 0], [2, 0, 1], [0, 3, 2], [1, 2, 3], [3, 1, 2], [2, 3, 1]],
            dtype=float,
        )
        tetra = np.vstack((np.zeros(3), np.eye(3)))
        points = ((anchors[:, None] + tetra) * 0.001).reshape(-1, 3)
        self.cells = np.arange(len(points)).reshape(-1, 4)
        self.states = [
            points,
            points + [0.03, -0.2, 0.1],
            points @ np.diag([1.01, 1.0, 0.98]),
        ]
        self.volumes = geom(points, self.cells)[0]
        self.groups = np.array([4, 4, 4, 8, 8, 8])

    def test_carrier_groups_and_pose_moments(self):
        for density, bounds in [(1000.0, (900.0, 1100.0)), (900.0, (850.0, 950.0))]:
            mass = density * self.volumes
            assigned = project_mass(
                self.states, self.states, self.cells, mass, self.groups, bounds
            )
            self.assertIsNotNone(assigned)
            for g in np.unique(self.groups):
                rows = self.groups == g
                self.assertAlmostEqual(sum(assigned[rows]), sum(mass[rows]), places=17)
            for state in self.states:
                np.testing.assert_allclose(
                    (assigned - mass) @ state[self.cells].mean(1), 0.0, atol=1e-17
                )
            self.assertTrue(
                np.all(
                    (assigned / self.volumes >= bounds[0])
                    & (assigned / self.volumes <= bounds[1])
                )
            )

    def test_impossible_source_mass(self):
        self.assertIsNone(
            project_mass(
                self.states,
                self.states,
                self.cells,
                1200 * self.volumes,
                self.groups,
                (900.0, 1100.0),
            )
        )


if __name__ == "__main__":
    unittest.main()

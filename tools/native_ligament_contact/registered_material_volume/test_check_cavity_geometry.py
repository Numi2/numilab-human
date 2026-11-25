import unittest

import numpy as np

from check_cavity_geometry import external_incidence_conflicts


class CavityIncidenceTest(unittest.TestCase):
    def setUp(self):
        self.original = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        self.new = np.array([[3, 4, 0, 1], [3, 4, 1, 2], [3, 4, 2, 0]])
        self.rows = np.arange(2)

    def test_unchanged_boundary_neighbour(self):
        mesh = np.vstack((self.original, [0, 1, 3, 5]))
        self.assertEqual(external_incidence_conflicts(mesh, [(self.rows, self.new)]), [])

    def test_new_diagonal_already_used_outside_cavity(self):
        mesh = np.vstack((self.original, [3, 4, 5, 6]))
        self.assertEqual(external_incidence_conflicts(mesh, [(self.rows, self.new)]),
                         [dict(patch=0, simplex=[3, 4])])

    def test_overlapping_source_cavities(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            external_incidence_conflicts(self.original, [(self.rows, self.new)] * 2)


if __name__ == "__main__":
    unittest.main()

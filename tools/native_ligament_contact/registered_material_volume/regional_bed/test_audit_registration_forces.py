import unittest

import numpy as np

from audit_registration_forces import compare


class RegistrationForces(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0., 0, 0], [.02, 0, 0], [0, .02, 0], [0, 0, .02], [0, 0, -.02]])
        self.t = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        self.tag = np.array([15, 16])
        self.params = {15: [20000., 150000.], 16: [1400., 9200.]}

    def run_change(self, nodes, disp):
        return compare(self.p, self.p, self.t, self.tag, self.params, np.array(nodes, int), np.array(disp))

    def test_shared_interface_force_is_assembled_and_conserved(self):
        v = np.array([4e-4, -5e-4, 3e-4])
        combined = self.run_change([0], [v])
        self.assertEqual(combined["incident_cells"], 2)
        self.assertGreater(combined["peak_force_change_N"], 0)
        np.testing.assert_allclose(combined["net_force_change_N"], 0., atol=1e-13)
        by_node = {}
        for i in (0, 1):
            region = compare(self.p, self.p, self.t[i:i+1], self.tag[i:i+1], self.params, [0], [v])
            for r in region["largest_node_changes"]:
                n = r["node_row"]
                by_node[n] = by_node.get(n, np.zeros(3)) + r["force_delta_N"]
        for r in combined["largest_node_changes"]:
            np.testing.assert_allclose(r["force_delta_N"], by_node[r["node_row"]], atol=1e-13)

    def test_rigid_translation_preserves_energy_and_nodal_force(self):
        v = np.tile([.002, .003, .001], (len(self.p), 1))
        r = self.run_change(range(len(v)), v)
        self.assertLess(r["peak_force_change_N"], 1e-12)
        for before, after in r["incident_energy_J"].values():
            self.assertAlmostEqual(before, after, places=12)

    def test_reject_missing_or_duplicate_nodes_and_inversion(self):
        for ids in ([0, 0], [8]):
            with self.assertRaises(ValueError):
                self.run_change(ids, np.zeros((len(ids), 3)))
        with self.assertRaises(ValueError):
            self.run_change([3], [[0., 0., -.05]])


if __name__ == "__main__":
    unittest.main()

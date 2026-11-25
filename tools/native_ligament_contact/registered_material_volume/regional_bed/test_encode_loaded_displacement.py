import unittest

import numpy as np

from encode_loaded_displacement import encode_incident_displacements


class DisplacementTests(unittest.TestCase):
    def test_shared_node_exposes_neighbor_and_preserves_distant_nodes(self):
        ref = np.zeros((7, 3))
        wanted = ref + 0.12345678901
        cells = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        old = ref.copy()
        old[0] = wanted[0]
        def margins(x):
            single = x == x.astype(np.float32).astype(np.float64)
            return [1 if single[1, 0] else -1, 1 if single[5, 0] else -1]
        result, counts = encode_incident_displacements(ref, old, wanted, cells, [0], margins)
        self.assertEqual([(r['nodes'], r['rejected']) for r in counts], [(3, 1), (5, 0)])
        np.testing.assert_array_equal(result[[0, 6]], wanted[[0, 6]])

    def test_failure_never_escapes_with_unproved_geometry(self):
        points = np.full((4, 3), 0.1)
        with self.assertRaisesRegex(ValueError, 'cannot certify'):
            encode_incident_displacements(np.zeros_like(points), points * 0, points,
                                          np.array([[0, 1, 2, 3]]), [0], lambda _: [-1])

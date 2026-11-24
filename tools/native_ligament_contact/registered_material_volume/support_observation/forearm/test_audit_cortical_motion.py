import unittest

import manifold3d
import numpy as np
from scipy.spatial.transform import Rotation

from audit_cortical_motion import measure_pair


class CorticalMotionTest(unittest.TestCase):
    def test_containment_contact_and_separation_in_common_moving_frame(self):
        first = manifold3d.Manifold.cube([10, 10, 10], center=True)
        second = manifold3d.Manifold.cube([2, 2, 2], center=True)
        rotation = Rotation.from_euler('xyz', [23, -17, 43], degrees=True)
        origin = np.array([0.31, -0.87, 0.05])
        def frame(offset_mm):
            return {1: (origin, rotation),
                    2: (origin + rotation.apply([offset_mm / 1000, 0, 0]), rotation)}
        contained = measure_pair((1, first), (2, second), {32: frame(0)})
        self.assertEqual(contained['samples_with_overlap'], 1)
        self.assertAlmostEqual(contained['worst_overlap'][1], 8.0, places=10)
        separate = measure_pair((1, first), (2, second), {64: frame(8)})
        self.assertEqual(separate['samples_with_overlap'], 0)
        self.assertAlmostEqual(separate['gap_range_for_nonoverlapping_samples_mm'][0], 2.0)
        identity = Rotation.identity()
        contact = measure_pair((1, first), (2, second), {
            32: {1: (np.zeros(3), identity), 2: (np.array([.006, 0, 0]), identity)}})
        self.assertEqual(contact['samples_with_overlap'], 0)
        self.assertEqual(contact['gap_range_for_nonoverlapping_samples_mm'][0], 0.0)


if __name__ == '__main__':
    unittest.main()

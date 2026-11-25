import unittest

import numpy as np
from measure_exterior_footprint import projected_area


class FootprintTests(unittest.TestCase):
    def test_clipped_triangle_overlapping_other_winding(self):
        xyz = np.array([[0.,0,0],[1,0,1],[0,1,1]])
        faces = np.array([[0,1,2],[2,1,0]])
        for gap,expected in [(0,0),(.5,.125),(1,.5)]:
            self.assertAlmostEqual(projected_area(xyz,faces,[0,0,1],0,gap),expected)

import itertools
import unittest

import numpy as np

from measure_bed_footprint import area_fraction


class BedAreaTests(unittest.TestCase):
    def test_clip_permutation_and_ties(self):
        for heights, expected in [
            ((0, 2, 2), 0.25),
            ((0, 0, 2), 0.75),
            ((0, 1, 2), 0.5),
            ((0, 1, 1), 1),
            ((1, 1, 1), 0),
            ((0, 0, 0), 1),
            ((2, 2, 2), 0),
        ]:
            all_orders = np.array(list(itertools.permutations(heights)))
            np.testing.assert_allclose(area_fraction(all_orders, 1), expected)

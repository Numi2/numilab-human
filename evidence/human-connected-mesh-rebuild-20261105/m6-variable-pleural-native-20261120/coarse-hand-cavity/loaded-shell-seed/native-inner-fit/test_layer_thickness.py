"""Prevent the authoring step from emitting a fat column over its native limit."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_preload_cells import largest_admitted_move
from predict import read_model


class LayerThicknessTests(unittest.TestCase):
    def test_previous_proposal_is_projected_into_admitted_interval(self):
        model, _, world, rest, *_ = read_model()
        node = 3703
        delta = np.array((-0.3666383350281856, -0.020093339934801002, 0.11608633803762027)) * .001
        bounds = np.array(((.0005, .0025), (.0005, .010)))
        self.assertGreater(np.linalg.norm(rest[node] + delta - rest[node + model.outer]), .010)
        clipped = largest_admitted_move(rest, world, node, model.outer, delta, bounds)
        self.assertGreater(np.linalg.norm(clipped), .95 * np.linalg.norm(delta))
        for coordinates in (rest, world):
            neighbors = coordinates[np.array((-model.outer, model.outer)) + node]
            thickness = np.linalg.norm(coordinates[node] + clipped - neighbors, axis=1)
            self.assertTrue(np.all(thickness > bounds[:, 0]))
            self.assertTrue(np.all(thickness < bounds[:, 1]))


if __name__ == "__main__":
    unittest.main()

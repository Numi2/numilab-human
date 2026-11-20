"""Check the new paired triangle split and two-sided volume ancestry."""
import unittest

import numpy as np

from rebuild import determinant
from paired_surface import paired_triples, stellar_children
from run_patch import unchanged_boundaries
from verify_patch import typed_surface


class PairedSurfaceTests(unittest.TestCase):
    def test_split_shared_face_preserves_both_typed_cavities(self):
        world = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [0., 0., 1.], [0., 0., -1.]], dtype=float)
        old = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        face = np.array([0, 1, 2])
        new = np.vstack([stellar_children(cell, face, 5) for cell in old])
        world = np.vstack((world, [.30, .40, 0.]))
        self.assertTrue(np.all(determinant(world, new) > 0.))
        unchanged_boundaries(new, old, np.repeat(np.arange(2), 3), (world, 2. * world))
        surface = typed_surface(new, np.repeat((15, 16), 3))
        self.assertNotIn((0, 1, 2), surface)
        self.assertTrue({(0, 1, 5), (0, 2, 5), (1, 2, 5)} <= surface)
        self.assertEqual(len(surface), 9)

    def test_match_uses_column_ancestry_with_different_vertex_orders(self):
        columns = np.array([[0, 6, 4], [8, 3, 5], [2, 7, 1]])
        faces = np.array([[1, 5, 4], [2, 0, 8], [3, 7, 6]])
        _, key = paired_triples(columns, faces, ["deep", "outer", "interface"])
        self.assertEqual(key[(0, 1, 2)], {0: 1, 1: 2, 2: 0})


if __name__ == "__main__":
    unittest.main()

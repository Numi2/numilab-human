"""Verify that a constrained skin/fat cavity preserves every registered facet."""
import unittest
import numpy as np
from extract_cavity_domain import extract, faces


class MaterialDomainTest(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]], dtype=float)
        self.cells = np.array([[0,1,2,3],[0,2,1,4]], dtype=int)
        self.surface = faces(self.cells[:1])

    def test_extracts_only_material_side(self):
        for cells in (self.cells, self.cells[::-1]):
            np.testing.assert_array_equal(extract(self.points, cells, self.surface), self.cells[:1])

    def test_keeps_both_tissue_cells_across_unconstrained_face(self):
        f = faces(self.cells)
        _, first, count = np.unique(np.sort(f,axis=1), axis=0, return_index=True, return_counts=True)
        np.testing.assert_array_equal(extract(self.points, self.cells, f[first[count == 1]]), self.cells)

    def test_missing_constrained_face(self):
        bad = self.surface.copy()
        bad[0] = [0,3,4]
        with self.assertRaisesRegex(ValueError, 'missing'):
            extract(self.points, self.cells, bad)

    def test_rejects_inconsistent_orientation(self):
        bad = self.surface.copy()
        bad[0] = bad[0, ::-1]
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            extract(self.points, self.cells, bad)

    def test_rejects_duplicate_constraints(self):
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            extract(self.points, self.cells, np.vstack([self.surface,self.surface[0]]))


if __name__ == '__main__':
    unittest.main()

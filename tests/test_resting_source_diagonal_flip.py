import unittest
from fractions import Fraction

import numpy as np

from numilab_human.resting_source_diagonal_flip import flip_same_source_facet_diagonal


class SourceFacetDiagonalFlipTests(unittest.TestCase):
    def setUp(self):
        self.points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.labels = [('pinned-source', 134, 2952)] * 2
        self.lineage = [(3657,), (3660,)]

    def run_flip(self, **overrides):
        values = dict(vertices_exact=self.points, faces=self.faces,
                      source_facet_labels=self.labels, edge=(0, 2),
                      face_lineage=self.lineage)
        values.update(overrides)
        return flip_same_source_facet_diagonal(**values)

    def test_convex_patch_preserves_inputs_and_combines_ancestry(self):
        old = self.faces.copy()
        result, proof = self.run_flip()
        self.assertTrue(np.array_equal(self.faces, old))
        self.assertEqual(proof.source_ancestor_faces, (3657, 3660))
        self.assertEqual(proof.face_indices, (0, 1))
        self.assertEqual(sorted(map(tuple, result)), [(1, 2, 3), (1, 3, 0)])
        self.assertEqual(self.points, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])

    def test_exact_small_source_patch_survives_large_translation(self):
        scale = Fraction(1, 10**20)
        points = [(10**10 + scale*x, 10**10 + scale*y, Fraction(17))
                  for x, y, _ in self.points]
        result, _ = self.run_flip(vertices_exact=points)
        self.assertTrue(np.array_equal(result, self.run_flip()[0]))

    def test_different_source_facets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'facet identities'):
            self.run_flip(source_facet_labels=['original-A', 'original-B'])

    def test_missing_source_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'facet identities'):
            self.run_flip(source_facet_labels=[None, None])

    def test_exact_nonplanarity_is_rejected(self):
        points = self.points.copy()
        points[3] = (0, 1, Fraction(1, 10**30))
        with self.assertRaisesRegex(ValueError, 'exactly coplanar'):
            self.run_flip(vertices_exact=points)

    def test_native_rounded_points_cannot_authenticate_source(self):
        with self.assertRaisesRegex(ValueError, 'integers'):
            self.run_flip(vertices_exact=np.array(self.points, dtype=np.float32))

    def test_concave_source_patch_is_rejected_without_mutation(self):
        points = [(0, 0, 0), (1, 0, 0), (Fraction(1, 4), Fraction(1, 4), 0), (0, 1, 0)]
        original = self.faces.copy()
        with self.assertRaisesRegex(ValueError, 'convex source patch'):
            self.run_flip(vertices_exact=points)
        self.assertTrue(np.array_equal(original, self.faces))

    def test_protected_interface_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'protected interface'):
            self.run_flip(protected_faces=[1])

    def test_existing_replacement_diagonal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_flip(faces=np.array([[0, 1, 2], [0, 2, 3], [1, 3, 4]]),
                          vertices_exact=self.points+[(2, 2, 0)],
                          source_facet_labels=self.labels+['other'],
                          face_lineage=self.lineage+[(99,)])

    def test_nonmanifold_old_diagonal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            self.run_flip(faces=np.array([[0, 1, 2], [0, 2, 3], [2, 0, 4]]),
                          vertices_exact=self.points+[(2, 2, 0)],
                          source_facet_labels=self.labels+['other'],
                          face_lineage=self.lineage+[(99,)])

    def test_opposite_winding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'winding'):
            self.run_flip(faces=np.array([[0, 1, 2], [0, 3, 2]]))

    def test_malformed_indices_are_rejected(self):
        for faces in [self.faces.astype(float), self.faces.astype(bool),
                      np.array([[0, 1, -1], [0, 2, 3]])]:
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                self.run_flip(faces=faces)
        with self.assertRaises(ValueError):
            self.run_flip(edge=(False, 2))

    def test_incomplete_or_negative_ancestry_is_rejected(self):
        for ancestry in [[], [(3657,)], [(), (3660,)], [(-1,), (3660,)]]:
            with self.subTest(ancestry=ancestry), self.assertRaises(ValueError):
                self.run_flip(face_lineage=ancestry)


if __name__ == '__main__':
    unittest.main()

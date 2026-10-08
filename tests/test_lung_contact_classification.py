import unittest

from numilab_human.lung_contact_classification import (
    UNCLASSIFIED, UNCLASSIFIED_LOBE, classify_diaphragm_lobe,
    classify_lobe_lobe, classification_counts, point_on_segment,
)


class ExactContactClassificationTests(unittest.TestCase):
    def test_exact_lobe_vertex_edge_and_full_face_features(self):
        a = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        vertex_neighbor = ((0, 0, 0), (-2, 0, 0), (0, -2, 0))
        edge_neighbor = ((0, 0, 0), (4, 0, 0), (2, -2, 0))
        self.assertEqual(classify_lobe_lobe(a, vertex_neighbor, [(0, 0, 0)]),
                         "shared_exact_vertex")
        self.assertEqual(classify_lobe_lobe(a, edge_neighbor, [(1, 0, 0), (3, 0, 0)]),
                         "shared_exact_edge")
        self.assertEqual(classify_lobe_lobe(a, a, [(0, 0, 0), (4, 0, 0)]),
                         "coincident_exact_triangle")

    def test_mapped_boundary_requires_concrete_segment_support(self):
        mapped = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        unrelated = ((1, 1, 0), (5, 1, 0), (1, 5, 0))
        boundary = (((0, 0, 0), (4, 0, 0)),)
        self.assertEqual(classify_diaphragm_lobe(
            mapped, unrelated, [(1, 0, 0), (3, 0, 0)], boundary),
            "exact_declared_interface_boundary_contact")
        self.assertEqual(classify_diaphragm_lobe(
            mapped, unrelated, [(1, 1, 0)], boundary), UNCLASSIFIED)

    def test_points_on_different_boundary_edges_do_not_authorize_a_segment(self):
        mapped = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        unrelated = ((1, 1, 0), (5, 1, 0), (1, 5, 0))
        boundary = (((0, 0, 0), (4, 0, 0)), ((0, 0, 0), (0, 4, 0)))
        # Each endpoint lies on some boundary edge, but their connecting segment
        # is not contained in either edge and cannot be authorized as boundary contact.
        self.assertEqual(classify_diaphragm_lobe(
            mapped, unrelated, [(1, 0, 0), (0, 1, 0)], boundary), UNCLASSIFIED)

    def test_empty_boundary_cannot_authorize_contact(self):
        mapped = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        unrelated = ((1, 1, 0), (5, 1, 0), (1, 5, 0))
        # Regression for the vacuous all(... for p in []) boundary check.
        self.assertEqual(classify_diaphragm_lobe(
            mapped, unrelated, [(1, 1, 0)], ()), UNCLASSIFIED)

    def test_unknown_labels_and_pair_kinds_fail_closed(self):
        counts, unclassified = classification_counts(
            "diaphragm_lobe",
            ["exact_reciprocal_mapped_face", "unreachable", "future_unknown_label"],
        )
        self.assertEqual(counts["unreachable"], 1)
        self.assertEqual(unclassified, 2)
        self.assertEqual(classification_counts("unknown_pair", ["anything"])[1], 1)
        self.assertEqual(classification_counts(
            "lobe_lobe", ["shared_exact_vertex", UNCLASSIFIED_LOBE])[1], 1)

    def test_closed_segment_membership_is_exact_and_inclusive(self):
        a, b = (0, 0, 0), (4, 2, 0)
        self.assertTrue(point_on_segment((2, 1, 0), a, b))
        self.assertTrue(point_on_segment(a, a, b))
        self.assertTrue(point_on_segment(b, a, b))
        self.assertFalse(point_on_segment((3, 1, 0), a, b))
        self.assertFalse(point_on_segment((5, 2, 0), a, b))


if __name__ == "__main__":
    unittest.main()

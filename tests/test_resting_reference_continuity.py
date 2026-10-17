import unittest

import numpy as np

from numilab_human import resting_anatomy_verify as av


FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


class ReferenceContinuityTests(unittest.TestCase):
    def surface(self, points=POINTS, faces=FACES, key=(51005, 86)):
        return av.prepare_surface(key, points, faces)

    def test_closed_disjoint_cut_fails_before_expensive_predicates(self):
        reference = self.surface()
        # Both candidates are individually closed, outward and non-intersecting.
        points = np.concatenate((POINTS / 4, POINTS / 4 + 2))
        faces = np.concatenate((FACES, FACES + 4))
        candidate = self.surface(points, faces)
        report = av.compare_reference_surface_continuity(reference, candidate)
        self.assertTrue(candidate.closed)
        self.assertEqual(candidate.self_audit()["count"], 0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["component_count_unchanged"])
        self.assertEqual(len(report["reference_components"]), 1)
        self.assertEqual(len(report["candidate_components"]), 2)
        self.assertEqual(report["absolute_oriented_volume_ratio"], 1 / 32)

    def test_exact_seams_and_unused_allocations_do_not_split_reference(self):
        # A duplicated raw slot and an unused extreme point are not extra solids.
        points = np.vstack((POINTS, POINTS[0], [1e9, 1e9, 1e9])).astype(np.float32)
        faces = FACES.copy()
        faces[0, 0] = 4
        reference = self.surface(points, faces)
        before_points, before_faces = points.tobytes(), faces.tobytes()
        report = av.compare_reference_surface_continuity(reference, self.surface())
        self.assertTrue(report["passed"])
        self.assertEqual(report["absolute_oriented_volume_ratio"], 1)
        self.assertEqual(points.tobytes(), before_points)
        self.assertEqual(faces.tobytes(), before_faces)
        self.assertFalse(report["anatomical_admission"])

    def test_orientation_reversal_rejects_even_with_same_component_count(self):
        report = av.compare_reference_surface_continuity(
            self.surface(), self.surface(faces=FACES[:, ::-1].copy()))
        self.assertTrue(report["component_count_unchanged"])
        self.assertFalse(report["orientation_inventory_unchanged"])
        self.assertFalse(report["passed"])

    def test_report_does_not_impose_unjustified_anatomical_volume_threshold(self):
        report = av.compare_reference_surface_continuity(
            self.surface(), self.surface(points=POINTS * 2))
        self.assertTrue(report["passed"])
        self.assertEqual(report["absolute_oriented_volume_ratio"], 8)
        self.assertFalse(report["anatomical_admission"])
        self.assertFalse(report["native_admission"])

    def test_identity_and_open_boundary_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError, "same typed"):
            av.compare_reference_surface_continuity(
                self.surface(), self.surface(key=(51005, 84)))
        with self.assertRaisesRegex(ValueError, "closed oriented"):
            av.compare_reference_surface_continuity(
                self.surface(), self.surface(faces=FACES[:-1]))


if __name__ == "__main__":
    unittest.main()

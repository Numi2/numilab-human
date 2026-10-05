import unittest
import numpy as np

from numilab_human.resting_anatomy_joint_compiler import (
    PINNED_INPUT_SHA256,
    evaluate_costal_source_field,
    apply_costal_source_field_if_changed,
    record_reciprocal_child_pair,
    validate_impression_report,
)


class ReciprocalChildLineageTests(unittest.TestCase):
    def test_shared_quality_flip_children_count_once_but_retain_all_ancestry(self):
        rows = {}
        key = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        record_reciprocal_child_pair(rows, key, stable_id=15, diaphragm_face=9,
            liver_face=17, aggregate_parent_face=120, diaphragm_parent_faces={40},
            liver_parent_face=11)
        record_reciprocal_child_pair(rows, key, stable_id=15, diaphragm_face=9,
            liver_face=17, aggregate_parent_face=121, diaphragm_parent_faces={41},
            liver_parent_face=12)

        self.assertEqual(len(rows), 1)
        record = rows[key]
        self.assertEqual(record["owner"], (15, 9, 17))
        self.assertEqual(record["aggregate_parent_faces"], {120, 121})
        self.assertEqual(record["diaphragm_parent_faces"], {40, 41})
        self.assertEqual(record["liver_parent_faces"], {11, 12})

    def test_duplicate_coordinate_child_with_different_owner_fails(self):
        rows = {}
        key = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        record_reciprocal_child_pair(rows, key, stable_id=15, diaphragm_face=9,
            liver_face=17, aggregate_parent_face=120, diaphragm_parent_faces={40},
            liver_parent_face=11)
        with self.assertRaisesRegex(ValueError, "multiple reciprocal surface owners"):
            record_reciprocal_child_pair(rows, key, stable_id=16, diaphragm_face=9,
                liver_face=17, aggregate_parent_face=121, diaphragm_parent_faces={40},
                liver_parent_face=12)


class VisceralImpressionReceiptTests(unittest.TestCase):
    def _report(self):
        return {
            "candidate_sha256": "candidate",
            "liver_source_sha256": PINNED_INPUT_SHA256["liver_npz"],
            "preserved_diaphragm_interface_faces": 849,
            "lost_diaphragm_interface_faces": 0,
        }

    def test_accepts_only_exact_pinned_source_and_all_contact_faces(self):
        validate_impression_report(self._report(), "candidate")

    def test_rejects_lost_reciprocal_contact_face(self):
        report = self._report()
        report["preserved_diaphragm_interface_faces"] = 848
        with self.assertRaisesRegex(ValueError, "all 849"):
            validate_impression_report(report, "candidate")

    def test_rejects_report_for_a_different_surface(self):
        with self.assertRaisesRegex(ValueError, "exact liver candidate"):
            validate_impression_report(self._report(), "other-candidate")


class CostalSourceFieldTests(unittest.TestCase):
    def test_compact_support_and_anchor_fade(self):
        field = {
            "controls": np.asarray([[0.0, 0.0, 0.0]]),
            "targets": np.asarray([[0.001, 0.0, 0.0]]),
            "anchors": np.asarray([[0.0, 0.0, 0.0]]),
            "radius_m": 1.0,
        }
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        displacement = evaluate_costal_source_field(points, field)
        # The control point lies inside the fixed-anchor clearance, and the
        # compact support vanishes at its radius.
        np.testing.assert_array_equal(displacement, np.zeros((2, 3)))

    def test_control_target_is_used_away_from_anchor(self):
        field = {
            "controls": np.asarray([[0.0, 0.0, 0.0]]),
            "targets": np.asarray([[0.001, 0.0, 0.0]]),
            "anchors": np.asarray([[10.0, 0.0, 0.0]]),
            "radius_m": 1.0,
        }
        displacement = evaluate_costal_source_field(np.asarray([[0.0, 0.0, 0.0]]), field)
        self.assertAlmostEqual(float(displacement[0, 0]), 0.001)
        self.assertAlmostEqual(float(np.linalg.norm(displacement[0, 1:])), 0.0)

    def test_unmoved_float32_surface_keeps_source_row_and_normals(self):
        vertices = np.asarray([
            [10.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [11.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        row = {"vertices6": vertices, "faces": np.asarray([[0, 1, 2]], dtype=np.int64)}
        field = {
            "controls": np.asarray([[0.0, 0.0, 0.0]]),
            "targets": np.asarray([[0.001, 0.0, 0.0]]),
            "anchors": np.asarray([[0.0, 0.0, 0.0]]),
            "radius_m": 1.0,
        }
        updated, report = apply_costal_source_field_if_changed(row, field)
        self.assertIs(updated, row)
        self.assertEqual(report["changed_position_count"], 0)
        np.testing.assert_array_equal(updated["vertices6"], vertices)


if __name__ == "__main__":
    unittest.main()

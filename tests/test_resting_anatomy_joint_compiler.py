import unittest

from numilab_human.resting_anatomy_joint_compiler import record_reciprocal_child_pair


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


if __name__ == "__main__":
    unittest.main()

import unittest

from source_face_identity import primary_source_face


class SourceFaceIdentityTest(unittest.TestCase):
    def test_several_source_faces_keep_the_authored_group_identity(self):
        tag = (1 << 31) + 38
        self.assertEqual(primary_source_face(38, [14318, 15919]), tag)
        self.assertEqual(primary_source_face(38, [14318, 15919], tag), tag)
        self.assertNotEqual(primary_source_face(39, [14318, 15919], tag), tag)
        self.assertEqual(primary_source_face(38, [14318, 15919], 15919), 15919)

    def test_single_source_face_uses_the_actual_face_id(self):
        tag = (1 << 31) + 38
        self.assertEqual(primary_source_face(38, [14318], tag), 14318)
        self.assertEqual(primary_source_face(38, [14318, 14318], tag), 14318)

    def test_empty_support_and_invalid_groups_fail(self):
        with self.assertRaises(ValueError):
            primary_source_face(0, [])
        for group in (-1, 1 << 31):
            with self.assertRaises(ValueError):
                primary_source_face(group, [1, 3])


if __name__ == "__main__":
    unittest.main()

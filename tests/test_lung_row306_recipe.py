import unittest
import numpy as np
from numilab_human.lung_row306_recipe import filter_parallel_face_rows

class LungRow306RecipeTests(unittest.TestCase):
    def test_duplicate_face_compaction_filters_all_parallel_lineage_arrays(self):
        # Same unoriented triangle; metadata differs so a stale mask is exposed.
        faces = np.asarray([[3,7,9],[9,7,3],[3,9,11]], dtype=np.int64)
        origins = np.asarray([57527,57530,57531], dtype=np.int64)
        kinds = np.asarray([1,0,1], dtype=np.int8)
        keep = np.asarray([True,False,True], dtype=bool)
        before = (faces.copy(), origins.copy(), kinds.copy())
        out_f,out_o,out_k = filter_parallel_face_rows(faces,origins,kinds,keep)
        np.testing.assert_array_equal(out_f, [[3,7,9],[3,9,11]])
        np.testing.assert_array_equal(out_o, [57527,57531])
        np.testing.assert_array_equal(out_k, [1,1])
        self.assertEqual(len(out_f),len(out_o))
        self.assertEqual(len(out_f),len(out_k))
        for actual,expected in zip((faces,origins,kinds),before):
            np.testing.assert_array_equal(actual,expected)

    def test_filter_rejects_misaligned_provenance(self):
        with self.assertRaisesRegex(ValueError,"lengths must match"):
            filter_parallel_face_rows(np.zeros((2,3),dtype=np.int64),
                np.zeros(1,dtype=np.int64),np.zeros(2,dtype=np.int8),
                np.ones(2,dtype=bool))

if __name__ == "__main__":
    unittest.main()

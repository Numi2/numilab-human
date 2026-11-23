import unittest

import numpy as np

from verify_registered_frames import apply_nodes


class MaterialFramesTest(unittest.TestCase):
    def setUp(self):
        self.native = np.array([[0.00002, 0, 0], [0.000024, 0, 0]], dtype="f4")
        self.ids = np.array([0, 1])
        self.shift = np.array([2., 0., 0.])
        self.world = self.native.astype(float) + self.shift

    def apply(self, ids, scene, raw=None):
        return apply_nodes(self.native, self.world, ids, scene, np.eye(3), self.shift, raw)

    def test_keep_unedited_native_bits(self):
        emitted = self.world.astype("f4")
        self.assertFalse(np.array_equal((emitted-self.shift).astype("f4"), self.native))
        np.testing.assert_array_equal(self.apply(self.ids, emitted), self.native)
        emitted[1,0] += 0.001
        result = self.apply(self.ids, emitted)
        np.testing.assert_array_equal(result[0], self.native[0])
        np.testing.assert_array_equal(result[1], (emitted[1]-self.shift).astype("f4"))

    def test_verify_canonical_reconstruction(self):
        edited = self.native.copy()
        edited[1,1] = 0.025
        emitted = (edited.astype(float)+self.shift).astype("f4")
        np.testing.assert_array_equal(self.apply(self.ids, emitted, edited), edited)
        with self.assertRaises(ValueError):
            self.apply(self.ids, emitted, self.native)
        with self.assertRaises(ValueError):
            self.apply(self.ids[::-1], emitted, edited)


if __name__ == "__main__":
    unittest.main()

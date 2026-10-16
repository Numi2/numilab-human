import unittest
import numpy as np
from numilab_human.surface_precision_retriangulation import (
    flip_closed_surface_edge_preserving_allocations as flip,
)
from tests import test_surface_precision_retriangulation as fixtures


class AllocationPreservingFlipTests(unittest.TestCase):
    def setUp(self):
        self.p, self.f = fixtures.SurfacePrecisionRetriangulationTests._sliver_octahedron()

    def apply(self, p=None, f=None, edge=(0, 1)):
        return flip(self.p if p is None else p, self.f if f is None else f,
                    identity=(51004, 49), edge=edge, minimum_altitude_m=1e-3)

    def test_preserves_inactive_slots_and_exact_coordinate_seam_aliases(self):
        p = np.vstack((self.p, self.p[0], [[9., 8., 7.]])).astype(np.float32)
        f = self.f.copy()
        f[0, f[0] == 0] = 6
        before_p, before_f = p.copy(), f.copy()
        out, proof = self.apply(p, f)
        np.testing.assert_array_equal(p, before_p)
        np.testing.assert_array_equal(f, before_f)
        changed = proof["changed_parent_face_indices"]
        self.assertEqual(changed, [0, 3])
        untouched = sorted(set(range(len(f))) - set(changed))
        np.testing.assert_array_equal(out[untouched], f[untouched])
        self.assertNotIn(7, out)
        self.assertFalse(proof["anatomical_admission"])
        for row in proof["operation"]["output_face_lineage"]:
            self.assertEqual(row["source_parent_face_indices"], changed)

    def test_rejects_open_mesh_and_invalid_endpoint_without_mutation(self):
        p, f = self.p.copy(), self.f.copy()
        with self.assertRaisesRegex(ValueError, "closed"):
            self.apply(f=self.f[:-1])
        for edge in ((False, 1), (0, 0), (0, 9)):
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                self.apply(edge=edge)
        np.testing.assert_array_equal(self.p, p)
        np.testing.assert_array_equal(self.f, f)

    def test_rejects_nonfloat32_and_unreferenced_slot(self):
        with self.assertRaisesRegex(ValueError, "Float32"):
            self.apply(p=self.p.astype(float))
        p = np.vstack((self.p, [[9., 8., 7.]])).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "not referenced"):
            self.apply(p=p, edge=(0, 6))


if __name__ == "__main__":
    unittest.main()

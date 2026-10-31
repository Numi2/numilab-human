#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).with_name("audit_combined_declared_surface_diff_354.py")
spec = importlib.util.spec_from_file_location("combined_surface_diff", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class CombinedSurfaceDiffControls(unittest.TestCase):
    def test_declared_subset_and_false_noops(self):
        audit.require_declared_change("D", [], [], {"changed_roles":["I"]})
        audit.require_declared_change("I", [1], [2], {"changed_roles":["I"]})
        for role, changed, removed, config in [
            ("D", [1], [], {"changed_roles":["I"]}),
            ("I", [], [], {"changed_roles":["I"]}),
            ("I", [1], [], {"changed_roles":[]}),
            ("I", [1], [], {"changed_roles":["I","I"]}),
            ("I", [1], [], {"changed_roles":["X"]}),
        ]:
            with self.assertRaises(ValueError):
                audit.require_declared_change(role,changed,removed,config)

    def test_point_rows_reorder_by_exact_f32_identity(self):
        base = np.asarray([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        candidate = base[[2,0,1]]
        np.testing.assert_array_equal(audit.point_identity_map(base,candidate,"T"), [2,0,1])

    def test_point_map_rejects_moved_added_and_duplicate_rows(self):
        base = np.asarray([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        moved = base.copy(); moved[1,0] = np.nextafter(moved[1,0], np.float32(2))
        with self.assertRaisesRegex(ValueError, "moved or new"):
            audit.point_identity_map(base,moved,"T")
        with self.assertRaisesRegex(ValueError, "shape changed"):
            audit.point_identity_map(base,base[:2],"T")
        duplicate = base.copy(); duplicate[2] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicate F32"):
            audit.point_identity_map(base,duplicate,"T")

    def test_oriented_key_ignores_cyclic_rotation_but_not_reversal(self):
        self.assertEqual(audit.oriented_face_key((2,7,9)), audit.oriented_face_key((7,9,2)))
        self.assertEqual(audit.oriented_face_key((2,7,9)), audit.oriented_face_key((9,2,7)))
        self.assertNotEqual(audit.oriented_face_key((2,7,9)), audit.oriented_face_key((2,9,7)))

    def test_exact_oriented_face_delta_handles_row_reordering(self):
        base = np.asarray([[0,1,2],[0,2,3],[4,5,6]], dtype=np.int64)
        cand = np.asarray([[5,6,4],[2,3,0],[1,2,3]], dtype=np.int64)
        changed, removed = audit.face_delta(base,cand,"T")
        np.testing.assert_array_equal(changed, [2])
        np.testing.assert_array_equal(removed, [0])
        reversed_changed, reversed_removed = audit.face_delta(
            np.asarray([[0,1,2]],np.int64), np.asarray([[0,2,1]],np.int64), "R")
        np.testing.assert_array_equal(reversed_changed,[0])
        np.testing.assert_array_equal(reversed_removed,[0])

    def test_duplicate_and_degenerate_triangle_control(self):
        with self.assertRaisesRegex(ValueError, "duplicate unoriented"):
            audit.face_delta(np.asarray([[0,1,2]],np.int64),
                             np.asarray([[0,1,2],[2,1,0]],np.int64),"T")
        with self.assertRaisesRegex(ValueError,"degenerate candidate face"):
            audit.face_delta(np.asarray([[0,1,2]],np.int64),
                             np.asarray([[0,0,1]],np.int64),"T")

    def test_boundary_requires_closed_interior_orientation(self):
        faces = np.asarray([[0,1,2],[0,2,3]],np.int64)
        boundary = audit.directed_boundary_edges(faces)
        self.assertEqual(len(boundary),4)
        with self.assertRaisesRegex(ValueError,"same-direction"):
            audit.directed_boundary_edges(np.asarray([[0,1,2],[0,3,2]],np.int64))
        with self.assertRaisesRegex(ValueError,"nonmanifold"):
            audit.directed_boundary_edges(np.asarray([[0,1,2],[1,0,3],[0,1,4]],np.int64))

    def test_exact_owner_detects_crossing_changed_triangles(self):
        base = audit.import_path(
            "combined_surface_test_owner_197",
            audit.BASE_DRIVER)
        owner = base.module("combined_surface_test_owner_198",base.OWNER).load_exact_owner()
        pts = np.asarray([
            [0,0,0],[1,0,0],[0,1,0],
            [.25,.2,-1],[.25,.2,1],[.25,.8,0]
        ],np.float32)
        faces = np.asarray([[0,1,2],[3,4,5]],np.int64)
        raw,_ = owner.float32_surface_records(pts.astype(np.float64),faces,source_seam_groups=())
        recs = [(x[0],x[1],x[2],i,tuple(x[4])) for i,x in enumerate(raw)]
        result = owner._audit_record_pairs([(recs[0],recs[1])],same_surface=False)
        self.assertGreaterEqual(result["count"],1)

if __name__ == "__main__":
    unittest.main(verbosity=2)


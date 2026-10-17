import unittest
import numpy as np

from numilab_human.forearm_topology_patch import (
    check_closed_oriented_vertex_links,
    collapse_short_edge_in_two_frames,
)


def fixture(alias=True):
    a, e = np.float32(1e-8), np.float32(1e-9)
    p = np.asarray([[a, 0, 0], [a, e, 0], [0, 0, a],
                    [0, 0, -a], [-a, 0, 0], [0, -a, 0]], dtype="<f4")
    w = p.copy()
    if alias:
        w[1] = w[0]
    f = np.asarray([[0, 1, 2], [1, 4, 2], [4, 5, 2], [5, 0, 2],
                    [1, 0, 3], [4, 1, 3], [5, 4, 3], [0, 5, 3]], dtype="<u4")
    return p, w, f


class TwoFrameEdgeRetirementTests(unittest.TestCase):
    def call(self, p, w, f, **kw):
        args = dict(modifiable_vertex_mask=np.ones(len(p), dtype=bool),
                    maximum_edge_length_m=2e-9)
        args.update(kw)
        return collapse_short_edge_in_two_frames(f, p, w, 0, 1, **args)

    def test_captured_alias_retires_faces_and_preserves_allocations(self):
        p, w, f = fixture()
        old = tuple(x.tobytes() for x in (p, w, f))
        output, report = self.call(p, w, f)
        self.assertEqual(len(output), 6)
        self.assertEqual(report["incident_face_indices_removed"], [0, 4])
        self.assertEqual(report["source_world_zero_face_counts_before"], [0, 2])
        self.assertEqual(report["source_world_zero_face_counts_after"], [0, 0])
        self.assertEqual(report["rewritten_input_face_ids"], [1, 5])
        self.assertEqual(report["input_star_face_ids"], [0, 1, 3, 4, 5, 7])
        self.assertEqual(tuple(x.tobytes() for x in (p, w, f)), old)
        self.assertTrue(check_closed_oriented_vertex_links(output, len(p))[
            "closed_oriented_indexed_manifold"])
        self.assertFalse(report["anatomical_admission"])
        self.assertFalse(report["native_admission"])

    def test_source_alias_is_also_accounted(self):
        p, w, f = fixture()
        p[1] = p[0]
        _, report = self.call(p, w, f)
        self.assertEqual(report["source_world_zero_face_counts_before"], [2, 2])
        self.assertEqual(report["source_world_zero_face_counts_after"], [0, 0])

    def test_endpoint_protection(self):
        p, w, f = fixture()
        for endpoint in (0, 1):
            mask = np.ones(len(p), dtype=bool)
            mask[endpoint] = False
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                    ValueError, "explicitly modifiable"):
                self.call(p, w, f, modifiable_vertex_mask=mask)

    def test_both_frame_length_bound(self):
        p, w, f = fixture()
        w[1, 1] = 1e-6
        with self.assertRaisesRegex(ValueError, "metre bound"):
            self.call(p, w, f)

    def test_default_requires_a_real_degeneracy_reduction(self):
        p, w, f = fixture(False)
        with self.assertRaisesRegex(ValueError, "strictly improve"):
            self.call(p, w, f)

    def test_collision_driven_policy_still_retains_both_frame_checks(self):
        p, w, f = fixture(False)
        _, report = self.call(p, w, f, require_zero_face_reduction=False)
        self.assertEqual(report["source_world_zero_face_counts_after"], [0, 0])
        self.assertTrue(report["exact_self_and_neighbor_checks_required"])
        self.assertFalse(report["strict_zero_face_reduction_required"])

    def test_world_fold_cannot_hide_behind_valid_source(self):
        p, w, f = fixture(False)
        w *= 1e8
        w[0] = [-3, 0, 0]
        with self.assertRaisesRegex(ValueError, "world.*reverses"):
            self.call(p, w, f, maximum_edge_length_m=10.,
                      require_zero_face_reduction=False)

    def test_nonmanifold_input_is_rejected_before_repair(self):
        p, w, f = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate indexed face"):
            self.call(p, w, np.concatenate((f, f[[1]])))

    def test_tetrahedron_edge_cannot_collapse_to_duplicate_faces(self):
        p = np.asarray([[0, 0, 0], [1e-9, 0, 0], [0, 1e-8, 0],
                        [0, 0, 1e-8]], dtype="<f4")
        w = p.copy()
        w[1] = w[0]
        f = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
        with self.assertRaisesRegex(ValueError, "duplicate indexed face"):
            self.call(p, w, f)

    def test_malformed_types_and_nonfinite_values(self):
        p, w, f = fixture()
        cases = [
            (p.astype("<f8"), w, f, {}, "finite Float32"),
            (p, w, f.astype("<i8"), {}, "UInt32"),
            (p, w, f, {"modifiable_vertex_mask": np.ones(len(p))}, "boolean"),
            (p, w, f, {"maximum_edge_length_m": float("nan")}, "finite"),
            (p, w, f, {"maximum_edge_length_m": True}, "finite"),
            (p, w, f, {"require_zero_face_reduction": 1}, "boolean"),
        ]
        bad = w.copy()
        bad[3, 0] = np.nan
        cases.append((p, bad, f, {}, "finite Float32"))
        for source, world, faces, kw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.call(source, world, faces, **kw)


if __name__ == "__main__":
    unittest.main()

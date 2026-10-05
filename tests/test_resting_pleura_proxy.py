import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human.resting_pleura_proxy import derive_lung_union_exterior


def _tetra(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    if signed_volume(np.asarray(vertices, dtype=np.float64), faces) < 0:
        faces = faces[:, [0, 2, 1]]
    xyz = np.asarray(vertices, dtype=np.float32)
    normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (len(xyz), 1))
    return np.column_stack((xyz, normals)), faces


class RestingPleuraProxyTests(unittest.TestCase):
    def _two_lung_fixture(self):
        shared = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        left_top = _tetra(
            shared + [[0, 0, 1]],
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
        left_bottom = _tetra(
            shared + [[0, 0, -1]],
            [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
        )

        base = [[10, 0, 0], [11, 0, 0], [10, 1, 0]]
        right_top = _tetra(
            base + [[10, 0, 1]],
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
        right_bottom = _tetra(
            base + [[10, 0, -1]],
            [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
        )
        # This tetra shares one outer face of right_top. Its apex lies across
        # that face, so the three source cells form one closed right envelope.
        right_side = _tetra(
            [[10, 0, 0], [11, 0, 0], [10, 0, 1], [10.5, -1, 0.5]],
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
        return {
            305: left_bottom,
            306: right_bottom,
            307: right_top,
            308: left_top,
            309: right_side,
        }

    def test_external_union_cancels_only_reciprocal_faces(self):
        vertices6, faces, detail = derive_lung_union_exterior(self._two_lung_fixture())
        self.assertEqual(detail["exact_reciprocal_face_group_count"], 3)
        self.assertEqual(detail["cancelled_opposite_face_count"], 6)
        self.assertEqual(detail["retained_external_triangle_count"], 14)
        self.assertEqual(detail["retained_external_vertex_count"], 11)
        self.assertEqual(len(detail["components"]), 2)
        self.assertTrue(np.isfinite(vertices6).all())
        self.assertEqual(topology_report(faces)["boundary_edge_count"], 0)
        self.assertEqual(topology_report(faces)["nonmanifold_edge_count"], 0)
        self.assertEqual(topology_report(faces)["orientation_error_edge_count"], 0)
        self.assertTrue(detail["position_coordinates_are_exact_source_float32"])

    def test_same_wound_duplicate_interface_is_rejected(self):
        lobes = self._two_lung_fixture()
        v, f = lobes[308]
        lobes[308] = (v, f.copy())
        lobes[308][1][0] = lobes[305][1][0]
        with self.assertRaisesRegex(ValueError, "same-wound|do not have reciprocal"):
            derive_lung_union_exterior(lobes)

    def test_only_the_source_anchored_closed_microfragment_may_be_omitted(self):
        lobes = self._two_lung_fixture()
        anchors = np.asarray([
            [0.009118776768445969, 0.036569900810718536, -0.056037694215774536],
            [0.008264528587460518, 0.037101924419403076, -0.05697183683514595],
            [0.00812491588294506, 0.03752134367823601, -0.056306567043066025],
            [0.00816801656037569, 0.03760587051510811, -0.05599856749176979],
        ], dtype=np.float32)
        fragment_faces = [
            ((0, 1, 2), 305), ((0, 2, 3), 305),
            ((1, 3, 2), 308), ((0, 3, 1), 308),
        ]
        for stable_id in (305, 308):
            old_v, old_f = lobes[stable_id]
            new_v = np.vstack((old_v, np.column_stack((anchors, np.tile([0, 1, 0], (4, 1)))).astype(np.float32)))
            new_faces = [tuple(len(old_v) + index for index in tri)
                         for tri, owner in fragment_faces if owner == stable_id]
            lobes[stable_id] = (new_v, np.vstack((old_f, np.asarray(new_faces, dtype=np.int64))))

        vertices6, faces, detail = derive_lung_union_exterior(lobes)
        self.assertEqual(len(detail["excluded_source_microfragments"]), 1)
        fragment = detail["excluded_source_microfragments"][0]
        self.assertEqual(fragment["face_count"], 4)
        self.assertEqual(fragment["vertex_count"], 4)
        self.assertAlmostEqual(fragment["signed_volume_m3"], -6.212387788e-13, delta=1e-16)
        self.assertEqual(len(detail["components"]), 2)
        self.assertEqual(topology_report(faces)["boundary_edge_count"], 0)
        explicit_vertices, explicit_faces, explicit_detail = derive_lung_union_exterior(
            lobes, coordinate_quantization_m=0,
        )
        np.testing.assert_array_equal(vertices6, explicit_vertices)
        np.testing.assert_array_equal(faces, explicit_faces)
        self.assertEqual(detail, explicit_detail)

    def test_declared_source_quantum_is_applied_to_microfragment_anchors(self):
        lobes = self._two_lung_fixture()
        anchors = np.asarray([
            [0.009118776768445969, 0.036569900810718536, -0.056037694215774536],
            [0.008264528587460518, 0.037101924419403076, -0.05697183683514595],
            [0.00812491588294506, 0.03752134367823601, -0.056306567043066025],
            [0.00816801656037569, 0.03760587051510811, -0.05599856749176979],
        ], dtype=np.float64)
        quantum = 1e-6
        anchors = (np.rint(anchors / quantum) * quantum).astype(np.float32)
        fragment_faces = [
            ((0, 1, 2), 305), ((0, 2, 3), 305),
            ((1, 3, 2), 308), ((0, 3, 1), 308),
        ]
        for stable_id in (305, 308):
            old_v, old_f = lobes[stable_id]
            new_v = np.vstack((old_v, np.column_stack((anchors, np.tile([0, 1, 0], (4, 1))))))
            new_faces = [tuple(len(old_v) + index for index in tri)
                         for tri, owner in fragment_faces if owner == stable_id]
            lobes[stable_id] = (new_v, np.vstack((old_f, np.asarray(new_faces, dtype=np.int64))))

        _, faces, detail = derive_lung_union_exterior(
            lobes, coordinate_quantization_m=quantum,
        )
        self.assertEqual(detail["source_coordinate_quantization_m"], quantum)
        self.assertEqual(len(detail["excluded_source_microfragments"]), 1)
        self.assertEqual(len(detail["components"]), 2)
        self.assertEqual(topology_report(faces)["boundary_edge_count"], 0)

    def test_unrelated_detached_component_fails_closed(self):
        lobes = self._two_lung_fixture()
        old_v, old_f = lobes[305]
        xyz = np.asarray([[20, 0, 0], [21, 0, 0], [20, 1, 0], [20, 0, 1]], dtype=np.float32)
        normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1))
        tetra_v = np.column_stack((xyz, normals))
        tetra_f = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
        tetra_v, tetra_f = _tetra(xyz, tetra_f)
        lobes[305] = (np.vstack((old_v, tetra_v)), np.vstack((old_f, tetra_f + len(old_v))))
        with self.assertRaisesRegex(ValueError, "detached lung-union component"):
            derive_lung_union_exterior(lobes)


if __name__ == "__main__":
    unittest.main()

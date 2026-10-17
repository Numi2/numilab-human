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

    def _append_source_microfragment(self, lobes):
        anchors = np.asarray([
            [0.009118776768445969, 0.036569900810718536, -0.056037694215774536],
            [0.008264528587460518, 0.037101924419403076, -0.05697183683514595],
            [0.00812491588294506, 0.03752134367823601, -0.05630656704337071],
            [0.00816801656037569, 0.03760587051510811, -0.05599856749176979],
        ], dtype=np.float32)
        fragment_faces = [
            ((0, 1, 2), 305), ((0, 2, 3), 305),
            ((1, 3, 2), 308), ((0, 3, 1), 308),
        ]
        vertex_indices = {}
        face_indices = {}
        for stable_id in (305, 308):
            old_v, old_f = lobes[stable_id]
            first = len(old_v)
            extra = np.column_stack((anchors, np.tile([0, 1, 0], (4, 1)))).astype(np.float32)
            new_v = np.vstack((old_v, extra))
            local_faces = [tuple(first + index for index in tri)
                           for tri, owner in fragment_faces if owner == stable_id]
            face_indices[stable_id] = list(range(len(old_f), len(old_f) + len(local_faces)))
            vertex_indices[stable_id] = list(range(first, first + len(anchors)))
            lobes[stable_id] = (new_v, np.vstack((old_f, np.asarray(local_faces, dtype=np.int64))))
        return lobes, vertex_indices, face_indices

    @staticmethod
    def _copy_lobes(lobes):
        return {sid: (vertices.copy(), faces.copy()) for sid, (vertices, faces) in lobes.items()}

    def test_source_lineage_allows_only_same_registered_microfragment_faces_after_deformation(self):
        source, vertices_by_row, _ = self._append_source_microfragment(self._two_lung_fixture())
        for transform in ("translation", "deformation"):
            with self.subTest(transform=transform):
                candidate = self._copy_lobes(source)
                all_anchor_points = np.asarray(source[305][0][vertices_by_row[305], :3], dtype=np.float64)
                center = all_anchor_points.mean(axis=0)
                for stable_id in (305, 308):
                    vertices6, faces = candidate[stable_id]
                    indices = vertices_by_row[stable_id]
                    points = vertices6[indices, :3].astype(np.float64)
                    if transform == "translation":
                        points += np.asarray([0.02, -0.01, 0.005])
                    else:
                        points = center + (points - center) * np.asarray([1.01, 0.99, 1.02])
                    vertices6[indices, :3] = points.astype(np.float32)
                    candidate[stable_id] = (vertices6, faces)
                _, _, detail = derive_lung_union_exterior(
                    candidate, source_lineage_lobes=source,
                )
                self.assertEqual(len(detail["excluded_source_microfragments"]), 1)
                lineage = detail["excluded_source_microfragments"][0]["source_lineage"]
                self.assertEqual(lineage["recognition_mode"], "exact_source_face_and_vertex_inventory")
                self.assertEqual(set(lineage["reference_lobe_row_arrays"]), {"305", "306", "307", "308", "309"})
                self.assertTrue(all(len(row["vertices6_f32_sha256"]) == 64 and len(row["faces_i64_sha256"]) == 64
                                    for row in lineage["reference_lobe_row_arrays"].values()))
                self.assertEqual(lineage["source_faces"], [[305, 4], [305, 5], [308, 4], [308, 5]])
                self.assertGreater(lineage["minimum_candidate_to_reference_face_area_ratio"], 0.05)
                self.assertIn("source-face/vertex inventory", detail["source_microfragment_anchor_rule"])
                self.assertEqual(len(detail["components"]), 2)

    def test_source_lineage_does_not_relax_default_anchor_recognition(self):
        source, vertices_by_row, _ = self._append_source_microfragment(self._two_lung_fixture())
        candidate = self._copy_lobes(source)
        for stable_id in (305, 308):
            vertices6, faces = candidate[stable_id]
            vertices6[vertices_by_row[stable_id], :3] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)
            candidate[stable_id] = (vertices6, faces)
        with self.assertRaisesRegex(ValueError, "detached lung-union component"):
            derive_lung_union_exterior(candidate)
        # The same displaced component is accepted only when its exact source rows are supplied.
        _, _, detail = derive_lung_union_exterior(candidate, source_lineage_lobes=source)
        self.assertEqual(len(detail["excluded_source_microfragments"]), 1)

    def test_source_lineage_rejects_unrecognized_reference_and_changed_face_inventory(self):
        source, vertices_by_row, _ = self._append_source_microfragment(self._two_lung_fixture())
        unrecognized = self._copy_lobes(source)
        for stable_id in (305, 308):
            vertices6, faces = unrecognized[stable_id]
            vertices6[vertices_by_row[stable_id], :3] += np.asarray([0.2, 0.1, -0.1], dtype=np.float32)
            unrecognized[stable_id] = (vertices6, faces)
        with self.assertRaisesRegex(ValueError, "detached lung-union component"):
            derive_lung_union_exterior(source, source_lineage_lobes=unrecognized)

        for mutation in ("face_id", "connectivity"):
            with self.subTest(mutation=mutation):
                candidate = self._copy_lobes(source)
                vertices6, faces = candidate[305]
                if mutation == "face_id":
                    faces[[0, -1]] = faces[[-1, 0]]
                else:
                    faces[-1, 0] = faces[-1, 1]
                candidate[305] = (vertices6, faces)
                with self.assertRaisesRegex(ValueError, "face/index inventory"):
                    derive_lung_union_exterior(candidate, source_lineage_lobes=source)

    def test_explicit_ancestry_allows_unrelated_face_inventory_change(self):
        source, _, _ = self._append_source_microfragment(self._two_lung_fixture())
        candidate = self._copy_lobes(source)
        # Remove both reciprocal faces of an ordinary internal interface
        # outside the declared fragment. Their removal leaves the emitted
        # exterior unchanged and gives each candidate row a different face
        # inventory without compromising the actual shell.
        # This owner is classifying fragment lineage; the downstream topology
        # gate remains responsible for rejecting an incomplete external shell.
        for stable_id in (305, 308):
            vertices6, faces = candidate[stable_id]
            candidate[stable_id] = (vertices6, np.delete(faces, 0, axis=0))
        face_ancestry = {}
        vertex_ancestry = {}
        for stable_id in (305, 306, 307, 308, 309):
            source_face_count = len(source[stable_id][1])
            if stable_id in (305, 308):
                face_ancestry[stable_id] = [[i] for i in range(source_face_count) if i != 0]
            else:
                face_ancestry[stable_id] = [[i] for i in range(source_face_count)]
            vertex_ancestry[stable_id] = [[i] for i in range(len(source[stable_id][0]))]

        expected_vertices, expected_faces, _ = derive_lung_union_exterior(source)
        vertices6, faces, detail = derive_lung_union_exterior(
            candidate,
            source_lineage_lobes=source,
            source_face_ancestry=face_ancestry,
            source_vertex_ancestry=vertex_ancestry,
        )
        np.testing.assert_array_equal(vertices6, expected_vertices)
        np.testing.assert_array_equal(faces, expected_faces)
        lineage = detail["excluded_source_microfragments"][0]["source_lineage"]
        self.assertEqual(lineage["recognition_mode"], "exact_source_face_and_vertex_ancestry")
        self.assertTrue(lineage["source_face_connectivity_verified"])
        self.assertEqual(lineage["source_faces"], [[305, 4], [305, 5], [308, 4], [308, 5]])
        self.assertNotEqual(len(candidate[305][1]), len(source[305][1]))
        self.assertNotEqual(len(candidate[308][1]), len(source[308][1]))

    def test_explicit_ancestry_rejects_tampered_face_and_vertex_identity(self):
        source, _, fragment_faces = self._append_source_microfragment(self._two_lung_fixture())
        for mutation in ("face_identity", "vertex_identity", "connectivity", "fractional_identity"):
            with self.subTest(mutation=mutation):
                candidate = self._copy_lobes(source)
                face_ancestry = {
                    sid: [[i] for i in range(len(source[sid][1]))]
                    for sid in (305, 306, 307, 308, 309)
                }
                vertex_ancestry = {
                    sid: [[i] for i in range(len(source[sid][0]))]
                    for sid in (305, 306, 307, 308, 309)
                }
                if mutation == "face_identity":
                    # Swap one source fragment face identity with an unrelated
                    # source face while retaining a unique ancestry map.
                    face_ancestry[305][0], face_ancestry[305][fragment_faces[305][0]] = (
                        face_ancestry[305][fragment_faces[305][0]], face_ancestry[305][0]
                    )
                elif mutation == "vertex_identity":
                    # The vertex map is per vertex; target an actual
                    # source-fragment vertex.
                    fragment_vertex = len(source[305][0]) - 4
                    vertex_ancestry[305][fragment_vertex] = [0]
                elif mutation == "fractional_identity":
                    face_ancestry[305][fragment_faces[305][0]] = [4.5]
                else:
                    vertices6, faces = candidate[305]
                    face_id = fragment_faces[305][0]
                    faces[face_id, 2] = 0
                    candidate[305] = (vertices6, faces)
                with self.assertRaises(ValueError):
                    derive_lung_union_exterior(
                        candidate,
                        source_lineage_lobes=source,
                        source_face_ancestry=face_ancestry,
                        source_vertex_ancestry=vertex_ancestry,
                    )

    def test_explicit_ancestry_requires_source_and_both_ancestry_maps(self):
        source, _, _ = self._append_source_microfragment(self._two_lung_fixture())
        with self.assertRaisesRegex(ValueError, "authenticated source lineage"):
            derive_lung_union_exterior(
                source,
                source_face_ancestry={sid: [] for sid in (305, 306, 307, 308, 309)},
                source_vertex_ancestry={sid: [] for sid in (305, 306, 307, 308, 309)},
            )

    def test_explicit_ancestry_rejects_fractional_stable_ids_and_face_indices(self):
        source, _, _ = self._append_source_microfragment(self._two_lung_fixture())
        face_map = {sid: [[i] for i in range(len(source[sid][1]))]
                    for sid in (305, 306, 307, 308, 309)}
        vertex_map = {sid: [[i] for i in range(len(source[sid][0]))]
                      for sid in (305, 306, 307, 308, 309)}

        fractional_face_keys = dict(face_map)
        fractional_face_keys[305.1] = fractional_face_keys.pop(305)
        with self.assertRaisesRegex(ValueError, "integer or canonical decimal stable lobe IDs"):
            derive_lung_union_exterior(
                source, source_lineage_lobes=source,
                source_face_ancestry=fractional_face_keys,
                source_vertex_ancestry=vertex_map,
            )

        for changed_side in ("candidate", "source"):
            with self.subTest(changed_side=changed_side):
                candidate = self._copy_lobes(source)
                lineage = self._copy_lobes(source)
                target = candidate if changed_side == "candidate" else lineage
                vertices6, faces = target[305]
                target[305] = (vertices6, faces.astype(np.float64))
                with self.assertRaisesRegex(ValueError, "requires integer candidate and source face indices"):
                    derive_lung_union_exterior(
                        candidate, source_lineage_lobes=lineage,
                        source_face_ancestry=face_map,
                        source_vertex_ancestry=vertex_map,
                    )

        canonical_face_keys = {str(sid): value for sid, value in face_map.items()}
        canonical_vertex_keys = {str(sid): value for sid, value in vertex_map.items()}
        _, _, detail = derive_lung_union_exterior(
            source, source_lineage_lobes=source,
            source_face_ancestry=canonical_face_keys,
            source_vertex_ancestry=canonical_vertex_keys,
        )
        self.assertEqual(len(detail["excluded_source_microfragments"]), 1)

    def test_source_lineage_still_rejects_collapsed_fragment_geometry(self):
        source, vertices_by_row, _ = self._append_source_microfragment(self._two_lung_fixture())
        candidate = self._copy_lobes(source)
        points = candidate[305][0][vertices_by_row[305], :3].astype(np.float64)
        # Collapse one source-fragment triangle while keeping its indexed faces unchanged.
        points[2] = (points[0] + points[1]) * 0.5
        for stable_id in (305, 308):
            vertices6, faces = candidate[stable_id]
            vertices6[vertices_by_row[stable_id], :3] = points.astype(np.float32)
            candidate[stable_id] = (vertices6, faces)
        with self.assertRaisesRegex(ValueError, "zero-area candidate face|5 percent source bound|signed-volume bound"):
            derive_lung_union_exterior(candidate, source_lineage_lobes=source)

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

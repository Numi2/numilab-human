import copy
import hashlib
from fractions import Fraction
import json
import unittest
from unittest.mock import patch

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human import resting_anatomy_interface_patch as anatomy
from numilab_human.resting_pleura_proxy import derive_lung_union_exterior
from numilab_human import resting_pleura_proxy as pleura


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




    def test_owner_authenticates_subdivided_parent_then_local_edge_collapse(self):
        # Affinely transformed octahedron: face [4, 0, 2] matches the left-lung
        # reciprocal face, and a separate detached source fragment is preserved.
        lobes = self._two_lung_fixture()
        xyz = np.asarray([
            [1, 0, 0], [-1, 0, -1], [0, 1, 0], [0, -1, -1],
            [0, 0, 0], [0, 0, -1],
        ], dtype=np.float32)
        octa_faces = np.asarray([
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ], dtype=np.int64)
        lobes[305] = (np.column_stack((xyz, np.zeros_like(xyz))).astype(np.float32), octa_faces)
        source, _, _ = self._append_source_microfragment(lobes)
        source_vertices6, source_faces = source[305]

        # Parent row first subdivides a non-interface source face exactly.
        source_patch_face = 1
        center = source_vertices6[source_faces[source_patch_face], :3].astype(np.float64).mean(axis=0).astype(np.float32)
        parent_vertices6 = np.vstack((source_vertices6, np.r_[center, np.zeros(3, dtype=np.float32)]))
        parent_faces = np.vstack((
            source_faces[:source_patch_face],
            np.asarray([[4, 2, 10], [2, 1, 10], [1, 4, 10]], dtype=np.int64),
            source_faces[source_patch_face + 1:],
        ))
        parent_face_to_source = ([(i,) for i in range(source_patch_face)]
                                 + [(source_patch_face,)] * 3
                                 + [(i,) for i in range(source_patch_face + 1, len(source_faces))])
        parent_vertex_to_source = [(i,) for i in range(len(source_vertices6))] + [(1, 2, 4)]

        # Then collapse edge (3, 5), away from both the shared lobe face and
        # the registered detached microfragment.
        removed, retained = 3, 5
        retired = tuple(map(int, np.flatnonzero(
            np.sum(np.isin(parent_faces, (removed, retained)), axis=1) == 2)))
        old_to_new = {}
        candidate_vertices6 = []
        candidate_vertex_to_parent = []
        for parent_id, vertex in enumerate(parent_vertices6):
            if parent_id == removed:
                continue
            output_id = len(candidate_vertices6)
            old_to_new[parent_id] = output_id
            if parent_id == retained:
                collapsed_xyz = ((parent_vertices6[removed, :3].astype(np.float64)
                                  + parent_vertices6[retained, :3].astype(np.float64)) * 0.5).astype(np.float32)
                candidate_vertices6.append(np.r_[collapsed_xyz, parent_vertices6[parent_id, 3:]])
                candidate_vertex_to_parent.append((removed, retained))
            else:
                candidate_vertices6.append(vertex.copy())
                candidate_vertex_to_parent.append((parent_id,))
        old_to_new[removed] = old_to_new[retained]
        candidate_faces = []
        candidate_face_to_parent = []
        for parent_face_id, face in enumerate(parent_faces):
            if parent_face_id in retired:
                continue
            candidate_faces.append([old_to_new[int(vertex)] for vertex in face])
            candidate_face_to_parent.append((parent_face_id,))
        candidate_vertices6 = np.asarray(candidate_vertices6, dtype=np.float32)
        candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
        noncoplanar_survivors = []
        for output_face_id, parent_face_id in enumerate(candidate_face_to_parent):
            parent_face_id = parent_face_id[0]
            parent_triangle = parent_faces[parent_face_id]
            if np.sum(np.isin(parent_triangle, (removed, retained))) != 1:
                continue
            old_points = parent_vertices6[parent_triangle, :3].astype(np.float64)
            new_points = candidate_vertices6[candidate_faces[output_face_id], :3].astype(np.float64)
            old_normal = np.cross(old_points[1] - old_points[0], old_points[2] - old_points[0])
            signed_offsets = (new_points - old_points[0]) @ old_normal
            if np.any(signed_offsets != 0):
                noncoplanar_survivors.append(output_face_id)
        self.assertTrue(noncoplanar_survivors, "fixture must exercise a non-coplanar surviving face after collapse")

        candidate_lobes = self._copy_lobes(source)
        candidate_lobes[305] = (candidate_vertices6, candidate_faces)
        candidate_face_maps = {}
        candidate_vertex_maps = {}
        for stable_id in (305, 306, 307, 308, 309):
            if stable_id == 305:
                candidate_face_maps[stable_id] = [
                    tuple(sorted({source_id for parent_id in row for source_id in parent_face_to_source[parent_id]}))
                    for row in candidate_face_to_parent
                ]
                candidate_vertex_maps[stable_id] = [
                    tuple(sorted({source_id for parent_id in row for source_id in parent_vertex_to_source[parent_id]}))
                    for row in candidate_vertex_to_parent
                ]
            else:
                candidate_face_maps[stable_id] = [(i,) for i in range(len(source[stable_id][1]))]
                candidate_vertex_maps[stable_id] = [(i,) for i in range(len(source[stable_id][0]))]

        def payload_for(vertices6, faces):
            return anatomy.encode_payload(
                (b"NHANAT1\0", 5, 1, len(vertices6), faces.size, 0, bytes(32)),
                {305: {"body_index": 20, "layer": 8, "flags": 0,
                       "vertices6": vertices6, "faces": faces}},
            )

        def map_hash(rows, kind):
            return pleura._lineage_map_sha256(305, rows, kind=kind)

        identity_faces = [(i,) for i in range(len(source_faces))]
        identity_vertices = [(i,) for i in range(len(source_vertices6))]
        nested_payload = payload_for(source_vertices6, source_faces)
        nested_evidence = {
            "parent_vertices6": source_vertices6,
            "parent_faces": source_faces,
            "parent_payload_bytes": nested_payload,
            "parent_payload_sha256": hashlib.sha256(nested_payload).hexdigest(),
            "parent_surface_sha256": pleura._parent_surface_sha256(305, source_vertices6, source_faces),
            "candidate_face_ancestry": parent_face_to_source,
            "candidate_vertex_ancestry": parent_vertex_to_source,
            "parent_face_ancestry": identity_faces,
            "parent_vertex_ancestry": identity_vertices,
            "immediate_subdivision_face_ids": (source_patch_face,),
        }
        nested_evidence.update({
            "candidate_face_ancestry_sha256": map_hash(parent_face_to_source, "candidate-face-to-parent-face"),
            "candidate_vertex_ancestry_sha256": map_hash(parent_vertex_to_source, "candidate-vertex-to-parent-vertex"),
            "parent_face_ancestry_sha256": map_hash(identity_faces, "parent-face-to-source-face"),
            "parent_vertex_ancestry_sha256": map_hash(identity_vertices, "parent-vertex-to-source-vertex"),
        })
        lattice_scale = 1 << 149

        def exact_patch_geometry(vertices6):
            numerators = np.asarray([
                [str(int(Fraction.from_float(float(value)) * lattice_scale))
                 for value in point]
                for point in vertices6[:, :3]
            ], dtype="U256")
            return {
                "support_source_face_ids": (source_patch_face,),
                "raw_vertex_lattice_numerators": numerators,
                "raw_vertex_lattice_denominators": np.ones(numerators.shape, dtype=np.int64),
                "lattice_scale": lattice_scale,
            }

        parent_payload = payload_for(parent_vertices6, parent_faces)
        edge_evidence = {
            "parent_vertices6": parent_vertices6,
            "parent_faces": parent_faces,
            "parent_payload_bytes": parent_payload,
            "parent_payload_sha256": hashlib.sha256(parent_payload).hexdigest(),
            "parent_surface_sha256": pleura._parent_surface_sha256(305, parent_vertices6, parent_faces),
            "candidate_face_ancestry": candidate_face_to_parent,
            "candidate_vertex_ancestry": candidate_vertex_to_parent,
            "parent_face_ancestry": parent_face_to_source,
            "parent_vertex_ancestry": parent_vertex_to_source,
            "edge_endpoint_parent_vertex_ids": (removed, retained),
            "retired_parent_face_ids": retired,
            "retained_position_policy": "midpoint",
            "collapse_position_f32_m": candidate_vertices6[
                next(i for i, row in enumerate(candidate_vertex_to_parent) if row == (removed, retained)), :3
            ].copy(),
            "maximum_endpoint_displacement_bound_m": 0.6,
            "maximum_metric_deviation_bound_m": 1.0,
            "parent_source_subdivision_evidence": {
                "subdivided_source_face_ids": (source_patch_face,),
                "source_patch_support_face_ids": (source_patch_face,),
                "patch_geometry": exact_patch_geometry(parent_vertices6),
                "subdivision_parent_evidence": nested_evidence,
            },
        }
        for field, rows, kind in (
            ("candidate_face_ancestry_sha256", candidate_face_to_parent, "candidate-face-to-parent-face"),
            ("candidate_vertex_ancestry_sha256", candidate_vertex_to_parent, "candidate-vertex-to-parent-vertex"),
            ("parent_face_ancestry_sha256", parent_face_to_source, "parent-face-to-source-face"),
            ("parent_vertex_ancestry_sha256", parent_vertex_to_source, "parent-vertex-to-source-vertex"),
        ):
            edge_evidence[field] = map_hash(rows, kind)

        _, _, detail = pleura.derive_lung_union_exterior(
            candidate_lobes,
            source_lineage_lobes=source,
            source_face_ancestry=candidate_face_maps,
            source_vertex_ancestry=candidate_vertex_maps,
            allow_source_face_subdivisions={305: (source_patch_face,)},
            source_face_edge_collapse_parent_evidence={305: edge_evidence},
        )
        collapse_report = detail["source_face_edge_collapse_reports_by_stable_id"]["305"]
        self.assertTrue(collapse_report["passed"])
        self.assertTrue(collapse_report["parent_source_subdivision_evidence"][
            "parent_exact_subdivision_patch_geometry"]["exact_raw_and_float32_patch_boundary_conserved"])
        self.assertEqual(
            collapse_report["candidate_surface_sha256"],
            pleura._parent_surface_sha256(305, candidate_vertices6, candidate_faces),
        )
        self.assertTrue(collapse_report["exact_source_and_world_self_intersection_rechecks_required"])
        self.assertEqual(len(detail["components"]), 2)


class ImmediateSubdivisionParentTests(unittest.TestCase):
    def setUp(self):
        xyz = np.asarray([
            [0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0],
        ], dtype=np.float32)
        self.parent_vertices6 = np.column_stack((xyz[:4], np.zeros((4, 3), dtype=np.float32)))
        self.parent_faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
        self.candidate_vertices6 = np.column_stack((xyz, np.zeros((5, 3), dtype=np.float32)))
        self.candidate_faces = np.asarray(
            [[0, 2, 4], [0, 4, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
        self.candidate_face_to_parent = [(0,), (0,), (1,), (2,), (3,)]
        self.candidate_vertex_to_parent = [(0,), (1,), (2,), (3,), (1, 2)]
        self.parent_face_to_source = [(1,), (2,), (3,), (4,)]
        self.parent_vertex_to_source = [(0,), (1,), (2,), (3,)]
        # Original 015 is intentionally in another registered frame. It remains
        # the lineage reference but cannot serve as the immediate patch parent.
        self.original_vertices6 = self.parent_vertices6.copy()
        self.original_vertices6[:, :3] += np.float32(0.25)
        self.original_faces = np.vstack((self.parent_faces[0], self.parent_faces + 0))
        self.parent_payload = anatomy.encode_payload(
            (b"NHANAT1\0", 5, 1, 0, 0, 0, b"s" * 32),
            {305: {"body_index": 20, "layer": 1, "flags": 0,
                   "vertices6": self.parent_vertices6, "faces": self.parent_faces}},
        )
        self.ultimate_face_map = [(1,), (1,), (2,), (3,), (4,)]
        self.ultimate_vertex_map = [(0,), (1,), (2,), (3,), (1, 2)]

    def evidence(self):
        return {
            "parent_vertices6": self.parent_vertices6.copy(),
            "parent_faces": self.parent_faces.copy(),
            "parent_payload_bytes": self.parent_payload,
            "parent_payload_sha256": hashlib.sha256(self.parent_payload).hexdigest(),
            "parent_surface_sha256": pleura._parent_surface_sha256(
                305, self.parent_vertices6, self.parent_faces),
            "candidate_face_ancestry": copy.deepcopy(self.candidate_face_to_parent),
            "candidate_vertex_ancestry": copy.deepcopy(self.candidate_vertex_to_parent),
            "parent_face_ancestry": copy.deepcopy(self.parent_face_to_source),
            "parent_vertex_ancestry": copy.deepcopy(self.parent_vertex_to_source),
            "immediate_subdivision_face_ids": [0],
            "candidate_face_ancestry_sha256": pleura._lineage_map_sha256(
                305, self.candidate_face_to_parent, kind="candidate-face-to-parent-face"),
            "candidate_vertex_ancestry_sha256": pleura._lineage_map_sha256(
                305, self.candidate_vertex_to_parent, kind="candidate-vertex-to-parent-vertex"),
            "parent_face_ancestry_sha256": pleura._lineage_map_sha256(
                305, self.parent_face_to_source, kind="parent-face-to-source-face"),
            "parent_vertex_ancestry_sha256": pleura._lineage_map_sha256(
                305, self.parent_vertex_to_source, kind="parent-vertex-to-source-vertex"),
        }

    def patch_geometry(self):
        nums = self.candidate_vertices6[:, :3].astype(np.int64)
        return {
            "support_source_face_ids": [1],
            "raw_vertex_lattice_numerators": nums,
            "raw_vertex_lattice_denominators": np.ones_like(nums),
            "lattice_scale": 1,
        }

    def validate_chain(self, evidence=None, face_map=None, vertex_map=None, support_ids=None):
        return pleura._validate_source_face_subdivision_parent_chain(
            305, self.candidate_faces, self.candidate_vertices6,
            self.ultimate_face_map if face_map is None else face_map,
            self.ultimate_vertex_map if vertex_map is None else vertex_map,
            len(self.original_faces), len(self.original_vertices6),
            self.evidence() if evidence is None else evidence, [1],
            self.patch_geometry()["support_source_face_ids"] if support_ids is None else support_ids)

    def test_exact_patch_uses_authenticated_immediate_parent_and_composes_to_original(self):
        chain = self.validate_chain()
        self.assertTrue(chain["passed"])
        self.assertEqual(chain["immediate_subdivision_face_ids"], [0])
        self.assertEqual(chain["original_source_subdivision_face_ids"], [1])
        self.assertEqual(chain["immediate_patch_support_face_ids"], [0])
        self.assertEqual(chain["ultimate_patch_support_face_ids"], [1])
        self.assertEqual(chain["parent_payload_sha256"], hashlib.sha256(self.parent_payload).hexdigest())
        self.assertEqual(chain["face_ancestry_composition_checked"], len(self.candidate_faces))
        self.assertEqual(chain["vertex_ancestry_composition_checked"], len(self.candidate_vertices6))

        proof = pleura._source_face_subdivision_patch_report(
            self.candidate_vertices6, self.candidate_faces,
            chain["parent_vertices6"], chain["parent_faces"],
            chain["candidate_face_to_parent"],
            {**self.patch_geometry(), "support_source_face_ids": chain["immediate_patch_support_face_ids"]},
            chain["immediate_subdivision_face_ids"])
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["output_child_count_by_source_face"], {"0": 2})
        with self.assertRaisesRegex(ValueError, "outside its exact parent triangle"):
            pleura._source_face_subdivision_patch_report(
                self.candidate_vertices6, self.candidate_faces,
                self.original_vertices6, self.original_faces,
                self.ultimate_face_map, self.patch_geometry(), [1])

    def test_parent_geometry_and_ancestry_pins_reject_drift(self):
        evidence = self.evidence()
        evidence["parent_vertices6"][0, 0] += np.float32(0.125)
        with self.assertRaisesRegex(ValueError, "arrays differ from the independently pinned payload row"):
            self.validate_chain(evidence)

        evidence = self.evidence()
        evidence["parent_payload_bytes"] = evidence["parent_payload_bytes"] + b"tampered"
        with self.assertRaisesRegex(ValueError, "payload bytes differ from their independent SHA-256 pin"):
            self.validate_chain(evidence)

        evidence = self.evidence()
        evidence["parent_face_ancestry"][0] = (2,)
        evidence["parent_face_ancestry_sha256"] = pleura._lineage_map_sha256(
            305, evidence["parent_face_ancestry"], kind="parent-face-to-source-face")
        with self.assertRaisesRegex(ValueError, "face ancestry differs from the complete source face ancestry"):
            self.validate_chain(evidence)

        evidence = self.evidence()
        evidence["parent_payload_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "parent_payload_sha256 must be a lowercase SHA-256 pin"):
            self.validate_chain(evidence)

    def test_candidate_to_parent_ancestry_must_compose_to_complete_original_map(self):
        bad_faces = copy.deepcopy(self.ultimate_face_map)
        bad_faces[0] = (3,)
        with self.assertRaisesRegex(ValueError, "face ancestry differs from the complete source face ancestry"):
            self.validate_chain(face_map=bad_faces)

        bad_vertices = copy.deepcopy(self.ultimate_vertex_map)
        bad_vertices[-1] = (0, 1)
        with self.assertRaisesRegex(ValueError, "vertex ancestry differs from the complete source vertex ancestry"):
            self.validate_chain(vertex_map=bad_vertices)

    def test_parent_patch_support_must_compose_exactly_to_source_support(self):
        evidence = self.evidence()
        with self.assertRaisesRegex(ValueError, "complete original-source patch support"):
            self.validate_chain(support_ids=[0, 1])

    def test_parent_face_cannot_straddle_the_declared_patch_boundary(self):
        evidence = self.evidence()
        evidence["parent_face_ancestry"][1] = (2, 3)
        evidence["parent_face_ancestry_sha256"] = pleura._lineage_map_sha256(
            305, evidence["parent_face_ancestry"], kind="parent-face-to-source-face")
        ultimate_faces = copy.deepcopy(self.ultimate_face_map)
        ultimate_faces[2] = (2, 3)
        with self.assertRaisesRegex(ValueError, "crosses the declared subdivision patch boundary"):
            self.validate_chain(evidence, face_map=ultimate_faces, support_ids=[2])

    def test_owner_routes_opt_in_patch_to_intermediate_geometry_and_keeps_default_path(self):
        tetra_faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
        candidate_lobes = {sid: (self.candidate_vertices6.copy(), self.candidate_faces.copy())
                            for sid in pleura.LOBE_IDS}
        source_lobes = {sid: (self.original_vertices6.copy(), self.original_faces.copy())
                        for sid in pleura.LOBE_IDS}
        # Only row 305 is subdivided; the other rows are identity maps. The
        # sentinel stops after the owner has validated ancestry and selected
        # its exact parent geometry, before any regional topology audit.
        face_maps = {sid: (self.ultimate_face_map if sid == 305
                           else [(i,) for i in range(len(tetra_faces))])
                     for sid in pleura.LOBE_IDS}
        vertex_maps = {sid: (self.ultimate_vertex_map if sid == 305
                             else [(i,) for i in range(4)])
                       for sid in pleura.LOBE_IDS}
        candidate_lobes[305] = (self.candidate_vertices6.copy(), self.candidate_faces.copy())
        source_lobes[305] = (self.original_vertices6.copy(), self.original_faces.copy())
        parent_evidence = self.evidence()
        parent_evidence.update({
            "parent_vertices6": self.parent_vertices6.copy(),
            "parent_faces": self.parent_faces.copy(),
            "parent_payload_bytes": self.parent_payload,
            "parent_payload_sha256": hashlib.sha256(self.parent_payload).hexdigest(),
        })

        class StopAfterPatchParent(Exception):
            pass

        observed = []
        def stop(candidate_vertices6, candidate_faces, parent_vertices6,
                 parent_faces, parent_map, patch_evidence, repeated):
            observed.append((parent_vertices6.copy(), parent_faces.copy(), tuple(parent_map),
                             dict(patch_evidence), tuple(repeated)))
            raise StopAfterPatchParent

        arguments = dict(
            allow_source_microfragment=True,
            source_lineage_lobes=source_lobes,
            source_face_ancestry=face_maps,
            source_vertex_ancestry=vertex_maps,
            allow_source_face_subdivisions={305: [1]},
            source_face_subdivision_patch_geometry={305: self.patch_geometry()},
        )
        with patch.object(pleura, "_source_face_subdivision_patch_report", side_effect=stop):
            with self.assertRaises(StopAfterPatchParent):
                pleura.derive_lung_union_exterior(
                    candidate_lobes, **arguments,
                    source_face_subdivision_parent_evidence={305: parent_evidence})
        self.assertTrue(np.array_equal(observed[0][0], self.parent_vertices6))
        self.assertTrue(np.array_equal(observed[0][1], self.parent_faces))
        self.assertEqual(observed[0][2], tuple(self.candidate_face_to_parent))
        self.assertEqual(observed[0][3]["support_source_face_ids"], [0])

        observed.clear()
        with patch.object(pleura, "_source_face_subdivision_patch_report", side_effect=stop):
            with self.assertRaises(StopAfterPatchParent):
                pleura.derive_lung_union_exterior(candidate_lobes, **arguments)
        self.assertTrue(np.array_equal(observed[0][0], self.original_vertices6))
        self.assertTrue(np.array_equal(observed[0][1], self.original_faces))
        report = self.validate_chain()
        serializable_report = {key: value for key, value in report.items()
                              if key not in ("parent_vertices6", "parent_faces",
                                             "candidate_face_to_parent")}
        json.dumps(serializable_report, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human.resting_lung_edge_repair import (
    _collapse_free_diaphragm_microedges, _collapse_reciprocal_shared_edge, collapse_midpoint_edge,
)


class RestingLungEdgeRepairTests(unittest.TestCase):
    @staticmethod
    def _octahedron():
        xyz = np.asarray([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0],
            [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=np.float32)
        xyz *= np.float32(1e-5)
        faces = np.asarray([
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ], dtype=np.int64)
        if signed_volume(xyz.astype(np.float64), faces) < 0:
            faces = faces[:, [0, 2, 1]]
        normals = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        return np.column_stack((xyz, normals)).astype(np.float32), faces

    def test_midpoint_edge_collapse_preserves_sphere_and_face_lineage(self):
        vertices6, faces = self._octahedron()
        origins = np.arange(len(faces), dtype=np.int64) + 100
        v2, f2, o2, report = collapse_midpoint_edge(
            vertices6, faces, origins, remove_vertex=0, keep_vertex=2,
        )
        self.assertEqual(v2.shape, (5, 6))
        self.assertEqual(f2.shape, (6, 3))
        self.assertEqual(len(o2), len(f2))
        self.assertEqual(sorted(set(origins) - set(o2)), report["deleted_face_origins"])
        self.assertEqual(report["candidate_topology"]["boundary_edges"], 0)
        self.assertEqual(report["candidate_topology"]["nonmanifold_edges"], 0)
        self.assertEqual(report["candidate_topology"]["orientation_error_edges"], 0)
        self.assertEqual(report["candidate_topology"]["euler_characteristic"], 2)
        self.assertGreater(report["signed_volume_after_m3"], 0)
        self.assertLessEqual(abs(report["signed_volume_delta_m3"]), 1e-12)
        self.assertEqual(topology_report(f2)["boundary_edge_count"], 0)

    @staticmethod
    def _torus():
        import math

        n = m = 8
        major, minor = 30e-6, 10e-6
        points = []
        for i in range(n):
            u = 2.0 * math.pi * i / n
            for j in range(m):
                v = 2.0 * math.pi * j / m
                points.append([
                    (major + minor * math.cos(v)) * math.cos(u),
                    (major + minor * math.cos(v)) * math.sin(u),
                    minor * math.sin(v),
                ])
        xyz = np.asarray(points, dtype=np.float32)
        faces = []
        for i in range(n):
            for j in range(m):
                a = i * m + j
                b = ((i + 1) % n) * m + j
                c = ((i + 1) % n) * m + (j + 1) % m
                d = i * m + (j + 1) % m
                faces.extend(((a, b, c), (a, c, d)))
        faces = np.asarray(faces, dtype=np.int64)
        if signed_volume(xyz.astype(np.float64), faces) < 0:
            faces = faces[:, [0, 2, 1]]
        unit_normals = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        return np.column_stack((xyz, unit_normals)).astype(np.float32), faces

    def test_midpoint_edge_collapse_preserves_nonzero_genus_and_full_face_lineage(self):
        vertices6, faces = self._torus()
        edge_rows = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
        edge_count = len(np.unique(np.sort(edge_rows, axis=1), axis=0))
        euler_before = len(vertices6) - edge_count + len(faces)
        self.assertEqual(euler_before, 0)

        v2, f2, origins2, report = collapse_midpoint_edge(
            vertices6, faces, np.arange(len(faces), dtype=np.int64),
            remove_vertex=0, keep_vertex=1, max_endpoint_displacement_m=5e-6,
        )

        self.assertEqual(report["source_topology"]["genus"], 1)
        self.assertEqual(report["candidate_topology"]["genus"], 1)
        self.assertEqual(report["candidate_topology"]["euler_characteristic"], euler_before)
        self.assertEqual(report["source_face_indices_in_changed_one_ring"],
                         sorted(report["source_face_indices_in_changed_one_ring"]))
        self.assertTrue(report["surviving_changed_face_lineage"])
        self.assertEqual(len(origins2), len(f2))
        self.assertEqual(topology_report(f2)["boundary_edge_count"], 0)
        self.assertAlmostEqual(signed_volume(v2[:, :3].astype(np.float64), f2),
                               signed_volume(vertices6[:, :3].astype(np.float64), faces),
                               delta=1e-12)

    def test_midpoint_edge_collapse_enforces_explicit_displacement_bound(self):
        vertices6, faces = self._octahedron()
        with self.assertRaisesRegex(ValueError, "endpoint-displacement bound"):
            collapse_midpoint_edge(
                vertices6, faces, np.arange(len(faces)), remove_vertex=0, keep_vertex=2,
                max_endpoint_displacement_m=1e-9,
            )

    def test_keep_endpoint_collapse_preserves_shared_coordinate_exactly(self):
        vertices6, faces = self._octahedron()
        keep_before = vertices6[2, :3].copy()
        _, _, _, report = collapse_midpoint_edge(
            vertices6, faces, np.arange(len(faces), dtype=np.int64),
            remove_vertex=0, keep_vertex=2, position_policy="keep",
            max_endpoint_displacement_m=2e-5,
        )
        self.assertEqual(report["collapse_position_policy"], "keep")
        self.assertEqual(report["collapse_position_f32_m"], keep_before.tolist())
        self.assertEqual(report["endpoint_displacements_m"][1], 0.0)
        self.assertAlmostEqual(report["maximum_endpoint_displacement_m"],
                               report["edge_length_m"], delta=1e-12)

    def test_nonedge_is_rejected_without_mutating_inputs(self):
        vertices6, faces = self._octahedron()
        before_v, before_f = vertices6.copy(), faces.copy()
        with self.assertRaisesRegex(ValueError, "exactly two incident faces"):
            collapse_midpoint_edge(
                vertices6, faces, np.arange(len(faces)), remove_vertex=0, keep_vertex=1,
            )
        np.testing.assert_array_equal(vertices6, before_v)
        np.testing.assert_array_equal(faces, before_f)


    def test_reciprocal_shared_edge_collapse_is_transactional_and_preserves_patch(self):
        vertices6, faces = self._octahedron()
        reverse_faces = faces[:, [0, 2, 1]].copy()
        reverse_vertices6 = vertices6.copy()
        reverse_vertices6[:, 3:6] *= -1.0
        rows = {
            307: {"body_index": 20, "vertices6": vertices6.copy(), "faces": faces.copy()},
            309: {"body_index": 20, "vertices6": reverse_vertices6, "faces": reverse_faces},
        }
        before = {
            sid: (row["vertices6"].copy(), row["faces"].copy())
            for sid, row in rows.items()
        }

        candidate, report, ancestry = _collapse_reciprocal_shared_edge(
            rows, owner_pair=(307, 309),
            endpoint_coordinates_m=(vertices6[0, :3], vertices6[2, :3]),
            max_endpoint_displacement_m=20e-6,
        )

        self.assertEqual(report["source_shared_face_counts"], {"(307, 309)": 8})
        self.assertEqual(report["candidate_shared_face_counts"], {"(307, 309)": 6})
        self.assertEqual(report["source_shared_patch_topology"]["face_count"], 8)
        self.assertEqual(report["candidate_shared_patch_topology"]["face_count"], 6)
        self.assertEqual(report["candidate_shared_patch_topology"]["boundary_edge_count"], 0)
        self.assertEqual(report["candidate_shared_patch_topology"]["euler_characteristic"], 2)
        self.assertTrue(report["all_other_reciprocal_owner_maps_unchanged"])
        self.assertEqual(candidate[307]["faces"].shape, (6, 3))
        self.assertEqual(candidate[309]["faces"].shape, (6, 3))
        self.assertEqual(len(ancestry[307]), 6)
        self.assertEqual(len(ancestry[309]), 6)
        for sid in rows:
            np.testing.assert_array_equal(rows[sid]["vertices6"], before[sid][0])
            np.testing.assert_array_equal(rows[sid]["faces"], before[sid][1])
            self.assertEqual(topology_report(candidate[sid]["faces"])["boundary_edge_count"], 0)

    def test_reciprocal_shared_edge_collapse_rejects_unpropagated_third_owner(self):
        vertices6, faces = self._octahedron()
        reverse_faces = faces[:, [0, 2, 1]].copy()
        reverse_vertices6 = vertices6.copy()
        reverse_vertices6[:, 3:6] *= -1.0
        rows = {
            307: {"body_index": 20, "vertices6": vertices6.copy(), "faces": faces.copy()},
            308: {"body_index": 20, "vertices6": vertices6.copy(), "faces": faces.copy()},
            309: {"body_index": 20, "vertices6": reverse_vertices6, "faces": reverse_faces},
        }
        before = {
            sid: (row["vertices6"].copy(), row["faces"].copy())
            for sid, row in rows.items()
        }
        with self.assertRaisesRegex(ValueError, "third-owner"):
            _collapse_reciprocal_shared_edge(
                rows, owner_pair=(307, 309),
                endpoint_coordinates_m=(vertices6[0, :3], vertices6[2, :3]),
                max_endpoint_displacement_m=20e-6,
            )
        for sid in rows:
            np.testing.assert_array_equal(rows[sid]["vertices6"], before[sid][0])
            np.testing.assert_array_equal(rows[sid]["faces"], before[sid][1])



    def test_diaphragm_microedge_collapse_keeps_reciprocal_patch_and_maps_origins(self):
        vertices6, faces = self._torus()
        patch_face = next(
            fi for fi, face in enumerate(faces)
            if 1 in face and 0 not in face
        )
        patch_ids = set(map(int, faces[patch_face]))
        patch_before = {
            tuple(np.asarray(vertices6[i, :3], dtype="<f4").view("<u4").tolist())
            for i in patch_ids
        }
        rows = {
            305: {"vertices6": vertices6.copy(), "faces": faces.copy()},
            311: {"vertices6": vertices6.copy(), "faces": faces.copy()},
        }
        candidate, origins, report = _collapse_free_diaphragm_microedges(
            rows,
            interface_rows=[{"lung_stable_id": 305,
                "diaphragm_patch_face_start": patch_face,
                "diaphragm_patch_face_count": 1}],
            edge_vertex_pairs=[(0, 1)],
            max_endpoint_displacement_m=30e-6,
        )
        self.assertEqual(len(report["operations"]), 1)
        self.assertEqual(report["operations"][0]["removed_face_origins"],
                         report["operations"][0]["collapse"]["deleted_face_origins"])
        self.assertEqual(len(origins), len(candidate[311]["faces"]))
        self.assertEqual(set(range(len(faces))) - set(map(int, origins)),
                         set(report["operations"][0]["removed_face_origins"]))
        self.assertEqual(topology_report(candidate[311]["faces"])["boundary_edge_count"], 0)
        final_patch = set()
        for out_index, origin in enumerate(origins):
            if int(origin) == patch_face:
                final_patch = {
                    tuple(np.asarray(candidate[311]["vertices6"][int(i), :3], dtype="<f4").view("<u4").tolist())
                    for i in candidate[311]["faces"][out_index]
                }
                break
        self.assertEqual(final_patch, patch_before)
        self.assertTrue(report["patch_coordinates_unchanged"])
        self.assertTrue(report["source_face_connectivity_changed_only_by_two_incident_face_deletions_per_collapse"])


    def test_diaphragm_microedge_collapse_rejects_removed_patch_vertex(self):
        vertices6, faces = self._torus()
        patch_face = next(
            fi for fi, face in enumerate(faces)
            if 0 in face and 1 not in face
        )
        rows = {
            305: {"vertices6": vertices6.copy(), "faces": faces.copy()},
            311: {"vertices6": vertices6.copy(), "faces": faces.copy()},
        }
        with self.assertRaisesRegex(ValueError, "remove a free vertex"):
            _collapse_free_diaphragm_microedges(
                rows,
                interface_rows=[{"lung_stable_id": 305,
                    "diaphragm_patch_face_start": patch_face,
                    "diaphragm_patch_face_count": 1}],
                edge_vertex_pairs=[(0, 1)],
            )


    def test_free_diaphragm_sidewall_clearance_preserves_patch_and_lobe(self):
        from numilab_human.resting_lung_edge_repair import _move_free_diaphragm_sidewall_clearance
        from numilab_human.resting_anatomy_interface_patch import normals

        lobe_xyz = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        lobe_faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        if signed_volume(lobe_xyz.astype(np.float64), lobe_faces) < 0:
            lobe_faces = lobe_faces[:, [0, 2, 1]]
        lobe_v6 = np.column_stack((lobe_xyz, normals(lobe_xyz.astype(float), lobe_faces))).astype(np.float32)
        diaphragm_xyz = np.asarray([
            [10, 0, 0], [11, 0, 0], [10, 1, 0],
            [.1, .1, 1.5], [.2, .2, .05], [.4, .1, 1.5],
        ], dtype=np.float32)
        diaphragm_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        diaphragm_v6 = np.column_stack((diaphragm_xyz, np.zeros_like(diaphragm_xyz))).astype(np.float32)
        rows = {
            305: {"body_index": 20, "vertices6": lobe_v6.copy(), "faces": lobe_faces.copy()},
            311: {"body_index": 20, "vertices6": diaphragm_v6.copy(), "faces": diaphragm_faces.copy()},
        }
        before_lobe = rows[305]["vertices6"].copy()
        before_patch = rows[311]["vertices6"][:3, :3].copy()
        before_faces = rows[311]["faces"].copy()
        candidate, report = _move_free_diaphragm_sidewall_clearance(
            rows, interface_rows=[{
                "lung_stable_id": 305, "diaphragm_patch_face_start": 0,
                "diaphragm_patch_face_count": 1,
            }],
            event_faces_by_lobe={305: [1]}, target_clearance_m=.1,
            max_vertex_displacement_m=.2,
        )
        np.testing.assert_array_equal(rows[305]["vertices6"], before_lobe)
        np.testing.assert_array_equal(candidate[305]["vertices6"], before_lobe)
        np.testing.assert_array_equal(candidate[311]["vertices6"][:3, :3], before_patch)
        np.testing.assert_array_equal(candidate[311]["faces"], before_faces)
        np.testing.assert_array_equal(rows[311]["vertices6"], diaphragm_v6)
        self.assertIn(4, report["changed_vertex_ids"])
        self.assertEqual(report["moved_vertex_count"], len(report["changed_vertex_ids"]))
        tol = report["per_vertex"][1]["clearance_rounding_bound_m"]
        self.assertGreaterEqual(report["per_vertex"][1]["candidate_signed_distance_m"], .1-tol)
        self.assertLess(tol, 1e-6)
        self.assertTrue(report["reciprocal_patch_positions_unchanged"])
        self.assertTrue(report["lobes_unchanged_bytewise"])


    def test_free_diaphragm_sidewall_clearance_rejects_reciprocal_patch_faces(self):
        from numilab_human.resting_lung_edge_repair import _move_free_diaphragm_sidewall_clearance
        from numilab_human.resting_anatomy_interface_patch import normals

        xyz = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        faces = np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=np.int64)
        if signed_volume(xyz.astype(float), faces) < 0:
            faces = faces[:, [0,2,1]]
        lobe = np.column_stack((xyz, normals(xyz.astype(float), faces))).astype(np.float32)
        dxyz = np.asarray([[10,0,0],[11,0,0],[10,1,0]], dtype=np.float32)
        dfaces = np.asarray([[0,1,2]], dtype=np.int64)
        d = np.column_stack((dxyz, np.zeros_like(dxyz))).astype(np.float32)
        rows = {305: {"vertices6": lobe, "faces": faces},
                311: {"vertices6": d, "faces": dfaces}}
        before = rows[311]["vertices6"].copy()
        with self.assertRaisesRegex(ValueError, "itself a reciprocal patch face"):
            _move_free_diaphragm_sidewall_clearance(
                rows, interface_rows=[{"lung_stable_id":305,
                    "diaphragm_patch_face_start":0,"diaphragm_patch_face_count":1}],
                event_faces_by_lobe={305:[0]}, target_clearance_m=.1,
                max_vertex_displacement_m=.2)
        np.testing.assert_array_equal(rows[311]["vertices6"], before)


    def test_free_diaphragm_sidewall_clearance_enforces_displacement_bound(self):
        from numilab_human.resting_lung_edge_repair import _move_free_diaphragm_sidewall_clearance
        from numilab_human.resting_anatomy_interface_patch import normals

        xyz = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        faces = np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=np.int64)
        if signed_volume(xyz.astype(float), faces) < 0:
            faces = faces[:, [0,2,1]]
        lobe = np.column_stack((xyz, normals(xyz.astype(float), faces))).astype(np.float32)
        dxyz = np.asarray([[10,0,0],[11,0,0],[10,1,0],
                           [.1,.1,1.5],[.2,.2,.05],[.4,.1,1.5]], dtype=np.float32)
        dfaces = np.asarray([[0,1,2],[3,4,5]], dtype=np.int64)
        d = np.column_stack((dxyz, np.zeros_like(dxyz))).astype(np.float32)
        rows = {305: {"vertices6": lobe, "faces": faces},
                311: {"vertices6": d, "faces": dfaces}}
        before = rows[311]["vertices6"].copy()
        with self.assertRaisesRegex(ValueError, "exceeds displacement bound"):
            _move_free_diaphragm_sidewall_clearance(
                rows, interface_rows=[{"lung_stable_id":305,
                    "diaphragm_patch_face_start":0,"diaphragm_patch_face_count":1}],
                event_faces_by_lobe={305:[1]}, target_clearance_m=.1,
                max_vertex_displacement_m=.1)
        np.testing.assert_array_equal(rows[311]["vertices6"], before)


    def test_source_indices_reject_boolean_fractional_and_string_coercion(self):
        from numilab_human.resting_lung_edge_repair import _strict_source_index
        for value in (True, np.bool_(False), 1.5, 1.0, "1", float("nan")):
            with self.subTest(value=str(value)):
                with self.assertRaisesRegex(ValueError, "source indices"):
                    _strict_source_index(value)
        self.assertEqual(_strict_source_index(np.int64(7)), 7)

    def test_closest_surface_rejects_nonfinite_query_and_triangle(self):
        from numilab_human.resting_lung_edge_repair import _closest_point_on_oriented_triangles
        triangle = np.asarray([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    _closest_point_on_oriented_triangles([value, 0., 0.], triangle)
                invalid = triangle.copy(); invalid[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    _closest_point_on_oriented_triangles([0., 0., 1.], invalid)


if __name__ == "__main__":
    unittest.main()

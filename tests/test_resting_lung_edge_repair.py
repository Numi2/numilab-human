import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human.resting_lung_edge_repair import collapse_midpoint_edge


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


if __name__ == "__main__":
    unittest.main()

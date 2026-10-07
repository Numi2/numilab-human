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

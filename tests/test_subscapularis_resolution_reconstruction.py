import unittest

import numpy as np

from numilab_human.subscapularis_resolution_reconstruction import (
    collapse_subscapularis_resolution_edges,
)
from numilab_human.resting_lung_edge_repair import normals


class SubscapularisResolutionReconstructionTests(unittest.TestCase):
    @staticmethod
    def octahedron():
        xyz = np.asarray([
            [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0],
            [0, -1, 0], [0, 0, -1],
        ], dtype="<f4")
        faces = np.asarray([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
        ], dtype=np.int64)
        n = normals(xyz.astype(np.float64), faces).astype("<f4")
        v6 = np.column_stack((xyz, n)).astype("<f4")
        maps = np.asarray([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3],
                           [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]], dtype="<f4")
        return v6, faces, np.arange(len(faces), dtype=np.int64), maps

    def test_midpoint_simplification_preserves_topology_lineage_and_interpolates_maps(self):
        v, f, origins, maps = self.octahedron()
        vo, fo, oo, mo, steps = collapse_subscapularis_resolution_edges(
            v, f, origins, maps, [(0, 1)], resolution_m=1.5,
        )
        self.assertEqual(len(vo), 5)
        self.assertEqual(len(fo), 6)
        self.assertEqual(set(oo), {1, 2, 4, 5, 6, 7})
        self.assertEqual(len(steps), 1)
        merged = [i for i, row in enumerate(vo) if np.array_equal(row[:3], np.asarray([0.5, 0.0, 0.5], dtype="<f4"))]
        self.assertEqual(len(merged), 1)
        np.testing.assert_allclose(mo[merged[0]], np.asarray([0.7, 0.3], dtype="<f4"), atol=1.0e-7, rtol=0.0)
        self.assertFalse(steps[0]["weight_renormalization_performed"])
        self.assertEqual(steps[0]["candidate_topology"]["boundary_edges"], 0)
        self.assertEqual(steps[0]["candidate_topology"]["nonmanifold_edges"], 0)
        self.assertEqual(steps[0]["candidate_topology"]["orientation_error_edges"], 0)

    def test_attachment_lock_rejects_selected_core_edge(self):
        v, f, origins, maps = self.octahedron()
        with self.assertRaisesRegex(ValueError, "locked attachment/core"):
            collapse_subscapularis_resolution_edges(
                v, f, origins, maps, [(0, 1)], resolution_m=1.5,
                locked_source_vertex_ids=[1],
            )

    def test_declared_resolution_rejects_longer_edges(self):
        v, f, origins, maps = self.octahedron()
        with self.assertRaisesRegex(ValueError, "exceeds the declared"):
            collapse_subscapularis_resolution_edges(
                v, f, origins, maps, [(0, 1)], resolution_m=1.0,
            )

    def test_overlapping_selected_edges_are_rejected(self):
        v, f, origins, maps = self.octahedron()
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            collapse_subscapularis_resolution_edges(
                v, f, origins, maps, [(0, 1), (0, 2)], resolution_m=2.0,
            )

    def test_disjoint_edges_keep_initial_ids_after_compaction(self):
        v, f, origins, _ = self.octahedron()
        maps = np.eye(len(v), dtype="<f4")
        vo, fo, oo, mo, steps = collapse_subscapularis_resolution_edges(
            v, f, origins, maps, [(0, 1), (4, 5)], resolution_m=1.5)
        self.assertEqual(len(vo), 4)
        self.assertEqual([row["merged_source_vertex_ids"] for row in steps], [[0, 1], [4, 5]])
        for point, ids in [([0.5, 0, 0.5], [0, 1]), ([0, -0.5, -0.5], [4, 5])]:
            index = next(i for i, row in enumerate(vo) if np.array_equal(row[:3], point))
            expected = np.zeros(6, dtype="<f4"); expected[ids] = 0.5
            np.testing.assert_array_equal(mo[index], expected)

    def test_later_attachment_lock_uses_original_source_id(self):
        v, f, origins, maps = self.octahedron()
        with self.assertRaisesRegex(ValueError, "locked attachment/core"):
            collapse_subscapularis_resolution_edges(
                v, f, origins, maps, [(0, 1), (2, 3)], resolution_m=2,
                locked_source_vertex_ids=[2])

    def test_lossy_source_identifier_coercion_is_rejected(self):
        v, f, origins, maps = self.octahedron()
        for kwargs in [dict(faces=f.astype(float) + 0.25),
                       dict(face_origins=origins.astype(float) + 0.25),
                       dict(locked_source_vertex_ids=[1.25]),
                       dict(locked_source_vertex_ids=[True])]:
            values = dict(vertices6=v, faces=f, face_origins=origins,
                          canonical_weights=maps, selected_edges=[(0, 1)], resolution_m=1.5)
            values.update(kwargs)
            with self.assertRaisesRegex(ValueError, "integer source identifiers"):
                collapse_subscapularis_resolution_edges(**values)

    def test_source_arrays_survive_success_and_later_rejection_unchanged(self):
        for edges in [[(0, 1)], [(0, 1), (0, 2)]]:
            arrays = self.octahedron(); snapshots = [a.copy() for a in arrays]
            try:
                collapse_subscapularis_resolution_edges(*arrays, edges, resolution_m=2)
            except ValueError:
                self.assertEqual(edges, [(0, 1), (0, 2)])
            for original, snapshot in zip(arrays, snapshots):
                np.testing.assert_array_equal(original, snapshot)


if __name__ == "__main__":
    unittest.main()

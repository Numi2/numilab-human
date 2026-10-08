import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human.resting_lobe_fan_refinement import refine_shared_vertex_fans


def _tetra(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    if signed_volume(xyz.astype(np.float64), faces) < 0:
        faces = faces[:, [0, 2, 1]]
    normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1))
    return np.column_stack((xyz, normals)), faces


def _face_for_points(vertices6, faces, points):
    wanted = {tuple(np.asarray(point, dtype=np.float32)) for point in points}
    matches = []
    for face_index, face in enumerate(faces):
        actual = {tuple(vertices6[int(v), :3]) for v in face}
        if actual == wanted:
            matches.append(face_index)
    if len(matches) != 1:
        raise AssertionError(f"expected one triangle {wanted}, found {matches}")
    return matches[0]


class RestingLobeFanRefinementTests(unittest.TestCase):
    def _fixture(self):
        base = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        bottom = _tetra(base + [[0, 0, -1]])
        top = _tetra(base + [[0, 0, 1]])
        rows = {305: bottom, 308: top}
        for sid, offset in ((306, 10), (307, 20), (309, 30)):
            rows[sid] = _tetra([[offset, 0, 0], [offset+1, 0, 0],
                                [offset, 1, 0], [offset, 0, 1]])
        first = _face_for_points(bottom[0], bottom[1],
                                 [base[0], base[1], [0, 0, -1]])
        second = _face_for_points(top[0], top[1],
                                  [base[1], base[2], [0, 0, 1]])
        return rows, ((305, first), (308, second))

    def test_local_fan_is_closed_and_keeps_reciprocal_children_exact(self):
        rows, seeds = self._fixture()
        before = {sid: (v.copy(), f.copy()) for sid, (v, f) in rows.items()}
        refined, ancestry, report = refine_shared_vertex_fans(
            rows, seed_faces=seeds, expected_shared_vertex_m=(1, 0, 0))
        self.assertEqual(report["source_exact_reciprocal_face_group_count"], 1)
        self.assertEqual(report["candidate_exact_reciprocal_child_face_group_count"], 3)
        self.assertTrue(report["exact_reciprocal_child_geometry_and_winding_preserved"])
        self.assertTrue(report["original_source_positions_moved"] is False)
        self.assertEqual(report["split_edge_count_by_lobe"], {"305": 3, "308": 3})
        for sid in rows:
            v0, f0 = before[sid]
            v1, f1 = refined[sid]
            topo = topology_report(f1)
            self.assertEqual(topo["boundary_edge_count"], 0)
            self.assertEqual(topo["nonmanifold_edge_count"], 0)
            self.assertEqual(topo["orientation_error_edge_count"], 0)
            np.testing.assert_array_equal(v1[:len(v0), :3], v0[:, :3])
            self.assertAlmostEqual(
                signed_volume(v0[:, :3].astype(np.float64), f0),
                signed_volume(v1[:, :3].astype(np.float64), f1), places=12)
        for sid, fi in seeds:
            self.assertEqual(len(ancestry[sid][fi]), 3)

    def test_wrong_pinned_vertex_fails_closed(self):
        rows, seeds = self._fixture()
        with self.assertRaisesRegex(ValueError, "pinned point"):
            refine_shared_vertex_fans(
                rows, seed_faces=seeds, expected_shared_vertex_m=(1, 1, 1))


if __name__ == "__main__":
    unittest.main()


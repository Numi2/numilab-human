import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
from numilab_human.resting_anatomy_conforming_refinement import _split_triangle, edge_key
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

    def test_float32_edge_midpoint_cannot_form_endpoint_midpoint_endpoint_sliver(self):
        # This is the exact geometry that previously produced a 1.76nm child
        # when rounded Float32 midpoints were treated as non-collinear ears.
        a = np.asarray([-0.0013838123995810747, 0.04402492940425873,
                        -0.0619685985147953], dtype=np.float32)
        b = np.asarray([-0.0005217144498601556, 0.043026700615882874,
                        -0.06243239343166351], dtype=np.float32)
        c = np.asarray([-0.0017699382733553648, 0.04414953663945198,
                        -0.06242293119430542], dtype=np.float32)
        midpoint_ac = ((a.astype(np.float64)+c.astype(np.float64))*0.5).astype(np.float32)
        midpoint_ab = ((a.astype(np.float64)+b.astype(np.float64))*0.5).astype(np.float32)
        positions = np.asarray([a, b, c, midpoint_ac, midpoint_ab], dtype=np.float32)
        vertices6 = np.column_stack((positions, np.tile([0, 0, 1], (5, 1)).astype(np.float32)))
        splits = {edge_key(0, 1), edge_key(0, 2)}
        midpoint_ids = {edge_key(0, 2): 3, edge_key(0, 1): 4}
        children = _split_triangle(vertices6, (1, 2, 0), splits, midpoint_ids)
        self.assertEqual(len(children), 3)
        forbidden = ({0, 1, 4}, {0, 2, 3})
        altitudes = []
        for child in children:
            ids = set(map(int, child))
            self.assertNotIn(ids, forbidden)
            tri = positions[np.asarray(child)].astype(np.float64)
            edges = [np.linalg.norm(tri[1]-tri[2]), np.linalg.norm(tri[2]-tri[0]),
                     np.linalg.norm(tri[0]-tri[1])]
            twice_area = np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))
            altitudes.extend(twice_area/edge for edge in edges)
        self.assertGreaterEqual(min(altitudes), 128e-9)

    def test_wrong_pinned_vertex_fails_closed(self):
        rows, seeds = self._fixture()
        with self.assertRaisesRegex(ValueError, "pinned point"):
            refine_shared_vertex_fans(
                rows, seed_faces=seeds, expected_shared_vertex_m=(1, 1, 1))


if __name__ == "__main__":
    unittest.main()


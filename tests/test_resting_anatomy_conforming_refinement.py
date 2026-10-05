import unittest

import numpy as np

from numilab_human.resting_anatomy_conforming_refinement import (
    edge_key,
    interface_face_mates,
    propagate_interface_edges,
    refine_surface_edges,
)
from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report


class ConformingRefinementTests(unittest.TestCase):
    def test_closed_tetra_edge_refinement_preserves_volume_and_orientation(self):
        vertices = np.asarray([
            [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1],
        ], dtype=np.float32)
        faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
        before = abs(signed_volume(vertices[:, :3].astype(np.float64), faces))
        refined_v, refined_f, mapping, detail = refine_surface_edges(
            vertices, faces, {edge_key(0, 1), edge_key(1, 2), edge_key(2, 0)}
        )
        after = abs(signed_volume(refined_v[:, :3].astype(np.float64), refined_f))
        topo = topology_report(refined_f)
        self.assertEqual(len(refined_v), 7)
        self.assertEqual(len(refined_f), 10)
        self.assertEqual(len(mapping[0]), 4)
        self.assertEqual(detail["affected_face_count"], 4)
        self.assertAlmostEqual(after, before, places=12)
        self.assertEqual(topo["boundary_edge_count"], 0)
        self.assertEqual(topo["nonmanifold_edge_count"], 0)
        self.assertEqual(topo["orientation_error_edge_count"], 0)

    def test_reciprocal_split_propagates_from_either_surface(self):
        xyz = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1))
        vertices = np.concatenate([xyz, normals], axis=1)
        lobe_faces = np.asarray([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int64)
        diaphragm_faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
        rows = {
            305: {"vertices6": vertices.copy(), "faces": lobe_faces},
            311: {"vertices6": vertices.copy(), "faces": diaphragm_faces},
        }
        receipt = {"provenance": {"diaphragm_lung_interface": {"interface_rows": [{
            "lung_stable_id": 305, "registered_lung_face_index_ranges": [[0, 1]],
            "diaphragm_patch_face_start": 0, "diaphragm_patch_face_count": 1,
        }]}}}
        mates, reverse = interface_face_mates(rows, receipt)
        self.assertEqual(mates[(305, 0)], (311, 0))
        self.assertEqual(reverse[(311, 0)], (305, 0))

        split = propagate_interface_edges(rows, receipt, {311: {edge_key(0, 1)}})
        self.assertIn(305, split)
        self.assertEqual(len(split[305]), 1)
        mapped_lobe = next(iter(split[305]))
        mapped_diaphragm = next(iter(split[311]))
        self.assertEqual(
            {tuple(rows[305]["vertices6"][i, :3]) for i in mapped_lobe},
            {tuple(rows[311]["vertices6"][i, :3]) for i in mapped_diaphragm},
        )

        refined = {}
        for sid, edges in split.items():
            refined[sid] = refine_surface_edges(rows[sid]["vertices6"], rows[sid]["faces"], edges)
        self.assertEqual(refined[305][2][0], refined[311][2][0])

        def oriented(surface, child_ids):
            vertices6, faces = surface[0], surface[1]
            return {tuple(sorted(tuple(vertices6[int(i), :3]) for i in faces[child])) for child in child_ids}

        self.assertEqual(oriented(refined[305], refined[305][2][0]), oriented(refined[311], refined[311][2][0]))


if __name__ == "__main__":
    unittest.main()

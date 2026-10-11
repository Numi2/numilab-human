import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import signed_volume
from numilab_human.surface_precision_retriangulation import (
    build_surface_adjacency,
    flip_interior_edge,
    validate_closed_oriented_surface,
)


class SurfacePrecisionRetriangulationTests(unittest.TestCase):
    @staticmethod
    def _sliver_octahedron():
        # A closed oriented octahedral surface with one exact Float32 sliver.
        # Flipping edge 0--1 replaces the 1e-9 m altitude face by two broad
        # triangles while retaining every source coordinate and face slot.
        vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0e-9, 0.0],
            [2.0, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.asarray([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
        ], dtype=np.int64)
        if signed_volume(vertices.astype(np.float64), faces) < 0.0:
            faces = faces[:, [0, 2, 1]]
        return vertices, faces

    def test_flip_preserves_positions_topology_orientation_and_two_parent_lineage(self):
        vertices, faces = self._sliver_octahedron()
        source_vertices = vertices.copy()
        source_faces = faces.copy()
        adjacency = build_surface_adjacency(faces)
        pair = adjacency.edge_faces[(0, 1)]
        origins = np.arange(len(faces), dtype=np.int64) + 700
        output, report = flip_interior_edge(
            vertices,
            faces,
            edge=(0, 1),
            face_indices=pair,
            minimum_altitude_m=1.0e-3,
            adjacency=adjacency,
            face_origins=origins,
        )

        np.testing.assert_array_equal(vertices, source_vertices)
        np.testing.assert_array_equal(faces, source_faces)
        self.assertEqual(output.shape, faces.shape)
        untouched = sorted(set(range(len(faces))) - set(pair))
        np.testing.assert_array_equal(output[untouched], faces[untouched])
        self.assertEqual(report["vertices_modified"], False)
        self.assertGreater(report["candidate_pair_minimum_altitude_m"], 1.0e-3)
        self.assertGreater(
            report["candidate_pair_minimum_altitude_m"],
            report["source_pair_minimum_altitude_m"],
        )
        for row in report["output_face_lineage"]:
            self.assertEqual(row["source_parent_face_indices"], list(pair))
            self.assertEqual(row["source_parent_face_origins"], [origins[i] for i in pair])
        topology = validate_closed_oriented_surface(vertices, output)
        self.assertEqual(topology["closed_oriented"], True)
        self.assertEqual(topology["vertex_count"], 6)
        self.assertEqual(topology["edge_count"], 12)
        self.assertEqual(topology["face_count"], 8)
        self.assertEqual(topology["connected_component_count"], 1)
        self.assertEqual(topology["euler_characteristic"], 2)
        self.assertEqual(topology["zero_area_face_count"], 0)

        old_volume = signed_volume(vertices.astype(np.float64), faces)
        new_volume = signed_volume(vertices.astype(np.float64), output)
        self.assertAlmostEqual(
            new_volume - old_volume,
            report["signed_volume_delta_m3"],
            delta=1.0e-14,
        )
        self.assertLess(report["oriented_area_vector_delta_norm_m2"], 1.0e-14)
        self.assertTrue(all(value > 0.0 for value in report["local_orientation_alignment_cosines"]["candidate_faces"]))
        self.assertEqual(report["face_count_preserved"], True)
        self.assertEqual(report["replacement_edge"], [2, 4])
        updates = {tuple(row["edge"]): row for row in report["edge_incidence_updates"]}
        self.assertEqual(updates[(0, 1)]["after_face_indices"], [])
        self.assertEqual(updates[(2, 4)]["after_face_indices"], list(pair))

    def test_in_place_mode_writes_only_the_validated_face_pair(self):
        vertices, source_faces = self._sliver_octahedron()
        working_faces = source_faces.copy()
        adjacency = build_surface_adjacency(source_faces)
        output, report = flip_interior_edge(
            vertices, working_faces, edge=(0, 1), face_indices=adjacency.edge_faces[(0, 1)],
            minimum_altitude_m=1.0e-3, adjacency=adjacency, in_place=True,
        )
        self.assertIs(output, working_faces)
        self.assertFalse(np.array_equal(working_faces, source_faces))
        np.testing.assert_array_equal(vertices, self._sliver_octahedron()[0])
        self.assertAlmostEqual(
            signed_volume(vertices.astype(np.float64), working_faces)
            - signed_volume(vertices.astype(np.float64), source_faces),
            report["signed_volume_delta_m3"],
            delta=1.0e-14,
        )

        rejected_faces = source_faces.copy()
        rejected_before = rejected_faces.copy()
        with self.assertRaisesRegex(ValueError, "minimum altitude bound"):
            flip_interior_edge(
                vertices, rejected_faces, edge=(0, 1), face_indices=adjacency.edge_faces[(0, 1)],
                minimum_altitude_m=2.0, adjacency=adjacency, in_place=True,
            )
        np.testing.assert_array_equal(rejected_faces, rejected_before)

    def test_below_requested_altitude_fails_without_mutation(self):
        vertices, faces = self._sliver_octahedron()
        vertices_before, faces_before = vertices.copy(), faces.copy()
        adjacency = build_surface_adjacency(faces)
        with self.assertRaisesRegex(ValueError, "minimum altitude bound"):
            flip_interior_edge(
                vertices, faces, edge=(0, 1), face_indices=adjacency.edge_faces[(0, 1)],
                minimum_altitude_m=2.0, adjacency=adjacency,
            )
        np.testing.assert_array_equal(vertices, vertices_before)
        np.testing.assert_array_equal(faces, faces_before)

    def test_boundary_edge_and_same_direction_pair_are_rejected(self):
        vertices, faces = self._sliver_octahedron()
        open_faces = faces[:-1].copy()
        open_adjacency = build_surface_adjacency(open_faces)
        with self.assertRaisesRegex(ValueError, "exactly the two declared incident faces"):
            flip_interior_edge(
                vertices, open_faces, edge=(1, 5), face_indices=(4, 5),
                minimum_altitude_m=1.0e-3, adjacency=open_adjacency,
            )

        reversed_faces = faces.copy()
        reversed_faces[3] = reversed_faces[3, ::-1]
        reversed_adjacency = build_surface_adjacency(reversed_faces)
        with self.assertRaisesRegex(ValueError, "oppositely"):
            flip_interior_edge(
                vertices, reversed_faces, edge=(0, 1), face_indices=(0, 3),
                minimum_altitude_m=1.0e-3, adjacency=reversed_adjacency,
            )

    def test_existing_replacement_diagonal_is_rejected(self):
        vertices, faces = self._sliver_octahedron()
        diagonal_faces = np.vstack((faces, np.asarray([[2, 4, 5]], dtype=np.int64)))
        diagonal_adjacency = build_surface_adjacency(diagonal_faces)
        with self.assertRaisesRegex(ValueError, "replacement diagonal already exists"):
            flip_interior_edge(
                vertices, diagonal_faces, edge=(0, 1), face_indices=(0, 3),
                minimum_altitude_m=1.0e-3, adjacency=diagonal_adjacency,
            )

    def test_extra_nonfacial_common_neighbor_does_not_block_valid_flip(self):
        # A tetrahedral boundary with one face subdivided around vertex 4.
        # The edge 0--1 has an extra common neighbor 2 outside its two selected
        # incident faces. That is valid for a 2-to-2 flip; it is only an edge-
        # collapse link-condition concern.
        vertices = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.5, -0.3, 1.0], [0.5, 1.0e-9, 0.0],
        ], dtype=np.float32)
        faces = np.asarray([
            [0, 4, 2], [2, 4, 1], [1, 4, 0],
            [1, 0, 3], [2, 1, 3], [0, 2, 3],
        ], dtype=np.int64)
        adjacency = build_surface_adjacency(faces)
        pair = adjacency.edge_faces[(0, 1)]
        self.assertEqual(len(adjacency.vertex_neighbors[0] & adjacency.vertex_neighbors[1]), 3)
        output, report = flip_interior_edge(
            vertices, faces, edge=(0, 1), face_indices=pair,
            minimum_altitude_m=1.0e-6, adjacency=adjacency,
        )
        self.assertEqual(report["extra_common_neighbors_not_used_by_selected_faces"], [2])
        self.assertGreater(report["candidate_pair_minimum_altitude_m"], 1.0e-6)
        self.assertEqual(validate_closed_oriented_surface(vertices, output)["closed_oriented"], True)

    def test_unsigned_face_origins_preserve_lineage_and_reject_int64_overflow(self):
        vertices, faces = self._sliver_octahedron()
        adjacency = build_surface_adjacency(faces)
        origins = np.arange(len(faces), dtype=np.uint32) + np.uint32(700)
        pair = adjacency.edge_faces[(0, 1)]
        output, report = flip_interior_edge(
            vertices, faces, edge=(0, 1), face_indices=pair,
            minimum_altitude_m=1.0e-3, adjacency=adjacency, face_origins=origins,
        )
        self.assertEqual(output.shape, faces.shape)
        self.assertEqual(
            [row["source_parent_face_origins"] for row in report["output_face_lineage"]],
            [[int(origins[i]) for i in pair], [int(origins[i]) for i in pair]],
        )

        oversized = np.arange(len(faces), dtype=np.uint64)
        oversized[-1] = np.uint64(1 << 63)
        with self.assertRaisesRegex(ValueError, "signed 64-bit range"):
            flip_interior_edge(
                vertices, faces, edge=(0, 1), face_indices=pair,
                minimum_altitude_m=1.0e-3, adjacency=adjacency, face_origins=oversized,
            )

    def test_full_validator_rejects_open_and_zero_area_surfaces(self):
        vertices, faces = self._sliver_octahedron()
        with self.assertRaisesRegex(ValueError, "expected two"):
            validate_closed_oriented_surface(vertices, faces[:-1])
        zero_area = faces.copy()
        zero_area[0] = [0, 1, 5]
        zero_vertices = vertices.copy()
        zero_vertices[5] = [0.25, 0.0, 0.0]
        # The changed face lies on the x axis; global validation fails closed.
        with self.assertRaisesRegex(ValueError, "zero-area"):
            validate_closed_oriented_surface(zero_vertices, zero_area)


if __name__ == "__main__":
    unittest.main()

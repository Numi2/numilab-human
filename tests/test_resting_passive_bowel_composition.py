import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from numilab_human.resting_passive_viscera import (
    _area_weighted_vertex_normals,
    _owned_surface_rows,
    compose_pylorus_stomach_arrangement,
)


class PassiveBowelCompositionTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        self.faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        self.owners = np.asarray([398, 399], dtype=np.int32)

    def test_area_weighted_normal_field_is_finite_unit_and_oriented(self):
        normals = _area_weighted_vertex_normals(self.vertices, self.faces)
        self.assertEqual(normals.shape, self.vertices.shape)
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-7)
        np.testing.assert_array_equal(normals, np.tile([0.0, 0.0, 1.0], (4, 1)))

    def test_owner_rows_partition_faces_and_preserve_shared_coordinates(self):
        normals = _area_weighted_vertex_normals(self.vertices, self.faces)
        rows = _owned_surface_rows(self.vertices, normals, self.faces, self.owners, range(398, 400))
        self.assertEqual(set(rows), {398, 399})
        self.assertEqual(rows[398]["faces"].shape, (1, 3))
        self.assertEqual(rows[399]["faces"].shape, (1, 3))
        self.assertEqual(rows[398]["vertices6"].shape, (3, 6))
        self.assertEqual(rows[399]["vertices6"].shape, (3, 6))
        # Aggregate seam positions and their normals are copied identically
        # into each local NHANAT owner row.
        for source_vertex in (1, 2):
            local_a = int(np.flatnonzero(np.all(rows[398]["vertices6"][:, :3] == self.vertices[source_vertex], axis=1))[0])
            local_b = int(np.flatnonzero(np.all(rows[399]["vertices6"][:, :3] == self.vertices[source_vertex], axis=1))[0])
            self.assertEqual(rows[398]["vertices6"][local_a].tobytes(),
                             rows[399]["vertices6"][local_b].tobytes())

    def test_owner_map_must_be_complete_and_exact(self):
        normals = _area_weighted_vertex_normals(self.vertices, self.faces)
        with self.assertRaisesRegex(ValueError, "owner IDs are incomplete"):
            _owned_surface_rows(self.vertices, normals, self.faces, self.owners, range(398, 401))

    def test_zero_area_faces_fail_closed_before_serialization(self):
        degenerate = np.asarray([[0, 1, 1]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "zero-area"):
            _area_weighted_vertex_normals(self.vertices, degenerate)

    def test_pylorus_composition_fails_closed_on_missing_source_evidence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "must-not-be-created"
            with self.assertRaisesRegex(ValueError, "dependency is missing"):
                compose_pylorus_stomach_arrangement(
                    base_payload=root / "base.nhanatomy",
                    base_receipt=root / "base-receipt.json",
                    arranged_stomach=root / "stomach.npz",
                    arrangement_report=root / "arrangement.json",
                    interface_audit=root / "interface.json",
                    neighbor_equivalence_audit=root / "neighbors.json",
                    neighbor_screen=root / "screen.json",
                    output=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

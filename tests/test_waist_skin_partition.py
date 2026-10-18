import struct
import unittest

import numpy as np

from numilab_human.waist_skin_partition import (
    PlaneCutError,
    emit_nhskin_abi5_half,
    split_surface_by_signed_field,
)


def _cube():
    points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return points, triangles


class WaistSkinPartitionTests(unittest.TestCase):
    def test_owner_scalar_cut_caps_both_halves_and_interpolates_attributes(self):
        points, triangles = _cube()
        owner_field = points[:, 2] + 0.15 * points[:, 0] * points[:, 1]
        weights = np.column_stack((0.5 + 0.1 * points[:, 0],
                                   0.5 - 0.1 * points[:, 0]))
        positive, negative = split_surface_by_signed_field(
            points, triangles, owner_field,
            attributes={"full_binding_weights": weights},
        )
        for half in (positive, negative):
            self.assertEqual(len(half.boundary_edges), 0)
            self.assertTrue(np.any(half.triangle_kind == 1))
            self.assertTrue(np.allclose(
                half.attributes["full_binding_weights"].sum(axis=1), 1.0
            ))
            self.assertEqual(len(half.cap_loop_projection_residuals), 1)
            self.assertTrue(np.all(half.triangle_source_face[half.triangle_kind == 0] >= 0))
            self.assertTrue(np.all(half.triangle_source_face[half.triangle_kind == 1] == -1))

    def test_existing_boundaries_must_be_declared_exactly(self):
        points, triangles = _cube()
        open_faces = triangles[~np.all(np.isin(triangles, [4, 5, 6, 7]), axis=1)]
        declared = {(4, 5), (5, 6), (6, 7), (4, 7)}
        with self.assertRaisesRegex(PlaneCutError, "boundary differs"):
            split_surface_by_signed_field(points, open_faces, points[:, 2])
        positive, negative = split_surface_by_signed_field(
            points, open_faces, points[:, 2],
            allowed_existing_boundary_edges=declared,
        )
        self.assertEqual(len(positive.boundary_edges), 4)
        self.assertEqual(len(negative.boundary_edges), 0)

    def test_emitter_pads_diagnostic_indices_with_valid_zero_weight_row(self):
        points, triangles = _cube()
        weights = np.full((len(points), 2), 0.5, dtype=np.float64)
        positive, _ = split_surface_by_signed_field(
            points, triangles, points[:, 2],
            attributes={
                "full_binding_weights": weights,
                "registered_rest_normals": np.tile([0.0, 0.0, 1.0], (len(points), 1)),
            },
        )
        binding_rows = b"".join(
            struct.pack("<I8f", body, 0, 0, 0, 0, 0, 0, 1, 1)
            for body in (0, 1)
        )
        header = struct.pack(
            "<8s5I32s", b"NHSKIN1\0", 5, 2, len(points), triangles.size, 0, b"x" * 32
        )
        vertex_dtype = np.dtype([
            ("p", "<f4", (3,)), ("n", "<f4", (3,)),
            ("bi", "<u4", (4,)), ("bw", "<f4", (4,)),
        ])
        vertices = np.zeros(len(points), dtype=vertex_dtype)
        vertices["p"] = points
        vertices["n"] = [0, 0, 1]
        vertices["bi"][:, 0:2] = [0, 1]
        vertices["bw"][:, 0:2] = 0.5
        raw = (
            header + binding_rows + vertices.tobytes()
            + np.asarray(triangles, dtype="<u4").reshape(-1).tobytes()
            + np.asarray(weights, dtype="<f4").tobytes()
        )
        poses = {
            0: np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
            1: np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
        }
        skin = {
            "bindings": [
                (body, np.zeros(3), np.array([0, 0, 0, 1]), 1.0)
                for body in (0, 1)
            ]
        }
        payload, _ = emit_nhskin_abi5_half(
            raw, positive, {"core_poses": poses}, skin, {0}
        )
        rows = np.frombuffer(
            payload, dtype=vertex_dtype, count=len(positive.vertices), offset=60 + 36
        )
        # C++ loader checks every diagnostic index even when its weight is zero.
        self.assertTrue(np.all(rows["bi"] < 1))
        self.assertTrue(np.all(rows["bi"][:, 1:] == 0))
        self.assertTrue(np.all(rows["bw"][:, 1:] == 0))
        self.assertTrue(np.allclose(rows["bw"].sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()

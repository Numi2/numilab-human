"""Load-only carrier regression checks; these do not qualify anatomy."""
import dataclasses
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from numilab_human.passive_attachment_composition import (
    _biceps_row_arrays,
    _read_nhtiss4,
    _row_slices,
)
from numilab_human.passive_tissue_parent_carriers import (
    CarrierRow,
    encode_carriers,
    validate_compiled_core_weights,
)


class CarrierTests(unittest.TestCase):
    def test_keyed_compiled_partition_keeps_source_and_cast_guards_separate(self):
        gids = np.array([30, 10, 20], dtype=np.uint64)
        source = np.array([[.2, .3, .5], [.1, .2, .7], [.6, .2, .2]], dtype=np.float64)
        active = np.array([20, 30], dtype=np.uint64)
        compiled = source[[2, 0]].astype(np.float32)
        for field in (compiled, compiled.astype(np.float64)):
            proof = validate_compiled_core_weights(gids, source, [91, 92, 93],
                                                    active, field, [91, 92, 93])
            self.assertTrue(proof["compiled_rows_equal_keyed_source_float32_casts"])
            self.assertGreater(proof["compiled_partition_max_error_diagnostic"], 1e-12)
            self.assertFalse(proof["weights_modified"])
        with self.assertRaisesRegex(ValueError, "source Float64"):
            validate_compiled_core_weights(gids, source.astype(np.float32).astype(np.float64),
                                           [91, 92, 93], active, compiled, [91, 92, 93])
        with self.assertRaisesRegex(ValueError, "exact keyed"):
            validate_compiled_core_weights(gids, source, [91, 92, 93],
                                           active, compiled[::-1].copy(), [91, 92, 93])
        with self.assertRaisesRegex(ValueError, "core-body"):
            validate_compiled_core_weights(gids, source, [91, 92, 93],
                                           active, compiled, [93, 92, 91])
        with self.assertRaisesRegex(ValueError, "duplicate GID"):
            validate_compiled_core_weights(gids, source, [91, 92, 93],
                                           np.array([20, 20]), compiled, [91, 92, 93])

    def row(self, sid=132, normal=1):
        v = np.array([[0, 0, 0, 0, 0, normal],
                      [1, 0, 0, 0, 0, normal],
                      [0, 1, 0, 0, 0, normal]], dtype=np.float32)
        bindings = b"".join(struct.pack("<I8f", i, 0, 0, 0, 0, 0, 0, 1, 1)
                            for i in range(6))
        weights = np.tile(np.arange(1, 7, dtype=float)/21, (3, 1))
        return CarrierRow(sid, v, np.array([[0, 1, 2]]), bindings, weights)

    def encode(self, rows):
        return encode_carriers(rows, registration_fingerprint=42, source_digest=b"x"*32)

    def test_full_field_survives_existing_reader_and_parent_compilation(self):
        rows = [self.row(), self.row(134, -1)]
        payload, surfaces, proof = self.encode(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.nhtissue"
            path.write_bytes(payload)
            data = _read_nhtiss4(path)
            self.assertEqual(data["fingerprint"], 42)
            self.assertEqual(data["source_digest"], b"x"*32)
            compiled = []
            for row, rec, surface in zip(rows, data["records"], surfaces):
                decoded = _biceps_row_arrays(_row_slices(data, rec))
                self.assertEqual(len(decoded["vertices6"]), 18)
                self.assertEqual(decoded["faces"].tolist(), [[0, 6, 12]])
                field = []
                for original, terms in enumerate(surface["vertices"]):
                    self.assertEqual(len(terms), 6)  # Not nearest-four truncation.
                    influences = []
                    for source_index, sid, vertex, coefficient in terms:
                        self.assertEqual((source_index, sid), (0, row.stable_id))
                        np.testing.assert_array_equal(decoded["vertices6"][vertex], row.vertices6[original])
                        lane = decoded["binding_indices"][vertex, 0]
                        weight = np.float32(coefficient * float(decoded["weights"][vertex, 0]))
                        influences.append((int(lane), weight))
                    np.testing.assert_array_equal([x[1] for x in influences], row.weights[original].astype(np.float32))
                    field.append(influences)
                compiled.append(field)
            self.assertEqual(compiled[0], compiled[1])
            self.assertEqual(proof["compiled_positive_influences"], 36)

    def test_reject_triangle_collapsed_by_storage_rounding(self):
        row = self.row()
        v = row.vertices6.astype(np.float64)
        v[:, :3] = [[1, 0, 0], [1 + 2**-25, 0, 0], [1, 1, 0]]
        self.assertGreater(np.linalg.norm(np.cross(v[1, :3]-v[0, :3],
                                                  v[2, :3]-v[0, :3])), 0)
        with self.assertRaisesRegex(ValueError, "exactly degenerate triangle"):
            self.encode([dataclasses.replace(row, vertices6=v)])

    def test_reject_distinct_collinear_stored_vertices(self):
        row = self.row()
        v = row.vertices6.copy()
        v[:, :3] = [[0, 0, 0], [1, 1, 0], [2, 2, 0]]
        with self.assertRaisesRegex(ValueError, "exactly degenerate triangle"):
            self.encode([dataclasses.replace(row, vertices6=v)])

    def test_accept_nonzero_subnormal_stored_triangle(self):
        row = self.row()
        v = row.vertices6.copy()
        v[:, :3] = [[0, 0, 0], [2**-149, 0, 0], [0, 2**-149, 0]]
        payload, _, _ = self.encode([dataclasses.replace(row, vertices6=v)])
        self.assertTrue(payload)

    def test_shared_points_reject_different_fields(self):
        a, b = self.row(), self.row(134)
        w = b.weights.copy()
        w[0, :2] = w[0, :2][::-1]
        with self.assertRaisesRegex(ValueError, "different shared fields"):
            self.encode([a, dataclasses.replace(b, weights=w)])

    def test_shared_signed_zero_requires_identical_stored_coordinates(self):
        a, b = self.row(), self.row(134)
        v = b.vertices6.copy()
        v[0, 0] = -0.0
        with self.assertRaisesRegex(ValueError, "stored bits"):
            self.encode([a, dataclasses.replace(b, vertices6=v)])

    def test_reject_invalid_inputs_without_normalizing(self):
        r = self.row()
        cases = [
            (dataclasses.replace(r, weights=r.weights*0.9), "partition"),
            (dataclasses.replace(r, binding_bytes=r.binding_bytes[:36]*6), "repeated body"),
            (dataclasses.replace(r, vertices6=r.vertices6*0), "normals"),
            (dataclasses.replace(r, faces=np.array([[0, 1, 3]])), "faces"),
            (dataclasses.replace(r, faces=np.array([[0., 1., 2.]])), "integer"),
        ]
        tiny = r.weights.copy()
        tiny[:, 1] += tiny[:, 0]
        tiny[:, 0] = 1e-100
        cases.append((dataclasses.replace(r, weights=tiny), "underflows"))
        for bad, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.encode([bad])
        changed_bindings = struct.pack("<I8f", 99, 0, 0, 0, 0, 0, 0, 1, 1)+r.binding_bytes[36:]
        with self.assertRaisesRegex(ValueError, "canonical binding"):
            self.encode([r, dataclasses.replace(self.row(134), binding_bytes=changed_bindings)])

    def test_sparse_cardinal_weights_remain_exact(self):
        r = self.row()
        w = np.zeros_like(r.weights)
        w[:, 5] = 1
        _payload, surfaces, proof = self.encode([dataclasses.replace(r, weights=w)])
        self.assertEqual(proof["source_vertices"], 3)
        self.assertEqual(surfaces[0]["vertices"][2], [[0, 132, 2, 1.0]])


if __name__ == "__main__":
    unittest.main()

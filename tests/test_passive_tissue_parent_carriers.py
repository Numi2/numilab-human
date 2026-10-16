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
    merge_carrier_payloads,
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

    def test_merge_preserves_shared_source_rows_and_local_parent_addresses(self):
        base, base_terms, _ = self.encode([self.row(131), self.row(133)])
        extra, extra_terms, _ = self.encode([self.row(109), self.row(135)])
        merged, proof = merge_carrier_payloads(base, extra)
        with tempfile.TemporaryDirectory() as directory:
            decoded = []
            for name, payload in (("base", base), ("extra", extra), ("merged", merged)):
                path = Path(directory)/(name+".nhtissue")
                path.write_bytes(payload)
                data = _read_nhtiss4(path)
                decoded.append({int(r[6]): _row_slices(data, r) for r in data["records"]})
            for original in decoded[:2]:
                for sid, row in original.items():
                    for field in ("binding_bytes", "vertex_bytes", "local_faces"):
                        self.assertEqual(row[field], decoded[2][sid][field])
            for terms in base_terms+extra_terms:
                actual = _biceps_row_arrays(decoded[2][terms["stable_id"]])
                for vertex_terms in terms["vertices"]:
                    for _, sid, local, _ in vertex_terms:
                        self.assertEqual(sid, terms["stable_id"])
                        self.assertLess(local, len(actual["vertices6"]))
        self.assertEqual(proof["retained_stable_ids"], [131, 133])
        self.assertEqual(proof["appended_stable_ids"], [109, 135])
        self.assertFalse(proof["physical_state_added"])

    def test_merge_replacement_requires_exact_explicit_scope(self):
        base, _, _ = self.encode([self.row(131), self.row(133)])
        addition, _, _ = self.encode([self.row(133, -1)])
        with self.assertRaisesRegex(ValueError, "explicit replacement"):
            merge_carrier_payloads(base, addition)
        with self.assertRaisesRegex(ValueError, "explicit replacement"):
            merge_carrier_payloads(base, addition, replace_stable_ids=(131, 133))
        merged, proof = merge_carrier_payloads(base, addition, replace_stable_ids=(133,))
        self.assertNotEqual(merged, base)
        self.assertEqual(proof["retained_stable_ids"], [131])
        self.assertEqual(proof["replaced_stable_ids"], [133])

    def test_merge_rejects_source_mismatch_and_corrupt_allocations(self):
        base, _, _ = self.encode([self.row(131)])
        addition, _, _ = self.encode([self.row(109)])
        for offset, message in ((28, "registration"), (32, "source")):
            bad = bytearray(addition)
            bad[offset] ^= 1
            with self.assertRaisesRegex(ValueError, message):
                merge_carrier_payloads(base, bytes(bad))
        with self.assertRaisesRegex(ValueError, "byte ranges"):
            merge_carrier_payloads(base, addition[:-1])
        bad = bytearray(addition)
        struct.pack_into("<I", bad, 64, 1)
        with self.assertRaisesRegex(ValueError, "allocation"):
            merge_carrier_payloads(base, bytes(bad))
        bad = bytearray(addition)
        struct.pack_into("<I", bad, len(bad)-4, 2**32-1)
        with self.assertRaisesRegex(ValueError, "index outside"):
            merge_carrier_payloads(base, bytes(bad))

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


class CarrierRegistrationReemissionTests(unittest.TestCase):
    def fixture(self):
        # Existing ABI with one triangular load-only row and one Core influence.
        bindings = struct.pack("<I8f", 91, 1, 0, 0, 0, 0, 0, 1, 1)
        fresh = struct.pack("<I8f", 91, 2, 0, 0, 0, 0, 0, 1, 1)
        vertices = np.zeros(3, dtype=np.dtype([
            ("vertex", "<f4", (6,)), ("binding", "<u4", (4,)),
            ("weight", "<f4", (4,))]))
        vertices["vertex"] = [[0,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,0,1]]
        vertices["weight"][:,0] = 1
        raw = (struct.pack("<8s6I32s",b"NHTISS4\0",5,1,1,3,3,42,b"x"*32)
               + struct.pack("<8I",0,1,0,3,0,3,132,1) + bindings
               + vertices.tobytes() + struct.pack("<3I",0,1,2))
        return raw, {"previous_fingerprint": 42, "fingerprint": 43,
                         "source_digest": b"x"*32, "previous_bindings": {91:bindings},
                         "bindings": {91:fresh}, "coordinate_frame": "registered_source_atlas"}

    def test_changes_only_binding_and_registration(self):
        from numilab_human.passive_tissue_parent_carriers import (
            _retarget_carrier_binding_bytes,
        )
        raw,args = self.fixture()
        out,proof = _retarget_carrier_binding_bytes(raw,**args)
        self.assertEqual(out[64:96],raw[64:96])
        self.assertEqual(out[132:],raw[132:])
        self.assertEqual(out[96:132],args["bindings"][91])
        self.assertEqual(struct.unpack_from("<I",out,28)[0],43)
        self.assertTrue(proof["geometry_normals_weights_indices_and_rows_byte_exact"])

    def test_core_local_stays_exact_with_changed_atlas_registration(self):
        from numilab_human.passive_tissue_parent_carriers import (
            _retarget_carrier_binding_bytes,
        )
        raw,args = self.fixture()
        identity=struct.pack("<I8f",91,0,0,0,0,0,0,1,1)
        raw=raw[:96]+identity+raw[132:]
        args["coordinate_frame"]="core_inertial_local"
        out,proof=_retarget_carrier_binding_bytes(raw,**args)
        self.assertEqual(out[64:],raw[64:])
        self.assertEqual(proof["changed_binding_records"],[])

    def test_reject_source_frame_or_identity_drift(self):
        from numilab_human.passive_tissue_parent_carriers import (
            _retarget_carrier_binding_bytes,
        )
        raw,args=self.fixture()
        for delta in [
            {"previous_fingerprint":41}, {"source_digest":b"y"*32},
            {"bindings":{}}, {"previous_bindings":{92:args["previous_bindings"][91]}},
            {"coordinate_frame":"unknown"}, {"coordinate_frame":"core_inertial_local"},
            {"bindings":{91:struct.pack("<I8f",92,2,0,0,0,0,0,1,1)}},
            {"previous_bindings":{91:args["bindings"][91]}}
        ]:
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                _retarget_carrier_binding_bytes(raw,**{**args,**delta})

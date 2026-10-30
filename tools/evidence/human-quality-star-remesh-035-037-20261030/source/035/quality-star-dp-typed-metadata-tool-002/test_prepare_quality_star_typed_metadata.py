#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("prepare_quality_star_typed_metadata.py")
spec = importlib.util.spec_from_file_location("quality_star_writer", SCRIPT)
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)

class WriterTests(unittest.TestCase):
    def test_aligned_tuple_union_deduplicates_rows_not_lanes(self):
        arr = {
            "face_source_layer": np.array([0, 0, 0]),
            "face_source_body": np.array([-1, -1, -1]),
            "face_source_bone": np.array([2, 2, 3]),
            "face_source_face_id": np.array([9, 9, 9]),
            "face_source_original_id": np.array([100, 100, 101]),
            "face_source_identity0": np.array([1, 1, 2]),
            "face_source_identity1": np.array([7, 7, 8]),
        }
        got = writer.tuple_union(arr, np.array([0, 1, 2], dtype=np.int64))
        self.assertEqual(got, [(0, -1, 2, 9, 100, 1, 7), (0, -1, 3, 9, 101, 2, 8)])

    def test_csr_rebuild_keeps_retained_and_empty_new_face_rows(self):
        offsets = np.array([0, 2, 3, 3], dtype=np.int64)
        lanes = {
            "source_rows": np.array([5, 6, 7], dtype=np.int32),
            "source_tags": np.array([50, 60, 70], dtype=np.int64),
        }
        out_offsets, out = writer.remap_csr(offsets, lanes, np.array([2, -1, 0]), 3)
        np.testing.assert_array_equal(out_offsets, [0, 0, 0, 2])
        np.testing.assert_array_equal(out["source_rows"], [5, 6])
        np.testing.assert_array_equal(out["source_tags"], [50, 60])

    def test_compaction_does_not_coordinate_weld_distinct_rows(self):
        pts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,1,0],[0,1,1]], dtype=np.float64)
        faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
        row_map = np.array([10,11,12,13,14,15], dtype=np.int64)
        out_pts, out_faces, used, raw_to_compact, input_to_compact = writer.compact_rows(faces, pts, row_map)
        self.assertEqual(len(out_pts), 6)
        self.assertEqual(out_pts[0].tobytes(), out_pts[3].tobytes())
        self.assertNotEqual(int(out_faces[0,0]), int(out_faces[1,0]))
        self.assertEqual(int(input_to_compact[10]), 0)
        self.assertEqual(int(raw_to_compact[3]), 3)

    def test_end_to_end_derived_face_union_and_f64_point_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            batch = root / "batch"
            batch.mkdir()
            outdir = root / "out"
            outdir.mkdir()
            src = root / "source-I.npz"
            raw_path = batch / "O-fixed-boundary-star-batch-proposal.npz"
            pts = np.array([
                [0.,0.,0.], [1.,0.,0.], [0.,1.,0.],
                [1.,1.,0.], [2.,0.,0.], [2.,1.,0.],
            ], dtype=np.float64)
            faces = np.array([[0,1,2],[1,3,2],[1,4,3],[4,5,3]], dtype=np.int64)
            typed = {
                "face_source_layer": np.array([0,0,0,0], dtype=np.int64),
                "face_source_body": np.array([-1,-1,-1,-1], dtype=np.int64),
                "face_source_bone": np.array([-1,-1,-1,-1], dtype=np.int64),
                "face_source_face_id": np.array([10,11,12,13], dtype=np.int64),
                "face_source_original_id": np.array([1,1,1,1], dtype=np.int64),
                "face_source_identity0": np.array([100,101,102,103], dtype=np.int64),
                "face_source_identity1": np.array([200,201,202,203], dtype=np.int64),
            }
            schema = {"1":{"role":"O","path":"fixture-source","sha256":"fixture-hash",
                           "identity_keys":["face_source_identity0","face_source_identity1"]}}
            src_arrays = {
                "points_world_m": pts,
                "points_world_m_f32": pts.astype(np.float32),
                "faces": faces,
                **typed,
                "face_source_identity_schema_by_original_id_json": np.array([json.dumps(schema)]),
                "face_ancestry_offsets": np.array([0,1,2,3,4], dtype=np.int64),
                "face_ancestry_source_face_rows": np.array([10,11,12,13], dtype=np.int64),
                "face_ancestry_source_original_ids": np.array([1,1,1,1], dtype=np.int64),
                "face_ancestry_parent_face_rows": np.array([20,21,22,23], dtype=np.int64),
                "microedge_parent_input_face_offsets": np.array([0,1,2,3,4], dtype=np.int64),
                "microedge_parent_input_face_rows": np.array([30,31,32,33], dtype=np.int64),
                "microedge_parent_input_point_offsets": np.arange(7, dtype=np.int64),
                "microedge_parent_input_point_rows": np.arange(6, dtype=np.int64),
                "reference_vertex_parent_face_offsets": np.arange(7, dtype=np.int64),
                "reference_vertex_parent_face_rows": np.arange(6, dtype=np.int64),
                "reference_vertex_parent_face_barycentric": np.tile(np.array([[1.,0.,0.]]), (6,1)),
                "reference_subdivision_parent_face_rows": np.array([40,41,42,43], dtype=np.int64),
            }
            np.savez_compressed(src, **src_arrays)
            source_sha = writer.sha256(src)
            raw_to_input = np.arange(6, dtype=np.int64)
            input_to_raw = np.arange(6, dtype=np.int64)
            out_faces = np.array([[0,1,2],[1,3,2],[1,4,3],[4,5,3]], dtype=np.int64)
            out_to_input_faces = np.array([-1,-1,2,3], dtype=np.int64)
            raw = {
                "points_world_m": pts.copy(),
                "points_world_m_f32": pts.astype(np.float32),
                "faces": out_faces,
                "source_candidate_sha256": np.array([source_sha]),
                "point_output_to_input_rows": raw_to_input,
                "point_input_to_output_row": input_to_raw,
                "face_output_to_input_rows": out_to_input_faces,
                "retired_input_face_rows": np.array([0,1], dtype=np.int64),
                "new_faces_input_point_rows": faces[:2].copy(),
                "new_output_face_patch_ids": np.array([0,1], dtype=np.int64),
                "candidate_patch_operation_ids": np.array(["O-star-A","O-star-B"]),
            }
            np.savez_compressed(raw_path, **raw)
            lineages = []
            for face_row in (0,1):
                lineages.append({k:[int(typed[k][face_row])] for k in writer.TYPED_FIELDS})
            for i, pid in enumerate(("O-star-A","O-star-B")):
                retired = i
                patch = {
                    "patch_id": pid,
                    "retired_input_face_rows": [retired],
                    "new_faces_input_point_rows": faces[retired:retired+1].tolist(),
                    "new_output_face_rows": [i],
                    "source_lineage": lineages[i],
                }
                operation = {
                    "role":"O", "patch_id":pid,
                    "retired_input_face_rows":[retired],
                    "new_faces_input_point_rows":faces[retired:retired+1].tolist(),
                    "new_output_face_rows":[i],
                    "source_lineage":lineages[i],
                    "patches":[patch],
                }
                (batch / f"{pid}.json").write_text(json.dumps(operation))
            result = writer.build_role("O", src, raw_path, batch, [], outdir)
            candidate = load = writer.load_npz(Path(result["output_path"]))
            self.assertEqual(candidate["points_world_m"].dtype, np.float64)
            self.assertEqual(candidate["points_world_m"].tobytes(), pts.tobytes())
            self.assertEqual(candidate["faces"].shape, faces.shape)
            self.assertEqual(candidate["face_source_original_id"][0], 700_000_000)
            self.assertEqual(candidate["face_source_original_id"][1], 700_000_001)
            self.assertEqual(candidate["face_source_face_id"][:2].tolist(), [-1,-1])
            self.assertEqual(candidate["face_source_identity0"][:2].tolist(), [0,1])
            self.assertEqual(candidate["face_source_identity1"][:2].tolist(), [0,1])
            np.testing.assert_array_equal(candidate["face_ancestry_offsets"], [0,1,2,3,4])
            np.testing.assert_array_equal(candidate["face_ancestry_source_face_rows"], [10,11,12,13])
            np.testing.assert_array_equal(candidate["face_ancestry_source_original_ids"], [1,1,1,1])
            np.testing.assert_array_equal(candidate["face_ancestry_parent_face_rows"], [20,21,22,23])
            np.testing.assert_array_equal(candidate["quality_star_patch_union_offsets"], [0,1,2,2,2])
            for face_row, source_row in ((0,0),(1,1)):
                a,b = candidate["quality_star_patch_union_offsets"][face_row:face_row+2]
                got = np.column_stack([candidate[f"quality_star_patch_union_{k}"][a:b] for k in writer.TYPED_FIELDS])
                expected = np.array([[typed[k][source_row] for k in writer.TYPED_FIELDS]], dtype=np.int64)
                np.testing.assert_array_equal(got, expected)
            schema_out = json.loads(str(candidate["face_source_identity_schema_by_original_id_json"][0]))
            self.assertEqual(schema_out["700000000"]["role"], "DERIVED_PATCH")
            self.assertEqual(schema_out["700000001"]["complete_typed_parent_union"], [[0,-1,-1,11,1,101,201]])


    def test_current_input_face_ancestry_is_nonempty_and_tuple_aligned(self):
        source = {"face_source_original_id": np.array([30, 10, 30], dtype=np.int64)}
        output_to_input = np.array([-1, 2], dtype=np.int64)
        patch_rows = {0: {"retired": [0, 1]}}
        offsets, source_rows, original_ids, parent_rows = writer.build_current_input_face_ancestry(
            source, output_to_input, patch_rows)
        np.testing.assert_array_equal(offsets, [0, 2, 3])
        np.testing.assert_array_equal(source_rows, [0, 1, 2])
        np.testing.assert_array_equal(original_ids, [30, 10, 30])
        np.testing.assert_array_equal(parent_rows, [0, 1, 2])
        self.assertTrue(np.all(np.diff(offsets) > 0))

    def test_I_build_emits_217_expected_current_candidate_ancestry_alias(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            batch = root / "batch"; batch.mkdir()
            outdir = root / "out"; outdir.mkdir()
            src = root / "source-I.npz"
            raw_path = batch / "I-fixed-boundary-star-batch-proposal.npz"
            pts = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[2.,0.,0.],[2.,1.,0.]], dtype=np.float64)
            faces = np.array([[0,1,2],[1,3,2],[1,4,3],[4,5,3]], dtype=np.int64)
            typed = {
                "face_source_layer": np.array([1,1,1,1], dtype=np.int64),
                "face_source_body": np.array([23,23,23,23], dtype=np.int64),
                "face_source_bone": np.array([28,28,28,28], dtype=np.int64),
                "face_source_face_id": np.array([10,11,12,13], dtype=np.int64),
                "face_source_original_id": np.array([2,2,2,2], dtype=np.int64),
                "face_source_identity0": np.array([100,101,102,103], dtype=np.int64),
                "face_source_identity1": np.array([200,201,202,203], dtype=np.int64),
            }
            schema = {"2":{"role":"I","path":"fixture-source","sha256":"fixture-hash",
                           "identity_keys":["face_source_identity0","face_source_identity1"]}}
            src_arrays = {
                "points_world_m":pts, "points_world_m_f32":pts.astype(np.float32), "faces":faces, **typed,
                "face_source_identity_schema_by_original_id_json":np.array([json.dumps(schema)]),
                "face_parent_offsets":np.array([0,1,2,3,4]), "face_parent_rows":np.array([10,11,12,13]),
                "edge_flip_parent_offsets":np.array([0,1,2,3,4]), "edge_flip_parent_face_rows":np.array([20,21,22,23]),
                "microedge_parent_input_face_offsets":np.array([0,1,2,3,4]), "microedge_parent_input_face_rows":np.array([30,31,32,33]),
                "microedge_parent_input_point_offsets":np.arange(7), "microedge_parent_input_point_rows":np.arange(6),
                "reference_original_input_vertex_offsets":np.arange(7), "reference_original_input_vertex_rows":np.arange(6),
                "reference_vertex_parent_face_offsets":np.arange(7), "reference_vertex_parent_face_rows":np.arange(6),
                "reference_vertex_parent_face_barycentric":np.tile(np.array([[1.,0.,0.]]),(6,1)),
                "reference_incremental_parent_input_face_rows":np.array([40,41,42,43]),
            }
            np.savez_compressed(src, **src_arrays)
            src_sha = writer.sha256(src)
            raw = {
                "points_world_m":pts.copy(), "points_world_m_f32":pts.astype(np.float32), "faces":faces.copy(),
                "source_candidate_sha256":np.array([src_sha]), "point_output_to_input_rows":np.arange(6),
                "point_input_to_output_row":np.arange(6), "face_output_to_input_rows":np.array([-1,-1,2,3]),
                "retired_input_face_rows":np.array([0,1]), "new_faces_input_point_rows":faces[:2].copy(),
                "new_output_face_patch_ids":np.array([0,0]), "candidate_patch_operation_ids":np.array(["I-star-A"]),
            }
            np.savez_compressed(raw_path, **raw)
            patch_id = "I-star-A"
            source_lineage = {k:[int(typed[k][j]) for j in (0,1)] for k in writer.TYPED_FIELDS}
            patch = {"patch_id":patch_id,"retired_input_face_rows":[0,1],
                     "new_faces_input_point_rows":faces[:2].tolist(),
                     "new_output_face_rows":[0,1],"source_lineage":source_lineage}
            op = {"role":"I","patch_id":patch_id,"retired_input_face_rows":[0,1],
                  "new_faces_input_point_rows":faces[:2].tolist(),
                  "new_output_face_rows":[0,1],"source_lineage":source_lineage,"patches":[patch]}
            (batch / f"{patch_id}.json").write_text(json.dumps(op))
            report = writer.build_role("I", src, raw_path, batch, [], outdir)
            with np.load(report["output_path"], allow_pickle=False) as z:
                self.assertIn("face_ancestry_source_face_rows", z.files)
                self.assertNotIn("face_ancestry_source_rows", z.files)
                np.testing.assert_array_equal(z["face_ancestry_offsets"], [0,2,4,5,6])
                np.testing.assert_array_equal(z["face_ancestry_source_face_rows"], [0,1,0,1,2,3])
                np.testing.assert_array_equal(z["face_ancestry_source_original_ids"], [2,2,2,2,2,2])
                np.testing.assert_array_equal(z["face_ancestry_parent_face_rows"], [0,1,0,1,2,3])
                self.assertTrue(np.all(np.diff(z["face_ancestry_offsets"]) > 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)


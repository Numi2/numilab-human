import unittest
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from warning_row_rebase_041 import rebase_segments, run_manifest


def report(*segments):
    return {"roles": [{"role": "skin", "warnings": [{"warning_index": 4, "segments": list(segments)}]}]}


def segment(role_rows, coords):
    return {"role_rows": role_rows, "coordinates_f32_m": np.asarray(coords, np.float32).tolist()}


class WarningRowRebaseTests(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        self.candidate = {"skin": {"points_f32": self.points, "faces": np.asarray([[0, 1, 2]])}}

    def test_explicit_map_checks_coordinates_and_edge(self):
        # Warning rows 10 and 11 map to candidate rows 2 and 0.
        warning = report(segment([["skin", 10], ["skin", 11]], [self.points[2], self.points[0]]))
        row_map = np.full(12, -1, np.int64)
        row_map[10], row_map[11] = 2, 0
        got = rebase_segments(warning, self.candidate, {"skin": row_map})
        self.assertEqual(got["segments"][0]["current_rows"], [2, 0])
        self.assertEqual(got["segments"][0]["status"], "UNRESOLVED_CURRENT_EDGE")

    def test_exact_translated_map(self):
        warning = report(segment([["skin", 0], ["skin", 1]], [self.points[2], self.points[0]]))
        got = rebase_segments(warning, self.candidate, {"skin": np.asarray([2, 0])})
        self.assertEqual(got["segments"][0]["current_rows"], [2, 0])
        self.assertEqual(got["segments"][0]["status"], "UNRESOLVED_CURRENT_EDGE")

    def test_removed_endpoint_is_explicit(self):
        warning = report(segment([["skin", 0], ["skin", 1]], [self.points[0], self.points[1]]))
        got = rebase_segments(warning, self.candidate, {"skin": np.asarray([0, -1])})
        self.assertEqual(got["segments"][0]["status"], "ENDPOINT_REMOVED_OR_OUT_OF_RANGE")

    def test_coordinate_mismatch_never_nearest_matches(self):
        warning = report(segment([["skin", 0], ["skin", 1]], [[0.1, 0, 0], [1, 0, 0]]))
        got = rebase_segments(warning, self.candidate, {"skin": np.asarray([0, 1])})
        self.assertEqual(got["segments"][0]["status"], "FLOAT32_COORDINATE_MISMATCH")

    def test_absent_edge_is_distinguished(self):
        warning = report(segment([["skin", 0], ["skin", 1]], [self.points[0], self.points[1]]))
        no_edge_points = np.concatenate((self.points, np.asarray([[1, 1, 1]], np.float32)))
        no_edge = {"skin": {"points_f32": no_edge_points, "faces": np.asarray([[0, 2, 3]])}}
        got = rebase_segments(warning, no_edge, {"skin": np.asarray([0, 1])})
        self.assertEqual(got["segments"][0]["status"], "EDGE_REMOVED")

    def test_cross_role_segment_is_not_rebased(self):
        warning = report(segment([["skin", 0], ["fat", 1]], [self.points[0], self.points[1]]))
        got = rebase_segments(warning, self.candidate, {"skin": None})
        self.assertEqual(got["segments"][0]["status"], "CROSS_ROLE_SEGMENT")

    def test_manifest_identity_is_sha_bound_and_emits_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate.npz"
            np.savez(candidate_path, points_world_m_f32=self.points, faces=np.asarray([[0, 1, 2]], np.int64))
            warning_path = root / "warning.json"
            warning = report(segment([["skin", 0], ["skin", 1]], [self.points[0], self.points[1]]))
            warning_path.write_text(json.dumps(warning))
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            manifest_path = root / "inputs.json"
            manifest_path.write_text(json.dumps({
                "schema": "numi.exact-warning-row-rebase-inputs.v1",
                "warning_report": {"path": str(warning_path), "sha256": sha(warning_path)},
                "roles": {"skin": {
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha(candidate_path),
                    "points_key": "points_world_m_f32",
                    "identity_domain_sha256": sha(candidate_path),
                }},
            }))
            out = root / "receipt.json"
            result = run_manifest(manifest_path, out)
            self.assertEqual(result["status"], "MAPPED_WITH_UNRESOLVED_EDGES")
            self.assertTrue(out.is_file())
            self.assertEqual(json.loads(out.read_text())["segment_count"], 1)

    def test_manifest_rejects_stale_candidate_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate.npz"
            np.savez(candidate_path, points_world_m_f32=self.points, faces=np.asarray([[0, 1, 2]], np.int64))
            warning_path = root / "warning.json"
            warning_path.write_text(json.dumps(report()))
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            manifest_path = root / "inputs.json"
            manifest_path.write_text(json.dumps({
                "schema": "numi.exact-warning-row-rebase-inputs.v1",
                "warning_report": {"path": str(warning_path), "sha256": sha(warning_path)},
                "roles": {"skin": {
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": "0" * 64,
                    "identity_domain_sha256": "0" * 64,
                }},
            }))
            with self.assertRaisesRegex(ValueError, "candidate SHA mismatch"):
                run_manifest(manifest_path, root / "receipt.json")


if __name__ == "__main__":
    unittest.main()

"""Input-binding regressions for the shared resting geometry entrypoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

_SPEC = importlib.util.spec_from_file_location("verify_resting_entrypoint", Path(__file__).resolve().parents[1] / "tools/verify_resting_anatomy.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

KEY = (51005, 73)
POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4")
GIDS = np.arange(4, dtype="<u4") + 100


class RestingVerificationEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def candidate(self, **changes):
        arrays = dict(capture_positions_m=POINTS, capture_faces=FACES, capture_global_vertex_ids=GIDS)
        arrays.update(changes)
        path = self.root / "candidate.npz"
        np.savez(path, **arrays)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def surfaces(self):
        return {KEY: (POINTS.copy(), FACES.copy(), None, GIDS.copy())}

    def test_wrong_hash_rejected_without_loading_archive(self):
        path, _ = self.candidate()
        with patch.object(tool.np, "load", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                tool.apply_overlay(self.surfaces(), KEY, path, "0" * 64)

    def test_reordered_source_ids_rejected(self):
        path, digest = self.candidate(capture_global_vertex_ids=GIDS[::-1])
        with self.assertRaisesRegex(ValueError, "ordered captured source GID"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_float64_is_not_silently_rounded_for_predicates(self):
        path, digest = self.candidate(capture_positions_m=POINTS.astype("<f8"))
        with self.assertRaisesRegex(ValueError, "exact Float32"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_face_index_bounds_use_shared_preparation(self):
        bad = FACES.copy()
        bad[0, 0] = 999
        path, digest = self.candidate(capture_faces=bad)
        with self.assertRaises(ValueError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_invalid_normals_are_rejected_before_pair_predicates(self):
        path, digest = self.candidate(capture_normals=np.full_like(POINTS, np.nan))
        with self.assertRaises(ValueError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_no_guessing_faces_key_and_no_native_admission(self):
        path = self.root / "candidate.npz"
        np.savez(path, capture_positions_m=POINTS, capture_faces_candidate=FACES, capture_global_vertex_ids=GIDS)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(KeyError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)
        surfaces = self.surfaces()
        report = tool.apply_overlay(surfaces, KEY, path, digest, "capture_faces_candidate")
        self.assertTrue(report["world_positions_are_counterfactual"])
        self.assertTrue(report["all_ordered_capture_gids_exact"])
        self.assertFalse(report["candidate_normals_present"])
        self.assertIsInstance(surfaces[KEY], tool.verify.PreparedSurface)

    def test_bad_capture_pin_stops_before_capture_loader(self):
        pack, receipt = self.root / "p", self.root / "r"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        output = self.root / "report.json"
        with patch.object(tool.verify, "load_capture", side_effect=AssertionError("must not load")):
            result = tool.main(["--pack", str(pack), "--pack-sha256", "0"*64, "--capture-receipt", str(receipt),
                                "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                "--changed", "51005:73", "--output", str(output)])
        report = json.loads(output.read_text())
        self.assertEqual(result, 1)
        self.assertEqual(report["failed_stage"], "preparation")
        self.assertFalse(report["native_admission"])
        self.assertFalse(report["anatomical_admission"])


if __name__ == "__main__":
    unittest.main()

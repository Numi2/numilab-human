#!/usr/bin/env python3
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER_PATH = HERE / "registered_receipt_adapter_1171.py"
SPEC = importlib.util.spec_from_file_location("registered_adapter_test", ADAPTER_PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)
EXPECTED = 155000


class RegisteredReceiptAdapterTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        target = root / "output" / "scene" / "native.log"
        target.parent.mkdir(parents=True)
        target.write_text("registered native log\n")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        receipt = {"returncode": 0, "failure": None, "ended_at": "2026-10-09T00:00:00Z",
                   "files": {"scene/native.log": digest}}
        return td, root, receipt

    def test_valid_registered_fixture_without_scene_run_metadata(self):
        td, root, receipt = self.fixture()
        with td:
            self.assertFalse((root / "output" / "scene" / "run-metadata.json").exists())
            self.assertEqual(ADAPTER.validate_successful_exit(receipt, EXPECTED)["accepted_terminal_step"], EXPECTED)
            files = ADAPTER.verify_receipt_files(root, receipt["files"])
            self.assertEqual(len(files), 1)
            self.assertTrue(ADAPTER.validate_receipt_chain_links(
                "registration-seal", "started-seal", "process-seal",
                {"registration_sha256": "registration-seal"},
                {"binding": "started-seal"},
                {"started_sha256": "started-seal", "process_sha256": "process-seal"},
            ))

    def test_missing_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            ADAPTER.validate_successful_exit(None, EXPECTED)

    def test_receipt_chain_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not bind"):
            ADAPTER.validate_receipt_chain_links(
                "registration-seal", "started-seal", "process-seal",
                {"registration_sha256": "registration-seal"},
                {"binding": "different-start"},
                {"started_sha256": "started-seal", "process_sha256": "process-seal"},
            )

    def test_failed_registered_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed"):
            ADAPTER.validate_successful_exit(
                {"returncode": 2, "failure": "process_exit_2", "ended_at": "2026-10-09T00:00:00Z"},
                EXPECTED,
            )

    def test_registered_output_hash_mismatch_rejected(self):
        td, root, receipt = self.fixture()
        with td:
            path = root / "output" / "scene" / "native.log"
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                ADAPTER.verify_receipt_files(root, receipt["files"])

    def test_n_minus_one_terminal_rejected(self):
        receipt = {"returncode": 0, "failure": None, "ended_at": "2026-10-09T00:00:00Z"}
        with self.assertRaisesRegex(ValueError, "exact terminal step"):
            ADAPTER.validate_successful_exit(receipt, EXPECTED - 1)


    def test_p18_helpers_are_exactly_pinned_and_expose_capture_api(self):
        self.assertEqual(str(ADAPTER.P18),
                         "/Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-1170/package-v018")
        self.assertEqual(ADAPTER.ANALYZER_SHA256,
                         "defe06f8ded56d66fe5ba903445bb8dd81ef96ad57fa59e3be4c615c5892cb5c")
        self.assertEqual(ADAPTER.PREPARE_SHA256,
                         "f5d57e46afce90ea756e8eb8e23bfd1f4a6a6e9fcb971659f946ce29768531be")
        analyzer = ADAPTER._load(ADAPTER.ANALYZER_PATH, "registered_p18_analyzer_api_test")
        prepare = ADAPTER._load(ADAPTER.PREPARE_PATH, "registered_p18_prepare_api_test")
        self.assertTrue(callable(analyzer.derived_capture_trial))
        self.assertTrue(callable(analyzer.accepted_geometry_summary))
        self.assertTrue(callable(prepare.accepted_geometry_capture_schedule))
        self.assertTrue(callable(prepare.validate_capture_receipt))

if __name__ == "__main__":
    unittest.main()

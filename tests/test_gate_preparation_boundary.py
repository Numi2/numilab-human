import hashlib
import tempfile
import unittest
from pathlib import Path

from numilab_human.gate_preparation_boundary import (
    execute_after_preparation,
    freeze_preparation,
    verify_frozen_preparation,
)


class PreparationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.asset = self.root / "input.bin"
        self.asset.write_bytes(b"pinned source")
        self.pin = {
            str(self.asset): hashlib.sha256(self.asset.read_bytes()).hexdigest()
        }
        self.argv = ["/qualified/python", "/task/runner.py", "--execute"]
        self.proof = {
            "status": "PASS_INPUT_PREFLIGHT",
            "predicate_phase_started": False,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_freezes_canonical_pins_and_exact_argv_before_geometry(self):
        frozen = freeze_preparation(
            self.root / "proof.json",
            self.proof,
            runner_argv=self.argv,
            input_pins=self.pin,
        )
        proof = verify_frozen_preparation(frozen)
        called = []
        result = execute_after_preparation(
            frozen, lambda current: called.append(current) or "ran"
        )
        self.assertEqual(result, "ran")
        self.assertEqual(len(called), 1)
        self.assertEqual(proof["execution_boundary"]["runner_argv"], self.argv)
        self.assertEqual(
            proof["execution_boundary"]["input_pins"],
            {str(self.asset.resolve()): self.pin[str(self.asset)]},
        )

    def test_failed_preparation_never_invokes_geometry(self):
        called = []
        with self.assertRaises(ValueError):
            freeze_preparation(
                self.root / "proof.json",
                {"status": "FAIL", "predicate_phase_started": False},
                runner_argv=self.argv,
                input_pins=self.pin,
            )
        self.assertEqual(called, [])
        self.assertFalse((self.root / "proof.json").exists())

    def test_nonfinite_proof_or_missing_argv_fails_before_writing(self):
        path = self.root / "proof.json"
        with self.assertRaises(ValueError):
            freeze_preparation(
                path,
                {**self.proof, "value": float("nan")},
                runner_argv=self.argv,
                input_pins=self.pin,
            )
        with self.assertRaises(ValueError):
            freeze_preparation(path, self.proof, runner_argv=[], input_pins=self.pin)
        self.assertFalse(path.exists())

    def test_modified_proof_fails_before_geometry(self):
        path = self.root / "proof.json"
        frozen = freeze_preparation(
            path, self.proof, runner_argv=self.argv, input_pins=self.pin
        )
        path.write_text(path.read_text() + " ", encoding="utf-8")
        called = []
        with self.assertRaises(ValueError):
            execute_after_preparation(frozen, lambda current: called.append(current))
        self.assertEqual(called, [])

    def test_changed_input_bytes_after_freeze_fail_before_geometry(self):
        frozen = freeze_preparation(
            self.root / "proof.json",
            self.proof,
            runner_argv=self.argv,
            input_pins=self.pin,
        )
        self.asset.write_bytes(b"changed after freeze")
        called = []
        with self.assertRaises(ValueError):
            execute_after_preparation(frozen, lambda current: called.append(current))
        self.assertEqual(called, [])

    def test_retargeted_symlink_after_freeze_fails_before_geometry(self):
        other = self.root / "other.bin"
        other.write_bytes(b"different input")
        alias = self.root / "alias.bin"
        alias.symlink_to(self.asset)
        pin = {str(alias): hashlib.sha256(self.asset.read_bytes()).hexdigest()}
        frozen = freeze_preparation(
            self.root / "proof.json", self.proof, runner_argv=self.argv, input_pins=pin
        )
        alias.unlink()
        alias.symlink_to(other)
        called = []
        with self.assertRaises(ValueError):
            execute_after_preparation(frozen, lambda current: called.append(current))
        self.assertEqual(called, [])

    def test_rejects_duplicate_canonical_path_and_stale_input_hash(self):
        alias = self.root / "alias.bin"
        alias.symlink_to(self.asset)
        path = self.root / "proof.json"
        duplicate = {**self.pin, str(alias): self.pin[str(self.asset)]}
        with self.assertRaises(ValueError):
            freeze_preparation(
                path, self.proof, runner_argv=self.argv, input_pins=duplicate
            )
        self.asset.write_bytes(b"changed source")
        with self.assertRaises(ValueError):
            freeze_preparation(
                path, self.proof, runner_argv=self.argv, input_pins=self.pin
            )
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

"""Reject incomplete, corrupted and physically contradictory Human receipts."""
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1] / "Docs/media/source-compliant-equilibrium-20260910"
SPEC = importlib.util.spec_from_file_location("compliant_receipt", BASE / "verify_receipt.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class SourceCompliantEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = V.read(BASE / "isolation4.log")
        cls.static = V.read(BASE / "qualified.log")

    def mutate_trace(self, update, scenario="unavailable", kind="qv"):
        lines = self.trace.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("prepared_recruitment={"):
                continue
            row = json.loads(line.split("=", 1)[1])
            if row["scenario"] == scenario and row["root"] == 1 and row["kind"] == kind:
                update(row)
                lines[i] = "prepared_recruitment=" + json.dumps(row)
                return "\n".join(lines)
        raise AssertionError("fixture record missing")

    def replace_word(self, row, index, value):
        raw = bytearray(base64.b64decode(row["fp32_le_base64"]))
        struct.pack_into("<f", raw, index * 4, value)
        row["fp32_le_base64"] = base64.b64encode(raw).decode()

    def test_receipt(self):
        self.assertEqual(V.verify()["status"], "source_compliant_preparation_verified")

    def test_short_trace_replay_and_dropout(self):
        result = V.audit_trace(self.trace, 4)
        self.assertEqual(result["replay"], "bitwise")
        self.assertTrue(result["dropout_matches_zero"])

    def test_long_contradiction_is_retained_and_rejected(self):
        trace = V.read(BASE / "horizon-streamed.log.gz")
        with self.assertRaisesRegex(ValueError, "dropout drift"):
            V.audit_trace(trace, 64)
        self.assertEqual(V.audit_trace(trace, 64, require_equivalence=False)["first_dropout_difference"]["root"], 1)

    def test_missing_and_duplicate_root_record(self):
        lines = self.trace.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("prepared_recruitment={"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            V.audit_trace("\n".join(lines[:index] + lines[index+1:]), 4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            V.audit_trace(self.trace + "\n" + lines[index], 4)

    def test_nonfinite_state_and_bad_extent(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            V.audit_trace(self.mutate_trace(lambda r: self.replace_word(r, 0, float("nan"))), 4)
        with self.assertRaisesRegex(ValueError, "extent"):
            V.audit_trace(self.mutate_trace(lambda r: r.update(fp32_le_base64="AAAAAA==")), 4)

    def test_invalid_clock_and_schema(self):
        for update, message in [(lambda r: r.update(elapsed_microseconds=101), "clock"),
                                (lambda r: r.update(schema="unknown"), "schema")]:
            with self.assertRaisesRegex(ValueError, message):
                V.audit_trace(self.mutate_trace(update), 4)

    def test_dropout_motion_and_actuator_corruption(self):
        with self.assertRaisesRegex(ValueError, "dropout drift"):
            V.audit_trace(self.mutate_trace(lambda r: self.replace_word(r, 129, 0.5)), 4)
        with self.assertRaisesRegex(ValueError, "actuator bounds"):
            V.audit_trace(self.mutate_trace(lambda r: self.replace_word(r, 0, 1.1), kind="activation"), 4)
        with self.assertRaisesRegex(ValueError, "motor availability"):
            V.audit_trace(self.mutate_trace(lambda r: self.replace_word(r, 0, 0.1), kind="motor"), 4)

    def test_static_force_and_support_corruption(self):
        for field, value, message in [("force_residual", 1.0, "decomposition"),
                                      ("support_normal_force", -1.0, "support")]:
            lines = self.static.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("source_compliant_equilibrium="):
                    row = json.loads(line.split("=", 1)[1]); row[field][0] = value
                    lines[i] = "source_compliant_equilibrium=" + json.dumps(row)
            with self.assertRaisesRegex(ValueError, message):
                V.audit_static("\n".join(lines))

    def test_artifact_hash_and_inventory_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt"
            shutil.copytree(BASE, target, ignore=shutil.ignore_patterns("__pycache__"))
            receipt = json.loads((target / "receipt.json").read_text())
            receipt["artifacts"].pop()
            (target / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "inventory"):
                V.verify(target)
            shutil.copy(BASE / "receipt.json", target / "receipt.json")
            (target / "qualified.log").write_text(self.static + "corruption")
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                V.verify(target)


if __name__ == "__main__":
    unittest.main()

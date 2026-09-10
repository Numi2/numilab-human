import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coupled_pose_evidence", ROOT / "Docs/media/coupled-initial-poses-20260910/verify_receipt.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CoupledInitialPoseEvidenceTests(unittest.TestCase):
    def test_published_receipt(self):
        result = AUDIT.verify()
        self.assertTrue(result["repaired"]["dropout_matches_zero"])
        self.assertFalse(result["retained_failure"]["dropout_matches_zero"])

    def test_truncated_contact_trace_is_rejected(self):
        text = AUDIT.read(AUDIT.BASE / "horizon-support-trace.log.gz")
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("mrnx_initial_support="))
        del lines[index]
        with self.assertRaisesRegex(ValueError, "assembly trace incomplete"):
            AUDIT.audit_failure("\n".join(lines))

    def test_source_drift_cannot_be_called_contact_drift(self):
        lines = AUDIT.read(AUDIT.BASE / "horizon-support-trace.log.gz").splitlines()
        for i, line in enumerate(lines):
            if line.startswith("mrnx_initial_buffer="):
                row = json.loads(line.split("=", 1)[1])
                if row["name"] == "initial_q":
                    row["values"][0] += 0.01
                    lines[i] = "mrnx_initial_buffer=" + json.dumps(row)
                    break
        with self.assertRaisesRegex(ValueError, "source input diverged"):
            AUDIT.audit_failure("\n".join(lines))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("candidate_precision_evidence",
    ROOT / "Docs/media/candidate-precision-performance-20260911/verify_receipt.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CandidatePrecisionPerformanceEvidenceTests(unittest.TestCase):
    def read(self, label):
        return AUDIT.data(AUDIT.BASE / (label + ".log.gz")).decode()

    def test_receipt_preserves_failed_refinement(self):
        result = AUDIT.verify()
        self.assertFalse(result["refinement"]["refine-25us-components"]["completed"])
        self.assertGreater(result["contact_failure"]["support_residual_norm"], 0.005)

    def test_missing_root_cannot_pass(self):
        lines = self.read("refine-50us-native").splitlines()
        del lines[next(i for i, line in enumerate(lines) if line.startswith("prepared_recruitment={"))]
        with self.assertRaisesRegex(ValueError, "incomplete scenario or root"):
            AUDIT.audit_refinement("\n".join(lines), 50, 32, True)

    def test_atomic_record_after_partial_diagnostic_is_preserved(self):
        original = self.read("refine-50us-published")
        modified = original.replace("prepared_recruitment={", "partial diagnostic prepared_recruitment={", 1)
        self.assertEqual(AUDIT.audit_refinement(original, 50, 32, True),
                         AUDIT.audit_refinement(modified, 50, 32, True))

    def test_failed_prefix_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError, "incomplete scenario or root"):
            AUDIT.audit_refinement(self.read("refine-25us-components"), 25, 64, True)

    def test_missing_timestamp_cannot_establish_speedup(self):
        lines = self.read("timing-ancestry4").splitlines()
        del lines[next(i for i, line in enumerate(lines) if line.startswith("candidate_gpu_timing="))]
        with self.assertRaisesRegex(ValueError, "incomplete GPU timestamps"):
            AUDIT.audit_timing("\n".join(lines))

    def test_missing_final_contact_iterate_is_rejected(self):
        lines = self.read("refine-25us-iterate-trace").splitlines()
        indices = [i for i,line in enumerate(lines) if line.startswith("human_support_iterate=")]
        del lines[indices[-1]]
        with self.assertRaisesRegex(ValueError, "missing or duplicate Newton iterate"):
            AUDIT.audit_iterates("\n".join(lines))


if __name__ == "__main__":
    unittest.main()

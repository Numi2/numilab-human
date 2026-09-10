import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("muscle_numerics_evidence",
    ROOT / "Docs/media/prepared-muscle-numerics-20260911/verify_receipt.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PreparedMuscleNumericsEvidenceTests(unittest.TestCase):
    def test_published_receipt(self):
        evidence = AUDIT.verify()
        self.assertTrue(evidence["coupled"]["dropout_matches_zero"])
        self.assertLessEqual(evidence["maximum_coupled_fibre_publication_error_ulps"], 0.501)

    def test_missing_muscle_cannot_pass(self):
        lines = AUDIT.data(AUDIT.BASE / "published-100us.log.gz").decode().splitlines()
        del lines[next(i for i, line in enumerate(lines) if line.startswith("prepared_path={"))]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            AUDIT.audit_probe("\n".join(lines), AUDIT.force_scales(), True)

    def test_force_summary_must_match_rows(self):
        lines = AUDIT.data(AUDIT.BASE / "published-100us.log.gz").decode().splitlines()
        i = next(i for i, line in enumerate(lines) if line.startswith("prepared_path={"))
        row = json.loads(lines[i].split("=", 1)[1])
        row["metal_fp32_force_n"] += 10
        lines[i] = "prepared_path=" + json.dumps(row)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            AUDIT.audit_probe("\n".join(lines), AUDIT.force_scales(), True)

    def test_tolerance_cannot_be_relaxed(self):
        lines = AUDIT.data(AUDIT.BASE / "published-100us.log.gz").decode().splitlines()
        i = next(i for i, line in enumerate(lines) if line.startswith("prepared_path_summary="))
        summary = json.loads(lines[i].split("=", 1)[1])
        summary["normalized_force_tolerance"] = 0.1
        lines[i] = "prepared_path_summary=" + json.dumps(summary)
        with self.assertRaisesRegex(ValueError, "fibre publication gate"):
            AUDIT.audit_probe("\n".join(lines), AUDIT.force_scales(), True)


if __name__ == "__main__":
    unittest.main()

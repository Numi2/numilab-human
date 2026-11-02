import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SharedReferenceFixtureTests(unittest.TestCase):
    def test_row_table_and_fixture_domain(self):
        rows_path = ROOT / "fixture" / "selected-tet-rows.csv"
        fixture_path = ROOT / "fixture" / "selected-rest-f32.bin"
        with rows_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["tet_row"])
        ids = [int(row[0]) for row in rows[1:]]
        self.assertEqual(len(ids), 4329)
        self.assertEqual(ids, sorted(set(ids)))
        with fixture_path.open("rb") as stream:
            count = struct.unpack("<Q", stream.read(8))[0]
        self.assertEqual(count, len(ids))
        self.assertEqual(fixture_path.stat().st_size, 8 + count * 12 * 4)

    def test_report_keeps_diagnostic_only_status(self):
        report = json.loads((ROOT / "evidence" / "diagnostic-report.json").read_text())
        self.assertEqual(report["status"], "COMPLETED_SHARED_ARITHMETIC_DIAGNOSTIC_NO_NATIVE_EXECUTION")
        self.assertFalse(report["physical_admission"])
        self.assertEqual(report["native_steps"], 0)
        self.assertFalse(report["gpu_executed"])
        self.assertEqual(report["selected_rows"], 4329)
        self.assertEqual([row["nonpositive_or_nonfinite_J"] for row in report["arithmetic"]["cases"]], [0, 0, 0])

    def test_pinned_cpu_replay(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "verify_fixture.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "PASS_CPU_FIXTURE_REPLAY_NO_NATIVE")
        self.assertFalse(receipt["gpu_executed"])
        self.assertEqual(receipt["native_steps"], 0)
        self.assertFalse(receipt["physical_admission"])


if __name__ == "__main__":
    unittest.main()

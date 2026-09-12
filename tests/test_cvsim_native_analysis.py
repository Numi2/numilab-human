"""Small arithmetic evidence-tamper tests; no physiological simulation."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cvsim_native_analysis", ROOT / "tools/analyze_cvsim_native.py")
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class NativeTraceArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trace = Path(self.temp.name) / "fixture.csv"
        self.log = Path(self.temp.name) / "fixture.log"
        initial = json.loads((ANALYZE.SOURCE_EVIDENCE / "original-initial.json").read_text())
        self.reference = [[value * 1e-6 for value in initial["volume_mL"]] + [0.] * 24 for _ in range(10)]
        self.dt = ANALYZE.float32(.002)
        self.rows = []
        for step in (1, 2):
            expected = self.reference[step * 4]
            actual = expected.copy()
            actual[0] += 1e-8
            actual[1] -= 1e-8
            actual[22] += 2e-8
            self.rows.append([step * self.dt] + [value for pair in zip(actual, expected) for value in pair])
        self.header = ANALYZE.NATIVE_HEADER.copy()
        self.write_fixture()

    def write_fixture(self, *, coordinates="upstream_equation"):
        with self.trace.open("w", newline="") as file:
            csv.writer(file, lineterminator="\n").writerows([self.header, *self.rows])
        scale = ANALYZE.STATE_SCALE
        shift = ANALYZE.coordinate_translation(coordinates)
        initial = [ANALYZE.float32(ANALYZE.float32(a + b) / scale) * scale
                   for a, b in zip(self.reference[0], shift)]
        initial_sum = sum(initial[:21])
        max_volume = max_flow = drift = square = 0.
        for row in self.rows:
            actual, expected = row[1::2], row[2::2]
            errors = [abs(a - b) for a, b in zip(actual, expected)]
            max_volume = max(max_volume, max(errors[:21]))
            max_flow = max(max_flow, max(errors[21:]))
            square += sum((error / scale) ** 2 for error in errors)
            drift = max(drift, abs(sum(actual[:21]) - initial_sum) / initial_sum)
        self.log.write_text(
            "cvsim_device=Apple M4 Pro abi=28 world_fingerprint=1\n"
            "cvsim_source_run=pass accepted_steps=2 environments=2 failed_steps=0 "
            f"dt_seconds={self.dt!r} duration_seconds={2*self.dt!r} "
            "clock=exact_binary_128_rational_period replay_pair=bitwise snapshot_replay=bitwise "
            f"max_volume_error_m3={max_volume!r} max_flow_error_m3_per_s={max_flow!r} "
            f"scaled_state_rms_error={math.sqrt(square/(45*len(self.rows)))!r} "
            f"relative_blood_volume_invariant_error={drift!r} wall_seconds=1 "
            "qualification=source_variant_numerical_comparison biological_calibration=unqualified\n")

    def check(self, *, coordinates="upstream_equation"):
        return ANALYZE.audit_run(self.trace, self.log, dt=.002, steps=2,
                                 reference=self.reference, coordinates=coordinates)

    def test_complete_fixture_is_independently_recomputed(self):
        result = self.check()
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["max_volume_error_m3"], 1e-8)
        self.assertAlmostEqual(result["max_flow_error_m3_per_s"], 2e-8)

    def test_poisoned_reference_column_rejected_after_summary_resealed(self):
        self.rows[0][2] += 1e-7
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "independent source"):
            self.check()

    def test_shifted_sample_time_rejected(self):
        self.rows[0][0] = .002
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "accepted clock"):
            self.check()

    def test_equal_total_cannot_hide_negative_compartment(self):
        previous = self.rows[0][1]
        self.rows[0][1] = -1e-6
        self.rows[0][3] += previous + 1e-6
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "nonpositive absolute"):
            self.check()

    def test_negative_one_way_flow_rejected(self):
        self.rows[0][1 + 2 * 21] = -1e-12
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "negative one-way"):
            self.check()

    def test_total_blood_loss_rejected_after_summary_resealed(self):
        self.rows[0][1] += 1e-5
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "conservation"):
            self.check()

    def test_missing_coordinate_rejected(self):
        self.rows[0].pop()
        with self.trace.open("w", newline="") as file:
            csv.writer(file).writerows([self.header, *self.rows])
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "coordinate count"):
            self.check()

    def test_duplicate_coordinate_header_rejected(self):
        self.header[2] = "native_0"
        self.write_fixture()
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "paired SI coordinates"):
            self.check()

    def test_missing_accepted_row_rejected(self):
        with self.trace.open("w", newline="") as file:
            csv.writer(file).writerows([self.header, self.rows[0]])
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "row count is incomplete"):
            self.check()

    def test_extra_accepted_row_rejected(self):
        with self.trace.open("a", newline="") as file:
            csv.writer(file).writerow(self.rows[-1])
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "extra native rows"):
            self.check()

    def test_false_summary_rejected(self):
        self.log.write_text(self.log.read_text().replace("max_flow_error_m3_per_s=2e-08", "max_flow_error_m3_per_s=1"))
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "independently recomputed"):
            self.check()

    def test_failure_cannot_be_hidden_by_later_pass(self):
        self.log.write_text("cvsim_source_run=failed reason=retained\n" + self.log.read_text())
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "failed source run"):
            self.check()

    def test_heldt_translation_must_be_explicit(self):
        translation = ANALYZE.coordinate_translation("heldt_table_aligned")
        self.assertEqual(sum(translation), 0)
        self.assertEqual(translation[2], 184e-6)
        self.assertEqual(translation[5], -184e-6)
        for row in self.rows:
            for i, shift in enumerate(translation):
                row[1 + i * 2] += shift
                row[2 + i * 2] += shift
        self.write_fixture(coordinates="heldt_table_aligned")
        self.assertEqual(self.check(coordinates="heldt_table_aligned")["status"], "pass")
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "independent source"):
            self.check()

    def test_nonfinite_is_rejected(self):
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "nonfinite"):
            ANALYZE.numbers(["1", "nan"], 2, "fixture")


class NativeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.runs = {name: {"duration_seconds": 429 * ANALYZE.float32(.002),
                            **{key: 1. / (2**index) for key in ANALYZE.METRICS}}
                     for index, name in enumerate(ANALYZE.REFINEMENT_RUNS)}

    def test_all_error_groups_must_show_order(self):
        self.assertEqual(ANALYZE.evaluate_refinement(self.runs)["status"], "pass")
        self.runs["cycle-0p5ms"]["max_flow_error_m3_per_s"] = .6
        result = ANALYZE.evaluate_refinement(self.runs)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["metrics"]["max_flow_error_m3_per_s"]["passed"])

    def test_equal_duration_is_required(self):
        self.runs["cycle-0p5ms"]["duration_seconds"] *= .5
        with self.assertRaisesRegex(ANALYZE.EvidenceError, "same elapsed duration"):
            ANALYZE.evaluate_refinement(self.runs)

    def test_zero_error_does_not_fabricate_measured_order(self):
        self.runs["cycle-0p5ms"]["max_flow_error_m3_per_s"] = 0
        result = ANALYZE.evaluate_refinement(self.runs)
        self.assertEqual(result["status"], "fail")
        self.assertIsNone(result["metrics"]["max_flow_error_m3_per_s"]["observed_orders"][1])

    def test_cooked_grid_subsets_are_exact(self):
        self.assertEqual(429 * ANALYZE.float32(.002), 1716 * ANALYZE.float32(.0005))
        self.assertNotEqual(429 * .002, 429 * ANALYZE.float32(.002))


if __name__ == "__main__":
    unittest.main()

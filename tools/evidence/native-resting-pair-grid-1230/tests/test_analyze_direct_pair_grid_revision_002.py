import importlib.util
import struct
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "source" / "analyze_direct_pair_revision-002.py"
spec = importlib.util.spec_from_file_location("direct_pair_v2", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(steps, *, dt=0.002, time_overrides=None):
    time_overrides = time_overrides or {}
    native_dt = struct.unpack("<f", struct.pack("<f", dt))[0]
    return [{"step": step, "time_s": time_overrides.get(step, float(format(step * native_dt, ".12g")))} for step in steps]


class AcceptedGridTests(unittest.TestCase):
    def check(self, left, right, *, steps=24, dt=0.002):
        return module.verify_matched_accepted_grid(left, right, dt=dt, steps=steps, stride=8)

    def test_identical_complete_grid_passes(self):
        result = self.check(rows([8, 16, 24]), rows([8, 16, 24]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["row_count"], 3)
        self.assertEqual((result["first_step"], result["last_step"]), (8, 24))

    def test_missing_row_in_one_arm_fails(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check(rows([8, 16, 24]), rows([8, 24]))

    def test_missing_row_in_both_arms_still_fails_expected_grid(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check(rows([8, 24]), rows([8, 24]))

    def test_duplicate_step_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            self.check(rows([8, 16, 16, 24]), rows([8, 16, 24]))

    def test_out_of_order_step_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            self.check(rows([8, 24, 16]), rows([8, 16, 24]))

    def test_shifted_time_outside_declared_dt_tolerance_fails(self):
        with self.assertRaisesRegex(ValueError, "shifted from step\*Float32\(dt\)"):
            self.check(rows([8, 16, 24], time_overrides={16: 0.1}), rows([8, 16, 24]))

    def test_arm_time_shift_is_rejected_even_with_same_step_grid(self):
        with self.assertRaisesRegex(ValueError, "shifted from step\*Float32\(dt\)"):
            self.check(rows([8, 16, 24]), rows([8, 16, 24], time_overrides={16: 0.032000001520418}))

    def test_fractional_step_fails(self):
        with self.assertRaisesRegex(ValueError, "nonfinite/noninteger"):
            self.check(rows([8, 16, 24]), [{"step": 8, "time_s": rows([8])[0]["time_s"]},
                                          {"step": 16.5, "time_s": rows([16])[0]["time_s"]},
                                          {"step": 24, "time_s": rows([24])[0]["time_s"]}])

    def test_incompatible_declared_grid_fails(self):
        with self.assertRaisesRegex(ValueError, "invalid declared accepted grid"):
            module.verify_matched_accepted_grid(rows([8, 16]), rows([8, 16]), dt=0.002, steps=15, stride=8)


if __name__ == "__main__":
    unittest.main()

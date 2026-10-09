import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("postrun_descriptive_supplement.py")
spec = importlib.util.spec_from_file_location("descriptive_supplement_tested", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DescriptiveSupplementValidation(unittest.TestCase):
    def write_csv(self, content):
        temp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        temp.write(content)
        temp.close()
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return Path(temp.name)

    def test_missing_required_column_refuses(self):
        path = self.write_csv("time_s,step,blood_ml\n0,0,1\n1,1,2\n")
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            mod.read_trace(path, ("time_s", "step", "blood_ml", "PaO2_mmhg"))

    def test_duplicate_required_column_refuses(self):
        path = self.write_csv("time_s,step,blood_ml,blood_ml\n0,0,1,1\n1,1,2,2\n")
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            mod.read_trace(path, ("time_s", "step", "blood_ml"))

    def test_nonfinite_required_value_refuses(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            mod.number({"value": "NaN"}, "value")

    def test_empty_window_refuses(self):
        rows = [{"time_s": "1"}]
        with self.assertRaisesRegex(ValueError, "no accepted trace samples"):
            mod.select_window(rows, 2.0, 3.0)

    def test_registered_window_flags_come_from_started_payload_not_native_scene_argv(self):
        item = {
            "started": {"payload": {"argv": ["numi", "--start-s", "60.0", "--end-s", "100.0", "--window-s", "30.0"]}},
            "inv": {"argv": ["native", "--resting-scene", "reference.json", "/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-reference-respiration.json"]},
        }
        self.assertEqual(mod.registered_windows(item, 155000, 0.002),
                         {"pre": [30.0, 60.0], "dose": [70.0, 100.0], "recovery": [280.0, 310.0]})
        self.assertEqual(mod.extract_respiration_config(item["inv"]), Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-reference-respiration.json"))

    def test_terminal_time_is_required_and_matches_capture_schedule(self):
        context = {"geometry": {"terminal_step": 155000, "terminal_accepted_time_s": 310.0000147242099},
                   "schedule": {"terminal": {"accepted_time_s": 310.0000147242099}}}
        self.assertAlmostEqual(mod.checked_terminal_time(context, 155000), 310.0000147242099)
        context["geometry"]["terminal_accepted_time_s"] = None
        with self.assertRaisesRegex(ValueError, "missing or malformed"):
            mod.checked_terminal_time(context, 155000)

    def test_counter_interval_multiplicity_is_not_collapsed_to_single_beats(self):
        rows = [
            {"time_s": "10", "complete_filling_ejection_cycles": "4"},
            {"time_s": "11", "complete_filling_ejection_cycles": "6"},
            {"time_s": "12", "complete_filling_ejection_cycles": "6"},
            {"time_s": "13", "complete_filling_ejection_cycles": "7"},
        ]
        got = mod.cycle_multiplicity(rows, 10.0, 14.0)
        self.assertEqual(got["counter_advance_intervals"], 2)
        self.assertEqual(got["complete_counter_cycles"], 3)
        self.assertEqual(got["interval_multiplicity_histogram"], {"2": 1, "1": 1})
        self.assertEqual(got["intervals_with_more_than_one_counter_cycle"], 1)

    def test_descriptive_response_keeps_per_arm_recovery_minus_pre(self):
        rows = []
        for t, oxygen, co2, sat in ((1, 90, 40, .96), (2, 92, 39, .97), (3, 91, 40, .96)):
            rows.append({"time_s": str(t), "PaO2_mmhg": str(oxygen), "PaCO2_mmhg": str(co2),
                         "SaO2": str(sat), "lung_volume_ml": "2500", "alveolar_pa": "0",
                         "pleural_pa": "-500", "airflow_ml_s": "0"})
        got = mod.window_response(rows, {"pre": [1, 2], "dose": [2, 3], "recovery": [3, 4]})
        self.assertAlmostEqual(got["descriptive_changes"]["PaO2_mmhg"]["recovery_minus_pre"], 1.0)
        self.assertAlmostEqual(got["descriptive_changes"]["PaCO2_mmhg"]["recovery_minus_pre"], 0.0)


if __name__ == "__main__":
    unittest.main()

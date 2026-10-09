import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("postrun_descriptive_supplement.py")
spec = importlib.util.spec_from_file_location("descriptive_supplement_tested", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DescriptiveSupplementValidation(unittest.TestCase):
    def test_closed_trace_owner_numeric_parser_contract(self):
        analyzer,prepare,adapter,owner=mod.load_helpers()
        root=Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-baseline")
        ctx=adapter.load_registered_context(root,"control")
        numeric=owner.read_trace(ctx["item"]["csv"])
        self.assertIsInstance(numeric[0]["time_s"],float)
        result=owner.resting_reference_comparison(numeric,30.0,60.0)
        item=result["general_adult_resting_reference_comparisons"]["mean_PaCO2_mmhg"]
        expected=sum(r["PaCO2_mmhg"] for r in numeric if 30<=r["time_s"]<60)/sum(30<=r["time_s"]<60 for r in numeric)
        self.assertAlmostEqual(item["measured"],expected,places=10)

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

    def test_counter_interval_multiplicity_narration_matches_histogram(self):
        one_each = [
            {"time_s": "10", "complete_filling_ejection_cycles": "4"},
            {"time_s": "11", "complete_filling_ejection_cycles": "5"},
            {"time_s": "12", "complete_filling_ejection_cycles": "5"},
            {"time_s": "13", "complete_filling_ejection_cycles": "6"},
        ]
        got_one = mod.cycle_multiplicity(one_each, 10.0, 14.0)
        self.assertEqual(got_one["interval_multiplicity_histogram"], {"1": 2})
        self.assertEqual(got_one["intervals_with_more_than_one_counter_cycle"], 0)
        self.assertIn("incremented once", got_one["interpretation"])
        self.assertNotIn("include multiple", got_one["interpretation"])
        multiple = [
            {"time_s": "10", "complete_filling_ejection_cycles": "4"},
            {"time_s": "11", "complete_filling_ejection_cycles": "6"},
            {"time_s": "12", "complete_filling_ejection_cycles": "6"},
            {"time_s": "13", "complete_filling_ejection_cycles": "7"},
        ]
        got_multiple = mod.cycle_multiplicity(multiple, 10.0, 14.0)
        self.assertEqual(got_multiple["interval_multiplicity_histogram"], {"2": 1, "1": 1})
        self.assertEqual(got_multiple["intervals_with_more_than_one_counter_cycle"], 1)
        self.assertIn("include multiple", got_multiple["interpretation"])

    def test_sao2_fraction_is_converted_before_percent_reference_comparison(self):
        rows = [{"SaO2": "0.95"}, {"SaO2": "0.949"}, {"SaO2": "0.97"}]
        stats = mod.stats_scaled(rows, "SaO2", 100.0, "percent")
        counts = mod.reference_outlier_counts(rows, "SaO2", [95.0, None], scale=100.0, unit="percent")
        self.assertAlmostEqual(stats["mean"], 95.63333333333334)
        self.assertEqual(stats["unit"], "percent")
        self.assertEqual(stats["source_to_report_scale"], 100.0)
        self.assertEqual(counts["below_lower"], 1)
        self.assertEqual(counts["above_upper"], 0)
        self.assertEqual(counts["reported_unit"], "percent")
        self.assertEqual(counts["source_unit"], "fraction")

    def test_descriptive_response_keeps_per_arm_recovery_minus_pre(self):
        rows = []
        for t, oxygen, co2, sat in ((1, 90, 40, .96), (2, 92, 39, .97), (3, 91, 40, .96)):
            rows.append({"time_s": str(t), "PaO2_mmhg": str(oxygen), "PaCO2_mmhg": str(co2),
                         "SaO2": str(sat), "lung_volume_ml": "2500", "alveolar_pa": "0",
                         "pleural_pa": "-500", "airflow_ml_s": "0"})
        got = mod.window_response(rows, {"pre": [1, 2], "dose": [2, 3], "recovery": [3, 4]})
        self.assertAlmostEqual(got["descriptive_changes"]["PaO2_mmhg"]["recovery_minus_pre"], 1.0)
        self.assertAlmostEqual(got["descriptive_changes"]["PaCO2_mmhg"]["recovery_minus_pre"], 0.0)
        self.assertEqual(got["dose"]["SaO2_percent"]["unit"], "percent")
        self.assertAlmostEqual(got["descriptive_changes"]["SaO2_percent"]["dose_minus_pre"], 1.0)
        self.assertEqual(got["descriptive_changes"]["SaO2_percent"]["unit"], "percent")


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170-review/review_registered_arm_1174.py")
spec = importlib.util.spec_from_file_location("review_registered_arm_1174_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class RecordingReviewTests(unittest.TestCase):
    def test_inspector_summary_and_all_snapshots(self):
        text = "\n".join([
            "frames=12 timing_markers=1 first_wall_s=0.0 last_wall_s=3.5 max_gap_wall_s=0.5 duration_wall_s=4.0",
            "frame=initial index=0 simulated_s=0.0 wall_s=0.0 width=1280 height=900",
            "frame=middle index=6 simulated_s=2.0 wall_s=2.0 width=1280 height=900",
            "frame=final index=11 simulated_s=3.5 wall_s=3.5 width=1280 height=900",
            *[f"frame={name} index={i} simulated_s={i/2} wall_s={i/2} width=1280 height=900"
              for i, name in enumerate(("skin","muscles","skeleton","organs","lungs","heart","vessels"), 3)],
        ])
        summary, frames = module.parse_inspector_output(text, 12)
        self.assertEqual(summary["frame_count"], 12)
        self.assertFalse(summary["original_compressed_sample_decode_order_verified"])
        self.assertEqual(set(frames), set(module.EXPECTED_SNAPSHOTS))

    def test_inspector_rejects_missing_layer_and_count_mismatch(self):
        text = "frames=2 timing_markers=0 first_wall_s=0 last_wall_s=1 max_gap_wall_s=1 duration_wall_s=1\n"
        with self.assertRaisesRegex(ValueError, "frame count"):
            module.parse_inspector_output(text, 3)
        with self.assertRaisesRegex(ValueError, "all seven layer snapshots"):
            module.parse_inspector_output(text, 2)

    def test_surface_timeline_requires_strictly_increasing_finite_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surface.csv"
            path.write_text("time_s,value\n0,1\n0.5,2\n1,3\n")
            result = module.surface_time_summary(path)
            self.assertEqual(result["data_rows"], 3)
            path.write_text("time_s,value\n0,1\n0,2\n")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                module.surface_time_summary(path)

if __name__ == "__main__":
    unittest.main()

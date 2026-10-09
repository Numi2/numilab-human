import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return json.loads((ROOT / rel).read_text())

class PublicationBundleTest(unittest.TestCase):
    def test_observer_run_and_complete_streams(self):
        pins = read("run/build-pins.json")
        meta = read("run/run-metadata.json")
        inv = read("run/invocation.json")
        report = read("reports/observer-integrity/report.json")
        self.assertEqual(pins["schema"], "numi.human.support-drift-observer-build-pins.v1")
        self.assertEqual(meta["exit_code"], 0)
        self.assertEqual(meta["source_files_changed_during_run"], [])
        self.assertTrue(meta["loaded_metal_runtime"]["verified"])
        self.assertEqual(inv["asset_sha256"][meta["loaded_metal_runtime"]["expected_path"]],
                         meta["loaded_metal_runtime"]["expected_sha256"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["accepted_steps"]["count"], 19375)
        self.assertEqual(report["counts"]["body_motion_partition"]["body_count"], 157)
        self.assertEqual(report["counts"]["selected_skin_witness"]["regions_per_step"], 32)
        self.assertAlmostEqual(report["profile"]["mean_ms"], 6.0978374, places=7)

    def test_parsed_csv_and_geometry_scope(self):
        csv_report = read("reports/comparisons/parsed-csv-parity.json")
        self.assertEqual(csv_report["status"], "exact_candidate_prefix_on_shared_rows")
        for channel in csv_report["channels"].values():
            self.assertEqual(channel["numeric_or_text_value_mismatches"], 0)
            self.assertEqual(channel["reference_only_keys"], 0)
            self.assertEqual(channel["candidate_only_keys"], 0)
        geom = read("reports/comparisons/geometry-sections-2-through-5.json")
        self.assertEqual(geom["status"], "passed")
        self.assertEqual(len(geom["captures"]), 8)
        for capture in geom["captures"]:
            self.assertTrue(capture["all_geometry_sections_byte_identical"])
            self.assertEqual(set(capture["candidate"]["sections"]), {"2", "3", "4", "5"})

    def test_root_frame_identity_and_late_contact_observations(self):
        d = read("reports/root-frame-decomposition/report.json")
        late = d["decompositions"]["250s_to_terminal"]
        self.assertLess(max(map(abs, late["decomposition_residual_m"])), 1e-12)
        self.assertEqual(late["steps"], [125000, 155000])
        motion = read("reports/motion-250s-to-310s/report.json")
        self.assertEqual(motion["requested_accepted_steps"], [125000, 155000])
        self.assertLess(max(map(abs, motion["partition_identity_residual_m"])), 1e-12)
        self.assertAlmostEqual(motion["owner_total_com_delta_m"][0], late["world_com_delta_m"][0], places=14)
        region = motion["support_regions"]["22"]
        self.assertEqual(region["sampled_winner_changes"], 476)
        self.assertEqual(region["sampled_winner_changes_with_positive_normal_impulse"], 476)
        self.assertTrue(math.isfinite(region["positive_normal_pre_step_tangential_speed_mean_m_s"]))

    def test_motion_validation_pins_published_analyzer(self):
        validation = read("reports/motion-analysis-validation.json")
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["script_sha256"], "3fe71acfcc73789730cc3a661feb41ca70f5760583a88c4ea317cfa5e05ee652")
        endpoint = read("reports/endpoint-velocity-consistency/report.json")
        self.assertEqual(len(endpoint["windows"]), 2)
        self.assertEqual(endpoint["windows"][1]["cadence_steps"], 8)

    def test_readme_has_no_shell_marker_and_documents_rerun_outputs(self):
        text = (ROOT / "README.md").read_text()
        self.assertNotIn("REMOTE_E", text)
        self.assertIn("replace their adjacent report.json", text)
        self.assertIn("refuses an existing report.json", text)

    def test_publication_manifest_hashes_every_bundle_file(self):
        import hashlib
        manifest = read("publication-manifest.json")
        listed = manifest["files"]
        disk = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                if p.is_file() and p.name != "publication-manifest.json"
                and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        self.assertEqual(set(listed), disk)
        for rel, expected in listed.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected, rel)
        closure = read("closure.json")
        self.assertEqual(closure["status"], "observer_integrity_passed_drift_unresolved")
        checks = [
            (closure["motion_analysis"]["analyzer"], closure["motion_analysis"]["analyzer_sha256"]),
            ("reproduction/root-frame-decomposition/decompose.py", closure["drift_decomposition"]["script_sha256"]),
            ("reproduction/endpoint-velocity-consistency/endpoint_consistency.py", closure["endpoint_velocity_consistency"]["script_sha256"]),
            (closure["observer_integrity"]["report"], closure["observer_integrity"]["report_sha256"]),
            (closure["legacy_parity"]["report"], closure["legacy_parity"]["report_sha256"]),
            (closure["geometry_parity"]["report"], closure["geometry_parity"]["report_sha256"]),
            (closure["drift_decomposition"]["report"], closure["drift_decomposition"]["report_sha256"]),
            (closure["endpoint_velocity_consistency"]["report"], closure["endpoint_velocity_consistency"]["report_sha256"]),
        ]
        for rel, expected in checks:
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected, rel)

if __name__ == "__main__":
    unittest.main()

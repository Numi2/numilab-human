"""Mutation tests of copied evidence; no native execution or physical stepping."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("organ_circulation_verifier", ROOT / "tools/verify_organ_circulation_20260912.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class OrganCirculationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name) / "repository"
        self.bundle = self.repository / "evidence"
        shutil.copytree(ROOT / "Docs/media/organ-circulation-20260912", self.bundle)
        self.receipt_path = self.bundle / "receipt.json"
        self.receipt = VERIFY.read_json(self.receipt_path)
        for relative in self.receipt["source_sha256"]:
            target = self.repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)

    def run_verifier(self) -> dict:
        self.receipt_path.write_text(json.dumps(self.receipt))
        return VERIFY.verify(self.receipt_path, repository_root=self.repository)

    def reseal(self, name: str) -> None:
        for record in self.receipt["artifacts"]:
            if record["path"] == name:
                record["sha256"] = VERIFY.file_hash(self.bundle / name)
                return
        self.fail("test attempted to seal an unlisted artifact")

    def modify_log(self, old: str, new: str, name: str = "native-tests-detailed.txt") -> None:
        path = self.bundle / name
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))
        self.reseal(name)

    def rejected(self, expected: str) -> None:
        with self.assertRaisesRegex(VERIFY.EvidenceError, expected):
            self.run_verifier()

    def test_retained_receipt_proves_only_bounded_native_boundary(self) -> None:
        result = self.run_verifier()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scientific_status"], "unqualified")
        self.assertEqual(result["native_tests_passed"], 11)
        self.assertIn("not_full_human", result["boundary"])
        self.assertEqual(len(result["vascular_cases"]), 5)

    def test_artifact_hash_drift_rejected(self) -> None:
        with (self.bundle / "native-tests-detailed.txt").open("a") as stream:
            stream.write("uncaptured change\n")
        self.rejected("artifact hash drift")

    def test_live_human_source_drift_rejected(self) -> None:
        with (self.repository / "src/numilab_human/physiology.py").open("a") as stream:
            stream.write("\n# source drift\n")
        self.rejected("Human source hash drift")

    def test_resealed_log_cannot_drop_a_required_case(self) -> None:
        path = self.bundle / "native-tests-detailed.txt"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
                                  if not line.startswith("vascular_case=inertial_flow ")) + "\n")
        self.reseal(path.name)
        self.rejected("vascular case coverage")

    def test_scientific_promotion_rejected(self) -> None:
        self.receipt["scientific_status"] = "qualified"
        self.rejected("receipt does not meet")

    def test_resealed_invalid_error_metric_rejected(self) -> None:
        self.modify_log("fp64_normalized_max=1.535277601e-07", "fp64_normalized_max=0.00008")
        self.rejected("fp64_normalized_max exceeds")

    def test_resealed_nonfinite_metric_rejected(self) -> None:
        self.modify_log("relative_conservation_max=3.628520057e-08", "relative_conservation_max=nan")
        self.rejected("invalid relative_conservation_max")

    def test_resealed_failed_conservation_rejected(self) -> None:
        self.modify_log("relative_conservation_max=6.072223191e-08", "relative_conservation_max=0.1")
        self.rejected("relative_conservation_max exceeds")

    def test_resealed_rollback_failure_rejected(self) -> None:
        self.modify_log("isolated_rollback=pass", "isolated_rollback=fail")
        self.rejected("vascular case does not meet")

    def test_resealed_nondecreasing_refinement_rejected(self) -> None:
        self.modify_log("errors_m3=4.54144036e-09,2.296729443e-09,1.15518129e-09", "errors_m3=1e-9,2e-9,3e-9")
        self.rejected("timestep refinement fails")

    def test_resealed_admission_coverage_loss_rejected(self) -> None:
        self.modify_log("negative_cases=12", "negative_cases=11")
        self.rejected("native input admission does not meet")

    def test_resealed_alternate_runtime_or_biological_claim_rejected(self) -> None:
        self.modify_log("runtime_input=nmatterpack", "runtime_input=host_loop")
        self.rejected("native qualification does not meet")

    def test_ctest_summary_cannot_hide_missing_test(self) -> None:
        path = self.bundle / "native-ctest.txt"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
                                  if "1/11 Test " not in line) + "\n")
        self.reseal(path.name)
        self.rejected("CTest case coverage")

    def test_native_dirty_or_wrong_version_identity_rejected(self) -> None:
        path = self.bundle / "native-identity.json"
        identity = VERIFY.read_json(path)
        identity["worktree_status"] = " M matter/src/runtime.mm"
        path.write_text(json.dumps(identity))
        self.reseal(path.name)
        self.rejected("native identity does not meet")

    def test_missing_artifact_or_owner_pin_rejected(self) -> None:
        del self.receipt["source_sha256"]["src/numilab_human/physiology.py"]
        self.rejected("owning Human source pin is missing")

    def test_unsafe_relative_path_rejected(self) -> None:
        self.receipt["artifacts"][0]["path"] = "../outside.txt"
        self.rejected("unsafe evidence path")

    def test_uncaptured_artifact_rejected(self) -> None:
        (self.bundle / "later_failure.txt").write_text("failure not captured in receipt\n")
        self.rejected("inventory does not capture")

    def test_anatomical_missing_parameters_cannot_be_relabelled_complete(self) -> None:
        path = self.bundle / "anatomy-validation.json"
        anatomy = VERIFY.read_json(path)
        anatomy["missing_parameters"] = []
        anatomy["calibration_required"] = False
        path.write_text(json.dumps(anatomy))
        self.reseal(path.name)
        self.rejected("preserve all unresolved calibration")


if __name__ == "__main__":
    unittest.main()

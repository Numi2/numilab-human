"""Focused controls for mass trial status, without geometry/flow execution."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "mass_gate_1045", Path(__file__).with_name("run_refined_mass_gate_1045.py"))
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class TrialStatusCheck(unittest.TestCase):
    passed = "child_reference_transfer=PASS method=conservativeSpatialMassTransport mass_kg_parent=13.9"

    def test_explicit_mass_pass(self):
        self.assertTrue(owner.trial_passed(0, self.passed, ""))

    def test_successful_import_does_not_admit_mass(self):
        self.assertFalse(owner.trial_passed(0, "child_tetgen_reference_import=PASS", ""))

    def test_zero_exit_with_rejection_fails(self):
        self.assertFalse(owner.trial_passed(0, self.passed,
            "child_reference_transfer=REJECTED reason=flow_crosses_donor"))

    def test_nonzero_exit_with_pass_fails(self):
        self.assertFalse(owner.trial_passed(6, self.passed, ""))

    def test_duplicate_pass_fails(self):
        self.assertFalse(owner.trial_passed(0, self.passed + "\n" + self.passed, ""))

    def test_embedded_pass_token_fails(self):
        self.assertFalse(owner.trial_passed(0, "not_" + self.passed, ""))

    def test_wrong_method_suffix_fails(self):
        self.assertFalse(owner.trial_passed(0, self.passed.replace(
            "conservativeSpatialMassTransport", "conservativeSpatialMassTransportWrong"), ""))

    def test_missing_declared_trial_fails(self):
        self.assertEqual(owner.classify_trials([]), "FAIL")

    def test_passing_sensitivity_cannot_admit_failed_declared(self):
        self.assertEqual(owner.classify_trials([
            dict(factor=.5, mass_mapping_pass=True),
            dict(factor=1., mass_mapping_pass=False)]), "FAIL")

    def test_sensitivity_remains_distinct(self):
        self.assertEqual(owner.classify_trials([
            dict(factor=.5, mass_mapping_pass=False),
            dict(factor=1., mass_mapping_pass=True)]), "PASS")

    def test_duplicate_declared_trial_fails(self):
        self.assertEqual(owner.classify_trials([
            dict(factor=1., mass_mapping_pass=True),
            dict(factor=1., mass_mapping_pass=True)]), "FAIL")


if __name__ == "__main__":
    unittest.main()

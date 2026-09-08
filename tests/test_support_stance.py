import unittest

from numilab_human.support_stance import assess


class SupportStanceReceiptTests(unittest.TestCase):
    def summary(self):
        fields = {
            "numi_human_whole_body_support_wrench": "ok",
            "replay": "bitwise", "support_contacts": "10",
            "passive_joint_tissue": "none", "expected_weight_n": "10",
            "total_support_force_n": "10", "max_root_force_residual": "0",
            "internal_balanced": "false",
        }
        fields.update({f"contact_{i}_normal_force_n": "1" for i in range(10)})
        geometry = ("compiled_support_geometry=admissible compiled_support_min_gap_m=0 "
                    "compiled_support_gap_tolerance_m=0.000001 compiled_support_max_separated_force_n=0\n")
        return geometry * 2 + " ".join(f"{key}={value}" for key, value in fields.items())

    def test_wrench_success_retains_internal_failure(self):
        metrics, failures = assess(self.summary(), 0)
        self.assertEqual(failures, [])
        self.assertEqual(metrics["internal_balanced"], "false")

    def test_missing_or_duplicate_summary(self):
        self.assertTrue(assess("", 0)[1])
        self.assertTrue(assess(self.summary() + "\n" + self.summary(), 0)[1])

    def test_process_failure_cannot_be_promoted(self):
        for code in (1, None):
            self.assertTrue(assess(self.summary(), code)[1])

    def test_force_sum_is_independently_checked(self):
        text = self.summary().replace("contact_0_normal_force_n=1", "contact_0_normal_force_n=2")
        self.assertIn("unbalanced_or_inadmissible_wrench", assess(text, 0)[1])

    def test_negative_or_nonfinite_force_is_rejected(self):
        for value in ("-1", "nan", "inf"):
            text = self.summary().replace("contact_0_normal_force_n=1", f"contact_0_normal_force_n={value}")
            self.assertTrue(assess(text, 0)[1])

    def test_duplicate_and_missing_metrics_are_rejected(self):
        self.assertTrue(assess(self.summary() + " replay=bitwise", 0)[1])
        self.assertTrue(assess(self.summary().replace("internal_balanced=false", ""), 0)[1])

    def test_geometry_replay_and_separated_forces_are_required(self):
        self.assertTrue(assess(self.summary().splitlines()[-1], 0)[1])
        for old, new in (("min_gap_m=0", "min_gap_m=-0.01"),
                         ("max_separated_force_n=0", "max_separated_force_n=1"),
                         ("gap_tolerance_m=0.000001", "gap_tolerance_m=nan")):
            self.assertTrue(assess(self.summary().replace(old, new), 0)[1])


if __name__ == "__main__":
    unittest.main()

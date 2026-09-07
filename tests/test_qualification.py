import unittest

from numilab_human.qualification import assess


class QualificationEvidenceTests(unittest.TestCase):
    def summary(self, **changes):
        fields = dict(
            myosim_articulated_bodyparts_bone_visual="ok",
            persistent_completed_steps="16", pectoralis_fascia_steps="16",
            pectoralis_fascia_anchor_reaction_audited_steps="16",
            pectoralis_fascia_replay="bitwise", pectoralis_fascia_rollback="verified",
            stand_deterministic_replay="bitwise", persistent_root_assistance="none",
            tendon_step_transfers="13312", tendon_step_transaction="NHTENDON3",
            tendon_borrowed_consumer="same_command_buffer_exact_snapshot",
            tendon_rigid_state_effect="source_JT_share_replaced_by_same_command_continuum_anchor_reaction",
            pectoralis_fascia_directional_regions="26",
            pectoralis_fascia_directional_material_objects="12",
            pectoralis_fascia_minimum_J="0.99", pectoralis_fascia_applied_force_n="70",
            pectoralis_fascia_anchor_reaction_min_l1_n="3",
            pectoralis_fascia_max_displacement_m="0.0001",
            pectoralis_fascia_coupled_transaction_elapsed_ms="1000",
            pectoralis_fascia_device='"Apple M4 Pro"',
            muscle_force_metal_device='"Apple M4 Pro"',
        )
        fields.update(changes)
        return " ".join(f"{k}={v}" for k, v in fields.items())

    def test_complete_transaction_is_admitted(self):
        metrics, failures = assess(self.summary(), 16, 0)
        self.assertEqual(failures, [])
        self.assertEqual(metrics["pectoralis_fascia_device"], "Apple M4 Pro")

    def test_horizon_replay_reaction_and_device_fail_closed(self):
        for change in (
            {"persistent_completed_steps": "4"},
            {"tendon_step_transfers": "832"},
            {"tendon_rigid_state_effect": "duplicated"},
            {"pectoralis_fascia_anchor_reaction_audited_steps": "15"},
            {"pectoralis_fascia_replay": "none"},
            {"pectoralis_fascia_rollback": "none"},
            {"pectoralis_fascia_minimum_J": "-0.1"},
            {"pectoralis_fascia_anchor_reaction_min_l1_n": "0"},
            {"muscle_force_metal_device": '"CPU"'},
            {"persistent_root_assistance": "enabled"},
        ):
            with self.subTest(change=change):
                self.assertTrue(assess(self.summary(**change), 16, 0)[1])

    def test_nan_in_any_numeric_metric_is_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertTrue(assess(self.summary(unrelated_metric=value), 16, 0)[1])

    def test_missing_duplicate_truncated_and_failed_output_is_rejected(self):
        summary = self.summary()
        for text, code in (("", 0), (summary + ' broken="', 0), (summary + "\n" + summary, 0),
                           (summary + " persistent_completed_steps=16", 0),
                           ("myosim_articulated_bodyparts_bone_visual=ok", 0),
                           (summary, 1), (summary, None)):
            with self.subTest(text=text, code=code):
                self.assertTrue(assess(text, 16, code)[1])

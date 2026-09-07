import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from numilab_human.qualification import PAYLOADS, assess, qualify


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

    def test_failed_run_preserves_source_and_detects_changed_physics_library(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, build, inputs = (base / name for name in ("runtime", "build", "inputs"))
            for folder in (root, build, inputs):
                folder.mkdir()
            for name in PAYLOADS.values():
                (inputs / name).write_bytes(b"paired input")
            for name in ("bin/metalrobo_numilab_human_myosim_visual_probe",
                         "matter/shaders/NumiMatter.metallib", "shaders/MetalRobo.metallib"):
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen runtime")
            (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\n")
            (root / "new.mm").write_text("untracked source\n")
            def git(command, **kwargs):
                args = command[3:]
                value = (b"new.mm\0" if args[0] == "ls-files" else
                         b"exact patch\n" if args[0] == "diff" else
                         b"?? new.mm\n" if args[0] == "status" else b"a" * 40)
                return value.decode() if kwargs.get("text") else value
            def native(command, stdout, stderr, **kwargs):
                stderr.write("human_joint_failure completed_steps=7 matter_status=6\n")
                stderr.write("deformable_contact_failure step=7 slot=1040\n")
                (build / "shaders/MetalRobo.metallib").write_bytes(b"changed during execution")
                return SimpleNamespace(wait=lambda **_: 1)
            args = SimpleNamespace(runtime_root=root, runtime_build=build, input=inputs,
                                   output=base / "receipt", steps=[8, 16], timeout_seconds=1)
            with patch("numilab_human.qualification.platform.system", return_value="Darwin"), \
                    patch("numilab_human.qualification.subprocess.check_output", side_effect=git), \
                    patch("numilab_human.qualification.subprocess.Popen", side_effect=native):
                self.assertEqual(qualify(args), 1)
            receipt = json.loads((args.output / "receipt.json").read_text())
            self.assertEqual(len(receipt["runs"]), 1)
            self.assertEqual((args.output / "runtime.patch").read_bytes(), b"exact patch\n")
            self.assertEqual((args.output / "untracked-source/new.mm").read_text(), "untracked source\n")
            self.assertIn("articulated_metallib", receipt["artifacts"])
            self.assertEqual(receipt["runs"][0]["native_matter_status"], 6)
            self.assertIn("artifact_changed_during_run:articulated_metallib",
                          receipt["runs"][0]["failures"])

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from numilab_human.model import ImportError as HumanImportError
from numilab_human.shi_hose import CONFIG, canonical, compile_source, read_json
from numilab_human.shi_hose_step import CLOCK_NANOSECONDS, SCHEMA, simulate


class ShiHoseStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native, cls.lowering = compile_source(config=read_json(CONFIG))

    def test_source_step_accepts_activation_valves_and_conserves_volume(self) -> None:
        receipt = simulate(self.native, steps=100, timestep_s=1.0e-4)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["qualification"], "source_model_reproduction")
        self.assertEqual(receipt["accepted_steps"], 100)
        self.assertEqual(receipt["rejected_steps"], 0)
        self.assertTrue(receipt["conservation"]["volume_conserved"])
        self.assertGreater(receipt["activation"]["open_valve_evaluation_count"], 0)
        self.assertFalse(receipt["clock"]["exact"])
        self.assertEqual(len(receipt["activation"]["accepted_trace_sha256"]), 64)

    def test_exact_human_clock_is_recorded_and_conserved(self) -> None:
        receipt = simulate(self.native, steps=512, timestep_s=12_500e-9)
        self.assertEqual(receipt["clock"]["required_nanoseconds"], CLOCK_NANOSECONDS)
        self.assertTrue(receipt["clock"]["exact"])
        self.assertTrue(receipt["conservation"]["volume_conserved"])
        self.assertAlmostEqual(receipt["final_state"]["accepted_time_s"], 0.0064)

    def test_rejected_candidate_restores_source_state(self) -> None:
        rejected = simulate(self.native, steps=9, timestep_s=1.0e-4, reject_step=4)
        accepted = simulate(self.native, steps=8, timestep_s=1.0e-4)
        self.assertEqual(rejected["accepted_steps"], 8)
        self.assertEqual(rejected["rejected_steps"], 1)
        self.assertEqual(rejected["final_state"], accepted["final_state"])
        self.assertEqual(rejected["accepted_state_trace_sha256"], accepted["accepted_state_trace_sha256"])
        self.assertTrue(rejected["rollback"]["accepted_time_excludes_rejections"])

    def test_replay_is_canonical_and_does_not_mutate_native_source(self) -> None:
        before = canonical(self.native)
        first = simulate(self.native, steps=16, timestep_s=1.0e-4)
        second = simulate(self.native, steps=16, timestep_s=1.0e-4)
        self.assertEqual(canonical(first), canonical(second))
        self.assertEqual(canonical(self.native), before)

    def test_source_qualification_and_law_are_fail_closed(self) -> None:
        for field, value, message in (
            ("qualification", "fixture_only", "source-model reproduction"),
            ("law", "unbounded", "source law"),
        ):
            with self.subTest(field=field):
                native = copy.deepcopy(self.native)
                native[field] = value
                with self.assertRaisesRegex(HumanImportError, message):
                    simulate(native, steps=1, timestep_s=1.0e-4)

    def test_invalid_timestep_is_rejected(self) -> None:
        with self.assertRaisesRegex(HumanImportError, "timestep"):
            simulate(self.native, steps=1, timestep_s=0.0)

    def test_cli_output_is_immutable(self) -> None:
        from numilab_human.shi_hose_step import run

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "step.json"
            args = type("Args", (), {
                "source_directory": CONFIG.parents[1] / "third_party/physiome/shi_hose_2009",
                "config": CONFIG,
                "steps": 4,
                "timestep_seconds": 1.0e-4,
                "require_clock_nanoseconds": None,
                "reject_step": None,
                "output": output,
            })()
            self.assertEqual(run(args), 0)
            before = output.read_bytes()
            self.assertEqual(run(args), 0)
            self.assertEqual(output.read_bytes(), before)
            output.write_bytes(b"forged\n")
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                run(args)

    def test_cli_exact_clock_requirement_rejects_other_timestep(self) -> None:
        from numilab_human.shi_hose_step import run

        with tempfile.TemporaryDirectory() as temporary:
            args = type("Args", (), {
                "source_directory": CONFIG.parents[1] / "third_party/physiome/shi_hose_2009",
                "config": CONFIG,
                "steps": 1,
                "timestep_seconds": 1.0e-4,
                "require_clock_nanoseconds": CLOCK_NANOSECONDS,
                "reject_step": None,
                "output": Path(temporary) / "step.json",
            })()
            with self.assertRaisesRegex(HumanImportError, "required nanosecond clock"):
                run(args)


if __name__ == "__main__":
    unittest.main()

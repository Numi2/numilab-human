from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import ROOT, canonical, compile_graph, read_json
from numilab_human.physiology_step import SCHEMA, simulate


class PhysiologyStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        graph = read_json(ROOT / "config/physiology-passive-fixture.v1.json")
        cls.native = compile_graph(graph, sources=ROOT / "Sources")

    def test_accepted_transport_exchange_and_conservation(self) -> None:
        receipt = simulate(self.native, steps=32, timestep_s=1.0e-4)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["qualification"], "fixture_only")
        self.assertEqual(receipt["accepted_steps"], 32)
        self.assertEqual(receipt["rejected_steps"], 0)
        self.assertTrue(receipt["conservation"]["volume_conserved"])
        self.assertTrue(receipt["conservation"]["species_conserved"])
        self.assertGreater(
            receipt["final_state"]["tissue_reservoirs"][5]["species_mol"][0], 0.0
        )

    def test_rejected_candidate_does_not_advance_time_or_state(self) -> None:
        rejected = simulate(self.native, steps=9, timestep_s=1.0e-4, reject_step=4)
        accepted = simulate(self.native, steps=8, timestep_s=1.0e-4)
        self.assertEqual(rejected["accepted_steps"], 8)
        self.assertEqual(rejected["rejected_steps"], 1)
        self.assertAlmostEqual(rejected["final_state"]["accepted_time_s"], 8e-4)
        self.assertEqual(rejected["final_state"], accepted["final_state"])
        self.assertTrue(rejected["rollback"]["accepted_time_excludes_rejections"])

    def test_replay_is_bitwise_deterministic(self) -> None:
        first = simulate(self.native, steps=16, timestep_s=1.0e-4)
        second = simulate(self.native, steps=16, timestep_s=1.0e-4)
        self.assertEqual(canonical(first), canonical(second))

    def test_graph_without_tissue_reservoirs_remains_hydraulically_executable(self) -> None:
        native = copy.deepcopy(self.native)
        native["tissue_reservoirs"] = []
        native["exchanges"] = []
        receipt = simulate(native, steps=4, timestep_s=1.0e-4)
        self.assertEqual(receipt["accepted_steps"], 4)
        self.assertTrue(receipt["conservation"]["volume_conserved"])
        self.assertTrue(receipt["conservation"]["species_conserved"])

    def test_invalid_candidate_is_rejected_without_mutating_inputs(self) -> None:
        native = copy.deepcopy(self.native)
        native["connections"][0]["resistance_pa_s_per_m3"] = -1.0
        with self.assertRaisesRegex(HumanImportError, "resistance"):
            simulate(native, steps=1, timestep_s=1.0e-4)
        self.assertEqual(native["connections"][0]["resistance_pa_s_per_m3"], -1.0)

    def test_cli_output_is_immutable(self) -> None:
        from numilab_human.physiology_step import run
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = ROOT / "config/physiology-passive-fixture.v1.json"
            output = directory / "step.json"
            args = type("Args", (), {
                "graph": graph, "sources": ROOT / "Sources", "source_lock": ROOT / "sources.lock.json",
                "output": output, "steps": 4, "timestep_seconds": 1.0e-4, "reject_step": None,
            })()
            self.assertEqual(run(args), 0)
            before = output.read_bytes()
            self.assertEqual(run(args), 0)
            self.assertEqual(output.read_bytes(), before)
            output.write_bytes(b"forged\n")
            with self.assertRaisesRegex(HumanImportError, "immutable"):
                run(args)


if __name__ == "__main__":
    unittest.main()

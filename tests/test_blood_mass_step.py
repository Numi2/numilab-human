from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from numilab_human.blood_mass_step import SCHEMA, simulate, validate_owner_graph
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import ROOT, canonical, compile_graph, read_json


class BloodMassStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        graph = read_json(ROOT / "config/physiology-passive-fixture.v1.json")
        cls.native = compile_graph(graph, sources=ROOT / "Sources")
        cls.owner = read_json(ROOT / "config/blood-mass-transfer-fixture.v1.json")

    def test_conservative_mass_and_bidirectional_tissue_transfer(self) -> None:
        receipt = simulate(self.native, self.owner, steps=32, timestep_s=1.0e-4)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["qualification"], "fixture_only")
        self.assertTrue(receipt["mass_conservation"]["mass_conserved"])
        self.assertTrue(receipt["mass_conservation"]["owned_volume_conserved"])
        self.assertTrue(receipt["two_way_exchange"]["both_directions_observed"])
        self.assertGreater(receipt["transfer_counts"]["blood_to_tissue"], 0)
        self.assertGreater(receipt["transfer_counts"]["tissue_to_blood"], 0)
        self.assertNotEqual(
            receipt["initial_mass_state"]["tissue_reservoirs"][5]["mass_kg"],
            receipt["final_mass_state"]["tissue_reservoirs"][5]["mass_kg"],
        )

    def test_rejected_candidate_restores_hydraulic_and_mass_state(self) -> None:
        rejected = simulate(self.native, self.owner, steps=9, timestep_s=1.0e-4, reject_step=4)
        accepted = simulate(self.native, self.owner, steps=8, timestep_s=1.0e-4)
        self.assertEqual(rejected["accepted_steps"], 8)
        self.assertEqual(rejected["rejected_steps"], 1)
        self.assertEqual(canonical(rejected["final_mass_state"]), canonical(accepted["final_mass_state"]))
        self.assertEqual(canonical(rejected["final_hydraulic_totals"]), canonical(accepted["final_hydraulic_totals"]))
        self.assertEqual(rejected["accepted_state_trace_sha256"], accepted["accepted_state_trace_sha256"])
        self.assertTrue(rejected["rollback"]["accepted_time_excludes_rejections"])

    def test_owner_identity_is_single_and_hash_bound(self) -> None:
        mutated = copy.deepcopy(self.owner)
        mutated["tissue_owners"][0]["physical_volume_owner_id"] = "blood:pool_a"
        with self.assertRaisesRegex(HumanImportError, "physical volume owner is duplicated"):
            validate_owner_graph(self.native, mutated, steps=2)
        mutated = copy.deepcopy(self.owner)
        mutated["native_graph_sha256"] = "0" * 64
        with self.assertRaisesRegex(HumanImportError, "does not bind"):
            validate_owner_graph(self.native, mutated, steps=2)

    def test_negative_schedule_is_reversible_but_nonfinite_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.owner)
        mutated["exchange_schedules"][0]["flow_schedule_m3_per_s"] = [float("nan")]
        with self.assertRaisesRegex(HumanImportError, "must be finite"):
            validate_owner_graph(self.native, mutated, steps=2)

    def test_non_fixture_native_graph_cannot_be_run(self) -> None:
        native = copy.deepcopy(self.native)
        native["qualification"] = "source_model_variant"
        with self.assertRaisesRegex(HumanImportError, "only fixture graphs"):
            simulate(native, self.owner, steps=1, timestep_s=1.0e-4)

    def test_cli_output_is_immutable(self) -> None:
        from numilab_human.blood_mass_step import run
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "mass.json"
            args = type("Args", (), {
                "graph": ROOT / "config/physiology-passive-fixture.v1.json",
                "owners": ROOT / "config/blood-mass-transfer-fixture.v1.json",
                "sources": ROOT / "Sources", "source_lock": ROOT / "sources.lock.json",
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

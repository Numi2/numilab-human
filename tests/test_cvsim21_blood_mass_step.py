from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from numilab_human import cvsim21, cvsim21_blood_mass_step as mass


ROOT = Path(__file__).resolve().parents[1]


class CVSim21BloodMassStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native, cls.lowering = cvsim21.compile_source()
        cls.owner = cvsim21.read_json(ROOT / "config/cvsim21-blood-mass-owner.v1.json")

    def test_source_owner_covers_absolute_budget_without_anatomical_promotion(self) -> None:
        validated = mass._validate_owner(self.native, copy.deepcopy(self.owner))
        self.assertEqual(len(validated["owners"]), 21)
        self.assertEqual(len({row["physical_volume_owner_id"] for row in validated["owners"].values()}), 21)
        self.assertAlmostEqual(
            math.fsum(row["initial_mass_kg"] for row in validated["owners"].values()),
            5.459,
            places=12,
        )
        self.assertEqual(validated["density"], 1060.0)

    def test_exact_clock_mass_transfer_conserves_and_rolls_back(self) -> None:
        receipt = mass.simulate(
            self.native,
            copy.deepcopy(self.owner),
            steps=512,
            timestep_s=mass.CLOCK_SECONDS,
            reject_step=37,
        )
        self.assertEqual(receipt["accepted_steps"], 511)
        self.assertEqual(receipt["rejected_steps"], 1)
        self.assertTrue(receipt["conservation"]["mass_conserved"])
        self.assertTrue(receipt["conservation"]["volume_conserved"])
        self.assertTrue(receipt["rollback"]["accepted_time_excludes_rejections"])
        self.assertTrue(receipt["scope"]["source_aggregate_absolute_blood_mass"])
        self.assertFalse(receipt["scope"]["anatomical_registration"])
        self.assertFalse(receipt["scope"]["material_density_calibrated"])
        self.assertGreater(receipt["transfer_count"], 0)

    def test_rejection_is_state_neutral(self) -> None:
        rejected = mass.simulate(self.native, copy.deepcopy(self.owner), steps=9,
                                 timestep_s=1.0e-4, reject_step=4)
        accepted = mass.simulate(self.native, copy.deepcopy(self.owner), steps=8,
                                 timestep_s=1.0e-4)
        self.assertEqual(rejected["accepted_state_trace_sha256"],
                         accepted["accepted_state_trace_sha256"])
        self.assertEqual(rejected["final_state"], accepted["final_state"])

    def test_density_or_anatomical_promotion_is_rejected(self) -> None:
        altered = copy.deepcopy(self.owner)
        altered["density_provenance"]["kind"] = "source"
        with self.assertRaisesRegex(mass.StepError, "density must remain explicitly unresolved"):
            mass.simulate(self.native, altered, steps=1, timestep_s=mass.CLOCK_SECONDS)
        altered = copy.deepcopy(self.owner)
        altered["compartment_owners"][0]["anatomy_registration"] = "FMA:123"
        with self.assertRaisesRegex(mass.StepError, "cannot promote anatomy"):
            mass.simulate(self.native, altered, steps=1, timestep_s=mass.CLOCK_SECONDS)

    def test_exact_clock_requirement_rejects_wrong_period(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            args = [
                "--steps", "1", "--timestep-seconds", "0.00001",
                "--require-clock-nanoseconds", "12500", "--output", str(output),
            ]
            with self.assertRaises(SystemExit) as error:
                mass.main(args)
            self.assertNotEqual(error.exception.code, 0)
            self.assertFalse(output.exists())

    def test_cli_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            args = ["--steps", "2", "--reject-step", "2", "--output", str(output)]
            self.assertEqual(mass.main(args), 0)
            before = output.read_bytes()
            self.assertEqual(mass.main(args), 0)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(json.loads(before)["accepted_steps"], 1)


if __name__ == "__main__":
    unittest.main()

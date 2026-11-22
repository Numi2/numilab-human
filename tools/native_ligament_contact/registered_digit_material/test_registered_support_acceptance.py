import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from check_cavity_endpoint import endpoint_anatomy
from run_cavity_motion import copy_recruitment_cap, digest, execute_native


class RegisteredSupportAcceptanceTests(unittest.TestCase):
    def test_physical_acceptance_and_clearance_are_separate(self):
        report = dict(
            accepted_steps=2,
            skin_self_intersections=0,
            layer_crossing_pairs=0,
            physical_skin_tissue_pairs=0,
            material_audits=[
                dict(
                    phase="accepted",
                    material=str(i),
                    step="2",
                    nonfinite="0",
                    inverted="0",
                    volume_le_1e_18="0",
                    shape_lt_1e_5="0",
                )
                for i in [15, 16]
            ],
        )
        banks = dict(bone_pairs=0, ocular_source_pairs=10)
        self.assertTrue(endpoint_anatomy(report, banks)["passed"])
        report["material_audits"][1]["shape_lt_1e_5"] = "1"
        report["layer_crossing_pairs"] = 6
        self.assertEqual(
            endpoint_anatomy(report, banks)["failures"],
            {"material_layer_crossings": 6, "fat_shape_lt_1e_5": 1},
        )
        report["material_audits"].pop()
        with self.assertRaises(ValueError):
            endpoint_anatomy(report, banks)

    def test_storage_interruption_retains_receipt_without_admission(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = execute_native(
                ["/bin/sleep", "5"], root, {}, root, minimum_free_bytes=2**62
            )
            self.assertEqual(result, json.loads((root / "execution.json").read_text()))
            self.assertEqual(result["stop_reason"], "disk_reserve")
            self.assertNotEqual(result["exit"], 0)
            self.assertFalse((root / "admission.json").exists())
            result = execute_native(
                ["/usr/bin/true"], root, {}, root, minimum_free_bytes=0
            )
            self.assertEqual(result["exit"], 0)
            self.assertNotIn("stop_reason", result)

    def test_recruitment_requires_matched_assets_and_physical_clock(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            assets = [root / ("asset." + ext) for ext in ("nhrigid", "nhmyo", "nheq")]
            for p in assets:
                p.write_bytes(p.suffix.encode())
            argv = [*map(str, assets), "--muscle-activation", "0.0"]
            control = [*argv[:-1], "0.25"]
            (root / "launch.json").write_text(
                json.dumps(dict(argv=control, pins={str(p): digest(p) for p in assets}))
            )
            receipt = root / "terminal.receipt.json"
            receipt.write_text(
                json.dumps(dict(physical_endpoint="accepted", accepted_time_s=16.0))
            )
            copy = argv.copy()
            self.assertEqual(copy_recruitment_cap(copy, root)["activation_cap"], 0.25)
            self.assertEqual(copy, control)
            half = argv.copy()
            self.assertEqual(
                copy_recruitment_cap(half, root, 0.5)["selected_cap"], 0.125
            )
            self.assertEqual(half[-1], "0.125")
            with self.assertRaises(ValueError):
                copy_recruitment_cap(argv.copy(), root, 1.1)
            receipt.write_text(
                json.dumps(dict(physical_endpoint="rejected", accepted_time_s=16.0))
            )
            with self.assertRaises(ValueError):
                copy_recruitment_cap(argv.copy(), root)
            receipt.write_text(
                json.dumps(dict(physical_endpoint="accepted", accepted_time_s=16.0))
            )
            assets[-1].write_bytes(b"other")
            with self.assertRaises(ValueError):
                copy_recruitment_cap(argv.copy(), root)

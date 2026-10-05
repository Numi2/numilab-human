"""Admission regressions for the native launcher, not simulation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from numilab_human.model import ImportError as HumanImportError
from numilab_human.resting_run import command, run


class RestingRunAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        def asset(name):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        rigid = asset("reference/myosim-fullbody-core-reference.nhrigid")
        asset("reference/myosim-fullbody-muscle-reference.nhmyo")
        asset("reference/myosim-fullbody-joint-equalities.nheq")
        skin, contact, bones, organs = map(asset, ("skin.nhskin", "support.nhcnt", "bones.nhbones", "organs.nhanatomy"))
        surface = asset("bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
        tendon = asset("tendon.nhtendon")
        for name in ("bin/numi-human-native", "lib/libmetalrobo.dylib", "shaders/MetalRobo.metallib",
                     "shaders/MetalRoboHyperPolicy.metallib", "shaders/NumiNeuron.metallib",
                     "matter/shaders/HumanRespiration.metallib", "matter/shaders/NumiMatter.metallib",
                     "matter/shaders/NumiMatterPhysicalStateDigest.metallib",
                     "matter/tools/fixtures/cvsim21.native.v3.json", "matter/examples/resting-reference-respiration.json"):
            asset(name)
        self.scene = {"schema": "numi.human.resting-supine-source-scene.v1",
                      "pose": {"kind": "supine", "root_translation_xyz_m": [0, 0, .1],
                               "root_delta_quaternion_xyzw": [-.7, 0, 0, .714]},
                      "source": {"rigid": rigid, "skin": skin},
                      "outputs": {"support_contact": contact}}
        self.anatomy = {"schema": "numi.human.resting-anatomy-receipt.v1", "payload": organs,
                        "provenance": {"bones_payload": bones["path"], "native_muscle_surfaces": surface},
                        "functional_bindings": {"bones_payload_sha256": bones["sha256"]},
                        "mass_geometry_accounting": {"reference_total_mass_kg": 72}}
        self.args = argparse.Namespace(body_scene=self.root / "scene.json", anatomy_receipt=self.root / "anatomy.json",
                                       tendon=Path(tendon["path"]), build=self.root, lab=self.root,
                                       output=self.root / "run", seconds=310., dimension=512,
                                       mechanics_only=False, inspection_tour=False, drive_intervention=None)
        self.write_receipts()

    def write_receipts(self):
        self.args.body_scene.write_text(json.dumps(self.scene))
        self.args.anatomy_receipt.write_text(json.dumps(self.anatomy))

    def test_launch_binds_native_owners_and_full_duration(self):
        argv, hashes = command(self.args)
        self.assertEqual(argv[argv.index("--muscle-step-count") + 1], "310000")
        self.assertIn("--resting-movie", argv)
        self.assertIn(str(Path(self.scene["source"]["skin"]["path"]).resolve()), hashes)
        self.assertIn(str((self.root / "lib/libmetalrobo.dylib").resolve()), hashes)

    def test_changed_skin_is_rejected_before_launch(self):
        Path(self.scene["source"]["skin"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(HumanImportError, "receipt hash differs"):
            command(self.args)

    def test_changed_muscle_surface_is_rejected_before_launch(self):
        (self.root / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue").write_bytes(b"changed")
        with self.assertRaisesRegex(HumanImportError, "receipt hash differs"):
            command(self.args)

    def test_intervention_requires_recovery(self):
        self.args.drive_intervention = (100., 310., .5)
        with self.assertRaisesRegex(HumanImportError, "recovery interval"):
            command(self.args)
        self.args.drive_intervention = (100., 160., .5)
        argv, _ = command(self.args)
        self.assertEqual(argv[-4:], ["--resting-drive-intervention", "100.0", "160.0", "0.5"])

    def test_duration_must_respect_physical_cadence(self):
        self.args.seconds = .0015
        with self.assertRaisesRegex(HumanImportError, "integer number of 1 ms"):
            command(self.args)

    def test_cpu_solver_experiment_cannot_enter_native_run(self):
        with patch.dict("os.environ", {"NUMI_HUMAN_STAND_CPU_FACTOR": "1"}):
            with self.assertRaisesRegex(HumanImportError, "CPU solver experiment"):
                run(self.args)
        self.assertFalse((self.args.output / "native.log").exists())

    def test_existing_evidence_is_never_replaced(self):
        self.args.output.mkdir()
        sentinel = self.args.output / "native.log"
        sentinel.write_text("retained failed run")
        with self.assertRaisesRegex(HumanImportError, "output already exists"):
            run(self.args)
        self.assertEqual(sentinel.read_text(), "retained failed run")


if __name__ == "__main__":
    unittest.main()

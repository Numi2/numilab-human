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
                                       mechanics_only=False, inspection_tour=False, drive_intervention=None,
                                       circulation=None)
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

    def test_explicit_circulation_is_hashed_and_passed_to_native_owner(self):
        self.args.circulation = self.root / 'reference-circulation.json'
        self.args.circulation.write_bytes(b'explicit reference native payload')
        argv, hashes = command(self.args)
        owner = str(self.args.circulation.resolve())
        self.assertEqual(argv[argv.index('--resting-scene') + 1], owner)
        self.assertEqual(hashes[owner], hashlib.sha256(self.args.circulation.read_bytes()).hexdigest())

    def test_changed_muscle_surface_is_rejected_before_launch(self):
        (self.root / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue").write_bytes(b"changed")
        with self.assertRaisesRegex(HumanImportError, "receipt hash differs"):
            command(self.args)

    def test_composed_anatomy_uses_the_recorded_muscle_owner_without_a_sibling_copy(self):
        source = self.root / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
        owner = self.root / "retained-owner" / source.name
        owner.parent.mkdir()
        source.rename(owner)
        for declared in (str(owner), str(owner.relative_to(self.root))):
            self.anatomy["provenance"]["native_muscle_surfaces"]["payload_path"] = declared
            self.write_receipts()
            argv, hashes = command(self.args)
            self.assertEqual(argv[argv.index("--soft-tissue-payload") + 1], str(owner.resolve()))
            self.assertIn(str(owner.resolve()), hashes)
        owner.write_bytes(b"changed owner payload")
        with self.assertRaisesRegex(HumanImportError, "receipt hash differs"):
            command(self.args)

    def test_explicit_respiration_is_hashed_and_delivered_to_the_existing_owner(self):
        self.args.respiration = self.root / 'source-bound-respiration.json'
        self.args.respiration.write_bytes(b'explicit source-derived respiratory parameters')
        argv, hashes = command(self.args)
        owner = str(self.args.respiration.resolve())
        self.assertEqual(argv[argv.index('--resting-scene') + 2], owner)
        self.assertEqual(hashes[owner], hashlib.sha256(self.args.respiration.read_bytes()).hexdigest())
        self.assertNotIn(str((self.root / 'matter/examples/resting-reference-respiration.json').resolve()), hashes)

    def test_intervention_requires_recovery(self):
        self.args.drive_intervention = (100., 310., .5)
        with self.assertRaisesRegex(HumanImportError, "recovery interval"):
            command(self.args)
        self.args.drive_intervention = (100., 160., .5)
        argv, _ = command(self.args)
        self.assertEqual(argv[-4:], ["--resting-drive-intervention", "100.0", "160.0", "0.5"])
        self.args.drive_intervention = (100., 160., 2.1)
        with self.assertRaises(HumanImportError):
            command(self.args)

    def test_postural_initialization_options_remain_explicit(self):
        baseline, _ = command(self.args)
        for flag in ("--muscle-activation", "--resting-release-initialization",
                     "--persistent-source-passive-joint-tissue", "--resting-rigid-hands"):
            self.assertNotIn(flag, baseline)
        self.args.postural_activation_cap = .01
        self.args.release_initialization = True
        self.args.upper_passive_joints = True
        argv, _ = command(self.args)
        self.assertEqual(argv[argv.index("--muscle-activation") + 1], "0.01")
        self.assertIn("--resting-release-initialization", argv)
        self.assertIn("--persistent-source-passive-joint-tissue", argv)

    def test_rigid_hand_reduction_is_explicit(self):
        self.args.rigid_hands = True
        argv, _ = command(self.args)
        self.assertIn("--resting-rigid-hands", argv)
        self.assertNotIn("--persistent-source-passive-joint-tissue", argv)

    def test_invalid_postural_cap_is_rejected_before_native_launch(self):
        for cap in (0., -.01, 1.01, float("nan"), float("inf")):
            with self.subTest(cap=cap):
                self.args.postural_activation_cap = cap
                with self.assertRaisesRegex(HumanImportError, "postural recruitment activation cap"):
                    command(self.args)

    def test_duration_must_respect_physical_cadence(self):
        self.args.seconds = .0015
        with self.assertRaisesRegex(HumanImportError, "integer number of native steps"):
            command(self.args)

    def test_explicit_timestep_preserves_requested_physical_time(self):
        self.args.dt = .002
        argv, _ = command(self.args)
        self.assertEqual(argv[argv.index("--muscle-step-count") + 1], "155000")
        self.assertEqual(argv[argv.index("--muscle-step-seconds") + 1], "0.002")
        for dt in (0., -.001, .0021, float('nan')):
            self.args.dt = dt
            with self.assertRaisesRegex(HumanImportError, "timestep"):
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

"""Admission regressions for the native launcher, not simulation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
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

    def test_individually_valid_scene_cannot_change_the_anatomical_skin_or_rigid_owner(self):
        for owner, record, key in (
            ("skin", self.anatomy["mass_geometry_accounting"], "skin_payload_sha256"),
            ("rigid", self.anatomy["provenance"], "rigid_payload_sha256"),
        ):
            with self.subTest(owner=owner):
                declared = self.scene["source"][owner]
                record[key] = declared["sha256"]
                self.write_receipts()
                command(self.args)
                path = Path(declared["path"])
                original = path.read_bytes()
                path.write_bytes(b"another independently valid owner payload")
                declared["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.write_receipts()
                with self.assertRaisesRegex(HumanImportError, f"body scene {owner} differs"):
                    command(self.args)
                path.write_bytes(original)
                declared["sha256"] = record[key]
                self.write_receipts()

    def test_declared_anatomical_owner_identity_cannot_be_empty_or_malformed(self):
        for owner, record, key in (
            ("skin", self.anatomy["mass_geometry_accounting"], "skin_payload_sha256"),
            ("rigid", self.anatomy["provenance"], "rigid_payload_sha256"),
        ):
            for invalid in (None, "", "not-a-sha256"):
                with self.subTest(owner=owner, invalid=invalid):
                    record[key] = invalid
                    self.write_receipts()
                    with self.assertRaisesRegex(HumanImportError, f"invalid anatomy {owner} identity"):
                        command(self.args)
            del record[key]

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

    def add_common_field(self, absolute=False):
        binding = {"schema": "numi.human.cardiac_common_field.v1"}
        for name in ("map", "polynomials", "domain_boxes"):
            path = self.root / "cardiac" / (name + ".bin")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(("retained-" + name).encode())
            binding[name] = {
                "path": str(path if absolute else path.relative_to(self.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        self.anatomy["provenance"]["cardiac_geometry_binding"] = {"common_field": binding}
        self.write_receipts()
        return binding

    def test_common_field_files_are_resolved_from_receipt_and_hashed(self):
        for absolute in (False, True):
            with self.subTest(absolute=absolute):
                binding = self.add_common_field(absolute)
                _, hashes = command(self.args)
                for name in ("map", "polynomials", "domain_boxes"):
                    path = self.root / "cardiac" / (name + ".bin")
                    self.assertEqual(hashes[str(path.resolve())], binding[name]["sha256"])

    def test_changed_or_unbound_common_field_is_rejected_before_launch(self):
        for name in ("map", "polynomials", "domain_boxes"):
            with self.subTest(owner=name):
                binding = self.add_common_field()
                path = self.root / "cardiac" / (name + ".bin")
                path.write_bytes(b"changed owner")
                with self.assertRaisesRegex(HumanImportError, "receipt hash differs"):
                    command(self.args)
                binding = self.add_common_field()
                del binding[name]["sha256"]
                self.write_receipts()
                with self.assertRaisesRegex(HumanImportError, "common cardiac .* identity"):
                    command(self.args)

    def test_common_field_mutation_during_run_fails_and_is_retained(self):
        self.add_common_field()
        changed = self.root / "cardiac" / "polynomials.bin"
        runtime = (self.root / "lib/libmetalrobo.dylib").resolve()

        def native(argv, **kwargs):
            kwargs["stdout"].write(
                f"dyld[123]: <BE23DF33-01EE-306E-A947-B7342DB0A863> {runtime}\n")
            changed.write_bytes(b"changed during native execution")
            return subprocess.CompletedProcess(argv, 0)

        with patch("numilab_human.resting_run.platform.platform", return_value="Darwin-test"), \
             patch("numilab_human.resting_run.subprocess.run", side_effect=native):
            self.assertEqual(run(self.args), 1)
        receipt = json.loads((self.args.output / "run-metadata.json").read_text())
        self.assertEqual(receipt["source_files_changed_during_run"], [str(changed.resolve())])
        self.assertTrue(receipt["loaded_metal_runtime"]["verified"])

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

    def test_contact_iterations_reach_existing_native_owner(self):
        baseline, _ = command(self.args)
        self.assertNotIn("--stand-contact-iterations", baseline)
        self.args.contact_iterations = 64
        argv, _ = command(self.args)
        self.assertEqual(argv[argv.index("--stand-contact-iterations") + 1], "64")
        for invalid in (0, 65, 1.5, True):
            with self.subTest(invalid=invalid):
                self.args.contact_iterations = invalid
                with self.assertRaisesRegex(HumanImportError, "contact iterations"):
                    command(self.args)

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

    def test_launcher_overrides_inherited_library_path_and_records_actual_image(self):
        runtime = (self.root / "lib/libmetalrobo.dylib").resolve()

        def native(argv, **kwargs):
            self.assertEqual(kwargs["env"]["DYLD_LIBRARY_PATH"],
                             f"{(self.root / 'lib').resolve()}:{(self.root / 'matter').resolve()}")
            self.assertEqual(kwargs["env"]["DYLD_PRINT_LIBRARIES"], "1")
            kwargs["stdout"].write(f"dyld[123]: <BE23DF33-01EE-306E-A947-B7342DB0A863> {runtime}\n")
            return subprocess.CompletedProcess(argv, 0)

        with patch("numilab_human.resting_run.platform.platform", return_value="Darwin-test"), \
             patch.dict("os.environ", {"DYLD_LIBRARY_PATH": "/unrelated/build/lib",
                                       "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
                                       "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "1",
                                       "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": "/run/failure.json"}), \
             patch("numilab_human.resting_run.subprocess.run", side_effect=native):
            self.assertEqual(run(self.args), 0)
        receipt = json.loads((self.args.output / "run-metadata.json").read_text())
        self.assertTrue(receipt["loaded_metal_runtime"]["verified"])
        self.assertEqual(receipt["loaded_metal_runtime"]["observed_images"][0]["path"], str(runtime))
        self.assertEqual(receipt["environment"]["DYLD_PRINT_LIBRARIES"], "1")
        self.assertEqual(receipt["environment"]["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT"], "1")
        self.assertEqual(receipt["environment"]["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"], "1")
        self.assertEqual(receipt["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"],
                         "/run/failure.json")

    def test_successful_child_with_another_runtime_is_rejected_and_evidence_retained(self):
        def native(argv, **kwargs):
            kwargs["stdout"].write("dyld[123]: <BE23DF33-01EE-306E-A947-B7342DB0A863> /another/libmetalrobo.dylib\n")
            return subprocess.CompletedProcess(argv, 0)

        with patch("numilab_human.resting_run.platform.platform", return_value="Darwin-test"), \
             patch("numilab_human.resting_run.subprocess.run", side_effect=native):
            self.assertEqual(run(self.args), 1)
        receipt = json.loads((self.args.output / "run-metadata.json").read_text())
        self.assertEqual(receipt["exit_code"], 0)
        self.assertFalse(receipt["loaded_metal_runtime"]["verified"])
        self.assertIn("/another/libmetalrobo.dylib", (self.args.output / "native.log").read_text())

    def test_missing_dyld_evidence_is_not_treated_as_a_verified_launch(self):
        with patch("numilab_human.resting_run.platform.platform", return_value="Darwin-test"), \
             patch("numilab_human.resting_run.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(run(self.args), 1)
        receipt = json.loads((self.args.output / "run-metadata.json").read_text())
        self.assertEqual(receipt["loaded_metal_runtime"]["observed_images"], [])
        self.assertFalse(receipt["loaded_metal_runtime"]["verified"])

    def test_existing_evidence_is_never_replaced(self):
        self.args.output.mkdir()
        sentinel = self.args.output / "native.log"
        sentinel.write_text("retained failed run")
        with self.assertRaisesRegex(HumanImportError, "output already exists"):
            run(self.args)
        self.assertEqual(sentinel.read_text(), "retained failed run")


if __name__ == "__main__":
    unittest.main()

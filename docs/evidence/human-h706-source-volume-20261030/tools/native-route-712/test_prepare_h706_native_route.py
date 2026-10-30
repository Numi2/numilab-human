import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-launch-route-preparation-712-integrity-derivative/prepare_h706_native_route.py")
spec = importlib.util.spec_from_file_location("h706_route", SCRIPT)
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)

class RouteSafetyTests(unittest.TestCase):
    def test_source_only_and_motion_modes_are_separate(self):
        suffix = {
            "MAX_PARENT_REGISTRATION_DISTANCE_M": "0.0004",
            "MAXIMUM_MASS_TRANSPORT_DISTANCE_M": "0.019353",
            "DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV": "/tmp/canonical-map.csv",
        }
        source = route.build_stage_env({}, suffix, source_only=True,
                                       canonical_export=True, operational_split=True)
        motion = route.build_stage_env({}, suffix, source_only=False,
                                       canonical_export=False, operational_split=True,
                                       operational_spatial_install=True)
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_SPATIAL_INSTALL"], "0")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONSERVATIVE_SPATIAL_REFERENCE"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"], "/tmp/canonical-map.csv")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_SPATIAL_INSTALL"], "1")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"], "1")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONSERVATIVE_SPATIAL_REFERENCE"], "1")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"], "0")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"], "0")
        self.assertNotIn("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV", motion)
        for env in (source, motion):
            self.assertEqual(env["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"], "1")
            self.assertEqual(env["NUMI_HUMAN_RESTING_SKIN_FAT_FEM"], "1")
            self.assertEqual(env["NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM"], "1")

    def test_stage_environment_expands_outputs_per_stage(self):
        template = {"NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT":
                    "{RUN_OUTPUT}/common-field-failure.json"}
        source = route.expand_stage_environment(template, Path("/tmp/source-preflight"))
        motion = route.expand_stage_environment(template, Path("/tmp/install-motion"))
        self.assertEqual(source["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"],
                         "/tmp/source-preflight/common-field-failure.json")
        self.assertEqual(motion["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"],
                         "/tmp/install-motion/common-field-failure.json")

    def test_motion_rejects_conservative_spatial_reference_without_install_optin(self):
        env = route.build_stage_env({}, {}, source_only=False, canonical_export=False,
                                    operational_split=True, operational_spatial_install=True)
        env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_SPATIAL_INSTALL"] = "0"
        with self.assertRaisesRegex(ValueError, "OPERATIONAL_SPATIAL_INSTALL=1"):
            route.validate_physical_spatial_install(env)

    def test_motion_requires_split_spatial_source_history_and_explicit_install(self):
        env = route.build_stage_env({}, {}, source_only=False, canonical_export=False,
                                    operational_split=True, operational_spatial_install=True)
        route.validate_physical_spatial_install(env)
        for key, value in (
            ("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT", "0"),
            ("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONSERVATIVE_SPATIAL_REFERENCE", "0"),
            ("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY", "1"),
            ("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT", "0"),
        ):
            bad = env.copy()
            bad[key] = value
            with self.assertRaises(ValueError):
                route.validate_physical_spatial_install(bad)

    def test_scene_output_placeholder_uses_fourth_positional_slot(self):
        valid = ["/rigid.nhrigid", "/muscle.nhmyo", "/bones.nhbones",
                 "{RUN_OUTPUT}", "--persistent-metal-stand", "--muscle-step-count", "10",
                 "--resting-movie", "{RUN_MOVIE}"]
        route.validate_scene_output_slot(valid)
        self.assertEqual(route.expand_scene_argv(valid, Path("/tmp/stage"))[valid.index("{RUN_MOVIE}")],
                         "/tmp/stage/native-viewer.mov")
        with self.assertRaisesRegex(ValueError, "stage path"):
            route.validate_scene_output_slot(valid[:-1] + ["/old/native-viewer.mov"])
        with self.assertRaisesRegex(ValueError, r"place \{RUN_OUTPUT\} fourth"):
            route.validate_scene_output_slot(valid[:3] + ["--persistent-metal-stand", "{RUN_OUTPUT}"])
        with self.assertRaisesRegex(ValueError, "first three"):
            route.validate_scene_output_slot(["--persistent-metal-stand", "/muscle.nhmyo",
                                              "/bones.nhbones", "{RUN_OUTPUT}"])

    def test_duplicate_or_noninteger_motion_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            route.argv_step_count(["--muscle-step-count", "2",
                                   "--muscle-step-count", "3"])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            route.argv_step_count(["--muscle-step-count", "+2"])

    def test_saved_plan_tampering_rejected_after_manifest_regeneration(self):
        import copy
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            config = {"schema": "fixture"}
            manifest.write_text(__import__("json").dumps(config))
            digest = hashlib.sha256(
                __import__("json").dumps(config, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()
            canonical = {
                "schema": "numi.human.h706.native-launch-plan.v1",
                "status": "SOURCE_PREFLIGHT_READY",
                "input_pins": {},
                "geometry_admission": {},
                "mass_admission": {},
                "stages": [{"argv": ["/native", "--persistent-metal-stand",
                                     "--muscle-step-count", "10", "{RUN_OUTPUT}"],
                            "environment": {"SOURCE_ONLY": "1"}}],
            }
            plan = copy.deepcopy(canonical)
            plan["manifest_path"] = str(manifest.resolve())
            plan["manifest_sha256"] = digest
            with patch.object(route, "verify_pinned_static"), \
                 patch.object(route, "verify_input_pins", return_value={}), \
                 patch.object(route, "check_claim_receipt", return_value=({}, "fixture")), \
                 patch.object(route, "build_plan", return_value=canonical):
                route.verify_saved_plan_inputs(plan)
                tampered = copy.deepcopy(plan)
                tampered["stages"][0]["argv"][3] = "1"
                with self.assertRaisesRegex(ValueError, "differs from the manifest-derived route"):
                    route.verify_saved_plan_inputs(tampered)

    def test_geometry_admission_rejects_partial_self_coverage(self):
        with self.assertRaisesRegex(ValueError, "complete exact-gate report"):
            route.validate_complete_role_self_coverage({})

    def test_geometry_admission_requires_face_domain_counts_to_match(self):
        claims = {}
        for role, count in (("O", 154939), ("I", 252017), ("D", 246763)):
            claims[f"{role}_full_self_pairs"] = {
                "path": f"full_current_self_results.{role}.count", "equals": 0}
            claims[f"{role}_full_self_selected_faces"] = {
                "path": f"stages.full_current_{role}_self.selected_faces", "equals": count}
            claims[f"{role}_full_self_complete_faces"] = {
                "path": f"stages.full_current_{role}_self.complete_faces", "equals": count}
            claims[f"{role}_candidate_faces"] = {
                "path": f"candidate_readback.{role}.candidate_faces", "equals": count}
            claims[f"{role}_full_self_status"] = {
                "path": f"stages.full_current_{role}_self.status", "equals": "PASS"}
        route.validate_complete_role_self_coverage(claims)
        claims["I_full_self_selected_faces"]["equals"] -= 1
        with self.assertRaisesRegex(ValueError, "does not cover every current candidate face"):
            route.validate_complete_role_self_coverage(claims)

    def test_geometry_receipt_integration_rejects_nonzero_current_full_self_pairs(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            report={
                "physical_admission": True,
                "full_current_self_results": {
                    role: {"count": (0 if role=="O" else (12 if role=="I" else 11))}
                    for role in ("O","I","D")
                },
                "stages": {
                    f"full_current_{role}_self": {
                        "selected_faces": count,
                        "complete_faces": count,
                        "status": ("PASS" if role=="O" else "FAIL"),
                    }
                    for role,count in (("O",154939),("I",252017),("D",246763))
                },
                "candidate_readback": {
                    role: {"candidate_faces": count}
                    for role,count in (("O",154939),("I",252017),("D",246763))
                },
            }
            report_path=root/"geometry.json"
            report_path.write_text(json.dumps(report))
            claims={"physical_admission":{"path":"physical_admission","equals":True}}
            for role,count in (("O",154939),("I",252017),("D",246763)):
                claims[f"{role}_full_self_pairs"]={
                    "path":f"full_current_self_results.{role}.count",
                    "equals":report["full_current_self_results"][role]["count"]}
                claims[f"{role}_full_self_selected_faces"]={
                    "path":f"stages.full_current_{role}_self.selected_faces","equals":count}
                claims[f"{role}_full_self_complete_faces"]={
                    "path":f"stages.full_current_{role}_self.complete_faces","equals":count}
                claims[f"{role}_candidate_faces"]={
                    "path":f"candidate_readback.{role}.candidate_faces","equals":count}
                claims[f"{role}_full_self_status"]={
                    "path":f"stages.full_current_{role}_self.status",
                    "equals":report["stages"][f"full_current_{role}_self"]["status"]}
            spec={"path":str(report_path),
                  "sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  "claims":claims}
            with self.assertRaisesRegex(ValueError,"full I self-pair count must be integer zero"):
                route.check_claim_receipt(
                    spec, route.ROLE_SELF_COVERAGE_CLAIM_NAMES, "geometry admission")

    def test_motion_rejects_embedded_preflight_failure_marker(self):
        stdout = "registered_volume_native_preflight=FAIL source_only=0"
        with self.assertRaisesRegex(ValueError, "failure"):
            route.require_no_failure_markers(
                stdout, "", ("registered_volume_native_preflight=FAIL",), "motion")

    def test_stale_parent_registration_bound_rejected(self):
        env = {key: "" for key in route.REQUIRED_ENV}
        env["MAX_PARENT_REGISTRATION_DISTANCE_M"] = "0.02"
        env["MAXIMUM_MASS_TRANSPORT_DISTANCE_M"] = route.MAX_MASS_TRANSPORT_M
        with self.assertRaisesRegex(ValueError, "0.0004 m bound"):
            route.validate_environment(env, {}, "/tmp/output.csv")

    def test_changed_mass_transport_bound_rejected(self):
        env = {key: "" for key in route.REQUIRED_ENV}
        env["MAX_PARENT_REGISTRATION_DISTANCE_M"] = route.MAX_NATIVE_PARENT_REGISTRATION_M
        env["MAXIMUM_MASS_TRANSPORT_DISTANCE_M"] = "0.02"
        with self.assertRaisesRegex(ValueError, "0.019353 m bound"):
            route.validate_environment(env, {}, "/tmp/output.csv")

    def test_readiness_reports_all_missing_static_inputs_and_missing_228_bundle(self):
        import json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps({
                "schema":"numi.human.h706.native-launch-inputs.v1",
                "input_pins":{str(root/"missing-a"): "a"*64,
                              str(root/"missing-b"): "b"*64},
                "native_preflight_env":{},
                "geometry_admission":None,
                "mass_admission":None,
                "scene":{"argv":[]},
            }))
            static = {
                str(route.BINARY): "fixture-binary-sha",
                "__compiled_probe_source_sha256__": "fixture-probe-sha",
                "__vfs_external_content_count__": "14",
                "__vfs_external_content__": {},
                **{path: digest for path, digest in route.RUNTIME_ASSET_PINS.items()},
            }
            with patch.object(route, "verify_pinned_static", return_value=static), \
                 patch.dict(route.PINNED_PARENT, {}, clear=True):
                result=route.readiness(manifest)
        self.assertTrue(result["scene_input_pin_error"].startswith("current H706 input: missing file"))
        self.assertEqual(len(result["scene_input_pin_errors"]),2)
        self.assertTrue(all("missing file" in error for error in result["scene_input_pin_errors"]))
        self.assertIn("current O/I/D and 217 volume/registration bundle inputs are incomplete",
                      result["blockers"])
        self.assertIn("candidate-specific geometry admission receipt is absent", result["blockers"])
        self.assertIn("candidate-specific conservative mass/density admission receipt is absent",
                      result["blockers"])
        self.assertIn("OPERATIONAL_SOURCE_VOLUME_ASSET",
                      result["candidate_registered_bundle_missing_fields"])

    def test_native721_has_fourteen_unique_contents_across_sixteen_vfs_mappings(self):
        static = route.verify_pinned_static()
        self.assertEqual(len(route.VFS_CONTENT_PINS), 14)
        self.assertEqual(static["__vfs_external_content_count__"], "14")
        self.assertEqual(len(static["__vfs_external_content__"]), 14)
        for path, row in static["__vfs_external_content__"].items():
            self.assertEqual(row["sha256"], route.VFS_CONTENT_PINS[path])
            self.assertIn(row["compile_time_guard"],
                          {"receipt_hash", "source_mtime_precedes_compile_receipt"})
        self.assertEqual(len(route.RUNTIME_ASSET_PINS), 7)
        for path, digest in {**route.VFS_CONTENT_PINS, **route.RUNTIME_ASSET_PINS}.items():
            self.assertEqual(static[path], digest)

    def test_vfs_overlay_rejects_missing_or_changed_external_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "compile-result.json"
            receipt.write_text("{}")
            os.utime(receipt, ns=(2_000_000_000, 2_000_000_000))
            paths = []
            pins = {}
            roots = []
            compile_hashes = {}
            for i in range(14):
                path = root / f"source-{i}.hpp"
                path.write_text(f"source-{i}")
                os.utime(path, ns=(1_000_000_000, 1_000_000_000))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                paths.append(path)
                pins[str(path)] = digest
                roots.append({"type": "file", "name": f"/vfs/source-{i}.hpp",
                              "external-contents": str(path)})
                if i < 8:
                    compile_hashes[str(path)] = digest
            for i in range(2):
                roots.append({"type": "file", "name": f"/vfs/alias-{i}.hpp",
                              "external-contents": str(paths[0])})
            report = route.verify_vfs_content_pins(
                {"roots": roots}, {"input_sha256": compile_hashes}, receipt, pins)
            self.assertEqual(len(report), 14)
            paths[-1].write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                route.verify_vfs_content_pins(
                    {"roots": roots}, {"input_sha256": compile_hashes}, receipt, pins)

    def test_vfs_overlay_rejects_wrong_root_mapping_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / f"source-{i}" for i in range(14)]
            pins = {}
            roots = []
            for i, path in enumerate(paths):
                path.write_text(str(i))
                pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                roots.append({"type": "file", "name": f"/vfs/{i}",
                              "external-contents": str(path)})
            roots.extend([
                {"type": "file", "name": "/vfs/alias-a", "external-contents": str(paths[0])},
                {"type": "file", "name": "/vfs/alias-b", "external-contents": str(paths[0])},
            ])
            with self.assertRaisesRegex(ValueError, "exactly 16"):
                route.verify_vfs_content_pins(
                    {"roots": roots[:-1]}, {"input_sha256": {}}, root, pins)

    def test_full_human_motion_rejects_preparation_and_identity_modes(self):
        valid = route.build_stage_env({}, {}, source_only=False, canonical_export=False,
                                      operational_split=True, operational_spatial_install=True)
        for token in ("--mechanics-only", "--parent-mass-capture-only",
                      "--identity-import"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "forbids"):
                route.validate_full_scene_mode(["--persistent-metal-stand", token], valid)
        fascia_off = valid.copy()
        fascia_off["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"] = "0"
        with self.assertRaisesRegex(ValueError, "intact resting tissue world"):
            route.validate_full_scene_mode(["--persistent-metal-stand"], fascia_off)
        route.validate_full_scene_mode(["--persistent-metal-stand"], valid)

    def test_report_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text("keep\n")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                route.dump(path, {"changed": True})
            self.assertEqual(path.read_text(), "keep\n")

if __name__ == "__main__":
    unittest.main()

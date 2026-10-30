import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-launch-route-preparation-711/prepare_h706_native_route.py")
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
                                       canonical_export=False, operational_split=False)
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"], "1")
        self.assertEqual(source["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"], "/tmp/canonical-map.csv")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"], "0")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"], "0")
        self.assertEqual(motion["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"], "0")
        self.assertNotIn("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV", motion)
        for env in (source, motion):
            self.assertEqual(env["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"], "1")
            self.assertEqual(env["NUMI_HUMAN_RESTING_SKIN_FAT_FEM"], "1")
            self.assertEqual(env["NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM"], "1")

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

    def test_native004_has_all_fourteen_vfs_contents_and_runtime_assets_pinned(self):
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
            report = route.verify_vfs_content_pins(
                {"roots": roots}, {"input_sha256": compile_hashes}, receipt, pins)
            self.assertEqual(len(report), 14)
            paths[-1].write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                route.verify_vfs_content_pins(
                    {"roots": roots}, {"input_sha256": compile_hashes}, receipt, pins)

    def test_vfs_overlay_rejects_non_fourteen_root_map(self):
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
            with self.assertRaisesRegex(ValueError, "exactly 14"):
                route.verify_vfs_content_pins(
                    {"roots": roots[:-1]}, {"input_sha256": {}}, root, pins)

    def test_full_human_motion_rejects_preparation_and_identity_modes(self):
        for token in ("--mechanics-only", "--parent-mass-capture-only",
                      "--identity-import"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "forbids"):
                route.validate_full_scene_mode(["--persistent-metal-stand", token], {})
        with self.assertRaisesRegex(ValueError, "intact resting tissue world"):
            route.validate_full_scene_mode(
                ["--persistent-metal-stand"], {"NUMI_HUMAN_RESTING_PASSIVE_FASCIA": "0"})
        route.validate_full_scene_mode(
            ["--persistent-metal-stand"], {"NUMI_HUMAN_RESTING_PASSIVE_FASCIA": "1"})

    def test_report_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text("keep\n")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                route.dump(path, {"changed": True})
            self.assertEqual(path.read_text(), "keep\n")

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER_PATH = HERE / "run_canonical_038.py"
BUILDER_PATH = HERE / "build_source_control_038.py"
GENERATOR_PATH = HERE / "prepare_run_manifest_038.py"

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

LAUNCHER = load(LAUNCHER_PATH, "test_launch_038")
BUILDER = load(BUILDER_PATH, "test_builder_for_launch_038")
GENERATOR = load(GENERATOR_PATH, "test_manifest_generator_038")

class Run038Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="h706-run-038-test-")
        cls.root = Path(cls.temp.name)
        cls.control, cls.closure = BUILDER.build(cls.root / "controls")
        cls.control_path = cls.root / "controls" / "source-control-manifest-038.json"
        cls.closure_path = cls.root / "controls" / "source-closure-038.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_manifest(self, name: str = "plan.json") -> Path:
        data_dir = self.root / f"inputs-{name}"
        data_dir.mkdir(exist_ok=True)
        builder_path = Path(self.control["receipt_builder"]["path"])
        builder = load(builder_path, "pinned_builder_for_test")
        named = {}
        for key in sorted(builder.INPUT_KEYS):
            path = data_dir / f"{key}.bin"
            path.write_bytes((key + "\n").encode("ascii"))
            named[key] = LAUNCHER.file_pin({"path": str(path.resolve()),
                                            "sha256": LAUNCHER.sha_file(path)},
                                           f"test input {key}")
        for fat_key, skin_key in (("child_fat_node", "child_skin_node"),
                                  ("child_fat_elements", "child_skin_elements"),
                                  ("child_fat_labels", "child_skin_labels")):
            named[fat_key] = named[skin_key]
        carrier = named["carrier_skin_packet"]
        carrier_fat = named["carrier_fat_packet"]
        witness_skin = named["carrier_witness_skin"]
        witness_fat = named["carrier_witness_fat"]
        out = self.root / f"flow-output-{name}"
        env = [
            "NUMI_H706_FRACTIONAL_ENABLE=1",
            f"NUMI_H706_FRACTIONAL_OUTPUT_DIR={out.resolve()}",
            f"NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ={carrier['path']}",
            f"NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256={carrier['sha256']}",
            f"NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ={carrier_fat['path']}",
            f"NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256={carrier_fat['sha256']}",
            f"NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV={witness_skin['path']}",
            f"NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256={witness_skin['sha256']}",
            f"NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV={witness_fat['path']}",
            f"NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256={witness_fat['sha256']}",
        ]
        run_config_path = data_dir / "run-config.json"
        run_config_path.write_text("{\"schema\":\"fixture\"}\n", encoding="utf-8")
        run_config_pin = LAUNCHER.file_pin(
            {"path": str(run_config_path.resolve()), "sha256": LAUNCHER.sha_file(run_config_path)},
            "test run config")
        env.append(f"NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256={run_config_pin['sha256']}")
        envelope_path = data_dir / "current-envelope.json"
        frame_sha = "a" * 64
        envelope_inputs = {
            "state": named["current_state"]["sha256"],
            "geometry_skin": named["candidate_skin_volume"]["sha256"],
            "geometry_fat": named["candidate_fat_volume"]["sha256"],
            "child_node": named["child_skin_node"]["sha256"],
            "child_element": named["child_skin_elements"]["sha256"],
            "child_labels": named["child_skin_labels"]["sha256"],
            "source330_mass": named["source330_mass"]["sha256"],
            "source330_labels": named["source330_labels"]["sha256"],
            "source330_nodes": named["source330_nodes"]["sha256"],
            "source330_tets": named["source330_tets"]["sha256"],
            "coordinate_frame": frame_sha,
        }
        envelope = {
            "schema": "numi.human.mass-transport-envelope.v1",
            "status": "PREDECLARED_CURRENT_GEOMETRY_ENVELOPE",
            "exact_decimal": "0.019353",
            "input_sha256": envelope_inputs,
        }
        envelope_receipt = {
            "schema": "numi-human.registered-volume-parent-identity-source-control.v1",
            "physical_admission": False,
            "native_steps": 0,
            "coordinate_frame_sha256": frame_sha,
            "source_body_count": 157,
            "mass_mapping_maximum_transport_distance_m": 0.019353,
            "mass_transport_envelope": envelope,
            "input_sha256": envelope_inputs,
        }
        envelope_path.write_text(json.dumps(envelope_receipt, indent=2, sort_keys=True), encoding="utf-8")
        data = {
            "schema": LAUNCHER.RUN_SCHEMA,
            "variant": LAUNCHER.VARIANT,
            "source": self.control["source"],
            "binary": self.control["binary"],
            "source_control_manifest": LAUNCHER.file_pin(
                {"path": str(self.control_path.resolve()),
                 "sha256": LAUNCHER.sha_file(self.control_path)}, "control"),
            "source_closure": LAUNCHER.file_pin(
                {"path": str(self.closure_path.resolve()),
                 "sha256": LAUNCHER.sha_file(self.closure_path)}, "closure"),
            "mass_transport_envelope_receipt": LAUNCHER.file_pin(
                {"path": str(envelope_path.resolve()),
                 "sha256": LAUNCHER.sha_file(envelope_path)}, "transport envelope"),
            "argv": [
                "/usr/bin/env", *env, self.control["binary"]["path"],
                "--current-fractional-semantic",
                "--parent-mass-csv", named["source330_mass"]["path"],
                "--labels-csv", named["source330_labels"]["path"],
                "--child-node-file", named["child_skin_node"]["path"],
                "--child-ele-file", named["child_skin_elements"]["path"],
                "--child-labels-csv", named["child_skin_labels"]["path"],
                "--coordinate-frame-sha256", "a" * 64,
                "--source-body-count", "157",
                "--maximum-transport-distance-m", "0.019353",
            ],
            "cwd": str(HERE.resolve()),
            "output_dir": str(out.resolve()),
            "maximum_wall_seconds": 30,
            "stdout_log": str((self.root / f"{name}.stdout").resolve()),
            "stderr_log": str((self.root / f"{name}.stderr").resolve()),
            "execution_json": str((self.root / f"{name}.execution.json").resolve()),
            "named_input_pins": named,
            "file_pins": [run_config_pin],
            "environment_digest_pins": [{
                "sha_variable": "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256",
                "pin": run_config_pin,
            }],
            "environment_file_pins": [
                {
                    "path_variable": "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ",
                    "sha_variable": "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256",
                    "pin": carrier,
                },
                {
                    "path_variable": "NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ",
                    "sha_variable": "NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256",
                    "pin": carrier_fat,
                },
                {
                    "path_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV",
                    "sha_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256",
                    "pin": witness_skin,
                },
                {
                    "path_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV",
                    "sha_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256",
                    "pin": witness_fat,
                },
            ],
            "physical_admission": False,
            "native_steps": 0,
        }
        path = self.root / name
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_canonical_preflight_is_not_execution(self):
        path = self.make_manifest()
        report = LAUNCHER.preflight(path)
        self.assertEqual(report["status"], "PASS_LAUNCH_READY_NOT_RUN")
        self.assertEqual(report["variant"], LAUNCHER.VARIANT)
        self.assertEqual(report["named_input_count"], 22)
        self.assertFalse(Path(json.loads(path.read_text())["output_dir"]).exists())
        self.assertFalse(Path(json.loads(path.read_text())["execution_json"]).exists())

    def test_wall_time_ceiling_accepts_601_and_1800_only_within_bound(self):
        path = self.make_manifest("wall-bound.json")
        data = json.loads(path.read_text())
        for seconds in (600, 601, 1800):
            data["maximum_wall_seconds"] = seconds
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(LAUNCHER.validate_manifest(path)["manifest"]["maximum_wall_seconds"], seconds)
        data["maximum_wall_seconds"] = 1801
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bounded to 1..1800"):
            LAUNCHER.validate_manifest(path)

    def test_unbound_input_path_rejects(self):
        path = self.make_manifest("negative.json")
        data = json.loads(path.read_text())
        data["environment_file_pins"] = []
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks a file-pin binding"):
            LAUNCHER.validate_manifest(path)

    def test_environment_digest_must_match_input_file(self):
        path = self.make_manifest("wrong-digest.json")
        data = json.loads(path.read_text())
        data["argv"] = [
            "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256=0000000000000000000000000000000000000000000000000000000000000000"
            if item.startswith("NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256=") else item
            for item in data["argv"]
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from argv or pin"):
            LAUNCHER.validate_manifest(path)

    def test_binary_and_output_identity_are_bound(self):
        path = self.make_manifest("identity.json")
        data = json.loads(path.read_text())
        binary_index = data["argv"].index(data["binary"]["path"])
        data["argv"][binary_index] = "/tmp/wrong-cli"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned binary immediately after environment assignments"):
            LAUNCHER.validate_manifest(path)

        path = self.make_manifest("output.json")
        data = json.loads(path.read_text())
        data["output_dir"] = str((self.root / "different-output").resolve())
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output directory differs"):
            LAUNCHER.validate_manifest(path)

    def test_manifest_generator_replaces_both_role_csvs_without_new_env_keys(self):
        template_path = self.make_manifest("generator-template.json")
        data = json.loads(template_path.read_text())
        original_env, _ = LAUNCHER._split_argv(data["argv"], data["binary"]["path"])
        extra_skin = self.root / "new-skin-witness.csv"
        extra_fat = self.root / "new-fat-witness.csv"
        extra_skin.write_text("row,value\n0,skin\n", encoding="utf-8")
        extra_fat.write_text("row,value\n0,fat\n", encoding="utf-8")
        generated = self.root / "generated"
        generated.mkdir()
        manifest_path = generated / "run.json"
        result = GENERATOR.build_manifest(
            template_path, self.control_path, self.closure_path,
            Path(data["mass_transport_envelope_receipt"]["path"]),
            manifest_path, generated / "output", generated / "stdout.log",
            generated / "stderr.log", generated / "execution.json",
            [
                {"path": str(extra_skin), "sha256": LAUNCHER.sha_file(extra_skin),
                 "path_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV",
                 "sha_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256",
                 "named_input_key": "carrier_witness_skin"},
                {"path": str(extra_fat), "sha256": LAUNCHER.sha_file(extra_fat),
                 "path_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV",
                 "sha_variable": "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256",
                 "named_input_key": "carrier_witness_fat"},
            ],
        )
        after_env, _ = LAUNCHER._split_argv(result["argv"], result["binary"]["path"])
        self.assertEqual(result["variant"], LAUNCHER.VARIANT)
        self.assertEqual(result["named_input_pins"]["carrier_witness_skin"]["path"], str(extra_skin.resolve()))
        self.assertEqual(result["named_input_pins"]["carrier_witness_fat"]["path"], str(extra_fat.resolve()))
        self.assertEqual(result["maximum_wall_seconds"], 1800)
        witness_vars = {key for key in after_env if "CARRIER_WITNESS" in key}
        self.assertEqual(witness_vars, {key for key in original_env if "CARRIER_WITNESS" in key})
        self.assertEqual(len(witness_vars), 4)
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV"], str(extra_skin.resolve()))
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256"], LAUNCHER.sha_file(extra_skin))
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV"], str(extra_fat.resolve()))
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256"], LAUNCHER.sha_file(extra_fat))
        for key in ("carrier_skin_packet", "carrier_fat_packet"):
            self.assertEqual(result["named_input_pins"][key], data["named_input_pins"][key])
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ"], original_env["NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ"])
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ"], original_env["NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ"])
        preflight = LAUNCHER.preflight(manifest_path)
        self.assertEqual(preflight["status"], "PASS_LAUNCH_READY_NOT_RUN")
        self.assertEqual(preflight["extra_input_count"], 5)
        self.assertEqual(preflight["maximum_transport_distance_m"], "0.019353")
        self.assertEqual(after_env["NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256"], result["run_config"]["sha256"])
        self.assertEqual(LAUNCHER.sha_file(Path(result["run_config"]["path"])), result["run_config"]["sha256"])
        current_config = json.loads(Path(result["run_config"]["path"]).read_text())
        self.assertEqual(current_config["named_input_pins"]["carrier_witness_skin"]["path"], str(extra_skin.resolve()))
        self.assertEqual(current_config["named_input_pins"]["carrier_witness_fat"]["path"], str(extra_fat.resolve()))

    def test_manifest_generator_cli_accepts_both_role_replacements(self):
        template_path = self.make_manifest("cli-generator-template.json")
        data = json.loads(template_path.read_text())
        skin = self.root / "cli-skin-witness.csv"
        fat = self.root / "cli-fat-witness.csv"
        skin.write_text("row,value\n0,skin\n", encoding="utf-8")
        fat.write_text("row,value\n0,fat\n", encoding="utf-8")
        out = self.root / "cli-generated"
        out.mkdir()
        args = [
            str(GENERATOR_PATH), "--template-manifest", str(template_path),
            "--source-control", str(self.control_path), "--source-closure", str(self.closure_path),
            "--envelope-receipt", data["mass_transport_envelope_receipt"]["path"],
            "--output-manifest", str(out / "run.json"), "--output-dir", str(out / "output"),
            "--stdout-log", str(out / "stdout.log"), "--stderr-log", str(out / "stderr.log"),
            "--execution-json", str(out / "execution.json"),
            "--extra-file", "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV",
            "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256", str(skin), "carrier_witness_skin",
            "--extra-file", "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV",
            "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256", str(fat), "carrier_witness_fat",
        ]
        with patch.object(sys, "argv", args), redirect_stdout(StringIO()):
            self.assertEqual(GENERATOR.main(), 0)
        manifest_path = out / "run.json"
        preflight = LAUNCHER.preflight(manifest_path)
        self.assertEqual(preflight["status"], "PASS_LAUNCH_READY_NOT_RUN")
        self.assertEqual(LAUNCHER.validate_manifest(manifest_path)["manifest"]["maximum_wall_seconds"], 1800)

    def test_actual_cli_path_arguments_are_bound_to_named_inputs(self):
        path = self.make_manifest("cli-path.json")
        data = json.loads(path.read_text())
        index = data["argv"].index("--child-ele-file") + 1
        data["argv"][index] = str((self.root / "wrong.ele").resolve())
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CLI --child-ele-file differs from named input"):
            LAUNCHER.validate_manifest(path)

    def test_hash_only_environment_requires_an_explicit_file_pin(self):
        path = self.make_manifest("hash-only.json")
        data = json.loads(path.read_text())
        current = data["named_input_pins"]["current_state"]
        data["argv"].insert(1, f"NUMI_H706_FRACTIONAL_SOURCE_STATE_SHA256={current['sha256']}")
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks a file-pin binding"):
            LAUNCHER.validate_manifest(path)
        data["environment_digest_pins"] = [
            *data.get("environment_digest_pins", []),
            {"sha_variable": "NUMI_H706_FRACTIONAL_SOURCE_STATE_SHA256", "pin": current},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(LAUNCHER.validate_manifest(path)["cli_options"]["--current-fractional-semantic"])

if __name__ == "__main__":
    unittest.main()


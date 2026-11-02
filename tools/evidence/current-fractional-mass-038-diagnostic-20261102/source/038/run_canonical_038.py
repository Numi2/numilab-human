#!/usr/bin/env python3
"""Canonical 030d launcher for the existing current fractional CLI run schema.

By default this only validates pins and writes a source-only launch preflight.
The full allocator is started only with --execute. Carrier/projection inputs
are data-driven through argv environment assignments and explicit file-pin
bindings; no proposal family or witness type is hard-wired here.
"""
from __future__ import annotations
import argparse
import hashlib
from decimal import Decimal, InvalidOperation
import importlib.util
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
READER_PATH = HERE / "recorded_flow_contract_010_030d.py"
READER_SHA256 = "f2ee1d53880cb12864f1c3afab6bf336544a8b98f64b60adbd1ab3d3cd4bdba2"
RUN_SCHEMA = "numi.h706.current-fractional-mass-cli-run.v1"
PLAN_SCHEMA = "numi.h706.current-fractional-mass-cli-run-plan.038.v1"
EXEC_SCHEMA = "numi.h706.current-fractional-mass-cli-execution.030d.v1"
VARIANT = "030d-contiguous-preference-lookup"
ENV_PREFIX = "NUMI_H706_FRACTIONAL_"
OUTPUT_ENV = "NUMI_H706_FRACTIONAL_OUTPUT_DIR"
ENABLE_ENV = "NUMI_H706_FRACTIONAL_ENABLE"
HEX = re.compile(r"^[0-9a-f]{64}$")

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def import_reader():
    path = READER_PATH.resolve(strict=True)
    if sha_file(path) != READER_SHA256:
        raise ValueError("pinned 030d source reader changed")
    spec = importlib.util.spec_from_file_location("recorded_flow_contract_010_030d_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import 030d source reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def file_pin(ref: Any, label: str) -> dict[str, Any]:
    if not isinstance(ref, dict) or set(ref) not in (
        {"path", "sha256"}, {"path", "sha256", "bytes"}
    ):
        raise ValueError(f"{label} pin schema mismatch")
    raw, digest = ref["path"], ref["sha256"]
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError(f"{label} SHA must be lowercase SHA-256")
    path = Path(raw)
    if str(path.resolve()) != raw or not path.is_file() or sha_file(path) != digest:
        raise ValueError(f"{label} path/SHA changed: {raw}")
    if "bytes" in ref and (type(ref["bytes"]) is not int or ref["bytes"] != path.stat().st_size):
        raise ValueError(f"{label} byte count changed: {raw}")
    return {"path": raw, "sha256": digest, "bytes": path.stat().st_size}

def _verify_contract_files(manifest: dict[str, Any], manifest_path: Path):
    reader = import_reader()
    control_ref = manifest.get("source_control_manifest")
    closure_ref = manifest.get("source_closure")
    control_pin = file_pin(control_ref, "source-control")
    closure_pin = file_pin(closure_ref, "source-closure")
    control = json.loads(Path(control_pin["path"]).read_text(encoding="utf-8"))
    closure_path = Path(closure_pin["path"])
    reader.verify_nested_control_refs(control)
    reader.BASE.verify_source_closure(
        closure_path, closure_pin["sha256"], Path(control_pin["path"]),
        control_pin["sha256"], control,
    )
    if control.get("variant") != VARIANT:
        raise ValueError("run source-control is not the current 030d variant")
    for field in ("source", "binary"):
        got = file_pin(manifest.get(field), f"run {field}")
        expected = reader.BASE._strip_verified_bytes(control[field])
        if {"path": got["path"], "sha256": got["sha256"]} != expected:
            raise ValueError(f"run {field} does not match the actual 030d source-control")
    if manifest.get("variant") != VARIANT:
        raise ValueError("run manifest variant differs from the current 030d source-control")
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("run manifest is not the canonical CLI run schema")
    return reader, control

CLI_FLAG_OPTIONS = {"--current-fractional-semantic"}
CLI_VALUE_OPTIONS = {
    "--parent-mass-csv", "--labels-csv", "--child-node-file",
    "--child-ele-file", "--child-labels-csv", "--coordinate-frame-sha256",
    "--source-body-count", "--maximum-transport-distance-m",
}


def _split_argv(argv: Any, binary_path: str) -> tuple[dict[str, str], list[str]]:
    if not isinstance(argv, list) or len(argv) < 4 or argv[0] != "/usr/bin/env":
        raise ValueError("run argv must use /usr/bin/env, pinned assignments, binary, and CLI options")
    values: dict[str, str] = {}
    index = 1
    while index < len(argv) and argv[index] != binary_path:
        item = argv[index]
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("run argv must place the pinned binary immediately after environment assignments")
        key, value = item.split("=", 1)
        if not key.startswith(ENV_PREFIX) or key in values or not value:
            raise ValueError("run argv has an unexpected, duplicate, or empty environment assignment")
        values[key] = value
        index += 1
    if index >= len(argv):
        raise ValueError("run argv does not contain the pinned 030d binary after environment assignments")
    if argv.count(binary_path) != 1:
        raise ValueError("run argv must contain the pinned 030d binary exactly once")
    return values, argv[index + 1:]


def _environment(manifest: dict[str, Any], binary_path: str) -> dict[str, str]:
    values, _ = _split_argv(manifest.get("argv"), binary_path)
    if values.get(ENABLE_ENV) != "1":
        raise ValueError("fractional CLI must be explicitly enabled")
    if values.get(OUTPUT_ENV) != manifest.get("output_dir"):
        raise ValueError("output directory differs from the pinned CLI environment")
    return values


def _cli_options(manifest: dict[str, Any], binary_path: str) -> dict[str, str | bool]:
    _, argv = _split_argv(manifest.get("argv"), binary_path)
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in CLI_FLAG_OPTIONS:
            if option in parsed:
                raise ValueError(f"duplicate CLI flag {option}")
            parsed[option] = True
            index += 1
        elif option in CLI_VALUE_OPTIONS:
            if option in parsed or index + 1 >= len(argv):
                raise ValueError(f"missing or duplicate CLI value for {option}")
            parsed[option] = argv[index + 1]
            index += 2
        else:
            raise ValueError(f"unexpected 030d CLI argument {option!r}")
    expected = CLI_FLAG_OPTIONS | CLI_VALUE_OPTIONS
    if set(parsed) != expected:
        raise ValueError("030d CLI arguments do not exactly cover the frozen invocation contract")
    frame = parsed["--coordinate-frame-sha256"]
    if not isinstance(frame, str) or not HEX.fullmatch(frame):
        raise ValueError("coordinate frame argument is not a lowercase SHA-256")
    body_count = parsed["--source-body-count"]
    if not isinstance(body_count, str) or not re.fullmatch(r"[1-9][0-9]*", body_count):
        raise ValueError("source body count must be a positive integer")
    distance = parsed["--maximum-transport-distance-m"]
    try:
        distance_value = float(distance)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("maximum transport distance must be numeric") from None
    if not math.isfinite(distance_value) or distance_value <= 0.0:
        raise ValueError("maximum transport distance must be positive and finite")
    for option in ("--parent-mass-csv", "--labels-csv", "--child-node-file",
                   "--child-ele-file", "--child-labels-csv"):
        value = parsed[option]
        if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
            raise ValueError(f"{option} must be an absolute canonical path")
    return parsed

def validate_transport_envelope(ref: Any, cli: dict[str, str | bool],
                                named_pins: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pin = file_pin(ref, "current mass-transport envelope receipt")
    receipt = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != "numi-human.registered-volume-parent-identity-source-control.v1":
        raise ValueError("transport envelope receipt schema mismatch")
    envelope = receipt.get("mass_transport_envelope")
    if not isinstance(envelope, dict) or envelope.get("schema") != "numi.human.mass-transport-envelope.v1" or envelope.get("status") != "PREDECLARED_CURRENT_GEOMETRY_ENVELOPE":
        raise ValueError("transport envelope receipt is not the current predeclared geometry envelope")
    exact = envelope.get("exact_decimal")
    top_distance = receipt.get("mass_mapping_maximum_transport_distance_m")
    if (not isinstance(exact, str) or not exact or isinstance(top_distance, bool) or
            not isinstance(top_distance, (int, float, str))):
        raise ValueError("transport envelope exact distance fields disagree")
    try:
        exact_decimal = Decimal(exact)
        top_decimal = Decimal(str(top_distance))
    except InvalidOperation:
        raise ValueError("transport envelope exact distance fields disagree") from None
    if (not exact_decimal.is_finite() or not top_decimal.is_finite() or
            exact_decimal <= 0 or top_decimal != exact_decimal):
        raise ValueError("transport envelope exact distance fields disagree")
    if receipt.get("physical_admission") is not False or receipt.get("native_steps") != 0:
        raise ValueError("transport envelope receipt contains physical/native claims")
    if cli.get("--maximum-transport-distance-m") != exact:
        raise ValueError("CLI transport distance differs from pinned current geometry envelope")
    frame = receipt.get("coordinate_frame_sha256")
    body_count = receipt.get("source_body_count")
    if cli.get("--coordinate-frame-sha256") != frame or cli.get("--source-body-count") != str(body_count):
        raise ValueError("CLI coordinate frame/body count differs from pinned geometry receipt")
    inputs = envelope.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("geometry receipt lacks its exact input hash map")
    required_input_aliases = {
        "state": "current_state",
        "geometry_skin": "candidate_skin_volume",
        "geometry_fat": "candidate_fat_volume",
        "child_node": "child_skin_node",
        "child_element": "child_skin_elements",
        "child_labels": "child_skin_labels",
        "source330_mass": "source330_mass",
        "source330_labels": "source330_labels",
        "source330_nodes": "source330_nodes",
        "source330_tets": "source330_tets",
    }
    for envelope_key, input_key in required_input_aliases.items():
        if inputs.get(envelope_key) != named_pins[input_key]["sha256"]:
            raise ValueError(f"geometry envelope does not bind exact current input {input_key}")
    if inputs.get("coordinate_frame") != frame:
        raise ValueError("geometry envelope coordinate-frame input differs from receipt")
    return {"pin": pin, "exact_decimal": exact, "receipt": receipt}


def validate_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")
    reader, control = _verify_contract_files(manifest, path)
    if manifest.get("physical_admission") is not False or manifest.get("native_steps") != 0:
        raise ValueError("038 CLI run must retain nonphysical/native-zero status")
    if type(manifest.get("maximum_wall_seconds")) is not int or not 1 <= manifest["maximum_wall_seconds"] <= 1800:
        raise ValueError("maximum_wall_seconds must be bounded to 1..1800")
    cwd = manifest.get("cwd")
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or str(Path(cwd).resolve()) != cwd or not Path(cwd).is_dir():
        raise ValueError("run cwd must be an existing canonical directory")
    output_dir = manifest.get("output_dir")
    if not isinstance(output_dir, str) or not Path(output_dir).is_absolute() or str(Path(output_dir).resolve()) != output_dir:
        raise ValueError("output_dir must be an absolute canonical path")
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("038 output directory already exists and is not empty")
    elif not output.parent.is_dir():
        raise ValueError("038 output directory parent does not exist")
    logs = {}
    for key in ("stdout_log", "stderr_log", "execution_json"):
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
            raise ValueError(f"{key} must be an absolute canonical path")
        target = Path(value)
        if target.exists():
            raise ValueError(f"{key} already exists")
        if not target.parent.is_dir():
            raise ValueError(f"{key} parent directory does not exist")
        logs[key] = value
    if len(set(logs.values())) != len(logs):
        raise ValueError("run log and execution paths must be distinct")

    argv_env = _environment(manifest, control["binary"]["path"])
    cli = _cli_options(manifest, control["binary"]["path"])
    builder_path = Path(control["receipt_builder"]["path"]).resolve(strict=True)
    builder_spec = importlib.util.spec_from_file_location("builder_keys_038", builder_path)
    if builder_spec is None or builder_spec.loader is None:
        raise RuntimeError("cannot load input-key contract")
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    pins = manifest.get("named_input_pins")
    if not isinstance(pins, dict) or set(pins) != set(builder.INPUT_KEYS):
        raise ValueError("named_input_pins do not exactly cover the frozen 013 input domain")
    canonical_pins = {}
    for name in sorted(pins):
        canonical_pins[name] = file_pin(pins[name], f"named input {name}")
    cli_input_paths = {
        "--parent-mass-csv": "source330_mass",
        "--labels-csv": "source330_labels",
        "--child-node-file": "child_skin_node",
        "--child-ele-file": "child_skin_elements",
        "--child-labels-csv": "child_skin_labels",
    }
    for option, input_name in cli_input_paths.items():
        if cli[option] != canonical_pins[input_name]["path"]:
            raise ValueError(f"CLI {option} differs from named input {input_name}")
    for skin_key, fat_key in (("child_skin_node", "child_fat_node"),
                              ("child_skin_elements", "child_fat_elements"),
                              ("child_skin_labels", "child_fat_labels")):
        if canonical_pins[skin_key] != canonical_pins[fat_key]:
            raise ValueError("current combined child inputs must have identical Skin/Fat pins")
    transport_envelope = validate_transport_envelope(
        manifest.get("mass_transport_envelope_receipt"), cli, canonical_pins)
    extras = manifest.get("file_pins")
    if not isinstance(extras, list):
        raise ValueError("file_pins must be a list")
    canonical_extras = [file_pin(ref, f"extra input {i}") for i, ref in enumerate(extras)]

    bindings = manifest.get("environment_file_pins")
    if not isinstance(bindings, list):
        raise ValueError("environment_file_pins must be a list")
    path_vars, sha_vars = set(), set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or set(binding) != {
            "path_variable", "sha_variable", "pin",
        }:
            raise ValueError(f"environment_file_pins[{index}] schema mismatch")
        path_var, sha_var = binding["path_variable"], binding["sha_variable"]
        if path_var in path_vars or sha_var in sha_vars:
            raise ValueError("environment file bindings contain duplicate variables")
        path_vars.add(path_var); sha_vars.add(sha_var)
        ref = file_pin(binding["pin"], f"environment input {index}")
        if argv_env.get(path_var) != ref["path"] or argv_env.get(sha_var) != ref["sha256"]:
            raise ValueError(f"environment file binding {index} differs from argv or pin")
        known = list(canonical_pins.values()) + canonical_extras
        if not any(row["path"] == ref["path"] and row["sha256"] == ref["sha256"] for row in known):
            raise ValueError(f"environment file binding {index} is not in the pinned input roster")
    digest_bindings = manifest.get("environment_digest_pins", [])
    if not isinstance(digest_bindings, list):
        raise ValueError("environment_digest_pins must be a list")
    digest_vars = set()
    known_refs = list(canonical_pins.values()) + canonical_extras + [
        file_pin(manifest["source_control_manifest"], "source-control"),
        file_pin(manifest["source_closure"], "source-closure"),
        transport_envelope["pin"],
    ]
    for index, binding in enumerate(digest_bindings):
        if not isinstance(binding, dict) or set(binding) != {"sha_variable", "pin"}:
            raise ValueError(f"environment_digest_pins[{index}] schema mismatch")
        sha_var = binding["sha_variable"]
        if not isinstance(sha_var, str) or sha_var in digest_vars or sha_var in sha_vars:
            raise ValueError("environment digest bindings contain duplicate variables")
        ref = file_pin(binding["pin"], f"environment digest {index}")
        if argv_env.get(sha_var) != ref["sha256"]:
            raise ValueError(f"environment digest binding {index} differs from argv or pin")
        if not any(row["path"] == ref["path"] and row["sha256"] == ref["sha256"] for row in known_refs):
            raise ValueError(f"environment digest binding {index} is not in the pinned input/control roster")
        digest_vars.add(sha_var)
    for key, value in argv_env.items():
        if value.startswith("/") and key != OUTPUT_ENV and key not in path_vars:
            raise ValueError(f"absolute-path environment input {key} lacks a file-pin binding")
        if key.endswith("_SHA256") and HEX.fullmatch(value) and key not in sha_vars and key not in digest_vars:
            raise ValueError(f"environment digest {key} lacks a file-pin binding")

    return {
        "manifest": manifest,
        "manifest_path": str(path),
        "manifest_sha256": sha_file(path),
        "control": control,
        "reader": reader,
        "named_input_pins": canonical_pins,
        "file_pins": canonical_extras,
        "environment": argv_env,
        "cli_options": cli,
        "mass_transport_envelope": transport_envelope,
    }

def preflight(path: Path) -> dict[str, Any]:
    validated = validate_manifest(path)
    manifest = validated["manifest"]
    return {
        "schema": "numi.h706.current-fractional-mass-cli-preflight-038.v1",
        "status": "PASS_LAUNCH_READY_NOT_RUN",
        "variant": VARIANT,
        "run_manifest": {"path": validated["manifest_path"], "sha256": validated["manifest_sha256"]},
        "source_control_sha256": manifest["source_control_manifest"]["sha256"],
        "source_closure_sha256": manifest["source_closure"]["sha256"],
        "binary_sha256": manifest["binary"]["sha256"],
        "mass_transport_envelope_receipt_sha256": validated["mass_transport_envelope"]["pin"]["sha256"],
        "maximum_transport_distance_m": validated["mass_transport_envelope"]["exact_decimal"],
        "named_input_count": len(validated["named_input_pins"]),
        "extra_input_count": len(validated["file_pins"]),
        "environment_assignment_count": len(validated["environment"]),
        "physical_admission": False,
        "native_steps": 0,
    }

def execute(path: Path) -> dict[str, Any]:
    validated = validate_manifest(path)
    manifest = validated["manifest"]
    output_dir = Path(manifest["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("refusing to execute into a nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path, execution_path = (Path(manifest[k]) for k in
        ("stdout_log", "stderr_log", "execution_json"))
    execution_path.write_text(json.dumps({
        "schema": EXEC_SCHEMA, "status": "RUNNING",
        "source": manifest["source"], "binary": manifest["binary"],
        "source_control_manifest": manifest["source_control_manifest"],
        "source_closure": manifest["source_closure"],
        "run_manifest": {"path": str(Path(path).resolve()), "sha256": sha_file(Path(path))},
        "full_allocator_run": True, "physical_admission": False, "native_steps": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIX)}
    start = time.monotonic()
    timed_out = False
    exit_code = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            proc = subprocess.run(
                manifest["argv"], cwd=manifest["cwd"], env=env,
                stdout=stdout, stderr=stderr,
                timeout=manifest["maximum_wall_seconds"], check=False,
            )
            exit_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
    elapsed = time.monotonic() - start
    status = ("FULL_ALLOCATOR_TIMEOUT" if timed_out else
              "CLI_EXIT_ZERO_AWAITING_SOURCE_RECEIPT" if exit_code == 0 else
              "DIAGNOSTIC_RUN_EXIT_NONZERO")
    result = {
        "schema": EXEC_SCHEMA, "status": status,
        "source": manifest["source"], "binary": manifest["binary"],
        "source_control_manifest": manifest["source_control_manifest"],
        "source_closure": manifest["source_closure"],
        "run_manifest": {"path": str(Path(path).resolve()), "sha256": sha_file(Path(path))},
        "command": manifest["argv"], "elapsed_wall_seconds": elapsed,
        "exit_code": exit_code, "timed_out": timed_out, "full_allocator_run": True,
        "stdout": {"path": str(stdout_path.resolve()), "sha256": sha_file(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr": {"path": str(stderr_path.resolve()), "sha256": sha_file(stderr_path), "bytes": stderr_path.stat().st_size},
        "physical_admission": False, "native_steps": 0,
    }
    execution_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--preflight-output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = preflight(args.run_manifest)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        if args.preflight_output.exists():
            raise ValueError("preflight output already exists")
        args.preflight_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.execute:
            result = execute(args.run_manifest)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["exit_code"] == 0 else int(result["exit_code"])
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())


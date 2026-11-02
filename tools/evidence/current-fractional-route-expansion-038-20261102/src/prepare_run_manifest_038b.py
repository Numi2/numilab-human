#!/usr/bin/env python3
"""Generate a canonical 030d run manifest from a pinned prior invocation.

The template preserves its source330/022 input lineage and non-core named
inputs. Extra CSVs are added or replace an exact named 013 input, with path,
SHA, environment assignment and receipt pin linked in one operation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LAUNCHER_PATH = HERE / "run_canonical_038.py"
LAUNCHER_SHA256 = "3d78bce1cd66aaa6ea12a5859ff5c90f285deb72df4c6548f42d02cbfb2d1bfc"
HEX = re.compile(r"^[0-9a-f]{64}$")
ENV_PREFIX = "NUMI_H706_FRACTIONAL_"
OUTPUT_ENV = ENV_PREFIX + "OUTPUT_DIR"
SOURCE_CONTROL_SHA_ENV = ENV_PREFIX + "SOURCE_CONTROL_SHA256"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def canonical_pin(ref: Any, label: str, launcher: Any) -> dict[str, Any]:
    value = launcher.file_pin(ref, label)
    return {"path": value["path"], "sha256": value["sha256"], "bytes": value["bytes"]}


def load_launcher():
    path = LAUNCHER_PATH.resolve(strict=True)
    if sha_file(path) != LAUNCHER_SHA256:
        raise ValueError("pinned canonical 038 launcher changed")
    spec = importlib.util.spec_from_file_location("canonical_launcher_038_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned 038 launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _merge_file_pins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, str] = {}
    result = []
    for row in rows:
        path, digest = row["path"], row["sha256"]
        previous = seen.get(path)
        if previous is not None:
            if previous != digest:
                raise ValueError(f"same path has conflicting SHA pins: {path}")
            continue
        seen[path] = digest
        result.append(row)
    return result


def _set_assignments(argv: list[str], binary_path: str,
                     updates: dict[str, str], append_order: list[str]) -> list[str]:
    env, cli_args = _split_for_template(argv, binary_path)
    merged = dict(env)
    merged.update(updates)
    ordered_keys = list(env)
    for key in append_order:
        if key not in ordered_keys:
            ordered_keys.append(key)
    for key in sorted(merged):
        if key not in ordered_keys:
            ordered_keys.append(key)
    return ["/usr/bin/env", *[f"{key}={merged[key]}" for key in ordered_keys], binary_path, *cli_args]


def _split_for_template(argv: list[str], binary_path: str) -> tuple[dict[str, str], list[str]]:
    if not isinstance(argv, list) or not argv or argv[0] != "/usr/bin/env":
        raise ValueError("template argv must start with /usr/bin/env")
    env: dict[str, str] = {}
    index = 1
    while index < len(argv) and argv[index] != binary_path:
        item = argv[index]
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("template binary must follow environment assignments")
        key, value = item.split("=", 1)
        if not key.startswith(ENV_PREFIX) or key in env or not value:
            raise ValueError("template has invalid or duplicate environment assignments")
        env[key] = value
        index += 1
    if index >= len(argv):
        raise ValueError("template argv lacks its pinned binary")
    return env, argv[index + 1:]


def _replace_cli_value(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    matches = [i for i, item in enumerate(out) if item == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(out):
        raise ValueError(f"template CLI must contain exactly one {flag}")
    out[matches[0] + 1] = value
    return out


def _load_builder_input_keys(control: dict[str, Any]) -> set[str]:
    ref = control.get("receipt_builder")
    if not isinstance(ref, dict) or set(ref) not in ({"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise ValueError("038 source control lacks the frozen 013 receipt-builder pin")
    p = Path(ref["path"]).resolve(strict=True)
    if sha_file(p) != ref["sha256"]:
        raise ValueError("frozen 013 receipt-builder hash changed")
    if "bytes" in ref and p.stat().st_size != ref["bytes"]:
        raise ValueError("frozen 013 receipt-builder size changed")
    spec = importlib.util.spec_from_file_location("builder_013_input_domain_038", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import 013 input domain")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.INPUT_KEYS)


def build_manifest(
    template_path: Path,
    source_control_path: Path,
    source_closure_path: Path,
    envelope_receipt_path: Path,
    output_path: Path,
    output_dir: Path,
    stdout_log: Path,
    stderr_log: Path,
    execution_json: Path,
    extra_files: list[dict[str, Any]] | None = None,
    maximum_wall_seconds: int = 1800,
) -> dict[str, Any]:
    launcher = load_launcher()
    template_path = template_path.resolve(strict=True)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or template.get("schema") != launcher.RUN_SCHEMA:
        raise ValueError("template is not a canonical current fractional CLI run manifest")
    if template.get("physical_admission") is not False or template.get("native_steps") != 0:
        raise ValueError("template has physical/native status inconsistent with a source-only launch")
    if type(maximum_wall_seconds) is not int or not 1 <= maximum_wall_seconds <= 1800:
        raise ValueError("maximum wall time must be an integer bounded to 1..1800")
    source_control_path = source_control_path.resolve(strict=True)
    source_closure_path = source_closure_path.resolve(strict=True)
    envelope_receipt_path = envelope_receipt_path.resolve(strict=True)
    control = json.loads(source_control_path.read_text(encoding="utf-8"))
    closure = json.loads(source_closure_path.read_text(encoding="utf-8"))
    control_ref = canonical_pin({"path": str(source_control_path), "sha256": sha_file(source_control_path)}, "source-control", launcher)
    closure_ref = canonical_pin({"path": str(source_closure_path), "sha256": sha_file(source_closure_path)}, "source-closure", launcher)
    envelope_pin = pin(envelope_receipt_path)
    if control.get("variant") != launcher.VARIANT or control.get("status") != "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN":
        raise ValueError("source control is not the immutable 030d controls-only record")
    if closure.get("source_control_manifest", {}).get("sha256") != control_ref["sha256"]:
        raise ValueError("038 source closure does not bind the selected source-control file")

    builder_keys = _load_builder_input_keys(control)
    raw_named = template.get("named_input_pins")
    if not isinstance(raw_named, dict) or not builder_keys.issubset(raw_named):
        raise ValueError("template does not contain the complete frozen 013 named input domain")
    named: dict[str, dict[str, Any]] = {
        key: canonical_pin(raw_named[key], f"named input {key}", launcher)
        for key in sorted(builder_keys)
    }
    file_rows: list[dict[str, Any]] = []
    for key in sorted(set(raw_named) - builder_keys):
        file_rows.append(canonical_pin(raw_named[key], f"upstream named input {key}", launcher))

    raw_file_pins = template.get("file_pins")
    if not isinstance(raw_file_pins, list):
        raise ValueError("template file_pins must be a list")
    file_rows.extend(canonical_pin(row, f"template file pin {index}", launcher)
                     for index, row in enumerate(raw_file_pins))
    file_rows.append(envelope_pin)

    argv = template.get("argv")
    prior_binary = template.get("binary", {}).get("path")
    if not isinstance(prior_binary, str):
        raise ValueError("template lacks its original binary path")
    env, cli_args = _split_for_template(argv, prior_binary)
    # Parse the old CLI strictly before changing only its source identity and
    # current geometry-derived distance/frame/body-count values.
    template_for_parse = dict(template)
    template_for_parse["binary"] = template["binary"]
    launcher._cli_options(template_for_parse, prior_binary)

    extra_files = list(extra_files or [])
    extra_rows: list[dict[str, Any]] = []
    occupied_vars: set[str] = set()
    for index, item in enumerate(extra_files):
        allowed = {"path", "sha256", "path_variable", "sha_variable", "named_input_key"}
        if not isinstance(item, dict) or not {"path", "sha256", "path_variable", "sha_variable"}.issubset(item) or set(item) - allowed:
            raise ValueError(f"extra file {index} schema mismatch")
        path_variable, sha_variable = item["path_variable"], item["sha_variable"]
        if (not isinstance(path_variable, str) or not path_variable.startswith(ENV_PREFIX) or
                not isinstance(sha_variable, str) or not sha_variable.startswith(ENV_PREFIX) or
                not sha_variable.endswith("_SHA256") or path_variable == sha_variable or
                path_variable in occupied_vars or sha_variable in occupied_vars):
            raise ValueError(f"extra file {index} environment variables are invalid or duplicated")
        occupied_vars.update((path_variable, sha_variable))
        extra_path = Path(item["path"]).resolve(strict=True)
        extra_pin = canonical_pin({"path": str(extra_path), "sha256": item["sha256"]}, f"extra file {index}", launcher)
        if not (extra_pin["path"].endswith(".csv") or extra_pin["path"].endswith(".csv.gz")):
            raise ValueError(f"extra file {index} must be a CSV or CSV.GZ")
        input_key = item.get("named_input_key")
        if input_key is not None:
            if not isinstance(input_key, str) or input_key not in builder_keys:
                raise ValueError(f"extra file {index} names an unsupported 013 input key")
            named[input_key] = extra_pin
        existing_path, existing_sha = path_variable in env, sha_variable in env
        if existing_path != existing_sha:
            raise ValueError(f"extra file {index} overrides only half of an existing environment pair")
        extra_rows.append({"path_variable": path_variable, "sha_variable": sha_variable,
                           "pin": extra_pin})
        file_rows.append(extra_pin)

    # Core file variables remain exact aliases of the frozen 013 named input
    # domain. New witness CSV replacements update the same named key above.
    standard_file_env = {
        "NUMI_H706_FRACTIONAL_P1_SKIN_NPZ": ("NUMI_H706_FRACTIONAL_P1_SKIN_SHA256", "p1_skin_support"),
        "NUMI_H706_FRACTIONAL_P1_FAT_NPZ": ("NUMI_H706_FRACTIONAL_P1_FAT_SHA256", "p1_fat_support"),
        "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ": ("NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256", "carrier_skin_packet"),
        "NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ": ("NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256", "carrier_fat_packet"),
        "NUMI_H706_FRACTIONAL_EXACT_PROJECTION_CSV": ("NUMI_H706_FRACTIONAL_EXACT_PROJECTION_SHA256", "projection_evidence"),
        "NUMI_H706_FRACTIONAL_SOURCE_CENTROID_CSV": ("NUMI_H706_FRACTIONAL_SOURCE_CENTROID_SHA256", "source_centroid_slots"),
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV": ("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256", "carrier_witness_skin"),
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV": ("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256", "carrier_witness_fat"),
        "NUMI_H706_FRACTIONAL_SOURCE330_NODES": ("NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256", "source330_nodes"),
        "NUMI_H706_FRACTIONAL_SOURCE330_TETS": ("NUMI_H706_FRACTIONAL_SOURCE330_TETS_SHA256", "source330_tets"),
    }
    updates: dict[str, str] = {
        OUTPUT_ENV: str(output_dir.resolve()),
        SOURCE_CONTROL_SHA_ENV: control_ref["sha256"],
    }
    extra_path_vars = {row["path_variable"]: row for row in extra_rows}
    for path_var, (sha_var, input_key) in standard_file_env.items():
        if path_var in extra_path_vars:
            row = extra_path_vars[path_var]
            if row["sha_variable"] != sha_var or row["pin"] != named[input_key]:
                raise ValueError(f"replacement {path_var} must update its exact named input {input_key}")
        else:
            updates[path_var] = named[input_key]["path"]
            updates[sha_var] = named[input_key]["sha256"]
    for row in extra_rows:
        updates[row["path_variable"]] = row["pin"]["path"]
        updates[row["sha_variable"]] = row["pin"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_STATE_SHA256"] = named["current_state"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_GEOMETRY_SKIN_SHA256"] = named["candidate_skin_volume"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_VOLUME_SKIN_SHA256"] = named["candidate_skin_volume"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_GEOMETRY_FAT_SHA256"] = named["candidate_fat_volume"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_VOLUME_FAT_SHA256"] = named["candidate_fat_volume"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256"] = named["source330_mass"]["sha256"]
    updates["NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256"] = control_ref["sha256"]

    envelope_receipt = json.loads(envelope_receipt_path.read_text(encoding="utf-8"))
    envelope = envelope_receipt.get("mass_transport_envelope", {})
    distance = envelope.get("exact_decimal")
    if not isinstance(distance, str):
        raise ValueError("current geometry receipt lacks the exact transport distance")
    cli_args = _replace_cli_value(cli_args, "--coordinate-frame-sha256", envelope_receipt["coordinate_frame_sha256"])
    cli_args = _replace_cli_value(cli_args, "--source-body-count", str(envelope_receipt["source_body_count"]))
    cli_args = _replace_cli_value(cli_args, "--maximum-transport-distance-m", distance)

    output_dir = output_dir.resolve()
    output_path = output_path.resolve()
    stdout_log, stderr_log, execution_json = (Path(x).resolve() for x in
                                               (stdout_log, stderr_log, execution_json))
    run_config_path = output_path.with_name("run-config-038.json")
    for path in (output_dir, stdout_log, stderr_log, execution_json, output_path, run_config_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite existing launch output: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"launch output parent directory does not exist: {path.parent}")
    if len({str(stdout_log), str(stderr_log), str(execution_json), str(output_path),
            str(run_config_path)}) != 5:
        raise ValueError("run manifest/config/log paths must be distinct")

    template_env, _ = _split_for_template(argv, prior_binary)
    template_run_config_sha = template_env.get("NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256")
    template_run_configs = [
        canonical_pin(row, "template run config", launcher)
        for row in [*raw_named.values(), *raw_file_pins]
        if isinstance(row, dict) and row.get("sha256") == template_run_config_sha
        and Path(str(row.get("path", ""))).name in {"run-config.json", "run-config-038.json"}
    ]
    if len(template_run_configs) != 1:
        raise ValueError("template RUN_CONFIG_SHA256 must identify exactly one pinned run-config.json")
    template_run_config_pin = template_run_configs[0]
    template_run_config_obj = json.loads(Path(template_run_config_pin["path"]).read_text(encoding="utf-8"))
    if not isinstance(template_run_config_obj, dict):
        raise ValueError("template run config must be a JSON object")
    run_config = {
        "schema": "numi.h706.diagnostic-run-config-038.v1",
        "template_run_manifest": pin(template_path),
        "template_run_config": template_run_config_pin,
        "variant": control["variant"],
        "source": control["source"],
        "binary": control["binary"],
        "source_control_manifest": control_ref,
        "source_closure": closure_ref,
        "mass_transport_envelope_receipt": envelope_pin,
        "named_input_pins": named,
        "upstream_named_input_pins": {
            key: value for key, value in raw_named.items() if key not in builder_keys
        },
        "supplemental_input_pins": _merge_file_pins(file_rows),
        "output_dir": str(output_dir),
        "maximum_wall_seconds": maximum_wall_seconds,
        "physical_admission": False,
        "native_steps": 0,
        "scope": "source-controlled allocator diagnostic plan; not executed; no physical admission",
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    run_config_pin = pin(run_config_path)
    file_rows.append(run_config_pin)
    updates["NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256"] = run_config_pin["sha256"]

    named_rows = list(named.values())
    file_rows = _merge_file_pins(file_rows)
    all_pins = named_rows + file_rows + [control_ref, closure_ref]
    env_map, _ = _split_for_template(argv, prior_binary)
    for key, value in env_map.items():
        if key.endswith("_SHA256") and key not in updates:
            matching = [row for row in all_pins if row["sha256"] == value]
            unique = {(row["path"], row["sha256"]): row for row in matching}
            if len(unique) == 1:
                # The pin is represented explicitly even though this key has
                # no path-valued sibling in the C++ interface.
                updates[key] = value
            elif len(unique) == 0:
                raise ValueError(f"hash-only runtime value is not among pinned files: {key}")
            else:
                # Resolve the only two fixed non-path digest inputs explicitly.
                semantic = {
                    "NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256": named["source330_mass"],
                    "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256": None,
                }
                chosen = semantic.get(key)
                if chosen is None and key == "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256":
                    matches = [row for row in matching if Path(row["path"]).name == "run-config.json"]
                    if len(matches) == 1:
                        chosen = matches[0]
                if chosen is None or chosen["sha256"] != value:
                    raise ValueError(f"ambiguous hash-only runtime input: {key}")
                updates[key] = value
    ordered_updates = [OUTPUT_ENV, SOURCE_CONTROL_SHA_ENV]
    argv = _set_assignments(argv, prior_binary, updates, ordered_updates)
    argv = list(argv)
    _, old_cli_args = _split_for_template(argv, prior_binary)
    # Replace source binary and preserve the exact validated CLI contract with
    # only current geometry frame/count/distance updates.
    _, old_env_values = launcher._split_argv(template["argv"], prior_binary)
    del old_env_values
    # _set_assignments placed the old CLI suffix; rebuild with the revised one.
    env_values, _ = _split_for_template(argv, prior_binary)
    argv = ["/usr/bin/env", *[f"{k}={v}" for k, v in env_values.items()],
            control["binary"]["path"], *cli_args]

    manifest = dict(template)
    raw_extra_named = {key: value for key, value in raw_named.items() if key not in builder_keys}
    manifest.update({
        "schema": launcher.RUN_SCHEMA,
        "variant": launcher.VARIANT,
        "source": control["source"],
        "binary": control["binary"],
        "source_control_manifest": {"path": str(source_control_path), "sha256": control_ref["sha256"]},
        "source_closure": {"path": str(source_closure_path), "sha256": closure_ref["sha256"]},
        "mass_transport_envelope_receipt": envelope_pin,
        "run_config": run_config_pin,
        "template_run_config": template_run_config_pin,
        "template_manifest": pin(template_path),
        "upstream_named_input_pins": raw_extra_named,
        "argv": argv,
        "cwd": str(Path(template["cwd"]).resolve()),
        "output_dir": str(output_dir),
        "maximum_wall_seconds": maximum_wall_seconds,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "execution_json": str(execution_json),
        "named_input_pins": named,
        "file_pins": file_rows,
        "environment_file_pins": [],
        "environment_digest_pins": [],
        "physical_admission": False,
        "native_steps": 0,
    })

    file_bindings = []
    for path_var, (sha_var, input_key) in standard_file_env.items():
        file_bindings.append({"path_variable": path_var, "sha_variable": sha_var,
                              "pin": named[input_key]})
    file_bindings.extend(row for row in extra_rows
                         if row["path_variable"] not in standard_file_env)
    # Any path-valued runtime variable outside the declared standard aliases
    # must be declared as an extra file. This rejects hidden or stale paths.
    env_values, _ = _split_for_template(argv, control["binary"]["path"])
    declared_path_vars = {row["path_variable"] for row in file_bindings}
    for key, value in env_values.items():
        if value.startswith("/") and key != OUTPUT_ENV and key not in declared_path_vars:
            raise ValueError(f"unrecognized absolute-path environment input: {key}")
    manifest["environment_file_pins"] = file_bindings
    digest_bindings = []
    path_sha_vars = {row["sha_variable"] for row in file_bindings}
    known_by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in all_pins:
        known_by_sha.setdefault(row["sha256"], []).append(row)
    for key, value in env_values.items():
        if not key.endswith("_SHA256") or key in path_sha_vars:
            continue
        if key == "NUMI_H706_FRACTIONAL_STATE_SHA256":
            ref = named["current_state"]
        elif key in ("NUMI_H706_FRACTIONAL_GEOMETRY_SKIN_SHA256", "NUMI_H706_FRACTIONAL_VOLUME_SKIN_SHA256"):
            ref = named["candidate_skin_volume"]
        elif key in ("NUMI_H706_FRACTIONAL_GEOMETRY_FAT_SHA256", "NUMI_H706_FRACTIONAL_VOLUME_FAT_SHA256"):
            ref = named["candidate_fat_volume"]
        elif key == "NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256":
            ref = named["source330_mass"]
        elif key == "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256":
            ref = control_ref
        else:
            matches = known_by_sha.get(value, [])
            unique = {(x["path"], x["sha256"]): x for x in matches}
            if len(unique) != 1:
                raise ValueError(f"hash-only runtime input must resolve to one file pin: {key}")
            ref = next(iter(unique.values()))
        if ref["sha256"] != value:
            raise ValueError(f"hash-only runtime value differs from expected input: {key}")
        digest_bindings.append({"sha_variable": key, "pin": ref})
    manifest["environment_digest_pins"] = digest_bindings

    if not output_path.parent.is_dir():
        raise ValueError("run manifest output parent directory does not exist")
    temp_path = output_path.with_name(output_path.name + ".tmp")
    if temp_path.exists() or output_path.exists():
        raise ValueError("refusing to overwrite manifest output")
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        launcher.validate_manifest(temp_path)
        os.replace(temp_path, output_path)
    except Exception:
        # Preserve an invalid draft as evidence when output validation fails.
        failed = output_path.with_name(output_path.name + ".rejected-draft.json")
        if temp_path.exists():
            os.replace(temp_path, failed)
        raise
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-manifest", required=True, type=Path)
    parser.add_argument("--source-control", required=True, type=Path)
    parser.add_argument("--source-closure", required=True, type=Path)
    parser.add_argument("--envelope-receipt", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    parser.add_argument("--execution-json", required=True, type=Path)
    parser.add_argument("--maximum-wall-seconds", type=int, default=1800)
    parser.add_argument("--extra-file", nargs="+", action="append", default=[],
                        metavar="ARG", help="repeatable: PATH_VAR SHA_VAR CSV_PATH [NAMED_INPUT_KEY]")
    args = parser.parse_args()
    extras = []
    for row in args.extra_file:
        if len(row) not in (3, 4):
            parser.error("--extra-file needs PATH_VAR SHA_VAR CSV_PATH [NAMED_INPUT_KEY]")
        item = {"path_variable": row[0], "sha_variable": row[1],
                "path": row[2], "sha256": sha_file(Path(row[2]).resolve(strict=True))}
        if len(row) == 4 and row[3] != "-":
            item["named_input_key"] = row[3]
        extras.append(item)
    try:
        build_manifest(args.template_manifest, args.source_control, args.source_closure,
                       args.envelope_receipt, args.output_manifest, args.output_dir,
                       args.stdout_log, args.stderr_log, args.execution_json,
                       extras, args.maximum_wall_seconds)
    except Exception as exc:
        parser.error(str(exc))
    print(f"run_manifest_038=PASS_SOURCE_CONTROLS_AND_INPUTS_NOT_RUN path={args.output_manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

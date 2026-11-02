#!/usr/bin/env python3
"""030d-only compatibility view for two separately pinned run-control hashes.

The frozen 006 owner authenticates the 27 app variables. 038 adds two control
hashes to its argv; this adapter verifies those against the actual run manifest
and exact files, then delegates the unchanged 006 app-variable checks.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

BASE_ADAPTER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-006-source-geometry-scope/prepare_current_native_1025_direct_o_006.py")
BASE_ADAPTER_SHA256 = "249ebe4e548da2d4f1633c03dd994de313b1c591b47d82591fb8c5ea2a12609d"
PREFIX = "NUMI_H706_FRACTIONAL_"
CONTROL_VARIABLES = {
    "RUN_CONFIG_SHA256": "run_config",
    "SOURCE_CONTROL_SHA256": "source_control_manifest",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_base():
    path = BASE_ADAPTER_PATH.resolve(strict=True)
    if sha_file(path) != BASE_ADAPTER_SHA256:
        raise ValueError("pinned 006 app identity adapter changed")
    spec = importlib.util.spec_from_file_location("current_native_006_030d_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned 006 app identity adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ORIGINAL = _load_base()
base = _ORIGINAL.base
route_module = _ORIGINAL.route_module


def validate_run_control_bindings(cfg: dict[str, Any], adapter_module: Any = _ORIGINAL) -> tuple[Path, list[str]]:
    """Authenticate full actual argv and the two non-app control files."""
    base_module = adapter_module.base
    source = cfg.get("h706_fractional_environment_source")
    if not isinstance(source, dict) or set(source) != {"path", "resolved_path", "sha256", "argv_sha256"}:
        raise ValueError("H706 app environment must bind the exact full argv manifest")
    source_path = Path(source["path"]) if isinstance(source.get("path"), str) else Path()
    if not source_path.is_absolute() or source["resolved_path"] != str(source_path.resolve()):
        raise ValueError("H706 argv source path binding changed")
    if not source_path.is_file() or sha_file(source_path) != source["sha256"]:
        raise ValueError("H706 argv source manifest hash mismatch")
    source_doc = base_module.read_json(source_path)
    argv = source_doc.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("H706 actual argv is malformed")
    if base_module.json_digest(argv) != source["argv_sha256"]:
        raise ValueError("H706 full argv digest mismatch")

    actual: dict[str, str] = {}
    for item in argv:
        if item.startswith(PREFIX):
            if "=" not in item:
                raise ValueError("H706 fractional argv item is malformed")
            key, value = item[len(PREFIX):].split("=", 1)
            if key in actual:
                raise ValueError(f"duplicate H706 fractional argv key: {key}")
            actual[key] = value
    expected = set(adapter_module.H706_FRACTIONAL_ENV_KEYS) | {"OUTPUT_DIR", *CONTROL_VARIABLES}
    if set(actual) != expected:
        raise ValueError(
            f"030d argv key schema mismatch: missing={sorted(expected - set(actual))}, "
            f"extra={sorted(set(actual) - expected)}")
    env = cfg.get("h706_fractional_env")
    if not isinstance(env, dict) or set(env) != adapter_module.H706_FRACTIONAL_ENV_KEYS:
        missing = sorted(adapter_module.H706_FRACTIONAL_ENV_KEYS - set(env or {}))
        extra = sorted(set(env or {}) - adapter_module.H706_FRACTIONAL_ENV_KEYS)
        raise ValueError(f"H706 app environment schema mismatch: missing={missing}, extra={extra}")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("H706 app environment values must be strings")
    app_actual = {key: value for key, value in actual.items()
                  if key in adapter_module.H706_FRACTIONAL_ENV_KEYS}
    if app_actual != env:
        raise ValueError("H706 app env differs from the exact 038 argv")
    if not Path(actual["OUTPUT_DIR"]).is_absolute():
        raise ValueError("H706 CLI-only OUTPUT_DIR must be absolute")

    run_pins = cfg.get("h706_fractional_run_control_pins")
    if not isinstance(run_pins, dict) or set(run_pins) != set(CONTROL_VARIABLES.values()):
        raise ValueError("H706 run-control pins must separately cover run_config and source_control_manifest")
    for variable, manifest_key in CONTROL_VARIABLES.items():
        pin = run_pins[manifest_key]
        manifest_ref = source_doc.get(manifest_key)
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise ValueError(f"H706 run-control {manifest_key} pin is malformed")
        if not isinstance(manifest_ref, dict):
            raise ValueError(f"actual run manifest lacks {manifest_key}")
        path = Path(pin["path"])
        if (not path.is_absolute() or str(path) != manifest_ref.get("path") or
                pin["sha256"] != manifest_ref.get("sha256") or
                actual[variable] != pin["sha256"]):
            raise ValueError(f"H706 run-control {manifest_key} differs from actual argv/manifest")
        if not path.is_file() or sha_file(path) != pin["sha256"]:
            raise ValueError(f"H706 run-control {manifest_key} file hash mismatch")
    filtered_argv = [item for item in argv if not any(
        item.startswith(PREFIX + variable + "=") for variable in CONTROL_VARIABLES)]
    return source_path, filtered_argv


def verify_h706_fractional_env(cfg: dict[str, Any]) -> dict[str, str]:
    """Verify full 038 argv, then delegate all frozen 006 app-input checks."""
    source_path, filtered_argv = validate_run_control_bindings(cfg)
    delegated_cfg = copy.deepcopy(cfg)
    delegated_cfg["h706_fractional_environment_source"]["argv_sha256"] = base.json_digest(filtered_argv)
    original_read_json = base.read_json

    def read_json_with_app_argv(path: Path):
        document = original_read_json(path)
        if Path(path).resolve() == source_path:
            document = dict(document)
            document["argv"] = filtered_argv
        return document

    base.read_json = read_json_with_app_argv
    try:
        return _ORIGINAL.verify_h706_fractional_env(delegated_cfg)
    finally:
        base.read_json = original_read_json

#!/usr/bin/env python3
"""Materialize a current-038 source-only identity config for the 010i route.

This binds the canonical 038 H706 app inputs without claiming mass admission.
The final native config is created only after a passing 015 receipt and the
verified recorded-flow bundle are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-010i-envelope-bound-v1/final-controls-004/current-native-launch-010-build014b.template.json")
EXPECTED_INPUTS = {
    "candidate_fat_volume", "candidate_skin_volume",
    "carrier_fat_packet", "carrier_skin_packet",
    "carrier_witness_fat", "carrier_witness_skin",
    "child_fat_elements", "child_fat_labels", "child_fat_node",
    "child_skin_elements", "child_skin_labels", "child_skin_node",
    "current_state", "p1_fat_support", "p1_skin_support",
    "projection_evidence", "projection_report", "source330_labels",
    "source330_mass", "source330_nodes", "source330_tets",
    "source_centroid_slots",
}
SOURCE_CONTROL_KEYS = {"NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256", "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256"}
APP_PREFIX = "NUMI_H706_FRACTIONAL_"
CHANGE_KEYS = {
    "CARRIERS_FAT_NPZ", "CARRIERS_FAT_SHA256",
    "CARRIERS_SKIN_NPZ", "CARRIERS_SKIN_SHA256",
    "CARRIER_WITNESS_FAT_CSV", "CARRIER_WITNESS_FAT_SHA256",
    "CARRIER_WITNESS_SKIN_CSV", "CARRIER_WITNESS_SKIN_SHA256",
    "SOURCE_CENTROID_CSV", "SOURCE_CENTROID_SHA256",
}
OLD_032D = {
    "CARRIERS_SKIN_NPZ": ("skin-adaptive-source-quadrature-032d.npz", "7f648262440539761f5cc6f14c9a2a7c5bf73c5cf75b310ec747872d83c0efb3"),
    "CARRIERS_FAT_NPZ": ("fat-adaptive-source-quadrature-032d.npz", "12068b0a15bcc24dd8adae0d452ad6bf49b443a8a4b78712093aaa26713126a8"),
}
V15_NAMES = {
    "carrier_witness_skin": ("skin-witnesses-currentR-sparse-032dplus-v15.csv", "CARRIER_WITNESS_SKIN_CSV", "CARRIER_WITNESS_SKIN_SHA256"),
    "carrier_witness_fat": ("fat-witnesses-currentR-sparse-032dplus-v15.csv", "CARRIER_WITNESS_FAT_CSV", "CARRIER_WITNESS_FAT_SHA256"),
}

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def load_obj(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value

def exact_ref(pin: Any, label: str, *, live_hash: bool = True) -> dict[str, Any]:
    if not isinstance(pin, dict) or set(pin) not in ({"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise ValueError(f"{label} must be an exact path/SHA reference")
    raw, digest = pin.get("path"), pin.get("sha256")
    if not isinstance(raw, str) or not Path(raw).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} has a malformed path/SHA")
    path = Path(raw)
    if str(path.resolve()) != raw or not path.is_file():
        raise ValueError(f"{label} file is missing or noncanonical")
    if "bytes" in pin and (type(pin["bytes"]) is not int or pin["bytes"] != path.stat().st_size):
        raise ValueError(f"{label} byte count changed")
    if live_hash and sha_file(path) != digest:
        raise ValueError(f"{label} SHA changed")
    return {"path": raw, "sha256": digest, **({"bytes": pin["bytes"]} if "bytes" in pin else {})}

def env_from_manifest(run: dict[str, Any], template_env: dict[str, str]) -> dict[str, str]:
    argv = run.get("argv")
    if not isinstance(argv, list):
        raise ValueError("038 run manifest lacks argv")
    app_env: dict[str, str] = {}
    controls: dict[str, str] = {}
    output_dir: str | None = None
    for arg in argv:
        if not isinstance(arg, str) or not arg.startswith(APP_PREFIX):
            continue
        if "=" not in arg:
            raise ValueError("038 H706 assignment is malformed")
        key, value = arg[len(APP_PREFIX):].split("=", 1)
        full_key = APP_PREFIX + key
        if full_key == APP_PREFIX + "OUTPUT_DIR":
            if output_dir is not None or not Path(value).is_absolute() or value != run.get("output_dir"):
                raise ValueError("038 H706 output directory assignment is missing, duplicated, or mismatched")
            output_dir = value
        elif full_key in SOURCE_CONTROL_KEYS:
            if full_key in controls:
                raise ValueError(f"duplicate run-control assignment: {full_key}")
            controls[full_key] = value
        else:
            if key in app_env:
                raise ValueError(f"duplicate H706 app assignment: {key}")
            app_env[key] = value
    if set(controls) != SOURCE_CONTROL_KEYS:
        raise ValueError("038 run manifest must contain both run-config and source-control hashes")
    for env_key, ref_key in (
        ("NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256", "run_config"),
        ("NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256", "source_control_manifest"),
    ):
        pinned = exact_ref(run.get(ref_key), f"run.{ref_key}")
        if controls[env_key] != pinned["sha256"]:
            raise ValueError(f"{env_key} differs from the run manifest's pinned file")
    if set(app_env) != set(template_env):
        raise ValueError(f"038 H706 app key set differs: missing={sorted(set(template_env)-set(app_env))}, extra={sorted(set(app_env)-set(template_env))}")
    if app_env.get("ENABLE") != "1":
        raise ValueError("038 H706 app environment is not enabled")
    changed = {key for key, value in app_env.items() if template_env.get(key) != value}
    if changed != CHANGE_KEYS:
        raise ValueError(f"unexpected 038 app-environment changes: {sorted(changed ^ CHANGE_KEYS)}")
    return app_env

def validate_038_identity(run_path: Path, preflight_path: Path, template: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    run = load_obj(run_path, "038 run manifest")
    preflight = load_obj(preflight_path, "038 preflight")
    run_sha = sha_file(run_path)
    if preflight.get("schema") != "numi.h706.current-fractional-mass-cli-preflight-038.v1" or preflight.get("status") != "PASS_LAUNCH_READY_NOT_RUN":
        raise ValueError("038 preflight is not the source-only launch-ready, not-run result")
    if preflight.get("physical_admission") is not False or preflight.get("native_steps") != 0:
        raise ValueError("038 preflight overclaims physical/native execution")
    if preflight.get("run_manifest") != {"path": str(run_path.resolve()), "sha256": run_sha}:
        raise ValueError("038 preflight does not bind the exact run manifest")
    if run.get("physical_admission") is not False or run.get("native_steps") != 0:
        raise ValueError("038 run manifest overclaims physical/native execution")
    if run.get("source_control_manifest", {}).get("sha256") != preflight.get("source_control_sha256"):
        raise ValueError("038 source-control binding differs from its preflight")
    if run.get("source_closure", {}).get("sha256") != preflight.get("source_closure_sha256"):
        raise ValueError("038 source-closure binding differs from its preflight")
    for field in ("source_control_manifest", "source_closure", "run_config", "mass_transport_envelope_receipt"):
        exact_ref(run.get(field), f"run.{field}")
    named = run.get("named_input_pins")
    if not isinstance(named, dict) or set(named) != EXPECTED_INPUTS:
        raise ValueError("038 must carry the exact 22-name mass input roster")
    for name, pin in named.items():
        exact_ref(pin, f"named_input_pins.{name}", live_hash=False)
    app_env = env_from_manifest(run, template["h706_fractional_env"])
    return run, preflight, app_env

def input_refs_from_config(cfg: dict[str, Any], app_env: dict[str, str], run: dict[str, Any]) -> dict[str, tuple[str, str]]:
    pins = cfg.get("h706_fractional_path_pins")
    native = cfg.get("native_preflight_env")
    if not isinstance(pins, dict) or not isinstance(native, dict):
        raise ValueError("identity config lacks source or native path pins")
    def p(pin_key: str) -> tuple[str, str]:
        item = pins[pin_key]
        return str(Path(item["path"]).resolve()), item["sha256"]
    def hfile(path_key: str, hash_key: str) -> tuple[str, str]:
        return str(Path(app_env[path_key]).resolve()), app_env[hash_key]
    def nfile(path_key: str, hash_key: str) -> tuple[str, str]:
        return str(Path(native[path_key]).resolve()), native[hash_key]
    return {
        "candidate_fat_volume": p("VOLUME_FAT_SHA256"),
        "candidate_skin_volume": p("VOLUME_SKIN_SHA256"),
        "carrier_fat_packet": hfile("CARRIERS_FAT_NPZ", "CARRIERS_FAT_SHA256"),
        "carrier_skin_packet": hfile("CARRIERS_SKIN_NPZ", "CARRIERS_SKIN_SHA256"),
        "carrier_witness_fat": hfile("CARRIER_WITNESS_FAT_CSV", "CARRIER_WITNESS_FAT_SHA256"),
        "carrier_witness_skin": hfile("CARRIER_WITNESS_SKIN_CSV", "CARRIER_WITNESS_SKIN_SHA256"),
        "child_fat_elements": nfile("TETGEN_ELEMENTS", "TETGEN_ELEMENTS_SHA256"),
        "child_fat_labels": nfile("CHILD_LABELS", "CHILD_LABELS_SHA256"),
        "child_fat_node": nfile("TETGEN_NODE", "TETGEN_NODE_SHA256"),
        "child_skin_elements": nfile("TETGEN_ELEMENTS", "TETGEN_ELEMENTS_SHA256"),
        "child_skin_labels": nfile("CHILD_LABELS", "CHILD_LABELS_SHA256"),
        "child_skin_node": nfile("TETGEN_NODE", "TETGEN_NODE_SHA256"),
        "current_state": p("STATE_SHA256"),
        "p1_fat_support": hfile("P1_FAT_NPZ", "P1_FAT_SHA256"),
        "p1_skin_support": hfile("P1_SKIN_NPZ", "P1_SKIN_SHA256"),
        "projection_evidence": hfile("EXACT_PROJECTION_CSV", "EXACT_PROJECTION_SHA256"),
        "projection_report": (str(Path(run["named_input_pins"]["projection_report"]["path"]).resolve()), run["named_input_pins"]["projection_report"]["sha256"]),
        "source330_labels": nfile("PARENT_LABELS_CSV", "PARENT_LABELS_SHA256"),
        "source330_mass": p("SOURCE330_MASS_SHA256"),
        "source330_nodes": hfile("SOURCE330_NODES", "SOURCE330_NODES_SHA256"),
        "source330_tets": hfile("SOURCE330_TETS", "SOURCE330_TETS_SHA256"),
        "source_centroid_slots": hfile("SOURCE_CENTROID_CSV", "SOURCE_CENTROID_SHA256"),
    }

def materialize(template_path: Path, run_path: Path, preflight_path: Path, out_path: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    template = load_obj(template_path, "010 build014b template")
    if template.get("schema") != "numi.human.current-native-launch-1025.v1":
        raise ValueError("unexpected direct-O native launch template schema")
    if template.get("mass_admission") is not None:
        raise ValueError("identity materializer accepts only the mass-pending base template")
    if not isinstance(template.get("geometry_admission"), dict):
        raise ValueError("source geometry receipt must remain bound in the direct-O template")
    env = template.get("h706_fractional_env")
    if not isinstance(env, dict) or not isinstance(template.get("h706_fractional_path_pins"), dict):
        raise ValueError("template lacks the closed H706 environment and path map")
    run, preflight, app_env = validate_038_identity(run_path, preflight_path, template)
    expected_refs = input_refs_from_config(template, app_env, run)
    actual_named = run["named_input_pins"]
    for name, (path, digest) in expected_refs.items():
        pin = actual_named[name]
        if (str(Path(pin["path"]).resolve()), pin["sha256"]) != (path, digest):
            raise ValueError(f"038 named input {name} differs from current native identity")
    for key, (basename, digest) in OLD_032D.items():
        if Path(app_env[key]).name != basename or app_env[key.replace("_NPZ", "_SHA256")] != digest:
            raise ValueError(f"old 032d carrier packet changed: {key}")
    for name, (basename, path_key, hash_key) in V15_NAMES.items():
        pin = actual_named[name]
        if Path(pin["path"]).name != basename or Path(app_env[path_key]).name != basename or app_env[hash_key] != pin["sha256"]:
            raise ValueError(f"current v15 witness pin changed: {name}")
    if not output_root.is_absolute() or output_root.exists():
        raise ValueError("native output root must be absolute and new")
    cfg = json.loads(json.dumps(template))
    cfg["h706_fractional_env"] = app_env
    cfg["h706_fractional_environment_source"] = {
        "path": str(run_path.resolve()), "resolved_path": str(run_path.resolve()),
        "sha256": sha_file(run_path), "argv_sha256": json_digest(run["argv"]),
    }
    cfg["h706_fractional_run_control_pins"] = {
        name: {"path": run[name]["path"], "sha256": run[name]["sha256"]}
        for name in ("run_config", "source_control_manifest")
    }
    updated_pins = cfg["h706_fractional_path_pins"]
    pair_map = {
        "P1_SKIN_NPZ":"P1_SKIN_SHA256", "P1_FAT_NPZ":"P1_FAT_SHA256",
        "SOURCE330_NODES":"SOURCE330_NODES_SHA256", "SOURCE330_TETS":"SOURCE330_TETS_SHA256",
        "CARRIERS_SKIN_NPZ":"CARRIERS_SKIN_SHA256", "CARRIERS_FAT_NPZ":"CARRIERS_FAT_SHA256",
        "SOURCE_CENTROID_CSV":"SOURCE_CENTROID_SHA256", "EXACT_PROJECTION_CSV":"EXACT_PROJECTION_SHA256",
        "CARRIER_WITNESS_SKIN_CSV":"CARRIER_WITNESS_SKIN_SHA256", "CARRIER_WITNESS_FAT_CSV":"CARRIER_WITNESS_FAT_SHA256",
    }
    for path_key, hash_key in pair_map.items():
        path = Path(app_env[path_key])
        updated_pins[path_key] = {"path": str(path), "resolved_path": str(path.resolve()), "sha256": app_env[hash_key]}
    projection = actual_named["projection_report"]
    cfg["h706_fractional_auxiliary_path_pins"] = {
        "projection_report": {"path": projection["path"], "resolved_path": str(Path(projection["path"]).resolve()), "sha256": projection["sha256"]}
    }
    cfg["mass_admission"] = None
    cfg.pop("recorded_flow_bundle", None)
    cfg["output_root"] = str(output_root)
    cfg["candidate_label"] = "current433_geometry-envelope-1058b_v15-032d"
    cfg["description"] = "Source-only 010i identity config derived from passing canonical 038 v15 witness inputs; 032d carrier NPZs preserved. Mass admission and recorded-flow bundle are intentionally absent."
    cfg["h706_fractional_source_preflight"] = {
        "path": str(preflight_path.resolve()), "sha256": sha_file(preflight_path),
        "status": preflight["status"], "physical_admission": False, "native_steps": 0,
    }
    receipt = {
        "schema": "numi.human.current-native-010i-038-identity-materialization.v1",
        "status": "PASS_SOURCE_ONLY_IDENTITY_READY_MASS_PENDING",
        "physical_admission": False, "native_steps": 0,
        "template": {"path": str(template_path.resolve()), "sha256": sha_file(template_path)},
        "run_manifest": cfg["h706_fractional_environment_source"],
        "run_preflight": cfg["h706_fractional_source_preflight"],
        "run_source_control": run["source_control_manifest"],
        "run_source_closure": run["source_closure"],
        "updated_h706_env_keys": sorted(CHANGE_KEYS),
        "preserved_032d_carrier_packets": {
            key: {"path": app_env[key], "sha256": app_env[key.replace("_NPZ", "_SHA256")]}
            for key in OLD_032D
        },
        "v15_witness_inputs": {
            name: {"path": actual_named[name]["path"], "sha256": actual_named[name]["sha256"], "bytes": actual_named[name]["bytes"]}
            for name in V15_NAMES
        },
        "source_centroid_slots": actual_named["source_centroid_slots"],
        "projection_report": actual_named["projection_report"],
        "source_geometry_receipt": cfg["geometry_admission"],
        "mass_admission": None, "recorded_flow_bundle": None,
        "measurement_steps": 2, "accepted_mrv_steps": [0, 1, 2],
        "scope": "configuration materialization only; no mass solve, source preflight, native viewer, GPU, or physical admission",
        "config_sha256": json_digest(cfg),
    }
    return cfg, receipt

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    preflight = args.preflight or args.run_manifest.with_name("preflight.json")
    if args.output_config.exists() or args.receipt.exists():
        parser.exit(2, "refusing to overwrite identity config or receipt\n")
    try:
        cfg, receipt = materialize(args.template, args.run_manifest, preflight, args.output_config, args.output_root)
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        receipt["config"] = {"path": str(args.output_config.resolve()), "sha256": sha_file(args.output_config)}
        receipt["config_sha256"] = receipt["config"]["sha256"]
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        parser.exit(2, f"identity materialization failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"status": receipt["status"], "config": str(args.output_config), "receipt": str(args.receipt), "mass_admission": None, "physical_admission": False, "native_steps": 0}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

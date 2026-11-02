#!/usr/bin/env python3
"""Fail-closed launch adapter for the existing native Human probe.

This wraps the reusable schema/owner functions from route 711 while binding
current build 1025 and candidate-specific 433 receipts. It only writes plans;
it never starts the native executable or a GPU job.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
NATIVE = Path("/Users/numi/Work/skin-fat-native-integration-209")
BRAIN = Path("/Users/numi/numi-human-resting-integration-20261005/numi-brain")
DEFAULT_MANIFEST = HERE / "current-native-launch-1025-full-capacity-003.template.json"
BASELINE_OUTPUT_ENV = {
    "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
    "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin-native.csv",
    "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": "common-field-failure.json",
    "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass-reference.txt",
    "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV": "skin-fat-parent-mass-ledger.csv",
}
ALLOWED_ARGV_OVERRIDES = {
    "--skin-payload", "--soft-tissue-payload", "--torso-anatomy-payload",
    "--resting-anatomy-receipt", "--support-contact-payload",
    "--tendon-payload", "--joint-equality-payload", "--pectoralis-fascia-payload",
}
OBSERVER_KEYS = {
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT",
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS",
    "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP",
    "NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE",
    "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
}
TRACE_FILES = (
    "resting-coupled.csv",
    "resting-com-momentum-diagnostic.csv",
    "resting-com-support-impulses.csv",
    "resting-body-motion-partition.csv",
    "resting-support-point-motion.csv",
    "resting-com-q-integration.csv",
    "resting-com-q-index-map.csv",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def streamed_command_sha(cwd: Path, *args: str) -> str:
    proc = subprocess.Popen(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    h = hashlib.sha256()
    for block in iter(lambda: proc.stdout.read(1 << 20), b""):
        h.update(block)
    stderr = proc.stderr.read() if proc.stderr else b""
    code = proc.wait()
    if code:
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.decode(errors='replace')}")
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    if cfg.get("schema") != "numi.human.current-native-launch-1025.v1":
        raise ValueError("unsupported current-native-1025 manifest schema")
    return cfg



def verify_static_serial_cmake_v2(cfg: dict[str, Any], pins: dict[str, Any],
                                  build_path: Path, build: dict[str, Any]) -> dict[str, Any]:
    if build.get("schema") != "numi.human.native-build-source-device-manifest.v2" or build.get("capture_status") != "PASS_SERIAL_CMAKE_NATIVE_TARGET_BUILD":
        raise ValueError("build manifest lacks a successful serial CMake native target record")
    if build.get("physical_admission") is not False or build.get("native_steps") != 0:
        raise ValueError("build manifest must remain source/build-only")
    boundary = build.get("execution_boundary", {})
    if isinstance(boundary, dict):
        if any(boundary.get(k) for k in ("native_binary_executed", "metal_gpu_work_executed", "native_launch_performed")):
            raise ValueError("build manifest execution boundary is not source/build-only")
    elif (not isinstance(boundary, str) or "no native/gpu execution" not in boundary.lower() or
          "no physical or motion admission" not in boundary.lower()):
        raise ValueError("build manifest execution boundary is not source/build-only")

    build_info = build.get("build", {})
    binary = str(NATIVE / "build-native/bin/numi-human-native")
    expected_argv = ["/opt/homebrew/bin/cmake", "--build", str(NATIVE / "build-native"),
                     "--target", "numi-human-native", "--parallel", "1"]
    if build_info.get("exit_code") != 0 or build_info.get("target") != "numi-human-native":
        raise ValueError("build manifest does not attest successful numi-human-native build")
    if build_info.get("argv") != expected_argv or build_info.get("command") != shlex.join(expected_argv):
        raise ValueError("build manifest command is not the pinned serial CMake target")
    if build_info.get("binary") != binary:
        raise ValueError("native binary path differs from the selected build target")
    if not re.fullmatch(r"[0-9a-f]{64}", str(build_info.get("binary_sha256", ""))):
        raise ValueError("native binary SHA is malformed")
    if not Path(binary).is_file() or sha_file(Path(binary)) != build_info["binary_sha256"]:
        raise ValueError("native binary changed after build capture")
    if not str(build_info.get("binary_file", "")).endswith("Mach-O 64-bit executable arm64"):
        raise ValueError("native target is not an arm64 Mach-O executable")
    outputs = build_info.get("outputs_sha256")
    if not isinstance(outputs, dict) or binary not in outputs or outputs[binary] != build_info["binary_sha256"]:
        raise ValueError("build output map does not bind the native executable")
    for output_path, digest in outputs.items():
        if not Path(output_path).is_file() or sha_file(Path(output_path)) != digest:
            raise ValueError(f"native build output changed after capture: {output_path}")

    cache = build_info.get("cmake_cache", {})
    if not isinstance(cache, dict) or not Path(cache.get("path", "")).is_file() or sha_file(Path(cache["path"])) != cache.get("sha256"):
        raise ValueError("CMake cache hash is absent or stale")
    selected = cache.get("selected_values", {})
    if (selected.get("CMAKE_GENERATOR") != "Ninja" or
            selected.get("CMAKE_HOME_DIRECTORY") != str(NATIVE) or
            selected.get("NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY") != "ON" or
            selected.get("CMAKE_BUILD_TYPE") != "Release"):
        raise ValueError("CMake generator, source root, full-capacity option, or build type changed")

    parent_spec = build.get("parent_manifest", {})
    parent_path = Path(pins.get("build_parent_manifest_path", ""))
    parent_sha = pins.get("build_parent_manifest_sha256")
    if not parent_path.is_absolute() or not parent_path.is_file() or sha_file(parent_path) != parent_sha:
        raise ValueError("pinned parent build manifest is missing or stale")
    if parent_spec != {"path": str(parent_path), "sha256": parent_sha}:
        raise ValueError("build manifest parent identity differs from config pin")
    parent = read_json(parent_path)
    correction_ref = build.get("evidence_correction_parent")
    correction_path = Path(pins.get("build_evidence_correction_parent_manifest_path", ""))
    correction_sha = pins.get("build_evidence_correction_parent_manifest_sha256")
    if (not correction_path.is_absolute() or not correction_path.is_file() or
            sha_file(correction_path) != correction_sha or correction_ref !=
            {"path": str(correction_path), "sha256": correction_sha}):
        raise ValueError("build manifest correction-parent identity is absent or stale")
    correction_parent = read_json(correction_path)
    correction = build.get("evidence_correction", {})
    if (not isinstance(correction.get("scope"), str) or not correction["scope"].strip() or
            correction.get("recorded_source_and_outputs_reverified") is not True):
        raise ValueError("build manifest lacks status-mode correction scope and recheck evidence")
    producer_ref = correction.get("build_producer", {})
    producer_path = Path(producer_ref.get("path", ""))
    if (not producer_path.is_file() or
            not re.fullmatch(r"[0-9a-f]{64}", str(producer_ref.get("sha256", ""))) or
            sha_file(producer_path) != producer_ref.get("sha256")):
        raise ValueError("status-mode correction producer is absent or stale")
    # The correction is metadata-only: old 014 source, compiler outputs, and build result remain identical.
    old_source = correction_parent.get("source", {})
    source = build.get("source", {})
    if (correction_parent.get("build") != build.get("build") or
            old_source.get("head") != source.get("head") or
            old_source.get("selected_source_sha256") != source.get("selected_source_sha256") or
            old_source.get("changed_selected_source_from_parent_manifest") !=
            source.get("changed_selected_source_from_parent_manifest") or
            old_source.get("tracked_diff_from_head_sha256") != source.get("tracked_diff_from_head_sha256") or
            old_source.get("staged_diff_sha256") != source.get("staged_diff_sha256") or
            old_source.get("dirty_entries_fingerprint_sha256") != source.get("dirty_entries_fingerprint_sha256") or
            old_source.get("dirty_entry_count") != source.get("dirty_entry_count") or
            old_source.get("dirty_entry_records") != source.get("dirty_entry_records") or
            old_source.get("untracked_path_count") != source.get("untracked_path_count") or
            old_source.get("untracked_path_inventory_sha256") != source.get("untracked_path_inventory_sha256")):
        raise ValueError("status-mode correction changed native source or build evidence")
    normal_status = subprocess.check_output(
        ["git", "status", "--porcelain=v2", "--untracked-files=normal"], cwd=NATIVE)
    if (sha_bytes(normal_status) != correction.get("normal_mode_fingerprint_sha256") or
            correction.get("all_mode_fingerprint_sha256") != source.get("dirty_entries_fingerprint_sha256")):
        raise ValueError("recorded normal/all status comparison fingerprints are stale")
    source = build.get("source", {})
    parent_source = parent.get("source", {})
    if source.get("native_repo") != str(NATIVE) or source.get("head") != parent_source.get("head"):
        raise ValueError("native repository or parent revision differs from the pinned build lineage")
    selected_sources = source.get("selected_source_sha256", {})
    if not isinstance(selected_sources, dict) or not selected_sources:
        raise ValueError("build manifest lacks selected native source hashes")
    for source_path, digest in selected_sources.items():
        if not Path(source_path).is_file() or sha_file(Path(source_path)) != digest:
            raise ValueError(f"selected native source changed after build: {source_path}")

    changed = source.get("changed_selected_source_from_parent_manifest")
    if not isinstance(changed, dict) or len(changed) != 2:
        raise ValueError("build manifest must identify the exact two changed importer headers")
    expected_names = {
        "NumiHumanCurrentFractionalRecordedFlowBinding.hpp",
        "NumiHumanRecordedCarrierAllocationImport.hpp",
    }
    if {Path(p).name for p in changed} != expected_names:
        raise ValueError("build manifest changed-source set is not the exact two importer headers")
    parent_sources = parent_source.get("selected_source_sha256", {})
    for source_path, delta in changed.items():
        if set(delta) != {"previous_sha256", "current_sha256"}:
            raise ValueError("changed header delta schema is malformed")
        if (parent_sources.get(source_path) != delta["previous_sha256"] or
                selected_sources.get(source_path) != delta["current_sha256"] or
                not Path(source_path).is_file() or sha_file(Path(source_path)) != delta["current_sha256"]):
            raise ValueError(f"changed importer header lineage or live hash mismatch: {source_path}")

    tests = build.get("tests", {})
    test_ref = tests.get("source_control_manifest")
    if not isinstance(test_ref, dict) or set(test_ref) != {"path", "sha256"}:
        raise ValueError("build manifest lacks the strict C++ source-control manifest reference")
    test_manifest_path = Path(test_ref["path"])
    if not test_manifest_path.is_file() or sha_file(test_manifest_path) != test_ref["sha256"]:
        raise ValueError("C++ source-control manifest is missing or stale")
    test_manifest = read_json(test_manifest_path)
    if (test_manifest.get("schema") != "numi.human.recorded-flow-dynamic-envelope-source-control.v1" or
            test_manifest.get("status") != "PASS_SOURCE_CONTROLS_DYNAMIC_ENVELOPE_AND_LEGACY" or
            test_manifest.get("physical_admission") is not False or
            test_manifest.get("native_steps") != 0):
        raise ValueError("C++ source-control manifest does not report the required source-only pass")
    control_counts = test_manifest.get("control_counts", {})
    if (control_counts.get("dynamic_envelope_contract_passes") != 30 or
            control_counts.get("dynamic_envelope_contract_failures") != 0 or
            control_counts.get("existing_recorded_carrier_roundtrip") != "PASS_CURRENT_FRACTIONAL_RECORDED_FLOW_ARTIFACTS"):
        raise ValueError("C++ envelope or legacy roundtrip controls did not pass")
    envelope_spec = cfg.get("mass_transport_envelope_receipt", {})
    envelope_path = Path(envelope_spec.get("path", ""))
    if test_manifest.get("tested_envelope_receipt") != {
            "path": envelope_spec.get("path"), "sha256": envelope_spec.get("sha256"),
            "bytes": envelope_path.stat().st_size if envelope_path.is_file() else None}:
        raise ValueError("C++ source-control manifest tests a different envelope receipt")
    artifacts = test_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("C++ source-control manifest lacks its artifact inventory")
    control_artifacts = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("C++ source-control artifact row is malformed")
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha_file(path) != item["sha256"]:
            raise ValueError(f"C++ source-control artifact changed: {item.get('path')}")
        control_artifacts.setdefault(path.name, set()).add(item["sha256"])
    for source_path in changed:
        name = Path(source_path).name
        if changed[source_path]["current_sha256"] not in control_artifacts.get(name, set()):
            raise ValueError(f"C++ source controls do not include the changed header: {name}")

    toolchain = build.get("toolchain", {})
    if not toolchain.get("clang_version") or not toolchain.get("cmake_version") or not toolchain.get("ninja_version"):
        raise ValueError("build manifest lacks its compiler/toolchain versions")
    for key in ("compile_commands_json", "app_object_dependencies"):
        item = toolchain.get(key)
        if not isinstance(item, dict) or not Path(item.get("path", "")).is_file():
            raise ValueError(f"build manifest lacks {key}")
        path = Path(item["path"])
        if ("bytes" in item and path.stat().st_size != item["bytes"]) or sha_file(path) != item.get("sha256"):
            raise ValueError(f"build toolchain artifact is stale: {key}")
    compile_entry = toolchain["compile_commands_json"].get("entry", {})
    if str(NATIVE / "build-native") not in str(compile_entry.get("directory", "")):
        raise ValueError("compile command database does not target the selected native build")
    ninja = str(selected.get("CMAKE_MAKE_PROGRAM", ""))
    if not Path(ninja).is_file():
        raise ValueError("pinned CMake generator executable is missing")
    ninja_commands = subprocess.check_output([ninja, "-C", str(NATIVE / "build-native"),
                                              "-t", "commands", "numi-human-native"])
    if (sha_bytes(ninja_commands) != toolchain.get("target_command_sequence_sha256") or
            len(ninja_commands.splitlines()) != toolchain.get("target_command_count") or
            b"NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1" not in ninja_commands):
        raise ValueError("generated native target command sequence differs from build manifest or capacity profile")

    native_repo = Path(source.get("native_repo", ""))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=native_repo).decode().strip()
    status_mode = source.get("dirty_status_mode")
    status_argv = source.get("dirty_status_argv")
    expected_status_argv = ["git", "status", "--porcelain=v2", "--untracked-files=all"]
    if status_mode != "all" or status_argv != expected_status_argv:
        raise ValueError("build manifest must declare the exact all-path porcelain-v2 status command")
    status = subprocess.check_output(status_argv, cwd=native_repo)
    status_rows = status.decode("utf-8", "replace").splitlines()
    if (head != source.get("head") or sha_bytes(status) != source.get("dirty_entries_fingerprint_sha256") or
            len(status_rows) != source.get("dirty_entry_count") or
            status_rows != source.get("dirty_entry_records")):
        raise ValueError("native HEAD or dirty worktree differs from the build manifest")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=native_repo)
    untracked_count = len([row for row in untracked.split(b"\0") if row])
    if (untracked_count != source.get("untracked_path_count") or
            sha_bytes(untracked) != source.get("untracked_path_inventory_sha256")):
        raise ValueError("native untracked-path inventory differs from the build manifest")
    if streamed_command_sha(native_repo, "diff", "--binary", "--no-ext-diff", "HEAD") != source.get("tracked_diff_from_head_sha256"):
        raise ValueError("native tracked diff changed after build")
    if streamed_command_sha(native_repo, "diff", "--cached", "--binary", "--no-ext-diff") != source.get("staged_diff_sha256"):
        raise ValueError("native staged diff changed after build")

    brain = build.get("brain", {})
    brain_root = Path(brain.get("path", ""))
    brain_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=brain_root).decode().strip()
    brain_status = subprocess.check_output(["git", "status", "--porcelain=v2", "--untracked-files=normal"], cwd=brain_root)
    if brain_head != brain.get("commit") or sha_bytes(brain_status) != brain.get("dirty_status_sha256"):
        raise ValueError("Brain revision/worktree differs from compile manifest")
    for path, digest in brain.get("headers", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"Brain ABI header changed after build: {path}")
    if not brain.get("compile_include_observed"):
        raise ValueError("compile manifest did not observe the pinned Brain ABI include")

    route_path = Path(pins.get("route_module_path", ""))
    if not route_path.is_file() or sha_file(route_path) != pins.get("route_module_sha256"):
        raise ValueError("route 711 module hash changed")
    baseline_path = Path(pins.get("baseline_launch_snapshot_path", ""))
    source_path = Path(pins.get("baseline_launch_source_path", ""))
    if sha_file(baseline_path) != pins.get("baseline_launch_sha256") or sha_file(source_path) != pins.get("baseline_launch_sha256"):
        raise ValueError("frozen 016 baseline launch changed")
    baseline = read_json(baseline_path)
    if not re.fullmatch(r"[0-9a-f]{64}", str(baseline.get("binary_sha256", ""))):
        raise ValueError("016 launch snapshot lacks its historical executable hash")
    if baseline.get("argv", [None])[0] != build_info.get("binary"):
        raise ValueError("016 executable path differs from current build path")
    for path, digest in cfg.get("baseline_input_pins", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"baseline scene input changed: {path}")

    device = build.get("device", {})
    if subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip() != device.get("hardware_model"):
        raise ValueError("hardware model differs from build manifest")
    if subprocess.check_output(["sw_vers", "-buildVersion"]).decode().strip() != device.get("os_build"):
        raise ValueError("macOS build differs from build manifest")
    if subprocess.check_output(["uname", "-m"]).decode().strip() != "arm64" or device.get("arm64_capable") != "1":
        raise ValueError("build device is not the pinned arm64 host")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    if int(device.get("memory_bytes", 0)) != memory:
        raise ValueError("physical memory differs from build manifest")

    normalized_correction = copy.deepcopy(build)
    for key in ("evidence_correction_parent", "evidence_correction", "correction_scope"):
        normalized_correction.pop(key, None)
    normalized_correction["captured_at_utc"] = correction_parent.get("captured_at_utc")
    normalized_source = normalized_correction.get("source", {})
    normalized_source.pop("dirty_status_mode", None)
    normalized_source.pop("dirty_status_argv", None)
    if normalized_correction != correction_parent:
        raise ValueError("corrected build manifest differs from its pinned original beyond status-capture metadata")

    return {"build": build, "baseline": baseline, "native_binary_sha256": build_info["binary_sha256"],
            "manifest_sha256": pins["build_manifest_sha256"], "route_sha256": pins["route_module_sha256"],
            "baseline_launch_sha256": pins["baseline_launch_sha256"],
            "baseline_binary_sha256": baseline["binary_sha256"]}


def verify_static(cfg: dict[str, Any]) -> dict[str, Any]:
    pins = cfg.get("static_pins")
    if not isinstance(pins, dict):
        raise ValueError("static_pins missing")
    build_path = Path(pins.get("build_manifest_path", ""))
    if sha_file(build_path) != pins.get("build_manifest_sha256"):
        raise ValueError("build/source/device manifest pin is stale")
    build = read_json(build_path)
    if build.get("schema") == "numi.human.native-build-source-device-manifest.v2":
        return verify_static_serial_cmake_v2(cfg, pins, build_path, build)
    if build.get("schema") != "numi.human.native-build-source-device-manifest.v1" or build.get("capture_status") != "PASS_NINJA_TARGET_CURRENT":
        raise ValueError("build manifest lacks a successful current Ninja target record")

def verify_static(cfg: dict[str, Any]) -> dict[str, Any]:
    pins = cfg.get("static_pins")
    if not isinstance(pins, dict):
        raise ValueError("static_pins missing")
    build_path = Path(pins.get("build_manifest_path", ""))
    if sha_file(build_path) != pins.get("build_manifest_sha256"):
        raise ValueError("build/source/device manifest pin is stale")
    build = read_json(build_path)
    if build.get("schema") == "numi.human.native-build-source-device-manifest.v2":
        return verify_static_serial_cmake_v2(cfg, pins, build_path, build)
    if build.get("schema") != "numi.human.native-build-source-device-manifest.v1" or build.get("capture_status") != "PASS_NINJA_TARGET_CURRENT":
        raise ValueError("build manifest lacks a successful current Ninja target record")
    boundary = build.get("execution_boundary", {})
    if any(boundary.get(k) for k in ("native_binary_executed", "metal_gpu_work_executed", "native_launch_performed")):
        raise ValueError("build manifest execution boundary is not source/build-only")
    if build.get("build", {}).get("exit_code") != 0 or build.get("build", {}).get("target") != "numi-human-native":
        raise ValueError("build manifest does not attest successful numi-human-native build")
    for path, digest in build.get("build", {}).get("outputs_sha256", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"native build output changed after capture: {path}")
    native_repo = Path(build["source"]["native_repo"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=native_repo).decode().strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v2", "--untracked-files=normal"], cwd=native_repo)
    if head != build["source"]["head"] or sha_bytes(status) != build["source"]["dirty_entries_fingerprint_sha256"]:
        raise ValueError("native repository HEAD or 71-entry worktree fingerprint changed after build")
    if len(status.decode("utf-8", "replace").splitlines()) != build["source"]["dirty_entry_count"]:
        raise ValueError("native dirty worktree record count changed after build")
    if streamed_command_sha(native_repo, "diff", "--binary", "--no-ext-diff", "HEAD") != build["source"]["tracked_diff_from_head_sha256"]:
        raise ValueError("tracked native source diff changed after build")
    if streamed_command_sha(native_repo, "diff", "--cached", "--binary", "--no-ext-diff") != build["source"]["staged_diff_sha256"]:
        raise ValueError("staged native source diff changed after build")
    for path, digest in build.get("source", {}).get("selected_source_sha256", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"selected native source changed after build: {path}")
    brain = build.get("brain", {})
    brain_root = Path(brain.get("path", ""))
    brain_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=brain_root).decode().strip()
    brain_status = subprocess.check_output(["git", "status", "--porcelain=v2", "--untracked-files=normal"], cwd=brain_root)
    if brain_head != brain.get("commit") or sha_bytes(brain_status) != brain.get("dirty_status_sha256"):
        raise ValueError("Brain revision/worktree differs from compile manifest")
    for path, digest in brain.get("headers", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"Brain ABI header changed after build: {path}")
    if not brain.get("compile_include_observed"):
        raise ValueError("compile manifest did not observe the pinned Brain ABI include")
    route_path = Path(pins.get("route_module_path", ""))
    if sha_file(route_path) != pins.get("route_module_sha256"):
        raise ValueError("route 711 module hash changed")
    route_record = build.get("legacy_launch_contract", {}).get("route_711", {})
    if route_record.get("path") != str(route_path) or route_record.get("sha256") != pins.get("route_module_sha256"):
        raise ValueError("build manifest route-711 identity differs from adapter pin")
    baseline_path = Path(pins.get("baseline_launch_snapshot_path", ""))
    source_path = Path(pins.get("baseline_launch_source_path", ""))
    if sha_file(baseline_path) != pins.get("baseline_launch_sha256"):
        raise ValueError("frozen 016 baseline launch snapshot changed")
    if sha_file(source_path) != pins.get("baseline_launch_sha256"):
        raise ValueError("original 016 launch differs from frozen snapshot")
    baseline = read_json(baseline_path)
    if not re.fullmatch(r"[0-9a-f]{64}", str(baseline.get("binary_sha256", ""))):
        raise ValueError("016 launch snapshot lacks its historical executable hash")
    if baseline.get("argv", [None])[0] != build["build"]["binary"]:
        raise ValueError("016 launch executable path differs from current build path")
    for path, digest in cfg.get("baseline_input_pins", {}).items():
        if not Path(path).is_file() or sha_file(Path(path)) != digest:
            raise ValueError(f"baseline scene input changed: {path}")
    # Device metadata is live-checked so a plan cannot silently move hosts.
    if subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip() != build["device"]["hardware_model"]:
        raise ValueError("hardware model differs from build manifest")
    if subprocess.check_output(["sw_vers", "-buildVersion"]).decode().strip() != build["device"]["os_build"]:
        raise ValueError("macOS build differs from build manifest")
    return {"build": build, "baseline": baseline, "native_binary_sha256": build["build"]["binary_sha256"],
            "manifest_sha256": pins["build_manifest_sha256"], "route_sha256": pins["route_module_sha256"],
            "baseline_launch_sha256": pins["baseline_launch_sha256"],
            "baseline_binary_sha256": baseline["binary_sha256"]}


def route_module(cfg: dict[str, Any]):
    static = cfg["static_pins"]
    path = Path(static["route_module_path"])
    spec = importlib.util.spec_from_file_location("pinned_route_711", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import existing route 711 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_pair_map(route: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in ("O", "I", "D"):
        for key in (f"SURFACE_{role}_NODES_CSV", f"SURFACE_{role}_FACES_CSV",
                    f"ROLE_{role}_SOURCE_ASSET", f"ROLE_{role}_FULL86_CSV",
                    f"ROLE_{role}_CONTACT_THICKNESS_CSV"):
            suffix = "_CSV" if key.endswith("_CSV") else "_ASSET"
            result[key] = key[:-len(suffix)] + "_SHA256"
    for key in route.REQUIRED_ENV:
        if key == "DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV":
            continue
        if key.endswith("_CSV"):
            result[key] = key[:-4] + "_SHA256"
        elif key.endswith("_NODE") or key.endswith("_ELEMENTS") or key.endswith("_ASSET") or key.endswith("_RECEIPT"):
            result[key] = key + "_SHA256"
    return result


def validate_suffix_env(route: Any, cfg: dict[str, Any], pins: dict[str, str], detail_target: str) -> dict[str, str]:
    env = cfg.get("native_preflight_env")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("native_preflight_env must be a string-to-string map")
    missing = route.REQUIRED_ENV - set(env) - {"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"}
    if missing:
        raise ValueError(f"current 433 registered-volume environment missing {sorted(missing)}")
    env = dict(env)
    env["DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"] = detail_target
    for path_key, digest_key in path_pair_map(route).items():
        path = env.get(path_key)
        digest = env.get(digest_key)
        if not path or not digest:
            raise ValueError(f"registered-volume environment requires {path_key} and {digest_key}")
        if pins.get(path) != digest:
            raise ValueError(f"registered-volume path/hash pair is not pinned exactly: {path_key}")
    # Keep environment serialization simple and fail closed on non-finite or empty values.
    for key in route.REQUIRED_ENV - {"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"}:
        if not env.get(key):
            raise ValueError(f"registered-volume environment field {key} is empty")
    return env


def merged_pins(cfg: dict[str, Any]) -> dict[str, str]:
    base = cfg.get("baseline_input_pins", {})
    candidate = cfg.get("input_pins", {})
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        raise ValueError("input pin maps must be objects")
    combined = dict(base)
    for path, digest in candidate.items():
        if path in combined and combined[path] != digest:
            raise ValueError(f"candidate pin conflicts with frozen baseline pin: {path}")
        combined[path] = digest
    return combined


def safe_scene(cfg: dict[str, Any], static: dict[str, Any], route: Any, output_dir: Path, steps: int) -> tuple[list[str], dict[str, str], list[str]]:
    baseline = static["baseline"]
    argv = list(baseline["argv"])
    if argv[0] != static["build"]["build"]["binary"]:
        raise ValueError("scene executable path is not the freshly built binary")
    if len(argv) < 5 or argv[4] != "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/output":
        raise ValueError("baseline argv output slot differs from the pinned 016 launch contract")
    overrides = cfg.get("argv_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - ALLOWED_ARGV_OVERRIDES:
        raise ValueError("argv_overrides contains an unsupported physical input option")
    for flag, path in overrides.items():
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(f"argv override {flag} must be an absolute pinned input path")
        indices = [i for i, item in enumerate(argv[:-1]) if item == flag]
        if len(indices) != 1:
            raise ValueError(f"expected one {flag} option in frozen native argv")
        argv[indices[0] + 1] = path
    for index, item in enumerate(argv):
        if item == "--mechanics-only":
            argv.pop(index)
            break
    if "--mechanics-only" in argv:
        raise ValueError("mechanics-only diagnostic mode cannot produce accepted motion/MRV evidence")
    validation_env = {k: v for k, v in baseline["environment"].items() if not k.startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_") and not k.startswith("NUMI_HUMAN_SKIN_FAT_CANDIDATE033_")}
    validation_env["NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY"] = "0"
    route.validate_full_scene_mode(argv, validation_env)
    if "--persistent-metal-stand" not in argv or "--stand-root-assistance" in argv or "--stand-remove-assistance" in argv:
        raise ValueError("baseline must preserve unassisted persistent accepted-state standing")
    if "--persistent-stand-trace" in argv:
        raise ValueError("persistent stand trace is incompatible with the registered continuum transaction")
    try:
        argv[argv.index("--muscle-step-count") + 1] = str(steps)
    except (ValueError, IndexError):
        raise ValueError("pinned native argv lacks --muscle-step-count")
    argv[4] = str(output_dir)
    scene_env = dict(baseline["environment"])
    stale = [key for key in scene_env if key.startswith("NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_") or key.startswith("NUMI_HUMAN_SKIN_FAT_CANDIDATE033_")]
    for key in stale:
        scene_env.pop(key)
    # These 016 diagnostic outputs are redirected into each new run directory; no old output is overwritten.
    for key, filename in BASELINE_OUTPUT_ENV.items():
        if key in scene_env:
            scene_env[key] = str(output_dir / "diagnostics" / filename)
    return argv, scene_env, stale


def validate_claims(route: Any, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(cfg.get("geometry_admission"), dict):
        raise ValueError("candidate-specific geometry admission receipt is absent")
    if not isinstance(cfg.get("mass_admission"), dict):
        raise ValueError("candidate-specific mass admission receipt is absent")
    geometry, geo_sha = route.check_claim_receipt(cfg["geometry_admission"], route.REQUIRED_GEOMETRY_CLAIMS, "geometry admission")
    mass, mass_sha = route.check_claim_receipt(cfg["mass_admission"], route.REQUIRED_MASS_CLAIMS, "mass admission")
    if geometry.get("physical_admission") is not True:
        raise ValueError("geometry admission must explicitly pass physical_admission=true before import planning")
    return ({"receipt_path": cfg["geometry_admission"]["path"], "receipt_sha256": geo_sha,
             "claims": cfg["geometry_admission"]["claims"]},
            {"receipt_path": cfg["mass_admission"]["path"], "receipt_sha256": mass_sha,
             "claims": cfg["mass_admission"]["claims"]})


def output_paths(cfg: dict[str, Any], *, allow_existing_root: bool = False) -> tuple[Path, Path]:
    root = Path(cfg.get("output_root", ""))
    detail = Path(cfg.get("detail_map_export_target", ""))
    if not root.is_absolute() or not detail.is_absolute() or (root.exists() and not allow_existing_root):
        raise ValueError("output_root/detail target must be absolute and output_root must not already exist for source planning")
    if root not in detail.parents or (detail.exists() and not allow_existing_root):
        raise ValueError("detail-map target must be a new path beneath output_root")
    return root, detail


def stage_shell(route: Any, stage: dict[str, Any], cwd: str) -> str:
    body = route.render_shell(stage)
    return "#!/bin/sh\nset -eu\ncd " + shlex.quote(cwd) + "\n" + body.split("\n", 2)[2]


def generate_source_plan(cfg_path: Path, out_path: Path) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    static = verify_static(cfg)
    route = route_module(cfg)
    pins = merged_pins(cfg)
    route.verify_input_pins(pins)
    for flag, candidate_path in cfg.get("argv_overrides", {}).items():
        if candidate_path not in pins:
            raise ValueError(f"argv override {flag} lacks an exact input pin: {candidate_path}")
    geometry, mass = validate_claims(route, cfg)
    root, detail = output_paths(cfg)
    suffix_env = validate_suffix_env(route, cfg, pins, str(detail))
    steps = cfg.get("measurement_steps")
    if not isinstance(steps, int) or steps not in (1, 2):
        raise ValueError("first measurement is bounded to one or two accepted physical steps")
    argv, physics_env, stale_removed = safe_scene(cfg, static, route, root / "source-preflight", steps)
    source_env = route.build_stage_env(physics_env, suffix_env, source_only=True, canonical_export=True, operational_split=True)
    stage = {"name": "source_preflight_and_canonical_detail_export", "argv": argv,
             "environment": source_env, "expected_runtime": False, "expected_accepted_steps": 0,
             "canonical_detail_map_output": str(detail), "cwd": static["baseline"].get("cwd")}
    manifest_sha = json_digest(cfg)
    plan = {
        "schema": "numi.human.h706.native-launch-plan.v1", "status": "SOURCE_PREFLIGHT_READY",
        "execution_performed": False, "native_runtime_executed": False, "GPU_executed": False,
        "manifest_path": str(cfg_path.resolve()), "manifest_sha256": manifest_sha,
        "adapter_sha256": sha_file(Path(__file__).resolve()),
        "build_manifest_sha256": static["manifest_sha256"], "route_711_sha256": static["route_sha256"],
        "baseline_launch_sha256": static["baseline_launch_sha256"],
        "baseline_launch_executable_sha256": static["baseline_binary_sha256"],
        "native_binary": {"path": static["build"]["build"]["binary"], "sha256": static["native_binary_sha256"],
                           "repo_head": static["build"]["source"]["head"], "dirty_fingerprint_sha256": static["build"]["source"]["dirty_entries_fingerprint_sha256"]},
        "geometry_admission": geometry, "mass_admission": mass, "input_pins": pins,
        "physics_environment_sha256": json_digest(physics_env), "removed_stale_diagnostic_environment_keys": stale_removed,
        "baseline_argv_without_executable": argv[1:], "cwd": stage["cwd"],
        "stages": [stage],
        "motion": {"requested_accepted_step_count": steps, "first_measurement": True,
                   "accepted_geometry_steps": [0, steps], "install_and_motion_are_same_probe_invocation": True,
                   "postrun_admission_required": True,
                   "trace_files_required": list(TRACE_FILES),
                   "note": "Source-only route returns before scene installation/runtime; install and motion have no separate native entrypoint."},
        "execution_policy": {"launch_performed": False, "native_runtime_executed": False, "gpu_executed": False},
    }
    write_once(out_path, plan)
    shell_path = out_path.with_suffix(".stage-01.sh")
    if shell_path.exists():
        raise FileExistsError(f"refusing to overwrite source command: {shell_path}")
    shell_path.write_text(stage_shell(route, stage, stage["cwd"]))
    shell_path.chmod(0o755)
    return plan


def install_route_build_verifier(route: Any, cfg: dict[str, Any]) -> None:
    # route 711's recorders are reused, but its old verify_pinned_static() binds native004/source330.
    # Replace only that stale static check with this frozen 1025 build/source/device verifier.
    route.verify_pinned_static = lambda: verify_static(cfg)


def record_source(plan_path: Path, stdout_path: Path, stderr_path: Path, code: int, out_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    cfg = load_config(Path(plan["manifest_path"]))
    verify_static(cfg)
    route = route_module(cfg)
    install_route_build_verifier(route, cfg)
    result = route.record_source_preflight(plan_path, stdout_path, stderr_path, code, out_path)
    return result


def frozen_plan(plan_path: Path, cfg_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    cfg = load_config(cfg_path)
    static = verify_static(cfg)
    route = route_module(cfg)
    plan = read_json(plan_path)
    if plan.get("manifest_path") != str(cfg_path.resolve()) or plan.get("manifest_sha256") != json_digest(cfg):
        raise ValueError("frozen source plan does not bind the current manifest bytes")
    if plan.get("build_manifest_sha256") != static["manifest_sha256"] or plan.get("native_binary", {}).get("sha256") != static["native_binary_sha256"]:
        raise ValueError("frozen plan native build identity is stale")
    route.verify_input_pins(plan.get("input_pins", {}))
    validate_claims(route, cfg)
    if plan.get("status") not in ("SOURCE_PREFLIGHT_READY", "MOTION_LAUNCH_READY_AFTER_SOURCE_PREFLIGHT"):
        raise ValueError("plan status is not eligible for this stage")
    return plan, cfg, route


def generate_measurement_plan(cfg_path: Path, source_plan_path: Path, source_receipt_path: Path,
                              out_path: Path, steps: int) -> dict[str, Any]:
    if steps not in (1, 2):
        raise ValueError("measurement launch is limited to one or two accepted steps")
    plan, cfg, route = frozen_plan(source_plan_path, cfg_path)
    receipt_sha = sha_file(source_receipt_path)
    receipt = read_json(source_receipt_path)
    if receipt.get("status") != "PASS_SOURCE_ONLY_NO_RUNTIME" or receipt.get("execution_performed") is not True:
        raise ValueError("source-only result has not passed the existing route recorder")
    if receipt.get("input_manifest_sha256") != plan["manifest_sha256"]:
        raise ValueError("source-only receipt belongs to another candidate manifest")
    detail_path = receipt.get("canonical_detail_map_path")
    detail_sha = receipt.get("canonical_detail_map_sha256")
    if detail_path != cfg.get("detail_map_export_target") or sha_file(Path(detail_path)) != detail_sha:
        raise ValueError("canonical detail map is stale or mismatched")
    static = verify_static(cfg)
    root, detail_target = output_paths(cfg, allow_existing_root=True)
    if not Path(detail_path).is_file() or Path(detail_path) != detail_target:
        raise ValueError("source-preflight output root exists without the exact canonical detail-map target")
    base_argv = list(plan["stages"][0]["argv"])
    base_argv[4] = str(root / "install-motion")
    try:
        base_argv[base_argv.index("--muscle-step-count") + 1] = str(steps)
    except (ValueError, IndexError):
        raise ValueError("frozen source launch argv lacks a step count")
    if "--mechanics-only" in base_argv or "--persistent-stand-trace" in base_argv:
        raise ValueError("live registered continuum route cannot use mechanics-only or persistent trace mode")
    suffix = dict(cfg["native_preflight_env"])
    suffix["DETAIL_TO_BOUNDARY_CSV"] = detail_path
    suffix["DETAIL_TO_BOUNDARY_SHA256"] = detail_sha
    suffix.pop("DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV", None)
    base_env, _, stale_removed = safe_scene(cfg, static, route, root / "install-motion", steps)
    scene_env = dict(read_json(Path(cfg["static_pins"]["baseline_launch_snapshot_path"]))["environment"])
    for key in stale_removed:
        scene_env.pop(key, None)
    for key, filename in BASELINE_OUTPUT_ENV.items():
        if key in scene_env:
            scene_env[key] = str(root / "install-motion" / "diagnostics" / filename)
    motion_env = route.build_stage_env(scene_env, suffix, source_only=False, canonical_export=False, operational_split=False)
    # Add observers only; the native physics and Brain configuration remain the frozen baseline.
    motion_env.update({
        "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "1",
        "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP": str(steps),
        "NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE": "0",
        "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS": f"0,{steps}",
    })
    route.validate_full_scene_mode(base_argv[1:], motion_env)
    if "--mechanics-only" in base_argv or "--persistent-stand-trace" in base_argv:
        raise ValueError("motion stage must use the live viewer path without the incompatible persistent trace")
    stage = {"name": "registered_volume_install_and_bounded_accepted_state_motion", "argv": base_argv,
             "environment": motion_env, "expected_runtime": True, "expected_accepted_steps": steps,
             "accepted_geometry_steps": [0, steps], "cwd": plan["cwd"]}
    plan["status"] = "MOTION_LAUNCH_READY_AFTER_SOURCE_PREFLIGHT"
    plan["execution_performed"] = False
    plan["native_runtime_executed"] = False
    plan["GPU_executed"] = False
    plan["source_preflight_receipt"] = receipt
    plan["source_preflight_receipt_path"] = str(source_receipt_path.resolve())
    plan["source_preflight_receipt_sha256"] = receipt_sha
    plan["stages"].append(stage)
    plan["motion"] = {"requested_accepted_step_count": steps, "first_measurement": True,
                       "accepted_geometry_steps": [0, steps], "maximum_mrv_captures": 8,
                       "install_and_motion_are_same_probe_invocation": True,
                       "postrun_admission_required": True,
                       "trace_files_required": list(TRACE_FILES),
                       "terminal_fem_audit_required": True,
                       "full_fem_endpoint_state_dump_requested": False,
                       "viewer_path_required": True,
                       "persistent_stand_trace_forbidden_with_continuum": True}
    write_once(out_path, plan)
    shell_path = out_path.with_suffix(".stage-02.sh")
    if shell_path.exists():
        raise FileExistsError(f"refusing to overwrite motion command: {shell_path}")
    shell_path.write_text(stage_shell(route, stage, stage["cwd"]))
    shell_path.chmod(0o755)
    return plan


def record_measurement(plan_path: Path, cfg_path: Path, stdout_path: Path, stderr_path: Path,
                       code: int, out_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, cfg, route = frozen_plan(plan_path, cfg_path)
    if plan.get("status") != "MOTION_LAUNCH_READY_AFTER_SOURCE_PREFLIGHT" or len(plan.get("stages", [])) != 2:
        raise ValueError("measurement plan is not the verified two-stage source/install route")
    requested = plan["motion"]["requested_accepted_step_count"]
    outdir = Path(plan["stages"][1]["argv"][4])
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    if code != 0:
        raise ValueError(f"native measurement exited with {code}")
    files = {}
    for name in TRACE_FILES:
        path = outdir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"accepted-state trace is missing or empty: {path}")
        files[str(path)] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    mrv = {}
    accepted = outdir / "accepted-geometry"
    for step in (0, requested):
        pack = accepted / f"step-{step}.mrvpack"
        receipt_path = accepted / f"step-{step}.receipt.json"
        if not pack.is_file() or not receipt_path.is_file():
            raise ValueError(f"accepted viewer-frame MRV/receipt missing for step {step}")
        receipt = read_json(receipt_path)
        if receipt.get("accepted_step") != step or receipt.get("physical_endpoint") != "accepted" or receipt.get("surface_audit_endpoint") != "passed":
            raise ValueError(f"accepted MRV receipt does not bind passed accepted endpoint {step}")
        pack_sha = sha_file(pack)
        if receipt.get("pack_file_sha256") != pack_sha:
            raise ValueError(f"accepted MRV pack hash differs from receipt at step {step}")
        mrv[str(step)] = {"pack_path": str(pack), "pack_sha256": pack_sha,
                          "receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path)}
    install_motion = out_path.with_name(out_path.stem + ".route.json")
    if install_motion.exists():
        raise FileExistsError(f"refusing to overwrite route receipt: {install_motion}")
    install_receipt = route.record_motion(plan_path, stdout_path, stderr_path, code, install_motion)
    observer = {"schema": "numi.human.current-native-measurement-observer-receipt.v1",
                "status": "PASS_BOUNDED_ACCEPTED_STATE_TRACE_AND_MRV_CHECKS",
                "plan_sha256": sha_file(plan_path), "stdout_sha256": sha_file(stdout_path),
                "stderr_sha256": sha_file(stderr_path), "requested_accepted_steps": requested,
                "accepted_trace_files": files, "accepted_geometry_frames": mrv,
                "terminal_fem_audit_verified": True,
                "full_fem_endpoint_state_dump_requested": False,
                "native_runtime_executed": True, "gpu_executed": True,
                "physical_admission_claimed": False, "long_horizon_qualification_claimed": False,
                "install_route_receipt": {"path": str(install_motion), "sha256": sha_file(install_motion),
                                           "status": install_receipt["status"]}}
    write_once(out_path, observer)
    return observer, install_receipt


def readiness(cfg_path: Path) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    blockers: list[str] = []
    static_info = None
    try:
        static_info = verify_static(cfg)
    except Exception as exc:
        blockers.append(f"static build/baseline verification: {type(exc).__name__}: {exc}")
    try:
        route = route_module(cfg)
    except Exception as exc:
        route = None
        blockers.append(f"route 711 import: {type(exc).__name__}: {exc}")
    if route is not None:
        env = cfg.get("native_preflight_env", {})
        missing = route.REQUIRED_ENV - set(env) - {"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"}
        if missing:
            blockers.append(f"current 433 native environment missing {len(missing)} required fields: {sorted(missing)}")
        if not cfg.get("input_pins"):
            blockers.append("current 433 source candidate input pins are not present")
        for label in ("geometry_admission", "mass_admission"):
            value = cfg.get(label)
            if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
                blockers.append(f"candidate-specific {label.replace('_', ' ')} receipt is absent")
    root = Path(cfg.get("output_root", ""))
    detail = Path(cfg.get("detail_map_export_target", ""))
    if root.exists():
        blockers.append(f"new output root already exists: {root}")
    if detail.exists():
        blockers.append(f"canonical detail map target already exists: {detail}")
    return {"schema": "numi.human.current-native-launch-readiness-1025.v1",
            "status": "READY_FOR_SOURCE_PLAN_ONLY" if not blockers else "WAITING_FOR_CURRENT_433_SOURCE_AND_ADMISSION_PINS",
            "blockers": blockers, "build_source_device_manifest_sha256": static_info["manifest_sha256"] if static_info else None,
            "binary_sha256": static_info["native_binary_sha256"] if static_info else None,
            "route_711_sha256": static_info["route_sha256"] if static_info else None,
            "baseline_016_launch_sha256": static_info["baseline_launch_sha256"] if static_info else None,
            "baseline_016_executable_sha256": static_info["baseline_binary_sha256"] if static_info else None,
            "baseline_scene_input_count": len(cfg.get("baseline_input_pins", {})),
            "baseline_scene_input_bytes": sum(Path(p).stat().st_size for p in cfg.get("baseline_input_pins", {}) if Path(p).is_file()),
            "native_launch_performed": False, "native_runtime_executed": False, "gpu_executed": False,
            "physical_admission_claimed": False,
            "route_contract": {"source_preflight": "route 711 source-only registered-volume import; split reference and canonical detail export",
                               "install_motion": "single existing native process installs volume then steps; no install-only entrypoint",
                               "first_measurement_steps": cfg.get("measurement_steps"),
                               "accepted_geometry_frames": "initial state and terminal step only, maximum eight frames",
                               "persistent_stand_trace": "forbidden with continuum transaction; COM/body/Q/FEM observers used"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("readiness"); p.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); p.add_argument("--out", required=True)
    p = sub.add_parser("plan-source"); p.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); p.add_argument("--out", required=True)
    p = sub.add_parser("record-source"); p.add_argument("--plan", required=True); p.add_argument("--stdout", required=True); p.add_argument("--stderr", required=True); p.add_argument("--exit-code", type=int, required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("plan-measurement"); p.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); p.add_argument("--source-plan", required=True); p.add_argument("--source-receipt", required=True); p.add_argument("--steps", type=int, choices=(1, 2), default=2); p.add_argument("--out", required=True)
    p = sub.add_parser("record-measurement"); p.add_argument("--plan", required=True); p.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); p.add_argument("--stdout", required=True); p.add_argument("--stderr", required=True); p.add_argument("--exit-code", type=int, required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        if a.command == "readiness":
            report = readiness(Path(a.manifest)); write_once(Path(a.out), report)
            print(json.dumps({"status": report["status"], "report": a.out, "blockers": len(report["blockers"]), "launch_performed": False}, sort_keys=True))
            return 0
        if a.command == "plan-source":
            plan = generate_source_plan(Path(a.manifest), Path(a.out))
            print(json.dumps({"status": plan["status"], "plan": a.out, "command": str(Path(a.out).with_suffix(".stage-01.sh")), "launch_performed": False}, sort_keys=True))
            return 0
        if a.command == "record-source":
            result = record_source(Path(a.plan), Path(a.stdout), Path(a.stderr), a.exit_code, Path(a.out))
            print(json.dumps({"status": result["status"], "receipt": a.out, "launch_performed": True}, sort_keys=True))
            return 0
        if a.command == "plan-measurement":
            plan = generate_measurement_plan(Path(a.manifest), Path(a.source_plan), Path(a.source_receipt), Path(a.out), a.steps)
            print(json.dumps({"status": plan["status"], "plan": a.out, "command": str(Path(a.out).with_suffix(".stage-02.sh")), "launch_performed": False}, sort_keys=True))
            return 0
        if a.command == "record-measurement":
            observer, route_receipt = record_measurement(Path(a.plan), Path(a.manifest), Path(a.stdout), Path(a.stderr), a.exit_code, Path(a.out))
            print(json.dumps({"status": observer["status"], "receipt": a.out, "route_receipt": observer["install_route_receipt"]["path"], "launch_performed": True}, sort_keys=True))
            return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

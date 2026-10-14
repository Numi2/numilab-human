#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, sys, time, traceback
from pathlib import Path

REPO = Path("/Users/n/numi-human-alltarget-handoff-20261014")
sys.path.insert(0, str(REPO / "src"))
from numilab_human import resting_anatomy_verify as verifier
from numilab_human import resting_brachiocephalic_current_capture_network_preparation as preparation
from numilab_human import resting_brachiocephalic_current_capture_network_runner as runner

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
SOURCE_DIR = ROOT / "current-capture-network-gate-028-current-owner-baseline-runner-preflight-001"
REPORT_PATH = SOURCE_DIR / "network-gate-report.json"
PREFLIGHT_PATH = SOURCE_DIR / "source-and-capture-preflight.json"
OUTPUT = ROOT / "current-capture-network-gate-029-source028-all-target-screen-004"
COMPAT_PATH = ROOT / "vascular028-shared-verifier-compatibility-002/report.json"
EXPECTED_REPORT_SHA = "8435c2ccf0808236839ff6f5011c2b5c2a21f90d0dcae44623971915d1adef90"
EXPECTED_COMPAT_SHA = "1ae155b8bf0d28ae6802279920ce0117ac26d86f0e8ebaf7d0252b54fdaeec6a"
EXPECTED_CURRENT_VERIFIER_SHA = "b60f17eeda39b8a0143bf13cdcd1e89c7af3460a074eff85314a3668a2e6b899"

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def selected_code_pins():
    paths = [
        REPO / "src/numilab_human/resting_anatomy_verify.py",
        REPO / "src/numilab_human/resting_pleura_proxy.py",
        REPO / "src/numilab_human/resting_brachiocephalic_geometry_gate.py",
        REPO / "src/numilab_human/resting_brachiocephalic_3d_junction.py",
        REPO / "src/numilab_human/resting_brachiocephalic_current_capture_network_preparation.py",
        REPO / "src/numilab_human/resting_brachiocephalic_current_capture_network_runner.py",
    ]
    return {p.name: {"path": str(p.resolve()), "sha256": sha_file(p)}
            for p in paths}

started = time.perf_counter()
require(OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == {"run_all_targets.py"}, f"refusing to overwrite or reuse non-fresh output directory {OUTPUT}")
progress_path = OUTPUT / "progress.json"
manifest_path = OUTPUT / "execution-manifest.json"
revalidation_path = OUTPUT / "source-revalidation.json"
gate_path = OUTPUT / "all-target-collision-gate.json"
progress = {
    "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
    "status": "STARTED_SOURCE_REVALIDATION",
    "recorded_start_epoch_s": time.time(),
    "geometry_predicates_run": False,
    "native_or_gpu_work_performed": False,
    "anatomical_admission": False,
}
write_json(progress_path, progress)
try:
    report_raw = REPORT_PATH.read_bytes()
    preflight_raw = PREFLIGHT_PATH.read_bytes()
    continuity_report = json.loads(report_raw)
    frozen_preflight = json.loads(preflight_raw)
    frozen_continuity = continuity_report["network_source_continuity"]
    frozen_inventory = continuity_report["candidate_and_source_baseline_inventory"]
    require(sha_bytes(report_raw) == EXPECTED_REPORT_SHA,
            "frozen 028 report bytes changed")
    require(continuity_report.get("geometry_predicates_run") is True
            and continuity_report.get("all_target_collision_gate", {}).get("passed") is False
            and frozen_continuity.get("status")
            == "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED"
            and frozen_continuity.get("passed") is True
            and frozen_continuity.get("evaluated_link_count") == 15
            and frozen_continuity.get("requested_link_count") == 15,
            "028 report is not the complete passing 15-link continuity source")
    old_pins = continuity_report["runner_code_pins"]
    old_verifier_sha = old_pins["shared_verifier"]["sha256"]
    compat_raw = COMPAT_PATH.read_bytes()
    compat = json.loads(compat_raw)
    current_verifier = REPO / "src/numilab_human/resting_anatomy_verify.py"
    require(sha_bytes(compat_raw) == EXPECTED_COMPAT_SHA
            and compat.get("old_sha256") == old_verifier_sha
            and compat.get("current_sha256") == EXPECTED_CURRENT_VERIFIER_SHA
            and compat.get("all_existing_definitions_equal") is True
            and compat.get("all_non_definition_nodes_equal") is True
            and sha_file(current_verifier) == EXPECTED_CURRENT_VERIFIER_SHA,
            "028 shared-verifier compatibility proof does not match current module")
    bundle_path = Path(frozen_preflight["execution_runner"]["capture_input_bundle_path"])
    bundle_raw = bundle_path.read_bytes()
    bundle = json.loads(bundle_raw)
    require(sha_bytes(bundle_raw)
            == frozen_preflight["execution_runner"]["capture_input_bundle_sha256"]
            == continuity_report["capture_input_bundle_sha256"],
            "frozen 028 capture input bundle changed")
    for key in ("capture_pack_path", "capture_receipt_path", "anatomy_receipt_path"):
        require(sha_file(bundle[key]) == bundle[key.replace("_path", "_sha256")],
                f"current input bytes changed: {bundle[key]}")
    require(bundle["anatomy_payload_sha256"]
            == frozen_preflight["actual_accepted_capture"][
                "common_field_source_anatomy_payload_sha256"],
            "input bundle anatomy payload differs from frozen capture receipt")
    progress.update({
        "status": "LOADING_CURRENT_CAPTURE_AND_REBUILDING_SOURCE_INVENTORY",
        "frozen_028_report_sha256": EXPECTED_REPORT_SHA,
        "frozen_028_preflight_sha256": sha_bytes(preflight_raw),
        "frozen_capture_bundle_sha256": sha_bytes(bundle_raw),
        "historical_shared_verifier_sha256": old_verifier_sha,
        "current_shared_verifier_sha256": EXPECTED_CURRENT_VERIFIER_SHA,
        "shared_verifier_compatibility_report_sha256": EXPECTED_COMPAT_SHA,
        "geometry_predicates_run": False,
    })
    write_json(progress_path, progress)

    live_preflight, runtime = preparation.build_preflight(
        bundle, return_runtime_inputs=True)
    live_preflight["capture_source_slot_identity_preflight"] = (
        runner._source_binding_preflight(runtime))
    require(live_preflight.get("status", "").startswith("SOURCE0_SEAMOWNER_")
            and live_preflight.get("all_target_geometry_predicates_run") is False
            and live_preflight.get("native_or_gpu_work_performed") is False,
            "current source/capture preparation is not ready without predicates")
    old_capture = frozen_preflight["actual_accepted_capture"]
    live_capture = live_preflight["actual_accepted_capture"]
    for key in ("accepted_step", "accepted_time_s", "capture_receipt_sha256",
                "common_field_source_anatomy_payload_sha256",
                "explicit_merged_capture_input_bundle"):
        require(live_capture.get(key) == old_capture.get(key),
                f"actual accepted capture identity differs from 028 for {key}")
    for key in ("network_topology", "capture_source_slot_identity_preflight",
                "all_nearby_aabb_targets_by_network_row",
                "aabb_target_count_by_network_row", "branch_identity_crosscheck",
                "source_and_parent_motion_ownership"):
        require(live_preflight.get(key) == frozen_preflight.get(key),
                f"source/capture preflight field differs from frozen 028: {key}")

    candidates, baselines, link_inputs, live_inventory = runner._bind_runtime_inputs(runtime)
    require(len(candidates) == 17 and len(baselines) == 17 and len(link_inputs) == 15,
            "current runtime does not bind exactly 17 surfaces and 15 links")
    for key in ("candidate_surfaces", "independent_source_baselines",
                "candidate_baseline_gid_identity_by_typed_row",
                "native_source_slot_identity", "source_face_ancestry_by_typed_row",
                "source_face_arrays_bound_by_exact_ordered_abi5_corner_ancestry",
                "candidate_gid_values_were_read_from_actual_capture_pack",
                "baseline_coordinates_are_independently_cpu_reconstructed",
                "baseline_gid_values_are_transported_current_nha_source_slots"):
        require(live_inventory.get(key) == frozen_inventory.get(key),
                f"live network source inventory differs from frozen 028: {key}")
    surfaces = dict(runtime["context"]["surfaces"])
    surfaces.update(candidates)
    require(set(candidates) == set((51011, sid) for sid in preparation.NETWORK_STABLE_IDS),
            "current candidate surface identities do not match the declared network")
    live_preflight_path = OUTPUT / "source-revalidation.json"
    write_json(live_preflight_path, {
        "schema": live_preflight.get("schema"),
        "status": live_preflight["status"],
        "source_repository": preparation._git_state(),
        "actual_accepted_capture": live_capture,
        "current_package_code_sha256": live_preflight.get("current_package_code_sha256"),
        "current_verifier": live_preflight.get("current_verifier"),
        "source_inventory": live_inventory,
        "aabb_target_count_by_network_row": live_preflight[
            "aabb_target_count_by_network_row"],
        "all_target_geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    })
    manifest = {
        "schema": "numi.human.brachiocephalic.current-network-all-target-execution.v1",
        "execution_mode": "all-target-only using retained 028 continuity result; no continuity predicates rerun",
        "execution_commit": "f7f72ca3978886e5162e42f41e1f53527c57f1c2",
        "launcher_path": str(Path(__file__).resolve()),
        "launcher_sha256": sha_file(Path(__file__).resolve()),
        "execution_repository": preparation._git_state(),
        "execution_code_pins": selected_code_pins(),
        "frozen_continuity_report_path": str(REPORT_PATH.resolve()),
        "frozen_continuity_report_sha256": EXPECTED_REPORT_SHA,
        "frozen_source_preflight_path": str(PREFLIGHT_PATH.resolve()),
        "frozen_source_preflight_sha256": sha_bytes(preflight_raw),
        "frozen_capture_input_bundle_path": str(bundle_path.resolve()),
        "frozen_capture_input_bundle_sha256": sha_bytes(bundle_raw),
        "frozen_028_shared_verifier_sha256": old_verifier_sha,
        "current_shared_verifier_sha256": EXPECTED_CURRENT_VERIFIER_SHA,
        "shared_verifier_compatibility_report_path": str(COMPAT_PATH.resolve()),
        "shared_verifier_compatibility_report_sha256": EXPECTED_COMPAT_SHA,
        "current_source_revalidation_path": str(live_preflight_path.resolve()),
        "current_source_revalidation_sha256": sha_file(live_preflight_path),
        "all_target_screen_path": str(gate_path.resolve()),
        "surface_cache_policy": (
            "Current 17 network candidate PreparedSurface objects are created once, reused by "
            "the central all-target verifier, and reused for non-edge network pairs; nonnetwork "
            "targets are prepared lazily once by the central verifier."),
        "stage_order": ["revalidate current source/capture arrays against frozen 028",
                        "full network-to-nonnetwork target screen",
                        "all non-edge network pair screen"],
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }
    write_json(manifest_path, manifest)

    phase = {"name": "source_handoff_validation", "oracle_calls": 0}
    progress.update({
        "status": "READY_TO_VALIDATE_FROZEN_028_NETWORK_HANDOFF",
        "execution_manifest_path": str(manifest_path.resolve()),
        "execution_manifest_sha256": sha_file(manifest_path),
        "source_revalidation_path": str(live_preflight_path.resolve()),
        "source_revalidation_sha256": sha_file(live_preflight_path),
        "network_candidate_surface_count": len(candidates),
        "network_link_count_reused_from_028": 15,
        "geometry_predicates_run": False,
    })
    write_json(progress_path, progress)

    original_audit = verifier.audit_surface_pair
    def monitored_audit_surface_pair(first, second, *args, **kwargs):
        result = original_audit(first, second, *args, **kwargs)
        phase["oracle_calls"] += 1
        first_key = getattr(first, "key", None)
        second_key = getattr(second, "key", None)
        summary = {
            "stage": phase["name"],
            "oracle_call_index": phase["oracle_calls"],
            "identities": [list(first_key) if first_key is not None else None,
                           list(second_key) if second_key is not None else None],
            "passed": result.get("passed") is True,
            "triangle_pair_count": result.get("triangle_pair_count"),
            "unexpected_triangle_pair_count": result.get("unexpected_triangle_pair_count"),
            "failure_stage": result.get("failed_stage"),
        }
        progress.update({
            "status": "RUNNING_ALL_TARGET_GEOMETRY_PREDICATES",
            "last_completed_exact_pair": summary,
            "oracle_pair_count": phase["oracle_calls"],
            "geometry_predicates_run": True,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        })
        if result.get("passed") is not True:
            progress["first_failed_exact_pair_witness"] = result
        write_json(progress_path, progress)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return result

    def stage_callback(name):
        phase["name"] = name
        progress.update({
            "status": name,
            "geometry_predicates_run": True,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
            "oracle_pair_count": phase["oracle_calls"],
        })
        write_json(progress_path, progress)
        print(name, flush=True)

    verifier.audit_surface_pair = monitored_audit_surface_pair
    try:
        result = preparation.run_all_target_network_screen(
            {**runtime["context"], "surfaces": surfaces},
            runtime["edge_evidence"], runtime["separate_evidence"],
            verified_network_continuity=frozen_continuity,
            verified_network_source_inventory=frozen_inventory,
            progress_callback=stage_callback)
    finally:
        verifier.audit_surface_pair = original_audit

    result["execution_manifest_path"] = str(manifest_path.resolve())
    result["execution_manifest_sha256"] = sha_file(manifest_path)
    result["source_revalidation_path"] = str(live_preflight_path.resolve())
    result["source_revalidation_sha256"] = sha_file(live_preflight_path)
    result["reused_continuity_report_path"] = str(REPORT_PATH.resolve())
    result["reused_continuity_report_sha256"] = EXPECTED_REPORT_SHA
    result["new_continuity_predicates_run"] = False
    result["native_or_gpu_work_performed"] = False
    result["anatomical_admission"] = False
    write_json(gate_path, result)
    progress.update({
        "status": result.get("status"),
        "all_target_gate_path": str(gate_path.resolve()),
        "all_target_gate_sha256": sha_file(gate_path),
        "oracle_pair_count": phase["oracle_calls"],
        "failed_stage": result.get("failed_stage"),
        "passed": result.get("passed") is True,
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_predicates_run": True,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    })
    if result.get("passed") is not True:
        progress["first_failure"] = result
    write_json(progress_path, progress)
    print(json.dumps({
        "status": result.get("status"),
        "passed": result.get("passed") is True,
        "failed_stage": result.get("failed_stage"),
        "gate_path": str(gate_path.resolve()),
        "gate_sha256": sha_file(gate_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha_file(manifest_path),
        "elapsed_seconds": progress["elapsed_seconds"],
    }, sort_keys=True), flush=True)
except Exception as exc:
    failure = {
        "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
        "status": "ABORTED_OR_REJECTED_WITH_RETAINED_ERROR",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_predicates_run": progress.get("geometry_predicates_run", False),
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }
    write_json(OUTPUT / "all-target-error.json", failure)
    progress.update(failure)
    write_json(progress_path, progress)
    print(json.dumps({"status": failure["status"],
                      "error_type": failure["error_type"],
                      "error": failure["error"],
                      "error_path": str((OUTPUT / "all-target-error.json").resolve())},
                     sort_keys=True), flush=True)
    raise

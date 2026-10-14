"""Frozen source-bound execution wrapper for an explicit current vascular capture.

The caller supplies the exact invoked anatomy receipt, accepted pack, and capture
receipt. Source preflight is written before predicates. Fifteen 3D source-link
continuity checks run fail-first; only a full pass proceeds to the current
shared verifier's all-target collision screen. No native/GPU run is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as verifier
from . import resting_brachiocephalic_3d_junction as network_owner
from . import resting_brachiocephalic_current_capture_network_preparation as preparation
from . import resting_brachiocephalic_geometry_gate as collision_owner
from . import resting_brachiocephalic_source007 as retained_owner

REPO = Path(__file__).resolve().parents[2]
RUNNER_SCHEMA = "numi.human.current-capture-vascular-network-execution.v1"
NETWORK_KEYS = tuple(sorted({
    key for edge in network_owner.declared_network_links() for key in edge
}))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("current-capture vascular network execution: " + message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _native_nha_gid_base_offset(context: dict) -> dict:
    """Derive the NHA global GID base from this capture's recorded skin range."""
    receipt = context.get("capture_receipt")
    _require(isinstance(receipt, dict),
             "actual capture context lacks its validated capture receipt")
    skin = receipt.get("skin_source_mapping")
    _require(isinstance(skin, dict),
             "actual capture receipt lacks the native skin source range")
    first = skin.get("pack_first_vertex")
    count = skin.get("source_vertex_count")
    total = receipt.get("vertex_count")
    _require(type(first) is int and first >= 0
             and type(count) is int and count > 0
             and type(total) is int and total > 0,
             "actual capture receipt has invalid skin/anatomy vertex ranges")
    base = first + count
    _require(base <= total,
             "native skin range ends beyond the actual captured vertex buffer")
    return {
        "native_nha_gid_base_offset": base,
        "skin_pack_first_vertex": first,
        "skin_source_vertex_count": count,
        "captured_vertex_count": total,
        "derivation": "skin pack_first_vertex + source_vertex_count",
        "identity_contract": (
            "captured GID = native NHA base offset + ABI5 record firstVertex "
            "+ source-local vertex index"),
        "bound_to_validated_actual_capture_receipt": True,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _runtime_code_pins() -> dict:
    paths = {
        "runner": Path(__file__).resolve(),
        "network_preparation": Path(preparation.__file__).resolve(),
        "network_3d_owner": Path(network_owner.__file__).resolve(),
        "collision_owner": Path(collision_owner.__file__).resolve(),
        "shared_verifier": Path(verifier.__file__).resolve(),
        "abi5_codec": Path(preparation.capture_gate.nhanat_codec.__file__).resolve(),
    }
    return {name: {
        "path": str(path),
        "sha256": _sha_file(path),
    } for name, path in paths.items()}


def _prepared_network_surfaces(surface_map: dict, label: str) -> tuple[dict, dict]:
    _require(isinstance(surface_map, dict)
             and set(NETWORK_KEYS).issubset(surface_map),
             f"{label} does not contain every typed network row")
    prepared = {}
    records = {}
    for key in NETWORK_KEYS:
        value = surface_map[key]
        if isinstance(value, verifier.PreparedSurface):
            _require(value.key == key,
                     f"{label} prepared surface key differs from {key}")
            surface = value
        else:
            _require(isinstance(value, (tuple, list)) and len(value) >= 4,
                     f"{label} surface {key} lacks positions, ABI5 faces, normals slot, or captured GIDs")
            positions = np.asarray(value[0])
            faces = np.asarray(value[1])
            gids = np.asarray(value[3])
            _require(positions.dtype == np.dtype("<f4")
                     and positions.ndim == 2 and positions.shape[1] == 3
                     and np.isfinite(positions).all(),
                     f"{label} position array is not actual finite Float32 XYZ for {key}")
            _require(faces.ndim == 2 and faces.shape[1] == 3
                     and faces.dtype.kind in "iu"
                     and len(faces) > 0
                     and int(faces.min()) >= 0
                     and int(faces.max()) < len(positions),
                     f"{label} face array is not valid local ABI5/capture topology for {key}")
            _require(gids.shape == (len(positions),)
                     and gids.dtype.kind in "iu"
                     and len(np.unique(gids)) == len(gids),
                     f"{label} source IDs are not unique actual global capture vertex IDs for {key}")
            normals = value[2]
            surface = verifier.prepare_surface(
                key, positions, faces, normals=normals, source_vertex_ids=gids)
        prepared[key] = surface
        records[f"{key[0]}:{key[1]}"] = {
            "typed_identity": list(key),
            "vertex_count": len(surface.positions),
            "face_count": len(surface.faces),
            "positions_f32_sha256": _sha_bytes(
                np.ascontiguousarray(surface.positions, dtype="<f4").tobytes()),
            "faces_u32_sha256": _sha_bytes(
                np.ascontiguousarray(surface.faces, dtype="<u4").tobytes()),
            "captured_global_vertex_ids_u64_sha256": _sha_bytes(
                np.ascontiguousarray(surface.source_vertex_ids, dtype="<u8").tobytes()),
            "captured_global_vertex_id_min": int(surface.source_vertex_ids.min()),
            "captured_global_vertex_id_max": int(surface.source_vertex_ids.max()),
            "face_index_space": "actual pack local indices bound to ABI5 row face-array bytes",
        }
    return prepared, records


def _bind_runtime_inputs(runtime: dict) -> tuple[dict, dict, tuple[dict, ...], dict]:
    candidates, candidate_records = _prepared_network_surfaces(
        runtime["candidate_surfaces"], "actual captured candidate")
    baselines, baseline_records = _prepared_network_surfaces(
        runtime["baseline_surfaces"], "independently reconstructed source baseline")
    source_face_ancestry = {}
    source_slot_identity = _native_nha_gid_base_offset(runtime.get("context", {}))
    native_nha_gid_base_offset = source_slot_identity["native_nha_gid_base_offset"]
    for key in NETWORK_KEYS:
        candidate, baseline = candidates[key], baselines[key]
        _require(candidate.faces.tobytes() == baseline.faces.tobytes(),
                 f"candidate/source baseline face order or ABI5 ancestry differs for {key}")
        _require(candidate.source_vertex_ids.tobytes()
                 == baseline.source_vertex_ids.tobytes(),
                 f"candidate/source baseline global GID ancestry differs for {key}")
        row = runtime["source_rows"].get(key[1])
        _require(isinstance(row, dict),
                 f"decoded ABI5 source row is missing for {key}")
        _vertex_map, ancestry = network_owner._source_face_capture_ancestry(
            row, candidate, key[1], "actual-capture",
            native_nha_gid_base_offset=native_nha_gid_base_offset)
        source_face_ancestry[f"{key[0]}:{key[1]}"] = {
            **ancestry,
            "source_local_vertex_to_capture_global_gid_sha256": (
                ancestry["source_local_vertex_to_actual_global_gid_sha256"]),
            "unused_source_vertex_slots_use_uint64_max_sentinel": True,
        }
    link_inputs = network_owner.prepare_network_link_audit_inputs(
        runtime["edge_evidence"], runtime["separate_evidence"],
        source_rows=runtime["source_rows"], source_map=runtime["source_map"],
        candidate_surfaces=candidates, baseline_surfaces=baselines)
    normalized_inputs = []
    for item in link_inputs:
        edge = (item["bridge_key"], item["target_key"])
        normalized_inputs.append({
            **item,
            "bridge_surface": candidates[edge[0]],
            "target_surface": candidates[edge[1]],
            "baseline_bridge_surface": baselines[edge[0]],
            "baseline_target_surface": baselines[edge[1]],
        })
    _require(len(normalized_inputs) == 15,
             "prepared link input count differs from the exact declared graph")
    inventory = {
        "candidate_surface_count": len(candidates),
        "independent_source_baseline_surface_count": len(baselines),
        "candidate_surfaces": candidate_records,
        "independent_source_baselines": baseline_records,
        "source_face_arrays_bound_by_exact_ordered_abi5_corner_ancestry": True,
        "native_source_slot_identity": source_slot_identity,
        "source_face_ancestry_by_typed_row": source_face_ancestry,
        "candidate_and_baseline_global_gid_arrays_identical_per_row": True,
        "global_vertex_ids_were_read_from_actual_capture_pack": True,
        "repeated_candidate_self_audit_policy": (
            "The same PreparedSurface objects are passed to all edge audits and then "
            "to the all-target verifier. Their existing self_audit cache is reused."),
    }
    return candidates, baselines, tuple(normalized_inputs), inventory


def _source_binding_preflight(runtime: dict) -> dict:
    "Freeze ABI5 face/GID identity for all 17 rows before any geometry predicate."
    candidates, _baselines, link_inputs, inventory = _bind_runtime_inputs(runtime)
    _require(len(candidates) == len(NETWORK_KEYS) == 17
             and len(link_inputs) == len(network_owner.declared_network_links()) == 15,
             "source binding preflight did not cover exact 17 rows and 15 links")
    return {
        "status": "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE",
        "typed_network_surface_count": len(candidates),
        "requested_network_surface_count": 17,
        "prepared_source_link_count": len(link_inputs),
        "requested_source_link_count": 15,
        "candidate_and_independent_baseline_geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "typed_row_inventory": inventory,
    }

def execute_ordered_network_gates(runtime: dict, *, manifold3d=None,
                                 progress_callback=None) -> dict:
    """Run the 15-link continuity owner first, then all-target collision on success."""
    candidates, _baselines, link_inputs, inventory = _bind_runtime_inputs(runtime)
    if manifold3d is None:
        manifold3d = retained_owner._load_retained_manifold()
    if progress_callback is not None:
        progress_callback("STARTING_FAIL_FIRST_15_LINK_CONTINUITY")
    continuity = network_owner.audit_brachiocephalic_network_links_3d(
        verifier, manifold3d, link_inputs,
        expected_nha_gid_base_offset=(
            inventory["native_source_slot_identity"]["native_nha_gid_base_offset"]))
    report = {
        "schema": RUNNER_SCHEMA,
        "status": continuity.get("status"),
        "network_source_continuity": continuity,
        "candidate_and_source_baseline_inventory": inventory,
        "all_target_collision_gate": {
            "status": ("NOT_RUN_NETWORK_CONTINUITY_FAILED"
                       if continuity.get("passed") is not True
                       else "PENDING"),
            "passed": False,
        },
        "anatomical_admission": False,
        "native_gpu_confirmation_required": True,
    }
    if continuity.get("passed") is not True:
        report["status"] = "STOPPED_AT_FIRST_FAILED_NETWORK_LINK"
        return report
    collision_surfaces = dict(runtime["context"]["surfaces"])
    collision_surfaces.update(candidates)
    if progress_callback is not None:
        progress_callback("STARTING_ALL_TARGET_COLLISION_AFTER_15_LINK_PASS")
    collision = preparation.run_all_target_network_screen(
        {**runtime["context"], "surfaces": collision_surfaces},
        runtime["edge_evidence"], runtime["separate_evidence"])
    report["all_target_collision_gate"] = collision
    report["status"] = (
        "NETWORK_CONTINUITY_PASSED_ALL_TARGET_COLLISION_PASSED_UNADMITTED"
        if collision.get("passed") is True
        else "NETWORK_CONTINUITY_PASSED_ALL_TARGET_COLLISION_REJECTED")
    return report


def _execution_manifest(bundle_path: Path, bundle_sha256: str) -> dict:
    return {
        "schema": RUNNER_SCHEMA,
        "path": str(Path(__file__).resolve()),
        "sha256": _sha_file(Path(__file__).resolve()),
        "code_pins": _runtime_code_pins(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha256,
        "stage_order": [
            "strict ABI5/source017 ancestry binding and independent source baseline",
            "fail-first 15-link closed-solid and own-station calibre continuity",
            "only after all 15 links pass: current shared-verifier all-target collision gate",
        ],
        "geometry_predicates_run_during_preflight": False,
        "historical_source007_P6_used_for_current_geometry": False,
    }


def prepare(capture_inputs_json: Path, output_dir: Path) -> dict:
    """Freeze source/capture inputs and runtime code without running geometry."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(),
             "output directory already exists; use a fresh immutable preflight path")
    destination.mkdir(parents=True)
    bundle_bytes = bundle_path.read_bytes()
    bundle_sha = _sha_bytes(bundle_bytes)
    progress_path = destination / "progress.json"
    start = {
        "schema": RUNNER_SCHEMA,
        "status": "RUNNING_SOURCE_PREPARATION",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha,
        "runner_code_pins": _runtime_code_pins(),
        "source_repository": preparation._git_state(),
        "geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }
    _write_json(progress_path, start)
    try:
        capture_inputs = json.loads(bundle_bytes)
        preflight, runtime = preparation.build_preflight(
            capture_inputs, return_runtime_inputs=True)
        preflight["execution_runner"] = _execution_manifest(bundle_path, bundle_sha)
        preflight["capture_source_slot_identity_preflight"] = (
            _source_binding_preflight(runtime))
        preflight_path = destination / "source-and-capture-preflight.json"
        _write_json(preflight_path, preflight)
        result = {
            **start,
            "status": preflight["status"],
            "source_preflight_path": str(preflight_path.resolve()),
            "source_preflight_sha256": _sha_file(preflight_path),
            "actual_capture_identity": preflight["actual_accepted_capture"],
            "geometry_predicates_run": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(progress_path, result)
        return result
    except Exception as error:
        failure = {
            **start,
            "status": "SOURCE_PREFLIGHT_FAILED",
            "recorded_finish_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": {"type": type(error).__name__, "message": str(error)},
            "geometry_predicates_run": False,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        }
        _write_json(destination / "source-preflight-error.json", failure)
        _write_json(progress_path, failure)
        raise


def _stable_preflight(value: dict) -> dict:
    result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    result.pop("recorded_at_utc", None)
    return result


def run(capture_inputs_json: Path, output_dir: Path) -> dict:
    """Execute only from a fresh, unchanged source-and-capture preflight."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    preflight_path = destination / "source-and-capture-preflight.json"
    progress_path = destination / "progress.json"
    report_path = destination / "network-gate-report.json"
    _require(destination.is_dir() and preflight_path.is_file()
             and progress_path.is_file() and not report_path.exists()
             and not (destination / "network-gate-error.json").exists(),
             "run requires a fresh prepared output directory with no prior execution")
    progress = json.loads(progress_path.read_text())
    _require(progress.get("geometry_predicates_run") is False
             and progress.get("status", "").startswith("SOURCE0_SEAMOWNER_"),
             "prepared output has already entered or failed an execution stage")
    bundle_bytes = bundle_path.read_bytes()
    bundle_sha = _sha_bytes(bundle_bytes)
    frozen_preflight = json.loads(preflight_path.read_text())
    manifest = frozen_preflight.get("execution_runner")
    _require(isinstance(manifest, dict)
             and manifest.get("capture_input_bundle_path") == str(bundle_path)
             and manifest.get("capture_input_bundle_sha256") == bundle_sha,
             "run capture bundle differs from the frozen source preflight")
    _require(manifest.get("code_pins") == _runtime_code_pins()
             and manifest.get("sha256") == _sha_file(Path(__file__).resolve())
             and frozen_preflight.get("source_repository") == preparation._git_state(),
             "runner code or source checkout changed after preflight")
    _write_json(progress_path, {
        "schema": RUNNER_SCHEMA,
        "status": "REVALIDATING_FROZEN_SOURCE_PREFLIGHT",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha,
        "source_preflight_path": str(preflight_path.resolve()),
        "source_preflight_sha256": _sha_file(preflight_path),
        "geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    })
    geometry_predicates_started = False
    try:
        capture_inputs = json.loads(bundle_bytes)
        live_preflight, runtime = preparation.build_preflight(
            capture_inputs, return_runtime_inputs=True)
        live_preflight["execution_runner"] = manifest
        live_preflight["capture_source_slot_identity_preflight"] = (
            _source_binding_preflight(runtime))
        _require(_stable_preflight(live_preflight) == _stable_preflight(frozen_preflight),
                 "live source/capture inputs differ from the frozen preflight")
        _require(live_preflight.get("status")
                 == "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE",
                 "explicit accepted capture is not source/owner ready for the network gate")
        _require(live_preflight["capture_source_slot_identity_preflight"].get(
                     "status")
                 == "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE",
                 "actual capture lacks frozen ABI5 global slot identity proof")
        baseline_report = live_preflight["source_bound_rigid_rest_baseline"]
        _require(baseline_report.get("meaningful_calibre_baseline_ready") is True
                 and baseline_report.get("network_owner_baseline_coverage_complete") is True
                 and baseline_report.get("network_calibre_gate_passed") is False,
                 "independent per-owner source-calibre baselines are incomplete or mislabeled")
        _require(live_preflight["network_topology"].get(
                     "pending_source_collar_edges") == [],
                 "one or more of the exact 15 network links lacks frozen source ancestry")

        def mark_geometry_stage(stage):
            nonlocal geometry_predicates_started
            geometry_predicates_started = True
            _write_json(progress_path, {
                "schema": RUNNER_SCHEMA,
                "status": stage,
                "recorded_start_utc": datetime.now(UTC).isoformat(),
                "capture_input_bundle_path": str(bundle_path),
                "capture_input_bundle_sha256": bundle_sha,
                "source_preflight_path": str(preflight_path.resolve()),
                "source_preflight_sha256": _sha_file(preflight_path),
                "geometry_predicates_run": True,
                "native_or_gpu_work_performed": False,
                "anatomical_admission": False,
            })

        result = execute_ordered_network_gates(
            runtime, progress_callback=mark_geometry_stage)
        result.update({
            "recorded_finish_utc": datetime.now(UTC).isoformat(),
            "capture_input_bundle_path": str(bundle_path),
            "capture_input_bundle_sha256": bundle_sha,
            "source_preflight_path": str(preflight_path.resolve()),
            "source_preflight_sha256": _sha_file(preflight_path),
            "source_preflight_repository": frozen_preflight["source_repository"],
            "source_preflight_input_sha256": frozen_preflight["input_sha256"],
            "runner_code_pins": _runtime_code_pins(),
            "elapsed_seconds": time.perf_counter() - started,
            "geometry_predicates_run": True,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        })
        _write_json(report_path, result)
        result["report_path"] = str(report_path.resolve())
        result["report_sha256"] = _sha_file(report_path)
        _write_json(progress_path, result)
        return result
    except Exception as error:
        failure = {
            "schema": RUNNER_SCHEMA,
            "status": "ABORTED_OR_REJECTED_WITH_RETAINED_ERROR",
            "recorded_finish_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "capture_input_bundle_path": str(bundle_path),
            "capture_input_bundle_sha256": bundle_sha,
            "source_preflight_path": str(preflight_path.resolve()),
            "source_preflight_sha256": _sha_file(preflight_path),
            "error": {"type": type(error).__name__, "message": str(error)},
            "geometry_predicates_run": geometry_predicates_started,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        }
        _write_json(destination / "network-gate-error.json", failure)
        _write_json(progress_path, failure)
        raise



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-inputs-json", type=Path, required=True,
                        help="exact seven-field pack/capture/anatomy receipt bundle")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="fresh immutable output directory, reused only for --run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true",
                      help="freeze source and capture inputs without geometry")
    mode.add_argument("--run", action="store_true",
                      help="execute 15-link continuity, then all-target only on full pass")
    args = parser.parse_args(argv)
    result = (prepare(args.capture_inputs_json, args.output_dir)
              if args.prepare_only else run(args.capture_inputs_json, args.output_dir))
    print(json.dumps({
        "status": result["status"],
        "report_path": result.get(
            "report_path", result.get("source_preflight_path")),
        "report_sha256": result.get(
            "report_sha256", result.get("source_preflight_sha256")),
    }, sort_keys=True))
    return 0 if result["status"].startswith((
        "SOURCE0_SEAMOWNER_", "STOPPED_AT_FIRST_FAILED_NETWORK_LINK",
        "NETWORK_CONTINUITY_PASSED_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

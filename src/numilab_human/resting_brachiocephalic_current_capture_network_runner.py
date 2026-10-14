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
BASELINE_ADDENDUM_NAME = "baseline-collar-addendum.json"
SOURCE_BASELINE_ADDENDUM_NAME = "source-baseline-collar-addendum.json"
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
                     and int(gids.min()) >= 0
                     and len(np.unique(gids)) == len(gids),
                     f"{label} source IDs are not unique nonnegative global vertex IDs for {key}")
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
            "coordinate_source": (
                "actual accepted capture pack Float32 positions"
                if label == "actual captured candidate"
                else "independently CPU-reconstructed source positions"),
            "global_vertex_id_namespace": (
                "actual accepted capture pack global vertex IDs"
                if label == "actual captured candidate"
                else "current invoked NHA material source slots transported through "
                     "the exact candidate vertex map"),
        }
    return prepared, records


def _bind_runtime_inputs(runtime: dict) -> tuple[dict, dict, tuple[dict, ...], dict]:
    candidates, candidate_records = _prepared_network_surfaces(
        runtime["candidate_surfaces"], "actual captured candidate")
    baselines, baseline_records = _prepared_network_surfaces(
        runtime["baseline_surfaces"], "independently reconstructed source baseline")
    source_face_ancestry = {}
    source_gid_identity = {}
    source_slot_identity = _native_nha_gid_base_offset(runtime.get("context", {}))
    native_nha_gid_base_offset = source_slot_identity["native_nha_gid_base_offset"]
    for key in NETWORK_KEYS:
        candidate, baseline = candidates[key], baselines[key]
        _require(candidate.faces.tobytes() == baseline.faces.tobytes(),
                 f"candidate/source baseline face order or ABI5 ancestry differs for {key}")
        candidate_gids_u64 = np.ascontiguousarray(
            candidate.source_vertex_ids, dtype="<u8")
        baseline_gids_u64 = np.ascontiguousarray(
            baseline.source_vertex_ids, dtype="<u8")
        _require(candidate_gids_u64.shape == baseline_gids_u64.shape
                 and candidate_gids_u64.tobytes() == baseline_gids_u64.tobytes(),
                 f"candidate/source baseline global GID values differ after uint64 "
                 f"canonicalization for {key}")
        gid_sha256 = _sha_bytes(candidate_gids_u64.tobytes())
        source_gid_identity[f"{key[0]}:{key[1]}"] = {
            "typed_identity": list(key),
            "vertex_count": len(candidate_gids_u64),
            "canonical_dtype": "little-endian uint64",
            "candidate_gid_values_sha256": gid_sha256,
            "baseline_gid_values_sha256": _sha_bytes(baseline_gids_u64.tobytes()),
            "numeric_gid_values_equal": True,
        }
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
        "candidate_baseline_gid_identity_by_typed_row": source_gid_identity,
        "candidate_baseline_gid_comparison_dtype": "little-endian uint64",
        "candidate_gid_values_were_read_from_actual_capture_pack": True,
        "baseline_coordinates_are_independently_cpu_reconstructed": True,
        "baseline_gid_values_are_transported_current_nha_source_slots": True,
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

def _baseline_collar_addendum(runtime: dict, preflight_report: dict) -> dict:
    """Recompute the sole narrowly allowed baseline P6/Rat collar extension."""
    _candidates, baselines, link_inputs, inventory = _bind_runtime_inputs(runtime)
    edge = ((51011, 475), (51011, 479))
    item = next((value for value in link_inputs
                 if (value["bridge_key"], value["target_key"]) == edge), None)
    _require(item is not None,
             "baseline collar addendum link 475->479 is absent from the declared graph")
    proof = network_owner.verify_baseline_matched_collar_addendum(
        verifier,
        bridge_key=edge[0], target_key=edge[1],
        source_link_evidence=item["source_link_evidence"],
        source_rows=runtime["source_rows"], source_map=runtime["source_map"],
        independent_source_baseline_bridge_surface=baselines[edge[0]],
        independent_source_baseline_target_surface=baselines[edge[1]],
        native_nha_gid_base_offset=(
            inventory["native_source_slot_identity"]["native_nha_gid_base_offset"]),
        baseline_binding=preflight_report,
    )
    return proof


def _source_baseline_collar_addendum(runtime: dict, preflight_report: dict) -> dict:
    """Replay the frozen 023 source-baseline additions for typed link 525->469."""
    _candidates, baselines, link_inputs, inventory = _bind_runtime_inputs(runtime)
    edge = network_owner.SOURCE_BASELINE_ADDITIONS_LINK
    item = next((value for value in link_inputs
                 if (value["bridge_key"], value["target_key"]) == edge), None)
    _require(item is not None,
             "source-baseline additions link 525->469 is absent from the declared graph")
    report_path = network_owner.SOURCE_BASELINE_OBSERVATION_REPORT.resolve(strict=True)
    report_sha = network_owner.SOURCE_BASELINE_OBSERVATION_REPORT_SHA256
    _require(_sha_file(report_path) == report_sha,
             "pinned 023 source-baseline report changed before replay")
    return network_owner.verify_baseline_matched_collar_addendum(
        verifier,
        bridge_key=edge[0], target_key=edge[1],
        source_link_evidence=item["source_link_evidence"],
        source_rows=runtime["source_rows"], source_map=runtime["source_map"],
        independent_source_baseline_bridge_surface=baselines[edge[0]],
        independent_source_baseline_target_surface=baselines[edge[1]],
        native_nha_gid_base_offset=(
            inventory["native_source_slot_identity"]["native_nha_gid_base_offset"]),
        baseline_binding=preflight_report,
        declarations=network_owner.SOURCE_BASELINE_ADDITIONS,
        baseline_observation={
            "report_path": str(report_path),
            "report_sha256": report_sha,
        },
    )


def prepare_baseline_collar_addendum(capture_inputs_json: Path,
                                     output_dir: Path) -> dict:
    """Run and freeze only the baseline 475->479 exact P6/Rat extension proof."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    preflight_path = destination / "source-and-capture-preflight.json"
    progress_path = destination / "progress.json"
    addendum_path = destination / BASELINE_ADDENDUM_NAME
    _require(destination.is_dir() and preflight_path.is_file()
             and progress_path.is_file() and not addendum_path.exists(),
             "baseline addendum requires a fresh source preflight and output path")
    frozen = json.loads(preflight_path.read_text())
    progress = json.loads(progress_path.read_text())
    bundle_sha = _sha_file(bundle_path)
    manifest = frozen.get("execution_runner", {})
    _require(manifest.get("capture_input_bundle_path") == str(bundle_path)
             and manifest.get("capture_input_bundle_sha256") == bundle_sha
             and manifest.get("code_pins") == _runtime_code_pins()
             and manifest.get("sha256") == _sha_file(Path(__file__).resolve())
             and frozen.get("source_repository") == preparation._git_state()
             and progress.get("status", "").startswith("SOURCE0_SEAMOWNER_")
             and progress.get("geometry_predicates_run") is False,
             "baseline addendum source/capture/code preflight is stale or already executed")
    capture_inputs = json.loads(bundle_path.read_bytes())
    live, runtime = preparation.build_preflight(
        capture_inputs, return_runtime_inputs=True)
    live["execution_runner"] = manifest
    live["capture_source_slot_identity_preflight"] = _source_binding_preflight(runtime)
    _require(_stable_preflight(live) == _stable_preflight(frozen),
             "live source/capture inputs differ from the frozen source preflight")
    proof = _baseline_collar_addendum(runtime, live)
    report = {
        "schema": "numi.human.current-capture-baseline-collar-addendum-execution.v1",
        "status": "VERIFIED_BASELINE_ONLY_475_479_EXACT_P6_RAT_COLLAR_ADDENDUM",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha,
        "source_preflight_path": str(preflight_path.resolve()),
        "source_preflight_sha256": _sha_file(preflight_path),
        "runner_code_pins": _runtime_code_pins(),
        "candidate_surface_predicates_run": False,
        "baseline_exact_p6_predicate_run": True,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "verified_addendum": proof,
        "scope": "only source-baseline face pair 1577/391 on typed link 475->479",
    }
    _write_json(addendum_path, report)
    progress.update({
        "status": "BASELINE_COLLAR_ADDENDUM_VERIFIED",
        "baseline_addendum_path": str(addendum_path.resolve()),
        "baseline_addendum_sha256": _sha_file(addendum_path),
        "geometry_predicates_run": True,
        "candidate_surface_predicates_run": False,
        "baseline_exact_p6_predicate_run": True,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "elapsed_baseline_addendum_seconds": time.perf_counter() - started,
    })
    _write_json(progress_path, progress)
    return {**report, "path": str(addendum_path.resolve()),
            "sha256": _sha_file(addendum_path)}


def prepare_source_baseline_collar_addendum(
    capture_inputs_json: Path, output_dir: Path,
) -> dict:
    """Freeze the exact source-baseline 525->469 multi-face collar proof."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    preflight_path = destination / "source-and-capture-preflight.json"
    progress_path = destination / "progress.json"
    prior_addendum_path = destination / BASELINE_ADDENDUM_NAME
    addendum_path = destination / SOURCE_BASELINE_ADDENDUM_NAME
    _require(destination.is_dir() and preflight_path.is_file()
             and progress_path.is_file() and prior_addendum_path.is_file()
             and not addendum_path.exists(),
             "source-baseline addendum requires the frozen preflight and legacy collar proof")
    progress = json.loads(progress_path.read_text())
    prior = json.loads(prior_addendum_path.read_text())
    frozen = json.loads(preflight_path.read_text())
    bundle_sha = _sha_file(bundle_path)
    manifest = frozen.get("execution_runner", {})
    _require(progress.get("status") == "BASELINE_COLLAR_ADDENDUM_VERIFIED"
             and progress.get("baseline_addendum_sha256")
             == _sha_file(prior_addendum_path)
             and prior.get("status")
             == "VERIFIED_BASELINE_ONLY_475_479_EXACT_P6_RAT_COLLAR_ADDENDUM"
             and prior.get("capture_input_bundle_path") == str(bundle_path)
             and prior.get("capture_input_bundle_sha256") == bundle_sha
             and prior.get("source_preflight_path") == str(preflight_path.resolve())
             and prior.get("source_preflight_sha256") == _sha_file(preflight_path)
             and prior.get("runner_code_pins") == _runtime_code_pins()
             and manifest.get("capture_input_bundle_path") == str(bundle_path)
             and manifest.get("capture_input_bundle_sha256") == bundle_sha
             and manifest.get("code_pins") == _runtime_code_pins()
             and manifest.get("sha256") == _sha_file(Path(__file__).resolve())
             and frozen.get("source_repository") == preparation._git_state(),
             "source-baseline addendum preflight/legacy proof is stale or mismatched")
    capture_inputs = json.loads(bundle_path.read_bytes())
    live, runtime = preparation.build_preflight(
        capture_inputs, return_runtime_inputs=True)
    live["execution_runner"] = manifest
    live["capture_source_slot_identity_preflight"] = _source_binding_preflight(runtime)
    _require(_stable_preflight(live) == _stable_preflight(frozen),
             "live source/capture inputs differ from the frozen source preflight")
    proof = _source_baseline_collar_addendum(runtime, live)
    report = {
        "schema": "numi.human.current-capture-source-baseline-collar-addendum-execution.v1",
        "status": "VERIFIED_SOURCE_BASELINE_ONLY_525_469_DECLARED_COLLAR_ADDITIONS",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha,
        "source_preflight_path": str(preflight_path.resolve()),
        "source_preflight_sha256": _sha_file(preflight_path),
        "runner_code_pins": _runtime_code_pins(),
        "source_baseline_report_path": str(
            network_owner.SOURCE_BASELINE_OBSERVATION_REPORT.resolve()),
        "source_baseline_report_sha256": (
            network_owner.SOURCE_BASELINE_OBSERVATION_REPORT_SHA256),
        "candidate_surface_predicates_run": False,
        "baseline_exact_p6_predicate_run": True,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "verified_addendum": proof,
        "scope": "exact three declared pairs on source-baseline link 525->469",
    }
    _write_json(addendum_path, report)
    progress.update({
        "status": "SOURCE_BASELINE_COLLAR_ADDITIONS_VERIFIED",
        "source_baseline_addendum_path": str(addendum_path.resolve()),
        "source_baseline_addendum_sha256": _sha_file(addendum_path),
        "geometry_predicates_run": True,
        "candidate_surface_predicates_run": False,
        "source_baseline_exact_p6_predicate_run": True,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "elapsed_source_baseline_addendum_seconds": time.perf_counter() - started,
    })
    _write_json(progress_path, progress)
    return {**report, "path": str(addendum_path.resolve()),
            "sha256": _sha_file(addendum_path)}


def execute_ordered_network_gates(runtime: dict, *, manifold3d=None,
                                 progress_callback=None,
                                 matched_collar_addendum=None,
                                 matched_collar_addenda=None,
                                 run_all_targets=True) -> dict:
    """Run fail-first 15-link continuity; all-target is an explicit later stage."""
    candidates, _baselines, link_inputs, inventory = _bind_runtime_inputs(runtime)
    if matched_collar_addenda is None:
        collar_addenda = {}
        if matched_collar_addendum is not None:
            addendum_edge = tuple(tuple(row) for row in
                                  matched_collar_addendum.get("typed_link", []))
            collar_addenda[addendum_edge] = matched_collar_addendum
    else:
        _require(matched_collar_addendum is None
                 and isinstance(matched_collar_addenda, dict),
                 "network execution accepts either one legacy addendum or a typed addendum map")
        collar_addenda = {
            tuple(tuple(row) for row in edge): proof
            for edge, proof in matched_collar_addenda.items()
        }
    declared_edges = set(network_owner.declared_network_links())
    _require(set(collar_addenda).issubset(declared_edges),
             "network execution received an addendum for an undeclared link")
    for edge, proof in collar_addenda.items():
        _require(isinstance(proof, dict) and proof.get("passed") is True
                 and proof.get("typed_link") == [list(edge[0]), list(edge[1])],
                 "network execution addendum does not bind its exact typed link")
        edge_item = next((item for item in link_inputs
                          if item["bridge_key"] == edge[0]
                          and item["target_key"] == edge[1]), None)
        _require(edge_item is not None,
                 f"network execution is missing typed link {edge[0][1]}->{edge[1][1]}")
        edge_item["matched_collar_addendum"] = proof
    if ((51011, 475), (51011, 479)) in collar_addenda:
        inventory["baseline_matched_collar_addendum"] = collar_addenda[
            ((51011, 475), (51011, 479))]
    if ((51011, 525), (51011, 469)) in collar_addenda:
        inventory["source_baseline_matched_collar_addendum"] = collar_addenda[
            ((51011, 525), (51011, 469))]
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
        "baseline_matched_collar_addendum": matched_collar_addendum,
        "baseline_matched_collar_addenda": [
            {"typed_link": [list(edge[0]), list(edge[1])], "proof": proof}
            for edge, proof in sorted(collar_addenda.items())
        ],
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
    if not run_all_targets:
        report["status"] = (
            "NETWORK_CONTINUITY_PASSED_ALL_TARGET_PENDING_VERIFIED_JUNCTION_HANDOFF")
        report["all_target_collision_gate"] = {
            "status": "NOT_RUN_VERIFIED_JUNCTION_HANDOFF_NOT_FROZEN",
            "passed": False,
            "reason": (
                "declared partial-overlap vessel junction proofs are not yet bound "
                "into the all-target screen"),
        }
        return report
    collision_surfaces = dict(runtime["context"]["surfaces"])
    collision_surfaces.update(candidates)
    if progress_callback is not None:
        progress_callback("STARTING_ALL_TARGET_COLLISION_AFTER_15_LINK_PASS")
    collision = preparation.run_all_target_network_screen(
        {**runtime["context"], "surfaces": collision_surfaces},
        runtime["edge_evidence"], runtime["separate_evidence"],
        verified_network_continuity=continuity,
        verified_network_source_inventory=inventory,
        progress_callback=progress_callback)
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
            ("15-link fail-first continuity only; all-target deferred until a verified "
             "junction handoff is frozen"),
        ],
        "geometry_predicates_run_during_preflight": False,
        "historical_source007_P6_used_for_current_geometry": False,
        "baseline_addendum_stages": [
            "exact P6/Rat source-baseline contact proof for one bounded 475->479 pair",
            "exact P6/Rat source-baseline contact proof for three declared 525->469 pairs",
        ],
        "candidate_geometry_remains_unaudited_until_fail_first_network_run": True,
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


def run(capture_inputs_json: Path, output_dir: Path, *,
        run_all_targets: bool = False) -> dict:
    """Execute only from a fresh, unchanged source-and-capture preflight."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    preflight_path = destination / "source-and-capture-preflight.json"
    progress_path = destination / "progress.json"
    report_path = destination / "network-gate-report.json"
    addendum_path = destination / BASELINE_ADDENDUM_NAME
    source_addendum_path = destination / SOURCE_BASELINE_ADDENDUM_NAME
    _require(destination.is_dir() and preflight_path.is_file()
             and progress_path.is_file() and addendum_path.is_file()
             and source_addendum_path.is_file()
             and not report_path.exists()
             and not (destination / "network-gate-error.json").exists(),
             "run requires a fresh prepared output directory with no prior execution")
    progress = json.loads(progress_path.read_text())
    _require(progress.get("geometry_predicates_run") is True
             and progress.get("status") == "SOURCE_BASELINE_COLLAR_ADDITIONS_VERIFIED"
             and progress.get("candidate_surface_predicates_run") is False
             and progress.get("baseline_exact_p6_predicate_run") is True
             and progress.get("baseline_addendum_path") == str(addendum_path.resolve())
             and progress.get("baseline_addendum_sha256") == _sha_file(addendum_path)
             and progress.get("source_baseline_addendum_path")
             == str(source_addendum_path.resolve())
             and progress.get("source_baseline_addendum_sha256")
             == _sha_file(source_addendum_path)
             and progress.get("source_baseline_exact_p6_predicate_run") is True,
             "run requires both separately frozen baseline-only collar proofs and no candidate predicates")
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
        _require(live_preflight.get("status", "").startswith("SOURCE0_SEAMOWNER_"),
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

        frozen_addendum = json.loads(addendum_path.read_text())
        _require(frozen_addendum.get("status")
                 == "VERIFIED_BASELINE_ONLY_475_479_EXACT_P6_RAT_COLLAR_ADDENDUM"
                 and frozen_addendum.get("capture_input_bundle_path") == str(bundle_path)
                 and frozen_addendum.get("capture_input_bundle_sha256") == bundle_sha
                 and frozen_addendum.get("source_preflight_path") == str(preflight_path.resolve())
                 and frozen_addendum.get("source_preflight_sha256") == _sha_file(preflight_path)
                 and frozen_addendum.get("runner_code_pins") == _runtime_code_pins(),
                 "baseline addendum does not bind the exact preflight, bundle, and code")
        replayed_addendum = _baseline_collar_addendum(runtime, live_preflight)
        _require(_stable_preflight(replayed_addendum) == _stable_preflight(
                     frozen_addendum.get("verified_addendum", {})),
                 "frozen baseline collar addendum does not replay against current source inputs")
        frozen_source_addendum = json.loads(source_addendum_path.read_text())
        _require(frozen_source_addendum.get("status")
                 == "VERIFIED_SOURCE_BASELINE_ONLY_525_469_DECLARED_COLLAR_ADDITIONS"
                 and frozen_source_addendum.get("capture_input_bundle_path") == str(bundle_path)
                 and frozen_source_addendum.get("capture_input_bundle_sha256") == bundle_sha
                 and frozen_source_addendum.get("source_preflight_path")
                 == str(preflight_path.resolve())
                 and frozen_source_addendum.get("source_preflight_sha256")
                 == _sha_file(preflight_path)
                 and frozen_source_addendum.get("runner_code_pins")
                 == _runtime_code_pins()
                 and frozen_source_addendum.get("source_baseline_report_path")
                 == str(network_owner.SOURCE_BASELINE_OBSERVATION_REPORT.resolve())
                 and frozen_source_addendum.get("source_baseline_report_sha256")
                 == network_owner.SOURCE_BASELINE_OBSERVATION_REPORT_SHA256,
                 "source-baseline collar addendum does not bind exact preflight, bundle, and 023 report")
        replayed_source_addendum = _source_baseline_collar_addendum(runtime, live_preflight)
        _require(_stable_preflight(replayed_source_addendum) == _stable_preflight(
                     frozen_source_addendum.get("verified_addendum", {})),
                 "frozen source-baseline collar addendum does not replay against current source inputs")
        result = execute_ordered_network_gates(
            runtime, progress_callback=mark_geometry_stage,
            matched_collar_addenda={
                ((51011, 475), (51011, 479)): replayed_addendum,
                ((51011, 525), (51011, 469)): replayed_source_addendum,
            },
            run_all_targets=run_all_targets)
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
    mode.add_argument("--prepare-baseline-addendum", action="store_true",
                      help="run only the exact independent-baseline P6/Rat 475->479 collar proof")
    mode.add_argument("--prepare-source-baseline-addendum", action="store_true",
                      help="run only the pinned 023 source-baseline P6/Rat proof for three 525->469 collar additions")
    mode.add_argument("--run", action="store_true",
                      help="execute only the fail-first 15-link continuity gate")
    mode.add_argument("--run-all-targets-after-continuity", action="store_true",
                      help="execute the 15-link continuity gate, then its verified all-target handoff on full pass")
    args = parser.parse_args(argv)
    if args.prepare_only:
        result = prepare(args.capture_inputs_json, args.output_dir)
    elif args.prepare_baseline_addendum:
        result = prepare_baseline_collar_addendum(
            args.capture_inputs_json, args.output_dir)
    elif args.prepare_source_baseline_addendum:
        result = prepare_source_baseline_collar_addendum(
            args.capture_inputs_json, args.output_dir)
    elif args.run:
        result = run(args.capture_inputs_json, args.output_dir)
    else:
        result = run(args.capture_inputs_json, args.output_dir, run_all_targets=True)
    print(json.dumps({
        "status": result["status"],
        "report_path": result.get(
            "report_path", result.get("path", result.get("source_preflight_path"))),
        "report_sha256": result.get(
            "report_sha256", result.get("sha256", result.get("source_preflight_sha256"))),
    }, sort_keys=True))
    return 0 if result["status"].startswith((
        "SOURCE0_SEAMOWNER_", "STOPPED_AT_FIRST_FAILED_NETWORK_LINK",
        "NETWORK_CONTINUITY_PASSED_",
        "VERIFIED_BASELINE_ONLY_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

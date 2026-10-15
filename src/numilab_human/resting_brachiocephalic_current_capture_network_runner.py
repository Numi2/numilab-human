"""Frozen source-bound execution wrapper for an explicit current vascular capture.

The caller supplies the exact invoked anatomy receipt, accepted pack, and capture
receipt. Source preflight is written before predicates. Fifteen 3D source-link
continuity checks run fail-first; only a full pass proceeds to the current
shared verifier's all-target collision screen. No native/GPU run is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

AORTIC_PROOF_ROOT = Path(
    "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378")
AORTIC_PROOF_DIR = AORTIC_PROOF_ROOT / "aortic-6-7-source-interface-proof-004"
AORTIC_PROOF_REPORT = AORTIC_PROOF_DIR / "execution-report.json"
AORTIC_PROOF_PREFLIGHT = AORTIC_PROOF_DIR / "preflight.json"
AORTIC_DESCENDING_PROOF_DIR = (
    AORTIC_PROOF_ROOT / "aortic-7-8-source-interface-proof-004")
AORTIC_DESCENDING_PROOF_REPORT = (
    AORTIC_DESCENDING_PROOF_DIR / "execution-report.json")
AORTIC_DESCENDING_PROOF_PREFLIGHT = AORTIC_DESCENDING_PROOF_DIR / "preflight.json"
AORTIC_DESCENDING_PROOF_HANDOFF = (
    AORTIC_DESCENDING_PROOF_DIR / "handoff-certificate.json")
AORTIC_PARENT_150_DIR = (
    AORTIC_PROOF_ROOT /
    "current-capture-network-gate-150-actual012-source179-313candidate-all-target-001")
AORTIC_PARENT_150_REPORT = AORTIC_PARENT_150_DIR / "terminal-report.json"
AORTIC_PARENT_150_REPORT_SHA256 = (
    "662c1a389ccf73c53c4b4a5594ea77e0e0842937d58b8abb7d24cfdae82bae0f")
AORTIC_TARGET_OVERLAY_RUNNER = AORTIC_PROOF_ROOT / "prepare_current012_network150.py"
AORTIC_TARGET_OVERLAY_RUNNER_SHA256 = (
    "261ac70eb41f2f23e2821d406f1b388880b32cc0186f0f8f110725fd6e05f021")
AORTIC_DESCENDING_PROOF_REPORT_SHA256 = (
    "d4905ba93f1e11829ad38d076056fa434fa70d37b9269eb9f1ef231e10aefcdf")
AORTIC_DESCENDING_PROOF_PREFLIGHT_SHA256 = (
    "e00854e2883508e92ae0234353ec94e9df82649834ad7b4169ffd83b98fd8d0c")
AORTIC_DESCENDING_PROOF_HANDOFF_SHA256 = (
    "56922c77b63edcdc2ac8164f70cb47485312d558055cad95547754e183985ae5")


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
            "current_source_surface_replay": (
                network_owner.SOURCE_BASELINE_SURFACE_REPLAY_POLICY),
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
                                 verified_nonnetwork_interface_certificates=None,
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
        verified_nonnetwork_interface_certificates=(
            verified_nonnetwork_interface_certificates),
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


def _aortic_arch_certificate_from_reports(
    runtime: dict, bundle_sha256: str, report: dict, preflight: dict, *,
    evidence_report_sha256: str, source_preflight_sha256: str,
) -> dict:
    """Bind the frozen 6↔7 proof to current rows from the same accepted pack."""
    edge = collision_owner.AORTIC_ARCH_SOURCE_INTERFACE
    _require(bundle_sha256 == collision_owner.AORTIC_ARCH_INTERFACE_CAPTURE_BUNDLE_SHA256
             and preflight.get("capture_input_bundle_sha256") == bundle_sha256
             and preflight.get("parent_150_terminal_report_sha256")
             == AORTIC_PARENT_150_REPORT_SHA256,
             "aortic proof does not belong to the exact run150 capture bundle")
    _require(report.get("schema") == "numi.human.aortic-6-7-source-interface-proof.v1"
             and report.get("status") == "PASS_SOURCE_BOUND_AORTA_ARCH_INTERFACE_DIAGNOSTIC"
             and report.get("passed") is True
             and report.get("typed_pair") == [list(edge[0]), list(edge[1])]
             and report.get("preflight_sha256") == source_preflight_sha256
             and report.get("all_15_flow_links_unchanged") is True
             and report.get("source_topology_passed") is True
             and report.get("interface_gate_passed") is True
             and report.get("solid_gate_passed") is True
             and report.get("independent_source_collar_calibre_passed") is True
             and report.get("candidate_pairs_equal_frozen_source_pairs") is True
             and report.get("baseline_pairs_equal_frozen_source_pairs") is True
             and report.get("candidate_baseline_exact_pairsets_equal") is True,
             "pinned aortic report no longer carries the required source-bound passes")

    source_identity = preflight.get("source_identity")
    _require(isinstance(source_identity, dict)
             and set(source_identity) == {
                 f"{edge[0][0]}:{edge[0][1]}",
                 f"{edge[1][0]}:{edge[1][1]}"},
             "aortic preflight does not bind exactly the two typed source rows")
    source_rows = {}
    candidate_bindings = {}
    for key, expected in collision_owner.AORTIC_ARCH_SOURCE_ROWS.items():
        label = f"{key[0]}:{key[1]}"
        row = source_identity[label]
        ancestry = row.get("source_to_capture_ancestry")
        _require(row.get("typed_identity", list(key)) == list(key)
                 and row.get("source_member") == expected["source_member"]
                 and row.get("fma") == expected["fma"]
                 and row.get("name") == expected["name"]
                 and row.get("body_index") == expected["body_index"]
                 and row.get("layer") == expected["layer"]
                 and row.get("first_vertex") == expected["first_vertex"]
                 and row.get("vertex_count") == expected["vertex_count"]
                 and row.get("face_count") == expected["face_count"]
                 and row.get("source_face_u32_sha256") == expected["faces_sha256"]
                 and row.get("actual_face_u32_sha256") == expected["faces_sha256"]
                 and isinstance(ancestry, dict)
                 and ancestry.get("status") == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
                 and ancestry.get("typed_identity") == list(key)
                 and ancestry.get("source_abi5_face_array_sha256") == expected["faces_sha256"]
                 and ancestry.get("capture_faces_u32_sha256") == expected["faces_sha256"]
                 and ancestry.get("ordered_source_face_ids_are_preserved") is True
                 and ancestry.get("capture_global_vertex_ids_are_actual_pack_values") is True
                 and ancestry.get("native_source_slot_identity_is_contiguous") is True
                 and ancestry.get("native_nha_gid_base_offset") == 766836
                 and ancestry.get("observed_gid_minus_nha_global_vertex_index_unique_count") == 1
                 and ancestry.get("observed_gid_minus_nha_global_vertex_index_min") == 766836
                 and ancestry.get("observed_gid_minus_nha_global_vertex_index_max") == 766836,
                 f"aortic source row/slot ancestry changed for {key}")
        source_rows[label] = dict(expected)
        value = runtime["candidate_surfaces"].get(
            key, runtime["context"]["surfaces"].get(key))
        _require(value is not None, f"current runtime is missing aortic surface {key}")
        fingerprint = collision_owner._aortic_arch_surface_fingerprint(value, key=key)
        _require(fingerprint["positions_f32_sha256"]
                 == row.get("actual_position_f32_sha256")
                 and fingerprint["faces_u32_sha256"]
                 == row.get("actual_face_u32_sha256")
                 and fingerprint["normals_f32_sha256"]
                 == row.get("actual_normal_f32_sha256")
                 and fingerprint["source_vertex_ids_u64_sha256"]
                 == row.get("actual_gid_u64_sha256")
                 and fingerprint["vertex_count"] == expected["vertex_count"]
                 and fingerprint["face_count"] == expected["face_count"],
                 f"aortic proof arrays differ from current actual surface {key}")
        candidate_bindings[label] = fingerprint

    pairs = report.get("frozen_source_pairset")
    _require(isinstance(pairs, list)
             and len(pairs) == collision_owner.AORTIC_ARCH_INTERFACE_PAIR_COUNT
             and all(isinstance(pair, list) and len(pair) == 2
                     and all(type(face) is int and face >= 0 for face in pair)
                     for pair in pairs)
             and tuple(tuple(pair) for pair in pairs) == tuple(sorted({tuple(pair) for pair in pairs}))
             and hashlib.sha256(np.asarray(pairs, dtype="<u4").tobytes()).hexdigest()
             == collision_owner.AORTIC_ARCH_INTERFACE_PAIRSET_SHA256,
             "aortic proof pairset is not the frozen exact 36-pair set")
    interface_match = report.get("interface_gate_pairset_matches_frozen")
    _require(interface_match == {
                 "candidate": True, "independent_source_baseline": True},
             "candidate and independent baseline interface pairs differ from source")
    for role in ("candidate", "independent_source_baseline"):
        interface = report["interface_gate"][role]
        witness = interface["exact_partial_overlap_witnesses"]
        boundary = witness["boundary_predicate"]
        conditions = boundary["conditions"]
        _require(interface.get("passed") is True
                 and interface.get("observed_pair_count") == len(pairs)
                 and interface.get("observed_pairs_within_frozen_face_patch_cross_product") is True
                 and witness.get("passed") is True
                 and boundary.get("passed") is True
                 and conditions.get("exact_interface_points_nonempty") is True
                 and conditions.get("symmetric_exact_inside_and_outside_face_witnesses") is True
                 and conditions.get("all_observed_pairs_inside_frozen_collar_patch") is True
                 and conditions.get("no_unexpected_pairs_outside_frozen_patch") is True
                 and conditions.get("no_other_declared_feature_contacts") is True,
                 f"{role} aortic exact collar interface proof is incomplete")
        direct = report["direct_pair_audits"][(
            "candidate_actual" if role == "candidate" else "independent_source_baseline")]
        _require(direct.get("ordered_pairset_sha256")
                 == collision_owner.AORTIC_ARCH_INTERFACE_PAIRSET_SHA256
                 and direct.get("unexpected_pair_count") == len(pairs)
                 and direct.get("unexpected_pairs") == pairs,
                 f"{role} direct current/source pair witness differs from the frozen collar")
    for role in ("candidate", "independent_source_baseline"):
        solid = report["solid_gate"][role]
        accounting = solid["volume_accounting"]
        _require(solid.get("passed") is True
                 and solid.get("manifold_inputs_passed") is True
                 and solid.get("positive_intersection_volume") is True
                 and solid.get("intersection_volume_m3", 0.0) > 0.0
                 and solid.get("union_component_count") == 1
                 and solid.get("union_connected_numerical_solid") is True
                 and solid.get("union_lineage_has_both_sources_without_generated_or_unmapped_faces")
                 is True
                 and solid.get("floating_boolean_exact_topology_proven") is False
                 and accounting.get("union_volume_not_less_than_either_source") is True,
                 f"{role} aortic solid/inside-both/connected-union proof is incomplete")
    topology = report["source_topology"]
    for name in ("candidate_arch", "candidate_ascending",
                 "baseline_arch", "baseline_ascending"):
        surface = topology[name]
        quotient = surface["prepared_surface"]["exact_coordinate_quotient_topology"]
        _require(surface.get("closed_self_free_single_component") is True
                 and surface.get("exact_self_audit_count") == 0
                 and surface.get("signed_volume_m3", 0.0) > 0.0
                 and surface.get("referenced_face_component_count") == 1
                 and surface["prepared_surface"].get("referenced_surface_closed") is True
                 and quotient.get("closed_oriented_manifold_candidate") is True
                 and quotient.get("face_component_count") == 1
                 and quotient.get("boundary_edge_count") == 0
                 and quotient.get("orientation_defect_edges_count") == 0,
                 f"{name} source/candidate closed-solid topology proof is incomplete")
    for role, owner_id in (("arch", 7), ("ascending", 6)):
        calibre = report["independent_source_collar_calibre"][role]
        station = calibre["station"]
        comparison = calibre["candidate_to_baseline_comparison"]
        ratios = comparison["candidate_to_baseline_ratios"]
        _require(calibre.get("passed") is True and calibre.get("owner_id") == owner_id
                 and calibre.get("clinical_calibre_claim") is False
                 and station.get("inferred") is True
                 and station.get("transported_from_source_vertex_ancestry") is True
                 and station.get("mesh_coordinates_modified") is False
                 and comparison.get("passed") is True
                 and comparison.get("wide_bounds_are_not_preservation_success") is True
                 and 0.8 <= ratios["area"] <= 1.25
                 and 0.9 <= ratios["minimum_feret_width"] <= 1.1,
                 f"{role} lacks independent source-station calibre preservation")
    return {
        "schema": "numi.human.verified-aortic-arch-interface-handoff.v1",
        "status": "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE",
        "passed": True,
        "typed_link": [list(edge[0]), list(edge[1])],
        "evidence_report_sha256": evidence_report_sha256,
        "source_preflight_sha256": source_preflight_sha256,
        "capture_input_bundle_sha256": bundle_sha256,
        "source_topology_passed": True,
        "interface_gate_passed": True,
        "solid_gate_passed": True,
        "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
        "exact_face_pairs": pairs,
        "exact_face_pairset_sha256": collision_owner.AORTIC_ARCH_INTERFACE_PAIRSET_SHA256,
        "candidate_surface_bindings": candidate_bindings,
        "source_rows": source_rows,
        "numerical_boolean_topology_is_diagnostic_only": True,
        "anatomical_admission": False,
    }


def _load_aortic_arch_interface_certificate(runtime: dict, bundle_sha256: str) -> dict:
    report_path = AORTIC_PROOF_REPORT.resolve(strict=True)
    preflight_path = AORTIC_PROOF_PREFLIGHT.resolve(strict=True)
    report_sha = _sha_file(report_path)
    preflight_sha = _sha_file(preflight_path)
    _require(report_sha == collision_owner.AORTIC_ARCH_INTERFACE_REPORT_SHA256
             and preflight_sha == collision_owner.AORTIC_ARCH_INTERFACE_PREFLIGHT_SHA256,
             "pinned aortic source-interface evidence changed")
    preflight = json.loads(preflight_path.read_text())
    report = json.loads(report_path.read_text())
    _require(preflight.get("schema") == "numi.human.aortic-6-7-source-interface-preflight.v1"
             and preflight.get("status") == "PREFLIGHT_READY_NO_EXACT_INTERSECTION_OR_SOLID_PREDICATE"
             and preflight.get("capture_input_bundle_sha256") == bundle_sha256
             and preflight.get("parent_150_terminal_report_sha256")
             == AORTIC_PARENT_150_REPORT_SHA256
             and Path(report.get("preflight_path", "")).resolve() == preflight_path
             and report.get("preflight_sha256") == preflight_sha,
             "aortic proof preflight is not bound to its exact parent capture and report")
    return _aortic_arch_certificate_from_reports(
        runtime, bundle_sha256, report, preflight,
        evidence_report_sha256=report_sha,
        source_preflight_sha256=preflight_sha)


def _load_aortic_arch_descending_interface_certificate(
    runtime: dict, bundle_sha256: str,
) -> dict:
    """Load the frozen 7↔8 report, its preflight, and the compact handoff."""
    _require(bundle_sha256
             == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_CAPTURE_BUNDLE_SHA256,
             "7↔8 source-interface evidence belongs to another capture bundle")
    report_path = AORTIC_DESCENDING_PROOF_REPORT.resolve(strict=True)
    preflight_path = AORTIC_DESCENDING_PROOF_PREFLIGHT.resolve(strict=True)
    handoff_path = AORTIC_DESCENDING_PROOF_HANDOFF.resolve(strict=True)
    report_sha = _sha_file(report_path)
    preflight_sha = _sha_file(preflight_path)
    handoff_sha = _sha_file(handoff_path)
    _require(report_sha == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_REPORT_SHA256
             == AORTIC_DESCENDING_PROOF_REPORT_SHA256
             and preflight_sha
             == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_PREFLIGHT_SHA256
             == AORTIC_DESCENDING_PROOF_PREFLIGHT_SHA256
             and handoff_sha
             == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_HANDOFF_SHA256
             == AORTIC_DESCENDING_PROOF_HANDOFF_SHA256,
             "pinned 7↔8 source-interface evidence changed")
    report = json.loads(report_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    certificate = json.loads(handoff_path.read_text())
    edge = collision_owner.AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE
    expected_pair = [list(edge[0]), list(edge[1])]
    _require(preflight.get("schema")
             == "numi.human.aortic-7-8-source-interface-preflight.v1"
             and preflight.get("status")
             == "PREFLIGHT_READY_NO_EXACT_INTERSECTION_OR_SOLID_PREDICATE"
             and preflight.get("typed_pair") == expected_pair
             and preflight.get("capture_input_bundle_sha256") == bundle_sha256
             and preflight.get("parent_150_terminal_report_sha256")
             == AORTIC_PARENT_150_REPORT_SHA256
             and preflight.get("captured_step") == 0
             and preflight.get("captured_time_s") == 0,
             "7↔8 preflight is not bound to the cached actual012 step-0 capture")
    _require(report.get("schema") == "numi.human.aortic-7-8-source-interface-proof.v1"
             and report.get("status")
             == "PASS_SOURCE_BOUND_AORTA_ARCH_DESCENDING_INTERFACE_DIAGNOSTIC"
             and report.get("passed") is True
             and report.get("typed_pair") == expected_pair
             and report.get("preflight_sha256") == preflight_sha
             and Path(report.get("preflight_path", "")).resolve() == preflight_path
             and report.get("all_15_flow_links_unchanged") is True
             and report.get("source_topology_passed") is True
             and report.get("interface_gate_passed") is True
             and report.get("solid_gate_passed") is True
             and report.get("independent_source_collar_calibre_passed") is True
             and report.get("candidate_pairs_equal_frozen_source_pairs") is True
             and report.get("baseline_pairs_equal_frozen_source_pairs") is True
             and report.get("candidate_baseline_exact_pairsets_equal") is True,
             "pinned 7↔8 report does not carry its required source-bound passes")
    pairs = report.get("frozen_source_pairset")
    _require(isinstance(pairs, list)
             and pairs == report.get("independent_source_baseline_pairset")
             and pairs == certificate.get("exact_face_pairs")
             and len(pairs) == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_PAIR_COUNT
             and report.get("independent_source_baseline_pairset_sha256")
             == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_PAIRSET_SHA256
             and report.get("interface_gate_pairset_matches_frozen")
             == {"candidate": True, "independent_source_baseline": True},
             "7↔8 candidate/source exact face-pair sets are not identical")
    for role in ("candidate", "independent_source_baseline"):
        interface = report["interface_gate"][role]
        witnesses = interface["exact_partial_overlap_witnesses"]
        transverse = witnesses["exact_transverse_overlap_side_witnesses"]
        _require(interface.get("passed") is True
                 and interface.get("observed_face_pairs") == pairs
                 and witnesses.get("passed") is True
                 and transverse.get("passed") is True
                 and transverse.get("requires_all_four_witnesses_from_same_face_pair")
                 is True
                 and transverse.get("requires_strict_positive_exact_barycentric_coordinates")
                 is True
                 and transverse.get("uses_exact_rational_point_location_in_opposite_closed_surface")
                 is True,
                 f"7↔8 {role} exact transverse interface witnesses are incomplete")
        direct = report["direct_pair_audits"][
            "candidate_actual" if role == "candidate" else "independent_source_baseline"]
        _require(direct.get("unexpected_pairs") == pairs
                 and direct.get("ordered_pairset_sha256")
                 == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_PAIRSET_SHA256,
                 f"7↔8 {role} direct exact pairset differs from the certificate")
        solid = report["solid_gate"][role]
        inside = solid.get("exact_inside_both_witness")
        _require(solid.get("passed") is True
                 and solid.get("positive_intersection_volume") is True
                 and solid.get("intersection_volume_m3", 0.0) > 0.0
                 and inside.get("first_location") == "inside"
                 and inside.get("second_location") == "inside"
                 and solid.get("union_component_count") == 1
                 and solid.get("union_lineage_has_both_sources_without_generated_or_unmapped_faces")
                 is True,
                 f"7↔8 {role} exact inside-both/solid proof is incomplete")
        topology_names = (
            ("candidate_arch", "candidate_descending") if role == "candidate" else
            ("baseline_arch", "baseline_descending"))
        for topology_name in topology_names:
            topology = report["source_topology"][topology_name]
            prepared = topology.get("prepared_surface", {})
            quotient = prepared.get("exact_coordinate_quotient_topology", {})
            _require(topology.get("closed_self_free_single_component") is True
                     and topology.get("exact_self_audit_count") == 0
                     and topology.get("signed_volume_m3", 0.0) > 0.0
                     and topology.get("referenced_face_component_count") == 1
                     and prepared.get("referenced_surface_closed") is True
                     and quotient.get("closed_oriented_manifold_candidate") is True
                     and quotient.get("face_component_count") == 1
                     and quotient.get("boundary_edge_count") == 0
                     and quotient.get("orientation_defect_edges_count") == 0,
                     f"7↔8 {topology_name} is not a closed self-free solid")
    for role, owner_id in (("arch", 7), ("descending", 8)):
        calibre = report["independent_source_collar_calibre"][role]
        station = calibre["station"]
        comparison = calibre["candidate_to_baseline_comparison"]
        ratios = comparison["candidate_to_baseline_ratios"]
        _require(calibre.get("passed") is True
                 and calibre.get("owner_id") == owner_id
                 and calibre.get("clinical_calibre_claim") is False
                 and station.get("inferred") is True
                 and station.get("transported_from_source_vertex_ancestry") is True
                 and station.get("mesh_coordinates_modified") is False
                 and comparison.get("passed") is True
                 and 0.8 <= ratios["area"] <= 1.25
                 and 0.9 <= ratios["minimum_feret_width"] <= 1.1,
                 f"7↔8 {role} independent source-station calibre proof is incomplete")
    _require(certificate.get("schema")
             == "numi.human.verified-aortic-source-interface-handoff.v1"
             and certificate.get("status") == "VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE"
             and certificate.get("passed") is True
             and certificate.get("typed_link") == expected_pair
             and certificate.get("evidence_report_sha256") == report_sha
             and certificate.get("source_preflight_sha256") == preflight_sha
             and certificate.get("capture_input_bundle_sha256") == bundle_sha256
             and certificate.get("source_rows")
             == {f"{key[0]}:{key[1]}": value
                 for key, value in collision_owner.AORTIC_ARCH_DESCENDING_SOURCE_ROWS.items()}
             and certificate.get("exact_face_pairset_sha256")
             == collision_owner.AORTIC_ARCH_DESCENDING_INTERFACE_PAIRSET_SHA256
             and certificate.get("source_topology_passed") is True
             and certificate.get("interface_gate_passed") is True
             and certificate.get("solid_gate_passed") is True
             and certificate.get("independent_source_collar_calibre_passed") is True,
             "7↔8 compact handoff does not bind the exact passing report/preflight")
    source_identity = preflight.get("source_identity")
    labels = {f"{key[0]}:{key[1]}" for key in edge}
    _require(isinstance(source_identity, dict) and set(source_identity) == labels,
             "7↔8 preflight source identities differ from its exact typed link")
    gid_base = _native_nha_gid_base_offset(runtime["context"])[
        "native_nha_gid_base_offset"]
    live_surfaces = {}
    for key, expected in collision_owner.AORTIC_ARCH_DESCENDING_SOURCE_ROWS.items():
        label = f"{key[0]}:{key[1]}"
        row = source_identity[label]
        ancestry = row.get("source_to_capture_ancestry")
        _require(row.get("typed_identity") == list(key)
                 and row.get("source_member") == expected["source_member"]
                 and row.get("fma") == expected["fma"]
                 and row.get("name") == expected["name"]
                 and row.get("body_index") == expected["body_index"]
                 and row.get("layer") == expected["layer"]
                 and row.get("first_vertex") == expected["first_vertex"]
                 and row.get("vertex_count") == expected["vertex_count"]
                 and row.get("face_count") == expected["face_count"]
                 and row.get("source_face_u32_sha256") == expected["faces_sha256"]
                 and row.get("actual_face_u32_sha256") == expected["faces_sha256"]
                 and isinstance(ancestry, dict)
                 and ancestry.get("status")
                 == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
                 and ancestry.get("typed_identity") == list(key)
                 and ancestry.get("source_abi5_face_array_sha256")
                 == expected["faces_sha256"]
                 and ancestry.get("capture_faces_u32_sha256")
                 == expected["faces_sha256"]
                 and ancestry.get("ordered_source_face_ids_are_preserved") is True
                 and ancestry.get("capture_global_vertex_ids_are_actual_pack_values")
                 is True
                 and ancestry.get("native_source_slot_identity_is_contiguous") is True
                 and ancestry.get("native_nha_gid_base_offset") == gid_base
                 and ancestry.get(
                     "observed_gid_minus_nha_global_vertex_index_unique_count") == 1
                 and ancestry.get(
                     "observed_gid_minus_nha_global_vertex_index_min") == gid_base
                 and ancestry.get(
                     "observed_gid_minus_nha_global_vertex_index_max") == gid_base,
                 f"7↔8 source row/slot ancestry changed for {key}")
        value = runtime["candidate_surfaces"].get(
            key, runtime["context"]["surfaces"].get(key))
        _require(value is not None, f"current runtime is missing 7↔8 surface {key}")
        live_fingerprint = collision_owner._aortic_arch_surface_fingerprint(
            value, key=key)
        _require(certificate["candidate_surface_bindings"].get(label)
                 == live_fingerprint
                 and live_fingerprint["positions_f32_sha256"]
                 == row.get("actual_position_f32_sha256")
                 and live_fingerprint["faces_u32_sha256"]
                 == row.get("actual_face_u32_sha256")
                 and live_fingerprint["normals_f32_sha256"]
                 == row.get("actual_normal_f32_sha256")
                 and live_fingerprint["source_vertex_ids_u64_sha256"]
                 == row.get("actual_gid_u64_sha256"),
                 f"7↔8 proof arrays differ from current actual surface {key}")
        live_surfaces[key] = value
    collision_owner._validate_aortic_arch_descending_interface_certificate(
        certificate, live_surfaces, set(NETWORK_KEYS))
    return certificate


def _load_cached_aortic_interface_certificates(
    runtime: dict, bundle_sha256: str,
) -> dict:
    """Load only the two separately pinned aortic interfaces in this capture."""
    certificates = {
        collision_owner.AORTIC_ARCH_SOURCE_INTERFACE:
            _load_aortic_arch_interface_certificate(runtime, bundle_sha256),
        collision_owner.AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE:
            _load_aortic_arch_descending_interface_certificate(runtime, bundle_sha256),
    }
    for edge, certificate in certificates.items():
        _require(certificate.get("typed_link") == [list(edge[0]), list(edge[1])],
                 f"aortic source-interface loader returned the wrong typed link {edge}")
    return certificates


def _validate_cached_network_inventory(live_inventory: dict, cached_inventory: dict) -> None:
    fields = (
        "candidate_surface_count", "independent_source_baseline_surface_count",
        "candidate_surfaces", "independent_source_baselines",
        "source_face_arrays_bound_by_exact_ordered_abi5_corner_ancestry",
        "native_source_slot_identity", "source_face_ancestry_by_typed_row",
        "candidate_and_baseline_global_gid_arrays_identical_per_row",
        "candidate_baseline_gid_identity_by_typed_row",
        "candidate_baseline_gid_comparison_dtype",
        "candidate_gid_values_were_read_from_actual_capture_pack",
        "baseline_coordinates_are_independently_cpu_reconstructed",
        "baseline_gid_values_are_transported_current_nha_source_slots",
    )
    _require(isinstance(cached_inventory, dict)
             and all(cached_inventory.get(key) == live_inventory.get(key)
                     for key in fields),
             "cached 15-link inventory differs from current typed source/pack rows")


def _validate_run150_explicit_capture_bundle(
    capture_inputs: dict, explicit_bundle: object,
) -> None:
    """Bind the seven-field bundle plus its separately declared payload file."""
    _require(isinstance(capture_inputs, dict) and isinstance(explicit_bundle, dict)
             and set(explicit_bundle) == set(capture_inputs) | {"payload_path"},
             "run150 explicit capture bundle has an unexpected field set")
    for key, value in capture_inputs.items():
        observed = explicit_bundle[key]
        if key.endswith("_path"):
            _require(isinstance(value, str) and isinstance(observed, str)
                     and Path(value).resolve(strict=True)
                     == Path(observed).resolve(strict=True),
                     f"run150 explicit capture path differs for {key}")
        else:
            _require(observed == value,
                     f"run150 explicit capture field differs for {key}")
    payload_path = explicit_bundle.get("payload_path")
    _require(isinstance(payload_path, str)
             and Path(payload_path).resolve(strict=True).is_file()
             and _sha_file(Path(payload_path).resolve(strict=True))
             == capture_inputs.get("anatomy_payload_sha256"),
             "run150 explicit anatomy payload does not match its pinned SHA-256")


def continue_cached_all_target_screen(
    capture_inputs_json: Path, output_dir: Path, *, cached_terminal_report: Path | None = None,
) -> dict:
    """Continue only the all-target stage from the exact cached 150 link pass."""
    started = time.perf_counter()
    bundle_path = Path(capture_inputs_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    terminal_path = Path(
        AORTIC_PARENT_150_REPORT if cached_terminal_report is None
        else cached_terminal_report).resolve(strict=True)
    _require(not destination.exists(),
             "continuation output directory already exists; use an immutable new path")
    bundle_bytes = bundle_path.read_bytes()
    bundle_sha = _sha_bytes(bundle_bytes)
    terminal_sha = _sha_file(terminal_path)
    report_path = destination / "all-target-continuation-report.json"
    progress_path = destination / "progress.json"
    preflight_path = destination / "continuation-preflight.json"
    start = {
        "schema": "numi.human.current012-cached-all-target-continuation.v1",
        "status": "RUNNING_CACHED_ALL_TARGET_CONTINUATION_PREFLIGHT",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "capture_input_bundle_path": str(bundle_path),
        "capture_input_bundle_sha256": bundle_sha,
        "cached_terminal_report_path": str(terminal_path),
        "cached_terminal_report_sha256": terminal_sha,
        "geometry_predicates_run": False,
        "continuity_predicates_rerun": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }
    destination.mkdir(parents=True)
    _write_json(progress_path, start)
    geometry_started = False
    try:
        _require(bundle_sha == collision_owner.AORTIC_ARCH_INTERFACE_CAPTURE_BUNDLE_SHA256
                 and terminal_path == AORTIC_PARENT_150_REPORT.resolve()
                 and terminal_sha == AORTIC_PARENT_150_REPORT_SHA256,
                 "cached continuation is not the exact immutable run150 capture/result")
        terminal = json.loads(terminal_path.read_text())
        capture_inputs = json.loads(bundle_bytes)
        explicit_bundle = terminal.get("current_capture", {}).get(
            "explicit_merged_capture_input_bundle")
        _validate_run150_explicit_capture_bundle(capture_inputs, explicit_bundle)
        live_preflight, runtime = preparation.build_preflight(
            capture_inputs, return_runtime_inputs=True)
        _require(live_preflight.get("status") == "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE"
                 and live_preflight.get("actual_accepted_capture")
                 == terminal.get("current_capture"),
                 "live source/capture preparation differs from cached run150 accepted capture")
        continuity = terminal.get("current012_network_continuity")
        _require(terminal.get("schema")
                 == "numi.human.current012-vascular-network-all-target-screen.v1"
                 and terminal.get("status")
                 == "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_REJECTED"
                 and terminal.get("failed_stage")
                 == "network_surface_against_nonnetwork_target"
                 and terminal.get("passed") is False
                 and isinstance(continuity, dict)
                 and continuity.get("status")
                 == "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED"
                 and continuity.get("passed") is True
                 and continuity.get("evaluated_link_count") == 15
                 and continuity.get("requested_link_count") == 15
                 and len(continuity.get("links", [])) == 15
                 and all(link.get("passed") is True for link in continuity["links"]),
                 "cached run150 did not pass all 15 continuity links before all-target")
        old_gate = terminal.get("all_target_screen", {})
        old_shared = old_gate.get("shared_verifier_result", {})
        old_rows = old_shared.get("rows", [])
        first_failure = next((row for row in old_rows if row.get("passed") is False), None)
        _require(old_gate.get("failed_stage") == "network_surface_against_nonnetwork_target"
                 and isinstance(first_failure, dict)
                 and first_failure.get("identities") == [[51011, 7], [51011, 6]],
                 "cached all-target failure is not the exact aortic arch/ascending interface")
        candidates, _baselines, _link_inputs, live_inventory = _bind_runtime_inputs(runtime)
        cached_inventory = terminal.get("network_inventory")
        _validate_cached_network_inventory(live_inventory, cached_inventory)
        certificate_map = _load_cached_aortic_interface_certificates(runtime, bundle_sha)
        aortic_certificate = certificate_map[
            collision_owner.AORTIC_ARCH_SOURCE_INTERFACE]
        descending_certificate = certificate_map[
            collision_owner.AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE]
        _require(first_failure.get("triangle_pair_count")
                 == collision_owner.AORTIC_ARCH_INTERFACE_PAIR_COUNT
                 and first_failure.get("unexpected_triangle_pairs")
                 == aortic_certificate["exact_face_pairs"],
                 "cached first-failure contacts differ from the certified exact aortic pairset")

        overlay_pin_map = terminal.get("runner_code_pins", {})
        _require(isinstance(overlay_pin_map, dict)
                 and overlay_pin_map.get(str(AORTIC_TARGET_OVERLAY_RUNNER.resolve()))
                 == AORTIC_TARGET_OVERLAY_RUNNER_SHA256
                 and _sha_file(AORTIC_TARGET_OVERLAY_RUNNER.resolve(strict=True))
                 == AORTIC_TARGET_OVERLAY_RUNNER_SHA256,
                 "run150 target-overlay preparation source changed")
        spec = importlib.util.spec_from_file_location(
            "numi_current012_network150_pinned_overlay",
            AORTIC_TARGET_OVERLAY_RUNNER.resolve())
        _require(spec is not None and spec.loader is not None,
                 "cannot load the pinned run150 target-overlay preparation")
        overlay_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(overlay_module)
        collision_surfaces, target_overlay = overlay_module.prepare_target_overlays(runtime)
        frozen_overlay = terminal.get("counterfactual_target_overlay")
        _require(_stable_preflight(target_overlay) == _stable_preflight(frozen_overlay),
                 "replayed run150 rest/313 counterfactual target overlays changed")
        collision_surfaces.update(candidates)
        preflight = {
            **start,
            "status": "PREFLIGHT_READY_EXACT_CACHED_CONTINUITY_AND_AORTIC_HANDOFFS",
            "runner_code_pins": _runtime_code_pins(),
            "captured_source_rows": live_preflight["actual_accepted_capture"],
            "continuity_report_sha256": terminal_sha,
            "continuity_report_status": continuity["status"],
            "continuity_link_count": 15,
            "cached_network_inventory_typed_row_count": len(
                cached_inventory["candidate_surfaces"]),
            "aortic_interface_certificate": {
                key: value for key, value in aortic_certificate.items()
                if key not in ("exact_face_pairs", "candidate_surface_bindings")
            },
            "aortic_interface_exact_face_pair_count": len(
                aortic_certificate["exact_face_pairs"]),
            "aortic_interface_candidate_surface_bindings":
                aortic_certificate["candidate_surface_bindings"],
            "aortic_arch_descending_interface_certificate": {
                key: value for key, value in descending_certificate.items()
                if key not in ("exact_face_pairs", "candidate_surface_bindings")
            },
            "aortic_arch_descending_interface_exact_face_pair_count": len(
                descending_certificate["exact_face_pairs"]),
            "aortic_arch_descending_interface_candidate_surface_bindings":
                descending_certificate["candidate_surface_bindings"],
            "counterfactual_target_overlay_summary": {
                "typed_identity": target_overlay["typed_identity"],
                "candidate_nha_sha256": target_overlay["candidate_nha_sha256"],
                "candidate_overlay_sha256": target_overlay["candidate_overlay_sha256"],
                "changed_actual012_face_slots": target_overlay["changed_actual012_face_slots"],
                "rest_source_rows": target_overlay["rest_source_rows"],
                "rest_source_nhtiss_sha256": target_overlay["rest_source_nhtiss_sha256"],
                "rest_source_rows_count": target_overlay["rest_source_rows_count"],
                "full_replay_sha256": _sha_bytes(json.dumps(
                    target_overlay, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode()),
            },
            "geometry_predicates_run": False,
            "continuity_predicates_rerun": False,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        }
        _write_json(preflight_path, preflight)
        _write_json(progress_path, {
            **start,
            "status": "STARTING_ALL_TARGET_AFTER_CACHED_15_LINK_PASS",
            "continuation_preflight_path": str(preflight_path.resolve()),
            "continuation_preflight_sha256": _sha_file(preflight_path),
            "geometry_predicates_run": True,
            "continuity_predicates_rerun": False,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        })

        def progress_callback(stage):
            _write_json(progress_path, {
                **start,
                "status": stage,
                "continuation_preflight_path": str(preflight_path.resolve()),
                "continuation_preflight_sha256": _sha_file(preflight_path),
                "geometry_predicates_run": True,
                "continuity_predicates_rerun": False,
                "native_or_gpu_work_performed": False,
                "anatomical_admission": False,
            })

        geometry_started = True
        collision = preparation.run_all_target_network_screen(
            {**runtime["context"], "surfaces": collision_surfaces},
            runtime["edge_evidence"], runtime["separate_evidence"],
            verified_network_continuity=continuity,
            verified_network_source_inventory=cached_inventory,
            verified_nonnetwork_interface_certificates=certificate_map,
            progress_callback=progress_callback)
        passed = collision.get("passed") is True
        result = {
            **preflight,
            "status": ("CACHED_ALL_TARGET_SCREEN_PASSED_UNADMITTED"
                       if passed else "CACHED_ALL_TARGET_SCREEN_REJECTED"),
            "passed": passed,
            "cached_continuity_predicates_rerun": False,
            "all_target_collision_gate": collision,
            "counterfactual_target_overlay": preflight[
                "counterfactual_target_overlay_summary"],
            "recorded_finish_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "geometry_predicates_run": True,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        }
        _write_json(report_path, result)
        result["report_path"] = str(report_path.resolve())
        result["report_sha256"] = _sha_file(report_path)
        _write_json(progress_path, result)
        return result
    except Exception as error:
        failure = {
            **start,
            "status": "CACHED_ALL_TARGET_CONTINUATION_ABORTED_OR_REJECTED",
            "recorded_finish_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": {"type": type(error).__name__, "message": str(error)},
            "continuation_preflight_path": (
                str(preflight_path.resolve()) if preflight_path.exists() else None),
            "continuation_preflight_sha256": (
                _sha_file(preflight_path) if preflight_path.exists() else None),
            "geometry_predicates_run": geometry_started,
            "continuity_predicates_rerun": False,
            "native_or_gpu_work_performed": False,
            "anatomical_admission": False,
        }
        _write_json(destination / "continuation-error.json", failure)
        _write_json(progress_path, failure)
        raise


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
    mode.add_argument("--continue-cached-all-targets", action="store_true",
                      help="continue only the exact run150 all-target stage using its cached 15-link pass")
    parser.add_argument("--cached-terminal-report", type=Path,
                        help="exact immutable run150 report (defaults to the pinned report)")
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
    elif args.continue_cached_all_targets:
        result = continue_cached_all_target_screen(
            args.capture_inputs_json, args.output_dir,
            cached_terminal_report=args.cached_terminal_report)
    else:
        result = run(args.capture_inputs_json, args.output_dir, run_all_targets=True)
    print(json.dumps({
        "status": result["status"],
        "report_path": result.get(
            "report_path", result.get("path", result.get("source_preflight_path"))),
        "report_sha256": result.get(
            "report_sha256", result.get("sha256", result.get("source_preflight_sha256"))),
    }, sort_keys=True))
    if args.continue_cached_all_targets:
        return 0 if result.get("passed") is True else 2
    return 0 if result["status"].startswith((
        "SOURCE0_SEAMOWNER_", "STOPPED_AT_FIRST_FAILED_NETWORK_LINK",
        "NETWORK_CONTINUITY_PASSED_",
        "VERIFIED_BASELINE_ONLY_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

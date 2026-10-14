"""Instrument one rejected brachiocephalic source port without changing its owners.

This diagnostic wraps immutable geometry helper functions at runtime to retain
station/section evidence. It evaluates only FJ3482 (stable ID 525) against
the frozen left internal-jugular port (stable ID 469).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
RUN_REPORT = A / "brachiocephalic-source-007-six-port-geometry-001/source-007-geometry-report.json"
RUN_RECEIPT = A / "brachiocephalic-source-007-six-port-geometry-001/source-007-execution-receipt-001.json"
CANDIDATE_DIR = A / "brachiocephalic-source-candidate-007-001"
PREPARATION_DIR = A / "brachiocephalic-common-field-preparation-007-001"
CAPTURE_MANIFEST = A / "accepted-capture-inputs-1398-001.json"
DIAGNOSTIC_DIR = A / "brachiocephalic-source-007-six-port-geometry-001/port-525-469-diagnostic-005"
PREFLIGHT_PATH = A / "brachiocephalic-source-007-six-port-geometry-001/port-525-469-diagnostic-preflight-005.json"
HP98_ROOT = Path("/Users/n/numi-human-resting-anatomy-gates-publish-1398")
HP98_VERIFIER = HP98_ROOT / "src/numilab_human/resting_anatomy_verify.py"
HP98_SHA256 = "87d21c53c4ac003fa8df7410cc681d91a90465af808217d8af4561155d5094bb"
RUN_REPORT_SHA256 = "9d0fb2fc1b0b68f111387f867aec35168587ca0bd1bada50519b4ccc9da1a612"
RUN_RECEIPT_SHA256 = "4ab6f9c475a90ac76df18989396098736b005b3a9a657aaf1b95a9e69641371f"
EXPECTED_COMMIT = "e921f79b9b078911c30f552890d22274fb8cf56e"
SOURCE007_GATE_RELATIVE_PATH = (
    "src/numilab_human/resting_brachiocephalic_geometry_gate.py")
SOURCE007_GATE_SHA256 = (
    "adf7120f4257d0579b70fb4c7c1c98bb49e479596e0bbc604015bdee9608e977")
RUNTIME_CODE_EXPECTED_SHA256 = {
    "resting_brachiocephalic_geometry_gate.py":
        "103bc36649ddb7fb921e2c77e72d24b6ae118f223b75025d2e1f7e1f094c3e0b",
    "resting_brachiocephalic_source007.py":
        "975b2124ccfc96131197986893ea79c18f2fb7943de27227ff103883c245c8d3",
    "resting_brachiocephalic_pose_predictor.py":
        "3d1d6f4d69cd2d92dee0aaeaf39b6060979984508cb4c19bbfd2768b7621d748",
    "resting_pleura_proxy.py":
        "d874f599b8aeb9e9f02318fbae30ed5bf017920170d41c8b775cc3a9f792d683",
}
BRIDGE_KEY = (51011, 525)
TARGET_KEY = (51011, 469)
REPLAY_SCOPE = {
    "bridge": list(BRIDGE_KEY),
    "target": list(TARGET_KEY),
    "port_count": 1,
    "source_candidate_compiler_invoked": False,
    "all_target_checks_invoked": False,
    "stop_after_single_port_result": True,
}
SECTION_STAGES = (
    "bridge_candidate", "bridge_baseline",
    "target_candidate", "target_baseline",
    "union_candidate_at_bridge_station", "union_baseline_at_bridge_station",
    "union_candidate_at_target_station", "union_baseline_at_target_station",
)
COMPARISON_STAGES = (
    "bridge_source_section", "target_source_section",
    "union_section_at_bridge_station", "union_section_at_target_station",
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("port 525-to-469 diagnostic: " + message)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(value: Any, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _actual_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def build_preflight() -> dict:
    require(not PREFLIGHT_PATH.exists() and not DIAGNOSTIC_DIR.exists(),
            "retain existing diagnostic output and choose a successor")
    require(RUN_REPORT.is_file() and sha_file(RUN_REPORT) == RUN_REPORT_SHA256,
            "source-007 failure report changed")
    require(RUN_RECEIPT.is_file() and sha_file(RUN_RECEIPT) == RUN_RECEIPT_SHA256,
            "source-007 execution receipt changed")
    run = json.loads(RUN_REPORT.read_text())
    run_receipt = json.loads(RUN_RECEIPT.read_text())
    require(run_receipt.get("exit_code") == 2
            and run_receipt.get("geometry_report_sha256") == RUN_REPORT_SHA256,
            "source-007 execution receipt does not match the failed run")
    require(run.get("failed_stage") == "six-port-overlap-and-sections"
            and run.get("six_port_source_geometry", {}).get("first_failed_port") == {
                "bridge": [51011, 525], "target": [51011, 469]},
            "source-007 did not stop at the frozen first failed port")
    require(run.get("six_port_source_geometry", {}).get("evaluated_port_count") == 2,
            "source-007 failed-port evaluation count changed")
    require(run.get("code_and_dependency_pins", {}).get(
                "source_repository", {}).get("head") == EXPECTED_COMMIT,
            "source-007 inputs were not produced by the pinned e921 runner")
    code_pins = run.get("code_and_dependency_pins", {}).get("source_code_sha256", {})
    repo = Path(__file__).resolve().parents[2]
    historical_gate_pins = []
    for raw, expected in code_pins.items():
        path = Path(raw)
        if (path.name == "resting_brachiocephalic_geometry_gate.py"
                and expected == SOURCE007_GATE_SHA256):
            historical = subprocess.run(
                ["git", "-C", str(repo), "show",
                 f"{EXPECTED_COMMIT}:{SOURCE007_GATE_RELATIVE_PATH}"],
                check=True, capture_output=True).stdout
            require(hashlib.sha256(historical).hexdigest() == SOURCE007_GATE_SHA256,
                    "source-007 pre-fix section-helper blob differs from its e921 pin")
            historical_gate_pins.append({
                "source_path": raw,
                "source_sha256": expected,
                "verified_from_commit": EXPECTED_COMMIT,
                "relative_path": SOURCE007_GATE_RELATIVE_PATH,
            })
            continue
        require(path.is_file() and sha_file(path) == expected,
                "source-007 code/helper pin changed: " + raw)
    require(len(historical_gate_pins) == 1,
            "source-007 must preserve exactly one historical section-helper pin")
    verifier_pin = code_pins.get(str(HP98_VERIFIER.resolve()))
    require(verifier_pin == HP98_SHA256 and sha_file(HP98_VERIFIER) == HP98_SHA256,
            "central HP98 verifier differs from the run's frozen owner")
    candidate_expected = run.get("candidate007_files_sha256", {})
    prep_expected = run.get("prep007_files_sha256", {})
    require(_actual_file_hashes(CANDIDATE_DIR) == candidate_expected,
            "candidate007 emitted files changed")
    require(_actual_file_hashes(PREPARATION_DIR) == prep_expected,
            "prep007 emitted files changed")
    source0 = run["source0"]
    for path_key, hash_key in (("pack_path", "pack_sha256"),
                               ("receipt_path", "receipt_sha256")):
        path = Path(source0[path_key])
        require(path.is_file() and sha_file(path) == source0[hash_key],
                "pinned source-0 " + path_key + " bytes changed")
    capture_manifest_sha = run.get("input_sha256", {}).get(str(CAPTURE_MANIFEST))
    require(capture_manifest_sha == source0.get("capture_manifest_sha256")
            and sha_file(CAPTURE_MANIFEST) == capture_manifest_sha,
            "accepted-capture manifest differs from source-007")
    code_path = Path(__file__).resolve()
    runtime_code = {}
    for name, expected in RUNTIME_CODE_EXPECTED_SHA256.items():
        path = code_path.with_name(name)
        require(path.is_file() and sha_file(path) == expected,
                "runtime owner code changed: " + str(path))
        runtime_code[str(path)] = sha_file(path)
    manifold_pins = run.get("code_and_dependency_pins", {}).get(
        "manifold3d_retained_wheel", {})
    require(bool(manifold_pins), "source-007 did not pin the retained Manifold wheel")
    for raw_path, expected in manifold_pins.items():
        path = Path(raw_path)
        require(path.is_file() and sha_file(path) == expected,
                "retained Manifold wheel file changed: " + raw_path)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    require(not git_status, "replay preflight requires a clean pinned source checkout")
    return {
        "schema": "numi.human.brachiocephalic-port-525-469-diagnostic-preflight.v2",
        "status": "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
        "diagnostic_code_path": str(code_path),
        "diagnostic_code_sha256": sha_file(code_path),
        "diagnostic_repository": {"head": commit, "dirty": bool(git_status),
                                  "status_lines": git_status},
        "replay_scope": REPLAY_SCOPE,
        "runtime_owner_code_sha256": runtime_code,
        "source_compiler_invoked": False,
        "source007_commit": EXPECTED_COMMIT,
        "source007_failure_report_sha256": RUN_REPORT_SHA256,
        "source007_execution_receipt_sha256": RUN_RECEIPT_SHA256,
        "candidate007_files_sha256": candidate_expected,
        "prep007_files_sha256": prep_expected,
        "source0_pack_path": source0["pack_path"],
        "source0_pack_sha256": source0["pack_sha256"],
        "source0_receipt_path": source0["receipt_path"],
        "source0_receipt_sha256": source0["receipt_sha256"],
        "hp98_verifier_path": str(HP98_VERIFIER),
        "hp98_verifier_sha256": HP98_SHA256,
        "source007_code_dependency_pins": code_pins,
        "source007_historical_section_helper_provenance": historical_gate_pins,
        "retained_manifold_wheel_pins": run.get(
            "code_and_dependency_pins", {}).get("manifold3d_retained_wheel"),
        "port": {"bridge": list(BRIDGE_KEY), "target": list(TARGET_KEY)},
        "geometry_arrays_loaded": False,
        "geometry_predicates_run": False,
        "source_or_helper_changed": False,
        "diagnostic_output_absent": True,
    }


def write_preflight() -> dict:
    report = build_preflight()
    _write_json(PREFLIGHT_PATH, report)
    report["report_path"] = str(PREFLIGHT_PATH)
    report["report_sha256"] = sha_file(PREFLIGHT_PATH)
    return report


def _section_summary(section: dict) -> dict:
    return {
        key: section.get(key) for key in (
            "passed", "loop_count", "vertex_count", "segment_count", "area_m2",
            "minimum_feret_width_m", "area_mm2", "minimum_feret_width_mm",
            "coordinate_dtype", "coordinate_ulp_tolerance_m",
            "positive_measure_floor_m", "section_node_count",
            "section_degree_histogram", "section_node_kind_counts",
            "section_edge_intersection_count", "minimum_section_segment_length_m",
            "exact_coordinate_quotient_vertex_count",
            "exact_coordinate_duplicate_vertex_count", "coplanar_edge_count",
            "isolated_tangent_vertex_ids", "coplanar_face_count",
            "reason")
        if key in section
    }


class Recorder:
    def __init__(self, progress_path: Path | None = None):
        self.progress_path = progress_path
        self.context = {}
        self.data = {
            "section_derivation": [],
            "section_calls": [],
            "section_comparisons": [],
            "material_station_transport": [],
            "port_interface_audit": [],
            "manifold_inputs": [],
            "inside_both_witnesses": [],
            "solid_boolean_operations": [],
        }

    def bind(self, context: dict) -> None:
        self.context = context
        self.flush()

    def record(self, collection: str, entry: dict) -> None:
        self.data[collection].append({
            **_jsonable(entry),
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        })
        self.flush()

    def flush(self) -> None:
        if self.progress_path is not None:
            _write_json(self.progress_path, {**self.context, "trace": self.data})


def _surface_ancestry(surface: Any) -> dict:
    result = {}
    for name in ("positions", "faces", "source_vertex_ids",
                 "quotient_positions", "quotient_faces",
                 "quotient_source_vertex_ids"):
        value = getattr(surface, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        result[name + "_sha256"] = _array_hash(array)
        result[name + "_shape"] = list(array.shape)
        result[name + "_dtype"] = str(array.dtype)
    return result


def _solid_metrics(solid: Any) -> dict:
    metrics = {}
    for name, getter in (
        ("status", lambda value: str(value.status())),
        ("volume_m3", lambda value: float(value.volume())),
        ("component_count", lambda value: len(value.decompose())),
        ("triangle_count", lambda value: int(value.num_tri())),
        ("vertex_count", lambda value: int(value.num_vert())),
        ("genus", lambda value: int(value.genus())),
    ):
        try:
            metrics[name] = getter(solid)
        except Exception as error:  # noqa: BLE001 - retain partial solid diagnostics
            metrics[name + "_error"] = {
                "type": type(error).__name__, "message": str(error)}
    return metrics


def _recording_manifold_proxy(manifold3d: Any, recorder: Recorder):
    """Wrap only static Boolean dispatch so port-level CSG metrics survive errors."""
    native_manifold = manifold3d.Manifold
    native_batch_boolean = native_manifold.batch_boolean

    class ObservedManifold:
        def __new__(cls, *args, **kwargs):
            return native_manifold(*args, **kwargs)

        @staticmethod
        def batch_boolean(solids, operation):
            inputs = list(solids)
            record = {
                "operation": str(operation),
                "inputs": [_solid_metrics(solid) for solid in inputs],
                "input_count": len(inputs),
            }
            try:
                result = native_batch_boolean(inputs, operation)
                record["result"] = _solid_metrics(result)
                recorder.record("solid_boolean_operations", record)
                return result
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                recorder.record("solid_boolean_operations", record)
                raise

    return type("ObservedManifoldModule", (), {
        "Manifold": ObservedManifold,
        "Mesh": manifold3d.Mesh,
        "OpType": manifold3d.OpType,
    })()


def install_recording_wrappers(gate: Any, verifier: Any, recorder: Recorder):
    """Wrap read-only helpers to persist station, section, and port witnesses."""
    originals = []
    section_count = 0
    comparison_count = 0
    station_count = 0
    transport_count = 0

    original_section = gate._section_loops
    def wrapped_section(positions, faces, center, tangent, *, derived_csg=False):
        nonlocal section_count
        index = section_count
        section_count += 1
        stage = SECTION_STAGES[index] if index < len(SECTION_STAGES) else f"section_{index}"
        point = np.asarray(center, dtype=np.float64)
        direction = np.asarray(tangent, dtype=np.float64)
        record = {
            "call_index": index,
            "stage": stage,
            "derived_csg": bool(derived_csg),
            "positions_sha256": _array_hash(positions),
            "positions_shape": list(np.asarray(positions).shape),
            "positions_dtype": str(np.asarray(positions).dtype),
            "faces_sha256": _array_hash(faces),
            "faces_shape": list(np.asarray(faces).shape),
            "center_world_m": point.tolist(),
            "center_world_sha256": _array_hash(point, "<f8"),
            "tangent_world_unit": direction.tolist(),
            "tangent_world_sha256": _array_hash(direction, "<f8"),
            "section_plane_signature_sha256": _json_hash({
                "center_f64": point.tolist(), "tangent_f64": direction.tolist()}),
        }
        try:
            result = original_section(positions, faces, center, tangent,
                                      derived_csg=derived_csg)
            record["section_metrics"] = _section_summary(result)
            recorder.record("section_calls", record)
            return result
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            recorder.record("section_calls", record)
            raise

    original_compare = gate._section_comparison
    def wrapped_comparison(candidate, baseline):
        nonlocal comparison_count
        index = comparison_count
        comparison_count += 1
        stage = COMPARISON_STAGES[index] if index < len(COMPARISON_STAGES) else f"comparison_{index}"
        record = {
            "call_index": index,
            "stage": stage,
            "candidate_section": _section_summary(candidate),
            "baseline_section": _section_summary(baseline),
            "candidate_section_sha256": _json_hash(candidate),
            "baseline_section_sha256": _json_hash(baseline),
        }
        try:
            result = original_compare(candidate, baseline)
            record["comparison"] = _jsonable(result)
            recorder.record("section_comparisons", record)
            return result
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            record["passed"] = False
            recorder.record("section_comparisons", record)
            raise

    original_station = gate._validated_station_spec
    def wrapped_station(surface, value, *, witness_points, collar_face_ids,
                        route_anchor=None):
        nonlocal station_count
        index = station_count
        station_count += 1
        role = ("bridge" if index == 0 else "target" if index == 1
                else f"station_{index}")
        points = np.asarray(witness_points, dtype=np.float64)
        faces = np.asarray(collar_face_ids, dtype="<i8")
        record = {
            "call_index": index,
            "role": role,
            "surface_ancestry": _surface_ancestry(surface),
            "exact_port_witness_points_world_m": points.tolist(),
            "exact_port_witness_points_sha256": _array_hash(points, "<f8"),
            "frozen_collar_face_ids": faces.tolist(),
            "frozen_collar_face_ids_sha256": _array_hash(faces, "<i8"),
            "input_station_spec_sha256": None if value is None else _json_hash(value),
            "route_anchor_world_m": None if route_anchor is None else np.asarray(
                route_anchor, dtype=np.float64).tolist(),
        }
        try:
            result = original_station(
                surface, value, witness_points=witness_points,
                collar_face_ids=collar_face_ids, route_anchor=route_anchor)
            record["station"] = _jsonable(result)
            record["station_center_sha256"] = _array_hash(
                result["section_center_world_m"], "<f8")
            record["station_tangent_sha256"] = _array_hash(
                result["tangent_world_unit"], "<f8")
            record["material_ancestry_sha256"] = _json_hash({
                key: result.get(key) for key in (
                    "material_anchor_local_vertex_ids", "material_anchor_source_vertex_ids",
                    "material_anchor_weights", "tangent_support_local_vertex_ids",
                    "tangent_support_source_vertex_ids", "frozen_collar_face_ids")
            })
            recorder.record("section_derivation", record)
            return result
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            recorder.record("section_derivation", record)
            raise

    original_transport = gate._transport_material_station
    def wrapped_transport(surface, spec):
        nonlocal transport_count
        index = transport_count
        transport_count += 1
        record = {
            "call_index": index,
            "role": ("candidate_bridge" if index == 0 else "candidate_target"
                     if index == 1 else f"transport_{index}"),
            "surface_ancestry": _surface_ancestry(surface),
            "input_station_spec": _jsonable(spec),
            "input_station_spec_sha256": _json_hash(spec),
        }
        try:
            result = original_transport(surface, spec)
            record["transported_station"] = _jsonable(result)
            record["center_world_sha256"] = _array_hash(
                result["section_center_world_m"], "<f8")
            record["tangent_world_sha256"] = _array_hash(
                result["tangent_world_unit"], "<f8")
            recorder.record("material_station_transport", record)
            return result
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            recorder.record("material_station_transport", record)
            raise

    original_make = gate._make_manifold
    def wrapped_make(surface, manifold3d):
        result, info = original_make(surface, manifold3d)
        recorder.record("manifold_inputs", {
            "surface_identity": list(surface.key),
            "surface_ancestry": _surface_ancestry(surface),
            "input_diagnostics": _jsonable(info),
            "manifold_status": str(result.status()),
            "manifold_volume_m3": float(result.volume()),
        })
        return result, info

    original_inside = gate._exact_inside_witness
    def wrapped_inside(verifier_module, first, second, intersection):
        result = original_inside(verifier_module, first, second, intersection)
        recorder.record("inside_both_witnesses", {
            "bridge_identity": list(first.key),
            "target_identity": list(second.key),
            "witness": _jsonable(result),
        })
        return result

    original_pair = verifier.audit_surface_pair
    def wrapped_pair(*args, **kwargs):
        result = original_pair(*args, **kwargs)
        recorder.record("port_interface_audit", {
            "declared_interface_triangle_pairs":
                _jsonable(result.get("declared_interface_triangle_pairs", [])),
            "unexpected_triangle_pairs":
                _jsonable(result.get("unexpected_triangle_pairs", [])),
            "passed": result.get("passed"),
            "status": result.get("status"),
            "triangle_pair_count": result.get("triangle_pair_count"),
            "result_keys": sorted(result),
        })
        return result

    replacements = [
        (gate, "_section_loops", wrapped_section),
        (gate, "_section_comparison", wrapped_comparison),
        (gate, "_validated_station_spec", wrapped_station),
        (gate, "_transport_material_station", wrapped_transport),
        (gate, "_make_manifold", wrapped_make),
        (gate, "_exact_inside_witness", wrapped_inside),
        (verifier, "audit_surface_pair", wrapped_pair),
    ]
    for module, name, replacement in replacements:
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def restore():
        for module, name, original in originals:
            setattr(module, name, original)
    return restore


def run_diagnostic() -> dict:
    require(PREFLIGHT_PATH.is_file(), "run the no-geometry preflight first")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    require(preflight.get("status") == "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
            "diagnostic preflight status is not valid")
    code_path = Path(__file__).resolve()
    require(preflight.get("diagnostic_code_sha256") == sha_file(code_path),
            "diagnostic source changed after preflight")
    for raw_path, expected in preflight.get("runtime_owner_code_sha256", {}).items():
        path = Path(raw_path)
        require(path.is_file() and sha_file(path) == expected,
                "runtime owner code changed after preflight: " + raw_path)
    require(preflight.get("replay_scope") == REPLAY_SCOPE
            and preflight.get("source_compiler_invoked") is False,
            "single-port replay scope changed after preflight")
    require(not DIAGNOSTIC_DIR.exists(), "retain diagnostic outputs and use a new directory")

    from . import resting_brachiocephalic_geometry_gate as gate
    from . import resting_brachiocephalic_source007 as source007
    from .resting_pleura_proxy import _parse_payload

    started = time.perf_counter()
    recorded_at = datetime.now(UTC).isoformat()
    DIAGNOSTIC_DIR.mkdir(parents=True)
    progress_path = DIAGNOSTIC_DIR / "progress.json"
    report_path = DIAGNOSTIC_DIR / "port-525-469-section-diagnostic.json"
    recorder = Recorder(progress_path)
    report = {
        "schema": "numi.human.brachiocephalic-port-525-469-section-diagnostic.v2",
        "status": "RUNNING_SINGLE_FROZEN_PORT_DIAGNOSTIC",
        "recorded_start_utc": recorded_at,
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "source007_failure_report_path": str(RUN_REPORT),
        "source007_failure_report_sha256": RUN_REPORT_SHA256,
        "candidate_payload_sha256": preflight["candidate007_files_sha256"]["resting-thorax.nhanatomy"],
        "preparation_report_sha256": preflight["prep007_files_sha256"][
            "brachiocephalic-common-field-preparation.json"],
        "port": {"bridge": list(BRIDGE_KEY), "target": list(TARGET_KEY)},
        "source_or_helper_changed": False,
        "other_ports_or_targets_evaluated": False,
        "replay_scope": REPLAY_SCOPE,
        "source_compiler_invoked": False,
        "all_target_checks_invoked": False,
        "native_or_gpu_work_performed": False,
        "trace": recorder.data,
    }
    recorder.bind(report)
    _write_json(progress_path, report)
    restore = None
    try:
        run = json.loads(RUN_REPORT.read_text())
        require(run.get("failed_stage") == "six-port-overlap-and-sections"
                and run.get("six_port_source_geometry", {}).get(
                    "first_failed_port", {}).get("target") == [51011, 469],
                "diagnostic target is not the frozen first source-007 failure")
        require(sha_file(RUN_REPORT) == RUN_REPORT_SHA256,
                "source-007 failure report changed after preflight")
        candidate_payload = CANDIDATE_DIR / "resting-thorax.nhanatomy"
        candidate_receipt = PREPARATION_DIR / "resting-anatomy-receipt.json"
        prep_report = PREPARATION_DIR / "brachiocephalic-common-field-preparation.json"
        require(sha_file(candidate_payload) == preflight["candidate007_files_sha256"][
                    "resting-thorax.nhanatomy"]
                and sha_file(candidate_receipt) == preflight["prep007_files_sha256"][
                    "resting-anatomy-receipt.json"]
                and sha_file(prep_report) == preflight["prep007_files_sha256"][
                    "brachiocephalic-common-field-preparation.json"],
                "emitted candidate or preparation bytes changed")
        _, _, rows = _parse_payload(candidate_payload.read_bytes())
        require(525 in rows and 10 in rows, "candidate lacks bridge525 or SVC10")
        av = source007._load_central_p6()
        capture_manifest = json.loads(CAPTURE_MANIFEST.read_text())
        captured, identity, capture_meta = source007._step0_source(capture_manifest, av)
        require(identity.get("pack_sha256") == preflight["source0_pack_sha256"]
                and identity.get("capture_receipt_sha256") == preflight["source0_receipt_sha256"],
                "source-0 accepted capture changed after diagnostic preflight")
        transform_f32 = source007._load_predictor_transform()
        body = capture_meta["body20"]
        bridge_xyz = transform_f32(
            np.asarray(body["quaternion_xyzw"], dtype="<f4"),
            np.asarray(body["position_m"], dtype="<f4"),
            np.ascontiguousarray(rows[525]["vertices6"][:, :3], dtype="<f4"))
        bridge_faces = np.ascontiguousarray(rows[525]["faces"], dtype="<u4")
        bridge_surface = (bridge_xyz, bridge_faces, None,
                          np.arange(len(bridge_xyz), dtype=np.uint64))
        target_surface = captured[TARGET_KEY]
        port_map, attribution, stationarity = source007._read_port_pairs()
        frozen_pairs = port_map[(BRIDGE_KEY, TARGET_KEY)]
        require(len(frozen_pairs) > 0, "frozen source port face-pair set is empty")
        require(attribution.get("candidate107_payload", {}).get("contains_FJ3482") is True
                and stationarity.get("status") == "PASS_SOURCE_PORT_SUPPORT_FIXED",
                "frozen source attribution/support changed")
        report["input_geometry"] = {
            "bridge_positions_sha256": _array_hash(bridge_xyz, "<f4"),
            "bridge_faces_sha256": _array_hash(bridge_faces, "<u4"),
            "target_positions_sha256": _array_hash(target_surface[0], "<f4"),
            "target_faces_sha256": _array_hash(target_surface[1], "<u4"),
            "bridge_source_vertex_count": len(bridge_xyz),
            "bridge_source_face_count": len(bridge_faces),
            "target_vertex_count": len(target_surface[0]),
            "target_face_count": len(target_surface[1]),
            "frozen_face_pair_count": len(frozen_pairs),
            "frozen_face_pairs_sha256": _json_hash(frozen_pairs),
            "body20_pose": _jsonable(body),
            "target_capture_pack_sha256": identity["pack_sha256"],
            "target_capture_receipt_sha256": identity["capture_receipt_sha256"],
            "bridge_and_target_are_same_as_source007_rigid_only_screen": True,
        }
        report["status"] = "RUNNING_PORT_525_469_WITH_SECTION_WITNESS_CAPTURE"
        _write_json(progress_path, report)
        manifold = source007._load_retained_manifold()
        observed_manifold = _recording_manifold_proxy(manifold, recorder)
        restore = install_recording_wrappers(gate, av, recorder)
        try:
            result = gate.audit_brachiocephalic_port_pair_continuity(
                av, observed_manifold,
                bridge_key=BRIDGE_KEY, bridge_surface=bridge_surface,
                target_key=TARGET_KEY, target_surface=target_surface,
                frozen_face_pairs=frozen_pairs,
                baseline_bridge_surface=bridge_surface,
                baseline_target_surface=target_surface,
            )
            report["helper_result"] = _jsonable(result)
            report["status"] = "DIAGNOSTIC_HELPER_RETURNED"
        except Exception as error:  # noqa: BLE001 - persist diagnostic error before return
            report["helper_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            report["failed_comparison_stage"] = (
                recorder.data["section_comparisons"][-1].get("stage")
                if recorder.data["section_comparisons"] else None)
            report["status"] = "DIAGNOSTIC_CAPTURED_HELPER_REJECTION"
        finally:
            restore()
            restore = None
        report["trace"] = recorder.data
        report["section_failure_interpretation"] = (
            "An invalid or collapsed computed section is a limitation of this inferred station/section method unless supported by other geometry evidence; it does not by itself prove that the source vessel physically collapses.")
        report["source_or_helper_changed"] = False
        report["other_ports_or_targets_evaluated"] = False
        report["native_or_gpu_work_performed"] = False
    except Exception as error:  # noqa: BLE001 - persist diagnostic error before return
        if restore is not None:
            restore()
        report["status"] = "DIAGNOSTIC_ABORTED_BEFORE_PORT_RESULT"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["trace"] = recorder.data
    report["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["candidate_payload_sha256_after"] = sha_file(
        CANDIDATE_DIR / "resting-thorax.nhanatomy")
    report["preparation_report_sha256_after"] = sha_file(
        PREPARATION_DIR / "brachiocephalic-common-field-preparation.json")
    report["source_geometry_mutated"] = False
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha_file(report_path)
    _write_json(progress_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-diagnostic", action="store_true",
                      help="evaluate only frozen bridge 525 against target 469")
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = write_preflight()
        print(json.dumps({
            "status": result["status"],
            "report": result["report_path"],
            "report_sha256": result["report_sha256"],
            "geometry_predicates_run": result["geometry_predicates_run"],
        }, sort_keys=True))
        return 0
    result = run_diagnostic()
    print(json.dumps({
        "status": result["status"],
        "report": result.get("report_path"),
        "report_sha256": result.get("report_sha256"),
        "failed_comparison_stage": result.get("failed_comparison_stage"),
        "helper_error": result.get("helper_error", result.get("failure")),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }, sort_keys=True))
    return 0 if result.get("status") in {
        "DIAGNOSTIC_CAPTURED_HELPER_REJECTION", "DIAGNOSTIC_HELPER_RETURNED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

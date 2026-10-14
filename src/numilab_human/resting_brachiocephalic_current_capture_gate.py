"""Current accepted-capture vascular junction gate.

This entrypoint consumes the actual accepted MRVPACK2 capture and its bound
anatomy receipt. Historical source-007 collar files are used only as face-patch
ancestry; source-007 capture geometry and predicted poses are never substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as shared_verifier
from . import resting_brachiocephalic_geometry_gate as geometry_gate
from . import resting_brachiocephalic_port_diagnostic as port_diagnostic
from . import resting_brachiocephalic_source007 as source007
from . import resting_pleura_proxy as nhanat_codec

DATA_ROOT = Path(
    "/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378"
)
CAPTURE_DIR = DATA_ROOT / "vein008-forearm474-back158-native-emission-diagnostic-002"
CAPTURE_PACK = CAPTURE_DIR / "native-run/accepted-geometry/step-0.mrvpack"
CAPTURE_RECEIPT = CAPTURE_DIR / "native-run/accepted-geometry/step-0.receipt.json"
ANATOMY_RECEIPT = DATA_ROOT / (
    "forearm474-vein008-back158-integrated-receipt-001/"
    "resting-anatomy-receipt.json"
)
PREPARED_SUCCESSOR_RECEIPT = DATA_ROOT / (
    "forearm474-vein008-back157158-integrated-receipt-001/"
    "resting-anatomy-receipt.json"
)
OUTPUT_DIR = DATA_ROOT / "current-capture-network-gate-001"
PREFLIGHT_PATH = OUTPUT_DIR / "current-capture-preflight-001.json"
PROGRESS_PATH = OUTPUT_DIR / "right-brachiocephalic-svc-port-progress-001.json"
REPORT_PATH = OUTPUT_DIR / "right-brachiocephalic-svc-port-001.json"

CAPTURE_PACK_SHA256 = (
    "ba3e3160c135a3b04b927f7d9a1d56ca7cb8e9f90982b50eebb6a70bb169c523"
)
CAPTURE_RECEIPT_SHA256 = (
    "6deba756ae333e0c4e8a7609d90c8f21aa8a5990c9708971dc7c75f4196e852f"
)
ANATOMY_RECEIPT_SHA256 = (
    "a7795cc7b6bde0eb1fade7fca0d792eb24cc9ac73a29fc3fe4b001b91ff692e5"
)
PREPARED_SUCCESSOR_RECEIPT_SHA256 = (
    "86a227bf20861cd68c9597e91d15fa3fd4d03c696d85c29229fa3f45c373ed31"
)
ANATOMY_PAYLOAD_SHA256 = (
    "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42"
)
FIRST_PORT = ((51011, 526), (51011, 10))
BRIDGE_KEYS = ((51011, 525), (51011, 526))
NETWORK_ROW_IDS = (7, 10, 468, 469, 471, 472, 473, 475, 477, 479,
                   481, 483, 485, 487, 489, 525, 526, 313, 315, 317)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("current-capture brachiocephalic gate: " + message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _git_state() -> dict:
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status_lines": status}


def _file_pins(paths) -> dict[str, str]:
    return {
        str(Path(path).resolve()): sha_file(Path(path))
        for path in sorted(set(map(Path, paths)), key=lambda item: str(item.resolve()))
    }


def _current_package_pins() -> dict[str, str]:
    package_root = Path(shared_verifier.__file__).resolve().parent
    package_files = sorted(package_root.glob("*.py"))
    explicitly_called = (
        Path(__file__).resolve(),
        Path(geometry_gate.__file__).resolve(),
        Path(port_diagnostic.__file__).resolve(),
        Path(source007.__file__).resolve(),
        Path(nhanat_codec.__file__).resolve(),
        package_root / "cardiac_cavity_intersections.py",
        package_root / "torso_anatomy_audit.py",
    )
    return _file_pins((*package_files, *explicitly_called))


def _aabb_overlaps(first_min, first_max, second_min, second_max) -> bool:
    first_min = np.asarray(first_min, dtype=np.float64)
    first_max = np.asarray(first_max, dtype=np.float64)
    second_min = np.asarray(second_min, dtype=np.float64)
    second_max = np.asarray(second_max, dtype=np.float64)
    return bool(np.all(first_min <= second_max) and np.all(second_min <= first_max))


def _surface_inventory(surfaces: dict, primitive_owners: dict) -> list[dict]:
    inventory = []
    for key in sorted(surfaces):
        positions, faces, _normals, source_ids = surfaces[key]
        xyz = np.ascontiguousarray(positions, dtype="<f4")
        tris = np.ascontiguousarray(faces, dtype="<u4")
        ids = np.ascontiguousarray(source_ids, dtype="<u8")
        _require(xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(),
                 f"current capture surface {key} has invalid Float32 positions")
        _require(tris.ndim == 2 and tris.shape[1] == 3 and len(tris) > 0,
                 f"current capture surface {key} has invalid triangle indices")
        owner_rows = primitive_owners.get(key, [])
        inventory.append({
            "typed_identity": list(key),
            "vertex_count": len(xyz),
            "triangle_count": len(tris),
            "captured_global_vertex_id_min": int(ids.min()),
            "captured_global_vertex_id_max": int(ids.max()),
            "captured_global_vertex_ids_sha256": _sha_bytes(ids.tobytes()),
            "positions_world_f32_sha256": _sha_bytes(xyz.tobytes()),
            "faces_u32_sha256": _sha_bytes(tris.tobytes()),
            "aabb_world_m": {
                "min": xyz.min(axis=0).astype(float).tolist(),
                "max": xyz.max(axis=0).astype(float).tolist(),
            },
            "primitive_owner_records": owner_rows,
        })
    return inventory


def _primitive_owner_inventory(pack_path: Path, keys: set[tuple[int, int]]) -> dict:
    sections = shared_verifier._pack_sections(pack_path)
    raw, count, stride = sections[4]
    _require(stride == 64 and len(raw) == count * stride,
             "accepted pack primitive records have unexpected ABI")
    primitives = np.frombuffer(raw, dtype="<u4").reshape(-1, 16)
    result = {key: [] for key in keys}
    for primitive_index, primitive in enumerate(primitives):
        key = (int(primitive[4]), int(primitive[7]))
        if key not in result:
            continue
        result[key].append({
            "primitive_index": primitive_index,
            "first_index": int(primitive[0]),
            "index_count": int(primitive[1]),
            "primitive_owner_body_index": int(primitive[6]),
        })
    return result


def _source_identity(source_map: dict, stable_id: int) -> dict:
    raw = source_map.get(str(stable_id))
    _require(isinstance(raw, dict), f"final receipt source_id_map lacks row {stable_id}")
    owner = raw.get("source_owner_metadata", {})
    result = {
        "source_member": raw.get("source_member"),
        "source_member_sha256": raw.get("source_sha256"),
        "name": raw.get("name"),
        "provider": raw.get("provider"),
        "body_index": raw.get("body_index"),
        "layer": raw.get("layer"),
        "source_owner_metadata": owner,
        "vascular_compartment_id_present": "vascular_compartment_id" in raw,
        "vascular_compartment_id": raw.get("vascular_compartment_id"),
        "repair": raw.get("repair"),
    }
    return result


def _source_row_record(rows: dict, stable_id: int) -> dict:
    row = rows.get(stable_id)
    _require(isinstance(row, dict), f"current ABI5 payload lacks source row {stable_id}")
    record = tuple(map(int, row.get("record", ())))
    _require(len(record) == 8 and record[5] == stable_id,
             f"current ABI5 record identity is invalid for row {stable_id}")
    return {
        "abi5_record_fields": list(record),
        "body_index": int(row["body_index"]),
        "layer": int(row["layer"]),
        "flags": int(row["flags"]),
        "local_vertex_count": int(row["record"][2]),
        "local_triangle_count": int(row["record"][4]) // 3,
        "vertices6_sha256": _sha_bytes(
            np.ascontiguousarray(row["vertices6"], dtype="<f4").tobytes()),
        "faces_sha256": _sha_bytes(
            np.ascontiguousarray(row["faces"], dtype="<u4").tobytes()),
    }


def _frozen_port_inputs() -> tuple[dict, dict]:
    pairs, attribution, stationarity = source007._read_port_pairs()
    _require(set(pairs) == set(source007.PORT_KEYS) and FIRST_PORT in pairs,
             "frozen source collar ancestry does not declare the requested first port")
    pin_paths = (
        source007.COLLAR_ANCESTRY,
        source007.COLLAR_ATTRIBUTION,
        source007.COLLAR_REPLAY,
        source007.COLLAR_STATIONARITY,
    )
    input_pins = _file_pins(pin_paths)
    expected_pins = {
        str(Path(source007.COLLAR_ANCESTRY).resolve()): source007.COLLAR_SHA["ancestry"],
        str(Path(source007.COLLAR_ATTRIBUTION).resolve()): source007.COLLAR_SHA["attribution"],
        str(Path(source007.COLLAR_REPLAY).resolve()): source007.COLLAR_SHA["replay"],
        str(Path(source007.COLLAR_STATIONARITY).resolve()): source007.COLLAR_SHA["stationarity"],
    }
    _require(input_pins == expected_pins, "frozen source collar ancestry bytes changed")
    return pairs, {
        "input_sha256": input_pins,
        "provenance_only": True,
        "historical_source_capture_geometry_used": False,
        "attribution_status": attribution.get("status"),
        "stationarity_status": stationarity.get("status"),
        "all_six_typed_interfaces": [
            {
                "bridge": list(pair[0]),
                "target": list(pair[1]),
                "face_pair_count": len(pairs[pair]),
                "ordered_face_pairs_sha256": _sha_bytes(
                    np.ascontiguousarray(pairs[pair], dtype="<i8").tobytes()),
            }
            for pair in source007.PORT_KEYS
        ],
    }


def _load_current_inputs() -> dict:
    for path, expected, label in (
        (CAPTURE_PACK, CAPTURE_PACK_SHA256, "accepted capture pack"),
        (CAPTURE_RECEIPT, CAPTURE_RECEIPT_SHA256, "accepted capture receipt"),
        (ANATOMY_RECEIPT, ANATOMY_RECEIPT_SHA256,
         "anatomy receipt actually invoked by the accepted capture"),
        (PREPARED_SUCCESSOR_RECEIPT, PREPARED_SUCCESSOR_RECEIPT_SHA256,
         "prepared successor anatomy receipt used only for source-term comparison"),
    ):
        _require(path.is_file() and sha_file(path) == expected,
                 f"pinned {label} is absent or changed: {path.resolve()}")
    anatomy_receipt = json.loads(ANATOMY_RECEIPT.read_text())
    successor_receipt = json.loads(PREPARED_SUCCESSOR_RECEIPT.read_text())
    payload = anatomy_receipt.get("payload", {})
    successor_payload = successor_receipt.get("payload", {})
    actual_source_map = anatomy_receipt.get("provenance", {}).get("source_id_map", {})
    successor_source_map = successor_receipt.get("provenance", {}).get("source_id_map", {})
    relevant_ids = sorted(set(NETWORK_ROW_IDS) & {
        int(value) for value in actual_source_map
    } & {int(value) for value in successor_source_map})
    relevant_rows_equal = {
        str(stable_id): actual_source_map[str(stable_id)]
        == successor_source_map[str(stable_id)]
        for stable_id in relevant_ids
    }
    required_compare_ids = {10, 469, 471, 473, 475, 477, 479, 481, 483, 485,
                            487, 489, 525, 526}
    _require(required_compare_ids.issubset(relevant_ids)
             and all(relevant_rows_equal.values())
             and payload.get("path") == successor_payload.get("path")
             and payload.get("sha256") == successor_payload.get("sha256")
             and payload.get("sha256") == ANATOMY_PAYLOAD_SHA256,
             "prepared successor changes a route-relevant source binding or NHA payload")
    successor_source_comparison = {
        "status": "ALL_PRESENT_NETWORK_ROW_SOURCE_TERMS_EQUAL",
        "actual_invoked_receipt_path": str(ANATOMY_RECEIPT.resolve()),
        "actual_invoked_receipt_sha256": ANATOMY_RECEIPT_SHA256,
        "prepared_successor_receipt_path": str(PREPARED_SUCCESSOR_RECEIPT.resolve()),
        "prepared_successor_receipt_sha256": PREPARED_SUCCESSOR_RECEIPT_SHA256,
        "actual_payload_sha256": payload.get("sha256"),
        "prepared_successor_payload_sha256": successor_payload.get("sha256"),
        "compared_network_row_ids": relevant_ids,
        "per_row_source_identity_equal": relevant_rows_equal,
        "comparison_scope": "receipt source_id_map entries and exact NHA payload identity only",
        "prepared_successor_used_as_capture_source": False,
    }
    payload_path = Path(payload.get("path", "")).resolve()
    _require(payload_path.is_file()
             and payload.get("sha256") == ANATOMY_PAYLOAD_SHA256
             and sha_file(payload_path) == ANATOMY_PAYLOAD_SHA256
             and payload.get("surface_count") == len(
                 anatomy_receipt.get("provenance", {}).get("source_id_map", {})),
             "final ABI5 receipt does not bind the pinned NHA payload/source map")
    source_map = anatomy_receipt["provenance"]["source_id_map"]
    header, records, rows = nhanat_codec._parse_payload(payload_path.read_bytes())
    _require(len(records) == int(payload["surface_count"])
             and set(rows) == {int(value) for value in source_map},
             "decoded final ABI5 rows differ from the bound receipt identity inventory")
    _require(int(header[1]) == 5 and int(header[2]) == int(payload["surface_count"]),
             "final appended payload is not the bound ABI5 surface table")

    pairs, collar_provenance = _frozen_port_inputs()
    surfaces, capture_identity = shared_verifier.load_capture(
        CAPTURE_PACK, CAPTURE_RECEIPT)
    _require(
        capture_identity.get("pack_sha256") == CAPTURE_PACK_SHA256
        and capture_identity.get("capture_receipt_sha256") == CAPTURE_RECEIPT_SHA256
        and capture_identity.get("accepted_step") == 0
        and abs(float(capture_identity.get("accepted_time_s", -1.0))) <= 1e-12,
        "current shared verifier did not return the exact accepted step-0 capture identity",
    )
    capture_receipt = json.loads(CAPTURE_RECEIPT.read_text())
    _require(
        capture_receipt.get("physical_endpoint") == "accepted"
        and capture_receipt.get("common_field_source_anatomy_payload_sha256")
        == ANATOMY_PAYLOAD_SHA256,
        "actual accepted capture is not bound to the current final NHA payload",
    )
    source_map_keys = {int(value) for value in source_map}
    _require({526, 10}.issubset(source_map_keys)
             and FIRST_PORT[0] in surfaces and FIRST_PORT[1] in surfaces,
             "actual capture or final receipt lacks typed right bridge/SVC identities")
    for stable_id in (525, 526):
        ident = _source_identity(source_map, stable_id)
        _require(ident["body_index"] == 20 and ident["layer"] == 2
                 and ident["vascular_compartment_id_present"]
                 and ident["vascular_compartment_id"] == 4,
                 f"bridge {stable_id} lost its explicit passive torso/CVSim-4 binding")
    svc_identity = _source_identity(source_map, 10)
    _require(svc_identity["body_index"] == 20 and svc_identity["layer"] == 2,
             "SVC10 source identity/owner changed")
    # Preserve the SVC record exactly; no compartment field is synthesized here.
    bridge_and_svc_keys = set(source007.REQUIRED_SOURCE_TARGETS) | set(BRIDGE_KEYS)
    owner_inventory = _primitive_owner_inventory(CAPTURE_PACK, bridge_and_svc_keys)
    surface_by_key = {tuple(row["typed_identity"]): row
                      for row in _surface_inventory(surfaces, owner_inventory)}
    for key in (FIRST_PORT[0], FIRST_PORT[1]):
        owners = {row["primitive_owner_body_index"]
                  for row in owner_inventory.get(key, [])}
        _require(owners == {20}, f"actual capture primitive owner is not Body20 for {key}")
        _captured_positions, captured_faces = surfaces[key][:2]
        source_row = rows[key[1]]
        _require(
            len(captured_faces) == len(source_row["faces"])
            and np.array_equal(
                np.ascontiguousarray(captured_faces, dtype="<u4"),
                np.ascontiguousarray(source_row["faces"], dtype="<u4")),
            f"current capture source face order differs from final ABI5 row {key[1]}",
        )
    first_ports = pairs[FIRST_PORT]
    _require(len(first_ports) > 0,
             "the 526-to-SVC10 frozen source collar face ancestry is empty")
    return {
        "anatomy_receipt": anatomy_receipt,
        "payload_path": payload_path,
        "header": header,
        "records": records,
        "rows": rows,
        "source_map": source_map,
        "surfaces": surfaces,
        "capture_identity": capture_identity,
        "capture_receipt": capture_receipt,
        "prepared_successor_source_comparison": successor_source_comparison,
        "port_pairs": pairs,
        "collar_provenance": collar_provenance,
        "surface_inventory": surface_by_key,
        "primitive_owner_inventory": owner_inventory,
    }


def _input_file_pins(context: dict) -> dict:
    paths = (
        CAPTURE_PACK, CAPTURE_RECEIPT, ANATOMY_RECEIPT,
        PREPARED_SUCCESSOR_RECEIPT, context["payload_path"],
        source007.COLLAR_ANCESTRY, source007.COLLAR_ATTRIBUTION,
        source007.COLLAR_REPLAY, source007.COLLAR_STATIONARITY,
    )
    return _file_pins(paths)


def _aabb_candidates(surfaces: dict) -> dict:
    output = {}
    bridges = [key for key in BRIDGE_KEYS if key in surfaces]
    for bridge_key in bridges:
        bridge_positions = np.asarray(surfaces[bridge_key][0], dtype="<f4")
        bmin, bmax = bridge_positions.min(axis=0), bridge_positions.max(axis=0)
        nearby = []
        for target_key in sorted(surfaces):
            if target_key == bridge_key:
                continue
            target_positions = np.asarray(surfaces[target_key][0], dtype="<f4")
            tmin, tmax = target_positions.min(axis=0), target_positions.max(axis=0)
            if _aabb_overlaps(bmin, bmax, tmin, tmax):
                nearby.append({
                    "typed_identity": list(target_key),
                    "aabb_world_m": {
                        "min": tmin.astype(float).tolist(),
                        "max": tmax.astype(float).tolist(),
                    },
                })
        output[str(bridge_key[1])] = nearby
    return output


def build_current_capture_preflight(context: dict) -> dict:
    source_map = context["source_map"]
    rows = context["rows"]
    surfaces = context["surfaces"]
    capture_identity = context["capture_identity"]
    selected_ids = sorted(set(NETWORK_ROW_IDS) & {int(value) for value in source_map})
    row_bindings = {}
    for stable_id in selected_ids:
        row_bindings[str(stable_id)] = {
            "source_identity": _source_identity(source_map, stable_id),
            "source_nha_row": _source_row_record(rows, stable_id),
        }
    target_inventory = _surface_inventory(
        surfaces, context["primitive_owner_inventory"])
    package_pins = _current_package_pins()
    file_pins = _input_file_pins(context)
    _require(file_pins[str(CAPTURE_PACK.resolve())] == CAPTURE_PACK_SHA256
             and file_pins[str(CAPTURE_RECEIPT.resolve())] == CAPTURE_RECEIPT_SHA256
             and file_pins[str(ANATOMY_RECEIPT.resolve())] == ANATOMY_RECEIPT_SHA256
             and file_pins[str(context["payload_path"])] == ANATOMY_PAYLOAD_SHA256,
             "inspected current-capture or final NHA input hashes changed")
    inventory_bytes = json.dumps(
        target_inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "schema": "numi.human.brachiocephalic.current-capture-preflight.v1",
        "status": "CURRENT_CAPTURE_METADATA_PINNED_NO_GEOMETRY_PREDICATE",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_repository": _git_state(),
        "current_package_code_sha256": package_pins,
        "input_sha256": file_pins,
        "actual_accepted_capture": {
            "pack_path": str(CAPTURE_PACK.resolve()),
            "pack_sha256": capture_identity["pack_sha256"],
            "receipt_path": str(CAPTURE_RECEIPT.resolve()),
            "receipt_sha256": capture_identity["capture_receipt_sha256"],
            "accepted_step": capture_identity["accepted_step"],
            "accepted_time_s": capture_identity["accepted_time_s"],
            "typed_surface_count": len(surfaces),
            "surface_audit_endpoint": context["capture_receipt"].get(
                "surface_audit_endpoint"),
            "common_field_mode": context["capture_receipt"].get(
                "cardiac_geometry_mode"),
            "common_field_source_anatomy_payload_sha256": context["capture_receipt"].get(
                "common_field_source_anatomy_payload_sha256"),
            "body20_pose": next(
                row for row in context["capture_receipt"]["accepted_body_poses"]
                if row["body_index"] == 20),
        },
        "capture_bound_nha_source": {
            "receipt_path": str(ANATOMY_RECEIPT.resolve()),
            "receipt_sha256": ANATOMY_RECEIPT_SHA256,
            "payload_path": str(context["payload_path"]),
            "payload_sha256": ANATOMY_PAYLOAD_SHA256,
            "abi": int(context["header"][1]),
            "surface_count": len(context["records"]),
            "source_id_map_count": len(source_map),
            "selected_row_record_and_identity_bindings": row_bindings,
            "prepared_successor_source_comparison":
                context["prepared_successor_source_comparison"],
            "added_physical_mass_kg": context["anatomy_receipt"].get(
                "provenance", {}).get("passive_brachiocephalic_vein_extension", {}).get(
                    "additional_physical_mass_kg"),
            "additional_blood_state_or_force": context["anatomy_receipt"].get(
                "provenance", {}).get("passive_brachiocephalic_vein_extension", {}).get(
                    "additional_force_or_state"),
            "existing_cvsim_compartment_id": {
                "bridge525": source_map["525"].get("vascular_compartment_id"),
                "bridge526": source_map["526"].get("vascular_compartment_id"),
                "SVC10_field_present": "vascular_compartment_id" in source_map["10"],
                "SVC10_value": source_map["10"].get("vascular_compartment_id"),
                "SVC10_metadata_inferred": False,
            },
        },
        "historical_collar_ancestry": context["collar_provenance"],
        "named_port_scope": {
            "priority_port": {
                "bridge": list(FIRST_PORT[0]),
                "target": list(FIRST_PORT[1]),
                "face_pair_count": len(context["port_pairs"][FIRST_PORT]),
                "face_pairs_sha256": _sha_bytes(
                    np.ascontiguousarray(context["port_pairs"][FIRST_PORT], dtype="<i8").tobytes()),
            },
            "all_six_bridge_ports": context["collar_provenance"]["all_six_typed_interfaces"],
            "not_used_as_all_target_collision_allowlist": True,
        },
        "target_inventory_sha256": _sha_bytes(inventory_bytes),
        "all_actual_capture_surfaces": target_inventory,
        "all_nearby_aabb_targets_by_bridge": _aabb_candidates(surfaces),
        "network_row_ids_available": selected_ids,
        "row_count_from_decoded_source_map": len(source_map),
        "all_target_geometry_predicates_run": False,
        "port_geometry_predicates_run": False,
        "native_or_gpu_work_performed_by_this_gate": False,
        "source007_old_capture_or_pose_used": False,
        "anatomical_admission": False,
    }


def _assert_frozen_preflight(preflight: dict) -> None:
    _require(preflight.get("status")
             == "CURRENT_CAPTURE_METADATA_PINNED_NO_GEOMETRY_PREDICATE",
             "current-capture preflight status is not executable")
    _require(preflight.get("source_repository") == _git_state(),
             "source revision or checkout status changed after preflight")
    for raw_path, expected in preflight.get("current_package_code_sha256", {}).items():
        path = Path(raw_path)
        _require(path.is_file() and sha_file(path) == expected,
                 "current package code changed after preflight: " + raw_path)
    for raw_path, expected in preflight.get("input_sha256", {}).items():
        path = Path(raw_path)
        _require(path.is_file() and sha_file(path) == expected,
                 "current-capture input changed after preflight: " + raw_path)


def run_first_right_svc_port() -> dict:
    _require(not OUTPUT_DIR.exists(),
             "retain existing current-capture gate output; choose a new immutable output directory")
    started = time.perf_counter()
    context = _load_current_inputs()
    preflight = build_current_capture_preflight(context)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json(PREFLIGHT_PATH, preflight)
    preflight_sha256 = sha_file(PREFLIGHT_PATH)
    _assert_frozen_preflight(preflight)
    progress = {
        "schema": "numi.human.brachiocephalic.current-capture-port-gate.v1",
        "status": "RUNNING_ONLY_526_TO_SVC10",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": preflight_sha256,
        "port": {"bridge": list(FIRST_PORT[0]), "target": list(FIRST_PORT[1])},
        "other_ports_evaluated": False,
        "all_target_geometry_predicate_run": False,
        "historical_source007_capture_geometry_used": False,
        "native_or_gpu_work_performed": False,
        "trace": {},
    }
    recorder = port_diagnostic.Recorder(PROGRESS_PATH)
    recorder.bind(progress)
    _write_json(PROGRESS_PATH, progress)
    restore = None
    try:
        _require(FIRST_PORT in context["port_pairs"],
                 "frozen source collar map lacks the first named port")
        manifold = source007._load_retained_manifold()
        observed_manifold = port_diagnostic._recording_manifold_proxy(manifold, recorder)
        restore = port_diagnostic.install_recording_wrappers(
            geometry_gate, shared_verifier, recorder)
        bridge_surface = context["surfaces"][FIRST_PORT[0]]
        target_surface = context["surfaces"][FIRST_PORT[1]]
        progress["actual_port_geometry"] = {
            "bridge_positions_sha256": _sha_bytes(
                np.ascontiguousarray(bridge_surface[0], dtype="<f4").tobytes()),
            "bridge_faces_sha256": _sha_bytes(
                np.ascontiguousarray(bridge_surface[1], dtype="<u4").tobytes()),
            "target_positions_sha256": _sha_bytes(
                np.ascontiguousarray(target_surface[0], dtype="<f4").tobytes()),
            "target_faces_sha256": _sha_bytes(
                np.ascontiguousarray(target_surface[1], dtype="<u4").tobytes()),
            "frozen_face_pair_count": len(context["port_pairs"][FIRST_PORT]),
            "baseline_geometry_source": (
                "same exact current accepted-capture material surfaces; this is a "
                "same-source section sanity comparison, not independent physical calibration"),
            "capture_step": 0,
            "capture_pack_sha256": CAPTURE_PACK_SHA256,
            "capture_receipt_sha256": CAPTURE_RECEIPT_SHA256,
        }
        progress["status"] = "RUNNING_EXACT_INTERFACE_AND_3D_SOLID_CALIBRE_GATE"
        _write_json(PROGRESS_PATH, progress)
        result = geometry_gate.audit_brachiocephalic_port_pair_continuity(
            shared_verifier,
            observed_manifold,
            bridge_key=FIRST_PORT[0],
            bridge_surface=bridge_surface,
            target_key=FIRST_PORT[1],
            target_surface=target_surface,
            frozen_face_pairs=context["port_pairs"][FIRST_PORT],
            baseline_bridge_surface=bridge_surface,
            baseline_target_surface=target_surface,
        )
        progress["port_result"] = result
        progress["status"] = (
            "RIGHT_BRACHIOCEPHALIC_TO_SVC10_PORT_PASSED_UNADMITTED"
            if result.get("passed") is True
            else "RIGHT_BRACHIOCEPHALIC_TO_SVC10_PORT_REJECTED")
    except Exception as error:  # noqa: BLE001 - preserve gate partial trace
        progress["status"] = "RIGHT_BRACHIOCEPHALIC_TO_SVC10_PORT_REJECTED"
        progress["port_failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if restore is not None:
            restore()
    progress["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    progress["elapsed_seconds"] = time.perf_counter() - started
    progress["trace"] = recorder.data
    progress["other_ports_evaluated"] = False
    progress["all_target_geometry_predicate_run"] = False
    progress["source007_old_capture_or_pose_used"] = False
    progress["native_or_gpu_work_performed"] = False
    _write_json(REPORT_PATH, progress)
    progress["report_path"] = str(REPORT_PATH)
    progress["report_sha256"] = sha_file(REPORT_PATH)
    _write_json(PROGRESS_PATH, progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-526-to-svc10", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_526_to_svc10:
        parser.error("use --run-526-to-svc10")
    report = run_first_right_svc_port()
    print(json.dumps({
        "status": report["status"],
        "preflight_path": report.get("preflight_path"),
        "preflight_sha256": report.get("preflight_sha256"),
        "report_path": report.get("report_path"),
        "report_sha256": report.get("report_sha256"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "port_failure": report.get("port_failure"),
    }, sort_keys=True))
    return 0 if report.get("port_result", {}).get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-emit the FJ2217 source cap into the existing NHANAT ABI5 receipt.

This narrowly restores source face 1858 by exact existing Float32 aliases. It
changes no positions, normals, owner maps, circulation bindings, or physical
state. It does not run geometry predicates or native simulation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import resting_anatomy_verify as anatomy_verify
from . import resting_intercostal_mesh_arrangement as abi5_owner
from . import resting_pleura_proxy as codec
from . import resting_vessel_owner_blend as cap_owner

DATA_ROOT = Path(
    "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378"
)
BASE_RECEIPT = DATA_ROOT / (
    "forearm474-vein008-back158-integrated-receipt-001/resting-anatomy-receipt.json"
)
BASE_RECEIPT_SHA256 = (
    "a7795cc7b6bde0eb1fade7fca0d792eb24cc9ac73a29fc3fe4b001b91ff692e5"
)
BASE_PAYLOAD_SHA256 = (
    "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42"
)
CANDIDATE_PAYLOAD = DATA_ROOT / "axillary-477-cap-reinstatement-001/resting-thorax.nhanatomy"
CANDIDATE_PAYLOAD_SHA256 = (
    "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
)
CANDIDATE_REPORT = DATA_ROOT / "axillary-477-cap-reinstatement-001/source-candidate-report.json"
CANDIDATE_REPORT_SHA256 = (
    "d3b51b16720b3ca6e5afe8af4edc057dcd00bb29d22da9940efd2079d45a7740"
)
CAPTURE_PACK = DATA_ROOT / (
    "vein008-forearm474-back158-native-emission-diagnostic-002/"
    "native-run/accepted-geometry/step-0.mrvpack"
)
CAPTURE_PACK_SHA256 = (
    "ba3e3160c135a3b04b927f7d9a1d56ca7cb8e9f90982b50eebb6a70bb169c523"
)
CAPTURE_RECEIPT = CAPTURE_PACK.with_name("step-0.receipt.json")
CAPTURE_RECEIPT_SHA256 = (
    "6deba756ae333e0c4e8a7609d90c8f21aa8a5990c9708971dc7c75f4196e852f"
)
OUTPUT_DIR = DATA_ROOT / "axillary-477-cap-reinstatement-successor-006"
PRIOR_PREPARATIONS = {
    "partial_transform_owner_path_failure": DATA_ROOT / "axillary-477-cap-reinstatement-successor-002",
    "owner_projection_wrong_record_offset": DATA_ROOT / "axillary-477-cap-reinstatement-successor-003",
    "owner_projection_nha_offset_assumption": DATA_ROOT / "axillary-477-cap-reinstatement-successor-004",
    "all_alias_class_spread_included_inactive_unused_source_slots": DATA_ROOT / "axillary-477-cap-reinstatement-successor-005",
}
SOURCE_FACE = (930, 1066, 1063)
ALIAS_FACE = (1068, 1065, 926)
ALIAS_CLASSES = ((930, 1067, 1068), (1066, 929, 1065), (1063, 926))
CARRIER_SOURCE_INDEX = 6
CARRIER_PAYLOAD_SHA256 = (
    "f77cf7d6da5677178e858f68abe92d7272925f5a783eb8739eba01ad14027393"
)
TRANSFORM_OWNER = Path(__file__).resolve().parents[2] / (
    "scripts/predict_brachiocephalic_captured_svc_poses.py"
)
TRANSFORM_OWNER_SHA256 = (
    "017a90ed6c9eadbdfda9c4d4b31a4966ca10c8dd1c79577686635a296145f7d2"
)
HISTORICAL_REPORTS = {
    "failed_indexed_retirement": (
        DATA_ROOT / "axillary-477-boundary-repair-diagnostic-001/report.json",
        "49dcb9c55d4df6cf177c94c96b8259f504b424a5780d9dcc97f60b6e931760cc",
    ),
    "indexed_vs_exact_quotient_correction": (
        DATA_ROOT / "axillary-477-boundary-repair-diagnostic-001/"
        "alias-repair-comparison-002.json",
        "1738bcab5cab3da9374695931f03347c91d0a76a9d0c16426d1b9dc63bde23cb",
    ),
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("FJ2217 cap successor: " + message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def _code_pin(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha_file(resolved)}


def _load_transform_owner():
    _need(_sha_file(TRANSFORM_OWNER) == TRANSFORM_OWNER_SHA256,
          "registered transform owner changed")
    spec = importlib.util.spec_from_file_location(
        "_fj2217_cap_registered_transform", TRANSFORM_OWNER
    )
    _need(spec is not None and spec.loader is not None,
          "registered transform owner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = getattr(module, "transform_f32", None)
    _need(callable(transform), "registered transform_f32 owner is missing")
    return transform


def _binding_record(carrier: dict, binding_index: int) -> tuple[int, ...]:
    raw = carrier["raw"]
    offset = int(carrier["binding_start"]) + 36 * int(binding_index)
    record = struct.unpack_from("<I8f", raw, offset)
    _need(
        record[1:4] == (0.0, 0.0, 0.0)
        and record[4:7] == (0.0, 0.0, 0.0)
        and record[7] == 1.0
        and record[8] == 1.0,
        "selected passive carrier binding is not its pinned identity core frame",
    )
    return record


def _project_alias_parent_fields(
    base_receipt: dict[str, Any],
    candidate_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    from .passive_attachment_composition import (
        _biceps_row_arrays,
        _read_nhtiss4,
        _row_slices,
    )

    _need(_sha_file(CAPTURE_PACK) == CAPTURE_PACK_SHA256
          and _sha_file(CAPTURE_RECEIPT) == CAPTURE_RECEIPT_SHA256,
          "actual accepted step-0 capture changed")
    capture_receipt = json.loads(CAPTURE_RECEIPT.read_text())
    _need(capture_receipt.get("accepted_step") == 0
          and capture_receipt.get("physical_endpoint") == "accepted",
          "owner projection is not bound to an accepted step-0 capture")
    poses = {
        int(item["body_index"]): item
        for item in capture_receipt.get("accepted_body_poses", [])
    }
    _need(20 in poses and 91 in poses, "accepted capture lacks body 20 or 91")

    parent = base_receipt.get("passive_tissue_parent_bindings", {})
    source_payloads = parent.get("source_payloads", [])
    _need(len(source_payloads) > CARRIER_SOURCE_INDEX,
          "actual receipt lacks the existing joint carrier source")
    carrier_record = source_payloads[CARRIER_SOURCE_INDEX]
    carrier_path = Path(carrier_record["path"])
    if not carrier_path.is_absolute():
        carrier_path = BASE_RECEIPT.parent / carrier_path
    carrier_path = carrier_path.resolve(strict=True)
    _need(_sha_file(carrier_path) == CARRIER_PAYLOAD_SHA256
          == carrier_record.get("sha256"),
          "actual bound source-6 joint carrier changed")

    carrier = _read_nhtiss4(carrier_path)
    records_by_id = {int(record[6]): record for record in carrier["records"]}
    carrier_rows = {
        stable_id: _biceps_row_arrays(
            _row_slices(carrier, records_by_id[stable_id])
        )
        for stable_id in (3, 4)
    }
    parent_row = next(
        item for item in parent.get("anatomy_surfaces", [])
        if int(item.get("stable_id", -1)) == 477
    )
    _need(len(parent_row.get("vertices", [])) == len(candidate_rows[477]["vertices6"]),
          "FJ2217 parent map does not cover its exact material vertex slots")

    transform_f32 = _load_transform_owner()
    selected_body_poses = {
        core: (
            np.asarray(poses[core]["quaternion_xyzw"], dtype="<f4"),
            np.asarray(poses[core]["position_m"], dtype="<f4"),
        )
        for core in (20, 91)
    }
    surfaces, identity = anatomy_verify.load_capture(CAPTURE_PACK, CAPTURE_RECEIPT)
    _need(identity.get("pack_sha256") == CAPTURE_PACK_SHA256
          and identity.get("capture_receipt_sha256") == CAPTURE_RECEIPT_SHA256
          and identity.get("accepted_step") == 0,
          "shared capture decoder did not load the pinned actual capture")
    actual_surface = surfaces[(51011, 477)]
    actual_ids = np.asarray(actual_surface[3], dtype="<u8")
    base_payload = Path(base_receipt["payload"]["path"]).resolve(strict=True).read_bytes()
    _, _, base_rows = codec._parse_payload(base_payload)
    base_row = base_rows[477]
    used_local_ids = np.unique(np.asarray(base_row["faces"], dtype="<u4").reshape(-1))
    _need(len(used_local_ids) == len(actual_ids),
          "captured FJ2217 vertex count differs from referenced source slots")
    local_to_capture_index = {
        int(local_id): index for index, local_id in enumerate(used_local_ids)
    }
    expected_capture_faces = np.asarray(
        [[local_to_capture_index[int(vertex)] for vertex in face]
         for face in base_row["faces"]],
        dtype="<u4",
    )
    actual_faces = np.ascontiguousarray(actual_surface[1], dtype="<u4")
    _need(expected_capture_faces.tobytes() == actual_faces.tobytes(),
          "captured FJ2217 face array does not match exact referenced-slot order")
    base_used_ids = set(map(int, used_local_ids))
    candidate_used_ids = set(map(
        int, np.unique(np.asarray(candidate_rows[477]["faces"], dtype="<u4").reshape(-1))
    ))
    per_vertex_projection = {}
    classes = []

    for alias_class in ALIAS_CLASSES:
        members = []
        for local_id in alias_class:
            terms = parent_row["vertices"][local_id]
            resolved_terms = []
            projected_position = np.zeros(3, dtype=np.float64)
            projected_normal = np.zeros(3, dtype=np.float64)
            compiled_weight_sum = 0.0
            for term in terms:
                source_index, carrier_surface, carrier_vertex, coefficient = term
                _need(int(source_index) == CARRIER_SOURCE_INDEX
                      and int(carrier_surface) in carrier_rows
                      and int(carrier_vertex) == local_id,
                      f"alias {local_id} has an unexpected parent/carrier identity")
                row = carrier_rows[int(carrier_surface)]
                weights = np.asarray(row["weights"][local_id], dtype="<f4")
                columns = np.flatnonzero(weights > 0)
                _need(len(columns) == 1 and weights[columns[0]] == np.float32(1.0),
                      f"alias {local_id} carrier term is not one exact core influence")
                column = int(columns[0])
                binding_index = int(row["binding_indices"][local_id, column])
                binding = _binding_record(carrier, binding_index)
                body_index = int(binding[0])
                _need(body_index in selected_body_poses,
                      f"alias {local_id} resolved to an unpinned core body")
                q, t = selected_body_poses[body_index]
                local6 = np.asarray(row["vertices6"][local_id], dtype="<f4")
                local_xyz = local6[:3]
                local_normal = local6[3:]
                world_xyz = transform_f32(q, t, local_xyz.reshape(1, 3))[0]
                world_normal = transform_f32(
                    q, np.zeros(3, dtype="<f4"), local_normal.reshape(1, 3)
                )[0]
                normal_length = float(np.linalg.norm(world_normal.astype(np.float64)))
                _need(np.isfinite(world_xyz).all() and np.isfinite(world_normal).all()
                      and normal_length > 0,
                      f"alias {local_id} resolved parent point/normal is invalid")
                world_normal = np.asarray(world_normal / np.float32(normal_length), dtype="<f4")
                coefficient_f32 = np.float32(float(coefficient) * float(weights[column]))
                _need(np.isfinite(coefficient_f32) and coefficient_f32 > 0,
                      f"alias {local_id} compiled positive weight is invalid")
                projected_position += world_xyz.astype(np.float64) * float(coefficient_f32)
                projected_normal += world_normal.astype(np.float64) * float(coefficient_f32)
                compiled_weight_sum += float(coefficient_f32)
                resolved_terms.append({
                    "source_payload_index": int(source_index),
                    "carrier_surface_id": int(carrier_surface),
                    "carrier_vertex_id": int(carrier_vertex),
                    "core_body_index": body_index,
                    "carrier_binding_index": binding_index,
                    "parent_coefficient_f64": float(coefficient),
                    "carrier_weight_f32": float(weights[column]),
                    "compiled_product_weight_f32": float(coefficient_f32),
                    "compiled_product_weight_f32_hex": coefficient_f32.tobytes().hex(),
                    "carrier_local_xyz_f32": local_xyz.astype(float).tolist(),
                    "carrier_local_normal_f32": local_normal.astype(float).tolist(),
                    "carrier_local_xyz_sha256": _sha_bytes(local_xyz.tobytes()),
                    "carrier_local_normal_sha256": _sha_bytes(local_normal.tobytes()),
                    "accepted_body_pose_q_f32": q.astype(float).tolist(),
                    "accepted_body_pose_t_m_f32": t.astype(float).tolist(),
                    "transformed_parent_xyz_m_f32": world_xyz.astype(float).tolist(),
                    "transformed_parent_normal_f32": world_normal.astype(float).tolist(),
                })
            normal_length = float(np.linalg.norm(projected_normal))
            _need(abs(compiled_weight_sum - 1.0) <= 2.0 ** -23
                  and normal_length > 0,
                  f"alias {local_id} resolved owner field is not a positive unit partition")
            projected_xyz_f32 = np.asarray(projected_position, dtype="<f4")
            projected_normal_f32 = np.asarray(projected_normal / normal_length, dtype="<f4")
            actual_index = local_to_capture_index.get(local_id)
            actual = None
            if actual_index is not None:
                global_id = int(actual_ids[actual_index])
                actual_xyz = np.asarray(actual_surface[0][actual_index], dtype="<f4")
                actual_normal = np.asarray(actual_surface[2][actual_index], dtype="<f4")
                position_error_mm = float(
                    np.linalg.norm(projected_xyz_f32.astype(np.float64)
                                   - actual_xyz.astype(np.float64)) * 1000.0
                )
                dot = float(np.dot(projected_normal_f32.astype(np.float64),
                                   actual_normal.astype(np.float64)))
                angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
                actual = {
                    "global_vertex_id": global_id,
                    "captured_vertex_index": int(actual_index),
                    "captured_world_xyz_m_f32": actual_xyz.astype(float).tolist(),
                    "captured_world_normal_f32": actual_normal.astype(float).tolist(),
                    "position_residual_mm": position_error_mm,
                    "normal_angle_residual_degrees": angle,
                }
            vertex_record = {
                "local_vertex_id": local_id,
                "nha_local_xyz_f32": candidate_rows[477]["vertices6"][local_id, :3].astype(float).tolist(),
                "nha_local_normal_f32": candidate_rows[477]["vertices6"][local_id, 3:].astype(float).tolist(),
                "nha_local_xyz_sha256": _sha_bytes(
                    candidate_rows[477]["vertices6"][local_id, :3].tobytes()
                ),
                "nha_local_normal_sha256": _sha_bytes(
                    candidate_rows[477]["vertices6"][local_id, 3:].tobytes()
                ),
                "ordered_resolved_parent_terms": resolved_terms,
                "compiled_core_weight_sum_f32_terms": compiled_weight_sum,
                "projected_owner_world_xyz_m_f32": projected_xyz_f32.astype(float).tolist(),
                "projected_owner_world_normal_f32": projected_normal_f32.astype(float).tolist(),
                "captured_step0_comparison": actual,
            }
            per_vertex_projection[local_id] = (projected_xyz_f32, projected_normal_f32)
            members.append(vertex_record)

        reference_xyz, reference_normal = per_vertex_projection[alias_class[0]]
        position_error_mm = []
        normal_angle_degrees = []
        for local_id in alias_class[1:]:
            xyz, normal = per_vertex_projection[local_id]
            position_error_mm.append(float(
                np.linalg.norm(xyz.astype(np.float64) - reference_xyz.astype(np.float64)) * 1000.0
            ))
            normal_angle_degrees.append(float(np.degrees(np.arccos(np.clip(
                np.dot(normal.astype(np.float64), reference_normal.astype(np.float64)),
                -1.0, 1.0,
            )))))
        for member in members:
            _need(
                member["nha_local_xyz_sha256"]
                == members[0]["nha_local_xyz_sha256"],
                f"alias class {alias_class} is not exact Float32 coordinate-equivalent",
            )
        active_members = sorted(set(alias_class) & base_used_ids)
        active_reference_xyz, active_reference_normal = per_vertex_projection[
            active_members[0]
        ]
        active_position_spread_mm = max(
            (
                float(np.linalg.norm(
                    per_vertex_projection[local_id][0].astype(np.float64)
                    - active_reference_xyz.astype(np.float64)
                ) * 1000.0)
                for local_id in active_members[1:]
            ),
            default=0.0,
        )
        active_normal_spread_degrees = max(
            (
                float(np.degrees(np.arccos(np.clip(
                    np.dot(
                        per_vertex_projection[local_id][1].astype(np.float64),
                        active_reference_normal.astype(np.float64),
                    ),
                    -1.0,
                    1.0,
                ))))
                for local_id in active_members[1:]
            ),
            default=0.0,
        )
        yield_item = {
            "source_alias_class_local_ids": list(alias_class),
            "members": members,
            "max_projected_owner_position_spread_mm": max(position_error_mm, default=0.0),
            "max_projected_owner_normal_angle_degrees": max(normal_angle_degrees, default=0.0),
            "active_base_local_vertex_ids": sorted(set(alias_class) & base_used_ids),
            "candidate_local_vertex_ids": sorted(set(alias_class) & candidate_used_ids),
            "inactive_original_source_slots": sorted(set(alias_class) - base_used_ids),
            "active_owner_position_spread_mm": active_position_spread_mm,
            "active_owner_normal_angle_spread_degrees": active_normal_spread_degrees,
            "all_members_have_bitwise_equal_source_xyz": True,
            "actual_capture_face_order_exact": True,
            "actual_step0_capture_is_only_a_source-owner comparison": True,
        }
        classes.append(yield_item)

    return {
        "schema": "numi.human.fj2217.cap.alias-owner-projection.v1",
        "status": "SOURCE_OWNER_PROJECTION_REPORTED_NO_ADMISSION",
        "actual_capture": {
            "pack_sha256": CAPTURE_PACK_SHA256,
            "capture_receipt_sha256": CAPTURE_RECEIPT_SHA256,
            "accepted_step": 0,
            "accepted_time_s": capture_receipt.get("accepted_time_s"),
            "identity_from_shared_loader": identity,
            "local_to_capture_vertex_map": "sorted local slots referenced by the exact base FJ2217 face array; verified by byte-exact remapped faces",
            "capture_face_array_sha256": _sha_bytes(actual_faces.tobytes()),
        },
        "source_parent_binding": {
            "source_payload_index": CARRIER_SOURCE_INDEX,
            "source_payload_sha256": CARRIER_PAYLOAD_SHA256,
            "carrier_surface_ids": [3, 4],
            "source_core_body_indices": [20, 91],
            "source_map_schema": "existing passive_tissue_parent_bindings anatomy_surfaces[477].vertices",
            "native_compiled_weight_rule": "float(parent_coefficient_f64 * double(carrier_weight_f32))",
            "projection_arithmetic": "accepted rigid transforms evaluated by the retained Float32 transform_f32 owner; weighted output accumulated in Float64 then cast to Float32 for comparison",
            "parent_maps_mutated": False,
            "source_mesh_or_capture_mutated": False,
        },
        "alias_classes": classes,
        "limitations": [
            "The restored cap is not present in the pinned actual capture; alias vertices are existing captured vertices.",
            "This is a step-0 source-owner projection, not actual native emission of the reinstated cap.",
            "No all-pose, collision, lumen, calibre, anatomy, or native admission was run.",
            "The original source vertex slots remain unused; the restored face uses the exact coordinate-equivalent existing alias IDs.",
        ],
    }


def prepare() -> dict[str, Any]:
    _need(not OUTPUT_DIR.exists(), "successor output already exists; refusing overwrite")
    for path, digest, label in (
        (BASE_RECEIPT, BASE_RECEIPT_SHA256, "actual invoked A779 receipt"),
        (CANDIDATE_PAYLOAD, CANDIDATE_PAYLOAD_SHA256, "cap candidate payload"),
        (CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256, "cap candidate report"),
        (CAPTURE_PACK, CAPTURE_PACK_SHA256, "actual step-0 capture"),
        (CAPTURE_RECEIPT, CAPTURE_RECEIPT_SHA256, "actual step-0 capture receipt"),
        *[(path, digest, name) for name, (path, digest) in HISTORICAL_REPORTS.items()],
    ):
        _need(path.is_file() and _sha_file(path) == digest,
              f"{label} is missing or changed: {path}")

    base_receipt = json.loads(BASE_RECEIPT.read_text())
    candidate_report = json.loads(CANDIDATE_REPORT.read_text())
    _need(base_receipt.get("payload", {}).get("sha256") == BASE_PAYLOAD_SHA256
          and Path(base_receipt["payload"]["path"]).resolve().is_file()
          and _sha_file(Path(base_receipt["payload"]["path"])) == BASE_PAYLOAD_SHA256,
          "actual A779 receipt does not bind the pinned base payload")
    _need(candidate_report.get("base_receipt_sha256") == BASE_RECEIPT_SHA256
          and candidate_report.get("base_payload_sha256") == BASE_PAYLOAD_SHA256
          and candidate_report.get("output_payload_sha256") == CANDIDATE_PAYLOAD_SHA256
          and candidate_report.get("repair", {}).get("source_positions_normals_and_all_parent_terms_unchanged") is True,
          "frozen candidate report does not bind the direct A779 cap operation")

    base_raw = Path(base_receipt["payload"]["path"]).read_bytes()
    candidate_raw = CANDIDATE_PAYLOAD.read_bytes()
    base_header, _base_records, base_rows = codec._parse_payload(base_raw)
    candidate_header, _candidate_records, candidate_rows = codec._parse_payload(candidate_raw)
    _need(len(base_rows) == len(candidate_rows) == 525
          and set(base_rows) == set(candidate_rows)
          and int(base_header[1]) == int(candidate_header[1]) == 5,
          "base/candidate ABI5 row identity or ABI changed")

    expected_route_row = dict(candidate_rows[477])
    expected_route_faces = np.ascontiguousarray(
        candidate_rows[477]["faces"], dtype="<u4"
    ).copy()
    expected_route_faces[1858] = np.asarray(SOURCE_FACE, dtype="<u4")
    expected_route_row["faces"] = expected_route_faces
    replacement, declaration = cap_owner.reinstate_source_face_through_exact_aliases(
        base_rows[477],
        1858,
        np.asarray(SOURCE_FACE, dtype="<u4"),
        source_row=expected_route_row,
    )
    _need(
        codec._record_content_bytes(replacement)
        == codec._record_content_bytes(candidate_rows[477]),
        "direct face reinstatement differs from the frozen cap candidate row",
    )
    payload, reemit_proof = abi5_owner.encode_payload_with_row_replacements(
        base_raw, {477: replacement}
    )
    _need(payload == candidate_raw and _sha_bytes(payload) == CANDIDATE_PAYLOAD_SHA256,
          "existing ABI5 row replacement owner did not reproduce exact candidate bytes")
    cap_proof = cap_owner.validate_local_source_face_reinstatement(
        base_rows[477], candidate_rows[477], declaration, reemit_proof,
        expected_source_row=expected_route_row,
    )

    receipt = copy.deepcopy(base_receipt)
    old_payload_sha = receipt["payload"]["sha256"]
    _need(old_payload_sha == BASE_PAYLOAD_SHA256,
          "source receipt payload identity changed before successor construction")
    OUTPUT_DIR.mkdir(parents=True)
    payload_path = OUTPUT_DIR / "resting-thorax.nhanatomy"
    payload_path.write_bytes(payload)
    payload_sha = _sha_file(payload_path)
    _need(payload_sha == CANDIDATE_PAYLOAD_SHA256, "successor payload write changed bytes")

    payload_info = receipt["payload"]
    payload_info.update(
        path=str(payload_path.resolve()),
        sha256=payload_sha,
        surface_count=int(candidate_header[2]),
        vertex_count=int(candidate_header[3]),
        index_count=int(candidate_header[4]),
    )
    receipt["functional_bindings"]["anatomy_payload_sha256"] = payload_sha
    parent_bindings = receipt["passive_tissue_parent_bindings"]
    _need(parent_bindings.get("anatomy_payload_sha256") == old_payload_sha,
          "passive parent binding is not bound to the invoked A779 payload")
    parent_bindings["anatomy_payload_sha256"] = payload_sha
    owner_map_proof = cap_owner.validate_fj2217_reinstatement_parent_maps_unchanged(
        base_receipt["passive_tissue_parent_bindings"], parent_bindings
    )

    common_field = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    _need(common_field.get("anatomy_payload_sha256") == old_payload_sha,
          "existing common cardiac map is not bound to the invoked A779 payload")
    common_field["anatomy_payload_sha256"] = payload_sha
    common_sidecars = []
    for key in ("map", "domain_boxes", "polynomials"):
        descriptor = common_field[key]
        relative = Path(descriptor["path"])
        _need(not relative.is_absolute() and ".." not in relative.parts,
              f"common-field {key} path is not the existing relative sidecar contract")
        source = (BASE_RECEIPT.parent / relative).resolve(strict=True)
        _need(_sha_file(source) == descriptor["sha256"],
              f"pinned common-field {key} bytes changed")
        destination = OUTPUT_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        _need(_sha_file(destination) == descriptor["sha256"],
              f"copied common-field {key} sidecar changed")
        common_sidecars.append({
            "field": key,
            "relative_path": relative.as_posix(),
            "sha256": descriptor["sha256"],
            "byte_identical_hardlink": True,
        })

    source_map = receipt["provenance"]["source_id_map"]
    repair = copy.deepcopy(source_map["477"].get("repair") or {})
    old_retirement = repair.pop("local_source_face_retirement", None)
    _need(isinstance(old_retirement, dict)
          and "historical_local_source_face_retirement" not in repair
          and "local_source_face_reinstatement" not in repair,
          "active/history FJ2217 source-face declarations are not in expected state")
    repair["historical_local_source_face_retirement"] = old_retirement
    repair["local_source_face_reinstatement"] = declaration
    source_map["477"]["repair"] = repair

    extension = receipt["provenance"]["passive_brachiocephalic_vein_extension"]
    owner_blend = extension["passive_tissue_owner_blend"]
    old_owner_retirement = owner_blend.pop("local_source_face_retirement", None)
    _need(isinstance(old_owner_retirement, dict)
          and "historical_local_source_face_retirement" not in owner_blend,
          "passive owner blend retirement history is not in expected state")
    owner_blend["historical_local_source_face_retirement"] = old_owner_retirement
    owner_blend["local_source_face_reinstatement"] = declaration
    owner_blend["abi5_reinstatement_reemit_proof"] = reemit_proof

    owner_projection = _project_alias_parent_fields(base_receipt, candidate_rows)
    cap_provenance = {
        "schema": "numi.human.axillary-477-cap-reinstatement-successor.v1",
        "status": "SOURCE_ONLY_EXACT_ALIAS_CAP_RESTORED_NOT_ADMITTED",
        "method": "re-emit only stable ID 477 face 1858 through existing NHANAT ABI5 row replacement owner using exact Float32 coordinate aliases",
        "base_receipt_path": str(BASE_RECEIPT.resolve()),
        "base_receipt_sha256": BASE_RECEIPT_SHA256,
        "base_payload_sha256": BASE_PAYLOAD_SHA256,
        "candidate_payload_path": str(CANDIDATE_PAYLOAD.resolve()),
        "candidate_payload_sha256": CANDIDATE_PAYLOAD_SHA256,
        "candidate_source_report_path": str(CANDIDATE_REPORT.resolve()),
        "candidate_source_report_sha256": CANDIDATE_REPORT_SHA256,
        "historical_failed_retirement_and_quotient_correction": {
            name: {"path": str(path.resolve()), "sha256": digest, "historical_only": True}
            for name, (path, digest) in HISTORICAL_REPORTS.items()
        },
        "prior_preparations_preserved_not_used_as_input": {
            name: {"path": str(path.resolve()), "not_used_as_input": True}
            for name, path in PRIOR_PREPARATIONS.items()
        },
        "reinstatement": declaration,
        "row_validation": cap_proof,
        "abi5_reemit_proof": reemit_proof,
        "parent_map_preservation": owner_map_proof,
        "owner_field_alias_projection": owner_projection,
        "preserved_common_field_sidecars": common_sidecars,
        "base_common_field_map_sha256": "0d88db155f7dfe9a93030badf6e570f49050164a7d0aa0ae011065aeada7d3da",
        "base_common_field_domains_sha256": "7652785c9ea164b16ee96f64a3dcf940398af36cfd535a5f9b7a7d6bd2c7a611",
        "base_common_field_polynomials_sha256": "7be4f62996c996c6cfdd49480d27766c8253387bb24aeb504fef73cd553ae699",
        "replaced_stable_ids": [477],
        "unmodified_rows_byte_exact": 524,
        "same_surface_owner_maps_and_source_payload_order": True,
        "same_vascular_compartment_bindings": True,
        "same_mass_geometry_accounting": True,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "anatomical_admission": False,
        "native_confirmation_required": True,
        "source_code_pins": {},
        "scope_limit": "No source self/closure/collision, all-pose, lumen, calibre, connected-network, or native admission is claimed.",
    }
    code_paths = (
        Path(__file__).resolve(),
        Path(cap_owner.__file__).resolve(),
        Path(abi5_owner.__file__).resolve(),
        Path(codec.__file__).resolve(),
        Path(anatomy_verify.__file__).resolve(),
        Path(__import__("numilab_human.passive_tissue_parent_carriers", fromlist=["x"]).__file__).resolve(),
        Path(__import__("numilab_human.passive_attachment_composition", fromlist=["x"]).__file__).resolve(),
        TRANSFORM_OWNER,
    )
    cap_provenance["source_code_pins"] = {
        str(path.resolve()): _sha_file(path) for path in code_paths
    }
    cap_provenance["source_code_pins"]["git_head"] = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    receipt["provenance"]["axillary_477_cap_reinstatement_successor"] = cap_provenance

    receipt_path = OUTPUT_DIR / "resting-anatomy-receipt.json"
    _write_json(receipt_path, receipt)
    report = {
        **cap_provenance,
        "output_payload_path": str(payload_path.resolve()),
        "output_payload_sha256": payload_sha,
        "output_receipt_path": str(receipt_path.resolve()),
        "output_receipt_sha256": _sha_file(receipt_path),
        "output_abi5_counts": {
            "typed_rows": int(candidate_header[2]),
            "vertices": int(candidate_header[3]),
            "indices": int(candidate_header[4]),
            "row477_faces": len(candidate_rows[477]["faces"]),
        },
        "qualification": "source-only and unadmitted",
    }
    _write_json(OUTPUT_DIR / "source-only-report.json", report)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    _need(args.prepare, "pass --prepare to run the frozen source-only composer")
    report = prepare()
    print(json.dumps({
        "status": report["status"],
        "payload_path": report["output_payload_path"],
        "payload_sha256": report["output_payload_sha256"],
        "receipt_path": report["output_receipt_path"],
        "receipt_sha256": report["output_receipt_sha256"],
        "report_path": str(OUTPUT_DIR / "source-only-report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

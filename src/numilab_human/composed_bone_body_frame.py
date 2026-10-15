"""Reframe already-composed rib partition rows into their common body owner."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import model
from .resting_anatomy_interface_patch import parse_payload as parse_nha_payload

_BONE_HEADER = struct.Struct("<8s5I32s")
_BONE_RECORD = struct.Struct("<6I8fI")
_BONE_VERTEX = struct.Struct("<6f")
_FRAME_REBASE_SCHEMA = "numi.human.composed-rib-body-local-frame-rebase.v1"
_PARTITION_SCHEMA = "numi.human.intercostal313-rib48-rib49-joint-priority-partition-face-ancestry.v1"
_RIBS = {
    48: ("FJ3228", "803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603", 12, 868),
    49: ("FJ3229", "631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58", 13, 1769),
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _float32(value: float) -> float:
    try:
        result = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error, TypeError, ValueError) as error:
        raise model.ImportError("composed rib frame rebase contains an invalid FP32 value") from error
    if not math.isfinite(result):
        raise model.ImportError("composed rib frame rebase contains a non-finite FP32 value")
    return result


def _rotate_native_f32(quaternion: tuple[float, ...], vector: tuple[float, ...]) -> tuple[float, float, float]:
    """Match the existing NHBONES quaternion-vector operation order."""
    ax, ay, az, qw = (_float32(value) for value in quaternion)
    vx, vy, vz = (_float32(value) for value in vector)
    f = _float32
    tx = f(f(2.0) * f(f(ay * vz) - f(az * vy)))
    ty = f(f(2.0) * f(f(az * vx) - f(ax * vz)))
    tz = f(f(2.0) * f(f(ax * vy) - f(ay * vx)))
    cx = f(f(qw * tx) + f(f(ay * tz) - f(az * ty)))
    cy = f(f(qw * ty) + f(f(az * tx) - f(ax * tz)))
    cz = f(f(qw * tz) + f(f(ax * ty) - f(ay * tx)))
    return f(vx + cx), f(vy + cy), f(vz + cz)


def _row_body_local(vertex: tuple[float, ...], pose: tuple[float, ...]) -> tuple[float, ...]:
    translation = tuple(_float32(value) for value in pose[:3])
    quaternion = tuple(_float32(value) for value in pose[3:7])
    scale = _float32(pose[7])
    position = _rotate_native_f32(quaternion, vertex[:3])
    normal = _rotate_native_f32(quaternion, vertex[3:6])
    return tuple(
        _float32(translation[index] + _float32(scale * position[index]))
        for index in range(3)
    ) + normal


def _point_key(point: Any) -> bytes:
    return struct.pack("<3f", *(float(value) for value in point))


def _decode_bones(raw: bytes) -> tuple[tuple[Any, ...], list[tuple[Any, ...]], int, int]:
    if len(raw) < _BONE_HEADER.size:
        raise model.ImportError("source003 NHBONES payload is truncated")
    header = _BONE_HEADER.unpack_from(raw)
    magic, abi, row_count, vertex_count, index_count, _, _ = header
    expected_size = _BONE_HEADER.size + row_count * _BONE_RECORD.size + vertex_count * 24 + index_count * 4
    if magic != b"NHBONES1" or abi != 3 or row_count != 185 or len(raw) != expected_size:
        raise model.ImportError("source003 NHBONES ABI, row count, or byte length changed")
    records = [
        _BONE_RECORD.unpack_from(raw, _BONE_HEADER.size + index * _BONE_RECORD.size)
        for index in range(row_count)
    ]
    vertex_offset = _BONE_HEADER.size + row_count * _BONE_RECORD.size
    index_offset = vertex_offset + vertex_count * _BONE_VERTEX.size
    return header, records, vertex_offset, index_offset


def _load_source003_partition_owner(source_report: dict[str, Any]) -> Any:
    """Load the exact pinned composition owner used to create source003."""
    pins = source_report.get("input_pins")
    if not isinstance(pins, dict):
        raise model.ImportError("source003 report lacks owner-module source pins")
    matches = [
        (Path(path).resolve(), digest)
        for path, digest in pins.items()
        if isinstance(path, str)
        and Path(path).name == "compose_intercostal313_rib48_partition_010_002.py"
    ]
    if len(matches) != 1:
        raise model.ImportError("source003 report does not uniquely pin its partition composition owner")
    path, expected_sha = matches[0]
    if not path.is_file() or not isinstance(expected_sha, str) or model.sha256(path) != expected_sha:
        raise model.ImportError("source003 partition composition owner is missing or changed")
    import sys
    name = f"_numi_source003_partition_owner_{expected_sha[:16]}"
    owner = sys.modules.get(name)
    if owner is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise model.ImportError("source003 partition composition owner cannot be loaded")
        owner = importlib.util.module_from_spec(spec)
        sys.modules[name] = owner
        try:
            spec.loader.exec_module(owner)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    required = ("inverse_body64", "inverse_bone64", "bone_to_body_f32")
    if any(not callable(getattr(owner, attribute, None)) for attribute in required):
        raise model.ImportError("source003 partition owner lacks a required declared transform")
    return owner


def _source_partition_interface_maps(
    partition_directory: Path,
    source_report: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    nha_rows: dict[int, dict[str, Any]],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, dict[str, Any]]]:
    ancestry_path = partition_directory / "partition-face-ancestry.json"
    meshes_path = partition_directory / "partition-region-meshes.npz"
    strict_path = partition_directory / "strict-float32-partition-audit.json"
    prep_path = partition_directory / "source-preparation-report.json"
    expected_files = {
        "partition_ancestry_sha256": ancestry_path,
        "partition_region_meshes_sha256": meshes_path,
        "strict_float32_audit_sha256": strict_path,
        "finalizer_report_sha256": prep_path,
    }
    for report_key, path in expected_files.items():
        if not path.is_file() or model.sha256(path) != source_report.get(report_key):
            raise model.ImportError(f"source003 partition input changed: {path.name}")
    ancestry = model.read_json(ancestry_path)
    if ancestry.get("schema") != _PARTITION_SCHEMA or ancestry.get("priority") != "rib_union":
        raise model.ImportError("source003 rib partition ancestry has an unsupported identity")
    audit = model.read_json(strict_path)
    if (
        audit.get("interiors_disjoint") is not True
        or audit.get("shared_interface_opposite_chain") is not True
        or audit.get("shared_interface_face_count") != 2914
    ):
        raise model.ImportError("source003 partition strict Float32 audit is not a pass")
    prep = model.read_json(prep_path)
    if prep.get("native_admission") is not False or prep.get("retained_motion_admission") is not False:
        raise model.ImportError("source003 partition preparation overstates its admission stage")

    with np.load(meshes_path, allow_pickle=False) as archive:
        meshes = {name: np.asarray(archive[name]).copy() for name in archive.files}
    muscle_positions = np.asarray(meshes["intercostal313_region_positions_m"], dtype="<f4")
    muscle_faces = np.asarray(meshes["intercostal313_region_faces"], dtype=np.int64)
    union_positions = np.asarray(meshes["rib_union_positions_m"], dtype="<f4")
    union_faces = np.asarray(meshes["rib_union_faces"], dtype=np.int64)
    if (
        muscle_positions.shape != (8870, 3) or muscle_faces.shape != (17764, 3)
        or union_positions.shape != (6761, 3) or union_faces.shape != (13514, 3)
        or not np.isfinite(muscle_positions).all() or not np.isfinite(union_positions).all()
    ):
        raise model.ImportError("source003 ordered partition mesh dimensions or values changed")
    base_nha_vertices = source_report.get("nha_composition", {}).get("old_vertices")
    if type(base_nha_vertices) is not int or base_nha_vertices < 0:
        raise model.ImportError("source003 NHA313 original vertex offset is invalid")
    nha_partition_positions = np.asarray(nha_rows[313]["vertices6"], dtype="<f4")[base_nha_vertices:, :3]
    if nha_partition_positions.shape != muscle_positions.shape:
        raise model.ImportError("source003 NHA313 appended position count differs from ordered partition mesh")

    muscle_lineage = ancestry.get("intercostal313_source_faces")
    rib_lineage = ancestry.get("rib_union_source_faces")
    parent_map = ancestry.get("shared_interface_parent_mapping")
    if (
        not isinstance(muscle_lineage, list) or len(muscle_lineage) != len(muscle_faces)
        or not isinstance(rib_lineage, list) or len(rib_lineage) != len(union_faces)
        or not isinstance(parent_map, list) or len(parent_map) != 2914
        or ancestry.get("shared_interface_face_count") != 2914
    ):
        raise model.ImportError("source003 partition ancestry inventory changed")

    face_ids: dict[int, list[int]] = {stable_id: [] for stable_id in _RIBS}
    for face_id, item in enumerate(rib_lineage):
        typed_row = item.get("typed_row") if isinstance(item, dict) else None
        if (
            not isinstance(typed_row, list) or len(typed_row) != 2
            or type(typed_row[0]) is not int or typed_row[0] != 51004
            or type(typed_row[1]) is not int or typed_row[1] not in _RIBS
        ):
            raise model.ImportError("source003 partition face has invalid typed rib ancestry")
        face_ids[typed_row[1]].append(face_id)
    expected_face_counts = {48: 4728, 49: 8786}
    original_vertex_counts = {48: 1641, 49: 2862}
    maps: dict[int, dict[int, int]] = {stable_id: {} for stable_id in _RIBS}
    used_interface_faces: set[tuple[str, int]] = set()
    for entry in parent_map:
        if not isinstance(entry, dict):
            raise model.ImportError("source003 shared-interface ancestry row is malformed")
        typed = entry.get("bone_typed_row")
        if (
            not isinstance(typed, list) or len(typed) != 2
            or type(typed[0]) is not int or typed[0] != 51004
            or type(typed[1]) is not int or typed[1] not in _RIBS
        ):
            raise model.ImportError("source003 shared-interface ancestry has the wrong rib owner")
        stable_id = typed[1]
        bone_face = entry.get("rib_union_region_face")
        muscle_face = entry.get("intercostal313_region_face")
        if (
            type(bone_face) is not int or not 0 <= bone_face < len(union_faces)
            or type(muscle_face) is not int or not 0 <= muscle_face < len(muscle_faces)
            or ("bone", bone_face) in used_interface_faces
            or ("muscle", muscle_face) in used_interface_faces
            or rib_lineage[bone_face].get("typed_row") != typed
        ):
            raise model.ImportError("source003 shared-interface face ancestry is duplicated or mistyped")
        muscle_parent = muscle_lineage[muscle_face]
        if (
            muscle_parent.get("kind") != "generated_reversed_bone_interface_face"
            or muscle_parent.get("bone_typed_row") != typed
        ):
            raise model.ImportError("source003 intercostal cap lacks reciprocal typed rib ancestry")
        bone_tri = union_faces[bone_face]
        muscle_tri = muscle_faces[muscle_face]
        bone_keys = [_point_key(union_positions[index]) for index in bone_tri]
        muscle_keys = [_point_key(muscle_positions[index]) for index in muscle_tri]
        if not any(
            muscle_keys == [bone_keys[permutation[index]] for index in range(3)]
            for permutation in ((0, 2, 1), (2, 1, 0), (1, 0, 2))
        ):
            # Exact reversed cyclic order, preserving the orientation contract.
            expected = (bone_keys[0], bone_keys[2], bone_keys[1])
            if not any(muscle_keys == [expected[(index + shift) % 3] for index in range(3)] for shift in range(3)):
                raise model.ImportError("source003 reciprocal interface triangles are not exact reversed FP32 faces")
        for union_vertex in bone_tri:
            key = _point_key(union_positions[union_vertex])
            matching = [int(index) for index in muscle_tri if _point_key(muscle_positions[index]) == key]
            if len(matching) != 1:
                raise model.ImportError("source003 interface corner is missing or duplicated in row313")
            muscle_vertex = matching[0]
            prior = maps[stable_id].get(muscle_vertex)
            if prior is not None and prior != int(union_vertex):
                raise model.ImportError("one source313 interface vertex maps to multiple rib output vertices")
            maps[stable_id][muscle_vertex] = int(union_vertex)
        used_interface_faces.add(("bone", bone_face))
        used_interface_faces.add(("muscle", muscle_face))
    if len(used_interface_faces) != 2 * len(parent_map):
        raise model.ImportError("source003 reciprocal interface ancestry is incomplete")

    # The partition NPZ is in the actual body20/world frame; source003 stores
    # NHA313 in body20-local coordinates. Reproduce the exact pinned composer
    # transform and its shared-rib interface overwrite before comparing bytes.
    pose = source_report.get("actual_body20_pose")
    if (
        not isinstance(pose, dict)
        or type(pose.get("body_index")) is not int
        or pose.get("body_index") != 20
        or not isinstance(pose.get("position_m"), list)
        or len(pose["position_m"]) != 3
        or not isinstance(pose.get("quaternion_xyzw"), list)
        or len(pose["quaternion_xyzw"]) != 4
    ):
        raise model.ImportError("source003 lacks its typed actual body20 transform")
    body_position = np.asarray(pose["position_m"], dtype=np.float32)
    body_quaternion = np.asarray(pose["quaternion_xyzw"], dtype=np.float32)
    if not np.isfinite(body_position).all() or not np.isfinite(body_quaternion).all():
        raise model.ImportError("source003 body20 transform is nonfinite")
    owner = _load_source003_partition_owner(source_report)
    expected_body_local = owner.inverse_body64(
        muscle_positions, body_position, body_quaternion,
    ).astype(np.float32)
    bone_local_partitions: dict[int, np.ndarray] = {}
    for stable_id, mapping in maps.items():
        row = rows.get(stable_id)
        record = row.get("record") if isinstance(row, dict) else None
        if (
            not isinstance(record, tuple)
            or len(record) != 15
            or type(record[0]) is not int
            or record[0] != 20
            or type(record[5]) is not int
            or record[5] != stable_id
        ):
            raise model.ImportError(f"source003 rib{stable_id} row lacks its exact typed bone record")
        region_faces = np.asarray(face_ids[stable_id], dtype=np.int64)
        used_union = np.unique(union_faces[region_faces].reshape(-1))
        union_to_local = {int(union_vertex): local for local, union_vertex in enumerate(used_union.tolist())}
        body_points64 = owner.inverse_body64(
            union_positions[used_union], body_position, body_quaternion,
        )
        bone_local = owner.inverse_bone64(body_points64, record).astype(np.float32)
        bone_local_partitions[stable_id] = bone_local
        body_native = owner.bone_to_body_f32(bone_local, record)
        for muscle_vertex, union_vertex in mapping.items():
            expected_body_local[int(muscle_vertex)] = body_native[union_to_local[int(union_vertex)]]
    if not np.array_equal(nha_partition_positions, expected_body_local):
        mismatch = np.argwhere(nha_partition_positions != expected_body_local)
        first = mismatch[0].tolist() if len(mismatch) else None
        raise model.ImportError(
            f"source003 NHA313 body-local append differs from exact pinned world-to-body composition "
            f"at slot/axis {first}"
        )

    base_nha_vertices = source_report.get("nha_composition", {}).get("old_vertices")
    if type(base_nha_vertices) is not int or len(nha_rows[313]["vertices6"]) != base_nha_vertices + len(muscle_positions):
        raise model.ImportError("source003 NHA313 row does not contain the exact partition vertex append")
    appended_faces = muscle_faces + base_nha_vertices
    if (
        source_report.get("nha_composition", {}).get("appended_vertices") != len(muscle_positions)
        or source_report.get("nha_composition", {}).get("appended_partition_faces") != len(muscle_faces)
        or not np.array_equal(nha_rows[313]["faces"][-len(appended_faces):], appended_faces)
    ):
        raise model.ImportError("source003 NHA313 output faces do not match the ordered partition mesh")

    out: dict[int, list[tuple[int, int]]] = {}
    row_metadata: dict[int, dict[str, Any]] = {}
    for stable_id, (member, member_sha, channel, expected_vertices) in _RIBS.items():
        if len(face_ids[stable_id]) != expected_face_counts[stable_id]:
            raise model.ImportError(f"source003 rib{stable_id} partition face count changed")
        used_union = np.unique(union_faces[np.asarray(face_ids[stable_id], dtype=np.int64)].reshape(-1))
        union_to_local = {int(index): position for position, index in enumerate(used_union.tolist())}
        anchor = None
        # The input manifest records the exact original source identity and the composed counts.
        for candidate in source_report.get("input_pins", {}):
            if candidate.endswith("bodyparts3d-myosim-major-bones.manifest.json"):
                anchor = candidate
        if not anchor:
            raise model.ImportError("source003 preparation report does not pin the NHBONES manifest")
        # The caller passes the validated manifest via row metadata below.
        row = rows[stable_id]
        compiled_offset = original_vertex_counts[stable_id]
        expected_local_faces = np.asarray(union_faces[np.asarray(face_ids[stable_id], dtype=np.int64)], dtype=np.int64)
        expected_local_faces = np.vectorize(union_to_local.__getitem__, otypes=[np.int64])(expected_local_faces)
        expected_local_faces += compiled_offset
        if not np.array_equal(row["faces"], expected_local_faces):
            raise model.ImportError(f"source003 rib{stable_id} faces differ from their exact ordered partition ancestry")
        if len(row["vertices6"]) != compiled_offset + len(used_union):
            raise model.ImportError(f"source003 rib{stable_id} partition vertex allocation changed")
        compiled_partition_positions = np.asarray(row["vertices6"], dtype="<f4")[compiled_offset:, :3]
        expected_partition_positions = bone_local_partitions[stable_id]
        if (
            compiled_partition_positions.shape != expected_partition_positions.shape
            or not np.array_equal(compiled_partition_positions, expected_partition_positions)
        ):
            raise model.ImportError(f"source003 rib{stable_id} appended positions differ from ordered partition mesh")
        maps_by_local = []
        for muscle_vertex, union_vertex in maps[stable_id].items():
            rib_local = compiled_offset + union_to_local[union_vertex]
            nha_local = base_nha_vertices + muscle_vertex
            if rib_local < compiled_offset or rib_local >= len(row["vertices6"]):
                raise model.ImportError("source003 rib interface local vertex escapes its typed row")
            maps_by_local.append((rib_local, nha_local))
        maps_by_local.sort()
        if len(maps_by_local) != expected_vertices or len({pair[0] for pair in maps_by_local}) != expected_vertices:
            raise model.ImportError(f"source003 rib{stable_id} interface vertex count or uniqueness changed")
        map_bytes = b"".join(struct.pack("<III", stable_id, bone_id, nha_id) for bone_id, nha_id in maps_by_local)
        out[stable_id] = maps_by_local
        row_metadata[stable_id] = {
            "member_id": member,
            "member_sha256": member_sha,
            "channel_index": channel,
            "compiled_vertex_count": len(row["vertices6"]),
            "compiled_triangle_count": len(row["faces"]),
            "partition_row_face_indices_sha256": _sha_bytes(np.asarray(expected_local_faces, dtype="<u4").reshape(-1).tobytes()),
            "raw_source_vertex_count": compiled_offset,
            "interface_vertex_count": expected_vertices,
            "interface_vertex_map_sha256": _sha_bytes(map_bytes),
            "interface_vertex_pairs": [[a, b] for a, b in maps_by_local],
        }
    if set(maps) != set(_RIBS):
        raise model.ImportError("source003 partition interface misses one of the two typed ribs")
    return out, row_metadata



def validate_composed_rib_reframe(
    artifact: Path,
    manifest: dict[str, Any],
    output_raw: bytes,
) -> dict[str, Any]:
    """Re-derive the source003 partition bake from pinned preimage and source proof."""
    artifact = artifact.resolve()
    bake_manifest = manifest.get("body_local_bakes")
    if (
        not isinstance(bake_manifest, dict)
        or bake_manifest.get("mode") != "reframe_existing_composed_partitioned_rows"
    ):
        raise model.ImportError("composed rib reframe validation received another manifest mode")
    declaration_path = artifact / str(bake_manifest.get("declaration_file", ""))
    if (
        not declaration_path.is_file()
        or model.sha256(declaration_path) != bake_manifest.get("declaration_sha256")
    ):
        raise model.ImportError("composed rib reframe declaration is missing or changed")
    declaration = model.read_json(declaration_path)
    if declaration.get("schema") != _FRAME_REBASE_SCHEMA:
        raise model.ImportError("composed rib reframe declaration has the wrong schema")
    inputs = declaration.get("input_paths")
    input_hashes = declaration.get("input_source003")
    partition_proof = declaration.get("partition_proof")
    if not isinstance(inputs, dict) or not isinstance(input_hashes, dict) or not isinstance(partition_proof, dict):
        raise model.ImportError("composed rib reframe declaration lacks its complete source pins")
    required_path_hashes = {
        "receipt": input_hashes.get("receipt_sha256"),
        "source_export_report": input_hashes.get("source_export_report_sha256"),
        "nha_payload": input_hashes.get("nha_payload_sha256"),
        "nhbones_payload": input_hashes.get("nhbones_payload_sha256"),
        "nhbones_manifest": input_hashes.get("nhbones_manifest_sha256"),
        "registration": input_hashes.get("registration_sha256"),
        "actual_capture_pack": input_hashes.get("actual_capture_pack_sha256"),
        "actual_capture_receipt": input_hashes.get("actual_capture_receipt_sha256"),
    }
    for key, expected_sha in required_path_hashes.items():
        path_value = inputs.get(key)
        if not isinstance(path_value, str) or not isinstance(expected_sha, str):
            raise model.ImportError(f"composed rib reframe source path pin {key} is malformed")
        source_path = Path(path_value).resolve()
        if not source_path.is_file() or model.sha256(source_path) != expected_sha:
            raise model.ImportError(f"composed rib reframe input {key} is missing or changed")
    source_receipt_path = Path(inputs["receipt"]).resolve()
    source_report_path = Path(inputs["source_export_report"]).resolve()
    source_receipt = model.read_json(source_receipt_path)
    source_nha_path = Path(inputs["nha_payload"]).resolve()
    source_bones_path = Path(inputs["nhbones_payload"]).resolve()
    source_manifest_path = Path(inputs["nhbones_manifest"]).resolve()
    registration_path = Path(inputs["registration"]).resolve()
    capture_path = Path(inputs["actual_capture_pack"]).resolve()
    capture_receipt_path = Path(inputs["actual_capture_receipt"]).resolve()
    source_report_copy = artifact / source_report_path.name
    output_receipt_path = artifact / source_receipt_path.name
    if (
        not source_report_copy.is_file()
        or model.sha256(source_report_copy) != input_hashes["source_export_report_sha256"]
        or model.sha256(source_report_path) != input_hashes["source_export_report_sha256"]
        or not output_receipt_path.is_file()
    ):
        raise model.ImportError("composed rib reframe source report or output receipt is missing or changed")
    output_receipt = model.read_json(output_receipt_path)
    if (
        Path(str(source_receipt.get("payload", {}).get("path", ""))).resolve() != source_nha_path
        or Path(str(source_receipt.get("provenance", {}).get("bones_payload", ""))).resolve() != source_bones_path
        or source_receipt.get("payload", {}).get("sha256") != input_hashes["nha_payload_sha256"]
        or source_receipt.get("provenance", {}).get("bones_payload_sha256") != input_hashes["nhbones_payload_sha256"]
    ):
        raise model.ImportError("composed rib reframe source receipt does not bind its source payload paths")
    source_report = model.read_json(source_report_copy)
    source_manifest = model.read_json(source_manifest_path)
    source_bones_raw = source_bones_path.read_bytes()
    current_nha_path = artifact / source_nha_path.name
    if (
        not current_nha_path.is_file()
        or model.sha256(current_nha_path) != input_hashes["nha_payload_sha256"]
        or model.sha256(source_nha_path) != input_hashes["nha_payload_sha256"]
        or model.sha256(source_manifest_path) != input_hashes["nhbones_manifest_sha256"]
        or model.sha256(registration_path) != input_hashes["registration_sha256"]
        or model.sha256(capture_path) != input_hashes["actual_capture_pack_sha256"]
        or model.sha256(capture_receipt_path) != input_hashes["actual_capture_receipt_sha256"]
        or model.sha256(source_receipt_path) != input_hashes["receipt_sha256"]
        or source_report.get("output_sha256", {}).get("nha") != input_hashes["nha_payload_sha256"]
        or source_report.get("output_sha256", {}).get("nhbones") != input_hashes["nhbones_payload_sha256"]
        or source_report.get("output_sha256", {}).get("bone_manifest") != input_hashes["nhbones_manifest_sha256"]
        or source_report.get("output_sha256", {}).get("receipt") != input_hashes["receipt_sha256"]
        or source_report.get("actual_capture_pack_sha256") != input_hashes["actual_capture_pack_sha256"]
        or source_report.get("actual_capture_receipt_sha256") != input_hashes["actual_capture_receipt_sha256"]
        or source_manifest.get("payload", {}).get("sha256") != input_hashes["nhbones_payload_sha256"]
        or source_manifest.get("source", {}).get("registration", {}).get("sha256") != input_hashes["registration_sha256"]
        or manifest.get("payload", {}).get("sha256") != _sha_bytes(output_raw)
        or _sha_bytes(output_raw) != bake_manifest.get("payload_sha256")
        or not isinstance(bake_manifest.get("proof_paths"), dict)
    ):
        raise model.ImportError("composed rib reframe source003/output hash chain is inconsistent")
    proof_paths = bake_manifest["proof_paths"]
    partition_directory = Path(str(proof_paths.get("partition_directory", ""))).resolve()
    if not partition_directory.is_dir():
        raise model.ImportError("composed rib reframe partition proof directory is absent")
    for file_key, proof_key in (
        ("partition_ancestry", "partition_ancestry_sha256"),
        ("partition_meshes", "partition_meshes_sha256"),
        ("strict_float32_audit", "strict_float32_audit_sha256"),
        ("partition_report", "source_preparation_report_sha256"),
    ):
        proof_path = partition_directory / str(proof_paths.get(file_key, ""))
        if (
            not proof_path.is_file()
            or Path(proof_paths.get(file_key, "")).name != proof_paths.get(file_key)
            or model.sha256(proof_path) != partition_proof.get(proof_key)
        ):
            raise model.ImportError(f"composed rib reframe partition proof {file_key} is missing or changed")

    source_header, source_records, source_vertex_offset, source_index_offset = _decode_bones(source_bones_raw)
    output_header, output_records, output_vertex_offset, output_index_offset = _decode_bones(output_raw)
    if (
        Path(str(output_receipt.get("payload", {}).get("path", ""))).resolve()
        != (artifact / source_nha_path.name).resolve()
        or Path(str(output_receipt.get("provenance", {}).get("bones_payload", ""))).resolve()
        != (artifact / source_bones_path.name).resolve()
        or output_receipt.get("payload", {}).get("sha256") != model.sha256(artifact / source_nha_path.name)
        or output_receipt.get("provenance", {}).get("bones_payload_sha256") != _sha_bytes(output_raw)
        or output_receipt.get("functional_bindings", {}).get("bones_payload_sha256") != _sha_bytes(output_raw)
    ):
        raise model.ImportError("composed rib reframe output receipt does not bind its exact output payloads")
    if (
        source_header != output_header
        or [int(record[5]) for record in source_records] != list(range(1, 186))
        or [int(record[5]) for record in output_records] != list(range(1, 186))
        or source_bones_raw[source_index_offset:] != output_raw[output_index_offset:]
    ):
        raise model.ImportError("composed rib reframe changed NHBONES identity, record order, or indices")
    source_record_by_id = {int(record[5]): record for record in source_records}
    output_record_by_id = {int(record[5]): record for record in output_records}
    target_ranges: list[tuple[int, int, int]] = []
    source_rows: dict[int, dict[str, Any]] = {}
    for stable_id in _RIBS:
        record = source_record_by_id[stable_id]
        first_vertex, count = int(record[1]), int(record[2])
        if first_vertex + count > source_header[3]:
            raise model.ImportError(f"composed rib reframe input row {stable_id} escapes the vertex table")
        target_ranges.append((first_vertex, first_vertex + count, stable_id))
        source_rows[stable_id] = {
            "vertices6": np.ndarray(
                (count, 6), dtype="<f4", buffer=source_bones_raw,
                offset=source_vertex_offset + first_vertex * 24,
            ).copy(),
            "faces": np.ndarray(
                (int(record[4]) // 3, 3), dtype="<u4", buffer=source_bones_raw,
                offset=source_index_offset + int(record[3]) * 4,
            ).astype(np.int64) - first_vertex,
            "record": record,
        }
    target_ranges.sort()
    if any(left[1] > right[0] for left, right in itertools.pairwise(target_ranges)):
        raise model.ImportError("composed rib reframe source rows overlap in the NHBONES vertex table")
    source_nha_header, source_nha_rows = parse_nha_payload(current_nha_path)
    if source_nha_header != parse_nha_payload(source_nha_path)[0]:
        raise model.ImportError("composed rib reframe NHA header differs from source003")
    maps, map_metadata = _source_partition_interface_maps(
        partition_directory, source_report, source_rows, source_nha_rows,
    )
    if declaration.get("shared_interface_map_sha256") != bake_manifest.get("shared_interface_map_sha256"):
        raise model.ImportError("composed rib reframe interface map identity changed")
    current_vertex_offset = output_vertex_offset
    maximum_residual = 0.0
    maximum_displacement = 0.0
    for stable_id in _RIBS:
        source_record = source_record_by_id[stable_id]
        output_record = output_record_by_id[stable_id]
        if (
            source_record[:6] != output_record[:6]
            or source_record[14] != output_record[14]
            or tuple(_float32(value) for value in output_record[6:14]) !=
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        ):
            raise model.ImportError(f"composed rib reframe row {stable_id} changed its typed owner or pose unexpectedly")
        row_declaration = next(
            (row for row in declaration.get("rows", []) if isinstance(row, dict) and row.get("stable_id") == stable_id),
            None,
        )
        manifest_row = next(
            (row for row in bake_manifest.get("rows", []) if isinstance(row, dict) and row.get("stable_id") == stable_id),
            None,
        )
        declaration_row_with_manifest_binding = (
            {
                **row_declaration,
                "declaration_file": bake_manifest.get("declaration_file"),
                "declaration_sha256": bake_manifest.get("declaration_sha256"),
            }
            if isinstance(row_declaration, dict) else None
        )
        if declaration_row_with_manifest_binding != manifest_row:
            raise model.ImportError(f"composed rib reframe row {stable_id} manifest/declaration differ")
        if (
            row_declaration.get("interface_vertex_pairs") != map_metadata[stable_id]["interface_vertex_pairs"]
            or row_declaration.get("interface_vertex_map_sha256") != map_metadata[stable_id]["interface_vertex_map_sha256"]
            or row_declaration.get("partition_row_face_indices_sha256") != map_metadata[stable_id]["partition_row_face_indices_sha256"]
        ):
            raise model.ImportError(f"composed rib reframe row {stable_id} source partition map changed")
        first_vertex, count = int(source_record[1]), int(source_record[2])
        stored = np.ndarray(
            (count, 6), dtype="<f4", buffer=source_bones_raw,
            offset=source_vertex_offset + first_vertex * 24,
        )
        pose = tuple(_float32(value) for value in source_record[6:14])
        expected = [_row_body_local(tuple(vertex), pose) for vertex in stored]
        for bone_local, nha_local in maps[stable_id]:
            expected_point = tuple(_float32(value) for value in source_nha_rows[313]["vertices6"][nha_local][:3])
            residual = max(abs(expected[bone_local][axis] - expected_point[axis]) for axis in range(3))
            displacement = math.sqrt(sum((expected[bone_local][axis] - expected_point[axis]) ** 2 for axis in range(3)))
            maximum_residual = max(maximum_residual, residual)
            maximum_displacement = max(maximum_displacement, displacement)
            expected[bone_local] = (*expected_point, *expected[bone_local][3:])
        actual = np.ndarray(
            (count, 6), dtype="<f4", buffer=output_raw,
            offset=current_vertex_offset + first_vertex * 24,
        )
        expected_array = np.asarray(expected, dtype="<f4")
        if not np.array_equal(actual, expected_array):
            raise model.ImportError(f"composed rib reframe row {stable_id} stored FP32 vertices do not match its exact bake")
        actual_faces = np.ndarray(
            (int(output_record[4]) // 3, 3), dtype="<u4", buffer=output_raw,
            offset=output_index_offset + int(output_record[3]) * 4,
        )
        row_local_faces = np.asarray(actual_faces, dtype=np.int64) - int(output_record[1])
        if np.any(row_local_faces < 0) or np.any(row_local_faces >= int(output_record[2])):
            raise model.ImportError(f"composed rib reframe row {stable_id} emitted a face outside its local vertex range")
        face_hash = _sha_bytes(np.asarray(row_local_faces, dtype="<u4").reshape(-1).tobytes())
        if face_hash != row_declaration.get("partition_row_face_indices_sha256"):
            raise model.ImportError(f"composed rib reframe row {stable_id} face-index sequence changed")
    target_ranges = sorted(target_ranges)
    cursor = 0
    for start, end, stable_id in target_ranges:
        if source_bones_raw[source_vertex_offset + cursor * 24:source_vertex_offset + start * 24] != output_raw[current_vertex_offset + cursor * 24:current_vertex_offset + start * 24]:
            raise model.ImportError("composed rib reframe changed a vertex outside rows 48 and 49")
        cursor = end
    if source_bones_raw[source_vertex_offset + cursor * 24:source_index_offset] != output_raw[current_vertex_offset + cursor * 24:output_index_offset]:
        raise model.ImportError("composed rib reframe changed the trailing non-target bone vertices")
    validation_basis = {
        "source003": input_hashes,
        "partition": partition_proof,
        "output_nha_sha256": model.sha256(current_nha_path),
        "output_nhbones_sha256": _sha_bytes(output_raw),
        "interface_map_sha256": bake_manifest.get("shared_interface_map_sha256"),
        "maximum_pre_bake_interface_residual_m": maximum_residual,
        "maximum_interface_bake_displacement_m": maximum_displacement,
        "method": "source003 composed row pose to body20 local; shared points copied from exact NHA313 FP32 table",
    }
    validation_sha = _sha_bytes(json.dumps(validation_basis, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return {
        "status": "PASS_SOURCE003_DERIVED_PARTITION_AND_BODY20_REFRAME_REDERIVED",
        "validation_sha256": validation_sha,
        "maximum_pre_bake_interface_residual_m": maximum_residual,
        "maximum_interface_bake_displacement_m": maximum_displacement,
        "checked_untouched_rows": 183,
        "checked_partition_rows": [48, 49],
        "checked_interface_vertices": {stable_id: len(maps[stable_id]) for stable_id in sorted(maps)},
    }


def reframe_composed_rib_payload(
    source_receipt_path: Path,
    registration_path: Path,
    partition_directory: Path,
    actual_capture_pack: Path,
    output: Path,
) -> dict[str, Any]:
    """Reframe source003's composed rows 48/49 while retaining their partition faces."""
    receipt_path = source_receipt_path.resolve()
    registration_path = registration_path.resolve()
    partition_directory = partition_directory.resolve()
    capture_path = actual_capture_pack.resolve()
    output_path = output.resolve()
    if output_path.exists():
        raise model.ImportError(f"refusing to overwrite composed body-local output: {output_path}")
    receipt = model.read_json(receipt_path)
    nha_path = Path(receipt.get("payload", {}).get("path", "")).resolve()
    bones_path = Path(receipt.get("provenance", {}).get("bones_payload", "")).resolve()
    manifest_path = bones_path.with_name("bodyparts3d-myosim-major-bones.manifest.json")
    source_report_path = receipt_path.with_name("source-export-report.json")
    capture_receipt_path = capture_path.with_name("step-0.receipt.json")
    if not all(path.is_file() for path in (
        nha_path, bones_path, manifest_path, source_report_path, registration_path,
        capture_path, capture_receipt_path,
    )):
        raise model.ImportError("source003 reframe input set is incomplete")
    source_report = model.read_json(source_report_path)
    report_sha = model.sha256(source_report_path)
    manifest = model.read_json(manifest_path)
    bones_raw = bones_path.read_bytes()
    _nha_header, nha_rows = parse_nha_payload(nha_path)
    if source_report.get("status") != "SOURCE_EXPORT_COMPLETE_NATIVE_AND_MOTION_NOT_ADMITTED":
        raise model.ImportError("source003 export report does not identify the expected source-only stage")
    if (
        source_report.get("output_sha256", {}).get("nha") != model.sha256(nha_path)
        or source_report.get("output_sha256", {}).get("nhbones") != _sha_bytes(bones_raw)
        or source_report.get("output_sha256", {}).get("bone_manifest") != model.sha256(manifest_path)
        or source_report.get("output_sha256", {}).get("receipt") != model.sha256(receipt_path)
        or receipt.get("payload", {}).get("sha256") != model.sha256(nha_path)
        or receipt.get("provenance", {}).get("bones_payload_sha256") != _sha_bytes(bones_raw)
        or receipt.get("functional_bindings", {}).get("bones_payload_sha256") != _sha_bytes(bones_raw)
        or Path(receipt.get("provenance", {}).get("bones_payload", "")).resolve() != bones_path
        or Path(receipt.get("payload", {}).get("path", "")).resolve() != nha_path
    ):
        raise model.ImportError("source003 receipt/report/payload hashes are inconsistent")
    if (
        source_report.get("actual_capture_pack_sha256") != model.sha256(capture_path)
        or source_report.get("actual_capture_receipt_sha256") != model.sha256(capture_receipt_path)
        or manifest.get("payload", {}).get("sha256") != _sha_bytes(bones_raw)
        or manifest.get("payload", {}).get("payload_abi") != 3
        or manifest.get("source", {}).get("registration", {}).get("sha256") != model.sha256(registration_path)
        or manifest.get("source", {}).get("registration", {}).get("sha256") != receipt.get("provenance", {}).get("bodyparts_registration_sha256")
    ):
        raise model.ImportError("source003 actual capture or body registration identity changed")
    pins = source_report.get("input_pins")
    if not isinstance(pins, dict):
        raise model.ImportError("source003 export report lacks immutable input pins")
    required_pins = {
        str(capture_path): model.sha256(capture_path),
        str(capture_receipt_path): model.sha256(capture_receipt_path),
    }
    for path_string, digest in required_pins.items():
        if pins.get(path_string) != digest:
            raise model.ImportError(f"source003 immutable source report does not pin {Path(path_string).name}")
    if source_report.get("native_admission") is not False or source_report.get("retained_motion_admission") is not False:
        raise model.ImportError("source003 report claims native or retained-motion admission")
    if source_report.get("nhbones_composition", {}).get("changed_ids") != [48, 49] or source_report.get("nhbones_composition", {}).get("untouched_bones_exact") != 183:
        raise model.ImportError("source003 is not the expected two-row partition composition")

    registration = model.read_json(registration_path)
    reg_sha = model.sha256(registration_path)
    if registration.get("status") not in {
        "provisional_visual_registration_not_admitted_to_collision_or_physics",
        "inferred_attachment_surface_visual_registration_not_admitted_to_collision_or_physics",
    }:
        raise model.ImportError("source003 rib frame registration is not a visual-only candidate")
    _header, records, vertex_offset, index_offset = _decode_bones(bones_raw)
    record_by_id = {int(record[5]): record for record in records}
    if (
        len(record_by_id) != 185 or set(record_by_id) != set(range(1, 186))
        or [int(record[5]) for record in records] != list(range(1, 186))
    ):
        raise model.ImportError("source003 NHBONES stable-ID record order is not canonical")
    if 313 not in nha_rows or (
        type(nha_rows[313]["body_index"]) is not int or nha_rows[313]["body_index"] != 20
        or type(nha_rows[313]["layer"]) is not int or nha_rows[313]["layer"] != 1
        or type(nha_rows[313]["flags"]) is not int or nha_rows[313]["flags"] != 0
    ):
        raise model.ImportError("source003 NHA313 typed owner changed")
    source_id = receipt.get("provenance", {}).get("source_id_map", {}).get("313", {})
    if (
        source_id.get("source_member") != "FJ1451M"
        or source_id.get("source_sha256") != "d4c021ba00dbe2bd3c7e722cb5982df7659d7d68fe6f4ec980588bfd65d31214"
        or source_id.get("body_index") != 20 or source_id.get("layer") != 1
    ):
        raise model.ImportError("source003 NHA313 source identity changed")

    # Source record geometry counts remain in the original owner anchors; partition counts live beside them.
    source_anchors = manifest.get("source", {}).get("anchors")
    if not isinstance(source_anchors, list) or len(source_anchors) != 185:
        raise model.ImportError("source003 NHBONES source member inventory is incomplete")
    row_data: dict[int, dict[str, Any]] = {}
    for stable_id, (member, member_sha, channel, _) in _RIBS.items():
        anchor = source_anchors[stable_id - 1]
        record = record_by_id[stable_id]
        partition = anchor.get("derived_visual_partition") if isinstance(anchor, dict) else None
        if (
            not isinstance(partition, dict) or partition.get("schema") != _PARTITION_SCHEMA
            or partition.get("stable_id") != stable_id
            or partition.get("runtime_channel_kind") != 5
            or partition.get("runtime_channel_index") != channel
            or anchor.get("member_id") != member or anchor.get("member_sha256") != member_sha
            or anchor.get("myosim_body") != "torso" or anchor.get("core_body_index") != 20
            or type(record[0]) is not int or record[0] != 20
            or type(record[5]) is not int or record[5] != stable_id
            or record[14] != anchor.get("source_record_index")
            or partition.get("partition_ancestry_sha256") != source_report.get("partition_ancestry_sha256")
            or partition.get("strict_float32_audit_sha256") != source_report.get("strict_float32_audit_sha256")
            or partition.get("source_preparation_report_sha256") != source_report.get("finalizer_report_sha256")
        ):
            raise model.ImportError(f"source003 composed rib{stable_id} owner/partition ancestry changed")
        raw_vertex_count = partition.get("original_compiled_vertex_count")
        raw_face_count = partition.get("original_compiled_triangle_count")
        compiled_vertex_count = partition.get("new_compiled_vertex_count")
        compiled_face_count = partition.get("new_compiled_triangle_count")
        if (
            type(raw_vertex_count) is not int or raw_vertex_count != anchor.get("vertex_count")
            or type(raw_face_count) is not int or raw_face_count != anchor.get("triangle_count")
            or type(compiled_vertex_count) is not int or type(compiled_face_count) is not int
            or record[2] != compiled_vertex_count or record[4] // 3 != compiled_face_count
        ):
            raise model.ImportError(f"source003 composed rib{stable_id} cardinality does not match ancestry")
        registration_anchor = registration.get("anchors", [])[stable_id - 1]
        transform = registration_anchor.get("registration", {}).get("source_obj_mm_to_core_inertial_body_m")
        if (
            registration_anchor.get("source", {}).get("member_id") != member
            or registration_anchor.get("source", {}).get("member_sha256") != member_sha
            or registration_anchor.get("target", {}).get("core_body_index") != 20
            or registration_anchor.get("target", {}).get("name") != "torso"
        ):
            raise model.ImportError(f"source003 registration row {stable_id} has wrong typed owner")
        translation, quaternion, scale = model._bodyparts_visual_local_pose(transform, member)
        expected_pose = tuple(_float32(value) for value in (*translation, *quaternion, scale))
        record_pose = tuple(_float32(value) for value in record[6:14])
        if record_pose != expected_pose:
            raise model.ImportError(f"source003 rib{stable_id} serialized pose differs from its exact registration")
        if anchor.get("source_record_index") != record[14]:
            raise model.ImportError(f"source003 rib{stable_id} source-record owner changed")
        bodyparts_archive_sha = registration_anchor.get("source", {}).get("archive_sha256")
        if not isinstance(bodyparts_archive_sha, str) or len(bodyparts_archive_sha) != 64 or any(
            character not in "0123456789abcdef" for character in bodyparts_archive_sha
        ):
            raise model.ImportError(f"source003 registration row {stable_id} lacks its exact BodyParts3D archive hash")
        row_data[stable_id] = {
            "record": record,
            "anchor": anchor,
            "partition": partition,
            "transform": transform,
            "bodyparts_archive_sha256": bodyparts_archive_sha,
            "source_pose": list(record_pose),
            "raw_source_vertex_count": raw_vertex_count,
            "raw_source_triangle_count": raw_face_count,
            "compiled_vertex_count": compiled_vertex_count,
            "compiled_triangle_count": compiled_face_count,
        }

    interface_maps, map_metadata = _source_partition_interface_maps(
        partition_directory, source_report,
        {stable_id: {
            "vertices6": np.ndarray((int(row_data[stable_id]["record"][2]), 6), dtype="<f4", buffer=bones_raw,
                offset=vertex_offset + int(row_data[stable_id]["record"][1]) * 24).copy(),
            "faces": np.ndarray((int(row_data[stable_id]["record"][4]) // 3, 3), dtype="<u4", buffer=bones_raw,
                offset=index_offset + int(row_data[stable_id]["record"][3]) * 4).astype(np.int64)
                - int(row_data[stable_id]["record"][1]),
            "record": row_data[stable_id]["record"],
        } for stable_id in _RIBS},
        nha_rows,
    )
    # The current source receipt independently binds these exact NHA313 attachment slots.
    for stable_id, pairs in interface_maps.items():
        attachment = receipt.get("functional_bindings", {}).get(f"intercostal_313_rib{stable_id}_motion_attachment")
        expected_nha_ids = sorted(nha_id for _, nha_id in pairs)
        source_attachment_ids = attachment.get("surface", {}).get("interface_local_vertex_ids") if isinstance(attachment, dict) else None
        if (
            not isinstance(source_attachment_ids, list)
            or any(type(value) is not int for value in source_attachment_ids)
            or source_attachment_ids != expected_nha_ids
            or attachment.get("bones_payload_sha256") != _sha_bytes(bones_raw)
            or attachment.get("surface", {}).get("stable_id") != 313
            or attachment.get("existing_rib_channel", {}).get("stable_id") != stable_id
            or attachment.get("existing_rib_channel", {}).get("channel_kind") != 5
        ):
            raise model.ImportError(f"source003 row313-to-rib{stable_id} attachment crosswalk changed")

    nha_row = nha_rows[313]
    output_raw = bytearray(bones_raw)
    record_pose_hashes: dict[int, str] = {}
    rows_report: list[dict[str, Any]] = []
    all_interface_maps: dict[str, list[list[int]]] = {}
    max_pre_interface_residual = 0.0
    max_interface_bake_displacement = 0.0
    transformed_record_offsets: set[int] = set()
    transformed_vertex_slots: set[int] = set()
    for stable_id, (member, member_sha, channel, _) in _RIBS.items():
        record = row_data[stable_id]["record"]
        first_vertex, count = int(record[1]), int(record[2])
        pose = row_data[stable_id]["source_pose"]
        record_pose_hash = _sha_bytes(struct.pack("<8f", *pose))
        record_pose_hashes[stable_id] = record_pose_hash
        row_vertices: list[list[float]] = []
        for local_id in range(count):
            global_id = first_vertex + local_id
            stored = _BONE_VERTEX.unpack_from(bones_raw, vertex_offset + global_id * _BONE_VERTEX.size)
            baked = _row_body_local(stored, pose)
            row_vertices.append(list(baked))
        interface_map = interface_maps[stable_id]
        pair_rows = []
        max_row_residual = 0.0
        max_row_displacement = 0.0
        for bone_local, nha_local in interface_map:
            if not (0 <= bone_local < count and 0 <= nha_local < len(nha_row["vertices6"])):
                raise model.ImportError("source003 row-local interface mapping is out of range")
            bone_before = tuple(row_vertices[bone_local][:3])
            nha_point = tuple(_float32(value) for value in nha_row["vertices6"][nha_local][:3])
            displacement = math.sqrt(sum((a - b) ** 2 for a, b in zip(bone_before, nha_point, strict=True)))
            max_row_displacement = max(max_row_displacement, displacement)
            max_row_residual = max(max_row_residual, max(abs(a - b) for a, b in zip(bone_before, nha_point, strict=True)))
            max_pre_interface_residual = max(max_pre_interface_residual, max_row_residual)
            row_vertices[bone_local][:3] = list(nha_point)
            pair_rows.append([bone_local, nha_local])
        max_interface_bake_displacement = max(max_interface_bake_displacement, max_row_displacement)
        for local_id, values in enumerate(row_vertices):
            global_id = first_vertex + local_id
            packed = _BONE_VERTEX.pack(*(_float32(value) for value in values))
            start = vertex_offset + global_id * _BONE_VERTEX.size
            output_raw[start:start + _BONE_VERTEX.size] = packed
            transformed_vertex_slots.add(global_id)
        identity_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        replacement = _BONE_RECORD.pack(*record[:6], *identity_pose, record[14])
        record_offset = _BONE_HEADER.size + (stable_id - 1) * _BONE_RECORD.size
        output_raw[record_offset:record_offset + _BONE_RECORD.size] = replacement
        transformed_record_offsets.add(record_offset)
        map_bytes = b"".join(struct.pack("<III", stable_id, bone_id, nha_id) for bone_id, nha_id in interface_map)
        all_interface_maps[str(stable_id)] = pair_rows
        row_report = {
            **map_metadata[stable_id],
            "stable_id": stable_id,
            "semantic": 51004,
            "member_id": member,
            "member_sha256": member_sha,
            "bodyparts_archive_sha256": row_data[stable_id]["bodyparts_archive_sha256"],
            "core_body_index": 20,
            "myosim_body": "torso",
            "source_record_index": int(record[14]),
            "source_transform_sha256": model._bodyparts_transform_matrix_sha256(row_data[stable_id]["transform"]),
            "source_obj_mm_to_core_inertial_body_m": row_data[stable_id]["transform"],
            "source_vertex_count": row_data[stable_id]["raw_source_vertex_count"],
            "source_triangle_count": row_data[stable_id]["raw_source_triangle_count"],
            "raw_source_triangle_count": row_data[stable_id]["raw_source_triangle_count"],
            "reframe_mode": "source003_composed_partition_row_to_body20_local",
            "source_record_pose_f32": list(pose),
            "source_record_pose_f32_sha256": record_pose_hash,
            "output_record_pose_f32": list(identity_pose),
            "stored_vertex_attributes_f32_sha256": _sha_bytes(b"".join(
                _BONE_VERTEX.pack(*values) for values in row_vertices
            )),
            "interface_vertex_map_sha256": _sha_bytes(map_bytes),
            "maximum_pre_bake_interface_residual_m": max_row_residual,
            "maximum_interface_bake_displacement_m": max_row_displacement,
            "post_bake_interface_residual_m": 0.0,
            "rounding": "source row pose applied with explicit native FP32 operation order; interface positions copied from NHA313 FP32 slots",
        }
        rows_report.append(row_report)

    if len(transformed_vertex_slots) != sum(row_data[sid]["compiled_vertex_count"] for sid in _RIBS):
        raise model.ImportError("source003 rib vertex ranges overlap or omit a reframe target")
    output_bytes = bytes(output_raw)
    output_sha = _sha_bytes(output_bytes)
    # The rewrite is restricted to the two declared row records and vertex ranges; indices and other rows remain exact.
    if output_bytes[:_BONE_HEADER.size] != bones_raw[:_BONE_HEADER.size]:
        raise model.ImportError("composed rib reframe unexpectedly changed the NHBONES header")
    if output_bytes[index_offset:] != bones_raw[index_offset:]:
        raise model.ImportError("composed rib reframe unexpectedly changed NHBONES indices")

    output_path.mkdir(parents=True, exist_ok=False)
    for source_file in receipt_path.parent.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, output_path / source_file.name)
    output_nha_path = output_path / nha_path.name
    output_bones_path = output_path / bones_path.name
    output_manifest_path = output_path / manifest_path.name
    output_receipt_path = output_path / receipt_path.name
    output_bones_path.write_bytes(output_bytes)

    declaration = {
        "schema": _FRAME_REBASE_SCHEMA,
        "status": "source_pinned_composed_visual_rows_reframed_to_existing_core_body20_no_native_admission",
        "input_source003": {
            "receipt_sha256": model.sha256(receipt_path),
            "source_export_report_sha256": report_sha,
            "nha_payload_sha256": model.sha256(nha_path),
            "nhbones_payload_sha256": _sha_bytes(bones_raw),
            "nhbones_manifest_sha256": model.sha256(manifest_path),
            "registration_sha256": reg_sha,
            "actual_capture_pack_sha256": model.sha256(capture_path),
            "actual_capture_receipt_sha256": model.sha256(capture_receipt_path),
        },
        "input_paths": {
            "receipt": str(receipt_path),
            "source_export_report": str(source_report_path),
            "nha_payload": str(nha_path),
            "nhbones_payload": str(bones_path),
            "nhbones_manifest": str(manifest_path),
            "registration": str(registration_path),
            "actual_capture_pack": str(capture_path),
            "actual_capture_receipt": str(capture_receipt_path),
        },
        "partition_proof": {
            "schema": _PARTITION_SCHEMA,
            "source_preparation_report_sha256": model.sha256(partition_directory / "source-preparation-report.json"),
            "partition_ancestry_sha256": model.sha256(partition_directory / "partition-face-ancestry.json"),
            "partition_meshes_sha256": model.sha256(partition_directory / "partition-region-meshes.npz"),
            "strict_float32_audit_sha256": model.sha256(partition_directory / "strict-float32-partition-audit.json"),
            "shared_interface_face_count": 2914,
        },
        "nha_row": {"semantic": 51010, "stable_id": 313, "body_index": 20, "layer": 1, "flags": 0,
                     "payload_sha256": model.sha256(nha_path), "local_vertex_count": len(nha_row["vertices6"])},
        "actual_reference_body20_pose": source_report.get("actual_body20_pose"),
        "rows": rows_report,
        "shared_interface_local_vertex_pairs": all_interface_maps,
        "shared_interface_map_sha256": _sha_bytes(b"".join(
            struct.pack("<III", stable_id, bone_id, nha_id)
            for stable_id in sorted(interface_maps)
            for bone_id, nha_id in interface_maps[stable_id]
        )),
        "residuals": {
            "maximum_pre_bake_interface_residual_m": max_pre_interface_residual,
            "maximum_interface_bake_displacement_m": max_interface_bake_displacement,
            "post_bake_interface_residual_m": 0.0,
            "post_bake_world_residual_claim": "not_measured_until_native_reemission",
        },
        "preservation": {
            "partition_face_indices_changed": False,
            "shared_interface_vertex_positions_source": "exact NHA313 stored FP32 row-local positions",
            "other_bone_rows_exact": 183,
            "physics_or_mass_change": False,
            "native_admission": False,
        },
    }
    declaration_path = output_path / "composed-rib-body-local-frame-rebase.json"
    declaration_path.write_text(json.dumps(declaration, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    declaration_sha = model.sha256(declaration_path)
    for row_item in rows_report:
        row_item["declaration_file"] = declaration_path.name
        row_item["declaration_sha256"] = declaration_sha

    manifest_out = model.read_json(output_manifest_path)
    manifest_out["payload"]["sha256"] = output_sha
    manifest_out["body_local_bakes"] = {
        "schema": model._BODYPARTS_BODY_LOCAL_BAKE_SCHEMA,
        "status": "explicit_visual_owner_frame_bake_not_physics_or_native_admission",
        "mode": "reframe_existing_composed_partitioned_rows",
        "declaration_file": declaration_path.name,
        "declaration_sha256": declaration_sha,
        "registration_sha256": reg_sha,
        "myosim_source_archive_sha256": manifest_out["source"]["myosim_source_archive_sha256"],
        "source003_nha_payload_sha256": model.sha256(nha_path),
        "source003_nhbones_payload_sha256": _sha_bytes(bones_raw),
        "source003_source_export_report_sha256": report_sha,
        "partition_ancestry_sha256": model.sha256(partition_directory / "partition-face-ancestry.json"),
        "payload_sha256": output_sha,
        "proof_paths": {
            "partition_directory": str(partition_directory),
            "partition_ancestry": "partition-face-ancestry.json",
            "partition_meshes": "partition-region-meshes.npz",
            "strict_float32_audit": "strict-float32-partition-audit.json",
            "partition_report": "source-preparation-report.json",
        },
        "shared_interface_map_sha256": declaration["shared_interface_map_sha256"],
        "rows": rows_report,
        "native_admission": False,
        "physics_or_mass_change": False,
    }
    for row_item in rows_report:
        source_anchor = manifest_out["source"]["anchors"][row_item["stable_id"] - 1]
        source_anchor["archive_sha256"] = row_item["bodyparts_archive_sha256"]
        source_anchor["body_local_bake"] = row_item
    model.write_json(output_manifest_path, manifest_out)
    manifest_sha = model.sha256(output_manifest_path)

    # Preserve every receipt field; rebind only typed NHBONES payload references and attach this source proof.
    receipt_out = model.read_json(output_receipt_path)
    receipt_out["payload"]["path"] = str(output_nha_path)
    receipt_out["provenance"]["bones_payload"] = str(output_bones_path)
    receipt_out["provenance"]["bones_payload_sha256"] = output_sha
    receipt_out["functional_bindings"]["bones_payload_sha256"] = output_sha
    for stable_id in _RIBS:
        key = f"intercostal_313_rib{stable_id}_motion_attachment"
        binding = receipt_out["functional_bindings"].get(key)
        if not isinstance(binding, dict) or binding.get("bones_payload_sha256") != _sha_bytes(bones_raw):
            raise model.ImportError(f"source003 receipt rib{stable_id} attachment does not bind the input NHBONES")
        binding["bones_payload_sha256"] = output_sha
    report = {
        "schema": "numi.human.composed-rib-body-local-frame-rebase-report.v1",
        "status": "PASS_SOURCE_REFRAME_NATIVE_EMISSION_AND_MOTION_NOT_ADMITTED",
        "source003": declaration["input_source003"],
        "partition_proof": declaration["partition_proof"],
        "output": {
            "nha_payload_sha256": model.sha256(output_nha_path),
            "nhbones_payload_sha256": output_sha,
            "nhbones_manifest_sha256": manifest_sha,
            "declaration_sha256": declaration_sha,
        },
        "rows": rows_report,
        "residuals": declaration["residuals"],
        "preservation": declaration["preservation"],
        "native_emission": False,
        "retained_motion_admission": False,
    }
    report_path = output_path / "composed-rib-body-local-frame-rebase-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    receipt_out["provenance"]["composed_rib_body_local_frame_rebase"] = {
        "schema": report["schema"],
        "status": report["status"],
        "report_file": report_path.name,
        "report_sha256": model.sha256(report_path),
        "declaration_file": declaration_path.name,
        "declaration_sha256": declaration_sha,
        "source003_nhbones_sha256": _sha_bytes(bones_raw),
        "partition_ancestry_sha256": declaration["partition_proof"]["partition_ancestry_sha256"],
        "physics_or_mass_change": False,
        "native_admission": False,
    }
    model.write_json(output_receipt_path, receipt_out)
    report["output"]["receipt_sha256"] = model.sha256(output_receipt_path)
    return report

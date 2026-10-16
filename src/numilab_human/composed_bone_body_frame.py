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
_SOURCE_EXPORT_PARTITION_SCHEMA = "numi.human.intercostal313-rib48-rib49-current467-local-remesh-face-ancestry.v1"
_RIBS = {
    48: ("FJ3228", "803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603", 12, 868),
    49: ("FJ3229", "631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58", 13, 1769),
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _runtime_attachment_expected_values(
    nha_row: dict[str, Any],
    interface_maps: dict[int, list[list[int]]],
    *,
    nha_stable_id: int,
) -> tuple[int, dict[int, tuple[list[int], int, str]]]:
    """Validate typed NHA/rib maps and derive the counts used by the native hook."""
    if (
        not isinstance(nha_row, dict)
        or type(nha_stable_id) is not int or nha_stable_id != 313
        or type(nha_row.get("body_index")) is not int or nha_row.get("body_index") != 20
        or type(nha_row.get("layer")) is not int or nha_row.get("layer") != 1
        or type(nha_row.get("flags")) is not int or nha_row.get("flags") != 0
        or not isinstance(nha_row.get("vertices6"), (list, np.ndarray))
    ):
        raise model.ImportError("runtime rib attachment requires the exact typed NHA 51010:313 body20 layer1 row")
    vertices6 = np.asarray(nha_row["vertices6"])
    if vertices6.ndim != 2 or vertices6.shape[1] != 6 or not np.isfinite(vertices6).all():
        raise model.ImportError("runtime rib attachment NHA row allocation is malformed")
    local_vertex_count = len(vertices6)
    if local_vertex_count <= 0 or set(interface_maps) != set(_RIBS):
        raise model.ImportError("runtime rib attachment map inventory or NHA row allocation is invalid")
    expected: dict[int, tuple[list[int], int, str]] = {}
    for stable_id in _RIBS:
        pairs = interface_maps.get(stable_id)
        if not isinstance(pairs, list) or not pairs:
            raise model.ImportError(f"runtime rib{stable_id} attachment map is missing")
        rib_ids: set[int] = set()
        nha_ids: set[int] = set()
        for pair in pairs:
            if (
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                or type(pair[0]) is not int or type(pair[1]) is not int
                or pair[0] < 0 or pair[1] < 0 or pair[1] >= local_vertex_count
                or pair[0] in rib_ids or pair[1] in nha_ids
            ):
                raise model.ImportError(f"runtime rib{stable_id} attachment map has invalid or duplicate typed slots")
            rib_ids.add(pair[0])
            nha_ids.add(pair[1])
        ordered = sorted(nha_ids)
        ids_sha = _sha_bytes(b"".join(struct.pack("<I", value) for value in ordered))
        expected[stable_id] = (ordered, len(ordered), ids_sha)
    return local_vertex_count, expected


def bind_intercostal_runtime_attachment_metadata(
    receipt: dict[str, Any],
    nha_row: dict[str, Any],
    interface_maps: dict[int, list[list[int]]],
    *,
    nha_stable_id: int,
    nha_payload_sha256: str,
    bones_payload_sha256: str,
) -> None:
    """Emit exact runtime attachment allocation metadata from parsed owner rows."""
    local_vertex_count, expected = _runtime_attachment_expected_values(
        nha_row, interface_maps, nha_stable_id=nha_stable_id,
    )
    for value in (nha_payload_sha256, bones_payload_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise model.ImportError("runtime rib attachment payload hash is malformed")
    bindings = receipt.get("functional_bindings") if isinstance(receipt, dict) else None
    payload = receipt.get("payload") if isinstance(receipt, dict) else None
    provenance = receipt.get("provenance") if isinstance(receipt, dict) else None
    if not isinstance(bindings, dict) or not isinstance(payload, dict) or not isinstance(provenance, dict):
        raise model.ImportError("runtime rib attachment receipt lacks payload, provenance, or functional bindings")
    payload["sha256"] = nha_payload_sha256
    provenance["bones_payload_sha256"] = bones_payload_sha256
    bindings["bones_payload_sha256"] = bones_payload_sha256
    for stable_id, (member, member_sha, channel, _) in _RIBS.items():
        key = f"intercostal_313_rib{stable_id}_motion_attachment"
        attachment = bindings.get(key)
        surface = attachment.get("surface") if isinstance(attachment, dict) else None
        rib_channel = attachment.get("existing_rib_channel") if isinstance(attachment, dict) else None
        source_partition = attachment.get("source_partition") if isinstance(attachment, dict) else None
        ids, count, ids_sha = expected[stable_id]
        if (
            not isinstance(surface, dict)
            or type(surface.get("semantic")) is not int or surface.get("semantic") != 51010
            or type(surface.get("stable_id")) is not int or surface.get("stable_id") != 313
            or type(surface.get("body_index")) is not int or surface.get("body_index") != 20
            or type(surface.get("layer")) is not int or surface.get("layer") != 1
            or not isinstance(surface.get("interface_local_vertex_ids"), list)
            or surface.get("interface_local_vertex_ids") != ids
            or surface.get("interface_local_vertex_ids_sha256") != ids_sha
            or not isinstance(rib_channel, dict)
            or type(rib_channel.get("stable_id")) is not int or rib_channel.get("stable_id") != stable_id
            or type(rib_channel.get("semantic")) is not int or rib_channel.get("semantic") != 51004
            or type(rib_channel.get("body_index")) is not int or rib_channel.get("body_index") != 20
            or type(rib_channel.get("channel_kind")) is not int or rib_channel.get("channel_kind") != 5
            or type(rib_channel.get("channel_index")) is not int or rib_channel.get("channel_index") != channel
            or rib_channel.get("source_member_id") != member
            or rib_channel.get("source_member_sha256") != member_sha
            or not isinstance(source_partition, dict)
        ):
            raise model.ImportError(f"runtime rib{stable_id} attachment identity or exact interface slots are invalid")
        surface["local_vertex_count"] = local_vertex_count
        source_partition["interface_vertex_count"] = count
        attachment["anatomy_payload_sha256"] = nha_payload_sha256
        attachment["bones_payload_sha256"] = bones_payload_sha256


def validate_intercostal_runtime_attachment_metadata(
    receipt: dict[str, Any],
    nha_row: dict[str, Any],
    interface_maps: dict[int, list[list[int]]],
    *,
    nha_stable_id: int,
    nha_payload_sha256: str,
    bones_payload_sha256: str,
) -> None:
    """Fail closed unless receipt counts and hashes match the decoded owner rows."""
    local_vertex_count, expected = _runtime_attachment_expected_values(
        nha_row, interface_maps, nha_stable_id=nha_stable_id,
    )
    bindings = receipt.get("functional_bindings") if isinstance(receipt, dict) else None
    payload = receipt.get("payload") if isinstance(receipt, dict) else None
    provenance = receipt.get("provenance") if isinstance(receipt, dict) else None
    if (
        not isinstance(bindings, dict) or not isinstance(payload, dict) or not isinstance(provenance, dict)
        or payload.get("sha256") != nha_payload_sha256
        or provenance.get("bones_payload_sha256") != bones_payload_sha256
        or bindings.get("bones_payload_sha256") != bones_payload_sha256
    ):
        raise model.ImportError("runtime rib attachment receipt payload hashes differ from exact owner rows")
    for stable_id in _RIBS:
        attachment = bindings.get(f"intercostal_313_rib{stable_id}_motion_attachment")
        surface = attachment.get("surface") if isinstance(attachment, dict) else None
        source_partition = attachment.get("source_partition") if isinstance(attachment, dict) else None
        ids, count, ids_sha = expected[stable_id]
        if (
            not isinstance(attachment, dict)
            or attachment.get("anatomy_payload_sha256") != nha_payload_sha256
            or attachment.get("bones_payload_sha256") != bones_payload_sha256
            or not isinstance(surface, dict)
            or type(surface.get("local_vertex_count")) is not int
            or surface.get("local_vertex_count") != local_vertex_count
            or surface.get("interface_local_vertex_ids") != ids
            or surface.get("interface_local_vertex_ids_sha256") != ids_sha
            or not isinstance(source_partition, dict)
            or type(source_partition.get("interface_vertex_count")) is not int
            or source_partition.get("interface_vertex_count") != count
        ):
            raise model.ImportError(f"runtime rib{stable_id} attachment counts differ from decoded typed rows")
    copied = json.loads(json.dumps(receipt))
    bind_intercostal_runtime_attachment_metadata(
        copied, nha_row, interface_maps,
        nha_stable_id=nha_stable_id,
        nha_payload_sha256=nha_payload_sha256,
        bones_payload_sha256=bones_payload_sha256,
    )


def _preserve_receipt_common_field_sidecars(
    receipt: dict[str, Any],
    source_directory: Path,
    output_directory: Path,
) -> dict[str, dict[str, Any]]:
    """Copy and hash-check the three relative cardiac field sidecars used by receipts."""
    provenance = receipt.get("provenance") if isinstance(receipt, dict) else None
    cardiac = provenance.get("cardiac_geometry_binding") if isinstance(provenance, dict) else None
    common = cardiac.get("common_field") if isinstance(cardiac, dict) else None
    if not isinstance(common, dict):
        raise model.ImportError("rib receipt lacks the existing common cardiac field binding")
    source_root = source_directory.resolve()
    output_root = output_directory.resolve()
    preserved: dict[str, dict[str, Any]] = {}
    for key in ("map", "domain_boxes", "polynomials"):
        entry = common.get(key)
        if not isinstance(entry, dict):
            raise model.ImportError(f"rib receipt common field omits {key}")
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(relative, str) or not relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in Path(relative).parts)
            or not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha)
        ):
            raise model.ImportError(f"rib receipt common field {key} path or hash is malformed")
        source = (source_root / relative).resolve()
        destination = output_root / relative
        try:
            source.relative_to(source_root)
            destination.resolve(strict=False).relative_to(output_root)
        except ValueError as error:
            raise model.ImportError(f"rib receipt common field {key} escapes its artifact directory") from error
        if not source.is_file() or model.sha256(source) != expected_sha:
            raise model.ImportError(f"rib receipt common field {key} source sidecar is missing or changed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if model.sha256(destination) != expected_sha:
                raise model.ImportError(f"rib receipt common field {key} copied sidecar has changed")
        else:
            shutil.copy2(source, destination)
        if model.sha256(destination) != expected_sha:
            raise model.ImportError(f"rib receipt common field {key} output sidecar failed verification")
        preserved[key] = {"path": relative, "sha256": expected_sha, "bytes": destination.stat().st_size}
    return preserved


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



def _validate_composed_rib_reframe_source003(
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
    preserved_common_field_sidecars = _preserve_receipt_common_field_sidecars(
        receipt, receipt_path.parent, output_path,
    )
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
    bind_intercostal_runtime_attachment_metadata(
        receipt_out,
        nha_row,
        interface_maps,
        nha_stable_id=313,
        nha_payload_sha256=model.sha256(output_nha_path),
        bones_payload_sha256=output_sha,
    )
    validate_intercostal_runtime_attachment_metadata(
        receipt_out,
        nha_row,
        interface_maps,
        nha_stable_id=313,
        nha_payload_sha256=model.sha256(output_nha_path),
        bones_payload_sha256=output_sha,
    )
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
        "preservation": {
            **declaration["preservation"],
            "common_field_sidecars": preserved_common_field_sidecars,
        },
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




def exact_interface_vertex_pairs(
    bone_vertices6: np.ndarray,
    nha_vertices6: np.ndarray,
    nha_vertex_ids: list[int],
    stable_id: int,
) -> list[list[int]]:
    """Bind listed NHA interface slots to unique, byte-identical bone points."""
    if type(stable_id) is not int or stable_id not in _RIBS:
        raise model.ImportError("rib interface map has an unexpected typed stable ID")
    bone = np.asarray(bone_vertices6, dtype="<f4")
    nha = np.asarray(nha_vertices6, dtype="<f4")
    if bone.ndim != 2 or bone.shape[1] != 6 or nha.ndim != 2 or nha.shape[1] != 6:
        raise model.ImportError("rib interface map requires six-float vertex rows")
    if not np.isfinite(bone).all() or not np.isfinite(nha).all():
        raise model.ImportError("rib interface map contains non-finite vertices")
    if any(type(index) is not int or not 0 <= index < len(nha) for index in nha_vertex_ids):
        raise model.ImportError("rib interface map has an out-of-range or untyped NHA slot")
    if len(nha_vertex_ids) != len(set(nha_vertex_ids)):
        raise model.ImportError("rib interface map repeats an NHA slot")
    bone_by_position: dict[bytes, list[int]] = {}
    for index, vertex in enumerate(bone):
        bone_by_position.setdefault(_point_key(vertex[:3]), []).append(index)
    pairs: list[list[int]] = []
    used_bone: set[int] = set()
    for nha_index in nha_vertex_ids:
        matches = bone_by_position.get(_point_key(nha[nha_index, :3]), [])
        if len(matches) != 1 or matches[0] in used_bone:
            raise model.ImportError("rib interface point is absent, duplicated, or not one-to-one")
        bone_index = matches[0]
        if not np.array_equal(bone[bone_index, :3], nha[nha_index, :3]):
            raise model.ImportError("rib interface point is not an exact FP32 coordinate match")
        used_bone.add(bone_index)
        pairs.append([bone_index, nha_index])
    pairs.sort()
    return pairs


def _validate_source_export_partition_rebind(
    artifact: Path,
    manifest: dict[str, Any],
    output_raw: bytes,
    bake_manifest: dict[str, Any],
    declaration: dict[str, Any],
) -> dict[str, Any]:
    """Re-derive source-export 005's current partition rows without rewriting payloads."""
    inputs = declaration.get("input_paths")
    pins = declaration.get("input_hashes")
    proof = declaration.get("partition_proof")
    if not isinstance(inputs, dict) or not isinstance(pins, dict) or not isinstance(proof, dict):
        raise model.ImportError("source-export partition rebind lacks complete input bindings")
    required = {
        "source_export_report", "source_export_receipt", "source_export_manifest", "source_face_crosswalk",
        "partition_ancestry", "partition_meshes", "partition_report", "strict_float32_audit",
        "base_receipt", "base_nha", "base_nhbones", "base_manifest", "registration",
        "actual_capture_pack", "actual_capture_receipt",
    }
    if set(inputs) != required or set(pins) != required:
        raise model.ImportError("source-export partition rebind input inventory is incomplete")
    resolved: dict[str, Path] = {}
    for key in sorted(required):
        value = inputs.get(key)
        expected = pins.get(key)
        if not isinstance(value, str) or not isinstance(expected, str) or len(expected) != 64:
            raise model.ImportError(f"source-export partition rebind pin {key} is malformed")
        path = Path(value).resolve()
        if not path.is_file() or model.sha256(path) != expected:
            raise model.ImportError(f"source-export partition rebind input {key} is missing or changed")
        resolved[key] = path

    source_report = model.read_json(resolved["source_export_report"])
    source_receipt = model.read_json(resolved["source_export_receipt"])
    source_manifest = model.read_json(resolved["source_export_manifest"])
    def manifest_without_rebound_metadata(value: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(value))
        normalized.pop("body_local_bakes", None)
        anchors = normalized.get("source", {}).get("anchors", [])
        for stable_id in (48, 49):
            index = stable_id - 1
            if not isinstance(anchors, list) or index >= len(anchors) or not isinstance(anchors[index], dict):
                raise model.ImportError("source-export partition manifest lacks the typed rib anchor")
            anchors[index].pop("body_local_bake", None)
            anchors[index].pop("derived_visual_partition", None)
        return normalized
    if manifest_without_rebound_metadata(source_manifest) != manifest_without_rebound_metadata(manifest):
        raise model.ImportError("source-export partition rebind changed manifest data outside the two verified row declarations")
    crosswalk_path = resolved["source_face_crosswalk"]
    crosswalk = model.read_json(crosswalk_path)
    if (
        declaration.get("verification_kind") != "source_export005_partition_metadata_rebind.v1"
        or source_report.get("status") != "SOURCE_EXPORT_COMPLETE_NATIVE_AND_MOTION_NOT_ADMITTED"
        or source_report.get("native_admission") is not False
        or source_report.get("retained_motion_admission") is not False
        or crosswalk.get("schema") != "numi.human.intercostal313-rib48-rib49-output-face-crosswalk.v1"
        or source_report.get("source_face_output_crosswalk", {}).get("sha256") != model.sha256(crosswalk_path)
        or source_report.get("source_face_output_crosswalk", {}).get("path") != crosswalk_path.name
    ):
        raise model.ImportError("source-export partition rebind does not identify the pinned 005 export")
    report_inputs = source_report.get("input_pins")
    if not isinstance(report_inputs, dict):
        raise model.ImportError("source-export report lacks its immutable input pin inventory")
    for key in ("base_receipt", "base_nha", "base_nhbones", "base_manifest", "actual_capture_pack", "actual_capture_receipt"):
        if report_inputs.get(str(resolved[key])) != pins[key]:
            raise model.ImportError(f"source-export report does not bind the declared {key} input")
    base_manifest = model.read_json(resolved["base_manifest"])
    registration = base_manifest.get("source", {}).get("registration")
    if (
        not isinstance(registration, dict)
        or registration.get("file") != resolved["registration"].name
        or registration.get("sha256") != pins["registration"]
        or model.sha256(resolved["registration"]) != pins["registration"]
    ):
        raise model.ImportError("source-export registration is not bound by the exact base manifest")
    if (
        source_report.get("base_nha_sha256") != pins["base_nha"]
        or source_report.get("base_nhbones_sha256") != pins["base_nhbones"]
        or source_report.get("base_receipt_sha256") != pins["base_receipt"]
        or source_report.get("actual_capture_pack_sha256") != pins["actual_capture_pack"]
        or source_report.get("actual_capture_receipt_sha256") != pins["actual_capture_receipt"]
    ):
        raise model.ImportError("source-export report base or native-capture lineage changed")
    output_hashes = source_report.get("output_sha256")
    if (
        not isinstance(output_hashes, dict)
        or output_hashes.get("nha") != model.sha256(artifact / "resting-thorax.nhanatomy")
        or output_hashes.get("nhbones") != _sha_bytes(output_raw)
        or output_hashes.get("bone_manifest") != model.sha256(resolved["source_export_manifest"])
    ):
        raise model.ImportError("source-export partition rebind payload or manifest differs from the exact 005 export")
    if (
        source_receipt.get("payload", {}).get("sha256") != output_hashes.get("nha")
        or source_receipt.get("provenance", {}).get("bones_payload_sha256") != output_hashes.get("nhbones")
        or source_report.get("output_sha256", {}).get("receipt") != pins["source_export_receipt"]
        or source_receipt.get("payload", {}).get("input_payload_sha256")
        != model.read_json(resolved["base_receipt"]).get("payload", {}).get("input_payload_sha256")
        or Path(str(source_receipt.get("payload", {}).get("path", ""))).resolve()
        != (resolved["source_export_receipt"].parent / "resting-thorax.nhanatomy").resolve()
        or Path(str(source_receipt.get("provenance", {}).get("bones_payload", ""))).resolve()
        != (resolved["source_export_receipt"].parent / "bodyparts3d-myosim-major-bones.nhbones").resolve()
    ):
        raise model.ImportError("source-export partition rebind receipt does not bind the pinned source and outputs")

    partition = resolved["partition_ancestry"].parent
    if (
        proof.get("schema") != _SOURCE_EXPORT_PARTITION_SCHEMA
        or proof.get("partition_ancestry_sha256") != model.sha256(resolved["partition_ancestry"])
        or proof.get("partition_meshes_sha256") != model.sha256(resolved["partition_meshes"])
        or proof.get("strict_float32_audit_sha256") != model.sha256(resolved["strict_float32_audit"])
        or proof.get("partition_report_sha256") != model.sha256(resolved["partition_report"])
        or proof.get("partition_directory") != str(partition.resolve())
    ):
        raise model.ImportError("source-export partition proof identity changed")
    ancestry = model.read_json(resolved["partition_ancestry"])
    strict = model.read_json(resolved["strict_float32_audit"])
    partition_report = model.read_json(resolved["partition_report"])
    if (
        ancestry.get("schema") != _SOURCE_EXPORT_PARTITION_SCHEMA
        or ancestry.get("declared_interface_face_pairs") != 2914
        or strict.get("interiors_disjoint") is not True
        or strict.get("shared_interface_opposite_chain") is not True
        or strict.get("shared_interface_face_count") != 2932
        or partition_report.get("status") != "PASS_LOCAL_EXACT_FACE_COVERAGE_AND_STRICT_FLOAT32_PARTITION_SOURCE_CANDIDATE"
        or partition_report.get("anatomical_admission") is not False
        or partition_report.get("native_emission") is not False
        or partition_report.get("retained_motion_admission") is not False
        or partition_report.get("ancestry_sha256") != proof.get("partition_ancestry_sha256")
        or partition_report.get("strict_float32_audit_sha256") != proof.get("strict_float32_audit_sha256")
    ):
        raise model.ImportError("source-export partition strict source proof is not a complete pass")

    with np.load(resolved["partition_meshes"], allow_pickle=False) as archive:
        meshes = {name: np.asarray(archive[name]).copy() for name in archive.files}
    muscle_positions = np.asarray(meshes.get("intercostal313_positions_m"), dtype="<f4")
    muscle_faces = np.asarray(meshes.get("intercostal313_faces"), dtype=np.int64)
    union_positions = np.asarray(meshes.get("rib_union_positions_m"), dtype="<f4")
    union_faces = np.asarray(meshes.get("rib_union_faces"), dtype=np.int64)
    if (
        muscle_positions.shape != (8886, 3) or muscle_faces.shape != (17796, 3)
        or union_positions.shape != (6777, 3) or union_faces.shape != (13546, 3)
        or not np.isfinite(muscle_positions).all() or not np.isfinite(union_positions).all()
        or len(crosswalk.get("interface_decomposition", {}).get("pairs", [])) != 2932
    ):
        raise model.ImportError("source-export partition mesh or interface dimensions changed")

    crosswalk_regions = crosswalk.get("regions")
    if not isinstance(crosswalk_regions, dict):
        raise model.ImportError("source-export partition face crosswalk lacks ordered regions")
    muscle_region = crosswalk_regions.get("intercostal313")
    union_region = crosswalk_regions.get("rib_union")
    if not isinstance(muscle_region, list) or not isinstance(union_region, list):
        raise model.ImportError("source-export partition face crosswalk region tables are absent")
    def ordered_region_faces(rows: list[Any], count: int, region_name: str) -> list[dict[str, Any]]:
        if len(rows) != count:
            raise model.ImportError(f"source-export {region_name} face crosswalk cardinality changed")
        out: list[dict[str, Any] | None] = [None] * count
        for item in rows:
            if not isinstance(item, dict):
                raise model.ImportError(f"source-export {region_name} face ancestry is malformed")
            face_id = item.get("output_face_id")
            if type(face_id) is not int or not 0 <= face_id < count or out[face_id] is not None:
                raise model.ImportError(f"source-export {region_name} output face IDs are not a bijection")
            if item.get("output_region") != region_name:
                raise model.ImportError(f"source-export {region_name} ancestry names another region")
            out[face_id] = item
        if any(item is None for item in out):
            raise model.ImportError(f"source-export {region_name} face crosswalk has a gap")
        return [item for item in out if item is not None]
    ordered_muscle = ordered_region_faces(muscle_region, len(muscle_faces), "intercostal313")
    ordered_union = ordered_region_faces(union_region, len(union_faces), "rib_union")
    typed_union: dict[int, list[int]] = {48: [], 49: []}
    for face_id, item in enumerate(ordered_union):
        parent = item.get("typed_source_parent")
        typed = parent.get("typed_row") if isinstance(parent, dict) else None
        if (
            not isinstance(typed, list) or len(typed) != 2
            or type(typed[0]) is not int or typed[0] != 51004
            or type(typed[1]) is not int or typed[1] not in typed_union
        ):
            raise model.ImportError("source-export rib face ancestry has an invalid typed source row")
        if typed[1] not in typed_union:
            raise model.ImportError("source-export rib face ancestry names an unexpected stable ID")
        typed_union[typed[1]].append(face_id)

    base_receipt = model.read_json(resolved["base_receipt"])
    if (
        base_receipt.get("payload", {}).get("sha256") != pins["base_nha"]
        or base_receipt.get("provenance", {}).get("bones_payload_sha256") != pins["base_nhbones"]
    ):
        raise model.ImportError("source-export partition base receipt does not bind the exact base payloads")
    base_nha_header, base_nha_rows = parse_nha_payload(resolved["base_nha"])
    output_nha_path = artifact / "resting-thorax.nhanatomy"
    output_nha_header, output_nha_rows = parse_nha_payload(output_nha_path)
    base_bones_raw = resolved["base_nhbones"].read_bytes()
    base_header, base_records, base_vertex_offset, base_index_offset = _decode_bones(base_bones_raw)
    output_header, output_records, output_vertex_offset, output_index_offset = _decode_bones(output_raw)
    if (
        base_nha_header[:2] != output_nha_header[:2]
        or base_nha_header[2] != output_nha_header[2]
        or base_header[:2] != output_header[:2]
        or base_header[2] != output_header[2]
    ):
        raise model.ImportError("source-export partition rebind changed a payload ABI or typed row count")
    base_bones_by_id = {int(row[5]): row for row in base_records}
    output_bones_by_id = {int(row[5]): row for row in output_records}
    if set(base_bones_by_id) != set(output_bones_by_id) or set(base_nha_rows) != set(output_nha_rows):
        raise model.ImportError("source-export partition rebind changed the surface or bone identity inventory")
    for stable_id in sorted(base_nha_rows):
        if stable_id == 313:
            continue
        before, after = base_nha_rows[stable_id], output_nha_rows[stable_id]
        if (
            before["body_index"] != after["body_index"] or before["layer"] != after["layer"]
            or before["flags"] != after["flags"]
            or not np.array_equal(before["vertices6"], after["vertices6"])
            or not np.array_equal(before["faces"], after["faces"])
        ):
            raise model.ImportError(f"source-export partition rebind changed unrelated NHA row {stable_id}")
    for stable_id in sorted(base_bones_by_id):
        if stable_id in _RIBS:
            continue
        before_rec, after_rec = base_bones_by_id[stable_id], output_bones_by_id[stable_id]
        before_vertices = np.ndarray((before_rec[2], 6), dtype="<f4", buffer=base_bones_raw,
            offset=base_vertex_offset + before_rec[1] * 24)
        after_vertices = np.ndarray((after_rec[2], 6), dtype="<f4", buffer=output_raw,
            offset=output_vertex_offset + after_rec[1] * 24)
        before_faces = np.ndarray((before_rec[4] // 3, 3), dtype="<u4", buffer=base_bones_raw,
            offset=base_index_offset + before_rec[3] * 4).astype(np.int64) - before_rec[1]
        after_faces = np.ndarray((after_rec[4] // 3, 3), dtype="<u4", buffer=output_raw,
            offset=output_index_offset + after_rec[3] * 4).astype(np.int64) - after_rec[1]
        if (
            before_rec[0] != after_rec[0] or before_rec[5] != after_rec[5]
            or before_rec[14] != after_rec[14]
            or not np.array_equal(before_vertices, after_vertices)
            or not np.array_equal(before_faces, after_faces)
        ):
            raise model.ImportError(f"source-export partition rebind changed unrelated NHBONES row {stable_id}")

    body_pose = source_report.get("actual_body20_pose")
    if not isinstance(body_pose, dict) or body_pose.get("body_index") != 20:
        raise model.ImportError("source-export partition rebind lacks the actual typed body20 pose")
    body_position = np.asarray(body_pose.get("position_m"), dtype=np.float32)
    body_quaternion = np.asarray(body_pose.get("quaternion_xyzw"), dtype=np.float32)
    if body_position.shape != (3,) or body_quaternion.shape != (4,) or not np.isfinite(body_position).all() or not np.isfinite(body_quaternion).all():
        raise model.ImportError("source-export partition rebind body20 pose is malformed")
    owner = _load_source003_partition_owner(source_report)

    row_bone_points: dict[int, np.ndarray] = {}
    row_bone_faces: dict[int, np.ndarray] = {}
    row_union_vertex_ids: dict[int, np.ndarray] = {}
    row_union_to_local: dict[int, dict[int, int]] = {}
    row_bone_body_native: dict[int, np.ndarray] = {}
    for stable_id in (48, 49):
        base_rec = base_bones_by_id[stable_id]
        out_rec = output_bones_by_id[stable_id]
        anchor = manifest.get("source", {}).get("anchors", [])[stable_id - 1]
        derived = anchor.get("derived_visual_partition") if isinstance(anchor, dict) else None
        selected_faces = typed_union[stable_id]
        used_union = np.unique(union_faces[np.asarray(selected_faces, dtype=np.int64)].reshape(-1))
        union_to_local = {int(vertex): index for index, vertex in enumerate(used_union.tolist())}
        expected_local_faces = np.asarray([
            [union_to_local[int(vertex)] for vertex in union_faces[face_id]] for face_id in selected_faces
        ], dtype=np.int64)
        actual_global_faces = np.ndarray((out_rec[4] // 3, 3), dtype="<u4", buffer=output_raw,
            offset=output_index_offset + out_rec[3] * 4)
        actual_local_faces = actual_global_faces.astype(np.int64) - out_rec[1]
        expected_row_faces = expected_local_faces + base_rec[2]
        if not np.array_equal(actual_local_faces, expected_row_faces):
            raise model.ImportError(f"source-export rib{stable_id} face order differs from exact typed partition crosswalk")
        if (
            not isinstance(derived, dict)
            or derived.get("stable_id") != stable_id
            or derived.get("partition_ancestry_sha256") != proof.get("partition_ancestry_sha256")
            or derived.get("strict_float32_audit_sha256") != proof.get("strict_float32_audit_sha256")
            or derived.get("source_preparation_report_sha256") != proof.get("partition_report_sha256")
            or derived.get("source_export_report_sha256") != pins["source_export_report"]
            or derived.get("new_compiled_vertex_count") != out_rec[2]
            or derived.get("new_compiled_triangle_count") != out_rec[4] // 3
            or derived.get("original_compiled_vertex_count") != base_rec[2]
            or derived.get("original_compiled_triangle_count") != base_rec[4] // 3
        ):
            raise model.ImportError(f"source-export rib{stable_id} manifest partition ancestry changed")
        expected_appended_vertices = len(used_union)
        if out_rec[2] != base_rec[2] + expected_appended_vertices:
            raise model.ImportError(f"source-export rib{stable_id} partition vertex allocation changed")
        before_prefix = np.ndarray((base_rec[2], 6), dtype="<f4", buffer=base_bones_raw,
            offset=base_vertex_offset + base_rec[1] * 24)
        actual_row = np.ndarray((out_rec[2], 6), dtype="<f4", buffer=output_raw,
            offset=output_vertex_offset + out_rec[1] * 24)
        if not np.array_equal(actual_row[:base_rec[2]], before_prefix):
            raise model.ImportError(f"source-export rib{stable_id} changed its pre-existing vertex prefix")
        world_points = union_positions[used_union].astype(np.float32, copy=True)
        body_local64 = owner.inverse_body64(world_points, body_position, body_quaternion)
        bone_local = owner.inverse_bone64(body_local64, base_rec).astype(np.float32)
        body_native = owner.bone_to_body_f32(bone_local, base_rec)
        expected_normals = owner.mesh_normals(bone_local, expected_local_faces, f"source-export rib{stable_id}")
        expected_attributes = np.column_stack((bone_local, expected_normals)).astype("<f4")
        if not np.array_equal(actual_row[base_rec[2]:], expected_attributes):
            raise model.ImportError(f"source-export rib{stable_id} appended FP32 geometry differs from its partition source")
        row_bone_points[stable_id] = bone_local
        row_bone_faces[stable_id] = expected_local_faces
        row_union_vertex_ids[stable_id] = used_union
        row_union_to_local[stable_id] = union_to_local
        row_bone_body_native[stable_id] = body_native

    output_pairs = crosswalk.get("interface_decomposition", {}).get("pairs")
    decomposition = source_report.get("interface_face_decomposition")
    if (
        not isinstance(output_pairs, list) or len(output_pairs) != 2932
        or decomposition != {"preserved_prior_faces": 2914, "new_reversed_caps": 18, "total_exact_opposed_faces": 2932}
    ):
        raise model.ImportError("source-export reciprocal interface decomposition changed")
    muscle_to_union_by_rib: dict[int, dict[int, int]] = {48: {}, 49: {}}
    mapped_pair_faces: set[tuple[int, int]] = set()
    mapped_union_faces: set[int] = set()
    mapped_muscle_faces: set[int] = set()
    for pair in output_pairs:
        if not isinstance(pair, dict):
            raise model.ImportError("source-export interface pair is malformed")
        typed = pair.get("bone_typed_row")
        sid = typed[1] if isinstance(typed, list) and len(typed) == 2 and typed[0] == 51004 and type(typed[1]) is int else None
        uf, mf = pair.get("rib_union_output_face_id"), pair.get("intercostal313_output_face_id")
        if sid not in (48, 49) or type(uf) is not int or type(mf) is not int or not (0 <= uf < len(union_faces) and 0 <= mf < len(muscle_faces)):
            raise model.ImportError("source-export interface pair has invalid typed face IDs")
        if (uf, mf) in mapped_pair_faces:
            raise model.ImportError("source-export interface face pair is duplicated")
        mapped_pair_faces.add((uf, mf))
        if uf in mapped_union_faces or mf in mapped_muscle_faces:
            raise model.ImportError("source-export interface face is paired more than once")
        mapped_union_faces.add(uf)
        mapped_muscle_faces.add(mf)
        utri, mtri = union_faces[uf], muscle_faces[mf]
        ukeys = [_point_key(union_positions[int(index)]) for index in utri]
        mkeys = [_point_key(muscle_positions[int(index)]) for index in mtri]
        # Exact reversed cyclic order, no coordinate tolerance.
        if len(set(ukeys)) != 3 or len(set(mkeys)) != 3 or not any(
            mkeys == [ukeys[(shift - i) % 3] for i in range(3)] for shift in range(3)
        ):
            raise model.ImportError("source-export interface pair is not an exact reversed FP32 triangle")
        for mvid, mkey in zip(mtri.tolist(), mkeys, strict=True):
            matches = [int(uvid) for uvid, ukey in zip(utri.tolist(), ukeys, strict=True) if ukey == mkey]
            if len(matches) != 1:
                raise model.ImportError("source-export interface triangle corner mapping is ambiguous")
            prior = muscle_to_union_by_rib[sid].get(int(mvid))
            if prior is not None and prior != matches[0]:
                raise model.ImportError("source-export interface NHA corner maps to multiple rib points")
            muscle_to_union_by_rib[sid][int(mvid)] = matches[0]

    if len(mapped_union_faces) != 2932 or len(mapped_muscle_faces) != 2932:
        raise model.ImportError("source-export interface does not cover each face exactly once on both regions")
    base_nha_row = base_nha_rows.get(313)
    out_nha_row = output_nha_rows.get(313)
    if base_nha_row is None or out_nha_row is None:
        raise model.ImportError("source-export partition rebind lacks typed NHA313")
    replace = source_report.get("current_partition_replacement", {})
    retained_count = source_report.get("nha_composition", {}).get("retained_original_faces")
    retired_count = source_report.get("nha_composition", {}).get("retired_original_faces")
    if (
        replace.get("base_nha_row") != [51010, 313]
        or type(retained_count) is not int or type(retired_count) is not int
        or retained_count + retired_count != len(base_nha_row["faces"])
        or replace.get("replaced_current_face_count") != retired_count
        or out_nha_row["body_index"] != 20 or out_nha_row["layer"] != 1 or out_nha_row["flags"] != 0
        or len(out_nha_row["vertices6"]) != len(base_nha_row["vertices6"]) + len(muscle_positions)
        or not np.array_equal(out_nha_row["vertices6"][:len(base_nha_row["vertices6"])], base_nha_row["vertices6"])
        or not np.array_equal(out_nha_row["faces"][:retained_count], base_nha_row["faces"][:retained_count])
    ):
        raise model.ImportError("source-export NHA313 retained source prefix or replacement inventory changed")
    expected_muscle_faces = muscle_faces + len(base_nha_row["vertices6"])
    if not np.array_equal(out_nha_row["faces"][retained_count:], expected_muscle_faces):
        raise model.ImportError("source-export NHA313 appended face order differs from the partition source")
    muscle_body = owner.inverse_body64(muscle_positions, body_position, body_quaternion).astype(np.float32)
    expected_maps: dict[int, list[list[int]]] = {}
    for sid in (48, 49):
        pairs: list[list[int]] = []
        for mvid, union_vid in muscle_to_union_by_rib[sid].items():
            bone_local_index = row_union_to_local[sid].get(union_vid)
            if bone_local_index is None:
                raise model.ImportError("source-export interface point is absent from its typed rib row")
            muscle_body[mvid] = row_bone_body_native[sid][bone_local_index]
            pairs.append([base_bones_by_id[sid][2] + bone_local_index, len(base_nha_row["vertices6"]) + mvid])
        pairs.sort()
        expected_maps[sid] = pairs
    expected_muscle_normals = owner.mesh_normals(muscle_body, muscle_faces, "source-export NHA313 partition")
    expected_muscle_attributes = np.column_stack((muscle_body, expected_muscle_normals)).astype("<f4")
    if not np.array_equal(out_nha_row["vertices6"][len(base_nha_row["vertices6"]):], expected_muscle_attributes):
        raise model.ImportError("source-export NHA313 appended FP32 geometry differs from the exact partition source")

    attachment_maps = declaration.get("shared_interface_local_vertex_pairs")
    if not isinstance(attachment_maps, dict):
        raise model.ImportError("source-export rebind declaration omits its typed interface map")
    combined = bytearray()
    for sid in (48, 49):
        row = next((item for item in bake_manifest.get("rows", []) if isinstance(item, dict) and item.get("stable_id") == sid), None)
        pairs = expected_maps[sid]
        if (
            row is None or row.get("interface_vertex_pairs") != pairs
            or attachment_maps.get(str(sid)) != pairs
            or row.get("interface_vertex_count") != len(pairs)
            or len({pair[0] for pair in pairs}) != len(pairs)
            or len({pair[1] for pair in pairs}) != len(pairs)
        ):
            raise model.ImportError(f"source-export rib{sid} interface local map differs from exact output geometry")
        map_bytes = b"".join(struct.pack("<III", sid, bone_id, nha_id) for bone_id, nha_id in pairs)
        if row.get("interface_vertex_map_sha256") != _sha_bytes(map_bytes):
            raise model.ImportError(f"source-export rib{sid} interface map digest changed")
        combined.extend(map_bytes)
        if row.get("compiled_vertex_count") != output_bones_by_id[sid][2] or row.get("compiled_triangle_count") != output_bones_by_id[sid][4] // 3:
            raise model.ImportError(f"source-export rib{sid} declared row counts differ from payload")
        vertices = np.ndarray((output_bones_by_id[sid][2], 6), dtype="<f4", buffer=output_raw,
            offset=output_vertex_offset + output_bones_by_id[sid][1] * 24)
        faces = np.ndarray((output_bones_by_id[sid][4] // 3, 3), dtype="<u4", buffer=output_raw,
            offset=output_index_offset + output_bones_by_id[sid][3] * 4).astype(np.int64) - output_bones_by_id[sid][1]
        face_sha = _sha_bytes(np.asarray(faces, dtype="<u4").reshape(-1).tobytes())
        if row.get("partition_row_face_indices_sha256") != face_sha or row.get("stored_vertex_attributes_f32_sha256") != _sha_bytes(np.asarray(vertices, dtype="<f4").tobytes()):
            raise model.ImportError(f"source-export rib{sid} stored geometry digest differs from payload")
        if manifest.get("source", {}).get("anchors", [])[sid - 1].get("body_local_bake") != row:
            raise model.ImportError(f"source-export rib{sid} source anchor does not repeat its body-local declaration")
    expected_map_sha = _sha_bytes(bytes(combined))
    if (
        declaration.get("shared_interface_map_sha256") != expected_map_sha
        or bake_manifest.get("shared_interface_map_sha256") != expected_map_sha
        or bake_manifest.get("partition_ancestry_sha256") != proof.get("partition_ancestry_sha256")
        or bake_manifest.get("payload_sha256") != _sha_bytes(output_raw)
    ):
        raise model.ImportError("source-export partition rebind shared map or payload binding changed")
    return {
        "status": "PASS_SOURCE_EXPORT005_PARTITION_AND_BODY20_ROWS_REREAD",
        "checked_untouched_nha_rows": len(base_nha_rows) - 1,
        "checked_untouched_nhbones_rows": len(base_bones_by_id) - 2,
        "checked_partition_rows": [48, 49],
        "checked_partition_faces": {"rib48": int(len(row_bone_faces[48])), "rib49": int(len(row_bone_faces[49])), "NHA313": int(len(muscle_faces))},
        "checked_reciprocal_interface_faces": len(output_pairs),
        "checked_interface_vertices": {"48": len(expected_maps[48]), "49": len(expected_maps[49])},
        "source_export_report_sha256": pins["source_export_report"],
        "partition_ancestry_sha256": proof["partition_ancestry_sha256"],
        "output_nha_sha256": output_hashes["nha"],
        "output_nhbones_sha256": output_hashes["nhbones"],
        "native_emission": False,
        "retained_motion_admission": False,
    }

def validate_composed_rib_reframe(artifact: Path, manifest: dict[str, Any], output_raw: bytes) -> dict[str, Any]:
    artifact = artifact.resolve()
    if "thumb_reference_surface" in manifest:
        source, prior_manifest, prior_raw, patch = validate_thumb_reference_surface(
            artifact, manifest, output_raw)
        prior = validate_composed_rib_reframe(source, prior_manifest, prior_raw)
        result = {**prior, "thumb_reference_surface": patch}
        result["validation_sha256"] = _sha_bytes(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        return result
    if "source_face_retirement" in manifest:
        prior_artifact, prior_manifest, prior_raw, retirement = validate_exact_opposed_bone_source_retirement(
            artifact, manifest, output_raw,
        )
        prior = validate_composed_rib_reframe(prior_artifact, prior_manifest, prior_raw)
        result = {**prior, "source_face_retirement": retirement}
        result["validation_sha256"] = _sha_bytes(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        return result
    bake_manifest = manifest.get("body_local_bakes")
    if isinstance(bake_manifest, dict):
        declaration_file = bake_manifest.get("declaration_file")
        if isinstance(declaration_file, str) and Path(declaration_file).name == declaration_file:
            declaration_path = artifact / declaration_file
            if declaration_path.is_file() and model.sha256(declaration_path) == bake_manifest.get("declaration_sha256"):
                declaration = model.read_json(declaration_path)
                if declaration.get("verification_kind") == "source_export005_partition_metadata_rebind.v1":
                    return _validate_source_export_partition_rebind(
                        artifact, manifest, output_raw, bake_manifest, declaration,
                    )
    return _validate_composed_rib_reframe_source003(artifact, manifest, output_raw)


def retire_exact_opposed_bone_source_faces(
    raw: bytes, *, stable_id: int, source_face_pair: tuple[int, int]
) -> tuple[bytes, dict[str, Any]]:
    """Retire one proven zero-volume internal face pair in the existing ABI.

    Every source vertex, normal, rigid owner and rest transform stays byte exact.
    This is source preparation; exact emitted topology, neighboring structures,
    retained poses and native admission remain separate requirements.
    """
    header, records, vertex_offset, index_offset = _decode_bones(raw)
    if type(stable_id) is not int or stable_id <= 0:
        raise model.ImportError("opposed face retirement needs a typed bone stable ID")
    selected = [i for i, row in enumerate(records) if row[5] == stable_id]
    if len(selected) != 1:
        raise model.ImportError("opposed face retirement needs one source bone row")
    if (not isinstance(source_face_pair, tuple) or len(source_face_pair) != 2
            or any(type(x) is not int or x < 0 for x in source_face_pair)
            or source_face_pair[0] >= source_face_pair[1]):
        raise model.ImportError("opposed source face IDs must be two increasing integers")
    row_index = selected[0]
    row = records[row_index]
    if row[4] % 3 or source_face_pair[1] >= row[4] // 3:
        raise model.ImportError("opposed source face ID is outside the selected bone")
    # Source rows partition the index stream. No shared/overlapping index owner
    # may be silently shifted by this local repair.
    next_index = 0
    for record in records:
        if record[3] != next_index or record[4] % 3:
            raise model.ImportError("bone source index ranges do not partition the stream")
        next_index += record[4]
    if next_index != header[4]:
        raise model.ImportError("bone source index stream is not fully owned")
    triangle = struct.Struct("<3I")
    pair_indices = [
        triangle.unpack_from(raw, index_offset + 4 * (row[3] + 3 * face))
        for face in source_face_pair
    ]
    if any(not row[1] <= i < row[1] + row[2]
           for face in pair_indices for i in face):
        raise model.ImportError("opposed source face references a different bone allocation")
    points = [
        tuple(struct.unpack_from("<3f", raw, vertex_offset + 24 * i)
              for i in face)
        for face in pair_indices
    ]
    if any(not all(math.isfinite(x) for x in point)
           for face in points for point in face):
        raise model.ImportError("opposed source coordinates are nonfinite")
    if len(set(points[0])) != 3:
        raise model.ImportError("opposed source face is degenerate")
    # Exact binary32 coordinate equality, including normal numeric zero identity.
    # No source or emitted distance tolerance is admitted.
    reverse = (points[1][0], points[1][2], points[1][1])
    rotations = [reverse[k:] + reverse[:k] for k in range(3)]
    if points[0] not in rotations:
        raise model.ImportError("source faces are not exactly coincident and oppositely wound")
    from fractions import Fraction
    exact = [tuple(Fraction.from_float(x) for x in point) for point in points[0]]
    u = tuple(exact[1][i] - exact[0][i] for i in range(3))
    w = tuple(exact[2][i] - exact[0][i] for i in range(3))
    cross = (u[1]*w[2] - u[2]*w[1], u[2]*w[0] - u[0]*w[2],
             u[0]*w[1] - u[1]*w[0])
    if not any(cross):
        raise model.ImportError("opposed source face has zero area")
    retired = {row[3] + 3 * face for face in source_face_pair}
    kept_global = [i for i in range(0, header[4], 3) if i not in retired]
    new_header = list(header)
    new_header[4] -= 6
    new_records = []
    for i, record in enumerate(records):
        changed = list(record)
        if i == row_index:
            changed[4] -= 6
        elif i > row_index:
            changed[3] -= 6
        new_records.append(tuple(changed))
    packed = bytearray(_BONE_HEADER.pack(*new_header))
    packed.extend(b"".join(_BONE_RECORD.pack(*record) for record in new_records))
    packed.extend(raw[vertex_offset:index_offset])
    packed.extend(b"".join(raw[index_offset + 4*i:index_offset + 4*i + 12]
                           for i in kept_global))
    result = bytes(packed)
    after_header, after_rows, after_vertex, after_index = _decode_bones(result)
    if result[after_vertex:after_index] != raw[vertex_offset:index_offset]:
        raise model.ImportError("opposed face retirement changed source vertex records")
    for i, (old, new) in enumerate(zip(records, after_rows, strict=True)):
        unchanged = list(new)
        unchanged[3] = old[3]
        unchanged[4] = old[4]
        if tuple(unchanged) != old:
            raise model.ImportError("opposed face retirement changed rigid owner/rest metadata")
        if i != row_index:
            old_faces = raw[index_offset + old[3]*4:index_offset + (old[3]+old[4])*4]
            new_faces = result[after_index + new[3]*4:after_index + (new[3]+new[4])*4]
            if old_faces != new_faces:
                raise model.ImportError("opposed face retirement changed another bone")
    return result, {
        "status": "PASS_EXACT_OPPOSED_SOURCE_FACE_RETIREMENT_ONLY",
        "typed_identity": [51004, stable_id],
        "source_face_ids_retired": list(source_face_pair),
        "retained_source_face_ids": [
            i for i in range(row[4] // 3) if i not in source_face_pair],
        "exact_source_float32_coincidence": True,
        "opposite_winding": True,
        "retired_pair_oriented_volume_sum_exact_zero": True,
        "all_vertex_normal_owner_rest_bytes_unchanged": True,
        "all_other_bone_face_bytes_unchanged": True,
        "physical_mass_or_state_changed": False,
        "anatomical_admission": False,
        "native_admission": False,
        "evidence_boundary": (
            "Source-only zero-volume paired face retirement, not a tolerance weld. "
            "All emitted topology, geometry, attachments, motion and native gates remain.")
    }


def validate_exact_opposed_bone_source_retirement(
    artifact: Path, manifest: dict[str, Any], output_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes, dict[str, Any]]:
    """Re-derive a local zero-volume face retirement from its retained input.

    The original manifest and composed-rib proof stay unchanged. Their validation
    is performed against the original bytes; the current bytes must equal the
    exact owner operation, with no other change to any bone or manifest field.
    """
    repair = manifest.get("source_face_retirement")
    required = {"input_artifact", "input_manifest_sha256", "input_payload_sha256",
                "stable_id", "source_face_pair", "proof"}
    if not isinstance(repair, dict) or set(repair) != required:
        raise model.ImportError("bone face retirement has incomplete provenance")
    source_artifact = Path(repair.get("input_artifact", "")).resolve()
    if source_artifact == artifact.resolve():
        raise model.ImportError("bone face retirement cannot reference itself")
    source_manifest_path = source_artifact / "bodyparts3d-myosim-major-bones.manifest.json"
    if (not source_manifest_path.is_file()
            or model.sha256(source_manifest_path) != repair.get("input_manifest_sha256")):
        raise model.ImportError("bone face retirement input manifest is missing or drifted")
    source_manifest = model.read_json(source_manifest_path)
    if "source_face_retirement" in source_manifest:
        raise model.ImportError("nested bone face retirement requires explicit composition")
    descriptor = source_manifest.get("payload", {})
    filename = descriptor.get("file")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or descriptor.get("sha256") != repair.get("input_payload_sha256")):
        raise model.ImportError("bone face retirement source descriptor is invalid")
    source_path = source_artifact / filename
    if not source_path.is_file() or model.sha256(source_path) != repair.get("input_payload_sha256"):
        raise model.ImportError("bone face retirement source payload is missing or drifted")
    source_raw = source_path.read_bytes()
    pair = repair.get("source_face_pair")
    if not isinstance(pair, list) or len(pair) != 2:
        raise model.ImportError("bone face retirement source face ancestry is incomplete")
    expected_raw, proof = retire_exact_opposed_bone_source_faces(
        source_raw, stable_id=repair.get("stable_id"), source_face_pair=tuple(pair))
    if output_raw != expected_raw or repair.get("proof") != proof:
        raise model.ImportError("bone face retirement bytes or exact ancestry differ")
    expected_manifest = json.loads(json.dumps(source_manifest))
    expected_descriptor = expected_manifest["payload"]
    expected_descriptor["sha256"] = _sha_bytes(expected_raw)
    expected_descriptor["bytes"] = len(expected_raw)
    expected_descriptor["index_count"] = _BONE_HEADER.unpack_from(expected_raw)[4]
    expected_manifest["source_face_retirement"] = repair
    if manifest != expected_manifest:
        raise model.ImportError("bone face retirement changed unrelated manifest or ownership")
    return source_artifact, source_manifest, source_raw, proof


def replace_thumb_reference_surface(
    raw: bytes, *, vertices6: Any, faces: Any,
    face_ancestry: list[dict[str, Any]], protected_source_slots: tuple[int, ...],
) -> tuple[bytes, dict[str, Any]]:
    """Repack the inferred thumb-93 inspection surface in the existing ABI.

    Retain original slots, source identity, rigid frame, and every other local
    bone allocation. This operation grants no geometry or native admission.
    Face ancestry may reference the original thumb or its two joint neighbors;
    borrowed joint faces must be declared as inferred reciprocal caps.
    """
    header, rows, vertex_offset, index_offset = _decode_bones(raw)
    row_index = next((i for i, r in enumerate(rows) if r[5] == 93), None)
    if row_index is None:
        raise model.ImportError("thumb reference patch needs typed bone 93")
    row = rows[row_index]
    verts = np.asarray(vertices6)
    triangles = np.asarray(faces)
    if (verts.ndim != 2 or verts.shape[1] != 6 or len(verts) < row[2]
            or verts.dtype.kind not in "fiu" or not np.isfinite(verts).all()):
        raise model.ImportError("thumb reference patch must retain finite original vertex slots")
    verts = np.asarray(verts, dtype="<f4")
    if (not np.isfinite(verts).all() or triangles.ndim != 2
            or triangles.shape[1] != 3 or not len(triangles)
            or triangles.dtype.kind not in "iu" or triangles.min() < 0
            or triangles.max() >= len(verts)):
        raise model.ImportError("thumb reference patch has invalid Float32 attributes or indices")
    triangles = np.asarray(triangles, dtype="<u4")
    if (not isinstance(protected_source_slots, tuple) or not protected_source_slots
            or any(type(i) is not int or not 0 <= i < row[2] for i in protected_source_slots)
            or len(set(protected_source_slots)) != len(protected_source_slots)):
        raise model.ImportError("thumb reference patch needs distinct protected original attachment slots")
    old = np.frombuffer(raw, dtype="<f4", count=header[3]*6, offset=vertex_offset).reshape(-1, 6)
    original = old[row[1]:row[1]+row[2]]
    slots = list(protected_source_slots)
    if not np.array_equal(verts[slots, :3].view("<u4"), original[slots, :3].view("<u4")):
        raise model.ImportError("thumb reference patch moved a protected attachment")
    if not isinstance(face_ancestry, list) or len(face_ancestry) != len(triangles):
        raise model.ImportError("thumb reference patch needs ancestry for every face")
    source_rows = {int(r[5]): r for r in rows}
    for record in face_ancestry:
        if (not isinstance(record, dict) or type(record.get("source_stable_id")) is not int
                or record["source_stable_id"] not in (88, 93, 94)
                or type(record.get("source_face_index")) is not int
                or not 0 <= record["source_face_index"] < source_rows[record["source_stable_id"]][4]//3
                or type(record.get("orientation")) is not int
                or record["orientation"] not in (-1, 1)):
            raise model.ImportError("thumb reference patch has invalid typed source-face ancestry")
        if record["source_stable_id"] != 93 and (
                record.get("inferred_articular_cap") is not True or record["orientation"] != -1):
            raise model.ImportError("borrowed joint faces require explicit inferred reciprocal caps")
    from .resting_anatomy_verify import prepare_surface
    surface = prepare_surface((51004, 93), verts[:, :3], triangles)
    if not surface.closed or surface.quotient_topology["face_component_count"] != 1:
        raise model.ImportError("thumb reference patch must be one closed oriented manifold")
    if surface.self_audit()["count"]:
        raise model.ImportError("thumb reference patch source Float32 self-intersects")
    vertices = []
    indices = []
    new_rows = []
    vertex_cursor = index_cursor = 0
    for i, current in enumerate(rows):
        if current[1] != sum(r[2] for r in rows[:i]) or current[3] != sum(r[4] for r in rows[:i]):
            raise model.ImportError("bone reference source ranges do not partition the allocation")
        source_vertices = raw[vertex_offset+current[1]*24:vertex_offset+(current[1]+current[2])*24]
        source_indices = np.frombuffer(raw, "<u4", count=current[4], offset=index_offset+current[3]*4)
        if len(source_indices) and (source_indices.min() < current[1]
                                   or source_indices.max() >= current[1]+current[2]):
            raise model.ImportError("bone reference source face escapes its local allocation")
        vertex_bytes = verts.tobytes() if i == row_index else source_vertices
        local_indices = triangles.reshape(-1) if i == row_index else source_indices-current[1]
        updated = list(current)
        updated[1:5] = [vertex_cursor, len(vertex_bytes)//24, index_cursor, len(local_indices)]
        new_rows.append(tuple(updated))
        vertices.append(vertex_bytes)
        indices.append((local_indices+vertex_cursor).astype("<u4").tobytes())
        vertex_cursor += updated[2]
        index_cursor += updated[4]
    if sum(r[2] for r in rows) != header[3] or sum(r[4] for r in rows) != header[4]:
        raise model.ImportError("bone reference source allocation has unowned records")
    new_header = list(header)
    new_header[3:5] = [vertex_cursor, index_cursor]
    output = (_BONE_HEADER.pack(*new_header)
              + b"".join(_BONE_RECORD.pack(*r) for r in new_rows)
              + b"".join(vertices) + b"".join(indices))
    ancestry_bytes = json.dumps(face_ancestry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return output, {
        "status": "PASS_INFERRED_THUMB_REFERENCE_SOURCE_PREPARATION_ONLY",
        "typed_identity": [51004, 93], "original_source_slots_retained": row[2],
        "appended_vertex_count": len(verts)-row[2], "compiled_face_count": len(triangles),
        "protected_source_slots": slots, "face_ancestry_sha256": _sha_bytes(ancestry_bytes),
        "rigid_owner_source_identity_and_rest_transform_unchanged": True,
        "all_other_local_bone_vertex_and_face_bytes_unchanged": True,
        "source_float32_closed_single_component_self_clear": True,
        "inferred_not_measured": True, "physical_mass_or_state_changed": False,
        "anatomical_admission": False, "native_admission": False,
    }


def validate_thumb_reference_surface(
    artifact: Path, manifest: dict[str, Any], output_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes, dict[str, Any]]:
    """Replay the pinned local patch while validating the original rib owner."""
    repair = manifest.get("thumb_reference_surface")
    keys = {"input_artifact", "input_manifest_sha256", "input_payload_sha256",
            "patch_path", "patch_sha256", "ancestry_path", "ancestry_sha256",
            "protected_source_slots", "proof"}
    if not isinstance(repair, dict) or set(repair) != keys:
        raise model.ImportError("thumb reference patch has incomplete provenance")
    source = Path(repair["input_artifact"]).resolve()
    if source == artifact.resolve():
        raise model.ImportError("thumb reference patch cannot reference itself")
    mp = source/"bodyparts3d-myosim-major-bones.manifest.json"
    if not mp.is_file() or model.sha256(mp) != repair["input_manifest_sha256"]:
        raise model.ImportError("thumb reference patch input manifest drifted")
    prior = model.read_json(mp)
    if "thumb_reference_surface" in prior:
        raise model.ImportError("nested thumb reference patches require explicit composition")
    payload = prior.get("payload", {})
    name = payload.get("file")
    if (not isinstance(name, str) or Path(name).name != name
            or payload.get("sha256") != repair["input_payload_sha256"]):
        raise model.ImportError("thumb reference patch source descriptor drifted")
    path = source/name
    if not path.is_file() or model.sha256(path) != repair["input_payload_sha256"]:
        raise model.ImportError("thumb reference patch source payload drifted")
    pinned = []
    for key in ("patch", "ancestry"):
        p = Path(repair[key+"_path"])
        if not p.is_absolute():
            p = artifact/p
        if not p.is_file() or model.sha256(p) != repair[key+"_sha256"]:
            raise model.ImportError("thumb reference patch input asset drifted")
        pinned.append(p)
    ancestry = model.read_json(pinned[1])
    with np.load(pinned[0], allow_pickle=False) as patch:
        if set(patch.files) != {"vertices6", "faces"}:
            raise model.ImportError("thumb reference patch NPZ arrays differ")
        expected, proof = replace_thumb_reference_surface(
            path.read_bytes(), vertices6=patch["vertices6"], faces=patch["faces"],
            face_ancestry=ancestry["face_ancestry"],
            protected_source_slots=tuple(repair["protected_source_slots"]))
    if expected != output_raw or proof != repair["proof"]:
        raise model.ImportError("thumb reference patch bytes or ancestry differ")
    wanted = json.loads(json.dumps(prior))
    h = _BONE_HEADER.unpack_from(expected)
    wanted["payload"].update(sha256=_sha_bytes(expected), bytes=len(expected),
                              vertex_count=h[3], index_count=h[4])
    wanted["thumb_reference_surface"] = repair
    if manifest != wanted:
        raise model.ImportError("thumb reference patch changed unrelated manifest or ownership")
    return source, prior, path.read_bytes(), proof

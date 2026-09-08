"""Bind cooked costal anatomy to a pinned Human reference frame offline.

NHTBIND1 is a compiler input, not a force-ownership or calibration certificate.
The native compiler consumes the original NHCART1 and NHRIGID2 bytes, cooks
registered tetrahedra, and derives the mass partition from its actual nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .model import (
    ImportError, read_json, write_json, _NUMI_HUMAN_COSTAL_CARTILAGE_MEMBERS,
    _NUMI_HUMAN_STERNAL_ATTACHMENT_MEMBERS,
)

HEADER = struct.Struct("<8s6I32s32s32s")
MATRIX = struct.Struct("<16d")
REGION = struct.Struct("<3I")  # donor, sternal anchor, rib anchor


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in value
    ):
        raise ImportError(f"invalid {context}")
    return [float(v) for v in value]


def _matrix(registration: dict[str, Any]) -> list[list[float]]:
    raw = registration.get("coordinate_system", {}).get("global_source_mm_to_myosim_world_m")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ImportError("missing common atlas registration")
    matrix = [_finite_vector(row, 4, "common atlas registration") for row in raw]
    if matrix[3] != [0, 0, 0, 1]:
        raise ImportError("common atlas registration is not affine")
    # NHCART1 is already in metres; source OBJ coordinates were millimetres.
    for row in matrix[:3]:
        for column in range(3):
            row[column] *= 1000.0
    a = matrix
    lengths = [sum(a[r][c] ** 2 for r in range(3)) for c in range(3)]
    scale2 = sum(lengths) / 3
    if not 0.25 <= scale2 <= 4.0 or any(abs(v - scale2) > 1e-10 * scale2 for v in lengths):
        raise ImportError("registration must have one positive anatomical scale")
    if any(abs(sum(a[r][i] * a[r][j] for r in range(3))) > 1e-10 * scale2
           for i in range(3) for j in range(i)):
        raise ImportError("registration contains shear")
    determinant = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                   - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                   + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if determinant <= 0:
        raise ImportError("registration reflection would reverse tissue orientation")
    return matrix


def compile_binding(*, cartilage_manifest: Path, human_manifest: Path,
                    registration: Path, output: Path) -> dict[str, Any]:
    cartilage = read_json(cartilage_manifest)
    human = read_json(human_manifest)
    registered = read_json(registration)
    payload_record = cartilage["payload"]
    rigid_record = human["payloads"]["rigid"]
    payload = (cartilage_manifest.parent / payload_record["file"]).read_bytes()
    rigid = (human_manifest.parent / rigid_record["file"]).read_bytes()
    for name, data, record in (("cartilage", payload, payload_record), ("rigid", rigid, rigid_record)):
        if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
            raise ImportError(f"{name} bytes disagree with their source manifest")
    if len(payload) < 72 or len(rigid) < 224:
        raise ImportError("truncated anatomy payload")
    cart = struct.unpack_from("<8s6I2f32s", payload)
    rigid_header = struct.unpack_from("<8s10I32s", rigid)
    if cart[0] != b"NHCART1\0" or cart[1] != 1 or cart[2] != 14:
        raise ImportError("costal binding requires NHCART1 ABI 1 with fourteen regions")
    if rigid_header[0] != b"NHRIGID2" or rigid_header[1] != 1:
        raise ImportError("costal binding requires NHRIGID2 ABI 1")
    if len(rigid) != (224 + 160 * rigid_header[4] + 144 * rigid_header[5]
                      + 64 * rigid_header[7] + 4 * (rigid_header[6] + rigid_header[7])
                      + 32 * rigid_header[3]):
        raise ImportError("rigid payload has inconsistent record counts")
    if len(payload) != 72 + 64 * cart[2] + 28 * cart[3] + 20 * cart[4]:
        raise ImportError("costal payload has inconsistent record counts")
    if cart[-1].hex() != cartilage["source"]["bodyparts3d_archive"]["sha256"]:
        raise ImportError("cartilage source archive identity drifted")
    if registered["source"]["myosim"]["source"] != human["source"]:
        raise ImportError("registration and Human use different source models")
    matrix = _matrix(registered)
    body_records = human["core_tree"]["source_body_records"]
    if len(body_records) != rigid_header[3]:
        raise ImportError("Human source body manifest coverage drifted")
    map_offset = len(rigid) - 32 * rigid_header[3]
    pose_offset = map_offset + 4 * rigid_header[3]
    bodies = {}
    for i, record in enumerate(body_records):
        index = record["core_body_index"]
        if (type(index) is not int or not 0 <= index < rigid_header[4] or index in bodies
                or index != struct.unpack_from("<I", rigid, map_offset + 4*i)[0]):
            raise ImportError("Human source body mapping disagrees with rigid payload")
        pose = (_finite_vector(record["default_com_position_world_m"], 3, "Human source position")
                + _finite_vector(record["default_inertial_quaternion_world_xyzw"], 4, "Human source orientation"))
        if struct.pack("<7f", *pose) != rigid[pose_offset + 28*i:pose_offset + 28*(i+1)]:
            raise ImportError("Human source pose manifest disagrees with rigid payload")
        bodies[index] = record
    anchors: dict[str, Any] = {}
    for anchor in registered["anchors"]:
        member = anchor["source"]["member_id"]
        if member in anchors:
            raise ImportError(f"duplicate registration member {member}")
        anchors[member] = anchor

    def owner(source: dict[str, Any]) -> int:
        member = source["member_id"]
        if member not in anchors:
            raise ImportError(f"missing source bone registration: {member}")
        anchor = anchors[member]
        if anchor["source"]["member_sha256"] != source["obj_sha256"]:
            raise ImportError(f"source bone mesh identity drifted: {member}")
        target = anchor["target"]
        body = bodies.get(target["core_body_index"])
        if body is None or any(target[k] != body[k] for k in ("source_body_id", "name")):
            raise ImportError(f"source bone owner drifted: {member}")
        # Refresh a historical registration only by proving its named source
        # body pose remains the same; never merely replace its old payload hash.
        for key, length in (("default_com_position_world_m", 3),
                            ("default_inertial_quaternion_world_xyzw", 4)):
            before = _finite_vector(target[key], length, "registration body pose")
            after = _finite_vector(body[key], length, "current body pose")
            error = max(abs(a - b) for a, b in zip(before, after))
            if length == 4:
                error = min(error, max(abs(a + b) for a, b in zip(before, after)))
            if error > 1e-10:
                raise ImportError(f"registration source pose drifted: {member}")
        return target["core_body_index"]

    sternal_sources = cartilage["source"]["sternal_members"]
    if [s["member_id"] for s in sternal_sources] != list(_NUMI_HUMAN_STERNAL_ATTACHMENT_MEMBERS):
        raise ImportError("source sternum coverage or ordering drifted")
    sternal = {owner(s) for s in sternal_sources}
    if len(sternal) != 1:
        raise ImportError("split sternum requires per-node bone correspondence")
    sternal_body = next(iter(sternal))
    ribs = cartilage["source"]["rib_members"]
    if len(ribs) != 14:
        raise ImportError("costal rib source coverage is incomplete")
    regions = []
    for i, rib in enumerate(ribs):
        rib_body = owner(rib)
        if rib_body != sternal_body:
            raise ImportError("articulated thorax requires an explicit volumetric donor partition")
        member = struct.unpack_from("<8s", payload, 72 + i * 64)[0].rstrip(b"\0").decode("ascii")
        expected = _NUMI_HUMAN_COSTAL_CARTILAGE_MEMBERS[i]
        if member != expected[0] or rib["member_id"] != expected[4]:
            raise ImportError("source cartilage to rib correspondence drifted")
        regions.append({"member_id": member, "donor_body": rib_body,
                        "sternal_body": sternal_body, "rib_body": rib_body})
    registration_digest = _digest(registration.read_bytes())
    binary = HEADER.pack(b"NHTBIND1", 1, 14, rigid_header[4], cart[3], cart[4], 0,
                         bytes.fromhex(_digest(payload)), bytes.fromhex(_digest(rigid)),
                         bytes.fromhex(registration_digest))
    binary += MATRIX.pack(*(v for row in matrix for v in row))
    binary += b"".join(REGION.pack(r["donor_body"], r["sternal_body"], r["rib_body"]) for r in regions)
    result = {
        "schema": "HumanPack.costal-tissue-binding.v1",
        "payload": {"file": "costal-tissue.nhtbind", "sha256": _digest(binary), "bytes": len(binary)},
        "inputs": {"cartilage_sha256": _digest(payload), "rigid_sha256": _digest(rigid),
                   "cartilage_manifest_sha256": _digest(cartilage_manifest.read_bytes()),
                   "human_manifest_sha256": _digest(human_manifest.read_bytes()),
                   "registration_sha256": registration_digest},
        "registration": {"atlas_m_to_world_m": matrix,
                         "historical_rigid_sha256": registered["source"]["myosim"]["payloads"]["rigid"]["sha256"],
                         "source_body_pose_continuity_checked": True,
                         "method": "one_common_similarity_no_independent_endpoint_fits",
                         "status": "source_bound_reference_candidate_not_anatomical_accuracy_validation"},
        "regions": regions,
        "mass_ownership": {"measure": "final_native_cooked_lumped_FEM_nodes",
                           "donor_assumption": "costal_volume_is_included_in_source_gross_torso_inertia",
                           "native_partition_required": True,
                           "frame_rebase_required": True},
        "qualification": {"production_owner_fraction": 0.0,
                          "rib_sternum_relative_articulation": False,
                          "tissue_calibration": False,
                          "accepted_root_integration": False},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "costal-tissue.nhtbind").write_bytes(binary)
    write_json(output / "costal-tissue-binding.json", result)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cartilage-manifest", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    try:
        result = compile_binding(cartilage_manifest=args.cartilage_manifest,
                                 human_manifest=args.human_manifest,
                                 registration=args.registration, output=args.output)
    except (KeyError, TypeError, ValueError, struct.error) as error:
        raise ImportError(f"invalid costal binding input: {error}") from error
    print(json.dumps(result, indent=2))
    return 0

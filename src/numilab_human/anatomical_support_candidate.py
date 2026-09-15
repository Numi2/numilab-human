"""Compile a source-bound plantar support candidate for the native Human owner.

The candidate converts MyoSim's authored NHCNT1 foot witnesses into compact
NHCNT2 ellipsoids.  It retains the source body and geometry identities, uses
the exact foot-frame quaternions from the paired NHBONES1 source payload, and
does not claim dynamic contact, internal force balance, standing, recovery, or
walking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.anatomical-support-candidate.v1"
NHCNT1_MAGIC = b"NHCNT1\0\0"
NHCNT2_MAGIC = b"NHCNT2\0\0"
NHBONES_MAGIC = b"NHBONES1"
NHRIGID_MAGIC = b"NHRIGID2"
NHCNT_ABI = 1
NHCNT2_ABI = 2
NHBONES_ABI = 2
NHRIGID_ABI = 1
CONTACT_HEADER = struct.Struct("<8s4I32s7f")
NHCNT1_RECORD = struct.Struct("<2I10f")
NHCNT2_RECORD = struct.Struct("<4I20f")
NHBONES_HEADER = struct.Struct("<8s5I32s")
NHBONES_RECORD = struct.Struct("<6I8f")
NHRIGID_HEADER = struct.Struct("<8s10I32s")

_SOURCE_GEOMETRIES = {
    408: ("bofoot_col1_l", 153),
    377: ("bofoot_col1_r", 139),
    409: ("bofoot_col2_l", 153),
    378: ("bofoot_col2_r", 139),
    405: ("foot_col1_l", 152),
    374: ("foot_col1_r", 138),
    406: ("foot_col3_l", 152),
    375: ("foot_col3_r", 138),
    407: ("foot_col4_l", 152),
    376: ("foot_col4_r", 138),
}
_SUPPORT_BODIES = frozenset({138, 139, 152, 153})
_RADII_M = (0.020, 0.014, 0.008)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("anatomical support candidate: " + message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite(value: Any, context: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{context} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{context} is not finite")
    return result


def _unit_quaternion(value: tuple[float, ...], context: str) -> tuple[float, ...]:
    _require(len(value) == 4, f"{context} is not four-dimensional")
    norm = math.sqrt(sum(component * component for component in value))
    _require(math.isfinite(norm) and abs(norm - 1.0) <= 2.0e-5,
             f"{context} is not a unit quaternion")
    return tuple(component / norm for component in value)


def _rotate_xyzw(quaternion: tuple[float, ...], vector: tuple[float, ...]) -> tuple[float, ...]:
    """Rotate a world vector through the source foot-frame quaternion."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _parse_nhcnt1(raw: bytes) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    _require(len(raw) >= CONTACT_HEADER.size, "NHCNT1 payload is truncated")
    header = CONTACT_HEADER.unpack_from(raw)
    magic, abi, body_count, contact_count, reserved, source_hash, *ground = header
    _require(magic == NHCNT1_MAGIC and abi == NHCNT_ABI,
             "support source is not NHCNT1 ABI 1")
    _require(reserved == 0 and contact_count == len(_SOURCE_GEOMETRIES),
             "support source inventory is not the reviewed ten-witness set")
    expected = CONTACT_HEADER.size + NHCNT1_RECORD.size * contact_count
    _require(len(raw) == expected, "NHCNT1 payload length is invalid")
    point = tuple(_finite(value, "support ground point") for value in ground[:3])
    normal = tuple(_finite(value, "support ground normal") for value in ground[3:6])
    norm = math.sqrt(sum(value * value for value in normal))
    _require(abs(norm - 1.0) <= 2.0e-5, "support ground normal is not unit length")
    friction = _finite(ground[6], "support ground friction")
    _require(friction >= 0.0, "support ground friction is negative")
    records = [NHCNT1_RECORD.unpack_from(raw, CONTACT_HEADER.size + i * NHCNT1_RECORD.size)
               for i in range(contact_count)]
    seen: set[int] = set()
    for index, record in enumerate(records):
        body, geometry = record[:2]
        _require(geometry in _SOURCE_GEOMETRIES, f"record {index} has an unknown source geometry")
        _require(geometry not in seen, f"source geometry {geometry} is duplicated")
        seen.add(geometry)
        expected_name, expected_body = _SOURCE_GEOMETRIES[geometry]
        _require(body == expected_body,
                 f"source geometry {geometry} ({expected_name}) is bound to body {body}")
        for lane, value in enumerate(record[2:], start=2):
            _finite(value, f"NHCNT1 record {index} lane {lane}")
        _require(record[8] >= 0.0, f"NHCNT1 record {index} friction is negative")
    _require(seen == set(_SOURCE_GEOMETRIES), "support source inventory is incomplete")
    return {
        "magic": magic.decode("ascii").rstrip("\0"),
        "payload_abi": abi,
        "body_count": body_count,
        "contact_count": contact_count,
        "embedded_source_sha256": source_hash.hex(),
        "ground_point_world_m": list(point),
        "ground_normal_world": list(normal),
        "ground_friction_tangential": friction,
    }, records


def _parse_nhbones(raw: bytes, source_hash: str) -> tuple[dict[str, Any], dict[int, tuple[float, ...]]]:
    _require(len(raw) >= NHBONES_HEADER.size, "NHBONES1 payload is truncated")
    header = NHBONES_HEADER.unpack_from(raw)
    magic, abi, bone_count, vertex_count, index_count, fingerprint, embedded_source = header
    _require(magic == NHBONES_MAGIC and abi == NHBONES_ABI,
             "bone source is not NHBONES1 ABI 2")
    _require(embedded_source.hex() == source_hash,
             "NHCNT1 and NHBONES1 source archive hashes differ")
    expected = (NHBONES_HEADER.size + NHBONES_RECORD.size * bone_count
                + 24 * vertex_count + 4 * index_count)
    _require(len(raw) == expected, "NHBONES1 payload length is invalid")
    orientations: dict[int, tuple[float, ...]] = {}
    for index in range(bone_count):
        record = NHBONES_RECORD.unpack_from(raw, NHBONES_HEADER.size + index * NHBONES_RECORD.size)
        body, *_, stable_id, tx, ty, tz, qx, qy, qz, qw, scale = record
        _require(stable_id == index + 1, "NHBONES1 stable record identity is not contiguous")
        _finite(tx, "NHBONES1 translation x"); _finite(ty, "NHBONES1 translation y")
        _finite(tz, "NHBONES1 translation z"); _finite(scale, "NHBONES1 scale")
        _require(scale > 0.0, "NHBONES1 scale is not positive")
        q = _unit_quaternion((qx, qy, qz, qw), f"NHBONES1 body {body} orientation")
        if body not in _SUPPORT_BODIES:
            continue
        prior = orientations.get(body)
        if prior is None:
            orientations[body] = q
        else:
            dot = abs(sum(prior[lane] * q[lane] for lane in range(4)))
            _require(dot >= 1.0 - 2.0e-5,
                     f"NHBONES1 body {body} has inconsistent foot-frame orientations")
    _require(set(orientations) == set(_SUPPORT_BODIES),
             "NHBONES1 does not register all calcaneus/toe support bodies")
    return {
        "magic": magic.decode("ascii"),
        "payload_abi": abi,
        "bone_count": bone_count,
        "vertex_count": vertex_count,
        "index_count": index_count,
        "registration_fingerprint32": f"{fingerprint:08x}",
        "embedded_source_sha256": embedded_source.hex(),
    }, orientations


def _parse_nhrigid(raw: bytes, source_hash: str) -> dict[str, Any]:
    _require(len(raw) >= NHRIGID_HEADER.size, "NHRIGID2 payload is truncated")
    header = NHRIGID_HEADER.unpack_from(raw)
    magic, abi, engine_abi, source_body_count, body_count, joint_count, nq, nv, reserved0, virtual_count, reserved1, embedded_source = header
    _require(magic == NHRIGID_MAGIC and abi == NHRIGID_ABI,
             "rigid source is not NHRIGID2 ABI 1")
    _require(reserved0 == 0 and reserved1 == 0 and embedded_source.hex() == source_hash,
             "NHRIGID2 source identity is not closed")
    _require(body_count == 157 and source_body_count > 0 and joint_count > 0 and nq == nv + 1,
             "NHRIGID2 is not the reviewed floating full-body source")
    return {
        "magic": magic.decode("ascii"), "payload_abi": abi, "engine_abi": engine_abi,
        "source_body_count": source_body_count, "engine_body_count": body_count,
        "joint_count": joint_count, "nq": nq, "nv": nv, "virtual_body_count": virtual_count,
        "embedded_source_sha256": embedded_source.hex(),
    }


def compile_anatomical_support_candidate(
    support_path: Path,
    bones_path: Path,
    rigid_path: Path,
    *,
    radii_m: tuple[float, float, float] = _RADII_M,
) -> tuple[dict[str, Any], bytes]:
    """Return deterministic NHCNT2 metadata and bytes from exact source payloads."""
    for value, name in ((support_path, "NHCNT1"), (bones_path, "NHBONES1"), (rigid_path, "NHRIGID2")):
        _require(Path(value).is_file() and not Path(value).is_symlink(), f"{name} source is unavailable")
    support_raw, bones_raw, rigid_raw = (Path(value).read_bytes() for value in (support_path, bones_path, rigid_path))
    support, records = _parse_nhcnt1(support_raw)
    source_hash = support["embedded_source_sha256"]
    bones, orientations = _parse_nhbones(bones_raw, source_hash)
    rigid = _parse_nhrigid(rigid_raw, source_hash)
    _require(len(radii_m) == 3 and all(math.isfinite(float(value)) and float(value) > 0.0 for value in radii_m),
             "ellipsoid radii are not positive finite values")
    radii = tuple(float(value) for value in radii_m)
    target_gap = min(float(record[9]) for record in records)
    output_records: list[bytes] = []
    manifest_records: list[dict[str, Any]] = []
    normal = tuple(support["ground_normal_world"])
    for index, record in enumerate(records):
        body, geometry = record[:2]
        name, _ = _SOURCE_GEOMETRIES[geometry]
        source_center = tuple(float(value) for value in record[2:5])
        source_gap = float(record[9])
        q = orientations[body]
        displacement = _rotate_xyzw(q, tuple(
            normal[axis] * (radii[2] + target_gap - source_gap) for axis in range(3)
        ))
        center = tuple(source_center[axis] + displacement[axis] for axis in range(3))
        friction = float(record[8])
        output_records.append(struct.pack(
            "<4I20f", body, geometry, 3, 0,
            *center, 0.0, *center, friction, target_gap, target_gap, 0.0, 0.0,
            *q, *radii, 0.0,
        ))
        manifest_records.append({
            "source_geometry_id": geometry, "source_name": name, "source_body_index": body,
            "source_gap_m": source_gap, "target_gap_m": target_gap,
            "center_local_com_m": list(center), "orientation_xyzw": list(q),
            "radii_m": list(radii), "friction_tangential": friction,
        })
    payload = CONTACT_HEADER.pack(
        NHCNT2_MAGIC, NHCNT2_ABI, support["body_count"], len(output_records), 0,
        bytes.fromhex(source_hash), *support["ground_point_world_m"],
        *support["ground_normal_world"], support["ground_friction_tangential"],
    ) + b"".join(output_records)
    _require(len(payload) == CONTACT_HEADER.size + NHCNT2_RECORD.size * len(output_records),
             "NHCNT2 payload length is invalid")
    metadata = {
        "schema": SCHEMA,
        "status": "candidate",
        "id": "registered_plantar_ellipsoid_support_candidate",
        "subject": "one adult male source package",
        "payload": {
            "file": "support-plantar-ellipsoid.nhcnt", "magic": "NHCNT2",
            "payload_abi": NHCNT2_ABI, "bytes": len(payload), "sha256": _sha(payload),
            "primitive_count": len(output_records), "expanded_contact_count": len(output_records),
        },
        "source": {
            "support_contact": {"file": Path(support_path).name, "sha256": _sha(support_raw), **support},
            "bones": {"file": Path(bones_path).name, "sha256": _sha(bones_raw), **bones},
            "rigid": {"file": Path(rigid_path).name, "sha256": _sha(rigid_raw), **rigid},
        },
        "ground": {
            "point_world_m": support["ground_point_world_m"],
            "normal_world": support["ground_normal_world"],
            "friction_tangential": support["ground_friction_tangential"],
        },
        "geometry": {
            "kind": "ellipsoid", "radii_m": list(radii), "target_min_gap_m": target_gap,
            "orientation_source": "paired NHBONES1 source foot-frame quaternion",
            "records": manifest_records,
        },
        "counts": {"support_bodies": len(_SUPPORT_BODIES), "source_witnesses": len(records),
                   "candidate_primitives": len(output_records)},
        "qualification": {
            "source_geometry_identity_bound": True,
            "source_foot_frame_bound": True,
            "anatomical_support_surface_candidate": True,
            "friction_bound": True,
            "static_unilateral_support_wrench": False,
            "dynamic_contact": False,
            "internal_generalized_equilibrium": False,
            "standing": False, "recovery": False, "walking": False,
        },
        "boundary": (
            "Source-bound registered plantar ellipsoid candidate derived from ten authored MyoSim "
            "foot witnesses and exact NHBONES1 foot frames. Native static wrench qualification, "
            "internal generalized equilibrium, dynamic contact, sustained standing, recovery and "
            "walking remain separate claims. No skin, fat, material, mass or subject calibration "
            "owner is assigned."
        ),
    }
    return metadata, payload


def _immutable_write(path: Path, raw: bytes) -> None:
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == raw, f"immutable output drifted: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def run(arguments: argparse.Namespace) -> int:
    metadata, payload = compile_anatomical_support_candidate(
        arguments.support_contact, arguments.bones, arguments.rigid,
    )
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _immutable_write(output / metadata["payload"]["file"], payload)
    receipt = canonical(metadata) + b"\n"
    _immutable_write(output / "receipt-v1.json", receipt)
    print(json.dumps({"schema": SCHEMA, "status": metadata["status"],
                      "payload_sha256": metadata["payload"]["sha256"],
                      "receipt_sha256": _sha(receipt), "output": str(output)}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--support-contact", type=Path, required=True)
    parser.add_argument("--bones", type=Path, required=True)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, struct.error, ValueError, TypeError) as error:
        parser.exit(2, f"anatomical-support-candidate: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

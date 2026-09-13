"""Bind registered vessel surfaces to the exact MyoSim source body frame.

This owner closes only the source-body/frame relationship.  It deliberately
does not turn a BodyParts3D surface into a tube, a lumen, a material field, a
blood mass owner, or a pressure-coupled mechanical body.
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
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-vessel-body-link-registration.v1"
RIGID_HEADER = struct.Struct("<8s10I32s")
EXPECTED_BODY_LINKS = {
    "FJ1932": "Abdomen",
    "FJ3411": "torso",
    "FJ3413": "torso",
    "FJ3427": "torso",
    "FJ3441": "Abdomen",
    "FJ3645": "torso",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ vessel body-link registration: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} is malformed")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _read(path: Path) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"input is not an immutable regular file: {path}")
    raw = path.read_bytes()
    require(path.stat().st_size == len(raw), f"input changed while reading: {path}")
    return raw


def _validate_rigid(manifest_path: Path, manifest: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    source = manifest.get("source")
    require(isinstance(source, dict), "Human manifest has no source identity")
    require(manifest.get("schema") == "numi.human.myosim-fullbody-reference.v1",
            "unsupported Human manifest schema")
    record = manifest.get("payloads", {}).get("rigid")
    require(isinstance(record, dict), "Human manifest has no rigid payload record")
    name = record.get("file")
    require(isinstance(name, str) and Path(name).name == name and name not in {"", ".", ".."},
            "rigid payload is not a sibling file")
    rigid_path = manifest_path.parent / name
    rigid = _read(rigid_path)
    require(len(rigid) == record.get("bytes") and _sha256(rigid) == record.get("sha256"),
            "rigid payload disagrees with the Human manifest")
    require(len(rigid) >= RIGID_HEADER.size, "rigid payload is truncated")
    header = RIGID_HEADER.unpack_from(rigid)
    (magic, abi, version, source_count, body_count, joints, nq, nv,
     flags, reserved, reserved2, archive_sha) = header
    require(magic == b"NHRIGID2" and abi == 1 and version == 5 and flags == 0
            and reserved2 == 0,
            "unsupported NHRIGID2 header")
    require(0 < source_count <= body_count <= 4096 and 0 < nq <= 8192 and 0 < nv <= 4096,
            "invalid NHRIGID2 dimensions")
    expected = (224 + 160 * body_count + 144 * joints + 64 * nv
                + 4 * (nq + nv) + 32 * source_count)
    require(len(rigid) == expected, "NHRIGID2 record count mismatch")
    archive_hex = archive_sha.hex()
    require(archive_hex == source.get("archive_sha256"), "rigid/source archive identity mismatch")
    source_records = manifest.get("core_tree", {}).get("source_body_records")
    require(isinstance(source_records, list) and len(source_records) == source_count,
            "Human source body catalog coverage disagrees with NHRIGID2")

    map_offset = len(rigid) - 32 * source_count
    pose_offset = map_offset + 4 * source_count
    by_name: dict[str, dict[str, Any]] = {}
    seen_ids: set[int] = set()
    seen_core: set[int] = set()
    for index, row in enumerate(source_records):
        require(isinstance(row, dict), f"source body record {index} is malformed")
        source_id = row.get("source_body_id")
        core_index = row.get("core_body_index")
        name = row.get("name")
        require(isinstance(source_id, int) and not isinstance(source_id, bool)
                and 0 <= source_id < 2**32 and source_id not in seen_ids,
                f"source body identity {index} is invalid")
        require(isinstance(core_index, int) and not isinstance(core_index, bool)
                and 0 <= core_index < body_count and core_index not in seen_core,
                f"core body identity {index} is invalid")
        require(isinstance(name, str) and bool(name) and name not in by_name,
                f"source body name {index} is invalid")
        require(struct.unpack_from("<I", rigid, map_offset + 4 * index)[0] == core_index,
                f"source/core mapping disagrees for {name}")
        pose = (_vector(row.get("default_com_position_world_m"), 3, f"{name} COM")
                + _vector(row.get("default_inertial_quaternion_world_xyzw"), 4, f"{name} quaternion"))
        require(struct.pack("<7f", *pose) == rigid[pose_offset + 28 * index:pose_offset + 28 * (index + 1)],
                f"source pose disagrees with NHRIGID2 for {name}")
        by_name[name] = {
            "source_body_id": source_id,
            "core_body_index": core_index,
            "source_record_index": index,
            "default_com_position_world_m": pose[:3],
            "default_inertial_quaternion_world_xyzw": pose[3:],
        }
        seen_ids.add(source_id)
        seen_core.add(core_index)
    return rigid, {
        "archive_sha256": archive_hex,
        "sha256": _sha256(rigid),
        "bytes": len(rigid),
        "source_count": source_count,
        "body_count": body_count,
        "joint_count": joints,
        "nq": nq,
        "nv": nv,
        "bodies": by_name,
    }


def load_body_catalog(human_manifest: Path) -> dict[str, Any]:
    """Load the validated source/core body catalog from a Human manifest.

    The returned catalog is frame metadata only.  It contains no mass or
    tissue mechanics admission.
    """
    human_manifest = Path(human_manifest)
    manifest = read_json(human_manifest)
    _, info = _validate_rigid(human_manifest, manifest)
    return info


def compile_body_links(*, registration: Path, human_manifest: Path) -> dict[str, Any]:
    registration = Path(registration)
    human_manifest = Path(human_manifest)
    base = read_json(registration)
    require(base.get("schema") == "HumanPack.organ-vessel-registration.v1",
            "unsupported source vessel registration schema")
    require(base.get("qualification", {}).get("source_to_world_frame_registered") is True,
            "source vessel registration is not frame-qualified")
    bindings = base.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == 6, "source vessel registration must contain six bindings")
    rigid, rigid_info = _validate_rigid(human_manifest, read_json(human_manifest))
    del rigid  # the bytes are fully validated; only their identity enters the receipt
    manifest = read_json(human_manifest)
    # The BodyParts3D vessel archive and MyoSim source archive are independent
    # authorities.  Their identities remain separate in the receipt; the
    # binding is through the explicit named body field in the vessel map.

    linked: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for row in sorted(bindings, key=lambda item: item.get("member_id", "")):
        require(isinstance(row, dict), "source vessel binding is malformed")
        member = row.get("member_id")
        body_name = row.get("myosim_body")
        require(isinstance(member, str) and member and member not in seen_members,
                "source vessel member identity is invalid")
        require(isinstance(body_name, str) and body_name, f"missing MyoSim body link for {member}")
        require(EXPECTED_BODY_LINKS.get(member) == body_name,
                f"source vessel body link disagrees with the pinned anatomy map: {member}")
        body = rigid_info["bodies"].get(body_name)
        require(body is not None, f"MyoSim body link is absent from NHRIGID2: {body_name}")
        linked.append({
            "member_id": member,
            "source_name": row["source_name"],
            "region_id": row["region_id"],
            "semantic_id": row["semantic_id"],
            "myosim_body": body_name,
            "source_body_id": body["source_body_id"],
            "source_record_index": body["source_record_index"],
            "core_body_index": body["core_body_index"],
            "default_com_position_world_m": body["default_com_position_world_m"],
            "default_inertial_quaternion_world_xyzw": body["default_inertial_quaternion_world_xyzw"],
            "body_link_registration": True,
            "mechanical_mass_owner": None,
            "tubular_field_registered": False,
            "centreline_registered": False,
            "cross_section_area_m2": None,
            "material_density_kg_per_m3": None,
            "pressure_gradient_momentum_transfer": False,
            "subject_calibration": False,
        })
        seen_members.add(member)

    identity = {
        "source_registration_sha256": _sha256(registration.read_bytes()),
        "human_manifest_sha256": _sha256(human_manifest.read_bytes()),
        "rigid_payload_sha256": rigid_info["sha256"],
        "source_archive_sha256": rigid_info["archive_sha256"],
        "bindings": linked,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-vessel-body-links.1",
        "source": {
            "vessel_registration": str(registration.relative_to(ROOT)) if registration.is_relative_to(ROOT) else str(registration),
            "human_manifest": str(human_manifest.relative_to(ROOT)) if human_manifest.is_relative_to(ROOT) else str(human_manifest),
            "human_manifest_sha256": identity["human_manifest_sha256"],
            "rigid_payload_sha256": rigid_info["sha256"],
            "source_archive_sha256": rigid_info["archive_sha256"],
            "rigid_payload_abi": 1,
            "body_count": rigid_info["body_count"],
            "source_body_count": rigid_info["source_count"],
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "bindings": linked,
        "qualification": {
            "source_to_world_frame_registered": True,
            "body_link_registration": True,
            "body_link_transform_registered": True,
            "tubular_vessel_field": False,
            "centreline_and_area": False,
            "material_density_calibrated": False,
            "blood_mass_owner": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Six source-bound BodyParts3D vessel surfaces now carry exact named "
            "MyoSim source/core body links and default body-frame transforms from "
            "the hash-bound NHRIGID2 payload. This is frame/body bookkeeping only: "
            "it does not assign a vessel tube, lumen area, wall material, density, "
            "blood mass, pressure reaction, tissue exchange, or subject calibration."
        ),
    }
    canonical(result)
    return result


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_body_links(registration=args.registration, human_manifest=args.human_manifest)
        digest = immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "bindings": len(result["bindings"]),
                          "body_link_registration": True}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"organ vessel body-link registration: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

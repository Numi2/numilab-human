"""Bind the native torso anatomy surfaces to exact MyoSim body frames.

This receipt joins the already-qualified BodyParts3D visual payload to the
hash-bound MyoSim source/core body catalog.  It is provenance and frame
bookkeeping only: it does not create organ mechanics, vessel tube mechanics,
neural mechanics, mass, materials, pressure, or calibrated physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, load_anatomy, read_json
from .vessel_body_links import load_body_catalog

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-torso-body-link-registration.v1"
MAP = ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json"
VISUAL_MANIFEST = ROOT / "Docs/media/native-torso-anatomy-20260913/bodyparts3d-myosim-torso-anatomy.manifest.json"
NATIVE_RECEIPT = ROOT / "Docs/media/native-torso-anatomy-20260913/receipt.json"
HUMAN_MANIFEST = ROOT / "Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("torso anatomy body-link registration: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    value = read_json(path)
    require(isinstance(value, dict), f"input is not an object: {path}")
    return value


def _hex(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"{label} is not a lowercase SHA-256")
    return value


def compile_body_links(*, sources: Path = ROOT / "Sources",
                       source_lock: Path = ROOT / "sources.lock.json",
                       anatomy_map: Path = MAP,
                       visual_manifest: Path = VISUAL_MANIFEST,
                       native_receipt: Path = NATIVE_RECEIPT,
                       human_manifest: Path = HUMAN_MANIFEST) -> dict[str, Any]:
    mapping = _read_json(Path(anatomy_map))
    require(mapping.get("schema") == "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1",
            "unsupported torso anatomy map schema")
    entries = mapping.get("entries")
    require(isinstance(entries, list) and len(entries) == 12,
            "torso anatomy map must contain twelve surfaces")

    anatomy = load_anatomy(Path(sources), Path(source_lock))
    by_member: dict[str, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "torso anatomy map entry is malformed")
        required = {"concept_id", "source_name", "member_id", "hierarchy", "layer", "myosim_body"}
        require(required <= entry.keys(), "torso anatomy map entry is incomplete")
        require(entry["hierarchy"] == "part_of", f"unsupported source hierarchy: {entry.get('member_id')}")
        member = entry["member_id"]
        require(isinstance(member, str) and member and member not in by_member,
                "duplicate torso anatomy member")
        relation = anatomy["tables"].get("part_of", {})
        key = (entry["concept_id"], entry["source_name"])
        require(member in relation.get(key, set()),
                f"torso anatomy source relation drifted: {member}")
        by_member[member] = entry

    payload = _read_json(Path(visual_manifest))
    require(payload.get("schema") == "numi.human.bodyparts3d-myosim-torso-anatomy-visual-payload.v1",
            "unsupported torso visual payload schema")
    source = payload.get("source")
    require(isinstance(source, dict), "torso visual payload has no source identity")
    source_map = source.get("surface_map")
    require(isinstance(source_map, dict) and source_map.get("file") == Path(anatomy_map).name and
            source_map.get("sha256") == _sha256(Path(anatomy_map).read_bytes()),
            "torso visual payload is not bound to the current anatomy map")
    source_myosim = source.get("myosim_manifest")
    require(isinstance(source_myosim, dict) and
            source_myosim.get("file") == Path(human_manifest).name and
            source_myosim.get("sha256") == _sha256(Path(human_manifest).read_bytes()),
            "torso visual payload is not bound to the authoritative MyoSim manifest")
    source_lock_doc = _read_json(Path(source_lock))
    partof_hash = source_lock_doc["sources"]["bodyparts3d_4"]["files"][
        "partof_BP3D_4.0_obj_99.zip"
    ]["sha256"]
    require(source.get("myosim_source_archive_sha256") ==
            _read_json(Path(human_manifest)).get("source", {}).get("archive_sha256"),
            "torso visual payload MyoSim archive identity drifted")
    bodyparts = source.get("bodyparts")
    require(isinstance(bodyparts, dict), "torso visual payload has no BodyParts3D identity")
    archives = bodyparts.get("archives")
    require(isinstance(archives, list) and any(
        isinstance(row, dict) and row.get("file") == "partof_BP3D_4.0_obj_99.zip" and
        row.get("sha256") == partof_hash for row in archives),
        "torso visual payload BodyParts3D archive identity drifted")

    visual_surfaces = source.get("surfaces")
    require(isinstance(visual_surfaces, list) and len(visual_surfaces) == len(entries),
            "torso visual payload surface count changed")
    visual_by_member: dict[str, dict[str, Any]] = {}
    for row in visual_surfaces:
        require(isinstance(row, dict), "torso visual surface is malformed")
        member = row.get("member_id")
        require(isinstance(member, str) and member in by_member and member not in visual_by_member,
                f"torso visual surface identity is invalid: {member}")
        entry = by_member[member]
        for key in ("concept_id", "label", "layer", "myosim_body"):
            expected = entry["source_name"] if key == "label" else entry[key]
            require(row.get(key) == expected, f"torso visual surface {member} disagrees on {key}")
        _hex(row.get("member_sha256"), f"torso visual member {member}")
        visual_by_member[member] = row

    native = _read_json(Path(native_receipt))
    require(native.get("schema") == "numi.human.native-torso-anatomy-visual-evidence.v1",
            "unsupported native torso receipt schema")
    native_payload = native.get("payload", {})
    require(native_payload.get("manifest", {}).get("sha256") == _sha256(Path(visual_manifest).read_bytes()),
            "native torso receipt manifest identity drifted")
    require(native.get("validation", {}).get("visual_probe") == "passed" and
            native.get("qualification", {}).get("source_to_world_visual_registration") is True,
            "native torso receipt is not visually qualified")

    catalog = load_body_catalog(Path(human_manifest))
    linked: list[dict[str, Any]] = []
    for member in sorted(by_member):
        entry = by_member[member]
        visual = visual_by_member[member]
        body_name = entry["myosim_body"]
        body = catalog["bodies"].get(body_name)
        require(body is not None, f"MyoSim body link is absent from NHRIGID2: {body_name}")
        linked.append({
            "member_id": member,
            "concept_id": entry["concept_id"],
            "source_name": entry["source_name"],
            "layer": entry["layer"],
            "source_member_sha256": visual["member_sha256"],
            "visual_stable_id": visual.get("stable_id"),
            "myosim_body": body_name,
            "source_body_id": body["source_body_id"],
            "source_record_index": body["source_record_index"],
            "core_body_index": body["core_body_index"],
            "default_com_position_world_m": body["default_com_position_world_m"],
            "default_inertial_quaternion_world_xyzw": body["default_inertial_quaternion_world_xyzw"],
            "source_to_world_visual_registration": True,
            "body_link_registration": True,
            "organ_fem_or_mpm": False,
            "vessel_tube_or_lumen_mechanics": False,
            "neural_mechanics": False,
            "mechanical_mass_owner": None,
            "material_density_calibrated": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
        })

    identity = {
        "anatomy_map_sha256": _sha256(Path(anatomy_map).read_bytes()),
        "visual_manifest_sha256": _sha256(Path(visual_manifest).read_bytes()),
        "native_receipt_sha256": _sha256(Path(native_receipt).read_bytes()),
        "human_manifest_sha256": _sha256(Path(human_manifest).read_bytes()),
        "rigid_payload_sha256": catalog["sha256"],
        "source_archive_sha256": catalog["archive_sha256"],
        "bindings": linked,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-torso-body-links.1",
        "source": {
            "anatomy_map": str(Path(anatomy_map).relative_to(ROOT)) if Path(anatomy_map).is_relative_to(ROOT) else str(anatomy_map),
            "anatomy_map_sha256": identity["anatomy_map_sha256"],
            "visual_manifest": str(Path(visual_manifest).relative_to(ROOT)) if Path(visual_manifest).is_relative_to(ROOT) else str(visual_manifest),
            "visual_manifest_sha256": identity["visual_manifest_sha256"],
            "native_receipt": str(Path(native_receipt).relative_to(ROOT)) if Path(native_receipt).is_relative_to(ROOT) else str(native_receipt),
            "native_receipt_sha256": identity["native_receipt_sha256"],
            "human_manifest": str(Path(human_manifest).relative_to(ROOT)) if Path(human_manifest).is_relative_to(ROOT) else str(human_manifest),
            "human_manifest_sha256": identity["human_manifest_sha256"],
            "rigid_payload_sha256": catalog["sha256"],
            "source_archive_sha256": catalog["archive_sha256"],
            "body_count": catalog["body_count"],
            "source_body_count": catalog["source_count"],
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "bindings": linked,
        "qualification": {
            "source_membership_and_hashes": True,
            "source_to_world_visual_registration": True,
            "body_link_registration": True,
            "organ_fem_or_mpm": False,
            "vessel_tube_or_lumen_mechanics": False,
            "neural_mechanics": False,
            "material_density_calibrated": False,
            "blood_mass_owner": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
            "anatomical_loading": False,
            "standing_walking": False,
        },
        "boundary": (
            "Twelve exact BodyParts3D torso surfaces (five organs, six vessels, "
            "and one spinal cord) are hash-bound to the native visual payload and "
            "named MyoSim source/core body frames. This closes source/body-frame "
            "bookkeeping only; it does not create organ FEM/MPM, vessel tube or "
            "lumen mechanics, neural mechanics, mass, material density, pressure "
            "reaction, tissue exchange, anatomical loading, subject calibration, "
            "standing, or walking."
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
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--anatomy-map", type=Path, default=MAP)
    parser.add_argument("--visual-manifest", type=Path, default=VISUAL_MANIFEST)
    parser.add_argument("--native-receipt", type=Path, default=NATIVE_RECEIPT)
    parser.add_argument("--human-manifest", type=Path, default=HUMAN_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_body_links(sources=args.sources, source_lock=args.source_lock,
                                    anatomy_map=args.anatomy_map,
                                    visual_manifest=args.visual_manifest,
                                    native_receipt=args.native_receipt,
                                    human_manifest=args.human_manifest)
        digest = immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "bindings": len(result["bindings"]),
                          "body_link_registration": True}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"torso anatomy body-link registration: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

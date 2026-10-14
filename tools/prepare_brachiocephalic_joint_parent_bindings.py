#!/usr/bin/env python3
"""Compose the existing ABI5 vessel and joint passive-parent maps by payload identity.

This is source preparation only. It chains the existing anatomy receipt composer
through the exact common -> back -> joint muscle ancestors, then overlays the
two vascular parent maps into the joint extension. It does not run geometry or
native admission predicates.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from numilab_human import passive_attachment_composition as parent_composer
from numilab_human import resting_brachiocephalic_source007 as source007
from numilab_human.resting_brachiocephalic_joint_parent_composition import (
    extend_joint_source_inventory,
    merge_vascular_parent_maps,
)

OWNER_MODULES = (
    "numilab_human.passive_attachment_composition",
    "numilab_human.resting_brachiocephalic_source007",
    "numilab_human.resting_brachiocephalic_joint_parent_composition",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_snapshot(value: Any) -> Any:
    """Normalize in-memory mappings exactly as their frozen JSON representation."""
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    _require(isinstance(value, dict), f"expected JSON object at {path}")
    return value


def _bound_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else base / path).resolve()


def _record_path(record: dict[str, Any], base: Path) -> Path:
    path = record.get("path")
    _require(isinstance(path, str) and bool(path), "source payload path is missing")
    return _bound_path(path, base)


def _copy_or_link(source: Path, destination: Path, expected_sha: str) -> None:
    _require(source.is_file() and _sha(source) == expected_sha,
             f"source payload identity changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _require(destination.is_file() and _sha(destination) == expected_sha,
                 f"output asset path is occupied by different bytes: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM}:
            raise
        shutil.copyfile(source, destination)
    _require(_sha(destination) == expected_sha,
             f"copied asset differs from its source: {destination}")


def _manifest_direct_parent(
    manifest: dict[str, Any], expected_parent_sha256: str,
) -> str | None:
    """Select the operation bound to the receipt's current muscle owner.

    Older manifests retain unrelated append history alongside a later reference
    composition. Match the receipt owner's hash exactly, as the existing
    bind_anatomy_receipt function does, rather than treating the last operation
    in an unrelated history list as the direct parent.
    """
    source = manifest.get("source", {})
    appends = source.get("passive_surface_appends", [])
    references = source.get("reference_surface_compositions", [])
    if appends and appends[-1].get("source_payload_sha256") == expected_parent_sha256:
        return appends[-1].get("source_payload_sha256")
    if references:
        return references[-1].get("source_payload_sha256")
    direct = source.get("reference_attachment_composition")
    if isinstance(direct, dict):
        return direct.get("source_payload_sha256")
    return None


def _surface_source_indices(extension: dict[str, Any], stable_id: int) -> set[int]:
    rows = {int(row["stable_id"]): row for row in extension.get("surfaces", [])}
    _require(stable_id in rows, f"joint muscle owner map {stable_id} is missing")
    return {
        int(term[0])
        for vertex_terms in rows[stable_id].get("vertices", [])
        for term in vertex_terms
    }


def _verify_joint_owner_roles(extension: dict[str, Any]) -> dict[str, Any]:
    required = {
        132: ((0,), "FDS"),
        134: ((5,), "FDP"),
        153: ((2,), "back-parent-0"),
        155: ((2,), "back-parent-0"),
        154: ((3,), "back-parent-1"),
        156: ((3,), "back-parent-1"),
        157: ((2, 3), "back-parent-blend"),
        158: ((3,), "back-parent-1"),
    }
    inventory = extension["source_payloads"]
    observed = {}
    for stable_id, (expected_indices, role) in required.items():
        indices = _surface_source_indices(extension, stable_id)
        _require(indices == set(expected_indices),
                 f"existing {role} source ownership changed for row {stable_id}")
        observed[str(stable_id)] = {
            "role": role,
            "source_indices": list(expected_indices),
            "source_sha256_by_index": {
                str(index): inventory[index]["sha256"] for index in expected_indices},
            "source_path_by_index": {
                str(index): inventory[index]["path"] for index in expected_indices},
        }
    return observed


def _resolve_referenced_sources(
    common_extension: dict[str, Any], joint_extension: dict[str, Any],
    common_source_paths: dict[str, str], joint_extension_dir: Path,
    output_joint_dir: Path,
) -> tuple[dict[int, str], dict[str, str], dict[str, Any],
           dict[str, str], dict[str, str]]:
    common_rows = {int(row["stable_id"]): row
                   for row in common_extension["anatomy_surfaces"]}
    referenced = sorted({
        int(term[0])
        for stable_id in (475, 477)
        for vertex_terms in common_rows[stable_id]["vertices"]
        for term in vertex_terms
    })
    joint_by_sha = {
        record["sha256"]: index
        for index, record in enumerate(joint_extension["source_payloads"])
    }
    append_paths: dict[int, str] = {}
    source_paths: dict[str, str] = {}
    carrier_input_pins: dict[str, str] = {}
    appended_output_pins: dict[str, str] = {}
    for common_index in referenced:
        record = common_extension["source_payloads"][common_index]
        digest = record["sha256"]
        raw_source = common_source_paths.get(str(common_index))
        _require(isinstance(raw_source, str),
                 f"validated source receipt lacks carrier path {common_index}")
        source = Path(raw_source).resolve()
        _require(source.is_file() and _sha(source) == digest,
                 f"referenced common carrier changed: {source}")
        source_paths[str(common_index)] = str(source)
        carrier_input_pins[str(source)] = digest
        if digest in joint_by_sha:
            joint_record = joint_extension["source_payloads"][joint_by_sha[digest]]
            joint_source = _record_path(joint_record, joint_extension_dir)
            _require(joint_source.is_file() and _sha(joint_source) == digest,
                     f"joint source payload changed: {joint_source}")
            continue
        target = output_joint_dir / "assets" / source.name
        if target.exists() and _sha(target) != digest:
            target = target.with_name(f"{digest[:12]}-{source.name}")
        append_paths[common_index] = f"assets/{target.name}"
        appended_output_pins[str(target)] = digest
    return (
        append_paths,
        source_paths,
        {
            "referenced_common_source_indices": referenced,
            "referenced_common_source_sha256": {
                str(index): common_extension["source_payloads"][index]["sha256"]
                for index in referenced
            },
        },
        carrier_input_pins,
        appended_output_pins,
    )


def _verify_joint_input_pins(joint_preparation: dict[str, Any]) -> dict[str, str]:
    pins = joint_preparation.get("input_sha256", {})
    _require(isinstance(pins, dict) and bool(pins),
             "joint preparation lacks its immutable input hash inventory")
    resolved: dict[str, str] = {}
    for raw_path, expected in pins.items():
        path = Path(raw_path).resolve()
        _require(path.is_file(), f"joint preparation input is missing: {path}")
        actual = _sha(path)
        _require(actual == expected, f"joint preparation input hash changed: {path}")
        resolved[str(path)] = actual
    return resolved


def _input_code_pins() -> dict[str, dict[str, str]]:
    result = {}
    for module_name in OWNER_MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        result[module_name] = {"path": str(path), "sha256": _sha(path)}
    script_path = Path(__file__).resolve()
    result["prepare_brachiocephalic_joint_parent_bindings"] = {
        "path": str(script_path), "sha256": _sha(script_path)}
    git = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    return {"modules": result, "git_head": git}


def _verify_direct_ancestor_chain(
    common_receipt: dict[str, Any],
    back_manifest: dict[str, Any],
    joint_manifest: dict[str, Any],
    back_sha: str,
    joint_child_sha: str,
) -> dict[str, str]:
    common_owner_sha = common_receipt["provenance"]["native_muscle_surfaces"]["sha256"]
    back_parent = _manifest_direct_parent(back_manifest, common_owner_sha)
    joint_parent = _manifest_direct_parent(joint_manifest, back_sha)
    _require(back_parent == common_owner_sha,
             "back muscle child is not a direct child of the common receipt owner")
    _require(joint_parent == back_sha,
             "joint muscle child is not a direct child of the back muscle child")
    return {
        "common_receipt_muscle_owner_sha256": common_owner_sha,
        "back_manifest_direct_parent_sha256": back_parent,
        "back_payload_sha256": back_sha,
        "joint_manifest_direct_parent_sha256": joint_parent,
        "joint_child_sha256": joint_child_sha,
    }


def _validate_common_source_assets(
    receipt_path: Path, report_path: Path,
) -> dict[str, Any]:
    """Recheck the immutable ABI5/common-field asset bindings without P6 replay.

    The retained source-007 work already has a separate header-only preflight.
    The common-only receipt does not consume P6; the historical P6 verifier pin
    remains context, not a live dependency. This check binds the exact NHA,
    source owner report, NHRIGID rest registration, common map prefix, and
    unchanged sidecars before parent-map composition.
    """
    common_field_owner = importlib.import_module(
        "numilab_human.resting_brachiocephalic_common_field")
    source_validation = source007._validate_source_candidate(receipt_path, report_path)
    receipt = _load(receipt_path)
    preparation = _load(report_path)
    payload = receipt.get("payload", {})
    payload_path = Path(payload.get("path", "")).resolve()
    _require(payload_path.is_file() and _sha(payload_path) == payload.get("sha256"),
             "common candidate receipt does not bind its exact NHA payload")
    _require(preparation.get("input_candidate_payload_sha256") == payload.get("sha256")
             and preparation.get("anatomy_payload_modified") is False
             and preparation.get("appended_record_count") == 651
             and preparation.get("input_common_map_record_count") == 620718
             and preparation.get("map_prefix_byte_exact") is True
             and preparation.get("volume_polynomials_and_domain_boxes_byte_preserved") is True
             and preparation.get("existing_cvsim_compartment_id") == 4
             and preparation.get("added_physical_mass_kg") == 0
             and preparation.get("additional_physical_state_or_force") is False,
             "common preparation report does not pin the unchanged 651-record field append")

    n27_receipt_path = Path(receipt["provenance"][
        "passive_brachiocephalic_vein_extension"]["base_receipt"]).resolve()
    extension = receipt["provenance"]["passive_brachiocephalic_vein_extension"]
    _require(n27_receipt_path == source007.BASE_RECEIPT.resolve()
             and _sha(n27_receipt_path) == source007.N27_RECEIPT_SHA
             and extension.get("base_receipt_sha256") == source007.N27_RECEIPT_SHA
             and extension.get("base_payload_sha256") == source007.N27_PAYLOAD_SHA,
             "common candidate no longer names the pinned direct N27 source")
    n27_receipt = _load(n27_receipt_path)
    n27_payload_path = Path(n27_receipt["payload"]["path"]).resolve()
    _require(n27_receipt["payload"]["sha256"] == source007.N27_PAYLOAD_SHA
             and n27_payload_path.is_file()
             and _sha(n27_payload_path) == source007.N27_PAYLOAD_SHA,
             "pinned N27 payload identity changed")
    _require(receipt["provenance"].get("rigid_payload_sha256") == source007.RIGID_SHA
             and receipt["provenance"].get("bodyparts_registration_sha256")
             == source007.REGISTRATION_SHA
             and _sha(Path(extension["verified_rigid_payload_path"]))
             == source007.RIGID_SHA
             and _sha(Path(extension["verified_registration_path"]))
             == source007.REGISTRATION_SHA,
             "common candidate source-rest or registration pins changed")
    registration = _load(Path(extension["verified_registration_path"]))
    body = common_field_owner._validated_torso_rest_body(
        n27_receipt, registration, Path(extension["verified_rigid_payload_path"]))
    recorded_rest = extension.get("verified_rest_body_context", {})
    _require(recorded_rest.get("validated_against_receipt_torso_rest") is True
             and recorded_rest.get("source_body_id") == body["torso"]["source_body_id"] == 8
             and recorded_rest.get("core_body_index") == body["torso"]["core_body_index"] == 20,
             "common candidate torso context differs from the pinned NHRIGID rest")
    for key in ("default_com_position_world_m", "default_inertial_quaternion_world_xyzw"):
        _require(len(recorded_rest.get(key, [])) == len(body["torso"][key])
                 and all(abs(float(actual) - float(expected)) <= 5e-9
                         for actual, expected in zip(
                             recorded_rest[key], body["torso"][key])),
                 "common candidate NHRIGID rest transform differs for " + key)

    source_receipt_path = Path(preparation.get("input_candidate_receipt_path", "")).resolve()
    _require(source_receipt_path.is_file()
             and _sha(source_receipt_path) == preparation.get(
                 "input_candidate_receipt_sha256"),
             "common preparation's source candidate receipt pin changed")
    source_receipt = _load(source_receipt_path)
    source_extension = source_receipt.get("passive_tissue_parent_bindings")
    output_extension = receipt.get("passive_tissue_parent_bindings")
    _require(isinstance(source_extension, dict) and isinstance(output_extension, dict)
             and source_extension.get("source_payloads")
             == output_extension.get("source_payloads")
             and source_extension.get("anatomy_surfaces")
             == output_extension.get("anatomy_surfaces"),
             "common candidate changed its retained source-owner inventory or maps")
    carrier_source_paths = {}
    for index, record in enumerate(source_extension["source_payloads"]):
        candidate_path = _record_path(record, source_receipt_path.parent)
        _require(candidate_path.is_file() and _sha(candidate_path) == record["sha256"],
                 f"source carrier {index} differs from its source-candidate receipt")
        carrier_source_paths[str(index)] = str(candidate_path)
    source_payload_path = Path(source_receipt["payload"]["path"]).resolve()
    _require(source_receipt["payload"]["sha256"] == payload.get("sha256")
             and source_payload_path == payload_path
             and _sha(source_payload_path) == payload.get("sha256"),
             "common preparation NHA differs from its pinned source candidate")
    with payload_path.open("rb") as stream:
        header_bytes = stream.read(struct.calcsize("<8sIIIIII"))
    _require(len(header_bytes) == struct.calcsize("<8sIIIIII"),
             "common candidate NHANAT header is truncated")
    magic, abi, row_count, vertex_count, index_count, registration_fp, _reserved = (
        struct.unpack("<8sIIIIII", header_bytes))
    expected_length = (common_field_owner.NHANAT_HEADER.size
                       + row_count * common_field_owner.NHANAT_RECORD.size
                       + vertex_count * 24 + index_count * 4)
    _require(magic == b"NHANAT1\0" and abi == 5 and row_count == 525
             and vertex_count == 2078718 and index_count == 11082252
             and registration_fp == int(source007.REGISTRATION_SHA[:8], 16)
             and payload_path.stat().st_size == expected_length,
             "common candidate NHANAT ABI5 header or exact payload size differs")
    historical_common_driver = importlib.import_module(
        "numilab_human.resting_brachiocephalic_common_field_007007")
    historical_owner_pins = {
        name: digest
        for name, digest in historical_common_driver.OWNER_SOURCE_SHA256.items()
        if name != "myosim_bone_proximity.py"
    }
    _require(preparation.get("source_code_sha256") == historical_owner_pins,
             "common preparation's historical published source-owner pins changed")
    source_archive_present = any(
        item.get("hierarchy") == "part_of"
        and item.get("sha256") == source007.SOURCE_ARCHIVE_SHA
        for item in receipt["provenance"].get("bodyparts3d_archives", []))
    _require(source_archive_present and _sha(source007.SOURCE_ARCHIVE)
             == source007.SOURCE_ARCHIVE_SHA,
             "common candidate BodyParts3D source archive identity changed")

    source_common = source_receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    output_common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    source_descriptor = json.loads(json.dumps(source_common))
    output_descriptor = json.loads(json.dumps(output_common))
    _require(source_descriptor["coordinate_order"] == output_descriptor["coordinate_order"]
             and source_descriptor["volume_stable_ids"] == output_descriptor["volume_stable_ids"],
             "common-field coordinate or physical volume identity changed")
    output_descriptor["anatomy_payload_sha256"] = source_descriptor[
        "anatomy_payload_sha256"]
    output_ranges = output_descriptor.get("passive_attachment_ranges", [])
    appended_ranges = [row for row in output_ranges if int(row["stable_id"]) in (525, 526)]
    retained_ranges = [row for row in output_ranges if int(row["stable_id"]) not in (525, 526)]
    _require([int(row["stable_id"]) for row in appended_ranges] == [525, 526]
             and retained_ranges == source_descriptor.get("passive_attachment_ranges"),
             "common field changed a prior passive attachment range")
    output_descriptor["passive_attachment_ranges"] = source_descriptor[
        "passive_attachment_ranges"]
    for owner in ("map", "polynomials", "domain_boxes"):
        output_descriptor[owner]["path"] = source_descriptor[owner]["path"]
    source_map_record = source_descriptor["map"]
    output_map_record = output_descriptor["map"]
    _require(output_map_record.get("record_count")
             == int(source_map_record["record_count"]) + 651
             and output_map_record.get("record_stride_bytes")
             == source_map_record.get("record_stride_bytes")
             and output_map_record.get("record_stride_bytes") == 112,
             "common map descriptor is not the exact 651-record ABI append")
    output_map_record["record_count"] = source_map_record["record_count"]
    output_map_record["sha256"] = source_map_record["sha256"]
    _require(output_descriptor == source_descriptor,
             "common descriptor changed outside the declared map append and new ranges")
    source_base_path = Path(source_receipt["provenance"][
        "passive_brachiocephalic_vein_extension"]["base_receipt"]).resolve()
    compared_sidecars = {}
    for key in ("map", "polynomials", "domain_boxes"):
        source_record, output_record = source_common[key], output_common[key]
        source_path = common_field_owner._resolve_sidecar(
            source_receipt_path, source_base_path, source_record)
        output_path = common_field_owner._resolve_sidecar(
            receipt_path, n27_receipt_path, output_record)
        source_sha, output_sha = _sha(source_path), _sha(output_path)
        _require(source_sha == source_record["sha256"]
                 and output_sha == output_record["sha256"],
                 "common-field sidecar hash differs for " + key)
        compared_sidecars[key] = {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "output_path": str(output_path),
            "output_sha256": output_sha,
        }
        if key == "map":
            stride = int(output_record["record_stride_bytes"])
            source_count = int(source_record["record_count"])
            output_count = int(output_record["record_count"])
            source_bytes, output_bytes = source_path.read_bytes(), output_path.read_bytes()
            _require(source_count == 620718 and output_count == 621369
                     and len(source_bytes) == source_count * stride
                     and len(output_bytes) == output_count * stride
                     and output_bytes[:len(source_bytes)] == source_bytes
                     and preparation.get("input_common_map_sha256") == source_sha
                     and preparation.get("output_map_sha256") == output_sha,
                     "common map does not preserve the exact 620718-record prefix")
            compared_sidecars[key]["byte_exact_prefix_records"] = source_count
            compared_sidecars[key]["appended_record_count"] = output_count - source_count
        else:
            _require(source_path.read_bytes() == output_path.read_bytes(),
                     "common-field non-map sidecar changed: " + key)
            report_key = ("input_common_polynomials_sha256" if key == "polynomials"
                          else "input_common_domain_boxes_sha256")
            _require(preparation.get(report_key) == source_sha,
                     "common-field input sidecar report hash differs: " + key)
            compared_sidecars[key]["whole_file_byte_exact"] = True

    source_ids = receipt["provenance"].get("source_id_map", {})
    for stable_id, member in ((525, "FJ3482"), (526, "FJ3583")):
        identity = source_ids.get(str(stable_id), {})
        _require(identity.get("source_member") == member
                 and int(identity.get("body_index", -1)) == 20
                 and int(identity.get("layer", -1)) == 2
                 and int(identity.get("vascular_compartment_id", -1)) == 4,
                 "common candidate source ID/body/layer/compartment changed for " + member)
    _require([int(row["stable_id"]) for row in preparation.get("ranges", [])]
             == [525, 526]
             and [int(row["stable_id"]) for row in output_common.get(
                 "passive_attachment_ranges", []) if int(row["stable_id"]) in (525, 526)]
             == [525, 526],
             "common-field append ranges do not bind exactly rows 525 and 526")
    prep_owner = receipt["provenance"].get(
        "passive_brachiocephalic_common_field_preparation", {})
    _require(prep_owner.get("appended_range_stable_ids") == [525, 526]
             and prep_owner.get("additional_physical_mass_kg") == 0
             and prep_owner.get("additional_physical_state_or_force") is False,
             "common-field receipt owner or passive-state declaration changed")

    root = source_receipt_path.parents[1]
    historical_source_preflight = root / "brachiocephalic-source-007-preflight-002.json"
    historical_common_preflight = root / (
        "brachiocephalic-common-field-preparation-007-007-preflight.json")
    _require(_sha(historical_source_preflight)
             == "e81f75064a736a200fbfdf4c809702c15b54264c4bd2cdf2d775d9217b4f8da7",
             "historical source-007 preflight hash changed")
    source_preflight = _load(historical_source_preflight)
    _require(source_preflight.get("status") == "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED"
             and source_preflight.get("candidate007_path")
             and source_preflight.get("nhanat_vertex_or_face_arrays_loaded") is False,
             "historical source-007 preflight does not match its declared header-only scope")
    _require(_sha(historical_common_preflight)
             == "2b5629f2505b51baa91ee62d9dddef899e2a373eb1114cc41f877840ca93f95d",
             "historical common-007 preflight hash changed")
    common_preflight = _load(historical_common_preflight)
    p6 = common_preflight.get("p6_provenance", {})
    _require(common_preflight.get("status") == "PREPARED_NOT_EXECUTED"
             and p6.get("status") == "HISTORICAL_UNCONSUMED_CONTEXT_ONLY"
             and p6.get("consumed_by_common_field_preparation") is False
             and p6.get("p6_predicates_invoked") is False,
             "historical P6 context is no longer explicitly unconsumed")

    return {
        "receipt": receipt,
        "preparation": preparation,
        "payload_path": payload_path,
        "source_map": source_ids,
        "common_field": output_common,
        "source_asset_validation": {
            "status": "PASS_EXISTING_SOURCE_VALIDATOR_AND_COMMON_PREFIX",
            "existing_source_validator": "resting_brachiocephalic_source007._validate_source_candidate",
            "existing_source_validator_checked_rows": len(source_validation["source_map"]),
            "nhanat_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "nhanat_rows": row_count,
            "nhanat_vertices": vertex_count,
            "nhanat_indices": index_count,
            "nhrigid_rest_body_index": body["torso"]["core_body_index"],
            "nhrigid_source_body_id": body["torso"]["source_body_id"],
            "common_sidecars": compared_sidecars,
            "historical_source_preflight_path": str(historical_source_preflight),
            "historical_source_preflight_sha256": _sha(historical_source_preflight),
            "historical_common_preflight_path": str(historical_common_preflight),
            "historical_common_preflight_sha256": _sha(historical_common_preflight),
            "historical_p6_status": p6.get("status"),
            "historical_p6_consumed_by_common_prep": False,
            "current_p6_or_geometry_predicates_run": False,
            "retained_carrier_inventory_exact": True,
            "carrier_source_paths": carrier_source_paths,
            "carrier_source_sha256_by_index": {
                str(index): record["sha256"]
                for index, record in enumerate(source_extension["source_payloads"])},
        },
        "carrier_source_paths": carrier_source_paths,
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    common_receipt_path = args.common_receipt.resolve()
    common_report_path = args.common_report.resolve()
    back_child_path = args.back_child.resolve()
    back_manifest_path = args.back_manifest.resolve()
    back_report_path = args.back_report.resolve()
    joint_prep_path = args.joint_preparation.resolve()
    joint_report_path = args.joint_report.resolve()
    joint_bindings_path = args.joint_bindings.resolve()
    joint_child_path = args.joint_child.resolve()
    joint_manifest_path = args.joint_manifest.resolve()

    paths = (
        common_receipt_path, common_report_path, back_child_path,
        back_manifest_path, back_report_path, joint_prep_path,
        joint_report_path, joint_bindings_path, joint_child_path,
        joint_manifest_path,
    )
    _require(all(path.is_file() for path in paths), "one or more merge inputs are missing")
    common_receipt = _load(common_receipt_path)
    back_manifest = _load(back_manifest_path)
    back_report = _load(back_report_path)
    joint_preparation = _load(joint_prep_path)
    joint_extension = _load(joint_bindings_path)
    joint_manifest = _load(joint_manifest_path)

    common_validation = _validate_common_source_assets(
        common_receipt_path, common_report_path)
    common_extension = common_receipt.get("passive_tissue_parent_bindings")
    _require(isinstance(common_extension, dict),
             "common-field candidate lacks passive vessel parent bindings")
    common_payload_sha = common_validation["receipt"]["payload"]["sha256"]
    n27_parent_sha = common_receipt["provenance"][
        "passive_brachiocephalic_vein_extension"]["base_payload_sha256"]
    _require(joint_extension.get("anatomy_payload_sha256") in
             {common_payload_sha, n27_parent_sha},
             "joint parent map is not based on the validated common NHA ancestry")
    joint_child_sha = _sha(joint_child_path)
    back_sha = _sha(back_child_path)
    _require(joint_preparation.get("child_sha256") == joint_child_sha
             and joint_preparation.get("child_path")
             and Path(joint_preparation["child_path"]).resolve() == joint_child_path,
             "joint preparation child identity differs")
    _require(joint_preparation.get("extension_sha256") == _sha(joint_bindings_path)
             and Path(joint_preparation.get("extension_path", "")).resolve()
             == joint_bindings_path,
             "joint preparation binding extension identity differs")
    _require(joint_preparation.get("composition_report_sha256") == _sha(joint_report_path),
             "joint preparation composition report identity differs")
    _require(back_report.get("payload_sha256") == back_sha
             and back_report.get("manifest_sha256") == _sha(back_manifest_path),
             "back muscle report differs from its child payload or manifest")
    _require(back_manifest.get("payload", {}).get("sha256") == back_sha
             and joint_manifest.get("payload", {}).get("sha256") == joint_child_sha,
             "muscle manifest payload identity differs")
    _require(back_manifest.get("payload", {}).get("registration_fingerprint32")
             == common_receipt["payload"]["registration_fingerprint32"]
             == joint_manifest.get("payload", {}).get("registration_fingerprint32"),
             "muscle and anatomy registration identities differ")
    _require(joint_extension.get("child_payload_sha256") == joint_child_sha,
             "joint parent binding does not identify the child payload")
    ancestor_chain = _verify_direct_ancestor_chain(
        common_receipt, back_manifest, joint_manifest, back_sha, joint_child_sha)
    joint_input_pins = _verify_joint_input_pins(joint_preparation)
    joint_roles = _verify_joint_owner_roles(joint_extension)

    append_paths, resolved_source_paths, _referenced, carrier_input_pins, appended_output_pins = (
        _resolve_referenced_sources(
            common_extension, joint_extension,
            common_validation["carrier_source_paths"],
            joint_bindings_path.parent, args.output.resolve()))
    extended_joint, source_map, inventory_audit = extend_joint_source_inventory(
        common_extension, joint_extension,
        append_paths_by_common_index=append_paths)
    merged_extension, map_audit = merge_vascular_parent_maps(
        common_extension, extended_joint,
        final_anatomy_sha256=common_payload_sha,
        final_child_sha256=joint_child_sha,
        common_source_to_joint=source_map)
    _require(merged_extension["surfaces"] == joint_extension["surfaces"],
             "joint muscle owner maps changed during source-only preflight")
    input_paths = {
        str(path): _sha(path)
        for path in paths
    }
    input_paths.update(joint_input_pins)
    input_paths.update(carrier_input_pins)
    input_paths.update({
        str(common_validation["payload_path"]): _sha(common_validation["payload_path"]),
        str(back_child_path): back_sha,
        str(joint_child_path): joint_child_sha,
    })
    return {
        "status": "source_merge_preflight_only_not_admission",
        "common_validation": {
            "nha_payload_sha256": common_payload_sha,
            "source_asset_validation": common_validation["source_asset_validation"],
            "surface_count": common_validation["preparation"].get(
                "original_523_row_byte_preservation", {}).get("unchanged_original_row_count"),
            "common_map_record_count": common_validation["common_field"][
                "map"]["record_count"],
            "anatomical_admission": False,
            "all_pose_admission": False,
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
        },
        "ancestor_chain": ancestor_chain,
        "joint_muscle_owner_indices": joint_roles,
        "common_to_joint_source_inventory": inventory_audit,
        "vascular_parent_map_composition": map_audit,
        "validated_common_carrier_paths_by_index": resolved_source_paths,
        "input_sha256": dict(sorted(input_paths.items())),
        "output_assets_sha256_if_appended": appended_output_pins,
        "owner_code": _input_code_pins(),
        "joint_native_run_performed": joint_preparation.get("native_run_performed"),
        "no_new_physical_mass_or_state": True,
        "geometry_or_native_admission": "not evaluated",
    }


def write_preflight(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.preflight_output.resolve()
    _require(not destination.exists(), "preserve any prior preflight; choose a new path")
    report = build_preflight(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    destination.write_bytes(raw)
    return {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest(),
            "status": report["status"], "report": report}


def _materialize_sources(
    common_extension: dict[str, Any], joint_extension: dict[str, Any],
    common_source_paths: dict[str, str], output_joint_dir: Path,
) -> tuple[dict[int, str], dict[int, str]]:
    common_rows = {int(row["stable_id"]): row
                   for row in common_extension["anatomy_surfaces"]}
    referenced = sorted({
        int(term[0])
        for stable_id in (475, 477)
        for vertex_terms in common_rows[stable_id]["vertices"]
        for term in vertex_terms
    })
    joint_by_sha = {record["sha256"]: index
                    for index, record in enumerate(joint_extension["source_payloads"])}
    append_paths: dict[int, str] = {}
    appended_input_paths: dict[int, str] = {}
    for common_index in referenced:
        record = common_extension["source_payloads"][common_index]
        digest = record["sha256"]
        if digest in joint_by_sha:
            continue
        raw_source = common_source_paths.get(str(common_index))
        _require(isinstance(raw_source, str),
                 f"validated source receipt lacks carrier path {common_index}")
        source = Path(raw_source).resolve()
        _require(source.is_file() and _sha(source) == digest,
                 f"validated common carrier changed before materialization: {source}")
        target = output_joint_dir / "assets" / source.name
        if target.exists() and _sha(target) != digest:
            target = target.with_name(f"{digest[:12]}-{source.name}")
        _copy_or_link(source, target, digest)
        append_paths[common_index] = f"assets/{target.name}"
        appended_input_paths[common_index] = str(source)
    return append_paths, appended_input_paths


def _assert_file_binding(
    path_value: Any, base: Path, expected_path: Path, expected_sha256: str,
    label: str,
) -> Path:
    _require(isinstance(path_value, str) and bool(path_value),
             f"{label} path is missing")
    resolved = _bound_path(path_value, base)
    expected = Path(expected_path).resolve()
    _require(resolved == expected and resolved.is_file()
             and _sha(resolved) == expected_sha256
             and _sha(expected) == expected_sha256,
             f"{label} path alias or SHA-256 does not resolve to its pinned input")
    return resolved


def _assert_final_source_provenance_preserved(
    source_receipt: dict[str, Any], final_receipt: dict[str, Any],
    source_receipt_path: Path, final_receipt_path: Path,
) -> dict[str, Any]:
    """Preserve every source field except the declared owner/map composition."""
    source_payload = dict(source_receipt.get("payload", {}))
    final_payload = dict(final_receipt.get("payload", {}))
    source_path = _bound_path(source_payload.get("path", ""), source_receipt_path.parent)
    final_path = _bound_path(final_payload.get("path", ""), final_receipt_path.parent)
    _require(source_path == final_path and source_path.is_file()
             and _sha(source_path) == source_payload.get("sha256")
             and _sha(final_path) == final_payload.get("sha256")
             and source_payload.get("sha256") == final_payload.get("sha256"),
             "parent binding changed the exact NHA payload file")
    source_payload["path"] = str(source_path)
    final_payload["path"] = str(final_path)
    _require(final_payload == source_payload,
             "parent binding changed NHA payload metadata or identity")

    allowed_root_changes = {"payload", "provenance", "passive_tissue_parent_bindings"}
    for key in set(source_receipt) | set(final_receipt):
        if key not in allowed_root_changes:
            _require(source_receipt.get(key) == final_receipt.get(key),
                     "parent binding changed unrelated receipt field: " + str(key))

    source_provenance = source_receipt.get("provenance", {})
    final_provenance = final_receipt.get("provenance", {})
    allowed_provenance_changes = {
        "native_muscle_surfaces",
        "reference_surface_composition_binding",
        "reference_surface_composition_binding_history",
        "brachiocephalic_joint_parent_binding_composition",
    }
    for key in set(source_provenance) | set(final_provenance):
        if key not in allowed_provenance_changes:
            _require(source_provenance.get(key) == final_provenance.get(key),
                     "final receipt changed source-bound provenance: " + str(key))
    return {
        "nha_payload_record_preserved": True,
        "source_identity_map_preserved": True,
        "nhrigid_registration_and_archives_preserved": True,
        "all_other_top_level_and_provenance_fields_preserved": True,
        "vascular_source_and_common_field_provenance_preserved": True,
        "local_intercostal_source_provenance_preserved": True,
        "native_muscle_owner_changed_only_by_existing_receipt_composer": True,
    }


def _validate_final_parent_extension(
    common_extension: dict[str, Any],
    original_joint_extension: dict[str, Any],
    final_extension: dict[str, Any],
    *,
    common_receipt_path: Path,
    final_receipt_path: Path,
    output_dir: Path,
    final_anatomy_sha256: str,
    final_child_sha256: str,
    declared_inventory_audit: dict[str, Any],
    declared_map_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild the final source inventory and maps from the two retained owners."""
    common_sources = common_extension.get("source_payloads")
    joint_sources = original_joint_extension.get("source_payloads")
    final_sources = final_extension.get("source_payloads")
    _require(isinstance(common_sources, list) and isinstance(joint_sources, list)
             and isinstance(final_sources, list),
             "final parent extension lacks source payload inventories")
    common_rows = {
        int(row["stable_id"]): row for row in common_extension.get("anatomy_surfaces", [])
    }
    _require(set(common_rows) == {475, 477},
             "common extension must contain exactly vascular rows 475 and 477")
    _require([len(common_rows[sid].get("vertices", [])) for sid in (475, 477)]
             == [902, 1089],
             "common vascular parent-map vertex counts differ from the pinned rows")
    referenced = sorted({
        int(term[0])
        for stable_id in (475, 477)
        for terms in common_rows[stable_id]["vertices"]
        for term in terms
    })
    common_hash_index = {
        record["sha256"]: index for index, record in enumerate(common_sources)
    }
    joint_hash_index = {
        record["sha256"]: index for index, record in enumerate(joint_sources)
    }
    _require(len(common_hash_index) == len(common_sources)
             and len(joint_hash_index) == len(joint_sources),
             "common or joint source inventory repeats a payload identity")
    expected_missing = [
        index for index in referenced
        if common_sources[index]["sha256"] not in joint_hash_index
    ]
    _require(len(final_sources) == len(joint_sources) + len(expected_missing)
             and final_sources[:len(joint_sources)] == joint_sources,
             "final source inventory changed the joint prefix or appended extra sources")

    append_paths: dict[int, str] = {}
    appended_records = final_sources[len(joint_sources):]
    for common_index, record in zip(expected_missing, appended_records, strict=True):
        _require(record.get("sha256") == common_sources[common_index]["sha256"],
                 "appended source inventory order differs from common source identity")
        append_paths[common_index] = record.get("path")

    expected_joint, source_map, inventory_audit = extend_joint_source_inventory(
        common_extension, original_joint_extension,
        append_paths_by_common_index=append_paths)
    normalized_inventory_audit = _json_snapshot(inventory_audit)
    _require(expected_joint["source_payloads"] == final_sources
             and normalized_inventory_audit == declared_inventory_audit,
             "final source inventory or its audit differs from identity-based extension")
    _require(inventory_audit["original_joint_source_count"] == len(joint_sources)
             and inventory_audit["final_joint_source_count"] == len(final_sources)
             and inventory_audit["referenced_common_source_indices"] == referenced,
             "final source inventory counts or referenced indices are inconsistent")

    joint_assets_dir = (Path(output_dir).resolve() / "joint" / "assets").resolve()
    source_paths = []
    for index, record in enumerate(final_sources):
        source_path = _record_path(record, final_receipt_path.parent)
        _require(source_path.is_file() and _sha(source_path) == record["sha256"],
                 f"final parent source {index} path or SHA-256 is invalid")
        if index >= len(joint_sources):
            _require(not Path(record["path"]).is_absolute()
                     and source_path.is_relative_to(joint_assets_dir),
                     "new parent source is not materialized inside the final joint assets")
        source_paths.append(str(source_path))
    for common_index in referenced:
        source_path = _record_path(common_sources[common_index], common_receipt_path.parent)
        _require(source_path.is_file()
                 and _sha(source_path) == common_sources[common_index]["sha256"],
                 f"common referenced source {common_index} path or SHA-256 is invalid")

    extended_joint = deepcopy(original_joint_extension)
    extended_joint["source_payloads"] = final_sources
    expected_extension, map_audit = merge_vascular_parent_maps(
        common_extension, extended_joint,
        final_anatomy_sha256=final_anatomy_sha256,
        final_child_sha256=final_child_sha256,
        common_source_to_joint=source_map,
    )
    _require(expected_extension == final_extension,
             "final vascular parent maps differ from the source-identity remap")
    normalized_map_audit = _json_snapshot(map_audit)
    _require(normalized_map_audit == declared_map_audit,
             "final parent-map audit differs from reconstructed map composition")
    _require(final_extension.get("anatomy_payload_sha256") == final_anatomy_sha256
             and final_extension.get("child_payload_sha256") == final_child_sha256,
             "final parent extension is not bound to its NHA and joint child")
    final_rows = {
        int(row["stable_id"]): row for row in final_extension.get("anatomy_surfaces", [])
    }
    _require(set(final_rows) == {475, 477}
             and [len(final_rows[sid].get("vertices", [])) for sid in (475, 477)]
             == [902, 1089],
             "final vascular map row IDs or vertex counts changed")
    _require(final_extension.get("surfaces") == original_joint_extension.get("surfaces"),
             "composition changed an existing joint muscle parent map")
    _require(source_map == inventory_audit["common_source_to_joint_source_index"]
             and map_audit["remapped_parent_term_count"]
             == sum(len(terms) for sid in (475, 477)
                    for terms in common_rows[sid]["vertices"]),
             "parent-term counts or source-index remap are inconsistent")
    _require(map_audit.get("physical_state_mass_and_force_fields_added") is False
             and map_audit.get("geometry_or_native_predicates_run") is False,
             "parent-map composition report overstates source-only work")
    return normalized_inventory_audit, normalized_map_audit, {
        "status": "PASS_SOURCE_IDENTITY_AND_PARENT_MAP_STRUCTURE_ONLY",
        "joint_source_prefix_count": len(joint_sources),
        "appended_source_count": len(appended_records),
        "final_source_count": len(final_sources),
        "source_paths_sha_verified": True,
        "common_to_joint_source_index": {
            str(index): target for index, target in source_map.items()},
        "parent_map_rows": {
            str(sid): len(final_rows[sid]["vertices"]) for sid in (475, 477)},
        "remapped_parent_term_count": map_audit["remapped_parent_term_count"],
        "geometry_or_anatomical_admission": "not evaluated",
    }


def _validate_final_composition_output(
    args: argparse.Namespace,
    frozen: dict[str, Any],
    *,
    back_receipt_path: Path,
    final_receipt_path: Path,
    report: dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all serialized parent-composition outputs without geometry predicates."""
    common_path = args.common_receipt.resolve()
    common = _load(common_path)
    back_path = Path(back_receipt_path).resolve()
    back = _load(back_path)
    final_path = Path(final_receipt_path).resolve()
    final = _load(final_path)
    output_dir = args.output.resolve()
    _require(final_path.parent == (output_dir / "joint").resolve(),
             "final receipt is outside the declared composition output")

    source_preservation = _assert_final_source_provenance_preserved(
        common, back, common_path, back_path)
    _require(source_preservation["all_other_top_level_and_provenance_fields_preserved"],
             "back binding changed unrelated receipt fields")
    _require(back.get("passive_tissue_parent_bindings")
             == common.get("passive_tissue_parent_bindings"),
             "back receipt changed the common passive parent-map extension")
    final_source_preservation = _assert_final_source_provenance_preserved(
        common, final, common_path, final_path)

    back_manifest_path = args.back_manifest.resolve()
    joint_manifest_path = args.joint_manifest.resolve()
    back_manifest = _load(back_manifest_path)
    joint_manifest = _load(joint_manifest_path)
    back_child_path = args.back_child.resolve()
    joint_child_path = args.joint_child.resolve()
    chain = _verify_direct_ancestor_chain(
        common, back_manifest, joint_manifest,
        _sha(back_child_path), _sha(joint_child_path))
    _require(chain == frozen.get("ancestor_chain"),
             "serialized muscle ancestry differs from frozen direct-parent chain")

    common_native = common["provenance"]["native_muscle_surfaces"]
    back_native = back["provenance"]["native_muscle_surfaces"]
    final_native = final["provenance"]["native_muscle_surfaces"]
    _require(common_native.get("sha256") == chain["common_receipt_muscle_owner_sha256"]
             and back_native.get("sha256") == chain["back_payload_sha256"]
             and final_native.get("sha256") == chain["joint_child_sha256"]
             and common_native.get("manifest_sha256") == _sha(
                 Path(common_native.get("manifest_path", "")))
             and back_native.get("manifest_sha256") == _sha(back_manifest_path)
             and final_native.get("manifest_sha256") == _sha(joint_manifest_path),
             "receipt muscle owners do not follow the common -> back -> joint ancestry")
    _assert_file_binding(common_native.get("payload_path"), common_path.parent,
                         Path(common_native["payload_path"]), common_native["sha256"],
                         "common muscle owner")
    _assert_file_binding(back_native.get("payload_path"), back_path.parent,
                         back_child_path, chain["back_payload_sha256"], "back muscle owner")
    _assert_file_binding(back_native.get("manifest_path"), back_path.parent,
                         back_manifest_path, _sha(back_manifest_path),
                         "back muscle manifest")
    _assert_file_binding(final_native.get("payload_path"), final_path.parent,
                         joint_child_path, chain["joint_child_sha256"], "joint muscle owner")
    _assert_file_binding(final_native.get("manifest_path"), final_path.parent,
                         joint_manifest_path, _sha(joint_manifest_path),
                         "joint muscle manifest")
    _require(final_native.get("manifest_sha256") == _sha(joint_manifest_path),
             "final muscle owner manifest hash changed")

    back_binding = back["provenance"].get("reference_surface_composition_binding", {})
    final_binding = final["provenance"].get("reference_surface_composition_binding", {})
    _assert_file_binding(back_binding.get("prior_receipt_path"), back_path.parent,
                         common_path, _sha(common_path), "back receipt parent")
    _assert_file_binding(final_binding.get("prior_receipt_path"), final_path.parent,
                         back_path, _sha(back_path), "joint receipt parent")
    _require(back_binding.get("composition_manifest_sha256") == _sha(back_manifest_path)
             and final_binding.get("composition_manifest_sha256") == _sha(joint_manifest_path),
             "receipt reference-composition manifest ancestry changed")
    common_history = common["provenance"].get("reference_surface_composition_binding_history", [])
    back_history = back["provenance"].get("reference_surface_composition_binding_history", [])
    final_history = final["provenance"].get("reference_surface_composition_binding_history", [])
    _require(isinstance(common_history, list) and isinstance(back_history, list)
             and isinstance(final_history, list)
             and back_history[:len(common_history)] == common_history
             and len(back_history) == len(common_history) + 1
             and back_history[-1] == back_binding
             and final_history[:len(back_history)] == back_history
             and len(final_history) == len(back_history) + 1
             and final_history[-1] == final_binding,
             "receipt reference-composition history is not a direct append chain")

    composition = final["provenance"].get(
        "brachiocephalic_joint_parent_binding_composition")
    _require(isinstance(composition, dict)
             and composition.get("scope")
             == "Existing ABI5 parent map composition; source preparation only."
             and composition.get("native_or_gpu_run") is False
             and composition.get("physical_mass_state_or_force_added") is False
             and composition.get("geometry_or_anatomical_admission") == "not evaluated",
             "final receipt lacks required source-only composition scope")
    expected_files = {
        "common_receipt_path": (args.common_receipt, "common_receipt_sha256"),
        "common_preparation_report_path": (
            args.common_report, "common_preparation_report_sha256"),
        "back_child_path": (args.back_child, "back_child_sha256"),
        "back_manifest_path": (args.back_manifest, "back_manifest_sha256"),
        "joint_preparation_path": (args.joint_preparation, "joint_preparation_sha256"),
        "joint_manifest_path": (args.joint_manifest, "joint_manifest_sha256"),
        "joint_child_path": (args.joint_child, "joint_child_sha256"),
    }
    for path_key, (expected_path, sha_key) in expected_files.items():
        expected_path = expected_path.resolve()
        expected_sha = _sha(expected_path)
        _require(composition.get(sha_key) == expected_sha,
                 "final receipt composition hash changed for " + path_key)
        _assert_file_binding(composition.get(path_key), final_path.parent,
                             expected_path, expected_sha, path_key)
    joint_bindings_path = args.joint_bindings.resolve()
    _require(composition.get("joint_parent_binding_sha256") == _sha(joint_bindings_path),
             "final receipt source parent-binding file hash changed")

    common_extension = common["passive_tissue_parent_bindings"]
    joint_extension = _load(joint_bindings_path)
    final_extension = final["passive_tissue_parent_bindings"]
    inventory_audit, map_audit, extension_proof = _validate_final_parent_extension(
        common_extension, joint_extension, final_extension,
        common_receipt_path=common_path,
        final_receipt_path=final_path,
        output_dir=output_dir,
        final_anatomy_sha256=common["payload"]["sha256"],
        final_child_sha256=chain["joint_child_sha256"],
        declared_inventory_audit=composition.get("source_inventory", {}),
        declared_map_audit=composition.get("vascular_parent_maps", {}),
    )

    provenance_proof = {
        "nha_payload_record_preserved": True,
        "source_identity_map_preserved": True,
        "nhrigid_registration_and_archives_preserved": True,
        "vascular_source_and_common_field_provenance_preserved": True,
        "local_intercostal_source_provenance_preserved": True,
        "native_muscle_owner_changed_only_by_existing_receipt_composer": True,
        "all_other_top_level_and_provenance_fields_preserved": True,
    }
    _require(final_source_preservation == provenance_proof,
             "final source-provenance preservation proof changed")

    proof = {
        "status": "PASS_SERIALIZED_SOURCE_COMPOSITION_ONLY_NOT_ADMISSION",
        "final_receipt_sha256": _sha(final_path),
        "nha_payload_sha256": final["payload"]["sha256"],
        "nha_payload_alias_resolved": True,
        "ancestor_chain": chain,
        "source_inventory": inventory_audit,
        "vascular_parent_maps": map_audit,
        "extension_structure": extension_proof,
        "source_provenance_preservation": final_source_preservation,
        "physical_mass_state_or_force_added": False,
        "geometry_or_anatomical_admission": "not evaluated",
        "all_pose_or_native_admission": "not evaluated",
    }

    if report is not None:
        _require(report_path is not None and Path(report_path).is_file(),
                 "composition report file is missing")
        report_path = Path(report_path).resolve()
        _require(report_path == output_dir / "source-parent-composition-report.json"
                 and _load(report_path) == _json_snapshot(report),
                 "serialized composition report differs from the checked report object")
        report_bindings = {
            "status": report.get("status") == "source_parent_binding_composition_only_not_admission",
            "ancestor_chain": report.get("ancestor_chain") == chain,
            "source_inventory": _json_snapshot(
                report.get("common_to_joint_source_inventory")) == inventory_audit,
            "parent_maps": _json_snapshot(report.get("vascular_parent_maps")) == map_audit,
            "source_provenance": report.get("final_source_provenance_preservation")
            == final_source_preservation,
            "final_receipt_sha256": report.get("final_receipt_sha256") == _sha(final_path),
        }
        _require(all(report_bindings.values()),
                 "composition report does not describe the exact serialized receipt: "
                 + ", ".join(key for key, ok in report_bindings.items() if not ok))
        common_validation = report.get("common_source_candidate_validation", {})
        common_field = common["provenance"]["cardiac_geometry_binding"]["common_field"]
        _require(common_validation.get("nha_payload_sha256") == final["payload"]["sha256"]
                 and common_validation.get("common_map_record_count")
                 == common_field["map"]["record_count"]
                 and common_validation.get("geometry_predicates_run") is False
                 and common_validation.get("native_or_gpu_run") is False
                 and report.get("joint_muscle_owner_indices")
                 == frozen.get("joint_muscle_owner_indices")
                 and report.get("common_source_candidate_validation", {}).get(
                     "source_asset_validation")
                 == frozen.get("common_validation", {}).get("source_asset_validation")
                 and report.get("final_muscle_payload_sha256") == _sha(joint_child_path)
                 and report.get("physical_mass_state_or_force_added") is False
                 and report.get("geometry_or_anatomical_admission") == "not evaluated"
                 and report.get("all_pose_or_native_admission") == "not evaluated",
                 "composition report contains incorrect source count or admission status")
        _assert_file_binding(report.get("final_receipt_path"), report_path.parent,
                             final_path, _sha(final_path), "report final receipt")
        _assert_file_binding(report.get("final_muscle_payload_path"), report_path.parent,
                             joint_child_path, _sha(joint_child_path),
                             "report joint muscle payload")
        if "final_output_structure_validation" in report:
            _require(report["final_output_structure_validation"] == proof,
                     "composition report output-structure proof changed")
    return proof


def _assert_composition_report(
    args: argparse.Namespace, frozen: dict[str, Any], *,
    final_receipt_path: Path, report_path: Path, report: dict[str, Any],
    output_proof: dict[str, Any],
) -> None:
    _require(report.get("final_output_structure_validation") == output_proof,
             "composition report omits the complete serialized-output validation")
    _validate_final_composition_output(
        args, frozen,
        back_receipt_path=args.output.resolve() / "back" / "resting-anatomy-receipt.json",
        final_receipt_path=final_receipt_path,
        report=report,
        report_path=report_path,
    )

def compose(args: argparse.Namespace) -> dict[str, Any]:
    frozen_path = args.preflight.resolve()
    _require(frozen_path.is_file(), "reviewed merge preflight is missing")
    _require(_sha(frozen_path) == args.preflight_sha256,
             "merge preflight does not match the reviewed SHA")
    frozen = _load(frozen_path)
    current = _json_snapshot(build_preflight(args))
    _require(current == frozen,
             "merge inputs, source owners, or source bytes differ from frozen preflight")
    output = args.output.resolve()
    _require(not output.exists(), "preserve prior composition output; choose a new path")
    common_receipt_path = args.common_receipt.resolve()
    common_receipt = _load(common_receipt_path)
    common_extension = common_receipt["passive_tissue_parent_bindings"]
    joint_extension = _load(args.joint_bindings.resolve())
    output.mkdir(parents=True)
    back_dir = output / "back"
    joint_dir = output / "joint"
    back_dir.mkdir()
    joint_dir.mkdir()
    # Keep the original child payload paths: existing manifest operations bind
    # their exact immediate parent by path as well as payload and manifest hash.
    back_payload = args.back_child.resolve()
    joint_payload = args.joint_child.resolve()
    _require(_sha(back_payload) == frozen["ancestor_chain"]["back_payload_sha256"]
             and _sha(joint_payload) == frozen["ancestor_chain"]["joint_child_sha256"],
             "frozen muscle child payload changed before receipt binding")
    back_receipt_path = back_dir / "resting-anatomy-receipt.json"
    parent_composer.bind_anatomy_receipt(
        common_receipt_path, back_payload, back_receipt_path)
    joint_receipt_path = joint_dir / "resting-anatomy-receipt.json"
    parent_composer.bind_anatomy_receipt(
        back_receipt_path, joint_payload, joint_receipt_path)

    append_paths, appended_inputs = _materialize_sources(
        common_extension, joint_extension,
        {str(index): path for index, path in frozen[
            "validated_common_carrier_paths_by_index"].items()},
        joint_dir)
    extended_joint, source_map, inventory_audit = extend_joint_source_inventory(
        common_extension, joint_extension,
        append_paths_by_common_index=append_paths)
    merged_extension, map_audit = merge_vascular_parent_maps(
        common_extension, extended_joint,
        final_anatomy_sha256=common_receipt["payload"]["sha256"],
        final_child_sha256=_sha(joint_payload),
        common_source_to_joint=source_map)
    final_receipt = _load(joint_receipt_path)
    final_receipt["passive_tissue_parent_bindings"] = merged_extension
    final_receipt["provenance"]["brachiocephalic_joint_parent_binding_composition"] = {
        "scope": "Existing ABI5 parent map composition; source preparation only.",
        "common_receipt_path": str(common_receipt_path),
        "common_receipt_sha256": _sha(common_receipt_path),
        "common_preparation_report_path": str(args.common_report.resolve()),
        "common_preparation_report_sha256": _sha(args.common_report.resolve()),
        "back_child_path": str(back_payload),
        "back_child_sha256": _sha(back_payload),
        "back_manifest_path": str(args.back_manifest.resolve()),
        "back_manifest_sha256": _sha(args.back_manifest.resolve()),
        "joint_preparation_path": str(args.joint_preparation.resolve()),
        "joint_preparation_sha256": _sha(args.joint_preparation.resolve()),
        "joint_parent_binding_sha256": _sha(args.joint_bindings.resolve()),
        "joint_child_path": str(joint_payload),
        "joint_child_sha256": _sha(joint_payload),
        "joint_manifest_path": str(args.joint_manifest.resolve()),
        "joint_manifest_sha256": _sha(args.joint_manifest.resolve()),
        "source_inventory": inventory_audit,
        "vascular_parent_maps": map_audit,
        "native_or_gpu_run": False,
        "geometry_or_anatomical_admission": "not evaluated",
        "physical_mass_state_or_force_added": False,
    }
    with joint_receipt_path.open("w") as stream:
        json.dump(final_receipt, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    source_receipt = _load(common_receipt_path)
    final_receipt = _load(joint_receipt_path)
    final_source_provenance = _assert_final_source_provenance_preserved(
        source_receipt, final_receipt, common_receipt_path, joint_receipt_path)
    validation = {
        "receipt": final_receipt,
        "common_field": common_receipt["provenance"]["cardiac_geometry_binding"][
            "common_field"],
        "source_asset_validation": frozen["common_validation"]["source_asset_validation"],
    }
    report = {
        "status": "source_parent_binding_composition_only_not_admission",
        "common_source_candidate_validation": {
            "nha_payload_sha256": validation["receipt"]["payload"]["sha256"],
            "source_asset_validation": validation["source_asset_validation"],
            "common_map_record_count": validation["common_field"]["map"]["record_count"],
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
        },
        "ancestor_chain": frozen["ancestor_chain"],
        "joint_muscle_owner_indices": frozen["joint_muscle_owner_indices"],
        "common_to_joint_source_inventory": inventory_audit,
        "vascular_parent_maps": map_audit,
        "final_source_provenance_preservation": final_source_provenance,
        "appended_source_origin_paths": appended_inputs,
        "final_receipt_path": str(joint_receipt_path),
        "final_receipt_sha256": _sha(joint_receipt_path),
        "final_muscle_payload_path": str(joint_payload),
        "final_muscle_payload_sha256": _sha(joint_payload),
        "physical_mass_state_or_force_added": False,
        "geometry_or_anatomical_admission": "not evaluated",
        "all_pose_or_native_admission": "not evaluated",
    }
    output_validation = _validate_final_composition_output(
        args, frozen,
        back_receipt_path=back_receipt_path,
        final_receipt_path=joint_receipt_path,
    )
    report["final_output_structure_validation"] = output_validation
    report_path = output / "source-parent-composition-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False) + "\n")
    _assert_composition_report(
        args, frozen,
        final_receipt_path=joint_receipt_path,
        report_path=report_path,
        report=report,
        output_proof=output_validation,
    )
    return {"path": str(report_path), "sha256": _sha(report_path), "report": report}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--compose", action="store_true")
    for name in (
        "common-receipt", "common-report", "back-child", "back-manifest",
        "back-report", "joint-preparation", "joint-report", "joint-bindings",
        "joint-child", "joint-manifest", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if args.preflight_only and (args.preflight_output is None
                                or args.preflight is not None
                                or args.preflight_sha256 is not None):
        parser.error("--preflight-only requires --preflight-output only")
    if args.compose and (args.preflight is None or not args.preflight_sha256
                         or args.preflight_output is not None):
        parser.error("--compose requires --preflight and --preflight-sha256")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result = write_preflight(args) if args.preflight_only else compose(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"{error}")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Proof-bound passive payload rebind for the existing common cardiac owner.

No map, volume polynomial, domain, physical state or deformation is generated.
The changed rows must be absent from both cardiac and tissue parent ownership.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .resting_brachiocephalic_common_field import (
    _payload_sections,
    _validate_common_field_descriptor,
)
from .resting_pleura_proxy import _parse_payload, _record_content_bytes


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


_CENTRAL_CANAL_ID = 12
_CENTRAL_CANAL_MEMBER = "FJ1737"
_CENTRAL_CANAL_MEMBER_SHA256 = "c86dd74640e7c820eda28783944a90998045ff42a131a7a1456a64fdff0e6471"
_CENTRAL_CANAL_BASE_NHA_SHA256 = "5c955be571416fed6a126d793a9737cf856c51b7355300761890e908f9b0921f"
_CENTRAL_CANAL_REBIND_NHA_SHA256 = "48781120419c325be12c2ee72d0b0af502baf86f01f8db9dac14433fdab86317"
_CENTRAL_CANAL_REBIND_RECEIPT_SHA256 = "6a762240e5e168db7231da8f89746c6b7a456aee090ce5ed7866af7006af3d40"
_CENTRAL_CANAL_REGISTRATION_SHA256 = "31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
_CENTRAL_CANAL_RIGID_MANIFEST_SHA256 = "99401961daa509af80bc089fd750ff9c83e0a2ae3a734d242efa920e12e067db"


def _sha256_file(path: str, expected: str, description: str) -> None:
    _require(isinstance(path, str) and bool(path), f"{description} path is missing")
    source = Path(path)
    _require(source.is_absolute() and source.is_file(), f"{description} file is unavailable")
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    _require(actual == expected, f"{description} file hash differs")


def _validate_central_canal_owner_migration(
    stable_id: int, base_row: dict, candidate_row: dict, base_identity: dict,
    migration: dict, base_sha: str,
) -> tuple[dict, dict, dict]:
    """Validate the one retained source-owner migration supported by this rebind."""
    _require(stable_id == _CENTRAL_CANAL_ID,
             "only authenticated central-canal row 12 has a passive owner-migration path")
    _require(isinstance(migration, dict)
             and isinstance(migration.get("receipt_raw"), bytes)
             and isinstance(migration.get("payload_raw"), bytes)
             and isinstance(migration.get("source_member_raw"), bytes),
             "central-canal owner migration requires exact receipt, payload and OBJ-member bytes")
    receipt_sha = hashlib.sha256(migration["receipt_raw"]).hexdigest()
    _require(receipt_sha == _CENTRAL_CANAL_REBIND_RECEIPT_SHA256
             and base_sha == _CENTRAL_CANAL_BASE_NHA_SHA256
             and hashlib.sha256(migration["source_member_raw"]).hexdigest()
             == _CENTRAL_CANAL_MEMBER_SHA256,
             "central-canal migration evidence or source-member bytes are not the retained inputs")
    try:
        migration_receipt = json.loads(migration["receipt_raw"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("central-canal owner-migration receipt is invalid JSON") from error
    migration_sha = hashlib.sha256(migration["payload_raw"]).hexdigest()
    _require(migration_sha == _CENTRAL_CANAL_REBIND_NHA_SHA256,
             "central-canal owner migration payload is not the retained registered payload")
    evidence_payload = migration_receipt.get("payload")
    _require(isinstance(evidence_payload, dict)
             and evidence_payload.get("sha256") == migration_sha
             and migration_receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == migration_sha,
             "central-canal owner-migration receipt does not bind its exact payload")
    proof = migration_receipt.get("provenance", {}).get("central_canal_owner_rebind")
    _require(isinstance(proof, dict)
             and proof.get("status") == "INFERRED_PASSIVE_OWNER_REBIND_SOURCE_FRAME_PRESERVED_NOT_GEOMETRY_ADMISSION"
             and proof.get("base_nha_sha256") == base_sha
             and proof.get("candidate_nha_sha256") == migration_sha
             and proof.get("physical_mass_or_state_changed") is False
             and proof.get("geometry_admission") is False,
             "central-canal owner migration is not bound to this base and candidate")
    frame = proof.get("owner_frame_audit")
    _require(isinstance(frame, dict)
             and frame.get("status") == "INFERRED_PASSIVE_VISUAL_OWNER_CORRECTION_SOURCE_FRAME_PRESERVED"
             and frame.get("stable_id") == stable_id
             and frame.get("source_member") == _CENTRAL_CANAL_MEMBER
             and frame.get("source_member_sha256") == _CENTRAL_CANAL_MEMBER_SHA256
             and frame.get("from") == {"myosim_body": "torso", "core_body_index": 20}
             and frame.get("to") == {"myosim_body": "head", "core_body_index": 23}
             and frame.get("source_face_order_unchanged") is True
             and frame.get("vertex_slot_order_unchanged") is True
             and frame.get("layer_unchanged") is True
             and frame.get("source_identity_correction_applied_before_owner_migration") is True
             and frame.get("inferred_from_reference_registration_not_subject_measurement") is True
             and frame.get("native_motion_admission") is False
             and frame.get("physical_mass_changed") is False
             and frame.get("physiological_state_or_controller_changed") is False,
             "central-canal owner-frame proof does not authorize this exact passive migration")
    _require(frame.get("registration_sha256") == _CENTRAL_CANAL_REGISTRATION_SHA256
             and frame.get("rigid_reference_manifest_sha256") == _CENTRAL_CANAL_RIGID_MANIFEST_SHA256,
             "central-canal owner-frame proof does not use the retained registration inputs")
    for value_name in ("default_pose_world_max_abs_delta_m",
                       "legacy_default_pose_world_max_abs_error_m",
                       "head_default_pose_world_max_abs_error_m"):
        value = frame.get(value_name)
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 f"central-canal owner-frame proof has invalid {value_name}")
    _require(frame["default_pose_world_max_abs_delta_m"] <= (
        frame["legacy_default_pose_world_max_abs_error_m"]
        + frame["head_default_pose_world_max_abs_error_m"]),
        "central-canal default-pose delta exceeds its two measured quantization errors")
    _sha256_file(frame.get("registration_path"), frame.get("registration_sha256"),
                 "central-canal registration")
    _sha256_file(frame.get("rigid_reference_manifest_path"),
                 frame.get("rigid_reference_manifest_sha256"),
                 "central-canal rigid reference manifest")

    _require(all(type(row.get(key)) is int for row in (base_row, candidate_row)
                 for key in ("body_index", "layer", "flags"))
             and base_row["body_index"] == 20 and candidate_row["body_index"] == 23
             and base_row["layer"] == candidate_row["layer"] == 3
             and candidate_row["flags"] == base_row["flags"],
             "central-canal typed row differs from the authenticated 20-to-23 migration")
    _require(isinstance(base_identity, dict)
             and base_identity.get("source_member") == _CENTRAL_CANAL_MEMBER
             and base_identity.get("source_sha256") == _CENTRAL_CANAL_MEMBER_SHA256
             and type(base_identity.get("body_index")) is int
             and base_identity.get("body_index") == 20
             and type(base_identity.get("layer")) is int
             and base_identity.get("layer") == 3,
             "base receipt does not identify the legacy central-canal row")

    _evidence_header, _evidence_records, evidence_rows = _parse_payload(migration["payload_raw"])
    evidence_row = evidence_rows.get(stable_id)
    _require(isinstance(evidence_row, dict)
             and all(type(evidence_row.get(key)) is int for key in ("body_index", "layer", "flags"))
             and evidence_row.get("body_index") == 23
             and evidence_row.get("layer") == 3
             and evidence_row.get("flags") == candidate_row.get("flags")
             and _record_content_bytes(evidence_row) == _record_content_bytes(candidate_row),
             "final central-canal row differs from the exact registered owner-rebind payload")
    evidence_identity = migration_receipt.get("provenance", {}).get("source_id_map", {}).get(str(stable_id))
    _require(isinstance(evidence_identity, dict)
             and type(evidence_identity.get("body_index")) is int
             and type(evidence_identity.get("layer")) is int
             and evidence_identity.get("source_member") == _CENTRAL_CANAL_MEMBER
             and evidence_identity.get("source_sha256") == _CENTRAL_CANAL_MEMBER_SHA256
             and evidence_identity.get("body_index") == 23
             and evidence_identity.get("layer") == 3
             and evidence_identity.get("name") == "Central canal of spinal cord"
             and evidence_identity.get("source_license") == "CC Attribution-Share Alike 2.1 Japan",
             "owner-rebind receipt lacks the authenticated central-canal identity")
    metadata = evidence_identity.get("source_owner_metadata")
    _require(isinstance(metadata, dict)
             and metadata.get("member_id") == _CENTRAL_CANAL_MEMBER
             and metadata.get("member_sha256") == _CENTRAL_CANAL_MEMBER_SHA256
             and metadata.get("myosim_body") == "head"
             and metadata.get("core_body_index") == 23
             and metadata.get("inferred_passive_owner_correction") == frame,
             "owner-rebind receipt metadata does not match the owner-frame proof")
    correction = metadata.get("source_identity_correction")
    _require(isinstance(correction, dict)
             and correction.get("status") == "SOURCE_HEADER_AUTHENTICATED_IDENTITY_CORRECTION"
             and correction.get("source_member") == _CENTRAL_CANAL_MEMBER
             and correction.get("source_member_sha256") == _CENTRAL_CANAL_MEMBER_SHA256
             and correction.get("stable_id_preserved") == stable_id
             and correction.get("embedded_obj_identity") == {
                 "concept_id": "FMA78497",
                 "hierarchy": "part_of",
                 "name": "Central canal of spinal cord",
             }
             and correction.get("embedded_obj_license") == "CC Attribution-Share Alike 2.1 Japan"
             and correction.get("source_archive_full_sha256_independently_verified") is False,
             "central-canal source identity/license correction is missing or inconsistent")
    return proof, evidence_identity, migration_receipt


def rebind_passive_rows(
    receipt: dict, base_raw: bytes, candidate_raw: bytes, *,
    changed_ids: tuple[int, ...], output_path: str,
    owner_migrations: dict[int, dict] | None = None,
) -> tuple[dict, dict]:
    """Rebind only after exact local geometry proves every owner unchanged."""
    _require(bool(changed_ids) and all(type(sid) is int and sid > 0 for sid in changed_ids)
             and len(set(changed_ids)) == len(changed_ids), "invalid changed anatomical IDs")
    base_sha = hashlib.sha256(base_raw).hexdigest()
    new_sha = hashlib.sha256(candidate_raw).hexdigest()
    _payload_sections(base_raw)
    _payload_sections(candidate_raw)
    bh, _br, before = _parse_payload(base_raw)
    ch, _cr, after = _parse_payload(candidate_raw)
    _require(bh[:3] == ch[:3] and bh[5:] == ch[5:]
             and list(before) == list(after), "source, registration or ordered IDs changed")
    _require(set(changed_ids).issubset(before), "changed row is missing")
    migrations = {} if owner_migrations is None else owner_migrations
    _require(isinstance(migrations, dict)
             and all(type(sid) is int and sid in changed_ids for sid in migrations),
             "owner migrations must be explicitly declared changed rows")
    _require(receipt["payload"]["sha256"] == base_sha
             and receipt["functional_bindings"]["anatomy_payload_sha256"] == base_sha,
             "receipt is not bound to the actual base payload")
    common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    _require(common.get("schema") == "numi.human.cardiac_common_field.v1"
             and common["anatomy_payload_sha256"] == base_sha, "current common owner mismatch")
    owned = set(common["volume_stable_ids"])
    for key in ("vertex_ranges", "passive_attachment_ranges"):
        for row in common[key]:
            sid = row["stable_id"]
            _require(type(sid) is int and sid in before, "invalid common-field row ID")
            _require(type(row["vertex_count"]) is int
                     and row["vertex_count"] == len(before[sid]["vertices6"]),
                     "common-field vertex count differs from actual anatomy")
            owned.add(sid)
    parents = receipt.get("passive_tissue_parent_bindings")
    if parents is not None:
        _require(parents["anatomy_payload_sha256"] == base_sha, "parent payload mismatch")
        for row in parents.get("anatomy_surfaces", []):
            _require(type(row["stable_id"]) is int, "invalid anatomy parent row ID")
            owned.add(row["stable_id"])
    _require(not owned.intersection(changed_ids), "changed row has a deformation owner")
    identity_map = receipt["provenance"]["source_id_map"]
    _require(all(str(sid) in identity_map for sid in before), "anatomical identity is missing")
    migration_proofs = {}
    migrated_identities = {}
    actual_changes = []
    for sid in before:
        a, b = before[sid], after[sid]
        typed_changed = any(a[key] != b[key] for key in ("body_index", "layer", "flags"))
        if typed_changed:
            _require(sid in migrations,
                     f"anatomical owner or layer changed without an owner proof for {sid}")
            proof, identity, _evidence_receipt = _validate_central_canal_owner_migration(
                sid, a, b, identity_map.get(str(sid), {}), migrations[sid], base_sha)
            migration_proofs[sid] = proof
            migrated_identities[sid] = identity
        else:
            _require(sid not in migrations,
                     f"owner migration was declared for unchanged typed row {sid}")
        if typed_changed or _record_content_bytes(a) != _record_content_bytes(b):
            actual_changes.append(sid)
    _require(set(actual_changes) == set(changed_ids),
             "geometry changed outside the declared passive rows or declared row is unchanged")
    result = copy.deepcopy(receipt)
    result["payload"].update(path=output_path, sha256=new_sha,
                             surface_count=ch[2], vertex_count=ch[3], index_count=ch[4])
    result["functional_bindings"]["anatomy_payload_sha256"] = new_sha
    for sid, identity in migrated_identities.items():
        result["provenance"]["source_id_map"][str(sid)] = copy.deepcopy(identity)
    if migration_proofs:
        result["provenance"].setdefault("verified_passive_owner_migrations", {}).update({
            str(sid): {
                "status": migration_proofs[sid]["status"],
                "migration_evidence_receipt_sha256": hashlib.sha256(
                    migrations[sid]["receipt_raw"]).hexdigest(),
                "migration_evidence_payload_sha256": migration_proofs[sid]["candidate_nha_sha256"],
                "source_member_sha256": hashlib.sha256(
                    migrations[sid]["source_member_raw"]).hexdigest(),
                "final_payload_sha256": new_sha,
                "source_member": _CENTRAL_CANAL_MEMBER,
                "owner_frame_audit": copy.deepcopy(
                    migration_proofs[sid]["owner_frame_audit"]),
                "geometry_admission": False,
                "native_motion_admission": False,
            } for sid in migration_proofs
        })
    candidate_common = result["provenance"]["cardiac_geometry_binding"]["common_field"]
    candidate_common["anatomy_payload_sha256"] = new_sha
    _validate_common_field_descriptor(candidate_common, common)
    if parents is not None:
        result["passive_tissue_parent_bindings"]["anatomy_payload_sha256"] = new_sha
    audit = {
        "status": "SOURCE_PREPARATION_ONLY_NOT_ADMISSION",
        "base_payload_sha256": base_sha,
        "candidate_payload_sha256": new_sha,
        "changed_passive_stable_ids": sorted(changed_ids),
        "changed_owner_migration_stable_ids": sorted(migration_proofs),
        "unchanged_row_count": len(before) - len(changed_ids),
        "common_owned_stable_ids": sorted(owned),
        "ordered_anatomical_identities_preserved": True,
        "owners_unchanged_except_explicit_proved_migrations": True,
        "common_field_descriptor_preserved_except_payload_sha256": True,
        "common_sidecars_must_be_verified_by_driver": True,
        "physical_state_mass_and_force_added": False,
        "native_admission": False, "anatomical_admission": False,
    }
    result["provenance"]["verified_passive_common_field_rebind"] = audit
    return result, audit

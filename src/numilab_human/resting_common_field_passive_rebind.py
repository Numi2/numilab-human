"""Proof-bound passive payload rebind for the existing common cardiac owner.

No map, volume polynomial, domain, physical state or deformation is generated.
The changed rows must be absent from both cardiac and tissue parent ownership.
"""
from __future__ import annotations

import copy
import hashlib

from .resting_brachiocephalic_common_field import (
    _payload_sections,
    _validate_common_field_descriptor,
)
from .resting_pleura_proxy import _parse_payload, _record_content_bytes


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def rebind_passive_rows(
    receipt: dict, base_raw: bytes, candidate_raw: bytes, *,
    changed_ids: tuple[int, ...], output_path: str,
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
    actual_changes = []
    for sid in before:
        a, b = before[sid], after[sid]
        _require(all(a[key] == b[key] for key in ("body_index", "layer", "flags")),
                 f"anatomical owner or layer changed for {sid}")
        if _record_content_bytes(a) != _record_content_bytes(b):
            actual_changes.append(sid)
    _require(set(actual_changes) == set(changed_ids),
             "geometry changed outside the declared passive rows or declared row is unchanged")
    result = copy.deepcopy(receipt)
    result["payload"].update(path=output_path, sha256=new_sha,
                             surface_count=ch[2], vertex_count=ch[3], index_count=ch[4])
    result["functional_bindings"]["anatomy_payload_sha256"] = new_sha
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
        "unchanged_row_count": len(before) - len(changed_ids),
        "common_owned_stable_ids": sorted(owned),
        "ordered_anatomical_identities_and_owners_preserved": True,
        "common_field_descriptor_preserved_except_payload_sha256": True,
        "common_sidecars_must_be_verified_by_driver": True,
        "physical_state_mass_and_force_added": False,
        "native_admission": False, "anatomical_admission": False,
    }
    result["provenance"]["verified_passive_common_field_rebind"] = audit
    return result, audit

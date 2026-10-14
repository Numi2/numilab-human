"""Narrow composition of brachiocephalic parent maps into a joint owner."""
from __future__ import annotations

import hashlib
import math
import re
import struct
from copy import deepcopy
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VASCULAR_IDS = (475, 477)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_hash_index(records: list[dict[str, Any]], label: str) -> dict[str, int]:
    _require(isinstance(records, list) and 0 < len(records) <= 8,
             f"{label} source inventory is outside the native ABI bound")
    result: dict[str, int] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"{label} source record is invalid")
        path, digest = record.get("path"), record.get("sha256")
        _require(isinstance(path, str) and bool(path),
                 f"{label} source path is missing")
        _require(isinstance(digest, str) and _SHA256.fullmatch(digest),
                 f"{label} source SHA-256 is invalid")
        _require(digest not in result, f"{label} source inventory repeats a payload hash")
        result[digest] = index
    return result


def _validated_vascular_rows(
    extension: dict[str, Any], source_count: int, label: str,
    expected_ids: tuple[int, ...], *, weights_are_float32: bool = True,
) -> dict[int, dict[str, Any]]:
    rows = extension.get("anatomy_surfaces")
    _require(isinstance(rows, list), f"{label} anatomy surface maps are missing")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), f"{label} anatomy surface row is invalid")
        stable_id = row.get("stable_id")
        _require(isinstance(stable_id, int) and not isinstance(stable_id, bool),
                 f"{label} anatomy stable ID is invalid")
        _require(stable_id not in by_id, f"{label} anatomy stable ID is duplicated")
        by_id[stable_id] = row
    _require(set(by_id) == set(expected_ids),
             f"{label} anatomy stable IDs differ from the declared scope")
    for stable_id, row in by_id.items():
        vertices = row.get("vertices")
        _require(isinstance(vertices, list) and len(vertices) > 0,
                 f"{label} row {stable_id} has no vertex ancestry")
        for vertex_index, terms in enumerate(vertices):
            _require(isinstance(terms, list) and bool(terms),
                     f"{label} row {stable_id} has an empty parent map")
            weights: list[float] = []
            for term in terms:
                _require(isinstance(term, list) and len(term) == 4,
                         f"{label} row {stable_id} has a malformed parent term")
                source_index, carrier_surface, carrier_vertex, weight = term
                _require(isinstance(source_index, int) and not isinstance(source_index, bool)
                         and 0 <= source_index < source_count,
                         f"{label} row {stable_id} parent source index is out of range")
                _require(isinstance(carrier_surface, int) and not isinstance(carrier_surface, bool)
                         and carrier_surface > 0,
                         f"{label} row {stable_id} carrier surface ID is invalid")
                _require(isinstance(carrier_vertex, int) and not isinstance(carrier_vertex, bool)
                         and carrier_vertex == vertex_index,
                         f"{label} row {stable_id} carrier vertex ancestry is not ordinal-bound")
                _require(isinstance(weight, (int, float)) and not isinstance(weight, bool)
                         and math.isfinite(weight) and weight > 0.0,
                         f"{label} row {stable_id} parent weight is not finite and positive")
                value = float(weight)
                try:
                    stored_f32 = struct.unpack("<f", struct.pack("<f", value))[0]
                except OverflowError:
                    stored_f32 = math.nan
                _require(math.isfinite(stored_f32) and stored_f32 > 0.0,
                         f"{label} row {stable_id} parent weight is not representable")
                if weights_are_float32:
                    _require(stored_f32 == value,
                             f"{label} row {stable_id} parent weight is not exact Float32")
                weights.append(value)
            # These receipt weights are the exact Float32 source values. Their
            # binary64 sum may differ from one by half an ULP at 1.0.
            tolerance = 2.0 ** -24 if weights_are_float32 else 1e-12
            _require(abs(math.fsum(weights) - 1.0) <= tolerance,
                     f"{label} row {stable_id} parent weights do not form a Float32 unit partition")
    return by_id



def _joint_vascular_ids(extension: dict[str, Any]) -> tuple[int, ...]:
    """Accept the original row477 owner or its existing two-row successor."""
    rows = extension.get("anatomy_surfaces")
    _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
             "joint anatomy surface maps are missing")
    ids = [row.get("stable_id") for row in rows]
    _require(all(type(sid) is int for sid in ids) and len(set(ids)) == len(ids)
             and set(ids) in ({477}, {475, 477}),
             "joint anatomy stable IDs differ from the supported vascular scopes")
    return tuple(sorted(ids))


def merge_vascular_parent_maps(
    common_extension: dict[str, Any],
    joint_extension: dict[str, Any],
    *,
    final_anatomy_sha256: str,
    final_child_sha256: str,
    common_source_to_joint: dict[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay rows 475/477 on the joint extension without reindexing owners.

    ``common_source_to_joint`` is derived by SHA-256 identity by the source
    driver. This pure function makes the index remap explicit and testable.
    All existing joint source records and muscle surface maps are copied in
    their original order and content.
    """
    _require(isinstance(final_anatomy_sha256, str)
             and _SHA256.fullmatch(final_anatomy_sha256),
             "final anatomy SHA-256 is invalid")
    _require(isinstance(final_child_sha256, str)
             and _SHA256.fullmatch(final_child_sha256),
             "final child SHA-256 is invalid")
    common_sources = common_extension.get("source_payloads")
    joint_sources = joint_extension.get("source_payloads")
    common_index = _source_hash_index(common_sources, "common")
    joint_index = _source_hash_index(joint_sources, "joint")
    common_rows = _validated_vascular_rows(
        common_extension, len(common_sources), "common", _VASCULAR_IDS)
    joint_rows = _validated_vascular_rows(
        joint_extension, len(joint_sources), "joint", _joint_vascular_ids(joint_extension),
        weights_are_float32=False)
    _require(joint_extension.get("child_payload_sha256") == final_child_sha256,
             "joint child hash differs from parent-binding extension")

    referenced = sorted({int(term[0]) for row in common_rows.values()
                         for terms in row["vertices"] for term in terms})
    _require(set(referenced).issubset(common_source_to_joint),
             "not every referenced common source has an identity-bound joint index")
    for common_source_index in referenced:
        _require(common_source_index in range(len(common_sources)),
                 "referenced common source index is out of range")
        joint_source_index = common_source_to_joint[common_source_index]
        _require(joint_source_index in range(len(joint_sources)),
                 "identity remap points outside the joint source inventory")
        digest = common_sources[common_source_index]["sha256"]
        _require(common_index.get(digest) == common_source_index,
                 "common source index does not match its payload identity")
        _require(joint_sources[joint_source_index]["sha256"] == digest,
                 "source-index remap does not preserve payload identity")
        _require(joint_index.get(digest) == joint_source_index,
                 "source-index remap is not the unique joint payload owner")

    merged = deepcopy(joint_extension)
    before_sources = deepcopy(joint_sources)
    before_surface_maps = deepcopy(joint_extension.get("surfaces"))
    existing_rows = {int(row["stable_id"]): deepcopy(row)
                     for row in joint_extension.get("anatomy_surfaces", [])}
    _require(477 in existing_rows,
             "joint extension does not contain its prior row 477 owner map")

    output_rows = []
    remapped_term_count = 0
    referenced_source_hashes: dict[str, int] = {}
    for stable_id in _VASCULAR_IDS:
        source_row = deepcopy(common_rows[stable_id])
        for vertex_index, terms in enumerate(source_row["vertices"]):
            for term in terms:
                original_index = int(term[0])
                mapped_index = common_source_to_joint[original_index]
                referenced_source_hashes[common_sources[original_index]["sha256"]] = mapped_index
                term[0] = mapped_index
                remapped_term_count += 1
        prior_row = existing_rows.get(stable_id)
        if prior_row is not None:
            for key, value in prior_row.items():
                if key == "vertices":
                    continue
                if key in source_row:
                    _require(source_row[key] == value,
                             f"row {stable_id} has conflicting retained source metadata")
                else:
                    source_row[key] = value
        output_rows.append(source_row)

    merged_rows = {int(row["stable_id"]): deepcopy(row)
                   for row in joint_extension.get("anatomy_surfaces", [])
                   if int(row["stable_id"]) not in _VASCULAR_IDS}
    merged_rows.update({int(row["stable_id"]): row for row in output_rows})
    merged["anatomy_surfaces"] = [merged_rows[key] for key in sorted(merged_rows)]
    merged["anatomy_payload_sha256"] = final_anatomy_sha256
    merged["child_payload_sha256"] = final_child_sha256

    _require(merged["source_payloads"] == before_sources,
             "composition changed existing joint source order or identity")
    _require(merged.get("surfaces") == before_surface_maps,
             "composition changed an existing muscle parent map")
    merged_ids = [int(row["stable_id"]) for row in merged["anatomy_surfaces"]]
    _require(len(merged_ids) == len(set(merged_ids)),
             "composition duplicated an anatomy parent-map stable ID")
    audit = {
        "status": "source_parent_binding_composition_only_not_admission",
        "overlaid_stable_ids": list(_VASCULAR_IDS),
        "input_common_stable_ids": list(_VASCULAR_IDS),
        "input_joint_stable_ids": sorted(joint_rows),
        "existing_joint_source_count_preserved": len(before_sources),
        "source_payload_inventory_prepared_before_overlay": True,
        "referenced_common_source_indices": referenced,
        "source_hash_to_joint_index": referenced_source_hashes,
        "remapped_parent_term_count": remapped_term_count,
        "joint_muscle_surface_maps_byte_semantics_preserved": True,
        "joint_source_payload_prefix_preserved": True,
        "physical_state_mass_and_force_fields_added": False,
        "geometry_or_native_predicates_run": False,
    }
    return merged, audit


def extend_joint_source_inventory(
    common_extension: dict[str, Any],
    joint_extension: dict[str, Any],
    *,
    append_paths_by_common_index: dict[int, str],
) -> tuple[dict[str, Any], dict[int, int], dict[str, Any]]:
    """Resolve referenced common parent payloads against the joint inventory.

    Existing joint records are never reordered or rewritten. Missing payloads
    are appended in ascending common-source-index order using caller-provided
    output paths, while identity is determined only by SHA-256.
    """
    common_sources = common_extension.get("source_payloads")
    joint_sources = joint_extension.get("source_payloads")
    common_index = _source_hash_index(common_sources, "common")
    joint_index = _source_hash_index(joint_sources, "joint")
    common_rows = _validated_vascular_rows(
        common_extension, len(common_sources), "common", _VASCULAR_IDS)
    _validated_vascular_rows(
        joint_extension, len(joint_sources), "joint", _joint_vascular_ids(joint_extension),
        weights_are_float32=False)
    referenced = sorted({int(term[0]) for row in common_rows.values()
                         for terms in row["vertices"] for term in terms})
    output = deepcopy(joint_extension)
    original_prefix = deepcopy(joint_sources)
    source_map: dict[int, int] = {}
    appended: list[dict[str, Any]] = []
    for common_source_index in referenced:
        digest = common_sources[common_source_index]["sha256"]
        _require(common_index.get(digest) == common_source_index,
                 "common source index does not match its payload identity")
        existing = joint_index.get(digest)
        if existing is not None:
            source_map[common_source_index] = existing
            continue
        path = append_paths_by_common_index.get(common_source_index)
        _require(isinstance(path, str) and bool(path),
                 "missing output path for an unowned common source payload")
        new_index = len(output["source_payloads"])
        _require(new_index < 8, "joint source inventory would exceed the native ABI bound")
        output["source_payloads"].append({"path": path, "sha256": digest})
        joint_index[digest] = new_index
        source_map[common_source_index] = new_index
        appended.append({
            "common_source_index": common_source_index,
            "joint_source_index": new_index,
            "sha256": digest,
            "path": path,
        })
    _require(output["source_payloads"][:len(original_prefix)] == original_prefix,
             "source extension changed an existing joint source prefix")
    return output, source_map, {
        "original_joint_source_count": len(original_prefix),
        "final_joint_source_count": len(output["source_payloads"]),
        "joint_source_prefix_preserved": True,
        "source_payloads_appended": appended,
        "common_source_to_joint_source_index": source_map,
        "referenced_common_source_indices": referenced,
    }

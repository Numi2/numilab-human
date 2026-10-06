"""Admit one repaired passive taenia through the existing NHANAT1 asset path.

This source-preparation step adds no runtime state, mass, force, or physiology.
It requires exact source-bound audits and explicit colon-interface witnesses.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .resting_anatomy_interface_patch import HEADER, RECORD, signed_volume
from .resting_passive_interfaces import _surface
from .resting_pleura_proxy import _parse_payload, _record_content_bytes

MEMBERS = {454: "FJ2566", 455: "FJ2567", 456: "FJ2568", 457: "FJ2569",
           458: "FJ2570", 459: "FJ2571", 460: "FJ2572"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError("passive taenia reference: " + message)


def _complete_zero(row):
    return (type(row.get("count")) is int and row["count"] == 0
            and row.get("audit_complete", True) and not row.get("count_is_lower_bound", False))


def build_candidate(base_payload: Path, base_receipt: Path, candidate_path: Path,
                    audit_path: Path, shape_path: Path, interface_path: Path,
                    output: Path) -> dict:
    _require(not output.exists(), "retain existing output; choose a new directory")
    raw = base_payload.read_bytes(); source_sha = hashlib.sha256(raw).hexdigest()
    header, records, rows = _parse_payload(raw)
    receipt = json.loads(base_receipt.read_text())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == source_sha
             and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_sha,
             "receipt does not identify the current payload")
    passive = set(receipt["functional_bindings"]["passive_viscera_geometry_binding"]["stable_ids"])
    source_map = receipt["provenance"]["source_id_map"]
    _require(set(MEMBERS).issubset(passive) and all(source_map.get(str(s), {}).get("source_member") == member
             for s, member in MEMBERS.items()), "colon source identities changed")
    _require(rows[457]["body_index"] == 20 and rows[457]["layer"] == 1,
             "taenia lacks the shared passive torso source frame")
    cardiac = receipt["provenance"]["cardiac_geometry_binding"]
    wall = cardiac["ventricular_wall_binding"]
    _require(cardiac.get("output_anatomy_payload_sha256") == source_sha
             and wall.get("output_anatomy_payload_sha256") == source_sha,
             "existing cardiac bindings are stale")
    _require("local_coefficient_refinement" not in wall.get("wall_map", {}),
             "attach source-bound cardiac correction after passive asset preparation")
    digest = _sha(candidate_path)
    audit = json.loads(audit_path.read_text()); shape = json.loads(shape_path.read_text())
    interfaces = json.loads(interface_path.read_text())
    _require(audit.get("source_payload_sha256") == source_sha
             and audit.get("candidate_sha256") == digest and _complete_zero(audit.get("self", {})),
             "candidate lacks a current complete exact self audit")
    ledger = {str(s): hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in passive}
    _require(audit.get("source_record_sha256") == ledger, "passive-neighbor audit sources changed")
    _require(shape.get("source_sha256") == audit.get("source_weld_sha256")
             and audit.get("source_weld_record_sha256") == ledger["457"],
             "shape reference does not bind the original taenia record")
    _require(shape.get("candidate_sha256") == digest and shape.get("sampled_distance_bound_m") == .0005
             and shape.get("relative_volume_bound") == .04,
             "shape evidence is absent or uses different bounds")
    _require(all(0 <= shape[k]["max_m"] <= .0005 for k in ("source_to_candidate", "candidate_to_source")),
             "candidate exceeds the sampled shape bound")
    _require(interfaces.get("candidate_sha256") == digest
             and interfaces.get("audit_sha256") == audit.get("exact_audit_sha256")
             and interfaces.get("predicate_sha256") == audit.get("predicate_sha256"),
             "interface witnesses do not bind the exact candidate audit")
    by_id = {}
    for row in audit.get("pairs", []):
        ids = row.get("ids", [])
        _require(len(ids) == 2 and ids[0] == 457 and ids[1] not in by_id, "invalid or repeated neighbor pair")
        by_id[ids[1]] = row
    _require(set(by_id) == passive - {457}, "audit does not cover every passive neighbor")
    classified = {row["stable_ids"][1]: row for row in interfaces.get("interfaces", [])}
    _require(len(classified) == len(interfaces.get("interfaces", [])), "repeated interface classification")
    positive = set()
    for sid, pair in by_id.items():
        _require(pair.get("audit_complete", True) and not pair.get("invalid_geometry")
                 and not pair.get("count_is_lower_bound", False) and type(pair.get("count")) is int,
                 "neighbor audit is incomplete or invalid")
        if _complete_zero(pair):
            continue
        positive.add(sid); item = classified.get(sid, {})
        _require(sid in MEMBERS and sid != 457 and pair["count"] > 0
                 and pair["count"] == len(pair.get("triangle_pairs", []))
                 and item.get("source_triangle_pairs") == pair["triangle_pairs"]
                 and item.get("source_members") == [MEMBERS[457], MEMBERS[sid]],
                 "crossing lacks its exact source-component interface witnesses")
        if sid in (456, 458):
            span = item.get("distance_from_caudal_ascending_extent_m", [])
            _require(item.get("interface") == "taenia_convergence_at_caudal_ascending_colon"
                     and len(span) == 2 and 0 <= span[0] <= span[1] <= .020,
                     "band crossing lies outside its localized source convergence")
        elif sid == 459:
            span = item.get("distance_from_cranial_rectal_extent_m", [])
            _require(item.get("interface") == "longitudinal_muscle_continuation_at_rectal_entry"
                     and len(span) == 2 and 0 <= span[0] <= span[1] <= .005,
                     "rectal crossing lies outside the source entry")
        else:
            _require(item.get("interface") == "longitudinal_muscle_component_of_colon_wall",
                     "colon wall ownership is not explicit")
    _require(set(classified) == positive, "interface list includes unobserved or missing crossings")
    with np.load(candidate_path, allow_pickle=False) as data:
        vertices, faces, quality = _surface(data["vertices"], data["faces"])
    old_volume = signed_volume(rows[457]["vertices6"][:, :3].astype(float), rows[457]["faces"])
    _require(old_volume > 0 and abs(quality["volume_m3"] / old_volume - 1) <= .04,
             "candidate exceeds the passive volume bound")
    _require(abs(shape["source_volume_m3"] - old_volume) <= 1e-12
             and abs(shape["candidate_volume_m3"] - quality["volume_m3"]) <= 1e-12,
             "shape evidence volumes differ from actual geometry")
    updated = copy.deepcopy(rows); updated[457]["vertices6"] = vertices; updated[457]["faces"] = faces
    chunks, vv, ff, nv, ni = [], [], [], 0, 0
    for record in records:
        sid = record[5]; row = updated[sid]; v, f = row["vertices6"], row["faces"]
        chunks.append(RECORD.pack(row["body_index"], nv, len(v), ni, f.size, sid, row["layer"], row["flags"]))
        vv.append(np.asarray(v, dtype="<f4").tobytes()); ff.append((f.reshape(-1)+nv).astype("<u4").tobytes())
        nv += len(v); ni += f.size
    result = HEADER.pack(header[0], header[1], len(records), nv, ni, header[5], header[6]) + b"".join(chunks+vv+ff)
    _, _, verified = _parse_payload(result)
    _require(all(_record_content_bytes(rows[s]) == _record_content_bytes(verified[s]) for s in rows if s != 457),
             "another anatomical surface changed")
    output_sha = hashlib.sha256(result).hexdigest()
    detail = {"interpretation": "inferred_passive_colon_muscle_reference_surface_not_measured_subject_geometry",
              "compiler_source_sha256": _sha(Path(__file__)), "base_payload_sha256": source_sha,
              "base_receipt_sha256": _sha(base_receipt), "candidate_sha256": digest,
              "audit_sha256": _sha(audit_path), "shape_sha256": _sha(shape_path), "interfaces_sha256": _sha(interface_path),
              "shape": shape, "quality": quality, "source_interfaces": interfaces,
              "additional_physical_mass_kg": 0, "additional_physiological_state": False,
              "all_other_record_geometry_bytes_preserved": True,
              "qualification": "Exact source self and non-colon clearance passed; source colon interfaces are explicitly localized. Accepted native breathing-cycle checks remain required."}
    receipt["provenance"]["passive_taenia_reference"] = detail
    source_map["457"]["reference_geometry_status"] = detail["interpretation"]
    receipt["payload"].update(path=str(output/"resting-thorax.nhanatomy"), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac["output_anatomy_payload_sha256"] = output_sha; wall["output_anatomy_payload_sha256"] = output_sha
    receipt["qualification"]["passive_taenia_reference"] = detail["qualification"]
    output.mkdir(parents=True); (output/"resting-thorax.nhanatomy").write_bytes(result)
    receipt_path = output/"resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"], "qualification": receipt["qualification"],
                "source_surfaces": source_map, "mass_geometry_accounting": receipt["mass_geometry_accounting"]}
    (output/"resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest

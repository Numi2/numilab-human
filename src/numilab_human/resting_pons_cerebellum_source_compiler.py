"""Source-only ABI5 compiler for the pinned fourth-ventricle/cerebellum successor.

This compiles stable IDs 323 and 348 through the shared ABI5 row-replacement
owner. The strict pair certificate is diagnostic evidence only; it does not
admit anatomy, valve behavior, motion, or a native combined scene.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from .resting_pleura_proxy import _parse_payload

BASE_NHA_SHA256 = "34d59a56552cb4b6be3a78174b0047c2a5f37461d089bbdf954e774c947a2ec1"
PREFLIGHT_SHA256 = "b3cd998a2e2262154a7f0396ca4d858a14a888e2f4ba677cf7c6f42fbc3407e7"
CANDIDATE_NPZ_SHA256 = "cbf6149a9b961368f1ea334db546a8b1520a6fa5dff556bff06dcea877200a4f"
LINEAGE_SHA256 = "6e3acef94d1eadb619cacb44f4b1a5a54179c4a8430ec123d12fd885025b6377"
STRICT_AUDIT_SHA256 = "63f4c83709146eb099543530a9fc3b3c2fbb7341bfaa5dc830407bd584451908"
PARENT_CANDIDATE_008_SHA256 = "36a4c384b6dc0d22ea289cc16f90348289ba6a88bf8fb9dbe6d10da0c5e12a7f"
PARENT_LINEAGE_008_SHA256 = "50fd859345447a38aa6ecb5772f43b159be0bd7d021f2cc06d17bb38ea94c376"
CERTIFICATE_SOURCE_SHA256 = "b5bf8ef8fb76657342bb9e21d4faa22fb38db8f335b3a8930295e90004ded1e1"

_ROWS = {
    323: ("row323_vertices6", "row323_faces", (11172, 15470)),
    348: ("row348_vertices6", "row348_faces", (12098, 23818)),
}
_EXPECTED_SOURCES = {
    323: {323: "FJ1731", 12: "FJ1737", 326: "FJ1738"},
    348: {348: "FJ1781", 323: "FJ1731"},
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_row_replacements(base_rows: dict[int, dict[str, Any]], arrays: Any) -> dict[int, dict[str, Any]]:
    """Use candidate arrays for 323/348 while retaining each complete ABI5 row owner."""
    _need(set(_ROWS).issubset(base_rows), "parent NHA lacks stable ID 323 or 348")
    replacements: dict[int, dict[str, Any]] = {}
    for stable_id, (vertex_key, face_key, source_counts) in _ROWS.items():
        base = base_rows[stable_id]
        _need((int(base["body_index"]), int(base["layer"]), int(base["flags"])) == (23, 1, 0),
              f"parent stable ID {stable_id} is not Body23/layer1/passive")
        vertices = np.ascontiguousarray(arrays[vertex_key], dtype="<f4")
        faces = np.ascontiguousarray(arrays[face_key], dtype="<i8")
        _need(vertices.ndim == 2 and vertices.shape[1] == 6 and len(vertices) >= source_counts[0],
              f"candidate vertex slots are malformed for stable ID {stable_id}")
        _need(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > source_counts[1],
              f"candidate faces are malformed for stable ID {stable_id}")
        _need(np.isfinite(vertices).all(), f"candidate row {stable_id} has non-finite ABI5 slots")
        _need(faces.size == 0 or (int(faces.min()) >= 0 and int(faces.max()) < len(vertices)),
              f"candidate row {stable_id} contains a nonlocal face index")
        replacement = dict(base)
        replacement["vertices6"] = vertices
        replacement["faces"] = faces
        _need((replacement["body_index"], replacement["layer"], replacement["flags"])
              == (base["body_index"], base["layer"], base["flags"]),
              f"candidate row {stable_id} changed an ABI5 owner field")
        replacements[stable_id] = replacement
    return replacements


def compile_payload(base_nha: bytes, replacements: dict[int, dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    """Re-emit through the existing ABI5 owner and assert only selected rows changed."""
    output, owner_report = encode_payload_with_row_replacements(base_nha, replacements)
    header, _records, rows = _parse_payload(output)
    _need(header[0] == b"NHANAT1\0" and int(header[1]) == 5,
          "existing ABI5 owner emitted an unexpected payload format")
    _need(owner_report.get("replaced_stable_ids") == [323, 348]
          and owner_report.get("row_count_preserved") == len(rows)
          and owner_report.get("unmodified_rows_byte_exact") == len(rows) - 2
          and owner_report.get("registration_and_source_header_bytes_preserved") is True,
          "existing ABI5 row-replacement owner did not preserve the full parent")
    for stable_id in _ROWS:
        _need((int(rows[stable_id]["body_index"]), int(rows[stable_id]["layer"]),
               int(rows[stable_id]["flags"])) == (23, 1, 0),
              f"compiled stable ID {stable_id} changed its parent owner fields")
    return output, owner_report


def _walk_lineage(entry: dict[str, Any]):
    yield entry
    for key in ("input_face_parent_ancestry", "previous_source_face_ancestry",
                "parent_row_face_lineage"):
        parent = entry.get(key)
        if isinstance(parent, dict):
            yield from _walk_lineage(parent)


def _verify_lineage(lineage: dict[str, Any], candidate_face_counts: dict[int, int]) -> dict[str, Any]:
    _need(lineage.get("schema") == "numi.human.fourth-ventricle-cerebellum-local-patch-successor-ancestry.v1",
          "candidate lineage schema changed")
    _need(lineage.get("anatomical_admission") is False
          and lineage.get("physical_state_mass_controller_changed") is False
          and lineage.get("source_geometry_inference_declared") is True
          and lineage.get("row323_faces_unchanged") is True
          and lineage.get("row323_row348_positions_unchanged") is True
          and lineage.get("unchanged_interface_face_count") == 1617,
          "candidate lineage scope or retained interface changed")
    rows = lineage.get("rows")
    _need(isinstance(rows, dict) and set(rows) == {"323", "348"},
          "candidate lineage does not cover exactly rows 323 and 348")
    observed: dict[str, set[str]] = {"323": set(), "348": set()}
    for stable_id, face_count in candidate_face_counts.items():
        entries = rows[str(stable_id)]
        _need(isinstance(entries, list) and len(entries) == face_count,
              f"candidate face ancestry count differs for stable ID {stable_id}")
        allowed = _EXPECTED_SOURCES[stable_id]
        for face_id, entry in enumerate(entries):
            _need(isinstance(entry, dict) and entry.get("output_face") == face_id
                  and entry.get("owner_stable_id") == stable_id,
                  f"candidate face {stable_id}:{face_id} lost its output identity")
            source_seen = False
            inherited_source_id = None
            for node in _walk_lineage(entry):
                source_id = node.get("source_stable_id")
                member = node.get("source_member")
                member_sha = node.get("source_member_sha256")
                if source_id is not None:
                    _need(type(source_id) is int and source_id in allowed,
                          f"candidate face {stable_id}:{face_id} has an unowned source stable ID")
                    inherited_source_id = source_id
                if member is not None or member_sha is not None:
                    _need(inherited_source_id in allowed
                          and member == allowed[inherited_source_id]
                          and isinstance(member_sha, str) and len(member_sha) == 64
                          and all(c in "0123456789abcdef" for c in member_sha),
                          f"candidate face {stable_id}:{face_id} has malformed source identity")
                    observed[str(stable_id)].add(
                        f"{inherited_source_id}:{member}:{member_sha}")
                    source_seen = True
            if entry.get("operation") == "inferred_fixed_boundary_local_retriangulation":
                patch_ids = entry.get("inferred_support_parent_source_stable_ids")
                support_faces = entry.get("inferred_support_parent_output_faces")
                support_sources = entry.get("inferred_support_parent_source_faces")
                support_sha = entry.get("inferred_support_lineage_sha256")
                _need(entry.get("direct_parent_face_claim") is False
                      and entry.get("positions_and_vertex_slots_preserved") is True
                      and entry.get("partition_input_row_stable_id") == stable_id
                      and entry.get("partition_input_row_face") is None
                      and isinstance(support_sha, str) and len(support_sha) == 64
                      and all(c in "0123456789abcdef" for c in support_sha)
                      and isinstance(support_faces, list) and support_faces
                      and all(type(value) is int and value >= 0 for value in support_faces)
                      and isinstance(support_sources, list) and support_sources
                      and all(type(value) is int and value >= 0 for value in support_sources)
                      and isinstance(patch_ids, list) and patch_ids
                      and all(type(value) is int and value in allowed for value in patch_ids),
                      f"candidate face {stable_id}:{face_id} has incomplete inferred-patch support")
                for source_stable_id in patch_ids:
                    source_member = allowed[source_stable_id]
                    _need(source_member,
                          f"candidate face {stable_id}:{face_id} patch support has unknown source owner")
                source_seen = True
            _need(source_seen, f"candidate face {stable_id}:{face_id} has no source-member ancestry")
    return {
        "every_candidate_face_has_nested_source_ancestry": True,
        "candidate_face_counts": {str(k): v for k, v in candidate_face_counts.items()},
        "observed_source_member_identities": {key: sorted(value) for key, value in observed.items()},
        "ancestry_sidecar_sha256": LINEAGE_SHA256,
    }


def _verify_gate_reports(preflight: dict[str, Any], strict: dict[str, Any]) -> dict[str, Any]:
    _need(preflight.get("schema") == "numi.human.fourth-ventricle-cerebellum-local-patch-successor-preflight.v1"
          and preflight.get("status") == "PASS_COMPLETE_NESTED_ANCESTRY_AND_SERIALIZATION_PREFLIGHT",
          "candidate preflight did not pass exact nested-ancestry serialization checks")
    _need(preflight.get("candidate_sha256") == CANDIDATE_NPZ_SHA256
          and preflight.get("lineage_sha256") == LINEAGE_SHA256
          and preflight.get("nested_parent_ancestry_complete") is True
          and preflight.get("fixed_boundary_exact") is True
          and preflight.get("local_screen_self_and_unexpected_cross_contacts_zero") is True
          and preflight.get("all_vertex_positions_slots_materials_unchanged") is True
          and preflight.get("retained_exact_interface_face_count") == 1617,
          "candidate preflight pins or exact fixed-interface gates changed")
    inputs = preflight.get("input_sha256", {})
    _need(inputs.get("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/pons-fourth-ventricle-reconstruction-001/fourth-ventricle-cerebellum-local-patch-008/prepared-candidate.npz") == PARENT_CANDIDATE_008_SHA256
          and inputs.get("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/pons-fourth-ventricle-reconstruction-001/fourth-ventricle-cerebellum-local-patch-008/prepared-face-ancestry.json.gz") == PARENT_LINEAGE_008_SHA256,
          "candidate lineage does not retain the authenticated 008 parent")
    _need(strict.get("schema") == "numi.human.fourth-ventricle-cerebellum-local-patch-successor-strict-audit.v1"
          and strict.get("status") == "PASS_STRICT_F32_SELF_CLOSED_INTERFACE_AND_SOLID_DISJOINTNESS"
          and strict.get("candidate_sha256") == CANDIDATE_NPZ_SHA256
          and strict.get("candidate_lineage_sha256") == LINEAGE_SHA256
          and strict.get("preflight_sha256") == PREFLIGHT_SHA256
          and strict.get("certificate_source_sha256") == CERTIFICATE_SOURCE_SHA256
          and strict.get("anatomical_admission") is False
          and strict.get("typed_pair") == [[51010, 323], [51010, 348]],
          "strict source-pair audit does not match this candidate and typed rows")
    result = strict.get("strict_result")
    _need(isinstance(result, dict) and result.get("interiors_disjoint") is True
          and result.get("cross_domain_contact_pairs") == 42374
          and result.get("biological_valve_interface") is False
          and result.get("mechanical_mass_assigned") is False,
          "strict source-pair report lacks its exact limited diagnostic result")
    return {
        "preflight_status": preflight["status"],
        "strict_pair_status": strict["status"],
        "typed_pair": strict["typed_pair"],
        "retained_exact_interface_face_count": 1617,
        "cross_domain_contact_pairs": 42374,
        "interiors_disjoint": True,
        "certificate_source_sha256": CERTIFICATE_SOURCE_SHA256,
        "anatomical_admission": False,
        "biological_valve_interface": False,
        "mechanical_mass_assigned": False,
    }


def compile_pinned_candidate(base_nha_path: Path, candidate_npz_path: Path,
                             preflight_path: Path, lineage_path: Path,
                             strict_audit_path: Path) -> tuple[bytes, dict[str, Any]]:
    paths = {
        "base_nha": (Path(base_nha_path), BASE_NHA_SHA256),
        "candidate_npz": (Path(candidate_npz_path), CANDIDATE_NPZ_SHA256),
        "preflight": (Path(preflight_path), PREFLIGHT_SHA256),
        "lineage": (Path(lineage_path), LINEAGE_SHA256),
        "strict_audit": (Path(strict_audit_path), STRICT_AUDIT_SHA256),
    }
    for label, (path, expected) in paths.items():
        _need(path.is_file() and sha256_file(path) == expected,
              f"pinned {label} bytes differ: {path}")
    base_raw = paths["base_nha"][0].read_bytes()
    preflight = json.loads(paths["preflight"][0].read_bytes())
    strict = json.loads(paths["strict_audit"][0].read_bytes())
    lineage = json.loads(gzip.decompress(paths["lineage"][0].read_bytes()))
    gate_evidence = _verify_gate_reports(preflight, strict)

    header, _records, base_rows = _parse_payload(base_raw)
    _need(header[0] == b"NHANAT1\0" and int(header[1]) == 5 and len(base_rows) == 525,
          "parent is not the pinned 525-row NHANAT1 ABI5 payload")
    with np.load(paths["candidate_npz"][0], allow_pickle=False) as arrays:
        replacements = build_row_replacements(base_rows, arrays)
        face_counts = {sid: len(replacements[sid]["faces"]) for sid in _ROWS}
        ancestry = _verify_lineage(lineage, face_counts)
        payload, owner = compile_payload(base_raw, replacements)
    out_header, _out_records, out_rows = _parse_payload(payload)
    _need(tuple(out_header[5:]) == tuple(header[5:])
          and list(out_rows) == list(base_rows)
          and owner["unmodified_rows_byte_exact"] == 523,
          "compiled payload changed parent row inventory or source header")
    rows = {}
    for stable_id in _ROWS:
        row = out_rows[stable_id]
        rows[str(stable_id)] = {
            "body_index": int(row["body_index"]), "layer": int(row["layer"]),
            "flags": int(row["flags"]), "vertex_count": len(row["vertices6"]),
            "face_count": len(row["faces"]),
            "vertices6_sha256": _sha256(np.ascontiguousarray(row["vertices6"], dtype="<f4").tobytes()),
            "faces_i64_sha256": _sha256(np.ascontiguousarray(row["faces"], dtype="<i8").tobytes()),
        }
    report = {
        "schema": "numi.human.pons-cerebellum-abi5-source-compile.v1",
        "status": "PASS_ABI5_SOURCE_ROWS_COMPILED",
        "base_nha_sha256": BASE_NHA_SHA256,
        "candidate_npz_sha256": CANDIDATE_NPZ_SHA256,
        "preflight_sha256": PREFLIGHT_SHA256,
        "lineage_sha256": LINEAGE_SHA256,
        "strict_pair_audit_sha256": STRICT_AUDIT_SHA256,
        "source_ancestry": ancestry,
        "strict_source_pair_diagnostic": gate_evidence,
        "abi5_owner": owner,
        "parent_row_count": len(base_rows),
        "unmodified_parent_rows_byte_exact": 523,
        "compiled_rows": rows,
        "compiled_payload_sha256": _sha256(payload),
        "compiled_payload_bytes": len(payload),
        "native_combined_scene_admission": False,
        "anatomical_admission": False,
        "motion_or_duration_qualification": False,
        "biological_valve_interface": False,
        "mechanical_mass_assigned": False,
        "scope": "Existing NHA ABI5 row-replacement owner; stable IDs 323 and 348 only. Source ancestry and strict F32 diagnostic are linked; neither establishes whole-anatomy admission or native scene behavior.",
    }
    return payload, report

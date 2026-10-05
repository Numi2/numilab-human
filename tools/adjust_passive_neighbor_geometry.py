#!/usr/bin/env python3
"""Compose the pinned passive-neighbor source rows into cardiac-wall-binding-005.

This overlays vertex+normal bytes only for stable IDs 4, 454, and 461. The
baseline, target, and candidate must agree exactly on those source vertex and
local-face arrays before authoring, and the candidate lineage/receipts are
pinned. No geometry is regenerated here and no native or GPU work is run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

HEADER = struct.Struct("<8s5I32s")
RECORD = struct.Struct("<8I")
FLOAT6 = struct.Struct("<6f")
STABLE_IDS = (4, 454, 461)
EXPECTED_OWNER = {
    4: ("FJ3147", "FMA7204"),
    454: ("FJ2566", "FMA7201"),
    461: ("FJ2817", "FMA7202"),
}
EVIDENCE_ROOT = Path("/Users/n/numi-human-resting-evidence-20261005")
BASE_DIR = EVIDENCE_ROOT / "thorax-conforming-passive-inputs-002"
CANDIDATE_DIR = EVIDENCE_ROOT / "passive-neighbor-candidate-004/output"
TARGET_DIR = EVIDENCE_ROOT / "cardiac-wall-binding-005"
OUTPUT_DIR = EVIDENCE_ROOT / "passive-neighbor-composed-native-inputs-001"
BASE_PAYLOAD = BASE_DIR / "resting-thorax.nhanatomy"
BASE_RECEIPT = BASE_DIR / "resting-anatomy-receipt.json"
CANDIDATE_PAYLOAD = CANDIDATE_DIR / "resting-thorax.nhanatomy"
CANDIDATE_RECEIPT = CANDIDATE_DIR / "resting-anatomy-receipt.json"
CANDIDATE_AUTHORING = CANDIDATE_DIR / "authoring.json"
CANDIDATE_PAIR_AUDIT = CANDIDATE_DIR / "neighbor-pair-audit.json"
TARGET_PAYLOAD = TARGET_DIR / "resting-thorax.nhanatomy"
TARGET_RECEIPT = TARGET_DIR / "resting-anatomy-receipt.json"
PREDICATE_PATH = Path("/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py")
CANDIDATE_PREDICATE_PATH = Path("/Users/n/numilab-human/src/numilab_human/cardiac_cavity_intersections.py")

PINNED_SHA256 = {
    "base_payload": "19a523a658519860416a53b1417615e43fff408b9f021126ffec40a5cf4db9bc",
    "base_receipt": "5b51441dc248e2980a2d2a1fca75436674b0f4156ee42c51b04d40110f7d994c",
    "candidate_payload": "6e8efce205aa564845dbf469afef0857565c205a0f76e54fe6cee888bea0d354",
    "candidate_receipt": "50f391aa1c352b98c3dce87a55ee9f5f3d201ee784edcb6d9d6ebea083ff262e",
    "candidate_authoring": "62be630888f780e3050996a2fa73c77ca7c07878a7be7fdeb2179ac02cc8ebee",
    "candidate_pair_audit": "9e14395b04bbf54f3a6870798eaceec9bd9aed4e02fc7c8436ffa1e59d522ae3",
    "target_payload": "c49590ba89257738d806a4748347990c458ef5e381a24835839410597fd15e7a",
    "target_receipt": "21f5f53b20115e627e577b84848e2f2ba4f9034fe2a6c6f7a95adcb32bcb204e",
    "predicate": "8b138882161c78312cb9b5aa46f49795092eb9b67ed9f705dc354787cb0a635b",
}


class CompositionError(ValueError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CompositionError(message)


def parse_payload(raw: bytes, label: str) -> dict[str, Any]:
    if len(raw) < HEADER.size:
        raise CompositionError(f"{label}: truncated NHANAT payload")
    magic, abi, count, vertex_count, index_count, registration_fp, source_sha = HEADER.unpack_from(raw)
    _need(magic == b"NHANAT1\0" and abi == 5, f"{label}: expected NHANAT1 ABI5")
    vertex_offset = HEADER.size + count * RECORD.size
    index_offset = vertex_offset + vertex_count * 24
    _need(len(raw) == index_offset + index_count * 4, f"{label}: exact payload byte length mismatch")
    rows: dict[int, dict[str, Any]] = {}
    vertex_ranges: list[tuple[int, int, int]] = []
    index_ranges: list[tuple[int, int, int]] = []
    for ordinal in range(count):
        fields = RECORD.unpack_from(raw, HEADER.size + ordinal * RECORD.size)
        body, vertex_start, vertex_n, index_start, index_n, stable_id, layer, flags = fields
        _need(stable_id not in rows, f"{label}: duplicate stable ID {stable_id}")
        _need(vertex_start + vertex_n <= vertex_count, f"{label}: invalid vertex range at stable ID {stable_id}")
        _need(index_start + index_n <= index_count and index_n % 3 == 0,
              f"{label}: invalid triangle-index range at stable ID {stable_id}")
        vertex_ranges.append((vertex_start, vertex_start + vertex_n, stable_id))
        index_ranges.append((index_start, index_start + index_n, stable_id))
        rows[stable_id] = {
            "fields": fields, "body": body, "vertex_start": vertex_start,
            "vertex_count": vertex_n, "index_start": index_start,
            "index_count": index_n, "layer": layer, "flags": flags,
        }
    for ranges, name in ((vertex_ranges, "vertex"), (index_ranges, "index")):
        ordered = sorted(ranges)
        for left, right in zip(ordered, ordered[1:]):
            _need(left[1] <= right[0], f"{label}: overlapping {name} ranges for {left[2]} and {right[2]}")
    return {
        "raw": raw, "header": (magic, abi, count, vertex_count, index_count, registration_fp, source_sha),
        "vertex_offset": vertex_offset, "index_offset": index_offset, "rows": rows,
    }


def vertex_bytes(view: dict[str, Any], stable_id: int) -> bytes:
    row = view["rows"][stable_id]
    start = view["vertex_offset"] + row["vertex_start"] * 24
    return view["raw"][start:start + row["vertex_count"] * 24]


def local_faces(view: dict[str, Any], stable_id: int) -> tuple[int, ...]:
    row = view["rows"][stable_id]
    count = row["index_count"]
    start = view["index_offset"] + row["index_start"] * 4
    global_indices = struct.unpack_from("<" + "I" * count, view["raw"], start)
    local = tuple(i - row["vertex_start"] for i in global_indices)
    _need(all(0 <= i < row["vertex_count"] for i in local),
          f"stable ID {stable_id}: index references outside its local vertex range")
    return local


def _source_identity(receipt: dict[str, Any], stable_id: int) -> tuple[str, str, str]:
    source = receipt.get("provenance", {}).get("source_id_map", {}).get(str(stable_id))
    _need(isinstance(source, dict), f"receipt lacks source identity for stable ID {stable_id}")
    metadata = source.get("source_owner_metadata") or {}
    return (str(source.get("source_member", "")),
            str(metadata.get("concept_id", "")),
            str(source.get("source_sha256", "")))


def _validate_receipt_binding(path: Path, receipt: dict[str, Any], expected_receipt_sha: str,
                              expected_payload_sha: str, label: str) -> None:
    _need(path.is_file(), f"{label}: missing receipt {path}")
    _need(sha_file(path) == expected_receipt_sha, f"{label}: receipt SHA-256 mismatch")
    _need(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1",
          f"{label}: unexpected receipt schema")
    payload = receipt.get("payload", {})
    _need(payload.get("sha256") == expected_payload_sha, f"{label}: receipt is not bound to the pinned payload")


def assert_exact_path(path: Path, expected_path: Path, label: str) -> None:
    _need(path.resolve() == expected_path.resolve(), f"{label}: path does not match pinned input identity")


def validate_pinned_inputs(paths: dict[str, Path], predicate_path: Path = PREDICATE_PATH) -> dict[str, str]:
    expected_paths = {
        "base_payload": BASE_PAYLOAD, "base_receipt": BASE_RECEIPT,
        "candidate_payload": CANDIDATE_PAYLOAD, "candidate_receipt": CANDIDATE_RECEIPT,
        "candidate_authoring": CANDIDATE_AUTHORING, "candidate_pair_audit": CANDIDATE_PAIR_AUDIT,
        "target_payload": TARGET_PAYLOAD, "target_receipt": TARGET_RECEIPT,
    }
    observed: dict[str, str] = {}
    for key, expected_path in expected_paths.items():
        path = paths[key].resolve()
        assert_exact_path(path, expected_path, key)
        _need(path.is_file(), f"{key}: missing pinned input {path}")
        observed[key] = sha_file(path)
        _need(observed[key] == PINNED_SHA256[key], f"{key}: SHA-256 mismatch")
    assert_exact_path(predicate_path, PREDICATE_PATH, "exact-predicate module")
    _need(predicate_path.is_file(), "pinned exact-predicate module is missing")
    observed["predicate"] = sha_file(predicate_path)
    _need(observed["predicate"] == PINNED_SHA256["predicate"],
          "exact-predicate module SHA-256 mismatch")
    return observed


def compose_payload(baseline_raw: bytes, candidate_raw: bytes, target_raw: bytes) -> tuple[bytes, dict[str, Any]]:
    baseline = parse_payload(baseline_raw, "passive baseline")
    candidate = parse_payload(candidate_raw, "neighbor candidate")
    target = parse_payload(target_raw, "cardiac target")
    for label, view in (("candidate", candidate), ("target", target)):
        h = view["header"]
        b = baseline["header"]
        _need(h[0:2] == b[0:2] and h[5:] == b[5:],
              f"{label}: source or registration identity differs from passive baseline")
    _need(set(baseline["rows"]) == set(candidate["rows"]) == set(target["rows"]),
          "baseline, candidate, and target stable-ID sets differ")
    _need(baseline["header"][2] == candidate["header"][2] == target["header"][2],
          "surface counts differ")
    report: dict[str, Any] = {"changed_stable_ids": list(STABLE_IDS), "surfaces": {}}
    target_view = target
    out = bytearray(target_raw)
    for stable_id in STABLE_IDS:
        b_row = baseline["rows"][stable_id]
        c_row = candidate["rows"][stable_id]
        t_row = target["rows"][stable_id]
        _need(b_row["body"] == c_row["body"] == t_row["body"] == 20,
              f"stable ID {stable_id}: expected preserved passive body-20 association")
        for row, label in ((c_row, "candidate"), (t_row, "target")):
            _need((row["vertex_count"], row["index_count"], row["layer"], row["flags"]) ==
                  (b_row["vertex_count"], b_row["index_count"], b_row["layer"], b_row["flags"]),
                  f"stable ID {stable_id}: {label} metadata/topology dimensions differ from baseline")
        baseline_vertices = vertex_bytes(baseline, stable_id)
        target_vertices = vertex_bytes(target, stable_id)
        candidate_vertices = vertex_bytes(candidate, stable_id)
        _need(target_vertices == baseline_vertices,
              f"stable ID {stable_id}: target original vertices/normals do not match passive baseline")
        baseline_faces = local_faces(baseline, stable_id)
        _need(local_faces(target, stable_id) == baseline_faces,
              f"stable ID {stable_id}: target local faces do not match passive baseline")
        _need(local_faces(candidate, stable_id) == baseline_faces,
              f"stable ID {stable_id}: candidate local faces do not match passive baseline")
        _need(candidate_vertices != baseline_vertices,
              f"stable ID {stable_id}: candidate contains no authored vertex/normal change")
        _need(len(candidate_vertices) == len(target_vertices),
              f"stable ID {stable_id}: vertex row byte length mismatch")
        for f in struct.iter_unpack("<f", candidate_vertices):
            _need(math.isfinite(f[0]), f"stable ID {stable_id}: candidate has non-finite geometry")
        # Exact-position groups must remain coincident after source-space edits.
        baseline_position_groups: dict[bytes, list[int]] = {}
        for i in range(b_row["vertex_count"]):
            pos = baseline_vertices[i * 24:i * 24 + 12]
            baseline_position_groups.setdefault(pos, []).append(i)
        for group in baseline_position_groups.values():
            if len(group) > 1:
                first = candidate_vertices[group[0] * 24:group[0] * 24 + 12]
                _need(all(candidate_vertices[i * 24:i * 24 + 12] == first for i in group[1:]),
                      f"stable ID {stable_id}: candidate split a baseline exact-position group")
        dest = out
        start = target["vertex_offset"] + t_row["vertex_start"] * 24
        dest[start:start + t_row["vertex_count"] * 24] = candidate_vertices
        report["surfaces"][str(stable_id)] = {
            "body_index": t_row["body"], "vertex_count": t_row["vertex_count"],
            "triangle_count": t_row["index_count"] // 3,
            "baseline_vertex_normal_sha256": sha_bytes(baseline_vertices),
            "target_before_vertex_normal_sha256": sha_bytes(target_vertices),
            "candidate_vertex_normal_sha256": sha_bytes(candidate_vertices),
            "local_face_index_sha256": sha_bytes(struct.pack("<" + "I" * len(baseline_faces), *baseline_faces)),
            "baseline_target_byte_identical_before_composition": True,
            "candidate_local_faces_byte_identical_to_baseline": True,
        }
    result = bytes(out)
    result_view = parse_payload(result, "composed output")
    _need(result_view["header"] == target_view["header"], "composition changed target NHA header")
    _need(result[HEADER.size:target["vertex_offset"]] == target_raw[HEADER.size:target["vertex_offset"]],
          "composition changed target record bytes")
    _need(result[target["index_offset"]:] == target_raw[target["index_offset"]:],
          "composition changed target index bytes")
    for stable_id in target["rows"]:
        if stable_id not in STABLE_IDS:
            _need(vertex_bytes(result_view, stable_id) == vertex_bytes(target, stable_id),
                  f"composition changed non-target stable ID {stable_id}")
    _need(len(result) == len(target_raw), "composition changed target payload length")
    report["preservation"] = {
        "target_header_and_records_unchanged": True,
        "target_indices_unchanged": True,
        "all_non_target_vertex_normal_rows_unchanged": True,
        "payload_length_unchanged": True,
        "modified_payload_sections": "vertex+normal bytes only for IDs 4, 454, and 461",
    }
    return result, report


def transition_bounds(raw: bytes, receipt: dict[str, Any]) -> dict[str, Any]:
    view = parse_payload(raw, "composed target")
    passive = receipt["functional_bindings"]["passive_viscera_geometry_binding"]
    upper_ids = [int(v) for v in passive["upper_anchor_stable_ids"]]
    pelvic_ids = [int(v) for v in passive["pelvic_anchor_stable_ids"]]
    def min_y(ids: list[int]) -> tuple[float, int]:
        candidates = []
        for stable_id in ids:
            row = view["rows"].get(stable_id)
            _need(row is not None, f"passive attachment anchor stable ID {stable_id} missing")
            raw_vertices = vertex_bytes(view, stable_id)
            for vertex in struct.iter_unpack("<6f", raw_vertices):
                _need(all(math.isfinite(x) for x in vertex), f"stable ID {stable_id}: non-finite anchor geometry")
                candidates.append((vertex[1], stable_id))
        return min(candidates)
    def max_y(ids: list[int]) -> tuple[float, int]:
        candidates = []
        for stable_id in ids:
            row = view["rows"].get(stable_id)
            _need(row is not None, f"passive attachment anchor stable ID {stable_id} missing")
            for vertex in struct.iter_unpack("<6f", vertex_bytes(view, stable_id)):
                _need(all(math.isfinite(x) for x in vertex), f"stable ID {stable_id}: non-finite anchor geometry")
                candidates.append((vertex[1], stable_id))
        return max(candidates)
    superior_value, superior_id = max_y(pelvic_ids)
    inferior_value, inferior_id = min_y(upper_ids)
    _need(superior_value < inferior_value,
          "recomputed passive attachment transition anchors are not ordered")
    return {
        "transition_superior_coordinates_m": [float(superior_value), float(inferior_value)],
        "pelvic_maximum_y_stable_id": int(superior_id),
        "upper_anchor_minimum_y_stable_id": int(inferior_id),
        "pelvic_anchor_stable_ids": pelvic_ids,
        "upper_anchor_stable_ids": upper_ids,
        "coordinate_basis": "canonical torso-20 local source coordinates",
    }


def _validate_source_receipts(base: dict[str, Any], candidate: dict[str, Any], target: dict[str, Any]) -> None:
    expected_candidate = candidate.get("provenance", {})
    first_stage = expected_candidate.get("passive_neighbor_geometry_adjustment", {})
    second_stage = expected_candidate.get("passive_neighbor_geometry_followup_adjustment", {})
    _need(first_stage.get("input_payload_sha256") == PINNED_SHA256["base_payload"],
          "candidate adjustment ancestry does not begin at the pinned passive baseline")
    _need(set(map(int, first_stage.get("changed_stable_ids", []))) == set(STABLE_IDS),
          "candidate first-stage changed-source ID set differs from selected composition scope")
    _need(set(map(int, second_stage.get("changed_stable_ids", []))) == {4, 461},
          "candidate follow-up changed-source ID set differs from pinned lineage")
    _need(second_stage.get("output_payload_sha256") == PINNED_SHA256["candidate_payload"],
          "candidate follow-up output does not bind the pinned candidate payload")
    for stable_id, (member, concept) in EXPECTED_OWNER.items():
        identities = [_source_identity(receipt, stable_id) for receipt in (base, candidate, target)]
        _need(identities[0] == identities[1] == identities[2],
              f"stable ID {stable_id}: source identity differs among baseline/candidate/target")
        _need(identities[0][0] == member and identities[0][1] == concept,
              f"stable ID {stable_id}: unexpected BodyParts3D source identity")


def compose_receipt(target: dict[str, Any], output_payload_path: Path, output_payload_sha: str,
                    output_raw: bytes, source_proofs: dict[str, Any], driver_sha: str) -> dict[str, Any]:
    receipt = copy.deepcopy(target)
    view = parse_payload(output_raw, "receipt output")
    magic, abi, surface_count, vertex_count, index_count, registration_fp, _ = view["header"]
    receipt["payload"].update({
        "path": str(output_payload_path), "sha256": output_payload_sha,
        "abi": int(abi), "surface_count": int(surface_count),
        "vertex_count": int(vertex_count), "index_count": int(index_count),
        "registration_fingerprint32": f"{registration_fp:08x}",
    })
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_payload_sha
    transition = transition_bounds(output_raw, receipt)
    receipt["functional_bindings"]["passive_viscera_geometry_binding"].update(
        {k: v for k, v in transition.items() if k == "transition_superior_coordinates_m"})
    cardiac = receipt["provenance"]["cardiac_geometry_binding"]
    cardiac["output_anatomy_payload_sha256"] = output_payload_sha
    if "ventricular_wall_binding" in cardiac:
        cardiac["ventricular_wall_binding"]["output_anatomy_payload_sha256"] = output_payload_sha
    receipt["provenance"]["passive_neighbor_geometry_composition"] = {
        "schema": "numi.human.passive-neighbor-geometry-composition.v1",
        "method": "byte-preserving source-row overlay into the downstream ABI-5 NHA payload",
        "functional_role": "inferred passive reference geometry only; no independent forces, mass, or physiology",
        "parameter_status": "inferred_source_boundary_adjustment_not_measured_subject_geometry",
        "changed_stable_ids": list(STABLE_IDS),
        "selected_surface_source_identity": {
            str(sid): {"source_member": EXPECTED_OWNER[sid][0], "FMA_concept_id": EXPECTED_OWNER[sid][1],
                       "source_sha256": _source_identity(receipt, sid)[2]}
            for sid in STABLE_IDS
        },
        "source_artifacts": source_proofs,
        "passive_transition_recomputed_from_composed_source_vertices": transition,
        "driver_sha256": driver_sha,
        "limitations": [
            "The earlier exact pair counts and authoring audits bind the candidate ancestry; pair predicates were not rerun on this cardiac-wall descendant.",
            "No native viewer or complete respiratory-cycle validation has yet been run on this composition.",
            "The small source-boundary adjustments are inferred reference geometry, not measured subject anatomy or organ mechanics.",
            "The broader passive-organ interface audit remains incomplete.",
        ],
    }
    return receipt


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def run(output_dir: Path = OUTPUT_DIR, predicate_path: Path = PREDICATE_PATH) -> dict[str, Any]:
    paths = {
        "base_payload": BASE_PAYLOAD, "base_receipt": BASE_RECEIPT,
        "candidate_payload": CANDIDATE_PAYLOAD, "candidate_receipt": CANDIDATE_RECEIPT,
        "candidate_authoring": CANDIDATE_AUTHORING, "candidate_pair_audit": CANDIDATE_PAIR_AUDIT,
        "target_payload": TARGET_PAYLOAD, "target_receipt": TARGET_RECEIPT,
    }
    inputs = validate_pinned_inputs(paths, predicate_path)
    base_rc = _read_json(BASE_RECEIPT)
    candidate_rc = _read_json(CANDIDATE_RECEIPT)
    target_rc = _read_json(TARGET_RECEIPT)
    _validate_receipt_binding(BASE_RECEIPT, base_rc, PINNED_SHA256["base_receipt"], PINNED_SHA256["base_payload"], "baseline")
    _validate_receipt_binding(CANDIDATE_RECEIPT, candidate_rc, PINNED_SHA256["candidate_receipt"], PINNED_SHA256["candidate_payload"], "candidate")
    _validate_receipt_binding(TARGET_RECEIPT, target_rc, PINNED_SHA256["target_receipt"], PINNED_SHA256["target_payload"], "target")
    _validate_source_receipts(base_rc, candidate_rc, target_rc)
    authoring = _read_json(CANDIDATE_AUTHORING)
    pair_audit = _read_json(CANDIDATE_PAIR_AUDIT)
    _need(authoring.get("output_payload_sha256") == PINNED_SHA256["candidate_payload"],
          "candidate authoring receipt does not bind candidate payload")
    _need(pair_audit.get("candidate_payload_sha256") == PINNED_SHA256["candidate_payload"],
          "candidate exact-pair audit does not bind candidate payload")
    _need(Path(authoring.get("exact_predicate_path", "")).resolve() == CANDIDATE_PREDICATE_PATH.resolve(),
          "candidate authoring predicate path differs from its recorded historical owner")
    _need(authoring.get("exact_predicate_sha256") == PINNED_SHA256["predicate"],
          "candidate authoring exact-predicate module hash differs")
    _need(CANDIDATE_PREDICATE_PATH.is_file() and sha_file(CANDIDATE_PREDICATE_PATH) == PINNED_SHA256["predicate"],
          "historical candidate predicate source file is missing or changed")
    _need(Path(authoring.get("output_receipt_path", "")).resolve() == CANDIDATE_RECEIPT.resolve(),
          "candidate authoring receipt path differs from pinned receipt")
    _need(authoring.get("output_receipt_sha256") == PINNED_SHA256["candidate_receipt"],
          "candidate authoring does not bind pinned output receipt")

    base_raw, candidate_raw, target_raw = BASE_PAYLOAD.read_bytes(), CANDIDATE_PAYLOAD.read_bytes(), TARGET_PAYLOAD.read_bytes()
    output_raw, compose_report = compose_payload(base_raw, candidate_raw, target_raw)
    output_payload = output_dir / "resting-thorax.nhanatomy"
    output_receipt = output_dir / "resting-anatomy-receipt.json"
    output_payload_sha = sha_bytes(output_raw)
    source_proofs = {
        "passive_baseline": {"payload_path": str(BASE_PAYLOAD), "payload_sha256": inputs["base_payload"],
                              "receipt_path": str(BASE_RECEIPT), "receipt_sha256": inputs["base_receipt"]},
        "candidate": {"payload_path": str(CANDIDATE_PAYLOAD), "payload_sha256": inputs["candidate_payload"],
                      "receipt_path": str(CANDIDATE_RECEIPT), "receipt_sha256": inputs["candidate_receipt"],
                      "authoring_path": str(CANDIDATE_AUTHORING), "authoring_sha256": inputs["candidate_authoring"],
                      "pair_audit_path": str(CANDIDATE_PAIR_AUDIT), "pair_audit_sha256": inputs["candidate_pair_audit"]},
        "downstream_target": {"payload_path": str(TARGET_PAYLOAD), "payload_sha256": inputs["target_payload"],
                              "receipt_path": str(TARGET_RECEIPT), "receipt_sha256": inputs["target_receipt"]},
        "exact_predicate_module": {"path": str(predicate_path), "sha256": inputs["predicate"]},
    }
    driver_sha = sha_file(Path(__file__).resolve())
    receipt = compose_receipt(target_rc, output_payload, output_payload_sha, output_raw, source_proofs, driver_sha)
    receipt_raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _need(receipt["payload"]["sha256"] == output_payload_sha, "composed receipt payload hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_payload.write_bytes(output_raw)
    output_receipt.write_bytes(receipt_raw)
    composition_audit = {
        "schema": "numi.human.passive-neighbor-composition-audit.v1",
        "execution_scope": "SSH Mac mini CPU only; no native build or GPU simulation",
        "source_files_sha256": {**inputs, "driver": driver_sha},
        "output_payload": {"path": str(output_payload), "sha256": output_payload_sha,
                           "byte_count": len(output_raw)},
        "output_receipt": {"path": str(output_receipt), "sha256": sha_file(output_receipt)},
        "input_identity": source_proofs,
        "composition": compose_report,
        "recomputed_passive_transition": transition_bounds(output_raw, receipt),
        "candidate_prior_pair_audit": pair_audit,
        "qualification_limit": "Composition/preservation evidence only; inherited static candidate pair audits do not establish final-descendant or full-cycle clearance.",
    }
    audit_path = output_dir / "composition-audit.json"
    audit_path.write_text(json.dumps(composition_audit, indent=2, sort_keys=True) + "\n")
    invocation = {
        "schema": "numi.human.passive-neighbor-composition-invocation.v1",
        "argv": [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output_dir),
                 "--predicate-path", str(predicate_path)],
        "driver_path": str(Path(__file__).resolve()), "driver_sha256": driver_sha,
        "input_sha256": inputs, "output_payload_sha256": output_payload_sha,
        "output_receipt_sha256": sha_file(output_receipt),
        "composition_audit_sha256": sha_file(audit_path),
        "execution_scope": "SSH Mac mini CPU only; no native build or GPU simulation",
    }
    (output_dir / "invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")
    return composition_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--predicate-path", type=Path, default=PREDICATE_PATH)
    args = parser.parse_args(argv)
    try:
        result = run(args.output_dir, args.predicate_path)
    except (CompositionError, OSError, json.JSONDecodeError, struct.error) as exc:
        print(f"passive-neighbor composition rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_payload": result["output_payload"],
                      "output_receipt": result["output_receipt"],
                      "changed_stable_ids": result["composition"]["changed_stable_ids"],
                      "preservation": result["composition"]["preservation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

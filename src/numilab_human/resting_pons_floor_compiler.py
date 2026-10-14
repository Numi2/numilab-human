"""Pinned compiler for the inferred fourth-ventricle floor source candidate.

This module only prepares an ABI5 source payload from the exact reviewed
candidate and focused source-gate artifacts. It does not admit motion/native
geometry or broaden the focused gate's scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from .resting_pleura_proxy import _parse_payload

BASE_NHA_SHA256 = "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42"
CANDIDATE_NPZ_SHA256 = "7c9f5fb245169ce033caf6919c1397b6084acab802c7a745792e282dd024c989"
CANDIDATE_REPORT_SHA256 = "0c66ba44381b6dc9011b11b4801fd8d4f9ee9b51984345b7fc8a360696aefda3"
DUAL_LINEAGE_SHA256 = "65fd020482a41b6d85d2ed91bd4d1bf68cb436487af77e093ab0f0cd0fe56458"
FOCUSED_GATE_SHA256 = "66792e15b47c922d4628ed5e00cca07629cda7f82b385d194dbb8474930c9d37"
FOCUSED_RUNNER_SHA256 = "eb6fa710aae6e861076f1a947cb580d5b5b1cc0160f8fcfcb88e312333ab3089"

# Stable ID -> (candidate vertex array, candidate face array, source vertex count).
_ROW_ARRAYS = {
    345: ("pons_vertices6_f32", "pons_faces_i64"),
    323: ("fourth_ventricle_vertices6_f32", "fourth_ventricle_faces_i64"),
}
_SOURCE_ROW_COUNTS = {
    345: (5154, 10304),
    323: (3567, 6940),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_pinned_inputs(
    base_nha: Path,
    candidate_npz: Path,
    candidate_report: Path,
    dual_lineage: Path,
    focused_gate_report: Path,
) -> dict[str, Any]:
    """Verify immutable source and gate artifacts before source compilation."""
    expected = (
        (base_nha, BASE_NHA_SHA256, "base NHA"),
        (candidate_npz, CANDIDATE_NPZ_SHA256, "floor candidate NPZ"),
        (candidate_report, CANDIDATE_REPORT_SHA256, "candidate report"),
        (dual_lineage, DUAL_LINEAGE_SHA256, "dual-lineage report"),
        (focused_gate_report, FOCUSED_GATE_SHA256, "focused gate report"),
    )
    for path, digest, label in expected:
        _need(path.is_file(), f"{label} is missing: {path}")
        _need(sha256_file(path) == digest, f"{label} SHA-256 mismatch: {path}")

    candidate = json.loads(candidate_report.read_text())
    _need(candidate.get("status") == "candidate_emitted_not_admitted",
          "candidate report status changed")
    _need(candidate.get("pins", {}).get("nha") == BASE_NHA_SHA256,
          "candidate is not derived from the pinned base NHA")
    _need(candidate.get("candidate_npz", {}).get("sha256") == CANDIDATE_NPZ_SHA256,
          "candidate report does not bind the pinned NPZ")
    _need(candidate.get("dual_lineage", {}).get("sha256") == DUAL_LINEAGE_SHA256,
          "candidate report does not bind the pinned dual lineage")
    _need(candidate.get("anatomy", {}).get("owner_body_index") == 23,
          "candidate owner is not the declared rigid head body")
    _need(
        {tuple(row) for row in candidate.get("anatomy", {}).get("typed_rows", [])}
        == {(51010, 323), (51010, 345)},
        "candidate typed-row identities changed",
    )

    lineage = json.loads(dual_lineage.read_text())
    _need(lineage.get("cap_face_count") == 3314
          and lineage.get("source_surface_stable_id") == 345
          and len(lineage.get("cap_faces", [])) == 3314,
          "dual-lineage cap ancestry does not match the candidate report")
    _need(candidate.get("reciprocal_floor", {}).get("cap_face_count") == 3314,
          "candidate reciprocal-floor cap count changed")
    _need(candidate.get("pons_candidate", {}).get("source_vertices") == 5154
          and candidate.get("pons_candidate", {}).get("source_faces") == 10304
          and candidate.get("fourth_ventricle_candidate", {}).get("source_vertices") == 3567
          and candidate.get("fourth_ventricle_candidate", {}).get("source_faces") == 6940,
          "candidate source-row topology counts changed")

    # The focused report is intentionally large. Its full-byte SHA is the
    # deterministic proof pin; never copy/re-emit its per-pair witness ledger.
    return {
        "base_nha_sha256": BASE_NHA_SHA256,
        "candidate_npz_sha256": CANDIDATE_NPZ_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "dual_lineage_sha256": DUAL_LINEAGE_SHA256,
        "focused_gate_report_sha256": FOCUSED_GATE_SHA256,
        "focused_gate_runner_sha256": FOCUSED_RUNNER_SHA256,
    }


def build_row_replacements(
    base_rows: dict[int, dict[str, Any]],
    candidate_arrays: Any,
    candidate_report: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Construct exactly the two candidate rows, checking source prefixes/maps."""
    _need(set(_ROW_ARRAYS).issubset(base_rows),
          "base NHA lacks one of the two typed source rows")
    _need(candidate_report.get("anatomy", {}).get("owner_body_index") == 23,
          "candidate owner body changed")
    pons_meta = candidate_report["pons_candidate"]
    vent_meta = candidate_report["fourth_ventricle_candidate"]
    row_metadata = {345: pons_meta, 323: vent_meta}
    typed_rows = {tuple(row) for row in candidate_report["anatomy"]["typed_rows"]}
    _need(typed_rows == {(51010, 323), (51010, 345)},
          "candidate typed-row list differs from the compiler scope")

    replacements: dict[int, dict[str, Any]] = {}
    for stable_id, (vertex_key, face_key) in _ROW_ARRAYS.items():
        base = base_rows[stable_id]
        _need(int(base["body_index"]) == 23 and int(base["layer"]) == 1,
              f"base row {stable_id} has an unexpected owner or layer")
        vertices = np.ascontiguousarray(candidate_arrays[vertex_key], dtype="<f4")
        faces = np.ascontiguousarray(candidate_arrays[face_key], dtype="<i8")
        meta = row_metadata[stable_id]
        source_vertex_count = int(meta.get("source_vertices", -1))
        source_face_count = int(meta.get("source_faces", -1))
        _need((source_vertex_count, source_face_count) == _SOURCE_ROW_COUNTS[stable_id],
              f"candidate source topology counts changed for row {stable_id}")
        _need(vertices.ndim == 2 and vertices.shape[1] == 6
              and vertices.shape[0] == meta.get("candidate_vertices"),
              f"candidate vertex shape disagrees for row {stable_id}")
        _need(faces.ndim == 2 and faces.shape[1] == 3
              and faces.shape[0] == meta.get("candidate_faces"),
              f"candidate face shape disagrees for row {stable_id}")
        _need(np.isfinite(vertices).all(), f"candidate row {stable_id} has non-finite slots")
        _need(faces.size == 0 or (int(faces.min()) >= 0
                                  and int(faces.max()) < len(vertices)),
              f"candidate row {stable_id} has a nonlocal face index")
        _need(_sha256(vertices.tobytes()) == meta.get("vertices_sha256"),
              f"candidate vertex digest disagrees for row {stable_id}")
        _need(_sha256(faces.tobytes()) == meta.get("faces_sha256"),
              f"candidate face digest disagrees for row {stable_id}")
        _need(len(base["vertices6"]) == source_vertex_count
              and len(vertices) >= source_vertex_count,
              f"source vertex prefix length disagrees for row {stable_id}")
        _need(len(base["faces"]) == source_face_count,
              f"source face count disagrees for row {stable_id}")
        source_prefix = np.ascontiguousarray(base["vertices6"], dtype="<f4")
        _need(vertices[:source_vertex_count].tobytes() == source_prefix.tobytes(),
              f"candidate changed original source vertex slots for row {stable_id}")

        replacement = dict(base)
        replacement["vertices6"] = vertices
        replacement["faces"] = faces
        replacements[stable_id] = replacement

    # Cross-check dual face-lineage arrays before they can be compiled into rows.
    cap_count = int(candidate_report["reciprocal_floor"]["cap_face_count"])
    _need(cap_count == 3314, "candidate cap count changed")
    for key in (
        "pons_face_source_parent_i64",
        "pons_face_child_order_i64",
        "fourth_ventricle_face_source_parent_i64",
        "fourth_ventricle_face_child_order_i64",
        "fourth_ventricle_face_source_surface_stable_id_i64",
    ):
        _need(key in candidate_arrays, f"candidate lineage array is missing: {key}")
    _need(len(candidate_arrays["pons_face_source_parent_i64"])
          == len(replacements[345]["faces"]),
          "Pons face ancestry length changed")
    _need(len(candidate_arrays["pons_face_child_order_i64"])
          == len(replacements[345]["faces"]),
          "Pons face child-order length changed")
    _need(len(candidate_arrays["fourth_ventricle_face_source_parent_i64"])
          == len(replacements[323]["faces"]),
          "ventricle face ancestry length changed")
    _need(len(candidate_arrays["fourth_ventricle_face_child_order_i64"])
          == len(replacements[323]["faces"]),
          "ventricle face child-order length changed")
    _need(len(candidate_arrays["pons_new_vertex_parent_face_i64"]) == 2848
          and candidate_arrays["pons_new_vertex_parent_weights_f64"].shape == (2848, 3)
          and candidate_arrays["pons_new_vertex_source_exact_xyz_f64"].shape == (2848, 3),
          "Pons inserted-vertex source ancestry changed")
    _need(len(candidate_arrays["fourth_ventricle_new_vertex_parent_face_i64"]) == 2848
          and candidate_arrays["fourth_ventricle_new_vertex_parent_weights_f64"].shape == (2848, 3)
          and candidate_arrays["fourth_ventricle_new_vertex_source_exact_xyz_f64"].shape == (2848, 3),
          "ventricle inserted-vertex source ancestry changed")
    _need(len(candidate_arrays["fourth_ventricle_cap_face_source_pons_face_i64"])
          == cap_count,
          "reciprocal cap face ancestry length changed")
    for key in (
        "fourth_ventricle_cap_face_source_pons_child_order_i64",
        "fourth_ventricle_cap_face_source_pons_output_face_i64",
        "fourth_ventricle_cap_face_source_ventricle_face_parent_i64",
    ):
        _need(len(candidate_arrays[key]) == cap_count,
              f"reciprocal cap ancestry length changed: {key}")
    _need(candidate_arrays["fourth_ventricle_cap_face_pons_barycentric_parent_weights_f64"].shape
          == (cap_count, 3, 3)
          and candidate_arrays["fourth_ventricle_cap_face_pons_source_exact_xyz_f64"].shape
          == (cap_count, 3, 3),
          "reciprocal cap barycentric/source coordinates changed")
    _need(
        dict(zip(*np.unique(
            candidate_arrays["fourth_ventricle_face_source_surface_stable_id_i64"],
            return_counts=True,
        ))) == {323: 9362, 345: 3314},
        "ventricle candidate has an unexpected source-surface lineage",
    )
    return replacements


def compile_payload(
    base_raw: bytes,
    replacements: dict[int, dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Use the existing ABI5 replacement owner and prove unrelated rows exact."""
    _need(set(replacements) == {323, 345},
          "compiler must replace exactly rows 323 and 345")
    base_header, _base_records, base_rows = _parse_payload(base_raw)
    _need(base_header[0] == b"NHANAT1\0" and base_header[1] == 5,
          "expected existing NHANAT1 ABI5")
    _need(len(base_rows) == 525,
          "base source surface count changed")
    output, reemit = encode_payload_with_row_replacements(base_raw, replacements)
    out_header, _out_records, out_rows = _parse_payload(output)
    _need(len(out_rows) == len(base_rows) and list(out_rows) == list(base_rows),
          "ABI5 replacement changed stable-ID order")
    _need(out_header[5:] == base_header[5:],
          "ABI5 replacement changed registration/source header")
    for stable_id, before in base_rows.items():
        after = out_rows[stable_id]
        if stable_id in replacements:
            _need(after["vertices6"].tobytes()
                  == replacements[stable_id]["vertices6"].tobytes()
                  and after["faces"].tobytes()
                  == replacements[stable_id]["faces"].tobytes(),
                  f"compiled replacement row {stable_id} differs from candidate")
        else:
            _need(after["vertices6"].tobytes() == before["vertices6"].tobytes()
                  and after["faces"].tobytes() == before["faces"].tobytes()
                  and after["body_index"] == before["body_index"]
                  and after["layer"] == before["layer"]
                  and after["flags"] == before["flags"],
                  f"ABI5 replacement changed unrelated row {stable_id}")
    _need(reemit["replaced_stable_ids"] == [323, 345],
          "existing ABI5 owner reports a different replacement set")
    return output, {
        "reemit_proof": reemit,
        "source_surface_count": len(base_rows),
        "compiled_surface_count": len(out_rows),
        "replaced_stable_ids": [323, 345],
        "unmodified_rows_content_exact": len(base_rows) - 2,
        "registration_and_source_header_unchanged": True,
    }


def compile_pinned_candidate(
    base_nha: Path,
    candidate_npz: Path,
    candidate_report_path: Path,
    dual_lineage: Path,
    focused_gate_report: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Verify pinned evidence and return the source-only ABI5 candidate bytes."""
    pins = verify_pinned_inputs(
        base_nha, candidate_npz, candidate_report_path, dual_lineage, focused_gate_report
    )
    candidate_report = json.loads(candidate_report_path.read_text())
    with np.load(candidate_npz, allow_pickle=False) as arrays:
        base_raw = base_nha.read_bytes()
        _header, _records, base_rows = _parse_payload(base_raw)
        replacements = build_row_replacements(base_rows, arrays, candidate_report)
        output, proof = compile_payload(base_raw, replacements)
    proof["pins"] = pins
    proof["scope"] = (
        "source-only ABI5 compile of the inferred fourth-ventricle floor candidate; "
        "no motion, native, all-neighbor, physiology, or individual-measurement admission"
    )
    return output, proof

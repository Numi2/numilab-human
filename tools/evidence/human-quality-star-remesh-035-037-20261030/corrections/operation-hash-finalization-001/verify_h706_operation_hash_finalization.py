#!/usr/bin/env python3
"""Read-only audit of operation JSON hash finalization for H706 036/037."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FINALIZATION_KEYS = ("new_output_face_rows", "patches")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_without_finalization(doc: dict[str, Any]) -> tuple[bytes, list[str]]:
    base = dict(doc)
    removed = []
    for key in FINALIZATION_KEYS:
        if key in base:
            removed.append(key)
            del base[key]
    if removed != list(FINALIZATION_KEYS):
        raise ValueError(f"expected exactly the two producer-finalization fields, got {removed}")
    return (json.dumps(base, sort_keys=True, indent=2) + "\n").encode("utf-8"), removed


def json_by_patch(entries: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError("report patch_face_map must be a list")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        patch_id = entry.get("patch_id")
        if not isinstance(patch_id, str) or patch_id in result:
            raise ValueError(f"missing or duplicate patch ID in report map: {patch_id!r}")
        result[patch_id] = entry
    return result


def check_stage(label: str, report_path: Path, output_root: Path) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema") not in {
        "numi.quality-star-connected-union-proposal.v1",
        "numi.warning-local-connected-union-proposal.v1",
    }:
        raise ValueError(f"unexpected {label} report schema {report.get('schema')!r}")
    role_reports = report["outputs"]
    role_records = []
    total_records = 0
    for role in ("O", "I", "D"):
        out_info = role_reports[role]
        npz_path = Path(out_info["path"])
        if not npz_path.is_absolute():
            npz_path = output_root / npz_path
        if sha256_file(npz_path) != out_info["sha256"]:
            raise ValueError(f"{label}/{role} raw NPZ SHA does not match report")
        with np.load(npz_path, allow_pickle=False) as z:
            raw_faces = np.asarray(z["faces"], dtype=np.int64)
            point_map = np.asarray(z["point_output_to_input_rows"], dtype=np.int64)
            face_input_map = np.asarray(z["face_output_to_input_rows"], dtype=np.int64)
            retired_raw = np.asarray(z["retired_input_face_rows"], dtype=np.int64)
            point_parent = np.asarray(z["point_output_to_parent_rows"], dtype=np.int64) if "point_output_to_parent_rows" in z else None
            faces_out = raw_faces.copy()
        report_patch_map = json_by_patch(out_info["patch_face_map"])
        role_ops = [x for x in report["accepted_patches"] if x["role"] == role]
        if set(report_patch_map) != {x["patch_id"] for x in role_ops}:
            raise ValueError(f"{label}/{role} report operation/map patch ID sets differ")
        role_operation_records = []
        expected_retired = []
        for operation_summary in role_ops:
            op_path = Path(operation_summary["source_operation_path"])
            if not op_path.is_absolute():
                op_path = output_root / op_path
            op_raw = op_path.read_bytes()
            current_sha = sha256_bytes(op_raw)
            expected_pre_sha = operation_summary["source_operation_sha256"]
            op = json.loads(op_raw)
            if op.get("role") != role or op.get("patch_id") != operation_summary["patch_id"]:
                raise ValueError(f"{label}/{role} operation identity does not match report")
            reconstructed, removed = canonical_without_finalization(op)
            reconstructed_sha = sha256_bytes(reconstructed)
            if reconstructed_sha != expected_pre_sha:
                raise ValueError(f"{label}/{role}/{op_path.name}: reconstructed pre-finalization hash mismatch")
            if current_sha == expected_pre_sha:
                raise ValueError(f"{label}/{role}/{op_path.name}: report hash unexpectedly matches current finalized bytes")

            patch_id = op["patch_id"]
            report_map = report_patch_map[patch_id]
            out_rows = [int(x) for x in op["new_output_face_rows"]]
            if out_rows != [int(x) for x in report_map["output_face_rows"]]:
                raise ValueError(f"{label}/{role}/{patch_id}: output row list differs from report map")
            if len(op["patches"]) != 1:
                raise ValueError(f"{label}/{role}/{patch_id}: expected one nested patch record")
            nested = op["patches"][0]
            for key in ("patch_id", "new_output_face_rows"):
                if nested.get(key) != op.get(key):
                    raise ValueError(f"{label}/{role}/{patch_id}: nested {key} differs from top level")
            for key in ("retired_input_face_rows", "new_faces_input_point_rows", "source_lineage"):
                if nested.get(key) != op.get(key):
                    raise ValueError(f"{label}/{role}/{patch_id}: nested {key} differs from top level")

            tri_out = faces_out[np.asarray(out_rows, dtype=np.int64)]
            if np.any(tri_out < 0) or np.any(tri_out >= len(point_map)):
                raise ValueError(f"{label}/{role}/{patch_id}: triangle point row out of range")
            mapped_source_tris = point_map[tri_out]
            expected_source_tris = np.asarray(op["new_faces_input_point_rows"], dtype=np.int64)
            if not np.array_equal(mapped_source_tris, expected_source_tris):
                raise ValueError(f"{label}/{role}/{patch_id}: original point-map triangles differ")
            if label == "037":
                if point_parent is None or "new_faces_036_point_rows" not in op:
                    raise ValueError(f"{label}/{role}/{patch_id}: immediate-parent point map is missing")
                mapped_parent_tris = point_parent[tri_out]
                expected_parent_tris = np.asarray(op["new_faces_036_point_rows"], dtype=np.int64)
                if not np.array_equal(mapped_parent_tris, expected_parent_tris):
                    raise ValueError(f"{label}/{role}/{patch_id}: 036 parent point-map triangles differ")
                if sorted(map(int, op["retired_input_face_rows"])) != sorted(map(int, report_map["retired_original_face_rows"])):
                    raise ValueError(f"{label}/{role}/{patch_id}: retired original face rows differ from report map")
                if sorted(map(int, op["source_036_face_rows"])) != sorted(map(int, report_map["retired_current_face_rows"])):
                    raise ValueError(f"{label}/{role}/{patch_id}: retired immediate 036 face rows differ from report map")
            else:
                if sorted(map(int, op["retired_input_face_rows"])) != sorted(map(int, report_map["retired_current_face_rows"])):
                    raise ValueError(f"{label}/{role}/{patch_id}: retired 035 face rows differ from report map")
                if len(op.get("source_initial_face_rows", [])) != int(operation_summary.get("source_initial_face_rows", -1)):
                    raise ValueError(f"{label}/{role}/{patch_id}: initial source face row count differs from report")
            if np.any(face_input_map[np.asarray(out_rows, dtype=np.int64)] != -1):
                raise ValueError(f"{label}/{role}/{patch_id}: derived output face unexpectedly maps to a source face")
            if not np.array_equal(np.sort(retired_raw), np.sort(np.asarray(op["retired_input_face_rows"], dtype=np.int64))):
                # The raw retired array is the complete role-wide set; check this operation's subset below.
                if not set(map(int, op["retired_input_face_rows"])).issubset(set(map(int, retired_raw))):
                    raise ValueError(f"{label}/{role}/{patch_id}: retired face rows absent from raw proposal map")
            expected_retired.extend(map(int, op["retired_input_face_rows"]))
            role_operation_records.append({
                "patch_id": patch_id,
                "operation_path": str(op_path),
                "report_pinned_pre_finalization_sha256": expected_pre_sha,
                "current_finalized_sha256": current_sha,
                "reconstructed_pre_finalization_sha256": reconstructed_sha,
                "removed_finalization_fields": removed,
                "output_face_row_count": len(out_rows),
                "reported_retired_face_row_count": len(op["retired_input_face_rows"]),
                "status": "PASS_HASH_AND_ROW_MAPS",
            })
            total_records += 1
        if len(set(expected_retired)) != len(expected_retired):
            raise ValueError(f"{label}/{role}: retired rows overlap across operation records")
        if label == "036":
            if set(expected_retired) != set(map(int, retired_raw)):
                raise ValueError(f"{label}/{role}: raw retired-face map differs from operation union")
        else:
            reported_retired_count = int(out_info["retired_original_face_count"])
            if len(set(map(int, retired_raw))) != reported_retired_count:
                raise ValueError(f"{label}/{role}: full original retired-face map count differs from report")
            if not set(expected_retired).issubset(set(map(int, retired_raw))):
                raise ValueError(f"{label}/{role}: operation retired rows absent from full original retired-face map")
        role_records.append({
            "role": role,
            "raw_proposal_path": str(npz_path),
            "raw_proposal_sha256": out_info["sha256"],
            "operation_count": len(role_operation_records),
            "retired_face_union_count": len(retired_raw),
            "operations": role_operation_records,
        })
    return {
        "stage": label,
        "report_path": str(report_path),
        "report_sha256": sha256_bytes(report_bytes),
        "operation_count": total_records,
        "roles": role_records,
        "status": "PASS_REPORT_HASH_RECONSTRUCTION_AND_OPERATION_ROW_MAPS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="H706 task root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    stages = [
        ("036", root / "quality-star-connected-union-036-run002" / "report.json", root / "quality-star-connected-union-036-run002"),
        ("037", root / "quality-warning-connected-union-037" / "run-002" / "report.json", root / "quality-warning-connected-union-037" / "run-002"),
    ]
    results = [check_stage(label, report, stage_root) for label, report, stage_root in stages]
    evidence_script = root / "quality-warning-connected-union-037" / "remesh_warning_union_037_reportfix.py"
    lines = evidence_script.read_text().splitlines()
    producer_ranges = {
        "report_operation_hash_captured": {"path": str(evidence_script), "sha256": sha256_file(evidence_script), "lines": [262, 264]},
        "operation_finalization_rewrite": {"path": str(evidence_script), "sha256": sha256_file(evidence_script), "lines": [301, 302]},
    }
    receipt = {
        "schema": "h706-operation-hash-finalization-correction.v1",
        "status": "PASS_110_OF_110_PRE_FINALIZATION_REPORT_HASHES_RECONSTRUCTED",
        "scope": "Read-only verification. Current enriched operation JSON files and original reports are not modified. The report hash refers to canonical operation JSON bytes before adding only new_output_face_rows and patches.",
        "canonical_reconstruction": "UTF-8 JSON with sort_keys=true, indent=2, and one final newline after removing exactly new_output_face_rows and patches.",
        "source_files_modified": False,
        "operation_count": sum(x["operation_count"] for x in results),
        "stages": results,
        "producer_source_evidence": producer_ranges,
        "limitations": [
            "The reconstructed pre-finalization bytes are verified in memory against the report pins; existing operation files remain in their enriched form.",
            "This corrects operation-hash provenance only. It does not alter or qualify geometry, volume, mass, TetGen, or native execution.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "operation_count": receipt["operation_count"], "out": str(args.out), "sha256": sha256_file(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

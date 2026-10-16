#!/usr/bin/env python3
"""Prepare pinned per-row geometry overlays for the shared anatomy verifier.

The two files produced here are counterfactual construction inputs only. They
contain the current-972 full21 replay world coordinates and candidate faces for
rows 51005:132 and 51005:134; source ancestry, parent terms, and native
admission remain bound by the separately pinned emission report/payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_row(arrays: dict[str, np.ndarray], row: int, expected: dict) -> dict:
    prefix = f"row{row}_"
    points = arrays[prefix + "positions_world_f32"]
    faces = arrays[prefix + "faces_local_u32"]
    gids = arrays[prefix + "canonical_material_gids_u64"]
    if points.dtype != np.dtype("<f4") or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"row {row} world positions must be Float32 metres")
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"row {row} faces must be ordered UInt32 triangles")
    if gids.dtype != np.dtype("<u8") or gids.shape != (len(points),):
        raise ValueError(f"row {row} canonical material GIDs have an invalid shape or dtype")
    if (not np.isfinite(points).all() or not len(faces) or
            int(faces.max()) >= len(points) or len(np.unique(gids)) != len(gids)):
        raise ValueError(f"row {row} has invalid world geometry or non-unique per-row canonical GIDs")
    if len(points) != expected["candidate_vertex_count"] or len(faces) != expected["candidate_face_count"]:
        raise ValueError(f"row {row} geometry dimensions differ from the pinned lineage report")
    return {
        "typed_identity": [51005, row],
        "vertices": int(len(points)),
        "faces": int(len(faces)),
        "canonical_material_gids_unique_within_row": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite retained output: {args.output}")
    report_sha = sha256(args.report)
    if report_sha != args.report_sha256:
        raise ValueError("source-emission report SHA-256 mismatch")
    report = json.loads(args.report.read_text())
    if (report.get("schema") != "numi.human.forearm-current972-full21-source-emission.v1"
            or report.get("status") != "SOURCE_EMISSION_READY_FOR_SHARED_COLLECTOR"
            or report.get("anatomical_admission") is not False
            or report.get("native_admission") is not False):
        raise ValueError("input report is not the pinned source-only forearm emission")
    world = Path(report["world_rows_path"])
    world_sha = sha256(world)
    if world_sha != report.get("world_rows_sha256"):
        raise ValueError("emitted world-row NPZ SHA-256 mismatch")
    child = Path(report["composed_child_path"])
    parent_extension = Path(report["parent_bindings_extension_path"])
    child_sha = sha256(child)
    parent_sha = sha256(parent_extension)
    if child_sha != report.get("composed_child_sha256"):
        raise ValueError("composed child payload SHA-256 mismatch")
    if parent_sha != report.get("parent_bindings_extension_sha256"):
        raise ValueError("parent-bindings extension SHA-256 mismatch")

    output = args.output
    output.mkdir(parents=True)
    with np.load(world, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    row_reports = {}
    for row in (132, 134):
        expected = report["candidate_lineage_preflight"][str(row)]
        row_reports[str(row)] = validate_row(arrays, row, expected)
        target = output / f"row{row}-full21-world-construction-overlay.npz"
        np.savez(
            target,
            capture_positions_m=arrays[f"row{row}_positions_world_f32"],
            capture_faces=arrays[f"row{row}_faces_local_u32"],
            capture_global_vertex_ids=arrays[f"row{row}_canonical_material_gids_u64"],
        )
        row_reports[str(row)]["overlay_path"] = str(target)
        row_reports[str(row)]["overlay_sha256"] = sha256(target)

    result = {
        "schema": "numi.human.forearm-shared-collector-inputs.v1",
        "status": "READY_COUNTERFACTUAL_GEOMETRY_INPUTS_ONLY",
        "anatomical_admission": False,
        "native_admission": False,
        "emission_report": {"path": str(args.report), "sha256": report_sha},
        "emitted_world_rows": {"path": str(world), "sha256": world_sha},
        "composed_child": {"path": str(child), "sha256": child_sha},
        "parent_bindings_extension": {"path": str(parent_extension), "sha256": parent_sha},
        "rows": row_reports,
        "source_report_declared_shared_birth_count": report.get("shared_birth_count"),
        "index_source_rows": [86, 104, 105, 106, 153, 158],
        "scope": "Counterfactual geometry input only. The shared verifier authenticates the six index-only rows against captured Float32 coordinates, validates these full21 emitted-world surfaces as constructions, and tests all selected exact targets. Source ancestry remains bound by the pinned emission report and payload; this helper adds no membership rule or geometry predicate.",
    }
    out_report = output / "report.json"
    out_report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "output": str(output),
        "report_sha256": sha256(out_report), "declared_shared_birth_count": report.get("shared_birth_count"),
        "rows": {k: {"vertices": v["vertices"], "faces": v["faces"],
                     "overlay_sha256": v["overlay_sha256"]}
                 for k, v in row_reports.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Independently compare contact-split PLYs with their retained raw controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from numilab_human.healthy_total_body_ct_surface_audit import (
    _bad_vertex_links,
    _edge_incidence_histogram,
    _read_binary_ply_gzip,
    _sha256_file,
)
from numilab_human.physiology import canonical


SCHEMA = "numi.healthy-total-body-ct-contact-split-controlled-comparison.v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("contact-split control comparison: " + message)


def _mesh_geometry(vertices: Any, faces: Any, np: Any) -> tuple[float, float]:
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    area = float((0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)).sum())
    relative = triangles - vertices.mean(axis=0)
    volume = float(
        np.einsum(
            "ij,ij->i",
            relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]),
        ).sum(dtype=np.float64)
        / 6.0
    )
    return area, volume


def compare(
    controls: list[Path],
    candidates: list[Path],
    control_audit_path: Path,
    candidate_audit_path: Path,
    np: Any,
) -> dict[str, Any]:
    _require(
        len(controls) == len(candidates) > 0,
        "provide matching control/candidate receipts",
    )
    control_audit = json.loads(control_audit_path.read_text(encoding="utf-8"))
    candidate_audit = json.loads(candidate_audit_path.read_text(encoding="utf-8"))
    controls_by_scan = {row["scan_id"]: row for row in control_audit.get("runs", [])}
    candidates_by_scan = {
        row["scan_id"]: row for row in candidate_audit.get("runs", [])
    }
    _require(
        len(controls_by_scan) == len(control_audit.get("runs", [])),
        "control audit repeats a scan",
    )
    _require(
        len(candidates_by_scan) == len(candidate_audit.get("runs", [])),
        "candidate audit repeats a scan",
    )
    all_rows = []
    for control_path, candidate_path in zip(controls, candidates, strict=True):
        control = json.loads(control_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        scan_id = candidate["scan"]["scan_id"]
        _require(
            control["scan"]["scan_id"] == scan_id,
            "control and candidate scan IDs differ",
        )
        _require(
            control.get("topology_method", "raw_voxel_boundary") == "raw_voxel_boundary"
            and candidate["topology_method"]
            == "voxel_boundary_contact_fan_split_candidate",
            "control or candidate topology method is unexpected",
        )
        _require(
            control["source"] == candidate["source"]
            and control["scan"] == candidate["scan"],
            "control and candidate source lineage differs",
        )
        control_independent = controls_by_scan.get(scan_id)
        _require(
            control_independent is not None
            and control_independent["compiler_receipt_sha256"]
            == _sha256_file(control_path)
            and control_independent[
                "all_independent_source_and_mesh_geometry_checks_pass"
            ],
            "control is absent from the passing independent source audit",
        )
        control_meshes = {row["label_id"]: row for row in control["meshes"]}
        candidate_meshes = {row["label_id"]: row for row in candidate["meshes"]}
        _require(control_meshes.keys() == candidate_meshes.keys(), "label sets differ")
        independent_run = candidates_by_scan.get(scan_id)
        _require(
            independent_run is not None
            and independent_run["compiler_receipt_sha256"]
            == _sha256_file(candidate_path)
            and independent_run["all_independent_source_and_mesh_geometry_checks_pass"],
            "candidate is absent from the passing independent audit",
        )
        audit_meshes = {row["label_id"]: row for row in independent_run["meshes"]}
        control_audit_meshes = {
            row["label_id"]: row for row in control_independent["meshes"]
        }
        for label_id in sorted(control_meshes):
            old = control_meshes[label_id]
            new = candidate_meshes[label_id]
            old_path = control_path.parent / old["mesh_file"]
            new_path = candidate_path.parent / new["mesh_file"]
            old_vertices, old_faces, _ = _read_binary_ply_gzip(old_path, np)
            new_vertices, new_faces, _ = _read_binary_ply_gzip(new_path, np)
            old_mesh_audit = control_audit_meshes[label_id]
            _require(
                old_mesh_audit["mesh_sha256"] == _sha256_file(old_path)
                and old_mesh_audit["compiler_topology_measurements_reproduced"],
                "raw control PLY does not match its independent audit",
            )
            same_triangles = bool(
                np.array_equal(old_vertices[old_faces], new_vertices[new_faces])
            )
            old_area, old_volume = _mesh_geometry(old_vertices, old_faces, np)
            new_area, new_volume = _mesh_geometry(new_vertices, new_faces, np)
            edge_histogram = _edge_incidence_histogram(new_faces, np)
            bad_vertices = _bad_vertex_links(new_faces, len(new_vertices))
            closed = bool(set(edge_histogram) == {"2"} and bad_vertices == 0)
            audit_mesh = audit_meshes[label_id]
            expected_volume = float(audit_mesh["source_voxel_occupancy_volume_mm3"])
            control_expected_volume = float(
                old["mesh_metrics"]["source_voxel_occupancy_volume_candidate_mm3"]
            )
            control_closed = bool(
                set(_edge_incidence_histogram(old_faces, np)) == {"2"}
                and _bad_vertex_links(old_faces, len(old_vertices)) == 0
            )
            topology_split = new["mesh_metrics"].get("topology_split", {})
            alternative_resolution = topology_split.get(
                "alternative_contact_resolution"
            )
            _require(
                control_closed == old["mesh_metrics"]["closed_two_manifold"],
                "raw control topology differs from its compiler receipt",
            )
            row = {
                "scan_id": scan_id,
                "label_id": label_id,
                "label_name": new["source_label_name"],
                "control_ply_sha256": old["mesh_file_sha256"],
                "candidate_ply_sha256": new["mesh_file_sha256"],
                "control_closed_two_manifold": control_closed,
                "candidate_closed_two_manifold": closed,
                "candidate_edge_incidence_histogram": edge_histogram,
                "candidate_nonmanifold_vertex_count": bad_vertices,
                "cloned_vertex_count": new["mesh_metrics"]
                .get("topology_split", {})
                .get(
                    "cloned_vertex_count",
                    0,
                ),
                "source_voxel_owner_contact_edge_count": topology_split.get(
                    "source_voxel_owner_contact_edge_count", 0
                ),
                "alternative_contact_resolution_count": topology_split.get(
                    "alternative_contact_resolution_count", 0
                ),
                "alternative_contact_resolution": alternative_resolution,
                "triangle_count_unchanged": len(old_faces) == len(new_faces),
                "triangle_coordinate_sequence_identical": same_triangles,
                "surface_area_absolute_delta_mm2": abs(new_area - old_area),
                "signed_volume_absolute_delta_mm3": abs(new_volume - old_volume),
                "candidate_relative_occupancy_volume_error": abs(
                    new_volume - expected_volume
                )
                / expected_volume,
                "control_relative_occupancy_volume_error": abs(
                    old_volume - control_expected_volume
                )
                / control_expected_volume,
                "independent_ply_source_audit_passed": audit_mesh[
                    "compiler_topology_measurements_reproduced"
                ]
                and audit_mesh["relative_signed_volume_error"] <= 1e-9,
            }
            all_rows.append(row)

    control_closed_count = sum(row["control_closed_two_manifold"] for row in all_rows)
    candidate_closed_count = sum(
        row["candidate_closed_two_manifold"] for row in all_rows
    )
    raw_defect_count = len(all_rows) - control_closed_count
    defects_fixed = sum(
        not row["control_closed_two_manifold"] and row["candidate_closed_two_manifold"]
        for row in all_rows
    )
    unresolved = [
        {
            "scan_id": row["scan_id"],
            "label_id": row["label_id"],
            "label_name": row["label_name"],
            "edge_incidence": row["candidate_edge_incidence_histogram"],
            "nonmanifold_vertex_count": row["candidate_nonmanifold_vertex_count"],
        }
        for row in all_rows
        if not row["candidate_closed_two_manifold"]
    ]
    alternative_rows = [
        {
            "scan_id": row["scan_id"],
            "label_id": row["label_id"],
            "label_name": row["label_name"],
            "alternative_contact_resolution_count": row[
                "alternative_contact_resolution_count"
            ],
            "resolution": row["alternative_contact_resolution"],
        }
        for row in all_rows
        if row["alternative_contact_resolution_count"] > 0
    ]
    return {
        "schema": SCHEMA,
        "status": "controlled_geometry_comparison_complete",
        "source_scans": sorted(candidates_by_scan),
        "surface_unit_count": len(all_rows),
        "raw_closed_surface_count": control_closed_count,
        "candidate_closed_surface_count": candidate_closed_count,
        "raw_defective_surface_count": raw_defect_count,
        "defective_surfaces_closed_by_candidate": defects_fixed,
        "alternative_contact_resolution_surface_count": len(alternative_rows),
        "alternative_contact_resolution_count": sum(
            row["alternative_contact_resolution_count"] for row in alternative_rows
        ),
        "all_alternative_contact_resolutions_unique": all(
            row["resolution"]["unique_closed_two_manifold_solution_count"] == 1
            for row in alternative_rows
        ),
        "alternative_contact_resolution_rows": alternative_rows,
        "unresolved_candidate_surfaces": unresolved,
        "all_triangle_counts_unchanged": all(
            row["triangle_count_unchanged"] for row in all_rows
        ),
        "all_triangle_coordinate_sequences_identical": all(
            row["triangle_coordinate_sequence_identical"] for row in all_rows
        ),
        "all_surface_area_absolute_deltas_zero": all(
            row["surface_area_absolute_delta_mm2"] == 0.0 for row in all_rows
        ),
        "maximum_candidate_relative_occupancy_volume_error": max(
            row["candidate_relative_occupancy_volume_error"] for row in all_rows
        ),
        "all_independent_ply_source_audits_passed": all(
            row["independent_ply_source_audit_passed"] for row in all_rows
        ),
        "maximum_absolute_signed_volume_delta_mm3": max(
            row["signed_volume_absolute_delta_mm3"] for row in all_rows
        ),
        "candidate_receipt_sha256": {
            str(path): _sha256_file(path) for path in candidates
        },
        "control_receipt_sha256": {str(path): _sha256_file(path) for path in controls},
        "control_independent_audit_sha256": _sha256_file(control_audit_path),
        "candidate_independent_audit_sha256": _sha256_file(candidate_audit_path),
        "analysis_source_sha256": _sha256_file(Path(__file__).resolve()),
        "rows": all_rows,
        "boundary": (
            "Per-scan, topology-only comparison for external automatic CT segmentation masks. "
            "It checks exact triangle-coordinate sequences, surface area, voxel-occupancy "
            "volume, and closed two-manifold topology after vertex-index reconnection. It "
            "does not establish expert segmentation accuracy, Numi subject registration, "
            "physical tissue ownership, mechanics, physiology, or clinical anatomy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-receipt", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-receipt", type=Path, nargs="+", required=True)
    parser.add_argument("--control-independent-audit", type=Path, required=True)
    parser.add_argument("--candidate-independent-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from numilab_human.healthy_total_body_ct_source import _load_numpy

    result = compare(
        args.control_receipt,
        args.candidate_receipt,
        args.control_independent_audit,
        args.candidate_independent_audit,
        _load_numpy(),
    )
    _require(
        not args.output.exists() and not args.output.is_symlink(),
        "output already exists",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "surface_unit_count": result["surface_unit_count"],
                "raw_closed_surface_count": result["raw_closed_surface_count"],
                "candidate_closed_surface_count": result[
                    "candidate_closed_surface_count"
                ],
                "defective_surfaces_closed_by_candidate": result[
                    "defective_surfaces_closed_by_candidate"
                ],
                "alternative_contact_resolution_surface_count": result[
                    "alternative_contact_resolution_surface_count"
                ],
                "alternative_contact_resolution_count": result[
                    "alternative_contact_resolution_count"
                ],
                "unresolved_candidate_surfaces": result[
                    "unresolved_candidate_surfaces"
                ],
                "all_independent_ply_source_audits_passed": result[
                    "all_independent_ply_source_audits_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

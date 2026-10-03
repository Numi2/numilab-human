from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from numilab_human.healthy_total_body_ct_surface import (
    _write_binary_ply_gzip,
    build_voxel_boundary_mesh,
)
from numilab_human.healthy_total_body_ct_surface_audit import (
    SCHEMA,
    _audit_mesh,
    _bad_vertex_links,
    _edge_incidence_histogram,
    audit_run,
)
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical


def _source_volume_metrics(mask: np.ndarray, affine: list[list[float]]) -> dict:
    vertices, faces, metrics = build_voxel_boundary_mesh(mask, affine, np=np)
    volume = int(mask.sum()) * abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
    metrics.update(
        {
            "source_voxel_occupancy_volume_candidate_mm3": volume,
            "relative_signed_volume_error": abs(metrics["signed_volume_mm3"] - volume)
            / volume,
        }
    )
    return vertices, faces, metrics


def test_independent_vertex_link_audit_detects_point_and_edge_contacts() -> None:
    affine = np.eye(4).tolist()
    single = np.ones((1, 1, 1), dtype=np.bool_)
    _, closed_faces, _ = build_voxel_boundary_mesh(single, affine, np=np)
    assert _bad_vertex_links(closed_faces, 8) == 0
    assert _edge_incidence_histogram(closed_faces, np) == {"2": 18}

    point_touch = np.zeros((2, 2, 2), dtype=np.bool_)
    point_touch[0, 0, 0] = point_touch[1, 1, 1] = True
    _, point_faces, _ = build_voxel_boundary_mesh(point_touch, affine, np=np)
    assert set(_edge_incidence_histogram(point_faces, np)) == {"2"}
    assert _bad_vertex_links(point_faces, 15) == 1

    edge_touch = np.zeros((2, 2, 2), dtype=np.bool_)
    edge_touch[0, 0, 0] = edge_touch[1, 1, 0] = True
    edge_vertices, edge_faces, _ = build_voxel_boundary_mesh(edge_touch, affine, np=np)
    assert _edge_incidence_histogram(edge_faces, np)["4"] == 1
    assert _bad_vertex_links(edge_faces, len(edge_vertices)) > 0


def test_independent_mesh_audit_reproduces_geometry_and_keeps_topology_failure(
    tmp_path: Path,
) -> None:
    mask = np.zeros((2, 2, 2), dtype=np.bool_)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    affine = np.eye(4).tolist()
    vertices, faces, metrics = _source_volume_metrics(mask, affine)
    mesh_path = tmp_path / "point-contact.ply.gz"
    _write_binary_ply_gzip(mesh_path, vertices, faces, np)
    mesh = {
        "mesh_file_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "mesh_file_bytes": mesh_path.stat().st_size,
        "mesh_metrics": metrics,
    }
    geometry = {
        "voxel_center_bounds_ijk": {
            "minimum_inclusive": [0, 0, 0],
            "maximum_inclusive": [1, 1, 1],
        }
    }

    result = _audit_mesh(mesh_path, mesh, geometry, affine, 2, np)

    assert result["relative_signed_volume_error"] == pytest.approx(0.0)
    assert result["independent_nonmanifold_vertex_count"] == 1
    assert result["closed_two_manifold"] is False
    assert result["compiler_topology_measurements_reproduced"] is True


def test_audit_run_rechecks_source_archive_plan_manifest_and_ply(
    tmp_path: Path,
) -> None:
    mask = np.ones((1, 1, 1), dtype=np.bool_)
    affine = np.eye(4).tolist()
    vertices, faces, metrics = _source_volume_metrics(mask, affine)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    mesh_path = run_directory / "scan-001-label-001-heart-boundary.ply.gz"
    _write_binary_ply_gzip(mesh_path, vertices, faces, np)
    nifti_payload = b"independent-test-nifti-payload"
    nifti_member = tmp_path / "member.nii.gz"
    with gzip.GzipFile(filename=str(nifti_member), mode="wb", mtime=0) as stream:
        stream.write(nifti_payload)
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(nifti_member, "scan-001.nii.gz")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    nifti_sha = hashlib.sha256(nifti_payload).hexdigest()
    geometry = {
        "voxel_count": 1,
        "voxel_center_bounds_ijk": {
            "minimum_inclusive": [0, 0, 0],
            "maximum_inclusive": [0, 0, 0],
        },
    }
    intake = {
        "schema": "HumanPack.external-segmentation-source-ingest.v2",
        "source": {
            "source_id": "healthy_total_body_cts_v3",
            "archive_sha256": archive_sha,
        },
        "labels": [{"label_id": 1, "name": "Heart"}],
        "scans": [
            {
                "scan_id": "001",
                "nifti_member": "scan-001.nii.gz",
                "nifti_uncompressed_sha256": nifti_sha,
                "voxel_to_world_affine": affine,
                "label_voxel_counts": {"1": 1},
                "label_spatial_geometry_candidates": {"1": geometry},
            }
        ],
    }
    intake_path = tmp_path / "intake.json"
    intake_path.write_bytes(canonical(intake) + b"\n")
    intake_sha = hashlib.sha256(intake_path.read_bytes()).hexdigest()
    compiler_sources = {"surface.py": "a" * 64}
    runtime = {
        "python_version": "3.11.7",
        "numpy_version": np.__version__,
        "scipy_version": "1.17.1",
    }
    plan = {
        "schema": "numi.healthy-total-body-ct-surface-trial-plan.v2",
        "input_intake_receipt_sha256": intake_sha,
        "source_archive_sha256": archive_sha,
        "scan_id": "001",
        "label_ids": [1],
        "compiler_sources_sha256": compiler_sources,
        "runtime": runtime,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan) + b"\n")
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    mesh = {
        "label_id": 1,
        "source_label_name": "Heart",
        "source_semantic_id": "healthy_total_body_cts_v3:segmentation-label:1",
        "mesh_file": mesh_path.name,
        "mesh_file_sha256": mesh_hash,
        "mesh_file_bytes": mesh_path.stat().st_size,
        "mesh_metrics": metrics,
    }
    compiler_receipt = {
        "schema": "HumanPack.external-segmentation-voxel-surface-candidates.v2",
        "compiler": "numilab-human.healthy-total-body-ct-surface.3",
        "compiler_sources_sha256": compiler_sources,
        "runtime": runtime,
        "source": {
            "intake_receipt_sha256": intake_sha,
            "archive_sha256": archive_sha,
        },
        "scan": {
            "scan_id": "001",
            "nifti_member": "scan-001.nii.gz",
            "nifti_uncompressed_sha256": nifti_sha,
        },
        "trial_plan_sha256": plan_sha,
        "meshes": [mesh],
    }
    receipt_path = run_directory / "receipt.json"
    receipt_path.write_bytes(canonical(compiler_receipt) + b"\n")
    checksums = {
        "receipt.json": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        mesh_path.name: mesh_hash,
    }
    (run_directory / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="ascii",
    )

    report = audit_run(
        intake_path=intake_path,
        archive_path=archive_path,
        trial_plan_path=plan_path,
        compiler_receipt_path=receipt_path,
        np=np,
    )

    assert report["schema"] == SCHEMA
    assert report["all_independent_source_and_mesh_geometry_checks_pass"] is True
    assert report["closed_two_manifold_mesh_count"] == 1
    assert report["qualification"]["current_numi_subject_binding"] is False
    assert report["qualification"]["mechanics_or_physiology"] is False

    mesh_path.write_bytes(mesh_path.read_bytes() + b"tampered")
    with pytest.raises(HumanImportError, match="checksum mismatch"):
        audit_run(
            intake_path=intake_path,
            archive_path=archive_path,
            trial_plan_path=plan_path,
            compiler_receipt_path=receipt_path,
            np=np,
        )

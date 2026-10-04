from __future__ import annotations

import gzip
import hashlib
import struct
import zipfile

import numpy as np
import pytest

from numilab_human.healthy_total_body_ct_ras_x_partition import (
    _audit_source_counts,
    _mesh_source_voxel_geometry,
    _partition_mask_by_ras_x,
    _ras_x_plane,
    _verify_plan,
    _verify_reference_surface_manifest,
)
from numilab_human.healthy_total_body_ct_surface import build_voxel_boundary_mesh
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def test_partition_plan_binds_predictions_acceptance_and_runtime(tmp_path) -> None:
    hashes = {"compiler.py": "a" * 64}
    runtime = {"python_version": "3.11", "numpy_version": "2.4", "scipy_version": "1.17"}
    plan = {
        "schema": "numi.healthy-total-body-ct-ras-x-partition-plan.v1",
        "input_intake_receipt_sha256": "a" * 64,
        "source_archive_sha256": "b" * 64,
        "source_surface_receipt_sha256": "c" * 64,
        "source_surface_plan_sha256": "d" * 64,
        "scan_id": "002",
        "label_ids": [1, 12],
        "compiler_sources_sha256": hashes,
        "runtime": runtime,
        "partition": {
            "coordinate_system": "NIfTI RAS+",
            "axis": "x",
            "coordinate_mm": 0.0,
            "positive_side": "left",
            "negative_side": "right",
            "voxel_center_rule": "assign each occupied voxel by the sign of its RAS-X center",
            "voxel_face_aligned": True,
            "split_contact_topology": True,
        },
        "predictions": [
            "adrenal label 1 occupies both RAS-X sides in scan 002",
            "lung label 12 occupies both RAS-X sides in scan 002",
            "midline assignment loses or duplicates no occupied source voxel",
        ],
        "acceptance": [
            "source label voxels partition exactly once by RAS-X center sign",
            "left and right candidates are independently closed two-manifolds",
            "each side mesh volume matches its source voxel occupancy",
            "all disconnected source fragments remain represented",
        ],
        "boundary": "Automatic segmentation candidates only; no clinical anatomy, registration, Numi subject, mechanics, or physiology claim.",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan) + b"\n")

    _verify_plan(
        plan_path,
        intake_sha256="a" * 64,
        archive_sha256="b" * 64,
        source_surface_receipt_sha256="c" * 64,
        source_surface_plan_sha256="d" * 64,
        scan_id="002",
        label_ids=[1, 12],
        compiler_sources_sha256=hashes,
        runtime=runtime,
    )

    plan["predictions"].pop()
    plan_path.write_bytes(canonical(plan) + b"\n")
    with pytest.raises(ImportError, match="does not bind the exact source"):
        _verify_plan(
            plan_path,
            intake_sha256="a" * 64,
            archive_sha256="b" * 64,
            source_surface_receipt_sha256="c" * 64,
            source_surface_plan_sha256="d" * 64,
            scan_id="002",
            label_ids=[1, 12],
            compiler_sources_sha256=hashes,
            runtime=runtime,
        )


def _tiny_nifti_zip(path, values: np.ndarray, affine: list[list[float]]) -> str:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *values.shape, 1, 1, 1, 1)
    struct.pack_into("<hh", header, 70, 16, 32)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[123] = 2
    struct.pack_into("<h", header, 254, 1)
    for index, offset in enumerate((280, 296, 312)):
        struct.pack_into("<4f", header, offset, *affine[index])
    header[344:348] = b"n+1\x00"
    nii = bytes(header) + np.asarray(values, dtype="<f4").tobytes(order="F")
    compressed = gzip.compress(nii, mtime=0)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("tiny.nii.gz", compressed)
    return hashlib.sha256(nii).hexdigest()


def test_ras_x_partition_is_disjoint_complete_and_meshes_both_sides() -> None:
    affine = [
        [-1.0, 0.0, 0.0, 1.5],
        [0.0, 1.0, 0.0, -0.5],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    mask = np.ones((4, 2, 1), dtype=np.bool_)

    left, right, audit = _partition_mask_by_ras_x(mask, affine, (0, 0, 0), np)

    assert audit["axis_ijk"] == 0
    assert audit["plane_index_coordinate"] == pytest.approx(1.5)
    assert audit["left_voxel_count"] == audit["right_voxel_count"] == 4
    assert np.array_equal(left | right, mask)
    assert not np.any(left & right)
    assert np.all(left[:2]) and not np.any(left[2:])
    assert np.all(right[2:]) and not np.any(right[:2])

    left_vertices, _, left_metrics = build_voxel_boundary_mesh(
        left, affine, split_contact_topology=True, np=np
    )
    right_vertices, _, right_metrics = build_voxel_boundary_mesh(
        right, affine, split_contact_topology=True, np=np
    )
    assert left_metrics["closed_two_manifold"]
    assert right_metrics["closed_two_manifold"]
    assert left_metrics["signed_volume_mm3"] == pytest.approx(8.0)
    assert right_metrics["signed_volume_mm3"] == pytest.approx(8.0)
    assert left_vertices[:, 0].min() >= 0.0
    assert right_vertices[:, 0].max() <= 0.0


def test_ras_x_partition_rejects_a_plane_through_voxel_centers() -> None:
    affine = np.eye(4).tolist()
    affine[0][3] = 1.0

    with pytest.raises(ImportError, match="does not lie on a voxel face"):
        _ras_x_plane(affine, np)


def test_mesh_geometry_audit_checks_source_voxel_grid_and_envelope() -> None:
    affine = np.array([
        [-1.0, 0.0, 0.0, 1.5],
        [0.0, 1.0, 0.0, -0.5],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    voxel_indices = np.array([
        [-0.5, -0.5, -0.5], [3.5, -0.5, -0.5],
        [-0.5, 1.5, 0.5], [3.5, 1.5, 0.5],
    ])
    vertices = voxel_indices @ affine[:3, :3].T + affine[:3, 3]
    bounds = {"minimum_inclusive": [0, 0, 0], "maximum_inclusive": [3, 1, 0]}

    assert _mesh_source_voxel_geometry(
        vertices, affine=affine.tolist(), voxel_center_bounds_ijk=bounds, np=np
    ) == {
        "vertices_within_source_voxel_envelope": True,
        "vertices_on_source_voxel_grid": True,
    }
    assert _mesh_source_voxel_geometry(
        vertices + np.array([0.1, 0.0, 0.0]),
        affine=affine.tolist(), voxel_center_bounds_ijk=bounds, np=np
    )["vertices_on_source_voxel_grid"] is False
    outside = vertices.copy()
    outside[0] = np.array([-1.5, -0.5, -0.5]) @ affine[:3, :3].T + affine[:3, 3]
    assert _mesh_source_voxel_geometry(
        outside, affine=affine.tolist(), voxel_center_bounds_ijk=bounds, np=np
    )["vertices_within_source_voxel_envelope"] is False


def test_ras_x_partition_rejects_oblique_x_rows() -> None:
    affine = np.eye(4).tolist()
    affine[0][1] = 0.25
    affine[0][3] = 0.75

    with pytest.raises(ImportError, match="align with one voxel-grid axis"):
        _ras_x_plane(affine, np)


def test_independent_source_reader_recounts_both_sides(tmp_path) -> None:
    affine = [
        [-1.0, 0.0, 0.0, 1.5],
        [0.0, 1.0, 0.0, -0.5],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    values = np.full((4, 2, 1), 12, dtype=np.float32)
    archive_path = tmp_path / "tiny.zip"
    expected_hash = _tiny_nifti_zip(archive_path, values, affine)
    scan = {"voxel_to_world_affine": affine}

    counts, nifti_hash, info = _audit_source_counts(
        archive_path,
        member_name="tiny.nii.gz",
        scan=scan,
        expected_nifti_sha256=expected_hash,
        label_ids=[12],
        np=np,
    )

    assert nifti_hash == expected_hash
    assert counts[12] == {"left": 4, "right": 4, "on_plane": 0}
    assert info["voxel_volume_in_source_units_cubed"] == pytest.approx(2.0)


def test_reference_surface_manifest_binds_receipt_and_mesh_files(tmp_path) -> None:
    receipt_bytes = b'{"schema":"fixture"}\n'
    mesh_bytes = b"retained candidate mesh"
    (tmp_path / "receipt.json").write_bytes(receipt_bytes)
    (tmp_path / "candidate.ply.gz").write_bytes(mesh_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
    receipt = {
        "meshes": [{"mesh_file": "candidate.ply.gz", "mesh_file_sha256": mesh_sha}]
    }
    (tmp_path / "SHA256SUMS").write_text(
        f"{receipt_sha}  receipt.json\n{mesh_sha}  candidate.ply.gz\n",
        encoding="ascii",
    )

    _verify_reference_surface_manifest(
        tmp_path, receipt, expected_receipt_sha256=receipt_sha
    )

    (tmp_path / "candidate.ply.gz").write_bytes(b"changed")
    with pytest.raises(ImportError, match="payload checksum mismatch"):
        _verify_reference_surface_manifest(
            tmp_path, receipt, expected_receipt_sha256=receipt_sha
        )

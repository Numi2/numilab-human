from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from numilab_human import healthy_total_body_ct_vessel_profiles as profiles
from numilab_human.physiology import canonical
from numilab_human.model import ImportError


def _nifti(values: np.ndarray) -> bytes:
    nz, ny, nx = values.shape
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, -1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[123] = 2
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 0)
    struct.pack_into("<3f", header, 268, 10.0, 20.0, 30.0)
    header[344:348] = b"n+1\x00"
    return gzip.compress(bytes(header) + values.astype("<f4").tobytes(), mtime=0)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    values = np.zeros((3, 3, 4), dtype=np.float32)
    values[0, 1, 1] = 2
    values[2, :, 2] = 2
    values[1, 0, :2] = 11
    compressed_nifti = _nifti(values)
    member_name = "nifti/scan-001.nii.gz"
    workbook_name = "labels.xlsx"
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, compressed_nifti)
        archive.writestr(workbook_name, b"fake workbook replaced by pinned parser")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    source_config = {
        "schema": "numi.human.external-segmentation-source.v1",
        "source_id": "healthy_total_body_cts_v3",
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "nifti_member_directory": "nifti/",
        "label_dictionary_member": workbook_name,
        "expected_shape_ijk": [4, 3, 3],
    }
    config_path = tmp_path / "source-config.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    from numilab_human.healthy_total_body_ct_source import parse_nifti_header
    header = gzip.decompress(compressed_nifti)[:352]
    geometry = parse_nifti_header(header)
    intake = {
        "schema": profiles.source.SCHEMA,
        "status": "partial_source_inventory",
        "source": {
            "archive_sha256": archive_sha,
            "source_config_sha256": hashlib.sha256(canonical(source_config)).hexdigest(),
        },
        "scans": [{
            "scan_id": "001",
            "nifti_member": member_name,
            "nifti_uncompressed_sha256": hashlib.sha256(gzip.decompress(compressed_nifti)).hexdigest(),
            "label_voxel_counts": {"2": 4, "11": 2},
            "voxel_spacing_mm": geometry["voxel_spacing"],
            "voxel_to_world_affine": geometry["voxel_to_world_affine"],
            "voxel_volume_mm3": geometry["voxel_volume_in_source_units_cubed"],
            "label_scaling_slope": geometry["label_scaling_slope"],
            "label_scaling_intercept": geometry["label_scaling_intercept"],
        }],
    }
    intake_path = tmp_path / "intake.json"
    intake_path.write_text(json.dumps(intake), encoding="utf-8")
    monkeypatch.setattr(profiles.source, "read_label_dictionary", lambda _: [
        {"label_id": 2, "name": "Aorta"}, {"label_id": 11, "name": "VCI"},
    ])
    return archive_path, config_path, intake_path, values


def test_hash_bound_vessel_profiles_measure_exact_mask_planes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, config, intake, _ = _fixture(tmp_path, monkeypatch)

    result = profiles.compile_profiles(
        archive_path=archive, source_config=config, intake_receipt=intake, scan_ids=("001",),
    )

    assert result["qualification"]["selected_uncompressed_nifti_hashes_match"] is True
    assert result["qualification"]["medial_axis_centreline"] is False
    scan = result["scans"][0]
    assert scan["slice_plane_voxel_area_mm2"] == pytest.approx(6.0)
    assert scan["voxel_volume_mm3"] == pytest.approx(24.0)
    aorta, vci = scan["vessels"]
    assert aorta["source_voxel_count"] == 4
    assert aorta["k_range_inclusive"] == [0, 2]
    assert aorta["internal_empty_plane_count"] == 1
    assert [row["plane_occupancy_area_mm2"] for row in aorta["mask_centroid_profile"]] == [6.0, 18.0]
    assert aorta["mask_centroid_profile"][0]["mask_centroid_ras_mm"] == pytest.approx([12.0, 23.0, 30.0])
    assert aorta["source_mask_face_connectivity_6"]["component_voxel_counts_descending"] == [3, 1]
    assert vci["source_voxel_count"] == 2
    assert vci["source_mask_face_connectivity_6"]["component_count"] == 1
    assert vci["voxel_occupancy_volume_ml"] == pytest.approx(0.048)


def test_face_connectivity_uses_only_shared_faces_not_diagonal_touching() -> None:
    planes = np.zeros((2, 4, 4), dtype=bool)
    planes[0, 0, 0:2] = True
    planes[0, 3, 3] = True
    planes[1, 0, 1:3] = True
    planes[1, 2, 2] = True

    components = [profiles._plane_four_connected_components(plane, np) for plane in planes]
    parents = [0]
    sizes = [0]
    previous = np.zeros((4, 4), dtype=np.int32)
    for rows in components:
        current = np.zeros((4, 4), dtype=np.int32)
        for cells in rows:
            node = len(parents)
            parents.append(node)
            sizes.append(len(cells))
            current.reshape(-1)[cells] = node
        for index in np.flatnonzero((current > 0) & (previous > 0)).tolist():
            profiles._union(parents, sizes, int(current.reshape(-1)[index]),
                            int(previous.reshape(-1)[index]))
        previous = current

    result = profiles._connectivity_result(parents, sizes)
    assert result["component_voxel_counts_descending"] == [4, 1, 1]


def test_vessel_profile_rejects_intake_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, config, intake, _ = _fixture(tmp_path, monkeypatch)
    doc = json.loads(intake.read_text(encoding="utf-8"))
    doc["scans"][0]["nifti_uncompressed_sha256"] = "0" * 64
    intake.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ImportError, match="uncompressed NIfTI SHA-256"):
        profiles.compile_profiles(
            archive_path=archive, source_config=config, intake_receipt=intake, scan_ids=("001",),
        )


def test_vessel_profile_rejects_intake_affine_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, config, intake, _ = _fixture(tmp_path, monkeypatch)
    doc = json.loads(intake.read_text(encoding="utf-8"))
    doc["scans"][0]["voxel_to_world_affine"][0][3] += 1.0
    intake.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ImportError, match="voxel-to-world affine differs"):
        profiles.compile_profiles(
            archive_path=archive, source_config=config, intake_receipt=intake, scan_ids=("001",),
        )


def test_vessel_profile_cli_summary_does_not_admit_lumen_or_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "scans": [{"vessels": [{"source_voxel_count": 5}, {"source_voxel_count": 7}]}],
        "qualification": {
            "exact_slice_plane_occupancy_areas_and_mask_centroids": True,
            "single_face_connected_source_mask_per_label": True,
        },
    }
    monkeypatch.setattr(profiles, "compile_profiles", lambda **_: result)
    monkeypatch.setattr(profiles, "immutable_write", lambda *_: "a" * 64)
    args = SimpleNamespace(archive=tmp_path / "archive", source_config=tmp_path / "config",
                           intake_receipt=tmp_path / "intake", scan_ids=["001"],
                           output=tmp_path / "output.json")

    assert profiles.run(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source_voxel_count"] == 12
    assert summary["exact_slice_profiles"] is True
    assert summary["source_mask_single_component"] is True
    assert summary["lumen_or_branch_flow_admitted"] is False
    assert summary["pressure_flow_admitted"] is False

from __future__ import annotations

import pytest

from numilab_human.healthy_total_body_ct_cohort import summarize_intake
from numilab_human.model import ImportError


def spatial_candidate(voxel_count: int, affine: list[list[float]]) -> dict:
    centroid_ijk = [2.0, 3.0, 4.0]
    centroid_ras = [
        sum(affine[axis][j] * centroid_ijk[j] for j in range(3)) + affine[axis][3]
        for axis in range(3)
    ]
    corners = [
        [
            sum(affine[axis][j] * point[j] for j in range(3)) + affine[axis][3]
            for axis in range(3)
        ]
        for point in (
            [i, j, k]
            for i in (-0.5, 4.5)
            for j in (0.5, 5.5)
            for k in (1.5, 6.5)
        )
    ]
    return {
        "voxel_count": voxel_count,
        "centroid_voxel_center_ijk": centroid_ijk,
        "voxel_center_bounds_ijk": {
            "minimum_inclusive": [0, 1, 2],
            "maximum_inclusive": [4, 5, 6],
        },
        "centroid_ras_mm": centroid_ras,
        "voxel_envelope_aabb_ras_mm": {
            "minimum": [min(point[axis] for point in corners) for axis in range(3)],
            "maximum": [max(point[axis] for point in corners) for axis in range(3)],
        },
    }


def scan(scan_id: str, *, voxel_volume: float, counts: dict[str, int]) -> dict:
    scale = voxel_volume ** (1 / 3)
    affine = [
        [scale, 0.0, 0.0, 10.0],
        [0.0, scale, 0.0, 20.0],
        [0.0, 0.0, -scale, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    volumes = {
        key: count * voxel_volume / 1000.0
        for key, count in counts.items()
        if key != "0"
    }
    return {
        "scan_id": scan_id,
        "nifti_uncompressed_sha256": (scan_id * 64)[:64],
        "affine_source": "qform",
        "voxel_spacing_mm": [scale] * 3,
        "voxel_volume_mm3": voxel_volume,
        "voxel_to_world_affine": affine,
        "label_voxel_counts": counts,
        "label_raster_volume_candidate_ml": volumes,
        "label_spatial_geometry_candidates": {
            key: spatial_candidate(count, affine)
            for key, count in counts.items()
            if key != "0"
        },
        "observed_nonbackground_label_ids": sorted(int(key) for key in counts if key != "0"),
    }


def receipt() -> dict:
    return {
        "schema": "HumanPack.external-segmentation-source-ingest.v2",
        "source": {
            "source_id": "healthy_total_body_cts_v3",
            "release": "fixture",
            "doi": "10.0000/example",
            "license": "CC BY 4.0",
            "archive_sha256": "a" * 64,
            "label_dictionary_sha256": "b" * 64,
            "coordinate_convention": "NIfTI RAS+ in millimetres",
            "segmentation_generation": "automatic source masks",
            "voxel_array_shape_ijk": [10, 10, 10],
        },
        "counts": {
            "scan_count": 2,
            "data_dictionary_label_count": 3,
            "observed_label_count": 2,
            "unobserved_dictionary_label_count": 1,
        },
        "labels": [
            {"label_id": 1, "name": "Heart", "scan_coverage_count": 2},
            {"label_id": 2, "name": "Kidneys", "scan_coverage_count": 1},
            {"label_id": 3, "name": "Lung", "scan_coverage_count": 0},
        ],
        "scans": [
            scan("001", voxel_volume=2.0, counts={"0": 9, "1": 2, "2": 3}),
            scan("002", voxel_volume=3.0, counts={"0": 4, "1": 6}),
        ],
        "qualification": {
            "release_archive_identity_verified": True,
            "zip_member_crc_verified": True,
            "all_registered_segmentation_masks_scanned": True,
            "nifti_geometry_and_units_verified": True,
            "integer_labels_joined_to_source_dictionary": True,
        },
    }


def test_cohort_summary_joins_names_reports_missingness_and_separate_frames() -> None:
    result = summarize_intake(receipt(), input_receipt_sha256="c" * 64)

    assert result["status"] == "source_cohort_geometry_summary"
    assert result["source"]["intake_receipt_sha256"] == "c" * 64
    assert result["cohort"]["scan_count"] == 2
    assert result["cohort"]["participant_frames_registered_to_one_another"] is False
    assert len(result["scan_frames"]) == 2

    heart = result["labels"][0]
    assert heart["source_semantic_id"] == "healthy_total_body_cts_v3:segmentation-label:1"
    assert heart["name"] == "Heart"
    assert heart["scan_coverage_count"] == 2
    volume = heart["voxel_occupancy_volume_candidate_ml_distribution"]
    assert volume == {
        "observed_scan_count": 2,
        "units": "mL",
        "minimum": pytest.approx(0.004),
        "q25_linear": pytest.approx(0.0075),
        "median": pytest.approx(0.011),
        "q75_linear": pytest.approx(0.0145),
        "maximum": pytest.approx(0.018),
    }
    assert heart["per_scan_spatial_candidates"][0]["centroid_ras_mm"] == pytest.approx(
        [10.0 + 2.0 ** (1 / 3) * 2.0, 20.0 + 2.0 ** (1 / 3) * 3.0,
         30.0 - 2.0 ** (1 / 3) * 4.0],
    )

    kidneys = result["labels"][1]
    assert kidneys["scan_coverage_count"] == 1
    assert kidneys["missing_scan_ids"] == ["002"]
    assert kidneys["voxel_count_distribution"]["median"] == 3.0

    absent = result["labels"][2]
    assert absent["scan_coverage_count"] == 0
    assert absent["missing_scan_ids"] == ["001", "002"]
    assert absent["voxel_occupancy_volume_candidate_ml_distribution"]["median"] is None
    assert result["qualification"]["physical_tissue_volume_owner"] is False


def test_cohort_summary_rejects_dictionary_coverage_drift() -> None:
    data = receipt()
    data["labels"][0]["scan_coverage_count"] = 1
    with pytest.raises(ImportError, match="scan coverage disagrees"):
        summarize_intake(data, input_receipt_sha256="c" * 64)


def test_cohort_summary_rejects_volume_or_spatial_count_drift() -> None:
    data = receipt()
    data["scans"][0]["label_raster_volume_candidate_ml"]["1"] = 999.0
    with pytest.raises(ImportError, match="raster volume disagrees"):
        summarize_intake(data, input_receipt_sha256="c" * 64)

    data = receipt()
    data["scans"][0]["label_spatial_geometry_candidates"]["1"]["voxel_count"] = 3
    with pytest.raises(ImportError, match="spatial count disagrees"):
        summarize_intake(data, input_receipt_sha256="c" * 64)


def test_cohort_summary_rejects_affine_centroid_and_envelope_drift() -> None:
    data = receipt()
    data["scans"][0]["label_spatial_geometry_candidates"]["1"]["centroid_ras_mm"][0] += 1.0
    with pytest.raises(ImportError, match="RAS centroid disagrees"):
        summarize_intake(data, input_receipt_sha256="c" * 64)

    data = receipt()
    data["scans"][0]["label_spatial_geometry_candidates"]["1"]["voxel_envelope_aabb_ras_mm"]["maximum"][0] += 1.0
    with pytest.raises(ImportError, match="RAS envelope disagrees"):
        summarize_intake(data, input_receipt_sha256="c" * 64)


def test_cohort_summary_rejects_unqualified_or_wrong_schema_input() -> None:
    data = receipt()
    data["qualification"]["zip_member_crc_verified"] = False
    with pytest.raises(ImportError, match="intake lacks"):
        summarize_intake(data, input_receipt_sha256="c" * 64)

    data = receipt()
    data["schema"] = "unknown"
    with pytest.raises(ImportError, match="not a supported"):
        summarize_intake(data, input_receipt_sha256="c" * 64)

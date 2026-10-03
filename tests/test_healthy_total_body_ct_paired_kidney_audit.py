from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from numilab_human.healthy_total_body_ct_paired_kidney_audit import (
    _stream_label_crop,
    _summarize_components,
)
from numilab_human.model import ImportError


def test_component_pair_uses_each_scan_affine_to_order_ras_x_and_measure_occupancy() -> None:
    mask = np.zeros((9, 5, 5), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    mask[6:8, 1:3, 1:3] = True
    # The I axis is reversed in this scan's RAS+ affine.
    affine = [
        [-2.0, 0.0, 0.0, 20.0],
        [0.0, 3.0, 0.0, -6.0],
        [0.0, 0.0, 4.0, 12.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    result = _summarize_components(mask, [0, 0, 0], affine, 24.0,
                                   np=np, ndimage=ndimage)

    assert result["component_count"] == 2
    assert result["total_voxel_count"] == 16
    assert result["left_by_ras_x"]["centroid_ras_mm"][0] == 7.0
    assert result["right_by_ras_x"]["centroid_ras_mm"][0] == 17.0
    assert result["right_minus_left_ras_x_mm"] == 10.0
    assert result["source_mask_occupancy_ml"] == pytest.approx(0.384)
    assert result["smaller_to_larger_occupancy_ratio"] == 1.0


def test_more_than_two_components_is_not_silently_described_as_a_pair() -> None:
    mask = np.zeros((7, 3, 3), dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[3, 1, 1] = True
    mask[6, 2, 2] = True
    with pytest.raises(ImportError, match="expected two 26-connected kidney components"):
        _summarize_components(mask, [0, 0, 0], np.eye(4).tolist(), 1.0,
                              np=np, ndimage=ndimage)


def test_streamed_crop_matches_fortran_order_mask_and_hashes_full_member() -> None:
    labels = np.zeros((4, 3, 5), dtype="<f4", order="F")
    labels[1:3, 1:3, 2:4] = 6
    raw = labels.tobytes(order="F")
    stream = gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(raw)), mode="rb")
    digest = hashlib.sha256()
    crop = _stream_label_crop(
        stream,
        {"shape_ijk": [4, 3, 5], "voxel_count": labels.size, "bitpix": 32,
         "byte_order": "little", "label_scaling_slope": 1.0,
         "label_scaling_intercept": 0.0},
        [1, 1, 2], [2, 2, 3], 6, np=np, digest=digest)

    assert crop.shape == (2, 2, 2)
    assert crop.all()
    assert digest.hexdigest() == hashlib.sha256(raw).hexdigest()


def test_retained_cohort_receipt_binds_all_scans_and_reports_pair_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "Docs/media/healthy-total-body-ct-paired-kidney-20261003"
    plan_path = folder / "preregistered-plan-v1.json"
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((folder / "receipt.json").read_text())

    assert receipt["trial_plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert receipt["predicate_source_sha256"] == plan["predicate_source_sha256"]
    assert receipt["source"]["intake_sha256"] == plan["input_intake_sha256"]
    assert receipt["source"]["archive_sha256"] == plan["source_archive_sha256"]
    assert receipt["counts"] == {
        "measured_scan_count": 30,
        "ordered_pair_scan_count": 30,
        "scan_count": 30,
        "two_component_scan_count": 30,
    }
    assert len(receipt["rows"]) == 30
    assert all(row["component_count"] == 2
               and row["source_label_voxel_count"] == row["total_voxel_count"]
               and row["right_minus_left_ras_x_mm"] > 0
               for row in receipt["rows"])
    assert receipt["qualification"]["numi_subject_binding"] is False
    assert receipt["qualification"]["physical_kidney_volume"] is False

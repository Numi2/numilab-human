from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from numilab_human.healthy_total_body_ct_psoas_component_audit import (
    _distribution,
    _summarize_mask,
)
from numilab_human.healthy_total_body_ct_paired_kidney_audit import _stream_label_crop


def test_three_components_keep_extra_island_size_and_ras_order_explicit() -> None:
    mask = np.zeros((11, 5, 5), dtype=np.bool_)
    mask[1:3, 1:3, 1:3] = True
    mask[5, 1, 1:4] = True
    mask[9, 3, 3] = True
    affine = [
        [-2.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    result = _summarize_mask(mask, [0, 0, 0], affine, 2.0,
                             np=np, ndimage=ndimage)

    assert result["component_count"] == 3
    components = result["components_ordered_by_ras_x"]
    assert [row["ras_x_order"] for row in components] == [1, 2, 3]
    assert [row["voxel_count"] for row in components] == [1, 3, 8]
    assert result["smallest_component_fraction_of_label_voxels"] == pytest.approx(1 / 12)
    assert result["two_largest_component_balance_ratio"] == pytest.approx(3 / 8)
    assert result["total_source_mask_occupancy_ml"] == pytest.approx(0.024)


def test_shared_streaming_reader_extracts_label_crop_without_changing_member_hash() -> None:
    labels = np.zeros((4, 3, 5), dtype="<f4", order="F")
    labels[1:3, 1:3, 2:4] = 36
    raw = labels.tobytes(order="F")
    stream = gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(raw)), mode="rb")
    digest = hashlib.sha256()
    crop = _stream_label_crop(
        stream,
        {"shape_ijk": [4, 3, 5], "voxel_count": labels.size, "bitpix": 32,
         "byte_order": "little", "label_scaling_slope": 1.0,
         "label_scaling_intercept": 0.0},
        [1, 1, 2], [2, 2, 3], 36, np=np, digest=digest)

    assert crop.shape == (2, 2, 2)
    assert crop.all()
    assert digest.hexdigest() == hashlib.sha256(raw).hexdigest()


def test_distribution_uses_sorted_extrema_when_exception_occurs_mid_cohort() -> None:
    assert _distribution([2.0, 3.0, 2.0]) == {
        "minimum": 2.0,
        "median": 2.0,
        "maximum": 3.0,
    }


def test_corrected_cohort_receipt_retains_the_scan024_small_component_exception() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "Docs/media/healthy-total-body-ct-psoas-components-20261003"
    plan_path = folder / "corrective-plan-v2.json"
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((folder / "corrected-receipt-v2.json").read_text())
    superseded = json.loads((folder / "superseded-receipt-v1.json").read_text())

    assert receipt["trial_plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert receipt["predicate_source_sha256"] == plan["predicate_source_sha256"]
    assert plan["corrective_revision"]["supersedes_receipt_sha256"] == hashlib.sha256(
        (folder / "superseded-receipt-v1.json").read_bytes()).hexdigest()
    assert superseded["distributions"]["full26_component_count"]["maximum"] == 2.0
    assert receipt["counts"]["two_component_scan_count"] == 29
    assert receipt["counts"]["three_component_scan_count"] == 1
    assert receipt["distributions"]["full26_component_count"]["maximum"] == 3.0
    exception, = receipt["exception_scans"]
    assert exception["scan_id"] == "024"
    assert exception["components_ordered_by_ras_x"][1]["voxel_count"] == 37
    assert exception["smallest_component_fraction_of_label_voxels"] == pytest.approx(
        0.0002011820806577)
    assert receipt["qualification"]["individual_muscle_identity"] is False

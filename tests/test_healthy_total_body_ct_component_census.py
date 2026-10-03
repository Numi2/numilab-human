from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from numilab_human.healthy_total_body_ct_component_census import (
    CODE_FILES,
    PLAN_SCHEMA,
    TARGET_LABELS,
    _component_stats,
    _validate_trial_plan,
)
from numilab_human.model import ImportError


def test_face_and_corner_connectivity_are_reported_separately() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True

    result = _component_stats(mask, np=np, ndimage=ndimage)

    assert result["voxel_count"] == 2
    assert result["face6"]["component_count"] == 2
    assert result["full26"]["component_count"] == 1
    assert result["full26"]["largest_component_fraction"] == 1.0


def test_face_adjacent_voxels_form_one_component_in_both_connectivities() -> None:
    mask = np.zeros((3, 2, 2), dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[1, 0, 0] = True

    result = _component_stats(mask, np=np, ndimage=ndimage)

    assert result["face6"]["component_count"] == 1
    assert result["full26"]["component_count"] == 1
    assert result["face6"]["largest_component_voxel_count"] == 2


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ImportError, match="nonempty three-dimensional mask"):
        _component_stats(np.zeros((2, 2, 2), dtype=np.bool_), np=np, ndimage=ndimage)


def test_trial_plan_binds_source_design_and_predicate_hashes(tmp_path) -> None:
    hashes = {name: "a" * 64 for name in CODE_FILES}
    label_ids = {name: index + 1 for index, name in enumerate(TARGET_LABELS)}
    plan = {
        "schema": PLAN_SCHEMA,
        "input_intake_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "scan_count": 30,
        "source_id": "healthy_total_body_cts_v3",
        "source_release": "Version 3",
        "labels": list(TARGET_LABELS),
        "label_ids": label_ids,
        "connectivity_modes": ["face6", "full26"],
        "predicate_source_sha256": hashes,
        "question": "Does voxel-mask connectivity vary across source scans?",
        "predictions": ["full26 component count is never greater than face6"],
        "validity_checks": ["member hash and selected voxel counts match intake"],
        "analysis": ["retain every scan-label row and summarize observed counts"],
        "stop_rule": "stop on any hash, affine, or voxel-count mismatch",
        "evidence_boundary": "source masks only; no Numi binding or mechanics",
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    digest = _validate_trial_plan(
        path,
        intake_sha="b" * 64,
        archive_sha="c" * 64,
        scan_count=30,
        source_id="healthy_total_body_cts_v3",
        source_release="Version 3",
        label_ids=label_ids,
        code_hashes=hashes,
    )

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    plan["source_archive_sha256"] = "d" * 64
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ImportError, match="does not bind this source"):
        _validate_trial_plan(
            path,
            intake_sha="b" * 64,
            archive_sha="c" * 64,
            scan_count=30,
            source_id="healthy_total_body_cts_v3",
            source_release="Version 3",
            label_ids=label_ids,
            code_hashes=hashes,
        )


def test_retained_cohort_census_is_complete_and_keeps_the_lung_exception() -> None:
    root = Path(__file__).resolve().parents[1]
    census_path = root / (
        "Docs/media/healthy-total-body-ct-component-census-20261003/receipt.json"
    )
    plan_path = root / (
        "Docs/media/healthy-total-body-ct-component-census-20261003/"
        "preregistered-plan-v1.json"
    )
    census, plan = json.loads(census_path.read_text()), json.loads(plan_path.read_text())

    assert census["trial_plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert census["predicate_source_sha256"] == plan["predicate_source_sha256"]
    assert census["source"]["intake_sha256"] == plan["input_intake_sha256"]
    assert census["source"]["archive_sha256"] == plan["source_archive_sha256"]
    assert census["counts"] == {
        "scan_count": 30,
        "selected_label_count": 15,
        "expected_scan_label_pairs": 450,
        "measured_scan_label_pairs": 450,
        "missing_scan_label_pairs": 0,
    }
    rows = census["rows"]
    assert len(rows) == 450
    assert all(row["status"] == "measured" for row in rows)
    assert all(row["full26"]["component_count"] <= row["face6"]["component_count"]
               for row in rows)
    assert all(0.0 < row[mode]["largest_component_fraction"] <= 1.0
               for row in rows for mode in ("face6", "full26"))
    lung = census["distributions"]["Lung"]["full26"]
    assert lung["single_component_scan_count"] == 25
    assert lung["maximum"] == 6
    assert census["qualification"]["source_vessel_lumen_connectivity"] is False
    assert census["qualification"]["current_numi_subject_binding"] is False

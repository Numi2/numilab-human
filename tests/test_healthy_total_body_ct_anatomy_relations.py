from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.healthy_total_body_ct_anatomy_relations import (
    INTAKE_SCHEMA,
    compile_relations,
)


LABELS = [
    {"label_id": 3, "name": "Bladder"},
    {"label_id": 4, "name": "Brain"},
    {"label_id": 5, "name": "Heart"},
    {"label_id": 6, "name": "Kidneys"},
    {"label_id": 7, "name": "Liver"},
    {"label_id": 12, "name": "Lung"},
]


def _scan(scan_id: str, superior_offset: float, *, invert_heart_liver: bool = False):
    z = {
        "Brain": 5.0,
        "Heart": 4.0,
        "Lung": 3.5,
        "Liver": 3.0,
        "Kidneys": 2.0,
        "Bladder": 1.0,
    }
    if invert_heart_liver:
        z["Heart"] = 2.5
    ids = {row["name"]: row["label_id"] for row in LABELS}
    geometry = {}
    for label in LABELS:
        name = label["name"]
        centroid = [0.0, 0.0, z[name]]
        geometry[str(ids[name])] = {
            "centroid_voxel_center_ijk": centroid,
            "centroid_ras_mm": [0.0, 0.0, centroid[2] + superior_offset],
            "voxel_count": 10,
        }
    return {
        "scan_id": scan_id,
        "affine_source": "qform",
        "voxel_to_world_affine": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, superior_offset],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "label_spatial_geometry_candidates": geometry,
    }


def _write_intake(path: Path, scans: list[dict]) -> Path:
    path.write_text(json.dumps({
        "schema": INTAKE_SCHEMA,
        "source": {
            "archive_sha256": "a" * 64,
            "coordinate_convention": "NIfTI RAS+ in millimetres",
        },
        "labels": LABELS,
        "scans": scans,
    }), encoding="utf-8")
    return path


def test_scan_frames_remain_separate_and_all_four_orderings_pass(tmp_path: Path):
    intake = _write_intake(tmp_path / "intake.json", [
        _scan("001", 0.0),
        _scan("002", 500.0),
    ])

    report = compile_relations(intake)

    assert report["status"] == "passed_external_source_centroid_ordering"
    assert report["source"]["scan_count"] == 2
    assert report["source"]["source_scan_frames_kept_separate"] is True
    assert report["counts"] == {
        "scan_count": 2,
        "relation_count": 4,
        "total_relation_rows": 8,
        "passed_relation_rows": 8,
        "failed_relation_rows": 0,
        "missing_relation_rows": 0,
    }
    assert [row["scan_id"] for row in report["rows"][:4]] == ["001"] * 4
    assert [row["scan_id"] for row in report["rows"][4:]] == ["002"] * 4
    assert report["rows"][0]["ras_superior_delta_mm"] == 1.0
    assert report["rows"][4]["ras_superior_delta_mm"] == 1.0
    assert report["qualification"]["current_numi_subject_binding"] is False
    assert report["qualification"]["clinical_anatomy"] is False


def test_failed_ordering_is_retained_instead_of_hidden(tmp_path: Path):
    intake = _write_intake(tmp_path / "intake.json", [
        _scan("001", 0.0, invert_heart_liver=True),
    ])

    report = compile_relations(intake)

    assert report["status"] == "partial_external_source_centroid_ordering"
    assert report["counts"]["failed_relation_rows"] == 1
    heart_liver = next(row for row in report["rows"]
                       if row["relation"] == "heart_superior_to_liver")
    assert heart_liver["status"] == "failed_ordering"
    assert heart_liver["ras_superior_delta_mm"] == -0.5
    assert not report["qualification"]["external_source_centroid_ordering_consistency"]


def test_centroids_must_reproduce_in_their_own_nifti_affine(tmp_path: Path):
    scan = _scan("001", 0.0)
    scan["label_spatial_geometry_candidates"]["5"]["centroid_ras_mm"][2] += 0.1
    intake = _write_intake(tmp_path / "intake.json", [scan])

    with pytest.raises(HumanImportError, match="does not reproduce through its affine"):
        compile_relations(intake)

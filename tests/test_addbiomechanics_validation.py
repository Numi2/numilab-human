from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from numilab_human.addbiomechanics_validation import (
    SCHEMA,
    compile_candidate,
    immutable_write,
)
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def _artifact(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _table(path: Path, offset: float = 0.0, *, opensim: bool = False) -> None:
    rows = [
        (0.0, 0.10 + offset, 700.0 + offset, 80.0 + offset),
        (0.01, 0.12 + offset, 710.0 + offset, 82.0 + offset),
        (0.02, 0.14 + offset, 720.0 + offset, 84.0 + offset),
    ]
    if opensim:
        path.write_text(
            "name fixture\ndatacolumns=4\ndatarows=3\nendheader\n"
            "time\tknee_angle_r\tgrf_vertical\tknee_moment_r\n"
            + "\n".join("\t".join(str(value) for value in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(
            "time,knee_angle_r,grf_vertical,knee_moment_r\n"
            + "\n".join(",".join(str(value) for value in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )


def _manifest(tmp_path: Path, *, opensim: bool = False) -> Path:
    source = tmp_path / "subject.b3d"
    source.write_bytes(b"fixture AddBiomechanics subject artifact")
    calibration_ref = tmp_path / ("calibration_ref.sto" if opensim else "calibration_ref.csv")
    calibration_pred = tmp_path / ("calibration_pred.sto" if opensim else "calibration_pred.csv")
    validation_ref = tmp_path / ("validation_ref.sto" if opensim else "validation_ref.csv")
    validation_pred = tmp_path / ("validation_pred.sto" if opensim else "validation_pred.csv")
    _table(calibration_ref, opensim=opensim)
    _table(calibration_pred, offset=0.001, opensim=opensim)
    _table(validation_ref, opensim=opensim)
    _table(validation_pred, offset=0.002, opensim=opensim)
    value = {
        "schema": SCHEMA,
        "dataset": {
            "id": "addbiomechanics-core-v1",
            "version": "1.0",
            "url": "https://www.addbiomechanics.org/download_data.html",
            "license": "fixture-only",
            "provenance": {"kind": "fixture_only", "note": "unit-test table; no scientific claim"},
            "source_artifact": _artifact(source),
        },
        "subject": {"id": "fixture-male-01", "sex": "male", "age_years": 30,
                    "height_m": 1.80, "mass_kg": 75.0},
        "time_column": "time",
        "time_tolerance_s": 1.0e-12,
        "required_activities": ["standing", "walking"],
        "channels": [
            {"id": "knee_angle_r", "kind": "joint_angle", "column": "knee_angle_r", "unit": "rad", "threshold": 1.0},
            {"id": "grf_vertical", "kind": "grf", "column": "grf_vertical", "unit": "N", "threshold": 1.0},
            {"id": "knee_moment_r", "kind": "joint_moment", "column": "knee_moment_r", "unit": "N*m", "threshold": 1.0},
        ],
        "trials": [
            {"id": "calibration-standing-01", "activity": "standing", "split": "calibration",
             "reference": _artifact(calibration_ref), "prediction": _artifact(calibration_pred)},
            {"id": "validation-walking-01", "activity": "walking", "split": "validation",
             "reference": _artifact(validation_ref), "prediction": _artifact(validation_pred)},
        ],
        "prediction": {"runtime": "fixture", "source_revision": "a" * 40,
                        "clock_nanoseconds": 12500, "fit_trial_ids": ["calibration-standing-01"]},
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical(value) + b"\n")
    return path


def test_fixture_evaluation_is_split_and_fail_closed(tmp_path: Path) -> None:
    result = compile_candidate(_manifest(tmp_path))
    assert result["status"] == "fixture_only"
    assert result["clock"] == {"nanoseconds": 12500, "seconds": 1.25e-05, "exact": True}
    assert result["activity_coverage"] == {"standing": True, "walking": True}
    assert result["splits"]["calibration"]["trial_count"] == 1
    assert result["splits"]["validation"]["trial_count"] == 1
    assert result["qualification"]["calibration_validation_disjoint"]
    assert result["qualification"]["canonical_clock_12_5_us"]
    assert not result["qualification"]["release_qualified"]
    assert not result["qualification"]["native_mechanics_qualified"]


def test_opensim_sto_header_is_supported(tmp_path: Path) -> None:
    result = compile_candidate(_manifest(tmp_path, opensim=True))
    assert len(result["trials"]) == 2
    assert result["trials"][0]["sample_count"] == 3


def test_source_data_hash_and_clock_are_required(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["dataset"]["source_artifact"]["sha256"] = "0" * 64
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="source artifact SHA-256 changed"):
        compile_candidate(path)

    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["prediction"]["clock_nanoseconds"] = 25000
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="canonical 12.5 microsecond"):
        compile_candidate(path)


def test_calibration_and_validation_cannot_share_trial_or_table(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["trials"][1]["id"] = value["trials"][0]["id"]
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="duplicate trial id"):
        compile_candidate(path)

    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["trials"][1]["reference"] = value["trials"][0]["reference"]
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="reuses a table path"):
        compile_candidate(path)


def test_result_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate(_manifest(tmp_path))
    output = tmp_path / "result.json"
    first = immutable_write(output, result)
    assert immutable_write(output, result) == first
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, {**result, "status": "changed"})


def test_cli_arguments_require_a_manifest() -> None:
    from numilab_human.addbiomechanics_validation import add_arguments
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parsed = parser.parse_args(["--manifest", "manifest.json"])
    assert parsed.manifest == Path("manifest.json")

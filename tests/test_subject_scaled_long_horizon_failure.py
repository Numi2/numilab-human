from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.subject_scaled_long_horizon_failure import (
    OUTPUT,
    _attempt,
    compile_receipt,
    immutable_write,
)


def test_subject_scaled_long_horizon_retains_both_native_wait_failures() -> None:
    result = compile_receipt()
    assert result["status"] == "failed"
    assert result["source"]["device"] == "Mac mini M4 Pro"
    assert result["inputs"]["scaled_rigid_sha256"] == (
        "303722b1f50ba603d75c52796aad0072f8a33776d5c3a24557ad9b393db3efb4"
    )
    assert [attempt["step_count_requested"] for attempt in result["attempts"]] == [128, 512]
    assert all(not attempt["completed"] for attempt in result["attempts"])
    assert all(attempt["observed"]["payload_admission_reached"] for attempt in result["attempts"])
    assert all(attempt["observed"]["metal_submission_wait_observed"] for attempt in result["attempts"])
    control = result["one_step_control"]
    assert control["persistent_completed_steps"] == 1
    assert control["dynamic_force_audit_rows"] == 128
    assert control["stand_deterministic_replay"] == "bitwise"
    assert control["persistent_max_penetration_m"] == 0.0
    assert control["trace_samples"] == 2
    assert control["muscle_force_metal_active_records"] == 416
    assert result["two_step_control"]["persistent_completed_steps"] == 2
    assert result["two_step_control"]["trace_samples"] == 3
    assert result["two_step_control"]["muscle_force_metal_active_records"] == 832
    assert result["eight_step_control"]["persistent_completed_steps"] == 8
    assert result["eight_step_control"]["trace_samples"] == 9
    assert result["eight_step_control"]["muscle_force_metal_active_records"] == 3328
    assert not result["qualification"]["long_horizon_completed"]
    assert not result["qualification"]["sustained_standing"]


def test_subject_scaled_long_horizon_rejects_completed_or_missing_sample(tmp_path: Path) -> None:
    source = Path("Docs/media/native-subject-scaled-runtime-128-failure-20260915")
    for name in ("native.stdout.txt", "native.stderr.txt", "metal-wait.sample.txt",
                 "terminated-at-unix.txt"):
        target = tmp_path / name
        target.write_bytes((source / name).read_bytes())
    (tmp_path / "native.stdout.txt").write_text(
        (tmp_path / "native.stdout.txt").read_text(encoding="utf-8")
        + "myosim_articulated_marker_visual=ok\n",
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="unexpectedly contains a completed"):
        _attempt(tmp_path, name="tampered", step_count=128,
                 validation_layer=False, observed_seconds=1.0)


def test_subject_scaled_long_horizon_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_receipt()
    path = tmp_path / "receipt.json"
    first = immutable_write(path, result)
    assert immutable_write(path, result) == first
    path.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(path, result)


def test_checked_in_receipt_matches_compiler() -> None:
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == compile_receipt()

from __future__ import annotations

from pathlib import Path

import pytest

import numilab_human.native_passive_stand_refinement as refinement
from numilab_human.model import ImportError
from numilab_human.native_passive_stand_refinement import compile_refinement, immutable_write
from numilab_human.physiology import canonical


def test_common_duration_refinement_keeps_dynamic_gate_open() -> None:
    result = compile_refinement()

    assert result["source"]["commit"] == "7625ec565e086faf0dcd349846dadc2d22d65e86"
    assert result["common_duration"]["duration_seconds"] == pytest.approx(0.0064)
    assert result["common_duration"]["timestep_nanoseconds"] == [100000, 50000, 25000, 12500]
    assert result["common_duration"]["step_counts"] == [64, 128, 256, 512]
    assert result["common_duration"]["static_residual_spread"] == 0.0
    assert [row["persistent_completed_steps"] for row in result["cases"]] == [64, 128, 256, 512]
    assert [row["persistent_max_penetration_m"] for row in result["cases"]] == [0.0] * 4
    assert [row["stand_deterministic_replay"] for row in result["cases"]] == ["bitwise"] * 4
    assert result["convergence"]["peak_acceleration_range_over_minimum"] > 6.5
    assert not result["qualification"]["force_convergence"]
    assert result["qualification"]["complete_static_generalized_balance"]
    assert result["blocker"]["status"] == "open"


def test_refinement_rejects_tampered_native_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = Path("Docs/media/native-passive-stand-refinement-20260915")
    root = tmp_path / "cases"
    for name in ("100us", "50us", "25us", "12p5us"):
        target = root / name
        target.mkdir(parents=True)
        for filename in ("stdout.txt", "stderr.txt"):
            (target / filename).write_bytes((source_root / name / filename).read_bytes())
    stdout = root / "25us" / "stdout.txt"
    value = stdout.read_text(encoding="utf-8").replace(
        "persistent_max_penetration_m=0", "persistent_max_penetration_m=0.001", 1
    )
    stdout.write_text(value, encoding="utf-8")
    monkeypatch.setattr(refinement, "ROOT", tmp_path)
    with pytest.raises(ImportError, match="native passive stand refinement"):
        compile_refinement(case_root=root)


def test_refinement_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_refinement()
    output = tmp_path / "receipt.json"
    first = immutable_write(output, result)
    assert immutable_write(output, compile_refinement()) == first
    changed = {**result, "status": "changed"}
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, changed)

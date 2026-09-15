from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.addbiomechanics_subject_binding import compile_binding, immutable_write
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def test_binding_retains_subject_mass_mismatch_and_open_gates() -> None:
    result = compile_binding()
    assert result["schema"] == "HumanPack.addbiomechanics-subject-binding.v1"
    assert result["subject"]["reference_trial_count"] == 4
    assert result["subject"]["walking_reference_present"]
    assert not result["subject"]["standing_reference_present"]
    assert result["mass_comparison"]["measured_subject_mass_kg"] == 65.5
    assert result["mass_comparison"]["current_rigid_body_mass_kg"] == 97.13195176621342
    assert result["mass_comparison"]["difference_kg"] == 31.63195176621342
    assert not result["mass_comparison"]["same_scaled_mechanical_subject"]
    assert not result["qualification"]["subject_mass_identity_match"]
    assert not result["qualification"]["prediction_comparison_complete"]
    assert {item["id"] for item in result["blockers"]} == {
        "subject_mass_identity", "prediction_comparison", "standing_recovery_reference",
    }


def test_binding_rejects_promoted_subject_calibration(tmp_path: Path) -> None:
    source = Path("Docs/media/current-human-evidence-join-20260915/receipt-v3.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["qualification"]["subject_calibration"] = True
    path = tmp_path / "current.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="promoted subject"):
        compile_binding(current_evidence=path)


def test_binding_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_binding()
    output = tmp_path / "binding.json"
    first = immutable_write(output, result)
    assert immutable_write(output, compile_binding()) == first
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, {**result, "status": "changed"})

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.body_composition_mass_ledger import (
    _immutable_write,
    compile_ledger,
)
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def test_one_male_mass_ledger_closes_scalar_target_without_promoting_candidates() -> None:
    result = compile_ledger()

    assert result["status"] == "partial"
    assert result["subject"] == {
        "id": "Falisse2017:subject_1",
        "age_years": 43,
        "sex": "male",
        "height_m": 1.78,
        "target_mass_kg": 65.5,
    }
    assert result["scalar_mass_closure"]["target_closed"]
    assert result["scalar_mass_closure"]["closure_error_kg"] == pytest.approx(
        -5.684341886080802e-14
    )
    assert result["candidate_scopes"]["organ_surface_candidates"]["mass_kg"] == pytest.approx(
        2.7781575033215167
    )
    assert result["candidate_scopes"]["blood_tissue_transfer"]["mass_kg"] == pytest.approx(
        5.079334472359306
    )
    assert result["candidate_scopes"]["skeletal_muscle_route_partition"]["volume_m3"] == pytest.approx(
        0.006471304532959316
    )
    assert result["candidate_scopes"]["fat"]["mass_candidate_count"] == 0
    assert result["scope_policy"]["candidate_mass_sum_kg"] is None
    assert result["scope_policy"]["candidate_mass_admitted_to_dynamics"] is False
    assert result["ownership"]["mechanical_blood_mass_owner_count"] == 0
    assert result["qualification"]["candidate_scope_non_additivity_checked"]
    assert not result["qualification"]["whole_body_mass_partition"]


def test_mass_ledger_rejects_a_promoted_candidate_owner(tmp_path: Path) -> None:
    source = Path("Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["qualification"]["mechanical_blood_mass_owner"] = True
    path = tmp_path / "blood.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="promoted mechanical_blood_mass_owner"):
        compile_ledger(blood_transfer=path)


def test_mass_ledger_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_ledger()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_ledger())
    output.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(output, result)

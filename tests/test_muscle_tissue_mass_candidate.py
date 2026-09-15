import json
from pathlib import Path

import pytest

from numilab_human.muscle_tissue_mass_candidate import (
    PROFILE,
    SCHEMA,
    VOLUME_RECEIPT,
    MuscleMassError,
    _immutable_write,
    compile_candidate,
)
from numilab_human.physiology import canonical


def test_compiles_explicit_candidate_mass_without_promoting_owner() -> None:
    result = compile_candidate()
    assert result["schema"] == SCHEMA
    assert result["counts"] == {
        "source_muscle_surface_count": 148,
        "closed_volume_count": 60,
        "unadmitted_surface_count": 88,
        "physical_volume_owner_count": 0,
        "mechanical_mass_owner_count": 0,
    }
    assert result["density"]["candidate_kg_per_m3"] == 1060.0
    assert result["density"]["subject_calibrated"] is False
    assert result["totals"]["candidate_mass_kg"] == pytest.approx(6.859582804936875)
    assert result["qualification"]["candidate_mass_budget"]
    assert result["qualification"]["skeletal_muscle_tissue_mass_owner"] is False
    assert result["qualification"]["disjoint_volume_partition"] is False


def test_rejects_promoted_volume_owner(tmp_path: Path) -> None:
    volume = json.loads(VOLUME_RECEIPT.read_text())
    volume["qualification"]["physical_volume_owner"] = True
    path = tmp_path / "volume.json"
    path.write_bytes(canonical(volume) + b"\n")
    with pytest.raises(MuscleMassError, match="promoted physical_volume_owner"):
        compile_candidate(volume_receipt=path)


def test_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate(profile=PROFILE)
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate(profile=PROFILE))

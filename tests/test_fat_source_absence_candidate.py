from pathlib import Path

import pytest

from numilab_human.fat_source_absence_candidate import (
    FatSourceAbsenceError,
    compile_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fat_source_absence_is_explicit_and_fail_closed() -> None:
    result = compile_candidate()
    assert result["schema"] == "HumanPack.fat-source-absence-candidate.v1"
    assert result["counts"]["fat_surface_count"] == 0
    assert result["qualification"]["fat_source_absence_bound"]
    assert not result["qualification"]["fat_mass_candidate"]
    assert not result["qualification"]["fat_mechanical_mass_owner"]


def test_fat_profile_cannot_be_promoted(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_bytes((ROOT / "config/fat-source-absence-candidate.v1.json").read_bytes()
                        .replace(b'"expected_fat_surface_count":0',
                                 b'"expected_fat_surface_count":1'))
    with pytest.raises(FatSourceAbsenceError, match="no-source contract"):
        compile_candidate(profile=profile)


def test_fat_receipt_keeps_all_physical_owners_zero() -> None:
    result = compile_candidate()
    assert result["counts"]["fat_physical_volume_owner_count"] == 0
    assert result["counts"]["fat_mechanical_mass_owner_count"] == 0
    assert not result["qualification"]["integrated_human_qualification"]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.tissue_mass_candidate import PROFILE, ROOT, compile_candidate
from numilab_human.physiology import canonical


def test_organ_tissue_mass_candidate_is_closed_but_not_promoted() -> None:
    result = compile_candidate()
    assert result["status"] == "partial"
    assert result["counts"]["organ_surface_mass_candidates"] > 0
    assert result["counts"]["shared_source_members"] == 8
    assert result["totals"]["candidate_surface_mass_kg"] > 0.0
    assert result["qualification"]["candidate_zeroth_first_second_moments"]
    assert not result["qualification"]["mechanical_mass_owner_assigned"]
    assert not result["qualification"]["fat_geometry_and_mass"]
    assert not result["qualification"]["skeletal_muscle_tissue_partition"]
    assert all(
        row["mass_admission"] == "shared_source_member_overlap_unresolved"
        for row in result["candidates"]
        if len(row["region_ids"]) > 1
    )


def test_profile_must_cover_all_regions(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["region_classes"].pop("liver")
    path = tmp_path / "profile.json"
    path.write_bytes(canonical(profile) + b"\n")
    with pytest.raises(ImportError, match="cover every source region"):
        compile_candidate(profile=path)


def test_component_moment_receipt_adds_only_the_seven_safe_sums() -> None:
    result = compile_candidate(
        moments=ROOT / "Docs/media/organ-geometry-component-moments-20260915/receipt-v1.json"
    )
    assert result["source"]["moments_schema"] == "HumanPack.organ-geometry-component-moments.v1"
    assert result["counts"]["organ_surface_mass_candidates"] == 349
    assert result["counts"]["source_members"] == 378
    assert result["totals"]["physical_mass_owner_count"] == 0
    assert not result["qualification"]["material_calibration"]

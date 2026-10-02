from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.physiology import canonical
from numilab_human.whole_body_overlap_organ_moments import compile_candidates


def test_overlap_only_organ_sources_get_nonowned_surface_moments() -> None:
    result = compile_candidates()

    assert result["status"] == "partial"
    assert result["counts"] == {
        "overlap_census_member_count": 104,
        "organ_mass_inventory_member_count": 378,
        "overlap_census_only_organ_member_count": 74,
        "computed_single_closed_surface_moment_count": 73,
        "unaggregated_multi_component_member_count": 1,
        "ontology_priority_class_counts": {
            "organ": 11,
            "organ_component": 4,
            "organ_region": 59,
        },
    }
    rows = result["members"]
    assert len(rows) == len({row["member_id"] for row in rows}) == 74
    assert all(row["physical_volume_m3"] is None for row in rows)
    assert all(row["density_kg_per_m3"] is None for row in rows)
    assert all(row["mechanical_mass_kg"] is None for row in rows)
    assert all(row["physical_volume_owner"] is False for row in rows)
    assert all(row["mechanical_mass_owner"] is False for row in rows)

    separated = next(row for row in rows if row["member_id"] == "FJ3150")
    assert separated["moment_status"] == "unaggregated_multi_component_member"
    assert separated["source_surface_moments"] is None
    assert separated["component_aabb_pairwise_disjoint"] is False
    assert len(separated["source_component_moments"]) == 2
    assert result["qualification"]["cross_surface_moments_summed"] is False
    assert result["qualification"]["physical_volume_owner"] is False


def test_source_member_hash_drift_is_rejected(tmp_path: Path) -> None:
    source = Path("Docs/media/whole-body-source-overlap-census-20261002/receipt-v4.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    member = next(row for row in value["source_members"] if row["source_member_id"] == "FJ2561")
    member["source_member_sha256"] = "0" * 64
    path = tmp_path / "overlap-census.json"
    path.write_bytes(canonical(value) + b"\n")

    with pytest.raises(ImportError, match="source member hash differs"):
        compile_candidates(census=path)

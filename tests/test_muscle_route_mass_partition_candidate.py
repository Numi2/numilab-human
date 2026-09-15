from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.muscle_route_mass_partition_candidate import (
    MusclePartitionError,
    compile_candidate,
)
from numilab_human.physiology import canonical


ROOT = Path(__file__).resolve().parents[1]


def test_route_mass_partition_closes_candidate_budget() -> None:
    result = compile_candidate()
    assert result["schema"] == "HumanPack.muscle-route-mass-partition-candidate.v1"
    assert result["counts"] == {
        "source_route_count": 416,
        "closed_surface_count": 60,
        "route_incidence_count": 82,
        "routes_with_candidate_budget": 78,
        "routes_without_closed_geometry": 338,
        "routes_without_surface_binding": 238,
        "unadmitted_surface_count": 88,
    }
    assert result["totals"]["candidate_partition_is_disjoint"] is True
    assert result["totals"]["candidate_is_mechanical_mass"] is False
    assert result["totals"]["candidate_mass_residual_kg"] == pytest.approx(0.0, abs=2e-12)
    assert result["totals"]["candidate_volume_residual_m3"] == pytest.approx(0.0, abs=2e-15)
    assert result["qualification"]["equal_incidence_candidate_partition"] is True
    assert result["qualification"]["skeletal_muscle_tissue_mass_owner"] is False
    assert all(row["physical_volume_owner"] is False for row in result["route_budgets"])
    assert result["route_budgets"][0]["mass_partition_status"] == "no_closed_surface"


def test_each_shared_surface_is_partitioned_once() -> None:
    result = compile_candidate()
    by_surface: dict[str, float] = {}
    for row in result["surface_allocations"]:
        by_surface[row["member_id"]] = by_surface.get(row["member_id"], 0.0) + row["allocation_fraction"]
    assert len(by_surface) == 60
    assert all(value == pytest.approx(1.0, abs=1e-15) for value in by_surface.values())


def test_route_partition_rejects_promoted_owner(tmp_path: Path) -> None:
    source = ROOT / "Docs/media/muscle-route-volume-join-candidate-20260915/receipt-v1.json"
    value = json.loads(source.read_text())
    value["qualification"]["volume_partition_owner"] = True
    path = tmp_path / "route.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(MusclePartitionError, match="promoted a physical owner"):
        compile_candidate(route_join=path)

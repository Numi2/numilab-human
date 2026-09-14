from __future__ import annotations

import pytest

from numilab_human.soft_tissue_surface_candidate import (
    SurfaceCandidateError,
    compile_candidate,
)


def test_source_route_and_surface_inventory_retains_unbound_routes() -> None:
    candidate = compile_candidate()
    assert candidate["counts"] == {
        "source_route_count": 416,
        "surface_count": 150,
        "muscle_surface_count": 148,
        "tendon_surface_count": 2,
        "routes_with_surface_binding": 178,
        "routes_without_surface_binding": 238,
        "fat_surface_count": 0,
        "skin_surface_count": 0,
        "physical_volume_owner_count": 0,
        "mechanical_mass_owner_count": 0,
        "volumetric_active_force_owner_count": 0,
    }
    assert candidate["qualification"]["source_route_identity_bound"]
    assert candidate["qualification"]["muscle_surface_identity_bound"]
    assert candidate["qualification"]["tendon_surface_identity_bound"]
    assert candidate["qualification"]["unbound_routes_retained"]
    assert not candidate["qualification"]["fat_geometry_present"]
    assert not candidate["qualification"]["physical_tissue_volume_owner"]
    assert not candidate["qualification"]["volumetric_active_muscle_owner"]


def test_surface_rows_and_route_rows_are_identity_consistent() -> None:
    candidate = compile_candidate()
    surfaces = {row["stable_id"]: row for row in candidate["surface_rows"]}
    routes = {row["source_actuator_index"]: row for row in candidate["route_rows"]}
    assert len(surfaces) == 150
    assert len(routes) == 416
    for row in routes.values():
        assert row["surface_count"] == len(row["surface_ids"])
        assert row["physical_volume_owner"] is None
        for stable_id in row["surface_ids"]:
            assert row["name"] in {
                match["name"] for match in surfaces[stable_id]["matched_muscles"]
            }
    assert any(row["layer"] == "tendon" for row in surfaces.values())


def test_profile_mutation_is_rejected(tmp_path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text('{"schema":"forged"}\n')
    with pytest.raises(SurfaceCandidateError, match="fields differ"):
        compile_candidate(profile=profile)

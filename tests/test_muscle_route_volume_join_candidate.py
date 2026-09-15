import json
from pathlib import Path

import pytest

from numilab_human.muscle_route_volume_join_candidate import (
    PROFILE,
    SURFACES,
    VOLUMES,
    SCHEMA,
    MuscleJoinError,
    compile_candidate,
)


def test_joins_routes_to_closed_geometry_without_promoting_volume() -> None:
    result = compile_candidate(surfaces=SURFACES, volumes=VOLUMES, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["counts"] == {
        "source_route_count": 416,
        "muscle_surface_count": 148,
        "tendon_surface_count": 2,
        "closed_geometry_volume_owner_count": 60,
        "routes_with_surface_binding": 178,
        "routes_without_surface_binding": 238,
        "routes_with_closed_geometry": 78,
        "closed_geometry_route_incidence_count": 82,
    }
    assert result["qualification"]["closed_geometry_volume_identity_joined"]
    assert result["qualification"]["shared_surface_incidence_explicit"]
    assert not result["qualification"]["volume_partition_owner"]
    assert not result["qualification"]["skeletal_muscle_tissue_mass_owner"]
    assert not result["qualification"]["activation_force_transfer"]


def test_rejects_route_volume_hash_mismatch(tmp_path: Path) -> None:
    volume = json.loads(VOLUMES.read_text())
    volume["owners"][0]["member_sha256"] = "0" * 64
    mutated = tmp_path / "volume.json"
    mutated.write_text(json.dumps(volume))
    with pytest.raises(MuscleJoinError, match="hashes differ"):
        compile_candidate(surfaces=SURFACES, volumes=mutated, profile=PROFILE)


def test_rejects_promoted_route_owner(tmp_path: Path) -> None:
    surface = json.loads(SURFACES.read_text())
    surface["route_rows"][0]["physical_volume_owner"] = "muscle:0"
    mutated = tmp_path / "surface.json"
    mutated.write_text(json.dumps(surface))
    with pytest.raises(MuscleJoinError, match="already owns"):
        compile_candidate(surfaces=mutated, volumes=VOLUMES, profile=PROFILE)

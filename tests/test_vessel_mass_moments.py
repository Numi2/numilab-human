from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.vessel_mass_moments import BODY_LINKS, REGISTRATION, compile_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_source_vessel_mass_moments_are_closed_without_promotion() -> None:
    result = compile_candidate(density_kg_m3=1060.0)
    assert result["status"] == "partial"
    assert result["totals"]["unique_member_count"] == 6
    assert result["totals"]["owner_count"] == 6
    assert result["totals"]["mass_kg"] > 0.0
    assert result["totals"]["registered_world_surface_integral_volume_m3"] > result["totals"]["source_surface_integral_volume_m3"]
    assert result["qualification"]["zeroth_first_second_mass_moments"]
    assert result["qualification"]["atomic_checkpoint_restore"]
    assert not result["qualification"]["anatomical_blood_mass_owner"]
    assert all(row["mass_kg"] > 0.0 for row in result["owners"])


def test_initial_momentum_is_explicit_and_deterministic() -> None:
    velocity = [0.1, -0.2, 0.3]
    result = compile_candidate(density_kg_m3=1060.0, initial_velocity_mps=velocity)
    assert all(row["initial_velocity_mps"] == velocity for row in result["owners"])
    assert all(row["source_surface_integral_volume_m3"] > 0.0 for row in result["owners"])
    expected = [result["totals"]["mass_kg"] * component for component in velocity]
    assert result["totals"]["linear_momentum_kg_m_per_s"] == pytest.approx(expected)


def test_existing_owner_or_body_link_tampering_is_rejected(tmp_path: Path) -> None:
    registration = json.loads(REGISTRATION.read_text())
    registration["bindings"][0]["mechanical_mass_owner"] = "forged-owner"
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(HumanImportError, match="registration already owns mass"):
        compile_candidate(registration=registration_path, density_kg_m3=1060.0)


def test_nonpositive_density_is_rejected() -> None:
    with pytest.raises(HumanImportError, match="density must be positive"):
        compile_candidate(density_kg_m3=0.0)

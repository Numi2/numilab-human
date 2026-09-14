from __future__ import annotations

import copy

import pytest

from numilab_human.organ_tissue_exchange_candidate import (
    CLOCK_SECONDS,
    ExchangeError,
    compile_candidate,
    simulate,
)


def test_source_bound_candidate_covers_seven_beds_without_physical_owners() -> None:
    candidate = compile_candidate()
    assert candidate["counts"] == {
        "bed_count": 7,
        "region_count": 7,
        "source_blood_owner_count": 6,
        "source_members_bound": 329,
        "tissue_candidate_volume_owner_count": 0,
        "tissue_exchange_owner_count": 0,
    }
    assert candidate["qualification"]["regional_blood_transport_bound"]
    assert candidate["qualification"]["regional_tissue_surface_volume_bound"]
    assert candidate["qualification"]["bidirectional_oxygen_amount_exchange_candidate"]
    assert not candidate["qualification"]["physical_tissue_volume_owner"]
    assert not candidate["qualification"]["tissue_exchange_owner"]
    assert not candidate["qualification"]["material_calibration"]
    for row in candidate["beds"]:
        assert row["tissue_candidate_volume_m3"] > 0.0
        assert row["physical_tissue_volume_owner"] is None
        assert row["tissue_exchange_owner"] is None


def test_exact_clock_exchange_conserves_amounts_and_rolls_back() -> None:
    candidate = compile_candidate()
    receipt = simulate(candidate, steps=512, timestep_s=CLOCK_SECONDS, reject_step=37)
    assert receipt["accepted_steps"] == 511
    assert receipt["rejected_steps"] == 1
    assert receipt["transfer_count"] == 3 * 7 * 511
    assert receipt["conservation"]["mass_conserved"]
    assert receipt["conservation"]["volume_conserved"]
    assert receipt["conservation"]["oxygen_conserved"]
    assert receipt["rollback"]["rejected_candidate_state_neutral"]

    accepted = simulate(candidate, steps=511, timestep_s=CLOCK_SECONDS)
    assert receipt["final_state"] == accepted["final_state"]
    assert receipt["accepted_state_trace_sha256"] == accepted["accepted_state_trace_sha256"]


def test_exchange_can_reverse_when_tissue_concentration_is_higher() -> None:
    candidate = compile_candidate()
    modified = copy.deepcopy(candidate)
    bed = modified["initial_state"]["beds"]["right_lung"]
    row = next(row for row in modified["beds"] if row["region_id"] == "right_lung")
    modified_tissue = row["tissue_candidate_volume_m3"] * 2.0
    bed["tissue_oxygen_mol"] = modified_tissue
    before_transit = bed["transit_oxygen_mol"]
    receipt = simulate(modified, steps=1, timestep_s=CLOCK_SECONDS)
    after = receipt["final_state"]["beds"]["right_lung"]
    assert after["tissue_oxygen_mol"] < modified_tissue
    assert after["transit_oxygen_mol"] > before_transit
    assert receipt["conservation"]["oxygen_conserved"]


def test_noncanonical_clock_is_rejected() -> None:
    candidate = compile_candidate()
    with pytest.raises(ExchangeError, match="canonical 12.5 us"):
        simulate(candidate, steps=1, timestep_s=1.0e-5)


def test_candidate_state_is_not_mutated_by_simulation() -> None:
    candidate = compile_candidate()
    before = copy.deepcopy(candidate["initial_state"])
    simulate(candidate, steps=2, timestep_s=CLOCK_SECONDS)
    assert candidate["initial_state"] == before

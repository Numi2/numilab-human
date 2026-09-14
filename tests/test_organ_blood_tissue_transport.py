from __future__ import annotations

import copy

import pytest

from numilab_human.organ_blood_tissue_transport import (
    CLOCK_SECONDS,
    TransportError,
    compile_candidate,
    simulate,
)


def test_regional_candidate_binds_seven_beds_without_promoting_mass_owners() -> None:
    candidate = compile_candidate()
    assert candidate["counts"] == {
        "bed_count": 7,
        "region_count": 7,
        "source_blood_owner_count": 6,
        "source_members_bound": 329,
    }
    assert candidate["qualification"]["regional_arterial_tissue_venous_transport"]
    assert not candidate["qualification"]["physical_tissue_volume_owner"]
    assert not candidate["qualification"]["mechanical_mass_owner"]
    assert not candidate["qualification"]["two_way_tissue_exchange"]
    for row in candidate["beds"]:
        assert row["allocation_share"] > 0.0
        assert row["physical_volume_owner"] is None
        assert row["mechanical_mass_owner"] is None
        assert row["tissue_exchange_owner"] is None
    for owner in candidate["source_owner_partition"]:
        assert owner["allocation_sum"] == pytest.approx(1.0, abs=1e-15)


def test_exact_clock_transport_conserves_and_rolls_back() -> None:
    candidate = compile_candidate()
    receipt = simulate(candidate, steps=512, timestep_s=CLOCK_SECONDS, reject_step=37)
    assert receipt["accepted_steps"] == 511
    assert receipt["rejected_steps"] == 1
    assert receipt["transfer_count"] == 2 * 7 * 511
    assert receipt["conservation"]["mass_conserved"]
    assert receipt["conservation"]["volume_conserved"]
    assert receipt["rollback"]["rejected_candidate_state_neutral"]

    accepted = simulate(candidate, steps=511, timestep_s=CLOCK_SECONDS)
    assert receipt["final_state"] == accepted["final_state"]
    assert receipt["accepted_state_trace_sha256"] == accepted["accepted_state_trace_sha256"]


def test_noncanonical_clock_is_rejected() -> None:
    candidate = compile_candidate()
    with pytest.raises(TransportError, match="canonical 12.5 us"):
        simulate(candidate, steps=1, timestep_s=1.0e-5)


def test_candidate_state_is_not_mutated_by_simulation() -> None:
    candidate = compile_candidate()
    before = copy.deepcopy(candidate["initial_state"])
    simulate(candidate, steps=2, timestep_s=CLOCK_SECONDS)
    assert candidate["initial_state"] == before

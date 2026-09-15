from __future__ import annotations

import copy

import pytest

from numilab_human.organ_blood_mass_transfer_candidate import (
    CLOCK_SECONDS,
    compile_candidate,
    simulate,
)
from numilab_human.model import ImportError


def test_mass_transfer_binds_seven_beds_and_conserves_both_directions() -> None:
    candidate = compile_candidate()
    assert candidate["status"] == "partial"
    assert candidate["counts"] == {
        "bed_count": 7,
        "region_count": 7,
        "source_blood_owner_count": 6,
        "source_members_bound": 329,
    }
    assert candidate["qualification"]["blood_tissue_zeroth_moment_mass_transfer"]
    assert not candidate["qualification"]["anatomical_vessel_lumen"]
    assert not candidate["qualification"]["mechanical_blood_mass_owner"]
    receipt = simulate(candidate, steps=512, timestep_s=CLOCK_SECONDS, reject_step=37)
    assert receipt["accepted_steps"] == 511
    assert receipt["rejected_steps"] == 1
    assert receipt["transfer_counts"]["blood_to_tissue"] > 0
    assert receipt["transfer_counts"]["tissue_to_blood"] > 0
    assert receipt["conservation"]["mass_conserved"]
    assert receipt["conservation"]["volume_conserved"]
    assert receipt["rollback"]["rejected_candidate_state_neutral"]


def test_mass_transfer_rejects_noncanonical_clock() -> None:
    candidate = compile_candidate()
    with pytest.raises(ImportError, match="canonical 12.5 us clock"):
        simulate(candidate, steps=1, timestep_s=25.0e-6)


def test_mass_transfer_rejects_promoted_mechanical_owner() -> None:
    candidate = copy.deepcopy(compile_candidate())
    candidate["qualification"]["mechanical_blood_mass_owner"] = True
    with pytest.raises(ImportError, match="promoted a mechanical or anatomical owner"):
        simulate(candidate, steps=1, timestep_s=CLOCK_SECONDS)

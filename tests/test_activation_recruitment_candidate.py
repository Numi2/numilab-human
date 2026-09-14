import json
from pathlib import Path

import pytest

from numilab_human.activation_recruitment_candidate import (
    RecruitmentError,
    compile_candidate,
)


def test_source_recruitment_is_complete_but_not_calibrated() -> None:
    candidate = compile_candidate()
    assert candidate["counts"] == {
        "source_routes": 416,
        "generalized_dofs": 128,
        "support_witnesses": 18,
        "nonzero_routes": 237,
        "upper_bound_routes": 47,
        "zero_routes": 179,
    }
    assert candidate["qualification"]["source_route_activation_complete"]
    assert candidate["qualification"]["nonmaximal_recruitment_candidate"]
    assert candidate["qualification"]["pose_recruitment_fiber_state_bound"]
    assert candidate["qualification"]["fp32_activation_transport_bounded"]
    assert candidate["qualification"]["generalized_force_decomposition_closed"]
    assert not candidate["qualification"]["activation_calibration"]
    assert not candidate["qualification"]["sustained_standing"]


def test_profile_mutation_rejects_uniform_maximal_candidate(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    from numilab_human.activation_recruitment_candidate import PROFILE
    mutated = json.loads(PROFILE.read_text())
    mutated["expected_nonzero_routes"] = 416
    profile.write_text(json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(RecruitmentError, match="nonzero recruitment count drifted"):
        compile_candidate(profile=profile)

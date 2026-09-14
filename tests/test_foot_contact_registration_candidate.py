from pathlib import Path

import pytest

from numilab_human.foot_contact_registration_candidate import (
    FootRegistrationError,
    compile_candidate,
)


def test_source_foot_registration_binds_geometry_and_support_witnesses() -> None:
    candidate = compile_candidate(sources=Path("Sources"))
    assert candidate["counts"] == {
        "foot_body_count": 4,
        "source_mesh_count": 60,
        "registered_source_mesh_count": 30,
        "unique_source_member_count": 30,
        "registered_source_member_count": 30,
        "support_witness_count": 18,
        "active_support_witness_count": 6,
    }
    assert candidate["qualification"]["source_foot_geometry_registered"]
    assert candidate["qualification"]["source_to_body_rest_transform_bound"]
    assert candidate["qualification"]["lower_limb_multi_pose_continuity"]
    assert candidate["qualification"]["support_witness_identity_bound"]
    assert not candidate["qualification"]["anatomical_collider_admitted"]
    assert not candidate["qualification"]["contact_material_calibration"]


def test_profile_mutation_rejects_wrong_source_mesh_count(tmp_path: Path) -> None:
    import json
    from numilab_human.foot_contact_registration_candidate import PROFILE

    profile = json.loads(PROFILE.read_text())
    profile["expected_source_mesh_count"] = 59
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(FootRegistrationError, match="expected_source_mesh_count differs"):
        compile_candidate(sources=Path("Sources"), profile=path)

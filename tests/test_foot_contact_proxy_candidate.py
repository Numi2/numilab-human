import json
from pathlib import Path

import pytest

from numilab_human.foot_contact_proxy_candidate import (
    PROFILE,
    REGISTRATION,
    SCHEMA,
    FootProxyError,
    compile_candidate,
)


def test_compiles_registered_body_frame_proxy_enclosures() -> None:
    result = compile_candidate(registration=REGISTRATION, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["counts"] == {
        "foot_body_count": 4,
        "registered_source_member_count": 30,
        "proxy_count": 30,
        "support_witness_count": 18,
        "active_support_witness_count": 6,
        "multi_pose_count": 7,
    }
    assert len(result["body_summaries"]) == 4
    assert all(row["body_frame_aabb_proxy"]["encloses_transformed_source_aabb"]
               for row in result["proxies"])
    assert result["qualification"]["source_triangle_enclosure_preserved"]
    assert result["qualification"]["conservative_body_frame_proxy_bounds"]
    assert not result["qualification"]["anatomical_collider_admitted"]
    assert not result["qualification"]["anatomical_supports_loading"]


def test_rejects_non_rigid_registration(tmp_path: Path) -> None:
    registration = json.loads(REGISTRATION.read_text())
    registration["registration"]["source_members"][0][
        "source_obj_mm_to_core_inertial_body_m"
    ][0][0] = 2.0
    mutated = tmp_path / "registration.json"
    mutated.write_text(json.dumps(registration))
    with pytest.raises(FootProxyError, match="anisotropic|unit length|orthogonal"):
        compile_candidate(registration=mutated, profile=PROFILE)


def test_rejects_profile_count_drift(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text())
    profile["expected_proxy_count"] = 29
    mutated = tmp_path / "profile.json"
    mutated.write_text(json.dumps(profile, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(FootProxyError, match="expected_proxy_count differs"):
        compile_candidate(registration=REGISTRATION, profile=mutated)

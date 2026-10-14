from __future__ import annotations

import pytest

from numilab_human.resting_anatomy import (
    HEAD_MEMBERSHIP_TABLE_SHA256,
    _head_member_identity,
)


def _atlas():
    return {
        "source": {
            "tables": [
                {
                    "file": "partof_element_parts.txt",
                    "sha256": HEAD_MEMBERSHIP_TABLE_SHA256,
                }
            ]
        },
        "tables": {
            "part_of": {
                ("FMA78469", "fourth ventricle"): {"FJ1731"},
                ("FMA78467", "cerebral aqueduct"): {"FJ1738"},
            }
        },
    }


def test_head_member_identity_uses_narrow_source_component_and_keeps_family():
    atlas = _atlas()

    ventricle = _head_member_identity(
        atlas,
        "FJ1731",
        "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8",
    )
    aqueduct = _head_member_identity(
        atlas,
        "FJ1738",
        "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3",
    )

    assert ventricle["concept_id"] == "FMA78469"
    assert ventricle["name"] == "fourth ventricle"
    assert aqueduct["concept_id"] == "FMA78467"
    assert aqueduct["name"] == "cerebral aqueduct"
    assert ventricle["family_concept_id"] == aqueduct["family_concept_id"] == "FMA50801"
    assert ventricle["separate_physiological_compartment"] is False
    assert _head_member_identity(atlas, "FJ1775", "0" * 64) is None


@pytest.mark.parametrize(
    ("member_id", "member_sha256"),
    [
        ("FJ1731", "0" * 64),
        ("FJ1738", "0" * 64),
    ],
)
def test_head_member_identity_rejects_unpinned_obj_bytes(member_id, member_sha256):
    with pytest.raises(ValueError, match="source member bytes changed"):
        _head_member_identity(_atlas(), member_id, member_sha256)


def test_head_member_identity_rejects_membership_drift():
    atlas = _atlas()
    atlas["tables"]["part_of"][("FMA78469", "fourth ventricle")].clear()

    with pytest.raises(ValueError, match="not in its declared anatomical component"):
        _head_member_identity(
            atlas,
            "FJ1731",
            "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8",
        )


def test_head_member_identity_rejects_wrong_membership_table_pin():
    atlas = _atlas()
    atlas["source"]["tables"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="pinned BodyParts3D"):
        _head_member_identity(
            atlas,
            "FJ1731",
            "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8",
        )

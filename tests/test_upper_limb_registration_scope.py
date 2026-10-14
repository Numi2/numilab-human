from copy import deepcopy

import pytest

from numilab_human.upper_limb_registration import (
    _hand_names,
    _scope_target_names,
    _upper_names,
    _validate_scope_anchor_preservation,
)


def test_bilateral_hand_scope_excludes_arm_and_shoulder_targets():
    hand_names = _hand_names("r") | _hand_names("l")
    assert _scope_target_names("bilateral_hands") == hand_names
    assert hand_names <= _scope_target_names("all")
    assert not hand_names & {
        "clavicle_l",
        "scapula_l",
        "humerus_l",
        "radius_l",
        "ulna_l",
        "clavicle_r",
        "scapula_r",
        "humerus_r",
        "radius_r",
        "ulna_r",
    }


def test_default_scope_preserves_previous_candidate_name_set():
    assert _scope_target_names("all") == (
        (_upper_names("r") | _upper_names("l")) - {"clavicle_r", "clavicle_l"}
    )


def test_unknown_scope_fails_closed():
    with pytest.raises(ValueError, match="unsupported upper-limb registration scope"):
        _scope_target_names("left_hand")


def test_scope_anchor_preservation_accepts_unchanged_out_of_scope_anchors():
    source = {
        "anchors": [
            {
                "source": {"archive_sha256": "a" * 64, "member_id": "FJ1", "member_sha256": "b" * 64, "concept_id": "FMA1"},
                "target": {"name": "radius_l"},
            },
            {
                "source": {"archive_sha256": "a" * 64, "member_id": "FJ2", "member_sha256": "c" * 64, "concept_id": "FMA2"},
                "target": {"name": "hand_l"},
            },
        ]
    }
    candidate = deepcopy(source)
    candidate["anchors"][1]["registration"] = {"candidate": True}
    _validate_scope_anchor_preservation(source, candidate, {"hand_l"})


def test_scope_anchor_preservation_rejects_out_of_scope_mutation_or_reordering():
    source = {
        "anchors": [
            {
                "source": {"archive_sha256": "a" * 64, "member_id": "FJ1", "member_sha256": "b" * 64, "concept_id": "FMA1"},
                "target": {"name": "radius_l"},
            },
            {
                "source": {"archive_sha256": "a" * 64, "member_id": "FJ2", "member_sha256": "c" * 64, "concept_id": "FMA2"},
                "target": {"name": "hand_l"},
            },
        ]
    }
    candidate = deepcopy(source)
    candidate["anchors"][0]["registration"] = {"moved": True}
    with pytest.raises(RuntimeError, match="out-of-scope anchor radius_l"):
        _validate_scope_anchor_preservation(source, candidate, {"hand_l"})

    candidate = deepcopy(source)
    candidate["anchors"].reverse()
    with pytest.raises(RuntimeError, match="reordered or replaced"):
        _validate_scope_anchor_preservation(source, candidate, {"hand_l"})

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from numilab_human.resting_anatomy import (
    HEAD_MEMBERSHIP_TABLE_SHA256,
    _head_member_identity,
    rebind_head_member_identity_metadata,
)
from numilab_human.resting_anatomy_interface_patch import encode_payload


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



def _payload(*, body_323=23, layer_323=1):
    row = {
        "vertices6": np.asarray(
            [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]],
            dtype="<f4",
        ),
        "faces": np.asarray([[0, 1, 2]], dtype="<i8"),
        "flags": 0,
    }
    rows = {
        323: {**row, "body_index": body_323, "layer": layer_323},
        326: {**row, "body_index": 23, "layer": 1},
    }
    return encode_payload(
        (b"NHANAT1\0", 5, 0, 0, 0, 0x31C77660, b"s" * 32),
        rows,
    )


def _receipt(payload_raw=None):
    payload_raw = _payload() if payload_raw is None else payload_raw
    return {
        "payload": {
            "abi": 5,
            "sha256": hashlib.sha256(payload_raw).hexdigest(),
            "path": "/pinned/resting.nhanatomy",
            "surface_count": 2,
            "vertex_count": 6,
            "index_count": 6,
            "registration_fingerprint32": "31c77660",
            "source_sha256": (b"s" * 32).hex(),
        },
        "functional_bindings": {"anatomy_payload_sha256": hashlib.sha256(payload_raw).hexdigest(), "channels": [1, 2]},
        "mass_geometry_accounting": {"total_mass_kg": 72.0},
        "provenance": {
            "source_id_map": {
                "323": {"name": "brain", "source_member": "FJ1731", "source_sha256": "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8", "layer": 1, "body_index": 23, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
                "326": {"name": "brain", "source_member": "FJ1738", "source_sha256": "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3", "layer": 1, "body_index": 23, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
                "325": {"name": "brain", "source_member": "FJ1775", "source_sha256": "c" * 64, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
            },
            "other": {"preserve": True},
        },
    }


def _member_hashes():
    return {
        "FJ1731": "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8",
        "FJ1738": "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3",
    }


def test_metadata_rebind_updates_only_the_two_verified_source_rows():
    payload_raw = _payload()
    receipt = _receipt(payload_raw)
    result, audit = rebind_head_member_identity_metadata(
        receipt, _atlas(), _member_hashes(), payload_raw=payload_raw
    )
    assert result["payload"] == receipt["payload"]
    assert result["functional_bindings"] == receipt["functional_bindings"]
    assert result["mass_geometry_accounting"] == receipt["mass_geometry_accounting"]
    assert result["provenance"]["source_id_map"]["323"]["name"] == "fourth ventricle"
    assert result["provenance"]["source_id_map"]["326"]["name"] == "cerebral aqueduct"
    assert result["provenance"]["source_id_map"]["325"] == receipt["provenance"]["source_id_map"]["325"]
    assert result["provenance"]["other"] == receipt["provenance"]["other"]
    assert audit["changed_stable_ids"] == ["323", "326"]
    assert audit["payload_typed_rows_verified"] is True
    assert [row["semantic_id_from_layer"] for row in audit["rows"]] == [51010, 51010]
    normalized = copy.deepcopy(result)
    normalized["provenance"]["source_id_map"] = copy.deepcopy(receipt["provenance"]["source_id_map"])
    assert normalized == receipt


def test_metadata_rebind_rejects_wrong_input_member_hash():
    with pytest.raises(ValueError, match="source member bytes changed"):
        rebind_head_member_identity_metadata(
            _receipt(), _atlas(), {"FJ1731": "0" * 64, "FJ1738": "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3"},
            payload_raw=_payload(),
        )


def test_metadata_rebind_rejects_wrong_receipt_stable_id():
    receipt = _receipt()
    receipt["provenance"]["source_id_map"]["999"] = receipt["provenance"]["source_id_map"].pop("323")
    with pytest.raises(ValueError, match="unexpected stable ID"):
        rebind_head_member_identity_metadata(
            receipt, _atlas(), _member_hashes(), payload_raw=_payload()
        )


def test_metadata_rebind_rejects_wrong_receipt_obj_hash():
    receipt = _receipt()
    receipt["provenance"]["source_id_map"]["323"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt source member hash differs"):
        rebind_head_member_identity_metadata(
            receipt, _atlas(), _member_hashes(), payload_raw=_payload()
        )


def test_metadata_rebind_rejects_membership_table_drift():
    atlas = _atlas()
    atlas["source"]["tables"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned BodyParts3D"):
        rebind_head_member_identity_metadata(
            _receipt(), atlas, _member_hashes(), payload_raw=_payload()
        )



def test_metadata_rebind_rejects_payload_bytes_from_another_receipt():
    with pytest.raises(ValueError, match="payload bytes differ"):
        rebind_head_member_identity_metadata(
            _receipt(_payload()), _atlas(), _member_hashes(),
            payload_raw=_payload(body_323=22),
        )


def test_metadata_rebind_rejects_wrong_decoded_typed_row():
    wrong = _payload(body_323=24)
    with pytest.raises(ValueError, match="NHANAT typed owner differs for head row 323"):
        rebind_head_member_identity_metadata(
            _receipt(wrong), _atlas(), _member_hashes(), payload_raw=wrong
        )


def test_metadata_rebind_rejects_wrong_receipt_typed_row():
    payload_raw = _payload()
    receipt = _receipt(payload_raw)
    receipt["provenance"]["source_id_map"]["326"]["layer"] = 2
    with pytest.raises(ValueError, match="receipt typed owner differs for head row 326"):
        rebind_head_member_identity_metadata(
            receipt, _atlas(), _member_hashes(), payload_raw=payload_raw
        )


def test_metadata_rebind_rejects_header_count_drift():
    payload_raw = _payload()
    receipt = _receipt(payload_raw)
    receipt["payload"]["index_count"] += 3
    with pytest.raises(ValueError, match="header index_count"):
        rebind_head_member_identity_metadata(
            receipt, _atlas(), _member_hashes(), payload_raw=payload_raw
        )

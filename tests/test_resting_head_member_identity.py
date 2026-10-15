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
                ("FMA62004", "Medulla oblongata"): {"FJ1769"},
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
    medulla = _head_member_identity(
        atlas,
        "FJ1769",
        "ab318e2ca0b27656d06edb4a3d3315669c70d7625989a3573fa66a0e586c6afc",
    )

    assert ventricle["family_concept_id"] == aqueduct["family_concept_id"] == medulla["family_concept_id"] == "FMA50801"
    assert medulla["concept_id"] == "FMA62004"
    assert medulla["name"] == "Medulla oblongata"
    assert medulla["geometry_role"] == "passive brainstem tissue surface"
    assert medulla["source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert medulla["separate_physiological_compartment"] is False
    assert ventricle["separate_physiological_compartment"] is False
    assert _head_member_identity(atlas, "FJ1775", "0" * 64) is None


@pytest.mark.parametrize(
    ("member_id", "member_sha256"),
    [
        ("FJ1731", "0" * 64),
        ("FJ1738", "0" * 64),
        ("FJ1769", "0" * 64),
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
        343: {**row, "body_index": 23, "layer": 1},
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
            "surface_count": 3,
            "vertex_count": 9,
            "index_count": 9,
            "registration_fingerprint32": "31c77660",
            "source_sha256": (b"s" * 32).hex(),
        },
        "functional_bindings": {"anatomy_payload_sha256": hashlib.sha256(payload_raw).hexdigest(), "channels": [1, 2]},
        "mass_geometry_accounting": {"total_mass_kg": 72.0},
        "provenance": {
            "source_id_map": {
                "323": {"name": "brain", "source_member": "FJ1731", "source_sha256": "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8", "layer": 1, "body_index": 23, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
                "326": {"name": "brain", "source_member": "FJ1738", "source_sha256": "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3", "layer": 1, "body_index": 23, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
                "343": {"name": "brain", "source_member": "FJ1769", "source_sha256": "ab318e2ca0b27656d06edb4a3d3315669c70d7625989a3573fa66a0e586c6afc", "source_license": "CC-BY-4.0", "layer": 1, "body_index": 23, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": {"label": "brain", "source_license": "CC-BY-4.0"}},
                "325": {"name": "brain", "source_member": "FJ1775", "source_sha256": "c" * 64, "head_family": {"concept_id": "FMA50801", "source_name": "brain", "hierarchy": "part_of"}, "source_owner_metadata": None},
            },
            "other": {"preserve": True},
        },
    }


def _member_hashes():
    return {
        "FJ1731": "b56e1cd574977821f5331174a31a830ec6a477e2c6c3f32667f7e46a6716eae8",
        "FJ1738": "089159be9815563e0645e5a28e7302fbc0bc842af760ac886bf3d0542f34a8f3",
        "FJ1769": "ab318e2ca0b27656d06edb4a3d3315669c70d7625989a3573fa66a0e586c6afc",
    }


def test_metadata_rebind_updates_only_the_three_verified_source_rows():
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
    medulla = result["provenance"]["source_id_map"]["343"]
    assert medulla["name"] == "Medulla oblongata"
    assert medulla["source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert medulla["source_owner_metadata"]["source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert medulla["source_owner_metadata"]["source_obj_identity"] == {
        "member_id": "FJ1769", "concept_id": "FMA62004",
        "name": "Medulla oblongata", "hierarchy": "part_of",
    }
    assert result["provenance"]["source_id_map"]["325"] == receipt["provenance"]["source_id_map"]["325"]
    assert result["provenance"]["other"] == receipt["provenance"]["other"]
    assert audit["changed_stable_ids"] == ["323", "326", "343"]
    medulla_audit = next(row for row in audit["rows"] if row["stable_id"] == "343")
    assert medulla_audit["before_source_license"] == "CC-BY-4.0"
    assert medulla_audit["after_source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert audit["payload_typed_rows_verified"] is True
    assert [row["semantic_id_from_layer"] for row in audit["rows"]] == [51010, 51010, 51010]
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


@pytest.mark.parametrize("field", ["layer", "body_index"])
def test_metadata_rebind_rejects_boolean_typed_owner(field):
    payload_raw = _payload()
    receipt = _receipt(payload_raw)
    receipt["provenance"]["source_id_map"]["323"][field] = True
    with pytest.raises(ValueError, match="receipt typed owner differs for head row 323"):
        rebind_head_member_identity_metadata(
            receipt, _atlas(), _member_hashes(), payload_raw=payload_raw
        )

def test_metadata_rebind_reports_actual_before_name_on_idempotent_rebind():
    payload_raw = _payload()
    receipt = _receipt(payload_raw)
    receipt["provenance"]["source_id_map"]["323"]["name"] = "fourth ventricle"
    result, audit = rebind_head_member_identity_metadata(
        receipt, _atlas(), _member_hashes(), payload_raw=payload_raw
    )
    row = next(item for item in audit["rows"] if item["stable_id"] == "323")
    assert row["before_name"] == "fourth ventricle"
    assert result["provenance"]["source_id_map"]["323"]["name"] == "fourth ventricle"

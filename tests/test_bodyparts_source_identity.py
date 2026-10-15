from __future__ import annotations

import hashlib

import pytest

from numilab_human import model
from numilab_human import resting_pleura_proxy
from numilab_human.resting_anatomy import rebind_central_canal_identity_metadata
from numilab_human.resting_anatomy_interface_patch import encode_payload

OBJ_HEADER = b"""#
# The license for this database is specified in the Creative Commons Attribution-Share Alike 2.1 Japan. If you use data from this database, please be sure attribute this database as follows: "BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan".
# http://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html
#
# Compatibility version : 4.0
# File ID : FJ1737
# Representation ID : BP6682
# Build-up logic : FMA 3.0 part_of
# Concept ID : FMA78497
# English name : Central canal of spinal cord
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""

EXPECTED_SHA256 = "c86dd74640e7c820eda28783944a90998045ff42a131a7a1456a64fdff0e6471"


def _specification(**overrides):
    result = {
        "concept_id": "FMA7647",
        "source_name": "spinal cord",
        "member_id": "FJ1737",
        "hierarchy": "part_of",
        "layer": "nerve",
        "myosim_body": "torso",
    }
    result.update(overrides)
    return result


def _expected_for_fixture():
    return {
        "stable_id": 12,
        "accepted_selection_identities": {
            ("FMA7647", "spinal cord"),
            ("FMA78497", "Central canal of spinal cord"),
        },
        "source_concept_id": "FMA78497",
        "source_label": "Central canal of spinal cord",
        "hierarchy": "part_of",
        "source_member_sha256": hashlib.sha256(OBJ_HEADER).hexdigest(),
        "layer": "nerve",
        "myosim_body": "torso",
        "core_body_index": 20,
        "embedded_license": "CC Attribution-Share Alike 2.1 Japan",
        "legacy_license_claim": "CC-BY-4.0",
    }


def test_fj1737_importer_identity_table_is_source_pinned():
    expected = model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS["FJ1737"]
    assert expected["stable_id"] == 12
    assert expected["source_concept_id"] == "FMA78497"
    assert expected["source_label"] == "Central canal of spinal cord"
    assert expected["source_member_sha256"] == EXPECTED_SHA256
    assert expected["embedded_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert expected["legacy_license_claim"] == "CC-BY-4.0"


def test_source_header_guard_checks_every_selected_member_and_license():
    header = model._bodyparts_validate_obj_source_header(
        _specification(), OBJ_HEADER, "FJ1737.obj",
    )
    assert header["member_id"] == "FJ1737"
    assert header["hierarchy"] == "part_of"
    assert header["license"] == "CC Attribution-Share Alike 2.1 Japan"

    with pytest.raises(model.ImportError, match="member header differs"):
        model._bodyparts_validate_obj_source_header(
            _specification(), OBJ_HEADER.replace(b"FJ1737", b"FJ1739"), "FJ1737.obj",
        )
    with pytest.raises(model.ImportError, match="hierarchy differs"):
        model._bodyparts_validate_obj_source_header(
            _specification(), OBJ_HEADER.replace(b"part_of", b"is_a  "), "FJ1737.obj",
        )


def test_source_header_identity_corrects_legacy_spinal_cord_map(monkeypatch):
    monkeypatch.setitem(
        model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS,
        "FJ1737",
        _expected_for_fixture(),
    )
    header = model._bodyparts_obj_source_header(OBJ_HEADER, "FJ1737.obj")
    assert header == {
        "member_id": "FJ1737",
        "hierarchy": "part_of",
        "concept_id": "FMA78497",
        "name": "Central canal of spinal cord",
        "license": "CC Attribution-Share Alike 2.1 Japan",
    }

    correction = model._bodyparts_member_identity_correction(
        12, _specification(), OBJ_HEADER, "FJ1737.obj",
    )
    assert correction["concept_id"] == "FMA78497"
    assert correction["label"] == "Central canal of spinal cord"
    assert correction["source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert correction["source_selection_identity"] == {
        "concept_id": "FMA7647",
        "label": "spinal cord",
        "hierarchy": "part_of",
    }
    evidence = correction["source_identity_correction"]
    assert evidence["stable_id_preserved"] == 12
    assert evidence["typed_owner_preserved"] == {
        "layer": "nerve",
        "core_body_index": 20,
    }
    assert evidence["selected_identity_before_correction"] == {
        "concept_id": "FMA7647",
        "label": "spinal cord",
    }
    assert evidence["embedded_obj_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert evidence["legacy_importer_license_claim"] == "CC-BY-4.0"
    assert "full source archive" in evidence["license_discrepancy"]
    assert evidence["geometry_changed"] is False


def test_canonicalized_selection_is_idempotent(monkeypatch):
    monkeypatch.setitem(
        model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS,
        "FJ1737",
        _expected_for_fixture(),
    )
    correction = model._bodyparts_member_identity_correction(
        12,
        _specification(
            concept_id="FMA78497",
            source_name="Central canal of spinal cord",
        ),
        OBJ_HEADER,
        "FJ1737.obj",
    )
    assert correction["concept_id"] == "FMA78497"
    assert correction["source_selection_identity"]["concept_id"] == "FMA78497"


@pytest.mark.parametrize(
    ("stable_id", "specification", "obj", "message"),
    [
        (13, _specification(), OBJ_HEADER, "stable row drifted"),
        (12, _specification(concept_id="FMA7647", source_name="medulla"), OBJ_HEADER,
         "configured identity drifted"),
        (12, _specification(layer="organ"), OBJ_HEADER, "typed owner/source hierarchy drifted"),
        (12, _specification(myosim_body="head"), OBJ_HEADER, "typed owner/source hierarchy drifted"),
        (12, _specification(), OBJ_HEADER.replace(b"FMA78497", b"FMA7647"),
         "source member bytes changed"),
    ],
)
def test_identity_table_rejects_source_or_owner_drift(monkeypatch, stable_id, specification, obj, message):
    monkeypatch.setitem(
        model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS,
        "FJ1737",
        _expected_for_fixture(),
    )
    with pytest.raises(model.ImportError, match=message):
        model._bodyparts_member_identity_correction(
            stable_id, specification, obj, "FJ1737.obj",
        )


def test_identity_table_rejects_embedded_license_or_header_drift(monkeypatch):
    expected = _expected_for_fixture()
    monkeypatch.setitem(model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS, "FJ1737", expected)
    changed_license = OBJ_HEADER.replace(
        b"CC Attribution-Share Alike 2.1 Japan",
        b"CC Attribution 4.0 International",
    )
    expected["source_member_sha256"] = hashlib.sha256(changed_license).hexdigest()
    with pytest.raises(model.ImportError, match="embedded identity/license changed"):
        model._bodyparts_member_identity_correction(
            12, _specification(), changed_license, "FJ1737.obj",
        )



def _central_canal_payload():
    row = {
        "vertices6": __import__("numpy").asarray(
            [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]],
            dtype="<f4",
        ),
        "faces": __import__("numpy").asarray([[0, 1, 2]], dtype="<i8"),
        "flags": 0,
    }
    return encode_payload(
        (b"NHANAT1\0", 5, 0, 0, 0, 0x31C77660, b"s" * 32),
        {12: {**row, "body_index": 20, "layer": 3}},
    )


def _central_canal_receipt(payload_raw, member_sha256):
    header, _records, _rows = resting_pleura_proxy._parse_payload(payload_raw)
    return {
        "payload": {
            "abi": int(header[1]),
            "surface_count": int(header[2]),
            "vertex_count": int(header[3]),
            "index_count": int(header[4]),
            "registration_fingerprint32": f"{int(header[5]):08x}",
            "source_sha256": header[6].hex(),
            "sha256": hashlib.sha256(payload_raw).hexdigest(),
            "path": "/pinned/resting.nhanatomy",
        },
        "functional_bindings": {"anatomy_payload_sha256": hashlib.sha256(payload_raw).hexdigest()},
        "mass_geometry_accounting": {"total_mass_kg": 72.0},
        "provenance": {
            "source_id_map": {
                "12": {
                    "name": "spinal cord",
                    "source_member": "FJ1737",
                    "source_sha256": member_sha256,
                    "layer": 3,
                    "body_index": 20,
                    "source_owner_metadata": {
                        "concept_id": "FMA7647",
                        "label": "spinal cord",
                        "member_id": "FJ1737",
                        "member_sha256": member_sha256,
                        "hierarchy": "part_of",
                        "layer": "nerve",
                        "myosim_body": "torso",
                        "core_body_index": 20,
                    },
                },
                "323": {"name": "fourth ventricle", "source_member": "FJ1731"},
            },
        },
    }


def test_metadata_rebind_corrects_only_typed_row12(monkeypatch):
    payload_raw = _central_canal_payload()
    member_sha256 = hashlib.sha256(OBJ_HEADER).hexdigest()
    expected = _expected_for_fixture()
    monkeypatch.setitem(model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS, "FJ1737", expected)
    receipt = _central_canal_receipt(payload_raw, member_sha256)

    result, audit = rebind_central_canal_identity_metadata(
        receipt, OBJ_HEADER, payload_raw=payload_raw,
    )
    row = result["provenance"]["source_id_map"]["12"]
    owner = row["source_owner_metadata"]
    assert row["name"] == "Central canal of spinal cord"
    assert row["source_license"] == "CC Attribution-Share Alike 2.1 Japan"
    assert owner["concept_id"] == "FMA78497"
    assert owner["label"] == "Central canal of spinal cord"
    assert owner["source_obj_identity"] == {"member_id": "FJ1737", "concept_id": "FMA78497", "name": "Central canal of spinal cord", "hierarchy": "part_of"}
    assert owner["source_selection_identity"]["concept_id"] == "FMA7647"
    assert owner["source_identity_correction"]["legacy_importer_license_claim"] == "CC-BY-4.0"
    assert result["provenance"]["source_id_map"]["323"] == receipt["provenance"]["source_id_map"]["323"]
    assert result["payload"] == receipt["payload"]
    assert result["functional_bindings"] == receipt["functional_bindings"]
    assert result["mass_geometry_accounting"] == receipt["mass_geometry_accounting"]
    assert audit["stable_id"] == "12"
    assert audit["native_payload_sha256_unchanged"] is True
    assert audit["source_archive_full_sha256_independently_verified"] is False
    normalized = result["provenance"]["source_id_map"]["12"]
    assert normalized["source_sha256"] == receipt["provenance"]["source_id_map"]["12"]["source_sha256"]


def test_metadata_rebind_rejects_duplicate_or_wrong_typed_row(monkeypatch):
    payload_raw = _central_canal_payload()
    member_sha256 = hashlib.sha256(OBJ_HEADER).hexdigest()
    monkeypatch.setitem(
        model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS,
        "FJ1737",
        _expected_for_fixture(),
    )
    receipt = _central_canal_receipt(payload_raw, member_sha256)
    receipt["provenance"]["source_id_map"]["99"] = {
        **receipt["provenance"]["source_id_map"]["12"],
        "source_member": "FJ1737",
    }
    with pytest.raises(ValueError, match="must map exactly once"):
        rebind_central_canal_identity_metadata(receipt, OBJ_HEADER, payload_raw=payload_raw)

    receipt = _central_canal_receipt(payload_raw, member_sha256)
    receipt["provenance"]["source_id_map"]["12"]["body_index"] = True
    with pytest.raises(ValueError, match="receipt typed owner differs"):
        rebind_central_canal_identity_metadata(receipt, OBJ_HEADER, payload_raw=payload_raw)


def test_metadata_rebind_rejects_source_member_or_payload_drift(monkeypatch):
    payload_raw = _central_canal_payload()
    member_sha256 = hashlib.sha256(OBJ_HEADER).hexdigest()
    monkeypatch.setitem(
        model.BODYPARTS3D_MEMBER_IDENTITY_CORRECTIONS,
        "FJ1737",
        _expected_for_fixture(),
    )
    receipt = _central_canal_receipt(payload_raw, member_sha256)
    wrong_obj = OBJ_HEADER.replace(b"FMA78497", b"FMA7647")
    with pytest.raises(ValueError, match="source member bytes differ"):
        rebind_central_canal_identity_metadata(receipt, wrong_obj, payload_raw=payload_raw)

    receipt = _central_canal_receipt(payload_raw, member_sha256)
    receipt["payload"]["index_count"] += 3
    with pytest.raises(ValueError, match="header index_count"):
        rebind_central_canal_identity_metadata(receipt, OBJ_HEADER, payload_raw=payload_raw)


def test_torso_anatomy_cli_describes_the_selected_central_canal():
    from numilab_human.cli import parser

    help_text = parser().format_help()
    assert "central-canal surfaces" in help_text
    assert "spinal-cord surfaces" not in help_text

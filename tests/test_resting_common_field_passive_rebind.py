from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

import numilab_human.resting_common_field_passive_rebind as rebind
from numilab_human.resting_anatomy_interface_patch import encode_payload

rebind_passive_rows = rebind.rebind_passive_rows


def fixture():
    vertices = np.array([[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 1]], dtype="<f4")
    rows = {sid: {"body_index": 23, "layer": 1, "flags": 0,
                  "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])}
            for sid in (1, 323, 343)}
    header = (b"NHANAT1\0", 5, 3, 9, 9, 42, b"a" * 32)
    raw = encode_payload(header, rows)
    sha = hashlib.sha256(raw).hexdigest()
    receipt = {
        "payload": {"sha256": sha},
        "functional_bindings": {"anatomy_payload_sha256": sha, "drive": [1, 2]},
        "mass_accounting": {"kg": 72},
        "passive_tissue_parent_bindings": {"anatomy_payload_sha256": sha,
                                          "anatomy_surfaces": []},
        "provenance": {
            "source_id_map": {str(sid): {"source": str(sid)} for sid in rows},
            "cardiac_geometry_binding": {"common_field": {
                "schema": "numi.human.cardiac_common_field.v1",
                "anatomy_payload_sha256": sha, "volume_stable_ids": [1],
                "vertex_ranges": [{"stable_id": 1, "vertex_count": 3, "first_vertex": 0}],
                "passive_attachment_ranges": [],
                **{k: {"path": k, "sha256": "b" * 64}
                   for k in ("map", "domain_boxes", "polynomials")},
            }},
        },
    }
    new_rows = copy.deepcopy(rows)
    for sid in (323, 343):
        new_rows[sid]["vertices6"][0, 0] = 0.25
    return receipt, raw, header, new_rows


def run(receipt, raw, header, rows, ids=(323, 343)):
    return rebind_passive_rows(receipt, raw, encode_payload(header, rows),
                               changed_ids=ids, output_path="/candidate.nhanatomy")


def test_rebind_preserves_function_mass_owner_and_source_identities():
    receipt, raw, h, rows = fixture()
    original = copy.deepcopy(receipt)
    result, audit = run(receipt, raw, h, rows)
    assert receipt == original
    assert result["mass_accounting"] == original["mass_accounting"]
    assert result["functional_bindings"]["drive"] == [1, 2]
    assert result["provenance"]["source_id_map"] == original["provenance"]["source_id_map"]
    assert audit["unchanged_row_count"] == 1
    assert result["passive_tissue_parent_bindings"]["anatomy_payload_sha256"] == result["payload"]["sha256"]


@pytest.mark.parametrize("mutation", ["owner", "extra", "parent", "common", "base", "source"])
def test_rebind_rejects_unproven_changes(mutation):
    r, raw, h, rows = fixture()
    if mutation == "owner":
        rows[323]["body_index"] = 20
    elif mutation == "extra":
        rows[1]["vertices6"][0, 0] = 0.5
    elif mutation == "parent":
        r["passive_tissue_parent_bindings"]["anatomy_surfaces"] = [{"stable_id": 323}]
    elif mutation == "common":
        r["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"] = "c" * 64
    elif mutation == "base":
        r["payload"]["sha256"] = "c" * 64
    else:
        h = (*h[:5], 43, h[6])
    with pytest.raises(ValueError):
        run(r, raw, h, rows)


@pytest.mark.parametrize("ids", [(True, 343), (323.0, 343), (323, 323), (), (323, 999)])
def test_strict_declared_ids(ids):
    r, raw, h, rows = fixture()
    with pytest.raises(ValueError):
        run(r, raw, h, rows, ids)


def central_canal_owner_migration_fixture(tmp_path):
    source_member_raw = b"authenticated FJ1737 OBJ member test bytes"
    source_member_sha = hashlib.sha256(source_member_raw).hexdigest()
    vertices = np.array([[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 1]], dtype="<f4")
    rows = {
        1: {"body_index": 23, "layer": 1, "flags": 0,
            "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])},
        12: {"body_index": 20, "layer": 3, "flags": 0,
             "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])},
        323: {"body_index": 23, "layer": 1, "flags": 0,
              "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])},
        343: {"body_index": 23, "layer": 1, "flags": 0,
              "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])},
    }
    header = (b"NHANAT1\0", 5, 4, 12, 12, 42, b"a" * 32)
    base_raw = encode_payload(header, rows)
    base_sha = hashlib.sha256(base_raw).hexdigest()
    base_map = {
        "1": {"body_index": 23, "layer": 1, "source_member": "other", "source_sha256": "x"},
        "12": {"body_index": 20, "layer": 3, "source_member": "FJ1737",
               "source_sha256": source_member_sha},
        "323": {"body_index": 23, "layer": 1, "source_member": "other", "source_sha256": "y"},
        "343": {"body_index": 23, "layer": 1, "source_member": "other", "source_sha256": "z"},
    }
    receipt = {
        "payload": {"sha256": base_sha},
        "functional_bindings": {"anatomy_payload_sha256": base_sha, "drive": [1, 2]},
        "mass_accounting": {"kg": 72},
        "passive_tissue_parent_bindings": {"anatomy_payload_sha256": base_sha,
                                            "anatomy_surfaces": []},
        "provenance": {
            "source_id_map": base_map,
            "cardiac_geometry_binding": {"common_field": {
                "schema": "numi.human.cardiac_common_field.v1",
                "anatomy_payload_sha256": base_sha, "volume_stable_ids": [1],
                "vertex_ranges": [{"stable_id": 1, "vertex_count": 3, "first_vertex": 0}],
                "passive_attachment_ranges": [],
                **{k: {"path": k, "sha256": "b" * 64}
                   for k in ("map", "domain_boxes", "polynomials")},
            }},
        },
    }
    migration_rows = copy.deepcopy(rows)
    migration_rows[12]["body_index"] = 23
    migration_raw = encode_payload(header, migration_rows)
    migration_sha = hashlib.sha256(migration_raw).hexdigest()
    registration = tmp_path / "registration.json"
    rigid_manifest = tmp_path / "rigid-reference.json"
    registration.write_bytes(b"exact registration source")
    rigid_manifest.write_bytes(b"exact rigid manifest")
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    frame = {
        "status": "INFERRED_PASSIVE_VISUAL_OWNER_CORRECTION_SOURCE_FRAME_PRESERVED",
        "stable_id": 12,
        "source_member": "FJ1737",
        "source_member_sha256": source_member_sha,
        "from": {"myosim_body": "torso", "core_body_index": 20},
        "to": {"myosim_body": "head", "core_body_index": 23},
        "source_face_order_unchanged": True,
        "vertex_slot_order_unchanged": True,
        "layer_unchanged": True,
        "source_identity_correction_applied_before_owner_migration": True,
        "inferred_from_reference_registration_not_subject_measurement": True,
        "native_motion_admission": False,
        "physical_mass_changed": False,
        "physiological_state_or_controller_changed": False,
        "default_pose_world_max_abs_delta_m": 1.5e-8,
        "legacy_default_pose_world_max_abs_error_m": 1.4e-8,
        "head_default_pose_world_max_abs_error_m": 2.0e-9,
        "registration_path": str(registration),
        "registration_sha256": file_sha(registration),
        "rigid_reference_manifest_path": str(rigid_manifest),
        "rigid_reference_manifest_sha256": file_sha(rigid_manifest),
    }
    correction = {
        "status": "SOURCE_HEADER_AUTHENTICATED_IDENTITY_CORRECTION",
        "source_member": "FJ1737",
        "source_member_sha256": frame["source_member_sha256"],
        "stable_id_preserved": 12,
        "embedded_obj_identity": {"concept_id": "FMA78497", "hierarchy": "part_of",
                                   "name": "Central canal of spinal cord"},
        "embedded_obj_license": "CC Attribution-Share Alike 2.1 Japan",
        "source_archive_full_sha256_independently_verified": False,
    }
    evidence_identity = {
        "body_index": 23, "layer": 3, "name": "Central canal of spinal cord",
        "source_member": "FJ1737", "source_sha256": frame["source_member_sha256"],
        "source_license": "CC Attribution-Share Alike 2.1 Japan",
        "source_owner_metadata": {
            "member_id": "FJ1737", "member_sha256": frame["source_member_sha256"],
            "myosim_body": "head", "core_body_index": 23,
            "inferred_passive_owner_correction": frame,
            "source_identity_correction": correction,
        },
    }
    migration_receipt = {
        "payload": {"sha256": migration_sha},
        "functional_bindings": {"anatomy_payload_sha256": migration_sha},
        "provenance": {
            "source_id_map": {"12": evidence_identity},
            "central_canal_owner_rebind": {
                "status": "INFERRED_PASSIVE_OWNER_REBIND_SOURCE_FRAME_PRESERVED_NOT_GEOMETRY_ADMISSION",
                "base_nha_sha256": base_sha, "candidate_nha_sha256": migration_sha,
                "physical_mass_or_state_changed": False, "geometry_admission": False,
                "owner_frame_audit": frame,
            },
        },
    }
    owner_migration = {"receipt_raw": json.dumps(migration_receipt).encode(),
                       "payload_raw": migration_raw,
                       "source_member_raw": source_member_raw}
    candidate_rows = copy.deepcopy(rows)
    candidate_rows[12]["body_index"] = 23
    candidate_rows[323]["vertices6"][0, 0] = 0.25
    candidate_rows[343]["vertices6"][0, 0] = 0.5
    return receipt, base_raw, header, candidate_rows, owner_migration


def bind_fixture_migration_pins(monkeypatch, base_raw, migration):
    proof_receipt = json.loads(migration["receipt_raw"])
    frame = proof_receipt["provenance"]["central_canal_owner_rebind"]["owner_frame_audit"]
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_MEMBER_SHA256",
                        hashlib.sha256(migration["source_member_raw"]).hexdigest())
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_BASE_NHA_SHA256",
                        hashlib.sha256(base_raw).hexdigest())
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_REBIND_NHA_SHA256",
                        hashlib.sha256(migration["payload_raw"]).hexdigest())
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_REBIND_RECEIPT_SHA256",
                        hashlib.sha256(migration["receipt_raw"]).hexdigest())
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_REGISTRATION_SHA256",
                        frame["registration_sha256"])
    monkeypatch.setattr(rebind, "_CENTRAL_CANAL_RIGID_MANIFEST_SHA256",
                        frame["rigid_reference_manifest_sha256"])


def test_owner_migration_can_precede_separate_registered_geometry_changes(tmp_path, monkeypatch):
    receipt, raw, header, rows, migration = central_canal_owner_migration_fixture(tmp_path)
    bind_fixture_migration_pins(monkeypatch, raw, migration)

    owner_result, owner_audit = rebind_passive_rows(
        receipt, raw, migration["payload_raw"], changed_ids=(12,),
        output_path="/owner-migrated.nhanatomy", owner_migrations={12: migration})
    assert owner_audit["changed_passive_stable_ids"] == [12]
    assert owner_audit["changed_owner_migration_stable_ids"] == [12]

    rows[12]["vertices6"][1, 0] = 0.75
    candidate = encode_payload(header, rows)
    final_result, geometry_audit = rebind_passive_rows(
        owner_result, migration["payload_raw"], candidate,
        changed_ids=(12, 323, 343), output_path="/final.nhanatomy")
    assert geometry_audit["changed_owner_migration_stable_ids"] == []
    assert geometry_audit["changed_passive_stable_ids"] == [12, 323, 343]
    assert final_result["provenance"]["source_id_map"]["12"]["body_index"] == 23


def test_rebind_accepts_only_source_bound_central_canal_owner_migration(tmp_path, monkeypatch):
    receipt, raw, header, rows, migration = central_canal_owner_migration_fixture(tmp_path)
    bind_fixture_migration_pins(monkeypatch, raw, migration)
    candidate = encode_payload(header, rows)
    result, audit = rebind_passive_rows(
        receipt, raw, candidate, changed_ids=(12, 323, 343), output_path="/candidate.nhanatomy",
        owner_migrations={12: migration})
    mapped = result["provenance"]["source_id_map"]["12"]
    assert mapped["body_index"] == 23
    assert mapped["source_owner_metadata"]["myosim_body"] == "head"
    assert audit["changed_owner_migration_stable_ids"] == [12]
    assert result["provenance"]["verified_passive_owner_migrations"]["12"][
        "native_motion_admission"] is False


@pytest.mark.parametrize("mutation", ["other_id", "wrong_destination", "layer", "flags",
                                      "source_member", "source_hash", "source_bytes"])
def test_rebind_rejects_unproven_owner_migration_changes(tmp_path, monkeypatch, mutation):
    receipt, raw, header, rows, migration = central_canal_owner_migration_fixture(tmp_path)
    bind_fixture_migration_pins(monkeypatch, raw, migration)
    migrations = {12: migration}
    if mutation == "other_id":
        rows[323]["body_index"] = 20
        migrations = {323: migration}
    elif mutation == "wrong_destination":
        rows[12]["body_index"] = 22
    elif mutation == "layer":
        rows[12]["layer"] = 1
    elif mutation == "flags":
        rows[12]["flags"] = 1
    else:
        if mutation == "source_member":
            receipt["provenance"]["source_id_map"]["12"]["source_member"] = "FJ1769"
        elif mutation == "source_hash":
            receipt["provenance"]["source_id_map"]["12"]["source_sha256"] = "0" * 64
        else:
            migration["source_member_raw"] = b"different OBJ bytes"
    with pytest.raises(ValueError):
        rebind_passive_rows(
            receipt, raw, encode_payload(header, rows), changed_ids=(12, 323, 343),
            output_path="/candidate.nhanatomy", owner_migrations=migrations)

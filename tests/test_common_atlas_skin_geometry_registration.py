import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from numilab_human import model
from numilab_human.common_atlas_skin_geometry_registration import derive_common_atlas_skin_geometry
from numilab_human.skin_source_payload_preflight import decode_payload


def _matrix(translation):
    return [
        [0.001, 0.0, 0.0, translation[0]],
        [0.0, 0.001, 0.0, translation[1]],
        [0.0, 0.0, 0.001, translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _fixture(tmp_path, monkeypatch):
    archive = tmp_path / "Sources" / "isa_BP3D_4.0_obj_99.zip"
    archive.parent.mkdir()
    obj = b"v 0 0 0\nv 100 0 0\nv 0 100 0\nf 1 2 3\n"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("FJ2810.obj", obj)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    registration_path = tmp_path / "registration.json"
    registration = {
        "schema": "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
        "source": {
            "bodyparts": {"id": "bodyparts3d_4", "version": "4.0", "archives": [
                {"file": archive.name, "hierarchy": "is_a", "sha256": archive_sha},
            ]},
            "myosim": {"source": {"archive_sha256": "ab" * 32}},
        },
        "coordinate_system": {"global_source_mm_to_myosim_world_m": _matrix([0.0, 0.0, 0.0])},
        "anchors": [],
    }
    runtime = {}
    for index in range(86):
        name, core, source_id = f"owner{index}", index, 1000 + index
        runtime[name] = (
            core,
            {"source_body_id": source_id, "position_world_m": [0.0, 0.0, 0.0], "rotation_world": np.eye(3).tolist()},
        )
        matrix = _matrix([0.01 if index == 1 else 0.0, 0.0, 0.0])
        registration["anchors"].append({
            "source": {"member_id": f"FJ{source_id}"},
            "target": {"name": name, "core_body_index": core, "source_body_id": source_id},
            "registration": {"source_obj_mm_to_core_inertial_body_m": matrix},
        })
    reg_raw = json.dumps(registration, separators=(",", ":")).encode()
    registration_path.write_bytes(reg_raw)
    registration_sha = hashlib.sha256(reg_raw).hexdigest()
    monkeypatch.setattr(model, "_bodyparts_runtime_bindings", lambda *_: (
        {"rigid": {"sha256": "fixture"}}, runtime,
    ))

    source_payload = tmp_path / "source.nhskin"
    binding_records = b"".join(
        struct.pack("<I8f", body_id, 0, 0, 0, 0, 0, 0, 1, 1)
        for body_id in range(86)
    )
    vertices = np.zeros((4, 14), dtype="<f4")
    vertices[:, :3] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.7, 0.8, 0.9]]
    vertices[:, 3:6] = [0.0, 0.0, 1.0]
    vertices[:, 6:10] = [[0, 1, 0, 0]]
    vertices[:, 10:14] = [[1.0 / 86.0, 1.0 / 86.0, 0, 0]]
    indices = np.array([0, 1, 2], dtype="<u4")
    weights = np.full((4, 86), 1.0 / 86.0, dtype="<f4")
    header = struct.pack("<8s5I32s", b"NHSKIN1" + bytes([0]), 5, 86, 4, 3, int(registration_sha[:8], 16), bytes.fromhex("ab" * 32))
    raw = header + binding_records + vertices.tobytes() + indices.tobytes() + weights.tobytes()
    source_payload.write_bytes(raw)
    raw_skin = tmp_path / "raw-source.nhskin"
    raw_skin.write_bytes(b"raw source identity")
    provenance = {
        "source_skin": {"path": str(raw_skin), "sha256": hashlib.sha256(raw_skin.read_bytes()).hexdigest()},
        "derived_skin": {"path": str(source_payload), "sha256": hashlib.sha256(raw).hexdigest()},
    }
    provenance_path = tmp_path / "source-receipt.json"
    provenance_path.write_text(json.dumps(provenance))
    return source_payload, registration_path, archive.parent, provenance_path, runtime, registration, registration_sha, raw


def test_geometry_bake_preserves_shared_atlas_binds_weights_and_topology(tmp_path, monkeypatch):
    source_payload, registration_path, sources, provenance_path, runtime, registration, registration_sha, raw = _fixture(tmp_path, monkeypatch)
    manifest = derive_common_atlas_skin_geometry(
        source_payload=source_payload,
        registration_path=registration_path,
        sources=sources,
        myosim_artifact=tmp_path / "unused-myosim",
        input_provenance_path=provenance_path,
        output_directory=tmp_path / "candidate",
    )
    candidate_path = tmp_path / "candidate" / source_payload.name
    candidate = candidate_path.read_bytes()
    before, after = decode_payload(raw), decode_payload(candidate)
    assert manifest["schema"] == "numi.human.common-atlas-skin-geometry-registration.v1"
    assert manifest["status"] == "inferred_common_atlas_geometry_candidate_pending_native_clearance_and_pose_checks"
    assert np.array_equal(before["bindings_u"], after["bindings_u"])
    assert np.array_equal(before["full_weights"], after["full_weights"])
    assert np.array_equal(before["indices"], after["indices"])
    assert np.array_equal(before["vertices_u"][3], after["vertices_u"][3])
    assert np.array_equal(before["vertices_u"][:, 6:14], after["vertices_u"][:, 6:14])
    assert not np.array_equal(before["vertices_u"][:3, :3], after["vertices_u"][:3, :3])
    np.testing.assert_allclose(after["vertices_f"][0, :3], [0.01 / 86.0, 0, 0], atol=1e-7, rtol=0)
    assert manifest["preservation"]["all_86_canonical_binding_records_byte_identical"]
    assert manifest["preservation"]["full_weight_matrix_byte_identical"]
    assert manifest["inputs"]["raw_FJ2810"]["member_id"] == "FJ2810"
    assert "not measured" in manifest["inputs"]["raw_FJ2810"]["interpretation"]
    assert manifest["qualification"]["registered_skeleton_and_organ_clearance"] == "pending"


def test_geometry_bake_rejects_canonical_binding_outside_global_rest_atlas(tmp_path, monkeypatch):
    source_payload, registration_path, sources, provenance_path, runtime, registration, _, raw = _fixture(tmp_path, monkeypatch)
    mutated = bytearray(raw)
    struct.pack_into("<f", mutated, 60 + 7 * 36 + 4, 0.003)
    source_payload.write_bytes(mutated)
    provenance = json.loads(provenance_path.read_text())
    provenance["derived_skin"]["sha256"] = hashlib.sha256(mutated).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with np.testing.assert_raises_regex(model.ImportError, "canonical NHSKIN shared-atlas rest transform differs"):
        derive_common_atlas_skin_geometry(
            source_payload=source_payload,
            registration_path=registration_path,
            sources=sources,
            myosim_artifact=tmp_path / "unused-myosim",
            input_provenance_path=provenance_path,
            output_directory=tmp_path / "rejected-binding",
        )
    assert not (tmp_path / "rejected-binding").exists()


def test_geometry_bake_rejects_nonshared_transform_anchors(tmp_path, monkeypatch):
    source_payload, registration_path, sources, provenance_path, runtime, registration, _, _ = _fixture(tmp_path, monkeypatch)
    registration["anchors"].append({
        "source": {"member_id": "FJ1001M"},
        "target": {"name": "owner1", "core_body_index": 1, "source_body_id": 1001},
        "registration": {"source_obj_mm_to_core_inertial_body_m": _matrix([0.012, 0, 0])},
    })
    registration_raw = json.dumps(registration).encode()
    registration_path.write_bytes(registration_raw)
    registration_sha = hashlib.sha256(registration_raw).hexdigest()
    raw = bytearray(source_payload.read_bytes())
    struct.pack_into("<I", raw, 24, int(registration_sha[:8], 16))
    source_payload.write_bytes(raw)
    provenance = json.loads(provenance_path.read_text())
    provenance["derived_skin"]["sha256"] = hashlib.sha256(raw).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with np.testing.assert_raises_regex(model.ImportError, "do not share one owner transform"):
        derive_common_atlas_skin_geometry(
            source_payload=source_payload,
            registration_path=registration_path,
            sources=sources,
            myosim_artifact=tmp_path / "unused-myosim",
            input_provenance_path=provenance_path,
            output_directory=tmp_path / "rejected",
        )


def test_retained_004_geometry_template_restores_canonical_binding_records(tmp_path, monkeypatch):
    source_payload, registration_path, sources, provenance_path, runtime, registration, registration_sha, raw = _fixture(tmp_path, monkeypatch)
    direct = derive_common_atlas_skin_geometry(
        source_payload=source_payload,
        registration_path=registration_path,
        sources=sources,
        myosim_artifact=tmp_path / "unused-myosim",
        input_provenance_path=provenance_path,
        output_directory=tmp_path / "direct",
    )
    legacy = bytearray(raw)
    struct.pack_into("<f", legacy, 60 + 1 * 36 + 4, 0.012)
    legacy_path = tmp_path / "retained-004-template.nhskin"
    legacy_path.write_bytes(legacy)
    legacy_manifest_path = tmp_path / "retained-004-manifest.json"
    legacy_manifest_path.write_text(json.dumps({
        "schema": "numi.human.skin-lower-limb-anchor-rebind-candidate.v1",
        "status": "source_identity_preserved_visual_skin_candidate_pending_native_geometry_audit",
        "output_payload": {"path": str(legacy_path), "sha256": hashlib.sha256(legacy).hexdigest()},
    }))
    manifest = derive_common_atlas_skin_geometry(
        source_payload=legacy_path,
        registration_path=registration_path,
        sources=sources,
        myosim_artifact=tmp_path / "unused-myosim",
        input_provenance_path=legacy_manifest_path,
        canonical_binding_reference=source_payload,
        canonical_binding_provenance_path=provenance_path,
        output_directory=tmp_path / "restored",
    )
    restored = Path(manifest["output_payload"]["path"]).read_bytes()
    direct_bytes = Path(direct["output_payload"]["path"]).read_bytes()
    assert restored == direct_bytes
    assert manifest["inputs"]["source_payload"]["route"] == "retained_004_invalid_binding_candidate_geometry_template_only"
    assert manifest["inputs"]["canonical_binding_reference"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["preservation"]["source_binding_rows_restored_to_canonical"] is True
    assert manifest["preservation"]["all_86_canonical_binding_records_byte_identical"] is True

    bad = bytearray(legacy)
    struct.pack_into("<f", bad, 60 + 86 * 36, 0.003)
    bad_path = tmp_path / "bad-004-template.nhskin"
    bad_path.write_bytes(bad)
    bad_manifest = json.loads(legacy_manifest_path.read_text())
    bad_manifest["output_payload"] = {"path": str(bad_path), "sha256": hashlib.sha256(bad).hexdigest()}
    bad_manifest_path = tmp_path / "bad-004-manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest))
    with np.testing.assert_raises_regex(model.ImportError, "differs from canonical source in positions"):
        derive_common_atlas_skin_geometry(
            source_payload=bad_path,
            registration_path=registration_path,
            sources=sources,
            myosim_artifact=tmp_path / "unused-myosim",
            input_provenance_path=bad_manifest_path,
            canonical_binding_reference=source_payload,
            canonical_binding_provenance_path=provenance_path,
            output_directory=tmp_path / "bad-output",
        )
    assert not (tmp_path / "bad-output").exists()


def test_common_atlas_accepts_canonical_bindings_and_rejects_004_substitution(tmp_path, monkeypatch):
    from numilab_human.resting_anatomy import _verify_common_atlas_skin_binding_records

    source, registration_path, sources, provenance_path, runtime, registration, registration_sha, raw = _fixture(tmp_path, monkeypatch)
    canonical_sha = hashlib.sha256(raw).hexdigest()
    accepted = _verify_common_atlas_skin_binding_records(raw, source, canonical_sha, registration)
    assert accepted["binding_record_bytes_exact"] is True
    assert accepted["full_weight_matrix_exact"] is True
    assert accepted["registered_owner_id_set_exact"] is True
    assert "not required or asserted" in accepted["owner_transform_equality_to_NHTISS4"]

    substituted = bytearray(raw)
    struct.pack_into("<f", substituted, 60 + 1 * 36 + 4, 0.012)
    with np.testing.assert_raises_regex(model.ImportError, "differ from canonical shared-atlas records"):
        _verify_common_atlas_skin_binding_records(bytes(substituted), source, canonical_sha, registration)

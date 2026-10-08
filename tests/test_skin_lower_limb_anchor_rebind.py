import hashlib
import json
import struct

import numpy as np
import pytest

from numilab_human import model
from numilab_human.skin_lower_limb_anchor_rebind import (
    LOWER_LIMB_SCHEMA,
    REGISTRATION_SCHEMA,
    TARGET_BODY_NAMES,
    _derive_candidate_bytes,
    _verify_nhtiss_owner_alignment,
    _verify_owner_mismatch_report,
)
from numilab_human.skin_source_payload_preflight import decode_payload


def _matrix(translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= 0.001
    matrix[:3, 3] = translation
    return matrix.tolist()


def _fixture():
    body_ids = list(range(100, 112)) + [999]
    binding_count = len(body_ids)
    vertex_count = 4
    source_hash = "ab" * 32
    registration_sha = "12" * 32

    binding_bytes = b"".join(
        struct.pack("<I8f", body_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        for body_id in body_ids
    )

    vertices = np.zeros((vertex_count, 14), dtype="<f4")
    vertices[:, :3] = [
        [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4],
    ]
    vertices[:, 3:6] = [[0.0, 0.0, 1.0]] * vertex_count
    # The fourth payload vertex is intentionally unindexed and its normal is
    # finite but not unit length, so accidental normalization is detectable.
    vertices[3, 3:6] = [0.123, 0.456, 0.789]
    vertices[:, 6:10] = [[1.0, 0.0, 0.0, 0.0]] * vertex_count
    vertices[:, 10:14] = [[0.0, 1.0, 0.0, 0.0]] * vertex_count

    indices = np.array([0, 1, 2], dtype="<u4")
    weights = np.zeros((vertex_count, binding_count), dtype="<f4")
    weights[0, [0, 6]] = [0.5, 0.5]
    weights[1, [2, 6]] = [0.25, 0.75]
    weights[2, [4, 5]] = [0.5, 0.5]
    weights[3, [0, 6]] = [0.3, 0.7]

    fingerprint = int(registration_sha[:8], 16)
    header = struct.pack(
        "<8s5I32s", b"NHSKIN1\0", 5, binding_count, vertex_count,
        len(indices), fingerprint, bytes.fromhex(source_hash),
    )
    raw = (header + binding_bytes + vertices.tobytes()
           + indices.tobytes() + weights.tobytes())

    anchors = []
    runtime_bodies = {}
    for index, name in enumerate(TARGET_BODY_NAMES):
        body_id = body_ids[index]
        source_id = index + 5000
        translation = [0.01 * (index + 1), -0.002 * index, 0.003 * index]
        anchors.append({
            "source": {"member_id": f"FJ{source_id}"},
            "target": {
                "name": name,
                "core_body_index": body_id,
                "source_body_id": source_id,
            },
            "registration": {
                "source_obj_mm_to_core_inertial_body_m": _matrix(translation),
            },
        })
        runtime_bodies[name] = (
            body_id,
            {
                "source_body_id": source_id,
                "position_world_m": [0.0, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
            },
        )
    pelvis_id, pelvis_source_id = body_ids[12], 9000
    anchors.append({
        "source": {"member_id": "FJ9000"},
        "target": {"name": "pelvis", "core_body_index": pelvis_id,
                   "source_body_id": pelvis_source_id},
        "registration": {
            "source_obj_mm_to_core_inertial_body_m": _matrix([0.0, 0.0, 0.0]),
        },
    })
    runtime_bodies["pelvis"] = (
        pelvis_id,
        {
            "source_body_id": pelvis_source_id,
            "position_world_m": [0.0, 0.0, 0.0],
            "rotation_world": np.eye(3).tolist(),
        },
    )
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "coordinate_system": {
            "global_source_mm_to_myosim_world_m": _matrix([0.0, 0.0, 0.0]),
        },
        "source": {
            "bodyparts": {"source": "bodyparts-fixture"},
            "myosim": {"source": {"archive_sha256": source_hash}},
        },
        "lower_limb_source_mesh_registration": {
            "schema": LOWER_LIMB_SCHEMA,
            "status": "candidate_passed_bilateral_source_mesh_and_default_pose_continuity_gates",
        },
        "anchors": anchors,
    }
    return raw, registration, registration_sha, runtime_bodies


def test_rebind_changes_only_registered_foot_binding_records():
    raw, registration, registration_sha, runtime_bodies = _fixture()
    candidate, manifest = _derive_candidate_bytes(
        raw, registration, registration_sha, {"reference": "fixture"}, runtime_bodies,
    )
    before = decode_payload(raw)
    after = decode_payload(candidate)

    assert candidate != raw
    assert len(candidate) == len(raw)
    assert np.array_equal(before["vertices_u"][:, :3], after["vertices_u"][:, :3])
    assert np.array_equal(before["vertices_u"][:, 6:14], after["vertices_u"][:, 6:14])
    assert np.array_equal(before["indices"], after["indices"])
    assert np.array_equal(before["full_weights"], after["full_weights"])
    assert np.array_equal(before["vertices_u"][3, 3:6], after["vertices_u"][3, 3:6])
    assert manifest["changed_skin_regions"]["combined_geometry"]["normal_recomputation"][
        "unreferenced_input_normal_bytes_preserved"
    ]
    assert np.array_equal(before["bindings_u"][12], after["bindings_u"][12])
    assert [row["myosim_body"] for row in manifest["changed_skin_regions"]["bindings"]] == list(
        TARGET_BODY_NAMES
    )
    assert manifest["preservation"]["source_position_coordinates_byte_identical"]
    assert manifest["preservation"]["rest_world_normals_recomputed_from_candidate_default_pose"]
    assert manifest["preservation"]["full_86_body_weight_matrix_byte_identical"]

    changed = {
        index for index in range(before["binding_count"])
        if not np.array_equal(before["bindings_u"][index], after["bindings_u"][index])
    }
    assert changed == set(range(12))
    for index, name in enumerate(TARGET_BODY_NAMES):
        translation = np.asarray(after["bindings_f"][index, 1:4], dtype=float)
        expected = np.array([0.01 * (index + 1), -0.002 * index, 0.003 * index])
        np.testing.assert_allclose(translation, expected, rtol=0.0, atol=1e-7)
    assert manifest["preservation"]["non_target_binding_record_count"] == 1
    assert manifest["changed_skin_regions"]["combined_geometry"]["dominant_anatomical_region_breakdown"]


def test_rebind_rejects_inconsistent_duplicate_source_anchor_for_same_body():
    raw, registration, registration_sha, runtime_bodies = _fixture()
    duplicate = dict(registration["anchors"][TARGET_BODY_NAMES.index("tibia_r")])
    duplicate["source"] = {"member_id": "FJ9999"}
    duplicate["registration"] = {
        "source_obj_mm_to_core_inertial_body_m": _matrix([0.9, 0.0, 0.0])
    }
    registration["anchors"].append(duplicate)

    with pytest.raises(model.ImportError, match="do not share one exact"):
        _derive_candidate_bytes(
            raw, registration, registration_sha, {"reference": "fixture"}, runtime_bodies,
        )


def test_rebind_rejects_missing_named_foot_anchor():
    raw, registration, registration_sha, runtime_bodies = _fixture()
    registration["anchors"] = [
        anchor for anchor in registration["anchors"]
        if anchor["target"]["name"] != "toes_l"
    ]

    with pytest.raises(model.ImportError, match="no exact registration anchor for toes_l"):
        _derive_candidate_bytes(
            raw, registration, registration_sha, {"reference": "fixture"}, runtime_bodies,
        )

def _nhtiss_fixture(tmp_path, candidate, registration, registration_sha, *, corrupt=False):
    decoded = decode_payload(candidate)
    source_sha = decoded["source_archive_sha256"]
    binding_records = []
    for index, name in enumerate(TARGET_BODY_NAMES):
        if name in {"talus_r", "talus_l"}:
            continue
        begin = 60 + index * 36
        record = bytearray(candidate[begin:begin + 36])
        if corrupt and name == "femur_r":
            struct.pack_into("<f", record, 4, struct.unpack_from("<f", record, 4)[0] + 0.001)
        binding_records.append(bytes(record))
    fingerprint = decoded["registration_fingerprint32"]
    raw = b"".join([
        struct.pack(
            "<8s6I32s", b"NHTISS4" + bytes([0]), 5, 1, len(binding_records),
            1, 0, fingerprint, bytes.fromhex(source_sha),
        ),
        struct.pack("<8I", 0, len(binding_records), 0, 1, 0, 0, 1, 1),
        *binding_records,
        bytes(56),
    ])
    payload_path = tmp_path / "tissue.nhtissue"
    manifest_path = tmp_path / "tissue.manifest.json"
    payload_path.write_bytes(raw)
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
        "payload": {
            "file": payload_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "registration_fingerprint32": f"{fingerprint:08x}",
            "surface_count": 1,
            "binding_count": len(binding_records),
            "vertex_count": 1,
            "index_count": 0,
            "bytes": len(raw),
        },
        "source": {
            "registration": {"sha256": registration_sha},
            "myosim_source_archive_sha256": source_sha,
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    return payload_path, manifest_path


def test_rebind_matches_registered_nhtiss_records_and_reports_absent_talus_routes(tmp_path):
    raw, registration, registration_sha, runtime_bodies = _fixture()
    candidate, _ = _derive_candidate_bytes(
        raw, registration, registration_sha, {"reference": "fixture"}, runtime_bodies,
    )
    tissue_payload, tissue_manifest = _nhtiss_fixture(
        tmp_path, candidate, registration, registration_sha,
    )
    result = _verify_nhtiss_owner_alignment(
        tissue_payload_path=tissue_payload,
        tissue_manifest_path=tissue_manifest,
        registration_sha256=registration_sha,
        source_archive_sha256="ab" * 32,
        registration_fingerprint32=int(registration_sha[:8], 16),
        candidate_skin=candidate,
        registration=registration,
    )
    assert result["present_lower_limb_owner_records_byte_exact"]
    assert sum(row["nhtiss4_binding_record_count"] > 0 for row in result["owner_checks"]) == 10
    assert sum(row["nhtiss4_has_no_route_surface_for_owner"] for row in result["owner_checks"]) == 2


def test_rebind_rejects_nhtiss_owner_transform_mismatch(tmp_path):
    raw, registration, registration_sha, runtime_bodies = _fixture()
    candidate, _ = _derive_candidate_bytes(
        raw, registration, registration_sha, {"reference": "fixture"}, runtime_bodies,
    )
    tissue_payload, tissue_manifest = _nhtiss_fixture(
        tmp_path, candidate, registration, registration_sha, corrupt=True,
    )
    with pytest.raises(model.ImportError, match="differs byte-for-byte from registered NHTISS4"):
        _verify_nhtiss_owner_alignment(
            tissue_payload_path=tissue_payload,
            tissue_manifest_path=tissue_manifest,
            registration_sha256=registration_sha,
            source_archive_sha256="ab" * 32,
            registration_fingerprint32=int(registration_sha[:8], 16),
            candidate_skin=candidate,
            registration=registration,
        )

def test_owner_comparison_report_pins_exact_12_vs_74_partition(tmp_path):
    raw, registration, registration_sha, _ = _fixture()
    skin_path = tmp_path / "skin.nhskin"
    registration_path = tmp_path / "registration.json"
    bone_path = tmp_path / "bones.nhbones"
    bone_manifest_path = tmp_path / "bones.manifest.json"
    report_path = tmp_path / "owner-comparison.json"
    skin_path.write_bytes(raw)
    registration_path.write_text(json.dumps(registration))
    registration_sha = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    bone_path.write_bytes(b"bones fixture")
    bone_sha = hashlib.sha256(bone_path.read_bytes()).hexdigest()
    bone_manifest = {
        "source": {"registration": {"sha256": registration_sha}},
        "payload": {"sha256": bone_sha},
    }
    bone_manifest_path.write_text(json.dumps(bone_manifest))
    bone_manifest_sha = hashlib.sha256(bone_manifest_path.read_bytes()).hexdigest()
    expected = {
        anchor["target"]["name"]: anchor["target"]["core_body_index"]
        for anchor in registration["anchors"]
        if anchor["target"]["name"] in TARGET_BODY_NAMES
    }
    report = {
        "inputs": {
            "skin_payload": {"path": str(skin_path), "sha256": hashlib.sha256(raw).hexdigest()},
            "bone_payload": {"path": str(bone_path), "sha256": bone_sha},
            "bone_manifest": {
                "path": str(bone_manifest_path),
                "sha256": bone_manifest_sha,
                "full_registration_file_sha256": registration_sha,
            },
        },
        "coverage": {
            "skin_owner_count": 86,
            "bone_owner_count": 86,
            "owners_with_skin_bone_transform_mismatch": 12,
            "owners_with_matching_transforms": 74,
            "unmatched_skin_core_owners": [],
            "unmatched_bone_core_owners": [],
            "owners_with_multiple_distinct_bone_transform_tuples": 0,
        },
        "mismatched_owners": [
            {"body": name, "core_body_index": index}
            for name, index in expected.items()
        ],
    }
    report_path.write_text(json.dumps(report))
    result = _verify_owner_mismatch_report(
        report_path=report_path,
        source_payload_path=skin_path,
        registration_path=registration_path,
        source_payload_sha256=hashlib.sha256(raw).hexdigest(),
        registration_sha256=registration_sha,
        registration=registration,
    )
    assert result["mismatched_owner_count"] == 12
    assert result["matching_owner_count"] == 74
    report["coverage"]["owners_with_matching_transforms"] = 73
    report_path.write_text(json.dumps(report))
    with pytest.raises(model.ImportError, match="12/74 owner partition"):
        _verify_owner_mismatch_report(
            report_path=report_path,
            source_payload_path=skin_path,
            registration_path=registration_path,
            source_payload_sha256=hashlib.sha256(raw).hexdigest(),
            registration_sha256=registration_sha,
            registration=registration,
        )


def test_public_per_owner_rebind_is_deprecated_before_writing(tmp_path):
    from numilab_human.skin_lower_limb_anchor_rebind import rebind_registered_lower_limb_skin_payload

    output = tmp_path / "candidate"
    with pytest.raises(model.ImportError, match="shared atlas"):
        rebind_registered_lower_limb_skin_payload(
            source_payload=tmp_path / "source.nhskin",
            registration_path=tmp_path / "registration.json",
            myosim_artifact=tmp_path / "myosim",
            output_directory=output,
            tissue_payload_path=tmp_path / "tissue.nhtissue",
            tissue_manifest_path=tmp_path / "tissue.json",
            owner_comparison_report_path=tmp_path / "owners.json",
        )
    assert not output.exists()


def test_canonical_binding_constructor_preserves_one_shared_source_atlas():
    global_matrix = _matrix([0.2, -0.1, 0.4])
    point_m = np.array([0.31, -0.22, 0.08])
    global_translation, global_quaternion, global_scale = model._bodyparts_visual_local_pose(
        global_matrix, "fixture common atlas",
    )
    global_rotation = np.asarray(model._myosim_matrix_from_quaternion_xyzw(global_quaternion))
    expected_world = np.asarray(global_translation) + global_scale * (global_rotation @ point_m)
    poses = [
        ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0]),
        ([-0.4, 0.7, 0.2], [0.0, 0.0, np.sin(0.31), np.cos(0.31)]),
    ]
    for position, quaternion in poses:
        local = model._bodyparts_local_registration_matrix(
            global_matrix, position, quaternion,
        )
        local_translation, local_quaternion, local_scale = model._bodyparts_visual_local_pose(
            local, "fixture canonical owner binding",
        )
        body_rotation = np.asarray(model._myosim_matrix_from_quaternion_xyzw(quaternion))
        local_rotation = np.asarray(model._myosim_matrix_from_quaternion_xyzw(local_quaternion))
        rendered = np.asarray(position) + body_rotation @ (
            np.asarray(local_translation) + local_scale * (local_rotation @ point_m)
        )
        np.testing.assert_allclose(rendered, expected_world, rtol=0.0, atol=2e-12)

    per_bone = _matrix([0.235, -0.1, 0.4])
    per_bone_translation, per_bone_quaternion, per_bone_scale = model._bodyparts_visual_local_pose(
        per_bone, "fixture individual bone transform",
    )
    altered = np.asarray(per_bone_translation) + per_bone_scale * (
        np.asarray(model._myosim_matrix_from_quaternion_xyzw(per_bone_quaternion)) @ point_m
    )
    assert np.linalg.norm(altered - expected_world) > 0.03

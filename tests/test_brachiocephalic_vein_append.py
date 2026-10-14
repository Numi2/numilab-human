from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from numilab_human import resting_major_vessels
from numilab_human import resting_passive_viscera


def test_brachiocephalic_quotient_preserves_source_faces_and_exact_seam():
    local = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    source_mm = local.copy()
    faces = np.array([
        [0, 2, 1],
        [4, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int64)

    quotient = resting_passive_viscera._exact_float32_seam_quotient(
        local, faces, source_mm, "FJ-test")

    assert quotient["vertices"].shape == (4, 6)
    assert quotient["faces"].tolist() == [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    assert quotient["source_vertex_to_quotient"].tolist() == [0, 1, 2, 3, 0]
    assert quotient["source_face_ids"].tolist() == [0, 1, 2, 3]
    assert np.array_equal(quotient["vertices"][:, :3], local[:4].astype("<f4"))


def test_brachiocephalic_quotient_rejects_distinct_source_points_merged_by_float32():
    source_mm = np.array([
        [0.0, 0.0, 0.0],
        [1.0e-50, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    local = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 2, 3]], dtype=np.int64)

    with pytest.raises(ValueError, match="merge distinct source coordinates"):
        resting_passive_viscera._exact_float32_seam_quotient(
            local, faces, source_mm, "FJ-test")


def test_torso_context_comes_from_receipt_pinned_rigid_source_pose():
    source_sha = "b" * 64
    rigid_sha = "a" * 64
    pose = np.array([-0.02500000037, 0.19939999282, 1.35720002651,
                     0.5, -0.5, -0.5, 0.5], dtype=np.float64)
    source_map = list(range(103))
    source_map[8] = 20
    source_map[20] = 38
    receipt = {"provenance": {
        "rigid_payload_sha256": rigid_sha,
        "myosim_source_archive_sha256": source_sha,
        "torso_rest_position_world_m": pose[:3].tolist(),
        "torso_rest_orientation_xyzw": pose[3:].tolist(),
        "source_id_map": {"10": {"body_index": 20, "source_owner_metadata": {
            "core_body_index": 20, "myosim_body": "torso"}}},
    }}
    registration = {"source": {"myosim": {"source": {"archive_sha256": source_sha}}}}
    rigid = {"source_sha256": source_sha, "source_count": 103, "source_map": source_map,
             "core_poses": {20: pose}}

    with patch.object(resting_passive_viscera.human, "sha256", return_value=rigid_sha),          patch.object(resting_passive_viscera, "load_rigid", return_value=rigid):
        bodies = resting_passive_viscera._validated_torso_rest_body(
            receipt, registration, Path("/tmp/pinned.nhrigid"))

    assert bodies == {"torso": {
        "core_body_index": 20, "source_body_id": 8,
        "default_com_position_world_m": pose[:3].tolist(),
        "default_inertial_quaternion_world_xyzw": pose[3:].tolist(),
    }}
    bad_rigid = dict(rigid, source_map=source_map.copy())
    bad_rigid["source_map"][20] = 20
    with patch.object(resting_passive_viscera.human, "sha256", return_value=rigid_sha),          patch.object(resting_passive_viscera, "load_rigid", return_value=bad_rigid):
        with pytest.raises(ValueError, match="source-body 8 to core-body 20"):
            resting_passive_viscera._validated_torso_rest_body(
                receipt, registration, Path("/tmp/pinned.nhrigid"))


def test_quotient_report_keeps_historical_parent_but_accepts_later_source_parent(tmp_path):
    historical = tmp_path / "060-producer-parent"
    current = tmp_path / "later-repaired-parent"
    sources = tmp_path / "sources"
    rigid = tmp_path / "pinned.nhrigid"
    pins = {
        str((historical / "receipt.json").resolve()): "1" * 64,
        str((historical / "resting-thorax.nhanatomy").resolve()): "2" * 64,
        str(rigid.resolve()): "3" * 64,
        str((sources / "partof_BP3D_4.0_obj_99.zip").resolve()): "4" * 64,
    }

    resting_passive_viscera._require_brachiocephalic_source_coverage(
        pins, rigid_path=rigid, sources_path=sources)

    pinned_paths = {Path(name) for name in pins}
    assert (current / "receipt.json").resolve() not in pinned_paths
    assert (current / "resting-thorax.nhanatomy").resolve() not in pinned_paths
    with pytest.raises(ValueError, match="rigid/archive inputs"):
        resting_passive_viscera._require_brachiocephalic_source_coverage(
            {name: digest for name, digest in pins.items() if name != str(rigid.resolve())},
            rigid_path=rigid, sources_path=sources)


def test_public_append_wrapper_requires_explicit_sources_and_binds_existing_compartment():
    paths = {
        "base_receipt": Path("/tmp/base-receipt.json"),
        "output": Path("/tmp/output"),
        "sources_path": Path("/tmp/sources"),
        "registration_path": Path("/tmp/registration.json"),
        "rigid_path": Path("/tmp/torso.nhrigid"),
        "quotient_npz_path": Path("/tmp/rows.npz"),
        "quotient_report_path": Path("/tmp/report.json"),
    }
    with patch.object(resting_major_vessels, "append_viscera", return_value={"ok": True}) as append:
        result = resting_major_vessels.append_brachiocephalic_veins(**paths)

    assert result == {"ok": True}
    args, kwargs = append.call_args
    assert args == (paths["base_receipt"], paths["output"])
    assert kwargs["families"] == (
        ("FMA4761", "left brachiocephalic vein", "torso", 1),
        ("FMA4751", "right brachiocephalic vein", "torso", 1),
    )
    assert kwargs["hierarchy"] == "part_of"
    assert kwargs["layer"] == 2
    assert kwargs["extension_key"] == "passive_brachiocephalic_vein_extension"
    assert kwargs["compartment_by_name"] == {
        "left brachiocephalic vein": 4,
        "right brachiocephalic vein": 4,
    }
    assert kwargs["members_by_name"] == {
        "left brachiocephalic vein": {"FJ3482"},
        "right brachiocephalic vein": {"FJ3583"},
    }
    assert kwargs["sources_path"] == paths["sources_path"]
    assert kwargs["registration_path"] == paths["registration_path"]
    assert kwargs["rigid_path"] == paths["rigid_path"]
    assert kwargs["quotient_npz_path"] == paths["quotient_npz_path"]
    assert kwargs["quotient_report_path"] == paths["quotient_report_path"]

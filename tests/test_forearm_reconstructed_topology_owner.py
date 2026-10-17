"""Reconstructed topology must not inherit captured-index ownership."""
import copy
import hashlib
import json

import numpy as np
import pytest

from numilab_human.forearm_actual_capture_owner import build_mixed_full21_candidate_owner_record
from numilab_human.resting_anatomy_verify import prepare_surface
from tests.test_resting_anatomy_verify import FACES, mixed_candidate_feature_surface, tetra


def fixture(tmp_path):
    positions = tetra()
    gids = np.arange(4, dtype="<u8") + 100
    slots = np.arange(4, dtype="<i8")
    surface = mixed_candidate_feature_surface((51005, 134), positions, FACES, gids, slots)
    proof = surface.core_mixed_candidate_owner_proof
    arrays = {k: v.copy() for k, v in proof["candidate_arrays"].items()}
    order = [1, 0, 2, 3]
    arrays["faces_u32"] = arrays["faces_u32"][order]
    lineage = [surface.source_face_lineage[i] for i in order]
    archive = tmp_path / "candidate.npz"
    names = {"gids_u64": "canonical_gids_u64"}
    saved = {"row134_" + names.get(k, k): v for k, v in arrays.items()}
    saved["row134_source_face_slots_u32"] = slots[arrays["faces_u32"]].astype("<u4")
    np.savez(archive, **saved)
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps({"rows": {"134": {"source_face_lineage": lineage}}}))
    binding = {
        "archive_path": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "array_prefix": "row134_", "lineage_path": str(lineage_path),
        "lineage_sha256": hashlib.sha256(lineage_path.read_bytes()).hexdigest(),
        "typed_identity": [51005, 134],
    }
    row = {
        "typed_identity": [51005, 134],
        "actual_capture_positions_f32": proof["actual_capture_positions_f32"],
        "actual_capture_faces_u32": proof["actual_capture_faces_u32"],
        "index_patch_faces_capture_u32": proof["candidate_arrays"]["faces_u32"],
        "canonical_gids_u64": arrays["gids_u64"],
        "capture_to_source_slot_i64": proof["actual_capture_source_slots_i64"],
        "full21_weights_f64": proof["actual_capture_weights_f64"],
        "full21_source_points_f32": proof["actual_capture_source_points_f32"],
        "full21_actual_capture_owner_proof": proof["actual_capture_owner_proof"],
    }
    kwargs = {
        "actual_capture_row": row,
        "candidate_positions_f32": arrays["positions_f32"],
        "candidate_faces_u32": arrays["faces_u32"],
        "candidate_gids_u64": arrays["gids_u64"],
        "candidate_source_slots_i64": arrays["source_slots_i64"],
        "candidate_weights_f64": arrays["weights_f64"],
        "candidate_source_points_f32": arrays["source_points_f32"],
        "predicted_gids_u64": np.empty(0, dtype="<u8"),
        "predicted_source_points_f32": np.empty((0, 3), dtype="<f4"),
        "predicted_lane_world_points_f32": np.empty((0, 21, 3), dtype="<f4"),
        "candidate_parent_payload_sha256": "a" * 64,
        "candidate_parent_extension_sha256": "b" * 64,
        "candidate_child_payload_sha256": "c" * 64,
        "candidate_archive_sha256": binding["archive_sha256"],
        "source_face_lineage": lineage,
        "source_vertex_lineage": surface.source_vertex_lineage,
        "candidate_topology_file_binding": binding,
    }
    return surface, kwargs, arrays, saved


def prepare(surface, proof, kwargs):
    arrays = proof["candidate_arrays"]
    return prepare_surface(
        surface.key, arrays["positions_f32"], arrays["faces_u32"],
        source_vertex_ids=arrays["gids_u64"],
        source_face_lineage=kwargs["source_face_lineage"],
        source_vertex_lineage=kwargs["source_vertex_lineage"],
        ordered_core_body_indices=list(range(21)),
        core_weight_gids=arrays["gids_u64"],
        core_weights_float64=arrays["weights_f64"],
        core_source_point_gids=arrays["gids_u64"],
        core_source_points_float32=arrays["source_points_f32"],
        core_capture_source_slots_i64=arrays["source_slots_i64"],
        core_mixed_candidate_owner_proof=proof,
    )


def test_separates_captured_indices_from_candidate_topology(tmp_path):
    surface, kwargs, arrays, _ = fixture(tmp_path)
    proof = build_mixed_full21_candidate_owner_record(**kwargs)
    assert proof["accepted_index_faces_u32"].tobytes() != arrays["faces_u32"].tobytes()
    result = prepare(surface, proof, kwargs)
    assert result.self_audit()["count"] == 0
    assert proof["actual_capture_owner_proof"] == surface.core_mixed_candidate_owner_proof["actual_capture_owner_proof"]


def test_default_still_rejects_unbound_topology(tmp_path):
    _, kwargs, _, _ = fixture(tmp_path)
    kwargs.pop("candidate_topology_file_binding")
    with pytest.raises(ValueError, match="1274 index"):
        build_mixed_full21_candidate_owner_record(**kwargs)


@pytest.mark.parametrize("field", ["archive_sha256", "lineage_sha256", "typed_identity", "array_prefix"])
def test_rejects_topology_binding_tampering(tmp_path, field):
    _, kwargs, _, _ = fixture(tmp_path)
    binding = kwargs["candidate_topology_file_binding"]
    binding[field] = [51005, 132] if field == "typed_identity" else ("bad_" if field == "array_prefix" else "0"*64)
    with pytest.raises((ValueError, KeyError)):
        build_mixed_full21_candidate_owner_record(**kwargs)


@pytest.mark.parametrize("field", [
    "positions_f32", "faces_u32", "canonical_gids_u64", "source_slots_i64",
    "weights_f64", "source_points_f32", "source_face_slots_u32",
])
def test_rejects_archive_array_mismatch_even_with_updated_file_pin(tmp_path, field):
    _, kwargs, _, saved = fixture(tmp_path)
    value = saved["row134_" + field].copy()
    value.flat[0] += 1
    saved["row134_" + field] = value
    archive = kwargs["candidate_topology_file_binding"]["archive_path"]
    np.savez(archive, **saved)
    digest = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    kwargs["candidate_topology_file_binding"]["archive_sha256"] = digest
    kwargs["candidate_archive_sha256"] = digest
    with pytest.raises(ValueError, match="candidate topology"):
        build_mixed_full21_candidate_owner_record(**kwargs)


def test_revalidates_files_and_accepted_indices_at_surface_preparation(tmp_path):
    surface, kwargs, _, _ = fixture(tmp_path)
    proof = build_mixed_full21_candidate_owner_record(**kwargs)
    bad = copy.deepcopy(proof)
    bad["accepted_index_faces_u32"] = bad["candidate_arrays"]["faces_u32"].copy()
    with pytest.raises(ValueError, match="displayed face"):
        prepare(surface, bad, kwargs)
    lineage_path = kwargs["candidate_topology_file_binding"]["lineage_path"]
    with open(lineage_path, "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="lineage SHA"):
        prepare(surface, proof, kwargs)


def test_rejects_changed_complete_lineage_with_updated_pin(tmp_path):
    _, kwargs, _, _ = fixture(tmp_path)
    binding = kwargs["candidate_topology_file_binding"]
    path = binding["lineage_path"]
    with open(path, "w") as stream:
        json.dump({"rows": {"134": {"source_face_lineage": []}}}, stream)
    binding["lineage_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with pytest.raises(ValueError, match="complete face lineage"):
        build_mixed_full21_candidate_owner_record(**kwargs)


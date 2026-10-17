"""Tests for the narrow full21 forearm parent-carrier bridge."""
import hashlib
import json
import struct

import numpy as np
import pytest

from numilab_human.forearm_parent_carrier_bridge import (
    append_full21_carrier_points,
    append_parent_surface_terms,
    full21_parent_terms,
    build_authenticated_full21_lane_owner_proof,
    accepted_poses_from_receipt,
)


def make_carrier(stable_id=132, source_count=2):
    core_ids = tuple(range(91, 91 + 21))
    binding_bytes = b"".join(
        struct.pack("<I8f", core, 0, 0, 0, 0, 0, 0, 1, 1)
        for core in core_ids
    )
    vertices = np.zeros(source_count * 21, dtype=np.dtype([
        ("vertex", "<f4", (6,)), ("binding", "<u4", (4,)), ("weight", "<f4", (4,))
    ]))
    points = np.asarray([[i, 2*i, 3*i, 0, 0, 1] for i in range(source_count)], dtype="<f4")
    vertices["vertex"] = np.repeat(points[:, None, :], 21, axis=1).reshape(-1, 6)
    vertices["binding"][:, 0] = np.tile(np.arange(21, dtype="<u4"), source_count)
    vertices["weight"][:, 0] = 1
    index = np.asarray([0, 21, 0, 0, 0, 21], dtype="<u4")
    row = struct.pack("<8I", 0, 21, 0, source_count * 21, 0, len(index), stable_id, 1)
    header = struct.pack("<8s6I32s", b"NHTISS4\0", 5, 1, 21,
                         source_count * 21, len(index), 0x1234, b"x" * 32)
    return header + row + binding_bytes + vertices.tobytes() + index.tobytes(), core_ids


def normalized_weights(count):
    values = np.arange(1, 22, dtype=np.float64)
    return np.tile(values / values.sum(), (count, 1))


def test_append_points_preserves_old_bytes_and_emits_all_lanes():
    raw, core_ids = make_carrier()
    old_nv = 42
    points = np.asarray([[7, 8, 9, 0, 1, 0], [10, 11, 12, 1, 0, 0]], dtype="<f4")
    emitted, first, report = append_full21_carrier_points(
        raw, stable_id=132, core_body_indices=core_ids, source_points6=points,
    )
    magic, abi, nr, nb, nv, ni, fp, digest = struct.unpack_from("<8s6I32s", emitted)
    assert (magic, abi, nr, nb, nv, ni, fp, digest) == (
        b"NHTISS4\0", 5, 1, 21, old_nv + 42, 6, 0x1234, b"x" * 32)
    assert first == old_nv
    assert report["old_header_record_binding_vertex_prefix_and_indices_byte_exact"]
    assert emitted[64 + 32 + 21*36:64 + 32 + 21*36 + old_nv*56] == raw[64 + 32 + 21*36:64 + 32 + 21*36 + old_nv*56]
    assert emitted[-24:] == raw[-24:]
    appended = np.frombuffer(emitted, dtype=np.dtype([
        ("vertex", "<f4", (6,)), ("binding", "<u4", (4,)), ("weight", "<f4", (4,))
    ]), count=42, offset=64 + 32 + 21*36 + old_nv*56)
    np.testing.assert_array_equal(appended["vertex"].reshape(2, 21, 6), np.repeat(points[:, None, :], 21, axis=1))
    np.testing.assert_array_equal(appended["binding"][:, 0], np.tile(np.arange(21, dtype="<u4"), 2))
    np.testing.assert_array_equal(appended["weight"][:, 0], np.ones(42, dtype="<f4"))


def test_full21_terms_keep_order_and_complete_parent_identity():
    weights = normalized_weights(2)
    terms = full21_parent_terms(source_index=5, stable_id=134,
                                first_parent_vertex=822_507, core_weights_float64=weights)
    assert len(terms) == 2 and all(len(row) == 21 for row in terms)
    assert terms[0][0] == [5, 134, 822_507, float(weights[0, 0])]
    assert terms[0][-1] == [5, 134, 822_527, float(weights[0, -1])]
    assert terms[1][0][2] == 822_528
    assert [term[2] for term in terms[1]] == list(range(822_528, 822_549))


@pytest.mark.parametrize("defect", ["missing_lane", "zero_weight"])
def test_full21_terms_reject_corrupt_material_contract(defect):
    weights = normalized_weights(1)
    if defect == "zero_weight":
        weights[0, 0] = 0
        weights[0, 1:] /= weights[0, 1:].sum()
    if defect == "missing_lane":
        weights = weights[:, :-1]
    with pytest.raises(ValueError):
        full21_parent_terms(source_index=0, stable_id=132,
                            first_parent_vertex=0, core_weights_float64=weights)


def test_parent_extension_update_appends_only_pinned_row_and_source():
    extension = {
        "child_payload_sha256": "a" * 64,
        "source_payloads": [{"path": "old.nhtissue", "sha256": "b" * 64},
                            {"path": "other.nhtissue", "sha256": "c" * 64}],
        "surfaces": [{"stable_id": 132, "vertices": [[[0, 132, i, 1 / 21] for i in range(21)]]},
                     {"stable_id": 134, "vertices": []}],
    }
    appended = full21_parent_terms(source_index=0, stable_id=132, first_parent_vertex=42,
                                   core_weights_float64=normalized_weights(1))
    result = append_parent_surface_terms(
        extension, stable_id=132, source_index=0,
        expected_child_payload_sha256="a" * 64, child_payload_sha256="d" * 64,
        expected_source_payload_sha256="b" * 64, source_payload_path="/immutable/new.nhtissue",
        source_payload_sha256="e" * 64, expected_existing_term_count=1,
        appended_terms=appended,
    )
    assert extension["child_payload_sha256"] == "a" * 64
    assert result["child_payload_sha256"] == "d" * 64
    assert result["source_payloads"][0] == {"path": "/immutable/new.nhtissue", "sha256": "e" * 64}
    assert result["source_payloads"][1] == extension["source_payloads"][1]
    assert result["surfaces"][0]["vertices"][0] == extension["surfaces"][0]["vertices"][0]
    assert len(result["surfaces"][0]["vertices"]) == 2
    assert result["surfaces"][1] == extension["surfaces"][1]


@pytest.mark.parametrize("defect", ["child_pin", "source_pin", "wrong_sid", "bad_count"])
def test_parent_extension_update_fails_closed(defect):
    extension = {
        "child_payload_sha256": "a" * 64,
        "source_payloads": [{"path": "old.nhtissue", "sha256": "b" * 64}],
        "surfaces": [{"stable_id": 132, "vertices": []}],
    }
    terms = full21_parent_terms(source_index=0, stable_id=132, first_parent_vertex=0,
                                core_weights_float64=normalized_weights(1))
    args = dict(stable_id=132, source_index=0, expected_child_payload_sha256="a"*64,
                child_payload_sha256="d"*64, expected_source_payload_sha256="b"*64,
                source_payload_path="/new.nhtissue", source_payload_sha256="e"*64,
                expected_existing_term_count=0, appended_terms=terms)
    if defect == "child_pin":
        args["expected_child_payload_sha256"] = "f" * 64
    elif defect == "source_pin":
        args["expected_source_payload_sha256"] = "f" * 64
    elif defect == "wrong_sid":
        args["stable_id"] = 134
    else:
        args["expected_existing_term_count"] = 1
    with pytest.raises(ValueError):
        append_parent_surface_terms(extension, **args)



def test_birth_source_point_uses_actual_accepted_pose_full21_edge_pullback():
    from numilab_human.forearm_parent_carrier_bridge import (
        derive_full21_birth_source_point,
        full21_parent_world_positions,
    )

    core_ids = tuple(range(21))
    row_data = {
        "stable_id": 132,
        "vertex_count": 21,
        "binding_indices": np.zeros((21, 4), dtype="<u4"),
        "weights": np.zeros((21, 4), dtype="<f4"),
        "bindings": [
            {"core": i, "translation": np.zeros(3, dtype="<f4"),
             "quaternion": np.asarray([0, 0, 0, 1], dtype="<f4"),
             "scale": np.float32(1)}
            for i in core_ids
        ],
    }
    row_data["binding_indices"][:, 0] = np.arange(21, dtype="<u4")
    row_data["weights"][:, 0] = np.float32(1)
    poses = {
        i: np.asarray([i * 1e-4, -i * 2e-5, i * 3e-5, 0, 0, 0, 1], dtype="<f4")
        for i in core_ids
    }
    points = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]], dtype="<f4")
    base = np.arange(1, 22, dtype=np.float64)
    weights = np.stack([base / base.sum(), base[::-1] / base.sum()])
    expected_endpoints = full21_parent_world_positions(row_data, poses, points, weights)
    t = 0.37
    source, birth_weights, target, report = derive_full21_birth_source_point(
        row_data, poses, points, weights, t,
    )
    expected_target = np.asarray(
        expected_endpoints[0].astype(np.float64) * (1 - t)
        + expected_endpoints[1].astype(np.float64) * t,
        dtype="<f4",
    )
    np.testing.assert_array_equal(target, expected_target)
    assert birth_weights.shape == (21,)
    assert np.all(birth_weights > 0)
    assert abs(float(birth_weights.sum()) - 1.0) <= 1e-12
    np.testing.assert_allclose(report["actual_accepted_pose_full21_replay_f32"],
                               target, atol=2e-7, rtol=0)
    assert report["static_candidate_positions_world_used"] is False
    assert np.isfinite(source).all()
    assert report["full21_weight_lane_count"] == 21


def test_child_birth_preserves_complete_four_lane_support_and_actual_pose():
    from numilab_human.forearm_parent_carrier_bridge import derive_child_four_lane_birth

    row = {
        "stable_id": 132,
        "vertex_count": 2,
        "positions": np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], dtype="<f4"),
        "normals": np.asarray([[0, 0, 1], [0, 0, 1]], dtype="<f4"),
        "binding_indices": np.zeros((2, 4), dtype="<u4"),
        "weights": np.zeros((2, 4), dtype="<f4"),
        "bindings": [
            {"core": 0, "translation": np.zeros(3, dtype="<f4"),
             "quaternion": np.asarray([0, 0, 0, 1], dtype="<f4"),
             "scale": np.float32(1)}
        ],
    }
    row["weights"][:, 0] = 1.0
    poses = {0: np.asarray([0, 0, 0, 0, 0, 0, 1], dtype="<f4")}
    target = np.asarray([0.2, 0.3, 0.4], dtype="<f4")
    vertex, binding_ids, lane_weights, report = derive_child_four_lane_birth(
        row, poses, np.asarray([0, 1], dtype=np.int64), target, 0.5,
    )
    np.testing.assert_array_equal(vertex[:3], target)
    assert binding_ids.tolist() == [0, 0, 0, 0]
    assert lane_weights.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert report["max_four_lane_replay_residual_m"] <= 2e-7
    assert report["four_lane_support_truncated"] is False


def test_child_birth_refuses_more_than_four_route_supports():
    from numilab_human.forearm_parent_carrier_bridge import derive_child_four_lane_birth

    row = {
        "stable_id": 132,
        "vertex_count": 2,
        "positions": np.zeros((2, 3), dtype="<f4"),
        "normals": np.asarray([[0, 0, 1], [0, 0, 1]], dtype="<f4"),
        "binding_indices": np.asarray([[0, 1, 2, 3], [4, 5, 0, 0]], dtype="<u4"),
        "weights": np.asarray([[.25, .25, .25, .25], [.5, .5, 0, 0]], dtype="<f4"),
        "bindings": [
            {"core": i, "translation": np.zeros(3, dtype="<f4"),
             "quaternion": np.asarray([0, 0, 0, 1], dtype="<f4"),
             "scale": np.float32(1)}
            for i in range(6)
        ],
    }
    poses = {i: np.asarray([0, 0, 0, 0, 0, 0, 1], dtype="<f4") for i in range(6)}
    with pytest.raises(ValueError, match="refusing truncation"):
        derive_child_four_lane_birth(
            row, poses, np.asarray([0, 1], dtype=np.int64),
            np.asarray([0, 0, 0], dtype="<f4"), .5,
        )


def test_canonical_gid_lineage_keeps_source_ids_and_slots_distinct():
    from numilab_human.forearm_parent_carrier_bridge import validate_source_gid_slot_lineage

    report = validate_source_gid_slot_lineage(
        np.asarray([0, -1, 12], dtype=np.int64),
        np.asarray([979, 9000000000000000001, 1002], dtype=np.uint64),
        {
            "979": {
                "source_face_support_provenance": "original-keyed-material-handoff-GID",
                "source_gid": 979,
                "source_vertex_id": 979,
            },
            "1002": {
                "source_face_support_provenance": "original-keyed-material-handoff-GID",
                "source_gid": 1002,
                "source_vertex_id": 1002,
            },
        },
    )
    assert report == {
        "validated_original_vertex_count": 2,
        "validated_distinct_source_slot_count": 2,
        "validated_source_vertex_id_count": 2,
        "canonical_gid_and_source_slot_are_vertex_aligned": True,
        "source_vertex_id_numeric_equality_to_local_slot_required": False,
        "pack_global_vertex_ids_used_for_identity": False,
    }


def test_canonical_gid_lineage_rejects_canonical_source_gid_conflict():
    from numilab_human.forearm_parent_carrier_bridge import validate_source_gid_slot_lineage

    with pytest.raises(ValueError, match="malformed source identity"):
        validate_source_gid_slot_lineage(
            np.asarray([0], dtype=np.int64),
            np.asarray([979], dtype=np.uint64),
            {
                "979": {
                    "source_face_support_provenance": "original-keyed-material-handoff-GID",
                    "source_gid": 980,
                    "source_vertex_id": 979,
                },
            },
        )


def test_birth_parent_gid_resolves_nonidentity_source_slots():
    from numilab_human.forearm_parent_carrier_bridge import resolve_birth_source_parent_slots

    slots = resolve_birth_source_parent_slots(
        [979, 1002],
        np.asarray([979, 1002, 7000000001], dtype=np.uint64),
        np.asarray([0, 12, -1], dtype=np.int64),
    )
    assert slots.tolist() == [0, 12]


def test_birth_parent_gid_rejects_inferred_parent():
    from numilab_human.forearm_parent_carrier_bridge import resolve_birth_source_parent_slots

    with pytest.raises(ValueError, match="recursive ancestry"):
        resolve_birth_source_parent_slots(
            [979, 1002],
            np.asarray([979, 1002], dtype=np.uint64),
            np.asarray([0, -1], dtype=np.int64),
        )


def test_auxiliary_route_remaps_by_exact_typed_transform_and_preserves_weights():
    from numilab_human.forearm_parent_carrier_bridge import (
        remap_child_four_lane_route_by_binding_identity,
        emit_child_with_inferred_four_lane_route,
    )

    def binding(core, translation=(0, 0, 0)):
        return {
            "core": core,
            "translation": np.asarray(translation, dtype="<f4"),
            "quaternion": np.asarray([0, 0, 0, 1], dtype="<f4"),
            "scale": np.float32(1),
        }

    source_bindings = [binding(100), binding(101)]
    target_bindings = [binding(101), binding(100)]
    source_indices = np.asarray([0, 1, 0, 0], dtype="<u4")
    source_weights = np.asarray([0.6, 0.4, 0, 0], dtype="<f4")
    target_indices, target_weights, proof = remap_child_four_lane_route_by_binding_identity(
        source_bindings, target_bindings, source_indices, source_weights,
    )
    assert target_indices.tolist() == [1, 0, 0, 0]
    assert target_weights.tobytes() == source_weights.tobytes()
    assert proof["route_support_truncated"] is False

    row = {
        "stable_id": 132,
        "vertex_count": 1,
        "positions": np.asarray([[0.1, 0.2, 0.3]], dtype="<f4"),
        "normals": np.asarray([[0, 0, 1]], dtype="<f4"),
        "binding_indices": np.zeros((1, 4), dtype="<u4"),
        "weights": np.asarray([[1, 0, 0, 0]], dtype="<f4"),
        "bindings": target_bindings,
    }
    poses = {
        core: np.asarray([0, 0, 0, 0, 0, 0, 1], dtype="<f4")
        for core in (100, 101)
    }
    target = np.asarray([0.5, 0.6, 0.7], dtype="<f4")
    emitted, indices, weights, replay = emit_child_with_inferred_four_lane_route(
        row, poses, target_indices, target_weights, target,
        np.asarray([0.1, 0.2, 0.3], dtype="<f4"),
        np.asarray([0, 0, 1], dtype="<f4"),
    )
    np.testing.assert_array_equal(emitted[:3], target)
    assert indices.tolist() == [1, 0, 0, 0]
    assert weights.tobytes() == source_weights.tobytes()
    assert replay["route_weights_byte_exact"]
    assert replay["max_four_lane_replay_residual_m"] <= 2e-6


def test_auxiliary_route_rejects_unmatched_typed_binding_transform():
    from numilab_human.forearm_parent_carrier_bridge import (
        remap_child_four_lane_route_by_binding_identity,
    )

    def binding(core):
        return {
            "core": core, "translation": np.zeros(3, dtype="<f4"),
            "quaternion": np.asarray([0, 0, 0, 1], dtype="<f4"),
            "scale": np.float32(1),
        }

    with pytest.raises(ValueError, match="exact typed transform matches"):
        remap_child_four_lane_route_by_binding_identity(
            [binding(100)], [binding(101)],
            np.asarray([0, 0, 0, 0], dtype="<u4"),
            np.asarray([1, 0, 0, 0], dtype="<f4"),
        )

def test_duplicate_birth_edge_collapse_requires_strict_link_and_changes_faces_only():
    from numilab_human.forearm_topology_patch import collapse_exact_duplicate_edge

    # Octahedral surface with A/B occupying the same emitted Float32 point.
    points = np.asarray([
        [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, -1],
        [-1, 0, 0], [0, -1, 0],
    ], dtype="<f4")
    gids = np.asarray([101, 202, 303, 404, 505, 606], dtype="<u8")
    faces = np.asarray([
        [0, 1, 2], [1, 4, 2], [4, 5, 2], [5, 0, 2],
        [0, 1, 3], [1, 4, 3], [4, 5, 3], [5, 0, 3],
    ], dtype="<u4")
    points_before, gids_before, faces_before = points.tobytes(), gids.tobytes(), faces.tobytes()
    collapsed, report = collapse_exact_duplicate_edge(
        faces, points, gids, 0, 1, expected_incident_faces=(0, 4),
    )
    assert collapsed.shape == (6, 3)
    assert report["strict_link_condition_passed"]
    assert report["incident_face_indices_removed"] == [0, 4]
    assert report["kept_canonical_material_gid"] == 101
    assert report["retired_canonical_material_gid"] == 202
    assert points.tobytes() == points_before and gids.tobytes() == gids_before
    assert faces.tobytes() == faces_before


def test_duplicate_birth_edge_collapse_rejects_nonlink_neighbor():
    from numilab_human.forearm_topology_patch import collapse_exact_duplicate_edge

    points = np.asarray([
        [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, -1],
        [-1, 0, 0], [0, -1, 0], [1, 1, 0],
    ], dtype="<f4")
    gids = np.arange(100, 107, dtype="<u8")
    faces = np.asarray([
        [0, 1, 2], [1, 4, 2], [4, 5, 2], [5, 0, 2], [0, 1, 3],
        [1, 4, 3], [4, 5, 3], [5, 0, 3],
        [0, 6, 2], [1, 6, 3],
    ], dtype="<u4")
    with pytest.raises(ValueError, match="link condition"):
        collapse_exact_duplicate_edge(faces, points, gids, 0, 1)



def test_authenticated_full21_lane_proof_hashes_files_and_replays_nonzero_step(tmp_path):
    from numilab_human import passive_attachment_composition as pc

    child_bytes, core_ids = make_carrier(stable_id=132, source_count=3)
    parent_bytes, _ = make_carrier(stable_id=132, source_count=3)
    child_path = tmp_path / "child.nhtissue"
    parent_path = tmp_path / "parent.nhtissue"
    extension_path = tmp_path / "parents.json"
    receipt_path = tmp_path / "step-4.receipt.json"
    child_path.write_bytes(child_bytes)
    parent_path.write_bytes(parent_bytes)

    parent_file = pc._read_nhtiss4(parent_path)
    parent_record = next(row for row in parent_file["records"] if int(row[6]) == 132)
    parent_row = pc._nhtiss_row_arrays(parent_file, parent_record)
    child_file = pc._read_nhtiss4(child_path)
    child_record = next(row for row in child_file["records"] if int(row[6]) == 132)
    child_row = pc._nhtiss_row_arrays(child_file, child_record)
    source_points = parent_row["positions"].copy()
    weights = normalized_weights(len(source_points))
    vertices = [
        [[0, 132, (vertex // 21) * 21 + lane, float(weights[vertex, lane])]
         for lane in range(21)]
        for vertex in range(len(source_points))
    ]
    extension = {
        "child_payload_sha256": hashlib.sha256(child_bytes).hexdigest(),
        "source_payloads": [{
            "path": str(parent_path.resolve()),
            "sha256": hashlib.sha256(parent_bytes).hexdigest(),
        }],
        "surfaces": [{"stable_id": 132, "vertices": vertices}],
    }
    extension_path.write_text(json.dumps(extension, sort_keys=True))
    body_poses = [{
        "body_index": index,
        "position_m": [index * 1.0e-6, -index * 2.0e-6, index * 3.0e-6],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    } for index in range(157)]
    receipt = {
        "schema": "numi.human.accepted-render-geometry.v1",
        "accepted_step": 4,
        "accepted_body_pose_count": len(body_poses),
        "accepted_body_poses": body_poses,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    poses, pose_report = accepted_poses_from_receipt(receipt_path)
    lane_world = pc._binding_world_positions(parent_row, source_points, poses)[:, :21, :]
    emitted = np.zeros_like(source_points, dtype=np.float32)
    weight_f32 = weights.astype("<f4")
    for lane in range(21):
        emitted = np.asarray(
            emitted + np.asarray(lane_world[:, lane, :] * weight_f32[:, lane, None],
                                 dtype=np.float32),
            dtype=np.float32)
    gids = np.arange(1001, 1001 + len(source_points), dtype=np.uint64)
    slots = np.arange(len(source_points), dtype=np.int64)
    faces = child_row["faces"].astype(np.uint32)
    face_lineage = [{"source_parent_face": int(index)} for index in range(len(faces))]
    vertex_lineage = {
        int(gid): {"source_gid": int(gid), "source_parent_gids": [int(gid)]}
        for gid in gids
    }
    proof_inputs = {
        "source_payload_path": child_path,
        "parent_extension_path": extension_path,
        "parent_payload_path": parent_path,
        "accepted_step_receipt_path": receipt_path,
        "stable_id": 132, "source_slots_i64": slots,
        "canonical_gids_u64": gids, "source_points_float32": source_points,
        "core_weights_float64": weights, "displayed_positions_float32": emitted,
        "faces_local_u32": faces, "source_face_lineage": face_lineage,
        "source_vertex_lineage": vertex_lineage,
        "expected_ordered_core_body_indices": core_ids,
    }
    lanes, proof, report = build_authenticated_full21_lane_owner_proof(**proof_inputs)
    assert proof["accepted_step"] == 4 == pose_report["accepted_step"]
    assert proof["capture_receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert proof["source_payload_sha256"] == hashlib.sha256(child_path.read_bytes()).hexdigest()
    assert proof["parent_payload_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert proof["parent_extension_sha256"] == hashlib.sha256(extension_path.read_bytes()).hexdigest()
    assert lanes.shape == (len(source_points), 21, 3)
    assert report["displayed_positions_exact_owner_blend"] is True

    # A well-formed digest-shaped string in the sidecar cannot authenticate
    # changed parent bytes; the builder reads and checks the actual files.
    parent_path.write_bytes(parent_bytes + b"tampered")
    with pytest.raises(ValueError, match="exact parent bytes"):
        build_authenticated_full21_lane_owner_proof(**proof_inputs)


def test_authenticated_full21_lane_proof_rejects_detached_display_positions(tmp_path):
    from numilab_human import passive_attachment_composition as pc

    child_bytes, core_ids = make_carrier(stable_id=132, source_count=3)
    parent_bytes, _ = make_carrier(stable_id=132, source_count=3)
    child_path = tmp_path / "child.nhtissue"
    parent_path = tmp_path / "parent.nhtissue"
    extension_path = tmp_path / "parents.json"
    receipt_path = tmp_path / "step-4.receipt.json"
    child_path.write_bytes(child_bytes)
    parent_path.write_bytes(parent_bytes)
    parent_file = pc._read_nhtiss4(parent_path)
    parent_record = next(row for row in parent_file["records"] if int(row[6]) == 132)
    parent_row = pc._nhtiss_row_arrays(parent_file, parent_record)
    child_file = pc._read_nhtiss4(child_path)
    child_record = next(row for row in child_file["records"] if int(row[6]) == 132)
    child_row = pc._nhtiss_row_arrays(child_file, child_record)
    source_points = parent_row["positions"].copy()
    weights = normalized_weights(len(source_points))
    extension = {
        "child_payload_sha256": hashlib.sha256(child_bytes).hexdigest(),
        "source_payloads": [{
            "path": str(parent_path.resolve()),
            "sha256": hashlib.sha256(parent_bytes).hexdigest(),
        }],
        "surfaces": [{"stable_id": 132, "vertices": [
            [[0, 132, (vertex // 21) * 21 + lane, float(weights[vertex, lane])]
             for lane in range(21)]
            for vertex in range(len(source_points))
        ]}],
    }
    extension_path.write_text(json.dumps(extension, sort_keys=True))
    body_poses = [{
        "body_index": index, "position_m": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    } for index in range(157)]
    receipt_path.write_text(json.dumps({
        "schema": "numi.human.accepted-render-geometry.v1", "accepted_step": 4,
        "accepted_body_pose_count": len(body_poses), "accepted_body_poses": body_poses,
    }, sort_keys=True))
    emitted = np.zeros_like(source_points, dtype=np.float32)
    for lane in range(21):
        emitted = np.asarray(
            emitted + np.asarray(source_points * weights[:, lane, None].astype("<f4"),
                                 dtype=np.float32),
            dtype=np.float32)
    emitted[0, 0] = np.nextafter(emitted[0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="exact accepted-pose full21 owner blend"):
        build_authenticated_full21_lane_owner_proof(
            source_payload_path=child_path,
            parent_extension_path=extension_path,
            parent_payload_path=parent_path,
            accepted_step_receipt_path=receipt_path,
            stable_id=132, source_slots_i64=np.arange(len(source_points), dtype=np.int64),
            canonical_gids_u64=np.arange(1001, 1001 + len(source_points), dtype=np.uint64),
            source_points_float32=source_points, core_weights_float64=weights,
            displayed_positions_float32=emitted,
            faces_local_u32=child_row["faces"].astype(np.uint32),
            source_face_lineage=[{"source_parent_face": int(index)}
                                 for index in range(len(child_row["faces"]))],
            source_vertex_lineage={
                gid: {"source_gid": int(gid), "source_parent_gids": [int(gid)]}
                for gid in range(1001, 1001 + len(source_points))
            },
            expected_ordered_core_body_indices=core_ids,
        )


def test_accepted_pose_reader_rejects_boolean_step(tmp_path):
    receipt_path = tmp_path / "bad-step.receipt.json"
    body_poses = [{
        "body_index": index, "position_m": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    } for index in range(157)]
    receipt_path.write_text(json.dumps({
        "schema": "numi.human.accepted-render-geometry.v1", "accepted_step": True,
        "accepted_body_pose_count": len(body_poses), "accepted_body_poses": body_poses,
    }))
    with pytest.raises(ValueError, match="nonnegative accepted render-geometry step"):
        accepted_poses_from_receipt(receipt_path)

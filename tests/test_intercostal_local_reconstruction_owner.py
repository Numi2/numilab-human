import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from numilab_human.resting_anatomy_interface_patch import encode_payload
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_intercostal_mesh_arrangement import (
    LOCAL_RECONSTRUCTION_ANCESTRY_SCHEMA,
    _authenticated_local_reconstruction_body20_pose,
    _normalize_local_reconstruction_parent_face_owners,
    _localize_body20_normals_f32,
    derive_local_reconstruction_attachment_maps,
    derive_local_reconstruction_face_rib_owners,
    derive_local_reconstruction_source_face_lineage,
    derive_local_reconstruction_973_runtime_face_owners,
    compile_local_reconstruction_intercostal_payload,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_tool_report(rib_id: int = 48, *, face_count: int = 20) -> bytes:
    return json.dumps({
        "rows": [{
            "source_bone": [51004, rib_id],
            "derived_source_id": 900000000 + rib_id,
            "tool_sha256": "e" * 64,
            "faces": face_count,
        }],
    }, sort_keys=True, separators=(",", ":")).encode()


def _mixed_source_tool_report() -> bytes:
    return json.dumps({
        "source_arrays_sha256": "a" * 64,
        "rows": [{
            "source_bone": [51004, bone_id],
            "derived_source_id": 900000000 + bone_id,
            "tool_sha256": f"{bone_id:064x}",
            "faces": 20,
        } for bone_id in (26, *range(48, 60), 178)],
    }, sort_keys=True, separators=(",", ":")).encode()


def _source_face_report(source_id: int = 900000048, source_face_id: int = 17) -> bytes:
    return json.dumps({
        "faces": [{
            "output_face": 0,
            "classification": "unchanged_parent_face",
            "source_rows": [[source_id, source_face_id]],
            "exact_source_face_claim": True,
        }],
    }, sort_keys=True, separators=(",", ":")).encode()


def _face_npz(faces: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, faces=np.asarray(faces, dtype="<u4"))
    return buffer.getvalue()


def _owner_report(parent_payload: bytes, ancestry: bytes, faces: np.ndarray) -> bytes:
    faces = np.ascontiguousarray(faces, dtype="<u4")
    report = {
        "schema": LOCAL_RECONSTRUCTION_ANCESTRY_SCHEMA,
        "stable_id": 313,
        "parent_payload_sha256": _sha(parent_payload),
        "operation_ancestry_sha256": _sha(ancestry),
        "parent_faces_sha256": _sha(faces.tobytes()),
        "parent_face_count": len(faces),
        "parent_face_owners": [{
            "parent_face_id": 0,
            "rib_stable_ids": [48],
            "evidence": [{
                "rib_stable_id": 48,
                "method": "direct_typed_rib_source",
                "source_report_sha256": _sha(ancestry),
                "source_face_ids": [17],
            }],
        }],
    }
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode()


def _accepted_receipt(pack_sha: str, source_sha: str) -> dict:
    pose = {"position_m": [0.1, 0.2, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    accepted = [{"body_index": i, **pose} for i in range(21)]
    registered = [{"body_index": 20, **pose}]
    return {
        "schema": "numi.human.accepted-render-geometry.v1",
        "physical_endpoint": "accepted",
        "pack_file_sha256": pack_sha,
        "accepted_step": 0,
        "accepted_time_s": 0.0,
        "accepted_body_pose_count": 21,
        "accepted_body_poses": accepted,
        "accepted_registered_body_poses": registered,
        "common_field_source_anatomy_payload_sha256": source_sha,
    }



def test_authenticated_1091_source_lineage_keeps_mixed_and_nonrib_faces_unowned(monkeypatch):
    import numilab_human.resting_intercostal_mesh_arrangement as owner

    stable_id = 313
    source_parent = _face_npz(np.asarray([[0, 1, 2], [1, 2, 3]], dtype="<u4"))
    floating = _face_npz(np.asarray([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype="<u4"))
    candidate = _face_npz(np.asarray([
        [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
    ], dtype="<u4"))
    eo_tool = _face_npz(np.asarray([[0, 1, 2]], dtype="<u4"))
    tool_report = _source_tool_report(48, face_count=1)
    script = (
        b"run_original_id=np.asarray([oid]\n"
        b"face_id=np.arange(len(f),dtype=np.uint64)\n"
        b"51010000+sid\n"
    )
    script_sha = _sha(script)
    pins = json.dumps({
        "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
        "full-anatomy-delivery-audit-1378/"
        "construct_intercostal_all_three_external_oblique68_gap_1091.py": script_sha,
        "pinned_parent_candidate": _sha(source_parent),
        "pinned_eo68_source": "d" * 64,
    }, sort_keys=True, separators=(",", ":")).encode()
    monkeypatch.setattr(owner, "LOCAL_1091_SOURCE_ASSIGNMENT_SCRIPT_SHA256", script_sha)
    monkeypatch.setattr(
        owner, "LOCAL_1091_SOURCE_ASSIGNMENT_INPUT_PINS_SHA256", _sha(pins))
    operation = json.dumps({
        "identity": [51010, stable_id],
        "parent_candidate_sha256": _sha(source_parent),
        "derived_original_id": 900020068,
        "target_identity": [51005, 68],
        "target_source_sha256": "d" * 64,
        "derived_tool_sha256": _sha(eo_tool),
    }, sort_keys=True, separators=(",", ":")).encode()
    ancestry = json.dumps({
        "parent_candidate_sha256": _sha(floating),
        "faces": [
            {"output_face": 0, "classification": "unchanged_parent_face",
             "source_rows": [[51010313, 0]], "exact_source_face_claim": True},
            {"output_face": 1, "classification": "derived_nonrib_tool",
             "source_rows": [[900020068, 0]], "exact_source_face_claim": False},
            {"output_face": 2, "classification": "mixed_source",
             "source_rows": [[51010313, 1], [900020068, 0]],
             "exact_source_face_claim": False},
            {"output_face": 3, "classification": "derived_typed_rib_tool",
             "source_rows": [[900000048, 0]], "exact_source_face_claim": True},
        ],
    }, sort_keys=True, separators=(",", ":")).encode()
    current_operation = json.dumps({
        "parent_candidate_sha256": _sha(floating),
        "candidate_sha256": _sha(candidate),
    }, sort_keys=True, separators=(",", ":")).encode()
    face_parents, direct_owners, summary = derive_local_reconstruction_source_face_lineage(
        ancestry, _sha(ancestry), stable_id=stable_id,
        candidate_npz_raw=candidate, expected_candidate_npz_sha256=_sha(candidate),
        current_operation_ancestry_raw=current_operation,
        current_parent_candidate_raw=floating,
        source_face_ancestry_parent_candidate_raw=floating,
        source_operation_ancestry_raw=operation,
        source_operation_parent_candidate_raw=source_parent,
        source_assignment_script_raw=script,
        source_assignment_input_pins_raw=pins,
        derived_nonrib_tool_raw=eo_tool,
        authenticated_source_report_raws=[tool_report],
        authenticated_source_report_sha256s=[_sha(tool_report)],
    )
    assert face_parents == [[0], [], [], []]
    assert direct_owners == [None, None, None, 48]
    assert summary["single_parent_face_count"] == 1
    assert summary["direct_typed_rib_face_count"] == 1
    assert summary["unowned_face_count"] == 2
    assert summary["anatomical_admission"] is False

    with pytest.raises(ValueError, match="tool bytes differ"):
        derive_local_reconstruction_source_face_lineage(
            ancestry, _sha(ancestry), stable_id=stable_id,
            candidate_npz_raw=candidate, expected_candidate_npz_sha256=_sha(candidate),
            current_operation_ancestry_raw=current_operation,
            current_parent_candidate_raw=floating,
            source_face_ancestry_parent_candidate_raw=floating,
            source_operation_ancestry_raw=operation,
            source_operation_parent_candidate_raw=source_parent,
            source_assignment_script_raw=script,
            source_assignment_input_pins_raw=pins,
            derived_nonrib_tool_raw=_face_npz(np.asarray([[0, 2, 1]], dtype="<u4")),
            authenticated_source_report_raws=[tool_report],
            authenticated_source_report_sha256s=[_sha(tool_report)],
        )


def test_authenticated_slot_preserving_face_ancestry_binds_exact_parent_slots():
    parent = _face_npz(np.asarray([[0, 1, 2], [1, 2, 3]], dtype="<u4"))
    candidate = _face_npz(np.asarray([[0, 1, 2], [1, 2, 3]], dtype="<u4"))
    ancestry = json.dumps({
        "parent_candidate_sha256": _sha(parent),
        "faces": [
            {"output_face": i, "source_rows": [[51010313, i]]}
            for i in range(2)
        ],
    }, sort_keys=True, separators=(",", ":")).encode()
    operation = json.dumps({
        "parent_candidate_sha256": _sha(parent),
        "parent_face_ancestry_sha256": _sha(ancestry),
        "candidate_sha256": _sha(candidate),
        "faces_retain_parent_slots": True,
        "source_face_ancestry_unchanged": True,
    }, sort_keys=True, separators=(",", ":")).encode()
    parents, owners, summary = derive_local_reconstruction_source_face_lineage(
        ancestry, _sha(ancestry), stable_id=313,
        candidate_npz_raw=candidate, expected_candidate_npz_sha256=_sha(candidate),
        current_operation_ancestry_raw=operation,
        current_parent_candidate_raw=parent,
        source_face_ancestry_parent_candidate_raw=parent,
    )
    assert parents == [[0], [1]]
    assert owners == [None, None]
    assert summary["preserved_face_slot_count"] == 2
    assert summary["native_admission"] is False

    changed = _face_npz(np.asarray([[0, 1, 2], [1, 3, 2]], dtype="<u4"))
    changed_operation = json.dumps({
        "parent_candidate_sha256": _sha(parent),
        "parent_face_ancestry_sha256": _sha(ancestry),
        "candidate_sha256": _sha(changed),
        "faces_retain_parent_slots": True,
        "source_face_ancestry_unchanged": True,
    }, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="changed the parent face slots"):
        derive_local_reconstruction_source_face_lineage(
            ancestry, _sha(ancestry), stable_id=313,
            candidate_npz_raw=changed, expected_candidate_npz_sha256=_sha(changed),
            current_operation_ancestry_raw=changed_operation,
            current_parent_candidate_raw=parent,
            source_face_ancestry_parent_candidate_raw=parent,
        )


def test_973_runtime_maps_project_only_exact_source_faces_and_reject_drift(monkeypatch):
    import numilab_human.resting_intercostal_mesh_arrangement as owner
    from numilab_human.composed_bone_body_frame import _RUNTIME_ATTACHMENT_RIBS

    vertices = np.asarray([
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
    ], dtype="<f4")
    faces = np.asarray([[0, 1, 2], [1, 2, 3], [0, 2, 3]], dtype="<u4")
    source = encode_payload(
        (b"NHANAT1\0", 5, 1, 4, 9, 0x12345678, b"s" * 32),
        {313: {"body_index": 20, "layer": 1, "flags": 0,
               "vertices6": vertices, "faces": faces}},
    )
    source_sha = _sha(source)
    bindings = {}
    for rib_id, ids in ((48, [0, 1, 2]), (49, [0, 2, 3])):
        member, member_sha, channel_index, _ = _RUNTIME_ATTACHMENT_RIBS[rib_id]
        bindings[f"intercostal_313_rib{rib_id}_motion_attachment"] = {
            "schema": f"numi.human.intercostal313-rib{rib_id}-interface-motion-attachment.v1",
            "anatomy_payload_sha256": source_sha,
            "surface": {
                "semantic": 51010, "stable_id": 313, "body_index": 20, "layer": 1,
                "local_vertex_count": len(vertices),
                "interface_local_vertex_ids": ids,
                "interface_local_vertex_ids_sha256": _sha(
                    np.asarray(ids, dtype="<u4").tobytes()),
            },
            "existing_rib_channel": {
                "stable_id": rib_id, "semantic": 51004, "body_index": 20,
                "channel_kind": 5, "channel_index": channel_index,
                "source_member_id": member, "source_member_sha256": member_sha,
            },
            "source_partition": {"native_admission": False},
        }
    receipt = json.dumps({
        "schema": "numi.human.resting-anatomy-receipt.v1",
        "payload": {"sha256": source_sha},
        "functional_bindings": bindings,
    }, sort_keys=True, separators=(",", ":")).encode()
    receipt_sha = _sha(receipt)
    monkeypatch.setattr(owner, "LOCAL_RECONSTRUCTION_973_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(owner, "LOCAL_RECONSTRUCTION_973_PAYLOAD_SHA256", source_sha)

    face_owners, report = derive_local_reconstruction_973_runtime_face_owners(
        receipt, receipt_sha, source, source_sha, stable_id=313,
    )
    assert face_owners == {0: 48, 1: None, 2: 49}
    assert report["authenticated_runtime_map_rib_ids"] == [48, 49]
    assert report["geometric_proximity_used_for_ownership"] is False
    assert report["anatomical_admission"] is False

    bad_bindings = json.loads(receipt)
    bad_bindings["functional_bindings"][
        "intercostal_313_rib48_motion_attachment"
    ]["surface"]["interface_local_vertex_ids_sha256"] = "f" * 64
    bad_receipt = json.dumps(
        bad_bindings, sort_keys=True, separators=(",", ":")
    ).encode()
    monkeypatch.setattr(owner, "LOCAL_RECONSTRUCTION_973_RECEIPT_SHA256", _sha(bad_receipt))
    with pytest.raises(ValueError, match="source vertex map differs"):
        derive_local_reconstruction_973_runtime_face_owners(
            bad_receipt, _sha(bad_receipt), source, source_sha, stable_id=313,
        )


def test_mixed_bone_tool_registry_keeps_only_ribs_and_rejects_ambiguous_ids():
    from numilab_human.resting_intercostal_mesh_arrangement import (
        _parse_local_reconstruction_source_reports,
    )

    source_face = _source_face_report()
    registry = _mixed_source_tool_report()
    tools, claims = _parse_local_reconstruction_source_reports([source_face, registry])
    assert set(tools) == {900000000 + rib_id for rib_id in range(48, 60)}
    assert tools[900000048]["rib_stable_id"] == 48
    assert tools[900000048]["face_count"] == 20
    assert _sha(source_face) in claims
    malformed = json.loads(registry)
    malformed["rows"][0]["source_bone"] = [51004, 27]
    with pytest.raises(ValueError, match="identity is malformed or unsupported"):
        _parse_local_reconstruction_source_reports([
            source_face,
            json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode(),
        ])
    with pytest.raises(ValueError, match="ambiguous across authenticated operation reports"):
        _parse_local_reconstruction_source_reports([registry, registry])


def test_parent_face_owners_bind_parent_operations_and_typed_source_ancestry():
    parent = b"pinned parent candidate bytes"
    ancestry = _source_face_report()
    tool = _mixed_source_tool_report()
    reports = [ancestry, tool]
    report_pins = [_sha(value) for value in reports]
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    raw = _owner_report(parent, ancestry, faces)
    owners, summary = _normalize_local_reconstruction_parent_face_owners(
        raw, _sha(raw), stable_id=313, parent_faces=faces,
        parent_payload_raw=parent, operation_ancestry_raw=ancestry,
        authenticated_source_report_raws=reports,
        authenticated_source_report_sha256s=report_pins,
    )
    assert owners == {0: 48}
    assert summary["explicit_parent_face_owner_count"] == 1

    with pytest.raises(ValueError, match="parent payload hash"):
        _normalize_local_reconstruction_parent_face_owners(
            raw, _sha(raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent + b"drift", operation_ancestry_raw=ancestry,
            authenticated_source_report_raws=reports,
            authenticated_source_report_sha256s=report_pins,
        )
    with pytest.raises(ValueError, match="operation ancestry hash"):
        _normalize_local_reconstruction_parent_face_owners(
            raw, _sha(raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=ancestry + b"drift",
            authenticated_source_report_raws=reports,
            authenticated_source_report_sha256s=report_pins,
        )

    bad_report = json.loads(raw)
    bad_report["parent_face_owners"][0]["evidence"][0]["source_report_sha256"] = "b" * 64
    bad_raw = json.dumps(bad_report, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="does not match supplied source report bytes"):
        _normalize_local_reconstruction_parent_face_owners(
            bad_raw, _sha(bad_raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=ancestry,
            authenticated_source_report_raws=reports,
            authenticated_source_report_sha256s=report_pins,
        )

    wrong_face = json.loads(raw)
    wrong_face["parent_face_owners"][0]["evidence"][0]["source_face_ids"] = [18]
    wrong_face_raw = json.dumps(wrong_face, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="not proven by the pinned typed source-face ancestry"):
        _normalize_local_reconstruction_parent_face_owners(
            wrong_face_raw, _sha(wrong_face_raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=ancestry,
            authenticated_source_report_raws=reports,
            authenticated_source_report_sha256s=report_pins,
        )

    moved_face_report = json.loads(ancestry)
    moved_face_report["faces"][0]["output_face"] = 1
    moved_ancestry = json.dumps(
        moved_face_report, sort_keys=True, separators=(",", ":")
    ).encode()
    moved_reports = [moved_ancestry, tool]
    moved_owner = _owner_report(parent, moved_ancestry, faces)
    with pytest.raises(ValueError, match="not proven by the pinned typed source-face ancestry"):
        _normalize_local_reconstruction_parent_face_owners(
            moved_owner, _sha(moved_owner), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=moved_ancestry,
            authenticated_source_report_raws=moved_reports,
            authenticated_source_report_sha256s=[_sha(value) for value in moved_reports],
        )

    wrong_tool = _source_tool_report(49)
    wrong_reports = [ancestry, wrong_tool]
    with pytest.raises(ValueError, match="not proven by the pinned typed source-face ancestry"):
        _normalize_local_reconstruction_parent_face_owners(
            raw, _sha(raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=ancestry,
            authenticated_source_report_raws=wrong_reports,
            authenticated_source_report_sha256s=[_sha(value) for value in wrong_reports],
        )

    with pytest.raises(ValueError, match="differ from their independent pins"):
        _normalize_local_reconstruction_parent_face_owners(
            raw, _sha(raw), stable_id=313, parent_faces=faces,
            parent_payload_raw=parent, operation_ancestry_raw=ancestry,
            authenticated_source_report_raws=reports,
            authenticated_source_report_sha256s=["0" * 64, report_pins[1]],
        )


def test_conflicting_parent_rib_owners_remain_unowned():
    owners = derive_local_reconstruction_face_rib_owners(
        [[0, 1]], {0: 48, 1: 49}, parent_face_count=2,
    )
    assert owners == [None]


def test_body20_pose_comes_from_receipt_bound_to_source_and_pack(tmp_path: Path):
    source_sha = "a" * 64
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"accepted pack")
    receipt = tmp_path / "step-0.receipt.json"
    raw = json.dumps(_accepted_receipt(_sha(pack.read_bytes()), source_sha)).encode()
    receipt.write_bytes(raw)
    q, t, receipt_sha, pack_sha = _authenticated_local_reconstruction_body20_pose(
        raw, expected_source_payload_sha256=source_sha,
        accepted_receipt_path=receipt, accepted_capture_pack_path=pack,
    )
    assert q.tobytes() == np.asarray([0, 0, 0, 1], dtype="<f4").tobytes()
    assert t.tobytes() == np.asarray([0.1, 0.2, 0.3], dtype="<f4").tobytes()
    assert receipt_sha == _sha(raw) and pack_sha == _sha(pack.read_bytes())

    drifted = json.dumps(_accepted_receipt(_sha(pack.read_bytes()), "b" * 64)).encode()
    receipt.write_bytes(drifted)
    with pytest.raises(ValueError, match="does not bind the exact source anatomy"):
        _authenticated_local_reconstruction_body20_pose(
            drifted, expected_source_payload_sha256=source_sha,
            accepted_receipt_path=receipt, accepted_capture_pack_path=pack,
        )


def test_body20_pose_rejects_pack_drift(tmp_path: Path):
    source_sha = "a" * 64
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"accepted pack")
    receipt = tmp_path / "step-0.receipt.json"
    raw = json.dumps(_accepted_receipt(_sha(pack.read_bytes()), source_sha)).encode()
    receipt.write_bytes(raw)
    pack.write_bytes(b"changed pack")
    with pytest.raises(ValueError, match="does not authenticate the pinned capture"):
        _authenticated_local_reconstruction_body20_pose(
            raw, expected_source_payload_sha256=source_sha,
            accepted_receipt_path=receipt, accepted_capture_pack_path=pack,
        )


def test_body20_pose_rejects_nonzero_accepted_step(tmp_path: Path):
    source_sha = "a" * 64
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"accepted pack")
    receipt = tmp_path / "step-0.receipt.json"
    value = _accepted_receipt(_sha(pack.read_bytes()), source_sha)
    value["accepted_step"] = 1
    raw = json.dumps(value).encode()
    receipt.write_bytes(raw)
    with pytest.raises(ValueError, match="requires the exact accepted step-0 receipt"):
        _authenticated_local_reconstruction_body20_pose(
            raw, expected_source_payload_sha256=source_sha,
            accepted_receipt_path=receipt, accepted_capture_pack_path=pack,
        )


def test_attachment_inventory_must_equal_observed_authenticated_owners():
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<u4")
    owners = [48, 58]
    interface = np.asarray([True, True], dtype=np.bool_)
    assert derive_local_reconstruction_attachment_maps(
        faces, owners, interface, vertex_count=6, required_rib_ids=[48, 58],
    ) == {48: [0, 1, 2], 58: [3, 4, 5]}
    with pytest.raises(ValueError, match="differs from observed authenticated"):
        derive_local_reconstruction_attachment_maps(
            faces, owners, interface, vertex_count=6,
            required_rib_ids=list(range(48, 60)),
        )


def test_attachment_map_rejects_non_vector_boolean_mask():
    with pytest.raises(ValueError, match="one-dimensional boolean array"):
        derive_local_reconstruction_attachment_maps(
            np.asarray([[0, 1, 2]], dtype="<u4"),
            [48],
            np.asarray([[True]], dtype=np.bool_),
            vertex_count=3,
            required_rib_ids=[48],
        )


def test_attachment_maps_reject_indices_outside_uint32_before_cast():
    faces = np.asarray([[0, 1, 2**32]], dtype="<u8")
    owners = [48]
    interface = np.asarray([True], dtype=np.bool_)
    with pytest.raises(ValueError, match="exceed the ABI5 uint32 range"):
        derive_local_reconstruction_attachment_maps(
            faces, owners, interface, vertex_count=2**32 + 1, required_rib_ids=[48],
        )


def test_shared_interface_vertices_with_conflicting_rib_owners_stay_unmapped():
    faces = np.asarray([[0, 1, 2], [2, 3, 4]], dtype="<u4")
    owners = [48, 58]
    interface = np.asarray([True, True], dtype=np.bool_)
    maps, diagnostics = derive_local_reconstruction_attachment_maps(
        faces, owners, interface, vertex_count=5, required_rib_ids=[48, 58],
        return_diagnostics=True,
    )
    assert maps == {48: [0, 1], 58: [3, 4]}
    assert diagnostics == {
        "ambiguous_shared_interface_vertex_count": 1,
        "ambiguous_shared_interface_vertices": [
            {"local_vertex_id": 2, "rib_stable_ids": [48, 58]},
        ],
    }


def test_finite_zero_normal_is_allowed_only_for_unused_vertex_slot():
    normals = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    localized = _localize_body20_normals_f32(
        normals, faces, np.asarray([0, 0, 0, 1], dtype="<f4"),
    )
    assert np.all(np.isfinite(localized))
    assert np.array_equal(localized[3], np.zeros(3, dtype="<f4"))
    faces_with_unused_zero = np.asarray([[0, 1, 3]], dtype="<u4")
    with pytest.raises(ValueError, match="referenced Body20 normal is zero"):
        _localize_body20_normals_f32(
            normals, faces_with_unused_zero, np.asarray([0, 0, 0, 1], dtype="<f4"),
        )


def test_local_reconstruction_compiler_pins_source_and_preserves_unselected_rows(tmp_path: Path):
    vertices = np.asarray([
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
    ], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    candidate_positions = np.vstack((
        vertices[:, :3],
        np.asarray([[2, 2, 2]], dtype="<f4"),
    ))
    unrelated_vertices = np.asarray([
        [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1],
    ], dtype="<f4")
    rows = {
        313: {"body_index": 20, "layer": 1, "flags": 0,
              "vertices6": vertices, "faces": faces},
        900: {"body_index": 10, "layer": 1, "flags": 0,
              "vertices6": unrelated_vertices, "faces": faces},
    }
    source = encode_payload(
        (b"NHANAT1\0", 5, 2, 6, 6, 0x12345678, b"s" * 32), rows,
    )
    source_sha = _sha(source)
    candidate_buffer = __import__("io").BytesIO()
    np.savez(candidate_buffer, positions_world_f32=candidate_positions, faces=faces)
    candidate_raw = candidate_buffer.getvalue()
    parent_buffer = __import__("io").BytesIO()
    np.savez(parent_buffer, faces=faces)
    parent_raw = parent_buffer.getvalue()
    ancestry = _source_face_report()
    tool_report = _source_tool_report()
    owner_raw = _owner_report(parent_raw, ancestry, faces)

    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"fixture accepted capture pack")
    receipt_path = tmp_path / "step-0.receipt.json"
    receipt_raw = json.dumps(
        _accepted_receipt(_sha(pack.read_bytes()), source_sha)
    ).encode()
    receipt_path.write_bytes(receipt_raw)

    spec = {
        "candidate_npz_raw": candidate_raw,
        "candidate_npz_sha256": _sha(candidate_raw),
        "world_positions_f32": candidate_positions,
        "faces": faces,
        "face_parent_ids": [[0]],
        "interface_face_mask": np.asarray([True], dtype=np.bool_),
        "parent_faces": faces,
        "parent_faces_key": "faces",
        "parent_payload_raw": parent_raw,
        "operation_ancestry_raw": ancestry,
        "authenticated_source_report_raws": [ancestry, tool_report],
        "authenticated_source_report_sha256s": [_sha(ancestry), _sha(tool_report)],
        "parent_face_owner_report_raw": owner_raw,
        "parent_face_owner_report_sha256": _sha(owner_raw),
        "required_rib_ids": [48],
    }
    output, report = compile_local_reconstruction_intercostal_payload(
        source, {313: spec},
        expected_source_payload_sha256=source_sha,
        accepted_receipt_raw=receipt_raw,
        accepted_receipt_path=receipt_path,
        accepted_capture_pack_path=pack,
    )
    before_header, _records, before_rows = _parse_payload(source)
    after_header, _records, after_rows = _parse_payload(output)
    assert after_header[5:] == before_header[5:]
    assert report["unmodified_rows_byte_exact"] == 1
    np.testing.assert_array_equal(after_rows[900]["vertices6"], before_rows[900]["vertices6"])
    np.testing.assert_array_equal(after_rows[900]["faces"], before_rows[900]["faces"])
    assert report["row_reports"]["313"]["observed_rib_channel_ids"] == [48]
    sidecar = report["runtime_attachment_map_sidecars"]["313"]["48"]
    assert sidecar["surface"]["stable_id"] == 313
    assert sidecar["surface"]["interface_local_vertex_ids"] == report[
        "runtime_attachment_maps"]["313"][48]
    assert sidecar["existing_rib_channel"]["source_member_id"] == "FJ3228"
    assert sidecar["existing_rib_channel"]["channel_index"] == 12
    assert sidecar["surface"]["interface_local_vertex_ids_sha256"] == _sha(
        np.asarray(sidecar["surface"]["interface_local_vertex_ids"], dtype="<u4").tobytes()
    )
    assert sidecar["source_partition"]["base_anatomy_payload_sha256"] == source_sha
    assert sidecar["source_partition"]["parent_owner_report_sha256"] == _sha(owner_raw)
    assert sidecar["source_partition"]["native_admission"] is False
    assert after_rows[313]["vertices6"].shape == (4, 6)
    np.testing.assert_array_equal(after_rows[313]["vertices6"][3, 3:], np.zeros(3, dtype="<f4"))
    assert report["anatomical_admission"] is False
    assert report["native_admission"] is False

    diagnostic_spec = {
        "candidate_npz_raw": candidate_raw,
        "candidate_npz_sha256": _sha(candidate_raw),
        "world_positions_f32": candidate_positions,
        "faces": faces,
        "source_emission_diagnostic_only": True,
    }
    diagnostic_output, diagnostic_report = compile_local_reconstruction_intercostal_payload(
        source, {313: diagnostic_spec},
        expected_source_payload_sha256=source_sha,
        accepted_receipt_raw=receipt_raw,
        accepted_receipt_path=receipt_path,
        accepted_capture_pack_path=pack,
    )
    _header, _records, diagnostic_rows = _parse_payload(diagnostic_output)
    assert diagnostic_rows[313]["vertices6"].shape == (4, 6)
    assert diagnostic_report["status"] == "PASS_ABI5_SOURCE_EMISSION_DIAGNOSTIC_ONLY"
    assert diagnostic_report["attachment_ownership_unresolved_stable_ids"] == [313]
    assert diagnostic_report["runtime_attachment_maps"] == {"313": {}}
    assert diagnostic_report["runtime_attachment_map_sidecars"] == {"313": {}}
    assert diagnostic_report["row_reports"]["313"]["unowned_child_face_count"] == 1
    assert diagnostic_report["native_admission"] is False

    mismatched_diagnostic_spec = {
        **diagnostic_spec,
        "world_positions_f32": candidate_positions.copy(),
    }
    mismatched_diagnostic_spec["world_positions_f32"][0, 0] += 1.0
    with pytest.raises(ValueError, match="diagnostic arrays differ from the exact NPZ"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: mismatched_diagnostic_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    reshaped_diagnostic_spec = {
        **diagnostic_spec,
        "world_positions_f32": candidate_positions.reshape(-1),
    }
    with pytest.raises(ValueError, match="diagnostic arrays differ from the exact NPZ"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: reshaped_diagnostic_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    wrapped_diagnostic_faces = np.asarray(faces, dtype=np.uint64).copy()
    wrapped_diagnostic_faces[0, 0] += 2**32
    wrapped_diagnostic_spec = {
        **diagnostic_spec,
        "faces": wrapped_diagnostic_faces,
    }
    with pytest.raises(ValueError, match="diagnostic arrays differ from the exact NPZ"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: wrapped_diagnostic_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    float64_npz_buffer = __import__("io").BytesIO()
    np.savez(
        float64_npz_buffer,
        positions_world_f32=candidate_positions.astype("<f8"),
        faces=faces,
    )
    float64_raw = float64_npz_buffer.getvalue()
    float64_spec = {
        **spec,
        "candidate_npz_raw": float64_raw,
        "candidate_npz_sha256": _sha(float64_raw),
    }
    with pytest.raises(ValueError, match="candidate NPZ positions must be finite typed Float32"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: float64_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    bad_npz_buffer = __import__("io").BytesIO()
    np.savez(
        bad_npz_buffer,
        positions_world_f32=candidate_positions,
        faces=np.asarray([[0, 1, 2**32]], dtype="<u8"),
    )
    bad_npz_raw = bad_npz_buffer.getvalue()
    bad_npz_spec = {
        **spec,
        "candidate_npz_raw": bad_npz_raw,
        "candidate_npz_sha256": _sha(bad_npz_raw),
    }
    with pytest.raises(ValueError, match="candidate NPZ face indices exceed the ABI5 uint32 range"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: bad_npz_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    bad_parent_buffer = __import__("io").BytesIO()
    np.savez(bad_parent_buffer, faces=np.asarray([[0, 1, 2**32]], dtype="<u8"))
    bad_parent_raw = bad_parent_buffer.getvalue()
    bad_parent_spec = {**spec, "parent_payload_raw": bad_parent_raw}
    with pytest.raises(ValueError, match="parent NPZ face indices exceed the ABI5 uint32 range"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: bad_parent_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )

    with pytest.raises(ValueError, match="source ABI5 payload bytes differ"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: spec},
            expected_source_payload_sha256="f" * 64,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )


def test_local_reconstruction_rebind_preserves_authenticated_source_row_positions(tmp_path: Path):
    local_vertices = np.asarray([
        [0.1234567, 0.2, 0.3, 0, 0, 1],
        [1.234567, 0.2, 0.3, 0, 0, 1],
        [0.1234567, 1.2, 0.3, 0, 0, 1],
        [2.345678, 2.2, 2.3, 0, 0, 0],
    ], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    unrelated_vertices = np.asarray([
        [0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1],
    ], dtype="<f4")
    rows = {
        313: {"body_index": 20, "layer": 1, "flags": 0,
              "vertices6": local_vertices, "faces": faces},
        900: {"body_index": 10, "layer": 1, "flags": 0,
              "vertices6": unrelated_vertices, "faces": faces},
    }
    source = encode_payload(
        (b"NHANAT1\0", 5, 2, 7, 6, 0x12345678, b"s" * 32), rows,
    )
    source_sha = _sha(source)
    local_xyz = np.ascontiguousarray(local_vertices[:, :3], dtype="<f4")
    translation = np.asarray([0.1, 0.2, 0.3], dtype="<f4")
    candidate_positions = np.ascontiguousarray(local_xyz + translation, dtype="<f4")
    candidate_buffer = __import__("io").BytesIO()
    np.savez(candidate_buffer, positions_world_f32=candidate_positions, faces=faces)
    candidate_raw = candidate_buffer.getvalue()
    parent_buffer = __import__("io").BytesIO()
    np.savez(parent_buffer, faces=faces)
    parent_raw = parent_buffer.getvalue()
    ancestry = _source_face_report()
    tool_report = _source_tool_report()
    owner_raw = _owner_report(parent_raw, ancestry, faces)
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"fixture accepted capture pack")
    receipt_path = tmp_path / "step-0.receipt.json"
    receipt_raw = json.dumps(
        _accepted_receipt(_sha(pack.read_bytes()), source_sha)
    ).encode()
    receipt_path.write_bytes(receipt_raw)

    spec = {
        "candidate_npz_raw": candidate_raw,
        "candidate_npz_sha256": _sha(candidate_raw),
        "world_positions_f32": candidate_positions,
        "faces": faces,
        "face_parent_ids": [[0]],
        "interface_face_mask": np.asarray([True], dtype=np.bool_),
        "parent_faces": faces,
        "parent_faces_key": "faces",
        "parent_payload_raw": parent_raw,
        "operation_ancestry_raw": ancestry,
        "authenticated_source_report_raws": [ancestry, tool_report],
        "authenticated_source_report_sha256s": [_sha(ancestry), _sha(tool_report)],
        "parent_face_owner_report_raw": owner_raw,
        "parent_face_owner_report_sha256": _sha(owner_raw),
        "required_rib_ids": [48],
        "preserve_source_row_local_positions": True,
        "source_row_local_positions_sha256": _sha(local_xyz.tobytes()),
    }
    output, report = compile_local_reconstruction_intercostal_payload(
        source, {313: spec},
        expected_source_payload_sha256=source_sha,
        accepted_receipt_raw=receipt_raw,
        accepted_receipt_path=receipt_path,
        accepted_capture_pack_path=pack,
    )
    _header, _records, before_rows = _parse_payload(source)
    _header, _records, after_rows = _parse_payload(output)
    assert after_rows[313]["vertices6"][:, :3].tobytes() == before_rows[313][
        "vertices6"][:, :3].tobytes()
    assert report["source_local_positions_preserved_stable_ids"] == [313]
    binding = report["row_reports"]["313"]["body20_source_position_binding"]
    assert binding["mode"] == "authenticated_source_row_local_positions_retained"
    assert binding["source_position_inverse_performed"] is False
    assert binding["source_local_positions_byte_exact"] is True
    assert report["row_reports"]["313"]["source_row_local_positions_sha256"] == _sha(
        local_xyz.tobytes())
    np.testing.assert_array_equal(after_rows[900]["vertices6"], before_rows[900]["vertices6"])
    assert report["anatomical_admission"] is False
    assert report["native_admission"] is False

    bad_spec = {**spec, "source_row_local_positions_sha256": "f" * 64}
    with pytest.raises(ValueError, match="source-row local positions differ from independent pin"):
        compile_local_reconstruction_intercostal_payload(
            source, {313: bad_spec},
            expected_source_payload_sha256=source_sha,
            accepted_receipt_raw=receipt_raw,
            accepted_receipt_path=receipt_path,
            accepted_capture_pack_path=pack,
        )


def test_body20_inverse_searches_adjacent_float32_ulps():
    from numilab_human.resting_intercostal_mesh_arrangement import (
        _body20_rotation64,
        _inverse_body20_positions_f32,
        _load_local_body20_transform,
    )

    q = np.asarray([0.137, -0.219, 0.314, 0.912], dtype="<f4")
    q = np.ascontiguousarray(q / np.linalg.norm(q), dtype="<f4")
    t = np.asarray([0.173, -0.287, 0.091], dtype="<f4")
    local = np.random.default_rng(8124).normal(0.0, 0.7, size=(256, 3)).astype("<f4")
    transform = _load_local_body20_transform()
    world = np.ascontiguousarray(transform(q, t, local), dtype="<f4")

    rotation = _body20_rotation64(q)
    base_local = np.ascontiguousarray(
        np.linalg.solve(rotation, (world.astype(np.float64) - t.astype(np.float64)).T).T,
        dtype="<f4",
    )
    base_replay = np.ascontiguousarray(transform(q, t, base_local), dtype="<f4")
    recovered, replay, report = _inverse_body20_positions_f32(world, q, t)

    base_delta = base_replay.astype(np.float64) - world.astype(np.float64)
    selected_delta = replay.astype(np.float64) - world.astype(np.float64)
    assert np.all(np.linalg.norm(selected_delta, axis=1) <= np.linalg.norm(base_delta, axis=1))
    assert report["neighbor_search"]["offsets_per_vertex"] == 27
    assert report["neighbor_search"]["exact_replay_vertex_count_after_search"] > (
        report["neighbor_search"]["exact_replay_vertex_count_from_base"]
    )
    assert np.count_nonzero(recovered.view("<u4") != base_local.view("<u4")) > 0
    assert np.isfinite(recovered).all() and np.isfinite(replay).all()


def _runtime_attachment_fixture():
    from numilab_human.resting_intercostal_mesh_arrangement import (
        _local_reconstruction_runtime_attachment_map_sidecars,
        bind_local_reconstruction_runtime_attachment_receipt,
    )

    vertices = np.asarray([
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
    ], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    row = {"body_index": 20, "layer": 1, "flags": 0, "vertices6": vertices, "faces": faces}
    from numilab_human.resting_anatomy_interface_patch import encode_payload
    payload = encode_payload((b"NHANAT1\0", 5, 1, 3, 3, 0x12345678, b"s" * 32), {315: row})
    payload_sha = _sha(payload)
    maps = {58: [0, 1, 2]}
    sidecars = _local_reconstruction_runtime_attachment_map_sidecars(
        nha_stable_id=315,
        interface_maps=maps,
        vertex_count=3,
        face_rib_owners=[58],
        interface_face_mask=np.asarray([True], dtype=np.bool_),
        source_payload_sha256="a" * 64,
        compiled_payload_sha256=payload_sha,
        candidate_npz_sha256="b" * 64,
        candidate_faces_sha256=_sha(faces.tobytes()),
        face_parent_lineage_sha256="c" * 64,
        accepted_receipt_sha256="d" * 64,
        accepted_capture_pack_sha256="e" * 64,
        parent_owner_report={
            "parent_owner_report_sha256": "f" * 64,
            "parent_payload_sha256": "1" * 64,
            "operation_ancestry_sha256": "2" * 64,
            "authenticated_source_report_sha256s": ["3" * 64],
        },
    )
    receipt = {
        "payload": {"sha256": payload_sha},
        "provenance": {"bones_payload_sha256": "4" * 64},
        "functional_bindings": {
            "anatomy_payload_sha256": payload_sha,
            "bones_payload_sha256": "4" * 64,
        },
    }
    return bind_local_reconstruction_runtime_attachment_receipt, receipt, payload, maps, sidecars


def test_runtime_attachment_sidecar_binds_typed_registry_and_exact_slots():
    bind, receipt, payload, maps, sidecars = _runtime_attachment_fixture()
    result = bind(
        receipt, payload, maps, nha_stable_id=315,
        bones_payload_sha256="4" * 64,
        runtime_attachment_map_sidecars=sidecars,
    )
    attachment = receipt["functional_bindings"]["intercostal_315_rib58_motion_attachment"]
    assert attachment["surface"]["interface_local_vertex_ids"] == [0, 1, 2]
    assert attachment["surface"]["local_vertex_count"] == 3
    assert attachment["source_partition"]["interface_vertex_count"] == 3
    assert attachment["existing_rib_channel"]["source_member_id"] == "FJ3226"
    assert result["interface_maps_validated_by_existing_runtime_owner"] is True
    assert receipt["functional_bindings"]["anatomy_payload_sha256"] == result[
        "compiled_payload_sha256"
    ]


@pytest.mark.parametrize("mutation", ["rib_member", "rib_channel", "slot_sha"])
def test_runtime_attachment_sidecar_rejects_typed_identity_or_slot_hash_drift(mutation):
    bind, receipt, payload, maps, sidecars = _runtime_attachment_fixture()
    altered = json.loads(json.dumps(sidecars))
    attachment = altered["58"]
    if mutation == "rib_member":
        attachment["existing_rib_channel"]["source_member_id"] = "wrong-member"
    elif mutation == "rib_channel":
        attachment["existing_rib_channel"]["channel_index"] = 59
    else:
        attachment["surface"]["interface_local_vertex_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime rib58 attachment identity"):
        bind(
            receipt, payload, maps, nha_stable_id=315,
            bones_payload_sha256="4" * 64,
            runtime_attachment_map_sidecars=altered,
        )

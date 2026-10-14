import hashlib

import numpy as np
import pytest

from numilab_human import resting_brachiocephalic_transfer_audit as audit


def _row(stable_id, faces=None, *, offset=0, body=20, layer=2, flags=0):
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    if faces is None:
        faces = np.asarray([[0, 1, 2], [0, 3, 1]], dtype="<i8")
    return {
        "record": (
            body,
            offset,
            len(vertices),
            offset * 3,
            len(faces) * 3,
            stable_id,
            layer,
            flags,
        ),
        "body_index": body,
        "layer": layer,
        "flags": flags,
        "vertices6": vertices,
        "faces": np.asarray(faces, dtype="<i8"),
    }


def _identity(stable_id=7, *, source_member="FJ3411", body=20, layer=2):
    return {
        "source_member": source_member,
        "source_sha256": "a" * 64,
        "body_index": body,
        "layer": layer,
        "name": "arch of aorta",
        "provider": "pinned source provider",
        "source_owner_metadata": {
            "member_id": source_member,
            "core_body_index": body,
            "layer": "vessel",
        },
        "vascular_compartment_id": 4,
        "repair": None,
    }


def _surface(row):
    return (
        row["vertices6"][:, :3].copy(),
        np.asarray(row["faces"], dtype="<u4"),
        row["vertices6"][:, 3:6].copy(),
        np.arange(len(row["vertices6"]), dtype=np.uint64),
    )


def test_normalized_row_bytes_ignore_only_absolute_buffer_offsets():
    first = _row(7, offset=0)
    shifted = _row(7, offset=81000)

    assert audit._canonical_row_bytes(first) == audit._canonical_row_bytes(shifted)
    changed = _row(7, offset=81000, flags=1)
    assert audit._canonical_row_bytes(first) != audit._canonical_row_bytes(changed)


def test_candidate_row_can_only_be_exact_reported_retained_face_projection():
    source = _row(313)
    candidate = _row(313, source["faces"][[0]], offset=400)
    repair = {
        "source_face_count": 2,
        "retained_face_count": 1,
        "removed_source_face_ids": [1],
    }

    retained = audit._retained_face_ids(source, candidate, repair)

    np.testing.assert_array_equal(retained, [0])
    altered = _row(313, np.asarray([[0, 2, 1]]), offset=400)
    with pytest.raises(ValueError, match="retained N27 source-face ancestry"):
        audit._retained_face_ids(source, altered, repair)


def test_transfer_binding_checks_row_bytes_identity_and_primitive_owner():
    source = _row(7)
    metadata = _identity()
    verifier = object()
    report, projected = audit._audit_capture_row_bindings(
        surfaces={(51011, 7): _surface(source)},
        captured_owners={(51011, 7): 20},
        transfer_rows={7: source},
        baseline_rows={7: _row(7, offset=900)},
        candidate_rows={7: _row(7, offset=1200)},
        baseline_source_map={"7": metadata},
        candidate_source_map={"7": dict(metadata)},
        repair_surfaces={},
        verifier=verifier,
        transfer_label="N22 late transfer capture",
    )

    assert report["passed"]
    assert report["whole_payload_hash_mismatch_is_not_used_as_row_failure"]
    assert report["per_target_rows"][0][
        "transfer_row_matches_N27_by_exact_normalized_ABI5_bytes"
    ]
    assert report["per_target_rows"][0]["source_identity_matches_N27"]
    assert report["per_target_rows"][0][
        "transfer_source_identity_resolved_through_exact_N27_row_match"
    ]
    assert report["per_target_rows"][0]["captured_primitive_owner"] == 20
    np.testing.assert_array_equal(projected[(51011, 7)][1], source["faces"])


def test_row_content_mismatch_fails_even_when_capture_owner_matches():
    transfer = _row(7)
    baseline = _row(7)
    baseline["vertices6"][0, 0] = np.float32(0.125)
    identity = _identity()

    report, _ = audit._audit_capture_row_bindings(
        surfaces={(51011, 7): _surface(transfer)},
        captured_owners={(51011, 7): 20},
        transfer_rows={7: transfer},
        baseline_rows={7: baseline},
        candidate_rows={7: _row(7)},
        baseline_source_map={"7": identity},
        candidate_source_map={"7": dict(identity)},
        repair_surfaces={},
        verifier=object(),
        transfer_label="N22 late transfer capture",
    )

    assert not report["passed"]
    assert any(
        "transfer source row bytes differ" in row["reason"]
        for row in report["failures"]
    )


def test_owner_mismatch_is_not_hidden_by_matching_row_bytes():
    row = _row(7)
    identity = _identity()

    report, _ = audit._audit_capture_row_bindings(
        surfaces={(51011, 7): _surface(row)},
        captured_owners={(51011, 7): 21},
        transfer_rows={7: row},
        baseline_rows={7: _row(7, offset=1)},
        candidate_rows={7: _row(7, offset=2)},
        baseline_source_map={"7": identity},
        candidate_source_map={"7": dict(identity)},
        repair_surfaces={},
        verifier=object(),
        transfer_label="N22 late transfer capture",
    )

    assert not report["passed"]
    assert any("rigid source body" in row["reason"] for row in report["failures"])


def test_repaired_target_uses_central_retained_source_face_helper_before_projection():
    source = _row(313, layer=1)
    candidate = _row(313, source["faces"][[0]], offset=100, layer=1)
    identity = _identity(source_member="FJ1451M", layer=1)
    identity["name"] = "external intercostal muscle"
    identity["source_owner_metadata"] = None
    repair = {
        "source_face_count": 2,
        "retained_face_count": 1,
        "removed_source_face_ids": [1],
    }

    class P6Stub:
        called = False

        def retain_bound_source_faces(
            self, key, surface, source_faces, offset, selected
        ):
            self.called = True
            assert key == (51010, 313)
            assert offset == 0
            np.testing.assert_array_equal(source_faces, source["faces"])
            np.testing.assert_array_equal(selected, [0])
            projected = (surface[0], surface[1][selected], surface[2], surface[3])
            return projected, {
                "all_ordered_source_facets_match": True,
                "captured_positions_normals_and_vertex_ids_unchanged": True,
                "source_face_count": 2,
                "retained_source_face_count": 1,
                "source_capture_vertex_namespace_offset": 0,
            }

    verifier = P6Stub()
    report, projected = audit._audit_capture_row_bindings(
        surfaces={(51010, 313): _surface(source)},
        captured_owners={(51010, 313): audit.DEFORMABLE_OWNER},
        transfer_rows={313: source},
        baseline_rows={313: _row(313, offset=1, layer=1)},
        candidate_rows={313: candidate},
        baseline_source_map={"313": identity},
        candidate_source_map={"313": dict(identity)},
        repair_surfaces={"313": repair},
        verifier=verifier,
        transfer_label="N27 accepted capture",
    )

    assert verifier.called
    assert report["retained_face_projection"][0]["retained_source_face_count"] == 1
    assert report["retained_face_projection"][0]["owner"] == (
        "central HP97/P6 retain_bound_source_faces"
    )
    np.testing.assert_array_equal(projected[(51010, 313)][1], candidate["faces"])


def test_source_identity_change_is_reported_as_a_transfer_binding_failure():
    row = _row(7)
    baseline_identity = _identity()
    candidate_identity = _identity(source_member="FJ9999")

    report, _ = audit._audit_capture_row_bindings(
        surfaces={(51011, 7): _surface(row)},
        captured_owners={(51011, 7): 20},
        transfer_rows={7: row},
        baseline_rows={7: _row(7, offset=1)},
        candidate_rows={7: _row(7, offset=2)},
        baseline_source_map={"7": baseline_identity},
        candidate_source_map={"7": candidate_identity},
        repair_surfaces={},
        verifier=object(),
        transfer_label="N22 late transfer capture",
    )

    assert not report["passed"]
    assert any("identity differs" in row["reason"] for row in report["failures"])


def test_appended_bridge_rows_match_compiler_source060_xyz_faces_and_owner():
    rows = {stable_id: _row(stable_id, body=20, layer=2) for stable_id in (525, 526)}
    source_map = {}
    preparation = {"bridges": {}}
    for stable_id, spec in audit.source007.BRIDGE_ROWS.items():
        row = rows[stable_id]
        member = spec["member"]
        source_map[str(stable_id)] = {
            "source_member": member,
            "source_sha256": "b" * 64,
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
            "source_owner_metadata": {
                "member_id": member,
                "core_body_index": 20,
                "layer": "vessel",
            },
        }
        preparation["bridges"][member] = {
            "vertex_ancestry": list(range(len(row["vertices6"]))),
            "source_local_xyz_sha256": hashlib.sha256(
                np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4").tobytes()
            ).hexdigest(),
            "source_faces_sha256": hashlib.sha256(
                np.ascontiguousarray(row["faces"], dtype="<i4").tobytes()
            ).hexdigest(),
        }

    result = audit._validate_appended_bridge_owner_inputs(rows, source_map, preparation)

    assert result["passed"]
    assert (
        result["rows"]["525"]["source_local_xyz_sha256"]
        == (
            preparation["bridges"][audit.source007.BRIDGE_ROWS[525]["member"]][
                "source_local_xyz_sha256"
            ]
        )
    )
    changed = {key: dict(row) for key, row in rows.items()}
    changed[526] = dict(rows[526])
    changed[526]["vertices6"] = rows[526]["vertices6"].copy()
    changed[526]["vertices6"][0, 0] = np.float32(0.125)
    with pytest.raises(ValueError, match="compiler's source-060 row pins"):
        audit._validate_appended_bridge_owner_inputs(changed, source_map, preparation)


def test_pose_continuity_routes_only_last_y_junction_through_3d_owner(monkeypatch):
    pairs = tuple(audit.source007.PORT_KEYS)
    calls = []
    surfaces = {key: ("candidate",) for pair in pairs for key in pair}

    def single_port(_verifier, _manifold, **kwargs):
        calls.append(("single", kwargs["bridge_key"], kwargs["target_key"]))
        return {
            "bridge": list(kwargs["bridge_key"]),
            "target": list(kwargs["target_key"]),
            "passed": True,
        }

    class JunctionOwner:
        @staticmethod
        def audit_brachiocephalic_3d_junction(_verifier, _manifold, **kwargs):
            calls.append(("3d", kwargs["bridge_key"], kwargs["target_key"]))
            return {
                "bridge": list(kwargs["bridge_key"]),
                "target": list(kwargs["target_key"]),
                "passed": True,
            }

    monkeypatch.setattr(
        audit.gate, "audit_brachiocephalic_port_pair_continuity", single_port
    )
    monkeypatch.setattr(audit, "_load_3d_junction_owner", lambda: JunctionOwner)
    report = audit._audit_pose_port_continuity(
        object(), object(), surfaces, surfaces, {pair: ((0, 0),) for pair in pairs}
    )

    assert report["passed"]
    assert report["requested_port_count"] == 6
    assert report["evaluated_port_count"] == 6
    assert [call[0] for call in calls] == ["single"] * 5 + ["3d"]
    assert calls[-1][1:] == ((51011, 526), (51011, 472))


def test_pose_continuity_stops_at_first_failed_port(monkeypatch):
    pairs = tuple(audit.source007.PORT_KEYS)
    calls = []
    surfaces = {key: ("candidate",) for pair in pairs for key in pair}

    def single_port(_verifier, _manifold, **kwargs):
        calls.append(kwargs["target_key"])
        passed = len(calls) < 2
        return {
            "bridge": list(kwargs["bridge_key"]),
            "target": list(kwargs["target_key"]),
            "passed": passed,
        }

    class JunctionOwner:
        @staticmethod
        def audit_brachiocephalic_3d_junction(*_args, **_kwargs):
            raise AssertionError("3D owner must not run after an earlier failed port")

    monkeypatch.setattr(
        audit.gate, "audit_brachiocephalic_port_pair_continuity", single_port
    )
    monkeypatch.setattr(audit, "_load_3d_junction_owner", lambda: JunctionOwner)
    report = audit._audit_pose_port_continuity(
        object(), object(), surfaces, surfaces, {pair: ((0, 0),) for pair in pairs}
    )

    assert not report["passed"]
    assert report["requested_port_count"] == 6
    assert report["evaluated_port_count"] == 2
    assert report["first_failed_port"]["target"] == list(pairs[1][1])
    assert len(calls) == 2


def test_capture_initialization_stops_before_prediction_when_row_binding_fails(monkeypatch):
    surfaces = {(51011, 7): object()}
    identity = {"pack_sha256": "a" * 64, "capture_receipt_sha256": "b" * 64}
    meta = {
        "accepted_time_s": 0.0,
        "body20": {"quaternion_xyzw": [0, 0, 0, 1], "position_m": [0, 0, 0]},
        "receipt": {},
    }
    monkeypatch.setattr(
        audit, "_load_capture", lambda *_args, **_kwargs: (surfaces, identity, meta)
    )
    monkeypatch.setattr(
        audit,
        "_verify_capture_common_field",
        lambda *_args, **_kwargs: {"capture_source_anatomy_payload_sha256": "c" * 64},
    )
    monkeypatch.setattr(audit, "_capture_owner_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        audit,
        "_audit_capture_row_bindings",
        lambda **_kwargs: ({"passed": False, "failures": [{"reason": "row mismatch"}]}, {}),
    )
    monkeypatch.setattr(
        audit.source007,
        "_bridge_surfaces",
        lambda *_args, **_kwargs: pytest.fail("prediction must stop after row-binding failure"),
    )
    monkeypatch.setattr(
        audit.source007,
        "_predict_step0_bridges",
        lambda *_args, **_kwargs: pytest.fail("prediction must stop after row-binding failure"),
    )

    record = audit._initialize_capture_record(
        {
            "verifier": object(),
            "retention_owner": object(),
            "preparation": {},
            "baseline_rows": {},
            "candidate_rows": {},
            "baseline_map": {},
            "candidate_map": {},
            "repair_surfaces": {},
            "transform": object(),
        },
        {"pack_path": "/unused/accepted.pack"},
        expected_step=0,
        expected_source_sha="c" * 64,
        transfer_rows={},
        transfer_label="N27 source-0 test capture",
    )

    assert record["passed"] is False
    assert record["status"] == "CAPTURE_DECODED_BUT_SOURCE_BINDING_REJECTED"
    assert record["geometry_predicates_invoked"] is False
    assert record["boolean_or_collision_work_invoked"] is False


def test_normalized_pin_map_resolves_symlinked_source_paths(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert audit._normalized_pin_map({str(alias / "source.bin"): "a" * 64}) == {
        str(real / "source.bin"): "a" * 64
    }


def test_source_binding_scope_preserves_other_capture_surfaces():
    source = _row(7)
    identity = _identity(source_member="FJ3411")
    route_key = (51011, 7)
    proxy_key = (51004, 1)
    proxy = _row(1, body=20)

    report, projected = audit._audit_capture_row_bindings(
        surfaces={route_key: _surface(source), proxy_key: _surface(proxy)},
        captured_owners={route_key: 20, proxy_key: 1},
        transfer_rows={7: source},
        baseline_rows={7: _row(7, offset=100)},
        candidate_rows={7: _row(7, offset=200)},
        baseline_source_map={"7": identity},
        candidate_source_map={"7": dict(identity)},
        repair_surfaces={},
        verifier=object(),
        transfer_label="N27 accepted capture",
        source_bound_keys={route_key},
    )

    assert report["passed"]
    assert report["all_requested_source_bound_targets_bound"]
    assert report["captured_target_count"] == 1
    assert report["capture_surface_count_total"] == 2
    assert report["unbound_capture_semantic_counts"] == {"51004": 1}
    assert report["all_captured_targets_source_bound"] is False
    assert set(projected) == {route_key, proxy_key}

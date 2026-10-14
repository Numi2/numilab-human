import json

import numpy as np
import pytest

from numilab_human.resting_vessel_owner_blend import (
    FJ2217_OBJ_SHA256,
    ISA_ARCHIVE_SHA256,
    _retire_isolated_source_face,
    apply_parent_surface_updates,
    apply_route110_bridge_row,
    geodesic_owner_partition,
    reinstate_source_face_through_exact_aliases,
    validate_fj2217_reinstatement_parent_maps_unchanged,
    validate_fj2217_route_face_operation,
    validate_local_source_face_reinstatement,
)


def _strip(with_island=False):
    vertices = np.asarray(
        [[x, y, 0.0] for x in range(4) for y in (0.0, 1.0)],
        dtype=np.float32,
    )
    faces = []
    for x in range(3):
        a = 2 * x
        faces.extend(((a, a + 2, a + 1), (a + 1, a + 2, a + 3)))
    if with_island:
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    [[6.0, 0.0, 0.0], [6.1, 0.0, 0.0], [6.0, 0.1, 0.0]],
                    dtype=np.float32,
                ),
            )
        )
        faces.append((8, 9, 10))
    return vertices, np.asarray(faces, dtype=np.int64)


def test_geodesic_owner_partition_has_exact_ports_and_positive_interior():
    vertices, faces = _strip()
    weights, report = geodesic_owner_partition(vertices, faces, {0, 1}, {6, 7})
    assert np.array_equal(weights[[0, 1]], [[1, 0], [1, 0]])
    assert np.array_equal(weights[[6, 7]], [[0, 1], [0, 1]])
    assert np.all(weights[2:6] > 0)
    assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-7
    assert report["weights_are_inferred_not_measured"] is True


def test_geodesic_owner_partition_rejects_overlapping_or_disconnected_ports():
    vertices, faces = _strip(with_island=True)
    with pytest.raises(ValueError, match="empty or overlapping"):
        geodesic_owner_partition(vertices, faces, {0}, {0})
    with pytest.raises(ValueError, match="unmapped disconnected"):
        geodesic_owner_partition(vertices, faces, {0, 1}, {6, 7})


def test_geodesic_owner_partition_preserves_explicit_distal_only_remainder():
    vertices, faces = _strip(with_island=True)
    weights, report = geodesic_owner_partition(
        vertices, faces, {0, 1}, {6, 7}, isolated_distal_remainder={8, 9, 10}
    )
    assert np.array_equal(weights[[8, 9, 10]], np.tile([0, 1], (3, 1)))
    assert report["isolated_distal_remainder_vertex_count"] == 3
    assert report["isolated_distal_remainder_face_count"] == 1


def test_scoped_source_face_retirement_keeps_vertex_prefix_and_other_faces():
    vertices, faces = _strip()
    row = {
        "record": (91, 1938892, len(vertices), 10268397, faces.size, 477, 2, 0),
        "vertices6": np.column_stack(
            (vertices, np.tile([0.0, 0.0, 1.0], (len(vertices), 1)))
        ).astype("<f4"),
        "faces": faces.astype("<u4"),
    }
    target = np.asarray(row["faces"][2], dtype=np.uint32)
    replacement, report = _retire_isolated_source_face(row, 2, target)
    assert replacement["vertices6"].tobytes() == row["vertices6"].tobytes()
    assert (
        replacement["faces"].tobytes()
        == np.concatenate((row["faces"][:2], row["faces"][3:])).tobytes()
    )
    assert replacement["record"][4] == replacement["faces"].size
    assert report["removed_face_count"] == 1
    assert report["output_vertex_count"] == report["input_vertex_count"]


def _cap_row():
    vertices6 = np.zeros((1089, 6), dtype="<f4")
    source_points = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype="<f4",
    )
    source_ids = (930, 1066, 1063)
    alias_ids = (1068, 1065, 926)
    for source_id, alias_id, point in zip(source_ids, alias_ids, source_points):
        vertices6[source_id, :3] = point
        vertices6[alias_id, :3] = point
        vertices6[source_id, 3:] = [0.0, 0.0, 1.0]
        vertices6[alias_id, 3:] = [0.0, 0.0, 1.0]
    faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (2110, 1))
    faces[0] = [1068, 0, 1]
    faces[1] = [1065, 1, 2]
    faces[2] = [926, 2, 3]
    faces[1858] = source_ids
    row = {
        "record": (91, 100, len(vertices6), 200, faces.size, 477, 2, 0),
        "vertices6": vertices6,
        "faces": faces,
    }
    return row


def _reemit_proof():
    return {
        "replaced_stable_ids": [477],
        "stable_id_order_preserved": True,
        "global_offsets_rebased_by_existing_abi5_encoder": True,
        "registration_and_source_header_bytes_preserved": True,
    }


def test_cap_reinstatement_uses_exact_aliases_and_preserves_every_other_byte():
    row = _cap_row()
    original_faces = row["faces"].copy()
    original_vertices = row["vertices6"].tobytes()
    replacement, declaration = reinstate_source_face_through_exact_aliases(
        row, 1858, np.asarray([930, 1066, 1063], dtype="<u4")
    )
    assert replacement["faces"][1858].tolist() == [1068, 1065, 926]
    assert np.array_equal(replacement["faces"][:1858], original_faces[:1858])
    assert np.array_equal(replacement["faces"][1859:], original_faces[1859:])
    assert replacement["vertices6"].tobytes() == original_vertices
    assert replacement["record"][4] == original_faces.size
    assert declaration["coordinate_aliases_float32_bit_exact"] is True
    assert declaration["physical_mass_or_blood_state_added"] is False
    proof = validate_local_source_face_reinstatement(
        row, replacement, declaration, _reemit_proof(), expected_source_row=row
    )
    assert proof["restored_face_count"] == 1
    assert proof["all_vertices_and_normals_unchanged"] is True


def test_cap_reinstatement_rejects_alias_coordinate_or_corner_drift():
    row = _cap_row()
    row["vertices6"][1068, 0] = np.nextafter(
        row["vertices6"][1068, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="Float32-coordinate exact"):
        reinstate_source_face_through_exact_aliases(
            row, 1858, np.asarray([930, 1066, 1063], dtype="<u4")
        )

    row = _cap_row()
    row["faces"][1858] = [1066, 930, 1063]
    with pytest.raises(ValueError, match="source face/order"):
        reinstate_source_face_through_exact_aliases(
            row, 1858, np.asarray([930, 1066, 1063], dtype="<u4")
        )


def test_cap_reinstatement_validator_rejects_unrelated_face_or_normal_edit():
    row = _cap_row()
    replacement, declaration = reinstate_source_face_through_exact_aliases(
        row, 1858, np.asarray([930, 1066, 1063], dtype="<u4")
    )
    bad_faces = {**replacement, "faces": replacement["faces"].copy()}
    bad_faces["faces"][100] = [4, 5, 6]
    with pytest.raises(ValueError, match="face ancestry or order"):
        validate_local_source_face_reinstatement(
            row, bad_faces, declaration, _reemit_proof(), expected_source_row=row
        )
    bad_vertices = {**replacement, "vertices6": replacement["vertices6"].copy()}
    bad_vertices["vertices6"][20, 3] = np.float32(0.5)
    with pytest.raises(ValueError, match="position, normal"):
        validate_local_source_face_reinstatement(
            row, bad_vertices, declaration, _reemit_proof(), expected_source_row=row
        )


def test_cap_reinstatement_can_restore_only_the_historical_ordered_hole():
    complete = _cap_row()
    retired = {
        **complete,
        "faces": np.concatenate((complete["faces"][:1858], complete["faces"][1859:])),
        "record": (*complete["record"][:4], complete["record"][4] - 3, *complete["record"][5:]),
    }
    replacement, declaration = reinstate_source_face_through_exact_aliases(
        retired,
        1858,
        np.asarray([930, 1066, 1063], dtype="<u4"),
        source_row=complete,
    )
    assert np.array_equal(replacement["faces"], np.concatenate(
        (complete["faces"][:1858], np.asarray([[1068, 1065, 926]], dtype="<u4"), complete["faces"][1859:])
    ))
    assert declaration["operation"] == "reinsert_retired_source_face_at_original_order_position"


def test_parent_binding_update_preserves_existing_sources_and_unrelated_maps():
    prior_surfaces = [{"name": f"tissue-{i}", "weight": i} for i in range(10)]
    prior_map = {
        "stable_id": 477,
        "vertices": [[[4, 2, 0, 1.0]]],
        "retained": "provenance",
    }
    base = {
        "anatomy_payload_sha256": "a" * 64,
        "child_payload_sha256": "b" * 64,
        "source_payloads": [
            {"path": f"assets/old-{i}.nhtissue", "sha256": str(i) * 64}
            for i in range(5)
        ],
        "surfaces": prior_surfaces,
        "anatomy_surfaces": [
            {"stable_id": 3, "vertices": [[[0, 1, 0, 1.0]]]},
            prior_map,
        ],
    }
    maps = {
        475: [[[5, 1, 0, 0.5], [5, 2, 0, 0.5]]],
        477: [[[5, 3, 0, 0.5], [5, 4, 0, 0.5]]],
    }
    updated, digest = apply_parent_surface_updates(
        base,
        {
            "carrier_source_path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
            "carrier_payload_sha256": "c" * 64,
            "parent_surface_updates": maps,
        },
        "d" * 64,
    )
    assert updated["anatomy_payload_sha256"] == "d" * 64
    assert updated["child_payload_sha256"] == base["child_payload_sha256"]
    assert updated["source_payloads"][:5] == base["source_payloads"]
    assert updated["source_payloads"][5] == {
        "path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
        "sha256": "c" * 64,
    }
    assert updated["surfaces"] == prior_surfaces
    rows = {row["stable_id"]: row for row in updated["anatomy_surfaces"]}
    assert rows[3] == base["anatomy_surfaces"][0]
    assert rows[477]["retained"] == "provenance"
    assert rows[477]["vertices"] == maps[477]
    assert rows[475] == {"stable_id": 475, "vertices": maps[475]}
    assert len(digest) == 64


def test_brachiocephalic_wrapper_forwards_owner_binding():
    from pathlib import Path
    from unittest.mock import patch

    from numilab_human import resting_major_vessels

    binding = {
        "n27_receipt_path": Path("N27.json"),
        "port_report_path": Path("086.json"),
    }
    with patch.object(
        resting_major_vessels, "append_viscera", return_value={"ok": True}
    ) as append:
        result = resting_major_vessels.append_brachiocephalic_veins(
            base_receipt=Path("base.json"),
            output=Path("out"),
            sources_path=Path("sources"),
            registration_path=Path("registration.json"),
            rigid_path=Path("rigid.nhrigid"),
            quotient_npz_path=Path("quotient.npz"),
            quotient_report_path=Path("quotient.json"),
            vessel_owner_blend_binding=binding,
        )
    assert result == {"ok": True}
    assert append.call_args.kwargs["vessel_owner_blend_binding"] == binding



def test_route110_bridge_row_transfer_requires_exact_compiler_input():
    before = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype="<f4",
    )
    after = before + np.asarray([0.0, 0.0, 0.001], dtype="<f4")
    normals = np.tile(np.asarray([0.0, 0.0, 1.0], dtype="<f4"), (3, 1))
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    compiled = {
        "record": (20, 0, 3, 0, 3, 525, 2, 0),
        "vertices6": np.column_stack((before, normals)).astype("<f4"),
        "faces": faces,
    }
    route = {
        "before_local_m": before,
        "after_local_m": after,
        "normals_after": normals,
        "faces": faces,
    }
    transferred = apply_route110_bridge_row(compiled, route)
    assert transferred["vertices6"][:, :3].tobytes() == after.tobytes()
    assert transferred["vertices6"][:, 3:].tobytes() == normals.tobytes()
    assert transferred["faces"].tobytes() == faces.tobytes()
    wrong = {**compiled, "vertices6": compiled["vertices6"].copy()}
    wrong["vertices6"][0, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError, match="existing ABI5 compiler output"):
        apply_route110_bridge_row(wrong, route)


def test_fj2217_reinstatement_parent_map_proof_is_separate_and_exact():
    base = {
        "child_payload_sha256": "a" * 64,
        "source_payloads": [{"path": "assets/core.nhtissue", "sha256": "b" * 64}],
        "surfaces": [{"body_index": 20, "transform": [0.0] * 8}],
        "anatomy_surfaces": [
            {"stable_id": 477, "vertices": [[[0, 3, 0, 0.25], [0, 4, 0, 0.75]]]},
            {"stable_id": 475, "vertices": [[[0, 1, 0, 1.0]]]},
        ],
    }
    candidate = json.loads(json.dumps(base))
    proof = validate_fj2217_reinstatement_parent_maps_unchanged(base, candidate)
    assert proof["all_parent_surface_maps_unchanged"] is True
    assert proof["ordered_parent_terms_unchanged"] is True
    assert proof["source_payload_order_unchanged"] is True

    candidate["anatomy_surfaces"][0]["vertices"][0].reverse()
    with pytest.raises(ValueError, match="anatomy_surfaces"):
        validate_fj2217_reinstatement_parent_maps_unchanged(base, candidate)


def _full_fj2217_row():
    vertices = np.zeros((1089, 6), dtype="<f4")
    vertices[:, 5] = 1.0
    for source_id, alias_id in ((930, 1068), (1066, 1065), (1063, 926)):
        vertices[alias_id, :3] = vertices[source_id, :3]
    faces = np.zeros((2110, 3), dtype="<u4")
    faces[:, 0] = 0
    faces[:, 1] = 1
    faces[:, 2] = 2
    faces[1858] = [930, 1066, 1063]
    faces[0] = [1068, 1065, 926]
    row = {
        "record": (91, 0, 1089, 0, faces.size, 477, 2, 0),
        "body_index": 91,
        "layer": 2,
        "flags": 0,
        "vertices6": vertices,
        "faces": faces,
    }
    return row


def _legacy_fj2217_retirement(row):
    face = row["faces"][1858].copy()
    faces = np.concatenate((row["faces"][:1858], row["faces"][1859:]))
    candidate = {
        **row,
        "faces": faces,
        "record": (*row["record"][:4], faces.size, *row["record"][5:]),
    }
    lineage = {
        "classification": "unexplained_isolated_raw_source_face; scoped_reference_retirement",
        "anatomical_interface_proven": False,
        "local_retirement_proposed": True,
        "retirement_operation_changes_no_mesh_coordinates": True,
        "source_archive_sha256": ISA_ARCHIVE_SHA256,
        "source_member_sha256": FJ2217_OBJ_SHA256,
        "source_face_and_nha_face_arrays_exact": True,
        "source_face_id_zero_based": 1858,
        "nha_local_face_id_zero_based": 1858,
        "ordered_source_vertex_ids_zero_based": face.astype(int).tolist(),
        "nha_local_vertex_ids_zero_based": face.astype(int).tolist(),
    }
    retirement = {
        "stable_id": 477,
        "removed_face_count": 1,
        "source_archive_sha256": ISA_ARCHIVE_SHA256,
        "source_member_sha256": FJ2217_OBJ_SHA256,
        "source_face_and_nha_ancestry_exact": True,
        "all_other_source_faces_retained_in_original_order": True,
        "unchanged_source_vertex_prefix_byte_exact": True,
        "input_face_count": 2110,
        "output_face_count": 2109,
        "input_vertex_count": 1089,
        "output_vertex_count": 1089,
        "retained_unused_vertex_slots": 3,
        "source_face_bytes_sha256": __import__("hashlib").sha256(
            face.astype("<u4").tobytes()
        ).hexdigest(),
        "source_face_id_zero_based": 1858,
        "lineage": lineage,
    }
    proof = {
        "replaced_stable_ids": [477],
        "stable_id_order_preserved": True,
        "global_offsets_rebased_by_existing_abi5_encoder": True,
        "registration_and_source_header_bytes_preserved": True,
    }
    return candidate, retirement, proof


def test_fj2217_exact_legacy_retirement_receipt_replays():
    full = _full_fj2217_row()
    retired, declaration, proof = _legacy_fj2217_retirement(full)
    result = validate_fj2217_route_face_operation(
        full,
        retired,
        full,
        reinstatement=None,
        active_retirement=declaration,
        historical_retirement=None,
        route_reemit_proof=proof,
        historical_retirement_reemit_proof=None,
    )
    assert result["receipt_operation"] == "legacy_scoped_retirement_replay"
    assert result["all_other_faces_exact_and_ordered"] is True
    assert result["anatomical_admission"] is False


def test_fj2217_reinsertion_replays_the_exact_historical_retirement():
    full = _full_fj2217_row()
    retired, history, proof = _legacy_fj2217_retirement(full)
    candidate, declaration = reinstate_source_face_through_exact_aliases(
        retired,
        1858,
        full["faces"][1858],
        source_row=full,
    )
    result = validate_fj2217_route_face_operation(
        retired,
        candidate,
        full,
        reinstatement=declaration,
        active_retirement=None,
        historical_retirement=history,
        route_reemit_proof=proof,
        historical_retirement_reemit_proof=proof,
    )
    assert result["receipt_operation"] == (
        "reinsert_retired_source_face_at_original_order_position"
    )
    assert result["all_other_faces_exact_and_ordered"] is True


def test_fj2217_fresh_reinstatement_does_not_fabricate_retirement():
    base = _full_fj2217_row()
    candidate, declaration = reinstate_source_face_through_exact_aliases(
        base,
        1858,
        base["faces"][1858],
        source_row=base,
    )
    proof = {
        "replaced_stable_ids": [477],
        "stable_id_order_preserved": True,
        "global_offsets_rebased_by_existing_abi5_encoder": True,
        "registration_and_source_header_bytes_preserved": True,
    }
    result = validate_fj2217_route_face_operation(
        base,
        candidate,
        base,
        reinstatement=declaration,
        active_retirement=None,
        historical_retirement=None,
        route_reemit_proof=proof,
        historical_retirement_reemit_proof=None,
    )
    assert result["receipt_operation"] == (
        "replace_original_face_corners_by_exact_coordinate_aliases"
    )
    assert result["all_vertices_and_normals_unchanged"] is True
    with pytest.raises(ValueError, match="must not fabricate historical retirement"):
        validate_fj2217_route_face_operation(
            base,
            candidate,
            base,
            reinstatement=declaration,
            active_retirement=None,
            historical_retirement={"source_face_id_zero_based": 1858},
            route_reemit_proof=proof,
            historical_retirement_reemit_proof=proof,
        )

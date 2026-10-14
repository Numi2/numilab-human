import numpy as np
import pytest

from numilab_human import (
    resting_brachiocephalic_current_capture_network_preparation as prep,
)


def _context_and_item():
    vertices6 = np.zeros((1089, 6), dtype="<f4")
    for original, alias in zip((930, 1066, 1063), (1068, 1065, 926)):
        vertices6[original, :3] = [float(original), 1.0, 2.0]
        vertices6[alias, :3] = vertices6[original, :3]
    faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (2110, 1))
    faces[1858] = [1068, 1065, 926]
    row = {"faces": faces, "vertices6": vertices6}
    reinstatement = {
        "schema": "numi.human.local-source-face-exact-alias-reinstatement.v1",
        "stable_id": 477,
        "source_member": "FJ2217",
        "source_face_id_zero_based": 1858,
        "source_vertex_ids_zero_based": [930, 1066, 1063],
        "alias_vertex_ids_zero_based": [1068, 1065, 926],
        "coordinate_aliases_float32_bit_exact": True,
        "source_face_ancestry_exact": True,
        "all_other_faces_byte_exact_and_ordered": True,
        "all_vertex_and_normal_bytes_unchanged": True,
        "physical_mass_or_blood_state_added": False,
        "input_face_count": 2109,
        "output_face_count": 2110,
    }
    retirement = {
        "source_member_sha256": "7b448d1f2a071e88d1a2ebe54e7e5d417719663a5b01d1226f492b71ebdc19cd",
        "source_face_id_zero_based": 1858,
        "output_face_count": 2109,
        "lineage": {
            "classification": "unexplained_isolated_raw_source_face; scoped_reference_retirement",
        },
    }
    context = {
        "source_map": {
            "477": {
                "repair": {
                    "historical_local_source_face_retirement": retirement,
                    "local_source_face_reinstatement": reinstatement,
                }
            },
            "473": {},
        },
        "rows": {477: row},
    }
    item = {"pairs": np.asarray([[1858, 0], [2000, 1]], dtype="<i8")}
    return context, item


def test_reinstated_cap_keeps_frozen_face_ids_without_decrement():
    context, item = _context_and_item()
    before = item["pairs"].copy()
    result = prep._apply_receipt_face_retirements(item, 477, 473, context)
    assert np.array_equal(result["pairs"], before)
    assert "source_face_index_remap" not in result
    identity = result["source_face_index_identity"]
    assert len(identity) == 1
    assert identity[0]["reinstated_source_face_id_zero_based"] == 1858
    assert identity[0]["historical_retirement_is_not_active"] is True


def test_reinstated_cap_rejects_non_exact_source_alias_coordinates():
    context, item = _context_and_item()
    context["rows"][477]["vertices6"][1068, 0] = np.nextafter(
        np.float32(930.0), np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="coordinate bits differ"):
        prep._apply_receipt_face_retirements(item, 477, 473, context)


def _owner_composition_fixture(tmp_path):
    carrier = tmp_path / "carrier.nhtissue"
    carrier.write_bytes(b"pinned appended source")
    digest = prep._sha_file(carrier)
    old_sources = [{"index": index, "sha256": f"{index + 1:064x}"}
                   for index in range(7)]
    new_sources = [dict(row) for row in old_sources] + [
        {"index": 7, "path": str(carrier), "sha256": digest}
    ]
    source_ids = sorted(set(prep.NETWORK_STABLE_IDS) | {313, 315, 317})
    source_map = {str(sid): {"stable_id": sid, "source": "same"} for sid in source_ids}
    old_maps = [
        {"stable_id": sid, "vertices": [[0, 1.0]]}
        for sid in (475, 477, 999)
    ]
    new_maps = [
        {"stable_id": sid,
         "vertices": [[0, 2.0 if sid in (475, 477) else 1.0]]}
        for sid in (475, 477, 999)
    ]
    common = {
        "anatomy_payload_sha256": prep.capture_gate.ANATOMY_PAYLOAD_SHA256,
        "sha256": "a" * 64, "map": {"record_count": 621369},
    }
    current_common = dict(common)
    current_common["anatomy_payload_sha256"] = prep.OWNER_COMPOSITION_PAYLOAD_SHA256
    old = {
        "payload": {"sha256": prep.capture_gate.ANATOMY_PAYLOAD_SHA256},
        "provenance": {
            "source_id_map": source_map,
            "cardiac_geometry_binding": {"common_field": common},
        },
        "passive_tissue_parent_bindings": {
            "source_payloads": old_sources,
            "anatomy_surfaces": old_maps,
        },
    }
    current = {
        "payload": {"sha256": prep.OWNER_COMPOSITION_PAYLOAD_SHA256},
        "provenance": {
            "source_id_map": source_map,
            "cardiac_geometry_binding": {"common_field": current_common},
            "passive_brachiocephalic_exact_coordinate_owner_composition": {
                "final_source_payload_count": 8,
                "existing_source_payload_prefix_semantics_preserved": True,
                "additional_physical_mass_kg": 0,
                "additional_force_or_state": False,
                "anatomical_admission": False,
            },
        },
        "passive_tissue_parent_bindings": {
            "source_payloads": new_sources,
            "anatomy_surfaces": new_maps,
        },
    }
    return old, current, carrier, digest



def _cap_row_transition_fixture():
    old_vertices = np.zeros((1089, 6), dtype="<f4")
    for source, alias in zip((930, 1066, 1063), (1068, 1065, 926)):
        old_vertices[source, :3] = [float(source), 1.0, 2.0]
        old_vertices[alias, :3] = old_vertices[source, :3]
    new_vertices = old_vertices.copy()
    old_cap_faces = np.zeros((2109, 3), dtype="<u4")
    new_cap_faces = np.concatenate(
        (old_cap_faces[:1858], np.asarray([[1068, 1065, 926]], dtype="<u4"),
         old_cap_faces[1858:]), axis=0)
    actual_rows = {
        476: {"vertices6": np.zeros((1, 6), dtype="<f4"),
              "faces": np.zeros((1, 3), dtype="<u4"),
              "record": (41, 0, 1, 0, 3, 476, 2, 0)},
        477: {"vertices6": old_vertices, "faces": old_cap_faces,
              "record": (91, 1, 1089, 3, 6327, 477, 2, 0)},
        478: {"vertices6": np.zeros((1, 6), dtype="<f4"),
              "faces": np.zeros((1, 3), dtype="<u4"),
              "record": (41, 1090, 1, 6330, 3, 478, 2, 0)},
    }
    current_rows = {
        476: {"vertices6": np.zeros((1, 6), dtype="<f4"),
              "faces": np.zeros((1, 3), dtype="<u4"),
              "record": (41, 0, 1, 0, 3, 476, 2, 0)},
        477: {"vertices6": new_vertices, "faces": new_cap_faces,
              "record": (91, 1, 1089, 3, 6330, 477, 2, 0)},
        478: {"vertices6": np.zeros((1, 6), dtype="<f4"),
              "faces": np.zeros((1, 3), dtype="<u4"),
              "record": (41, 1090, 1, 6333, 3, 478, 2, 0)},
    }
    actual_records = [actual_rows[sid]["record"] for sid in (476, 477, 478)]
    current_records = [current_rows[sid]["record"] for sid in (476, 477, 478)]
    return actual_rows, current_rows, actual_records, current_records


def test_all_row_transition_accepts_only_reinstated_cap_and_derived_offsets():
    actual, current, actual_records, current_records = _cap_row_transition_fixture()
    result = prep._compare_payload_rows(actual, current, actual_records, current_records)
    assert result["status"] == (
        "ALL_ABI5_ROWS_BYTE_EXACT_EXCEPT_DECLARED_FJ2217_CAP_FACE")
    assert result["cap_face"]["source_face_id_zero_based"] == 1858
    assert result["record_offset_effect"]["rows_after_477_index_start_increment"] == 3


def test_all_row_transition_rejects_unrelated_face_or_record_change():
    actual, current, actual_records, current_records = _cap_row_transition_fixture()
    current[476]["faces"][0, 0] = 1
    with pytest.raises(ValueError, match="outside the declared cap"):
        prep._compare_payload_rows(actual, current, actual_records, current_records)

    actual, current, actual_records, current_records = _cap_row_transition_fixture()
    current_records[2] = (41, 1090, 1, 6334, 3, 478, 2, 0)
    current[478]["record"] = current_records[2]
    with pytest.raises(ValueError, match="exact cap index-count/offset shift"):
        prep._compare_payload_rows(actual, current, actual_records, current_records)


def test_cap_transition_rejects_coordinate_alias_drift():
    actual, current, actual_records, current_records = _cap_row_transition_fixture()
    current[477]["vertices6"][1068, 0] = np.nextafter(
        np.float32(930.0), np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="position/normal bytes changed"):
        prep._compare_payload_rows(actual, current, actual_records, current_records)

def test_source_collar_search_counts_only_explicit_source_rows_and_step_zero_pairs():
    report = {
        "rows": [
            {"parent_stable_id": 479, "child_stable_id": 487},
            {"parent_stable_id": 481, "child_stable_id": 489},
        ],
        "poses": [
            {"step": 1, "network_pairs": {"479->487": {}}},
            {"step": 0, "network_pairs": {"479->483": {}}},
        ],
    }
    assert prep._source_edge_occurrences(report, 479, 487) == 1
    assert prep._source_edge_occurrences(report, 481, 489) == 1
    assert prep._source_edge_occurrences(report, 479, 483) == 1
    assert prep._source_edge_occurrences(report, 479, 489) == 0


def test_source0_network_inventory_is_exactly_13_plus_two_and_keeps_469_473_separate():
    assert len(prep.NETWORK_EDGES) == 13
    assert len(prep.SVC_PORTS) == 2
    assert len(prep.NETWORK_EDGES) + len(prep.SVC_PORTS) == 15
    assert ((51011, 479), (51011, 487)) in prep.NETWORK_EDGES
    assert ((51011, 481), (51011, 489)) in prep.NETWORK_EDGES
    assert prep.POTENTIAL_NONNETWORK_OVERLAP == ((51011, 469), (51011, 473))
    assert prep.POTENTIAL_NONNETWORK_OVERLAP not in prep.NETWORK_EDGES
    assert set(prep.NETWORK_STABLE_IDS) == {
        7, 10, 468, 469, 471, 472, 473, 475, 477, 479, 481, 483, 485,
        487, 489, 525, 526,
    }


def test_source0_gate_stays_blocked_until_current_native_capture_and_missing_collars():
    pending = ["479->487", "481->489"]
    owner_delta = {"current_capture_candidate_ready": False}
    assert prep._source0_preflight_status(pending, [], owner_delta) == (
        "SOURCE0_SEAMOWNER_PREPARED_PENDING_NATIVE_EMISSION_AND_TWO_SOURCE_COLLARS")
    assert prep._source0_preflight_status([], [], owner_delta) == (
        "SOURCE0_SEAMOWNER_PREPARED_PENDING_NATIVE_EMISSION_AND_TWO_SOURCE_COLLARS")
    owner_delta["current_capture_candidate_ready"] = True
    assert prep._source0_preflight_status([], [], owner_delta) == (
        "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE")


def test_owner_composition_delta_allows_only_475_477_and_preserves_eight_source_cap(
    tmp_path, monkeypatch
):
    old, current, carrier, digest = _owner_composition_fixture(tmp_path)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER", carrier)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER_SHA256", digest)
    result = prep._owner_composition_delta(old, current)
    assert result["owner_rows_changed_only"] == [475, 477]
    assert result["actual_parent_payload_count"] == 7
    assert result["current_parent_payload_count"] == 8
    assert result["prefix_source_payloads_exactly_preserved"] is True
    assert result["appended_source_index"] == 7
    assert result["additional_physical_mass_kg"] == 0
    assert result["additional_force_or_state"] is False
    assert result["current_capture_candidate_ready"] is False


def test_owner_composition_delta_rejects_an_unscoped_owner_row_change(tmp_path, monkeypatch):
    old, current, carrier, digest = _owner_composition_fixture(tmp_path)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER", carrier)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER_SHA256", digest)
    current["passive_tissue_parent_bindings"]["anatomy_surfaces"][-1]["vertices"][0][1] = 3.0
    with pytest.raises(ValueError, match="beyond the declared 475/477"):
        prep._owner_composition_delta(old, current)



def test_common_field_payload_rebind_preserves_every_semantic_field():
    actual = {
        "anatomy_payload_sha256": prep.capture_gate.ANATOMY_PAYLOAD_SHA256,
        "map": {"sha256": "a" * 64},
        "polynomials": {"sha256": "b" * 64},
        "domain_boxes": [[0.0, 1.0]],
    }
    current = dict(actual)
    current["anatomy_payload_sha256"] = prep.OWNER_COMPOSITION_PAYLOAD_SHA256
    result = prep._validate_common_field_descriptor_transition(actual, current)
    assert result["all_common_field_semantics_and_assets_exact"] is True
    current["domain_boxes"] = [[0.0, 2.0]]
    with pytest.raises(ValueError, match="common map, polynomials, domains"):
        prep._validate_common_field_descriptor_transition(actual, current)

def test_source_identity_crosscheck_uses_current_fma_and_member_labels():
    source_map = {
        "487": {
            "name": "left ulnar artery",
            "source_member": "FJ2258",
            "body_index": 92,
            "vascular_compartment_id": 3,
            "source_owner_metadata": {"concept_id": "FMA22798"},
        },
        "489": {
            "name": "left ulnar vein",
            "source_member": "FJ2260",
            "body_index": 92,
            "vascular_compartment_id": 4,
            "source_owner_metadata": {"concept_id": "FMA22952"},
        },
    }
    result = prep._source_identity_crosscheck(source_map)
    assert result["487"]["name"] == "left ulnar artery"
    assert result["489"]["name"] == "left ulnar vein"
    source_map["487"]["name"] = "left radial artery"
    with pytest.raises(ValueError, match="branch identity metadata changed"):
        prep._source_identity_crosscheck(source_map)

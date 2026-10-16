import copy
import json
from pathlib import Path

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
        {"stable_id": sid, "vertices": [[[0, 101, 0, 1.0]]]}
        for sid in (475, 477, 999)
    ]
    new_maps = [
        {"stable_id": sid,
         "vertices": ([[[0, 101, 0, 0.5], [7, 101, 0, 0.5]]]
                      if sid in (475, 477) else [[[0, 101, 0, 1.0]]])}
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


def test_parent_source_rebase_optional_capture_bundle_is_all_or_none(tmp_path):
    payload = tmp_path / "payload.nhanatomy"
    payload.write_bytes(b"payload")
    anatomy_receipt = tmp_path / "anatomy-receipt.json"
    anatomy_receipt.write_text(json.dumps({
        "payload": {"path": str(payload.resolve()), "sha256": prep._sha_file(payload)}
    }))
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"pack")
    capture_receipt = tmp_path / "step-0.receipt.json"
    capture_receipt.write_bytes(b"receipt")
    report = tmp_path / "source-rebase.json"
    report.write_bytes(b"source-only proof")
    bundle = {
        "capture_pack_path": str(pack.resolve()),
        "capture_pack_sha256": prep._sha_file(pack),
        "capture_receipt_path": str(capture_receipt.resolve()),
        "capture_receipt_sha256": prep._sha_file(capture_receipt),
        "anatomy_receipt_path": str(anatomy_receipt.resolve()),
        "anatomy_receipt_sha256": prep._sha_file(anatomy_receipt),
        "anatomy_payload_sha256": prep._sha_file(payload),
    }

    with pytest.raises(ValueError, match="seven pinned fields"):
        prep._validated_capture_input_bundle({
            **bundle, "parent_source_rebase_report_path": str(report.resolve())
        })

    normalized = prep._validated_capture_input_bundle({
        **bundle,
        "parent_source_rebase_report_path": str(report.resolve()),
        "parent_source_rebase_report_sha256": prep._sha_file(report),
    })
    assert normalized["parent_source_rebase_report_path"] == report.resolve()
    assert normalized["parent_source_rebase_report_sha256"] == prep._sha_file(report)


@pytest.fixture(scope="module")
def retained_parent_source_rebase_case():
    report_path = prep.DATA_ROOT / (
        "aortic7-connected-network036-deltoid868b-parent7-rest-registration-bundle-003/"
        "source-rest-registration-report.json")
    current_receipt_path = prep.DATA_ROOT / (
        "rib50-existing-native-owner-bindings-954/resting-anatomy-receipt.json")
    if not report_path.is_file() or not current_receipt_path.is_file():
        pytest.skip("retained 956 source-rebase evidence is not installed in this environment")
    report = json.loads(report_path.read_text())
    current_receipt = json.loads(current_receipt_path.read_text())
    base_receipt_path = Path(report["base_receipt"]["path"])
    output_payload_path = Path(report["anatomy_payload"]["path"])
    current_payload_path = Path(current_receipt["payload"]["path"])
    required = (base_receipt_path, output_payload_path, current_payload_path)
    if not all(path.is_file() for path in required):
        pytest.skip("retained source-rebase payload lineage is not installed")
    base_receipt = json.loads(base_receipt_path.read_text())
    _, _, output_rows_all = prep.capture_gate.nhanat_codec._parse_payload(
        output_payload_path.read_bytes())
    _, _, current_rows_all = prep.capture_gate.nhanat_codec._parse_payload(
        current_payload_path.read_bytes())
    row_ids = (475, 477)
    return {
        "path": report_path.resolve(),
        "sha256": prep._sha_file(report_path),
        "report": report,
        "current_receipt": current_receipt,
        "base_receipt": base_receipt,
        "output_rows": {stable_id: output_rows_all[stable_id] for stable_id in row_ids},
        "current_rows": {stable_id: current_rows_all[stable_id] for stable_id in row_ids},
    }


def _validate_retained_parent_source_rebase(case, *, report=None, current_receipt=None):
    return prep._validate_parent_source_rebase_evidence(
        proof_path=case["path"],
        proof_sha256=case["sha256"],
        report=case["report"] if report is None else report,
        current_receipt=(
            case["current_receipt"] if current_receipt is None else current_receipt),
        base_receipt=case["base_receipt"],
        reemitted_rows=case["output_rows"],
        current_source_rows=case["current_rows"],
    )


def test_parent_source_rebase_accepts_retained_actual_proof(retained_parent_source_rebase_case):
    case = retained_parent_source_rebase_case
    proof = prep._load_parent_source_rebase_proof(
        {"path": case["path"], "sha256": case["sha256"]},
        case["current_receipt"], case["current_rows"],
    )
    assert proof["status"] == "AUTHENTICATED_SOURCE_ONLY_PARENT_REEMISSION"
    assert proof["typed_rows"] == [[51011, 475], [51011, 477]]
    assert proof["parent_maps_byte_exact_to_immediate_base"] is True
    assert proof["source_slots_0_through_6_byte_exact_to_immediate_base"] is True
    assert proof["anatomical_admission"] is False
    assert proof["native_admission"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_report_rows", "does not prove the declared 475/477 slot-7 operation"),
        ("wrong_owner_identity", "changed typed source ownership"),
        ("changed_positive_parent_map", "full positive parent terms changed"),
        ("failed_forward_residual", "does not have a passing registered rest residual"),
    ],
)
def test_parent_source_rebase_rejects_stale_owner_or_map_mutations(
    retained_parent_source_rebase_case, mutation, message,
):
    case = retained_parent_source_rebase_case
    report = copy.deepcopy(case["report"])
    current_receipt = copy.deepcopy(case["current_receipt"])
    if mutation == "wrong_report_rows":
        report["current_nha_rows_used"] = [475, 476]
    elif mutation == "wrong_owner_identity":
        current_receipt["provenance"]["source_id_map"]["475"]["body_index"] = 20
    elif mutation == "changed_positive_parent_map":
        row = next(
            item for item in current_receipt["passive_tissue_parent_bindings"][
                "anatomy_surfaces"] if item["stable_id"] == 475
        )
        row["vertices"][0][0][3] *= 0.5
    elif mutation == "failed_forward_residual":
        report["rows"]["475"]["re_emitted_carrier_residual"][
            "max_euclidean_residual_m"] = 3e-6
        current_receipt["provenance"][
            prep.PARENT_SOURCE_REBASE_RECEIPT_FIELD][
                "max_reemitted_parent_rest_residual_m"] = 3e-6

    with pytest.raises(ValueError, match=message):
        _validate_retained_parent_source_rebase(
            case, report=report, current_receipt=current_receipt)


def test_parent_source_rebase_rejects_stale_report_digest(retained_parent_source_rebase_case):
    case = retained_parent_source_rebase_case
    with pytest.raises(ValueError, match="frozen input changed"):
        prep._load_parent_source_rebase_proof(
            {"path": case["path"], "sha256": "0" * 64},
            case["current_receipt"], case["current_rows"],
        )


def test_parent_pose_baseline_selects_current_receipt_carrier_not_legacy_index():
    payloads = [
        {"path": f"/tmp/unused-{index}.nhtissue", "sha256": str(index) * 64}
        for index in range(8)
    ]
    current_terms = [
        {"stable_id": 475, "vertices": [
            [[7, 1, 0, 0.25], [7, 2, 0, 0.75]],
        ]},
        {"stable_id": 477, "vertices": [
            [[7, 3, 0, 0.125], [7, 4, 0, 0.875]],
        ]},
    ]
    receipt = {"passive_tissue_parent_bindings": {
        "source_payloads": payloads,
        "anatomy_surfaces": current_terms,
    }}
    source_rows = {
        stable_id: {"vertices6": np.zeros((1, 6), dtype="<f4")}
        for stable_id in (475, 477)
    }

    index, payload, terms, hashes = prep._current_parent_pose_owner_inputs(
        receipt, source_rows)

    assert index == 7
    assert payload == payloads[7]
    assert terms[475] == current_terms[0]["vertices"]
    assert terms[477] == current_terms[1]["vertices"]
    assert set(hashes) == {"475", "477"}

    receipt["passive_tissue_parent_bindings"]["anatomy_surfaces"][1][
        "vertices"][0][0][0] = 6
    with pytest.raises(ValueError, match="do not share one registered carrier"):
        prep._current_parent_pose_owner_inputs(receipt, source_rows)


def test_current_parent_owner_terms_preserve_duplicate_source_aliases():
    parents = {
        3: np.asarray([[0.125, -0.25, 0.5], [0.125, -0.25, 0.5]], dtype="<f4"),
        4: np.asarray([[0.875, 0.5, -0.25], [0.875, 0.5, -0.25]], dtype="<f4"),
    }
    current_terms = [
        [[7, 3, 0, 0.25], [7, 4, 0, 0.75]],
        [[7, 3, 1, 0.25], [7, 4, 1, 0.75]],
    ]
    current, _ = prep._blend_parent_pose_vertices(
        current_terms, parents, expected_source_index=7, stable_id=477)
    assert current[0].tobytes() == current[1].tobytes()

    legacy_terms = [
        [[6, 3, 0, 0.25], [6, 4, 0, 0.75]],
        [[6, 3, 1, 0.5], [6, 4, 1, 0.5]],
    ]
    legacy, _ = prep._blend_parent_pose_vertices(
        legacy_terms, parents, expected_source_index=6, stable_id=477)
    assert legacy[0].tobytes() != legacy[1].tobytes()


def test_parent_pose_source_vertex_map_uses_ordered_face_corner_ancestry():
    source_faces = np.asarray([[0, 1, 2], [2, 1, 3]], dtype="<u4")
    capture_faces = np.asarray([[2, 0, 1], [1, 0, 3]], dtype="<u4")
    vertex_map, report = prep._source_to_capture_vertex_map(
        source_faces, capture_faces, 5, 4)
    np.testing.assert_array_equal(vertex_map, [2, 0, 1, 3, -1])
    assert report["status"] == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
    assert report["source_unreferenced_vertex_count"] == 1
    assert report["capture_unreferenced_vertex_count"] == 0


def test_parent_pose_source_vertex_map_rejects_incomplete_or_merged_ancestry():
    source_faces = np.asarray([[0, 1, 2]], dtype="<u4")
    no_match, report = prep._source_to_capture_vertex_map(
        source_faces, np.asarray([[0, 1, 2], [1, 0, 2]], dtype="<u4"), 3, 3)
    assert no_match is None
    assert report["status"] == "CAPTURE_TO_SOURCE_FACE_ANCESTRY_NOT_ALIGNED"

    with pytest.raises(ValueError, match="merges distinct source vertices"):
        prep._source_to_capture_vertex_map(
            np.asarray([[0, 1, 2], [3, 1, 2]], dtype="<u4"),
            np.asarray([[0, 1, 2], [0, 1, 2]], dtype="<u4"), 4, 3)


def test_parent_pose_baseline_blends_in_exact_float32_term_order():
    first = np.asarray([[16_777_216.0, 3.0, -4.0]], dtype="<f4")
    second = np.asarray([[-16_777_216.0, 9.0, 2.0]], dtype="<f4")
    terms = [[[6, 1, 0, 0.5], [6, 2, 0, 0.5]]]
    output, report = prep._blend_parent_pose_vertices(
        terms, {1: first, 2: second},
        expected_source_index=6, stable_id=475)
    expected = np.zeros(3, dtype="<f4")
    expected = np.asarray(
        expected + np.asarray(first[0] * np.float32(0.5), dtype="<f4"),
        dtype="<f4")
    expected = np.asarray(
        expected + np.asarray(second[0] * np.float32(0.5), dtype="<f4"),
        dtype="<f4")
    np.testing.assert_array_equal(output[0], expected)
    assert report["source_term_order_diagnostic"]
    assert report["bit_exact_native_accumulation_claim"] is False
    assert report["used_parent_source_rows"] == [1, 2]
    assert report["maximum_partition_residual_f64"] == 0.0


@pytest.mark.parametrize(
    ("terms", "parents", "source_index"),
    [
        ([[[5, 1, 0, 1.0]]], {1: np.zeros((1, 3), dtype="<f4")}, 6),
        ([[[6, 1, 1, 1.0]]], {1: np.zeros((1, 3), dtype="<f4")}, 6),
        ([[[6, 1, 0, 0.4]]], {1: np.zeros((1, 3), dtype="<f4")}, 6),
        ([[[6, 1, 0, True]]], {1: np.zeros((1, 3), dtype="<f4")}, 6),
    ],
)
def test_parent_pose_baseline_rejects_wrong_ancestry_or_nonpartition(
    terms, parents, source_index,
):
    with pytest.raises(ValueError, match="invalid parent baseline term|do not partition"):
        prep._blend_parent_pose_vertices(
            terms, parents, expected_source_index=source_index, stable_id=477)


def test_source0_gate_stays_blocked_until_current_native_capture_and_missing_collars():
    pending = ["479->487", "481->489"]
    owner_delta = {"current_capture_candidate_ready": False}
    assert prep._source0_preflight_status(pending, [], owner_delta) == (
        "SOURCE0_SEAMOWNER_PREPARED_PENDING_SOURCE_COLLARS_AND_NATIVE_EMISSION")
    assert prep._source0_preflight_status([], [], owner_delta) == (
        "SOURCE0_SEAMOWNER_PREPARED_PENDING_FINAL_NATIVE_EMISSION")
    owner_delta["current_capture_candidate_ready"] = True
    assert prep._source0_preflight_status([], [], owner_delta) == (
        "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE")


def test_ulnar_proof_separates_data_inputs_from_executable_dependency_pins(tmp_path):
    data_path = tmp_path / "source.json"
    code_path = tmp_path / "manifold.so"
    data_sha = "a" * 64
    code_sha = "b" * 64
    data, code = prep._validate_ulnar_proof_pin_namespaces(
        {str(data_path): data_sha},
        {str(code_path): code_sha},
        {str(data_path): data_sha},
        {str(code_path): code_sha},
    )
    assert data[str(data_path.resolve())] == data_sha
    assert code[str(code_path.resolve())] == code_sha

    with pytest.raises(ValueError, match="historical data-input identity"):
        prep._validate_ulnar_proof_pin_namespaces(
            {}, {str(code_path): code_sha},
            {str(data_path): data_sha}, {str(code_path): code_sha})
    with pytest.raises(ValueError, match="executable dependency identity"):
        prep._validate_ulnar_proof_pin_namespaces(
            {str(data_path): data_sha}, {},
            {str(data_path): data_sha}, {str(code_path): code_sha})


def test_common_field_coordinates_require_matching_step_zero_capture_and_csv():
    columns = prep.COMMON_FIELD_COORDINATE_COLUMNS
    header = "step,time_s,geometry_mode,common_coordinate_solver_status," + ",".join(columns)
    row = "0,0,common_seven_coordinate_v1,0," + ",".join(["0.25"] * 7)
    coordinates, count = prep._accepted_step_zero_common_coordinates(
        {"accepted_step": 0, "accepted_time_s": 0.0}, header + "\n" + row)
    np.testing.assert_array_equal(coordinates, np.full(7, 0.25, dtype="<f4"))
    assert count == 1

    with pytest.raises(ValueError, match="requires accepted step zero"):
        prep._accepted_step_zero_common_coordinates(
            {"accepted_step": 1, "accepted_time_s": 0.01}, header + "\n" + row)
    with pytest.raises(ValueError, match="requires accepted time zero"):
        prep._accepted_step_zero_common_coordinates(
            {"accepted_step": 0, "accepted_time_s": 0.01}, header + "\n" + row)


def test_common_field_coordinates_reject_disagreeing_duplicate_step_zero_q():
    columns = prep.COMMON_FIELD_COORDINATE_COLUMNS
    header = "step,time_s,geometry_mode,common_coordinate_solver_status," + ",".join(columns)
    first = "0,0,common_seven_coordinate_v1,0," + ",".join(["0.25"] * 7)
    second_values = ["0.25"] * 7
    second_values[3] = "0.5"
    second = "0,0,common_seven_coordinate_v1,0," + ",".join(second_values)
    with pytest.raises(ValueError, match="disagree on solved common coordinates"):
        prep._accepted_step_zero_common_coordinates(
            {"accepted_step": 0, "accepted_time_s": 0.0},
            header + "\n" + first + "\n" + second)


def test_network_screen_normalizes_exact_integer_pair_arrays_for_existing_oracle(monkeypatch):
    pair_map = {
        edge: np.asarray([[2, 3], [4, 5]], dtype="<i8")
        for edge in (*prep.NETWORK_EDGES, *prep.SVC_PORTS)
    }
    monkeypatch.setattr(
        prep, "_runtime_network_source_pairs", lambda _edges, _separate: pair_map)
    captured = {}

    def oracle(_verifier, surfaces, edges, source_pairs):
        captured["surfaces"] = surfaces
        captured["edges"] = edges
        captured["source_pairs"] = source_pairs
        return {"passed": False}

    monkeypatch.setattr(prep.geometry_gate, "audit_network_rows_against_all_targets", oracle)
    prep.run_all_target_network_screen({"surfaces": {"frozen": True}}, {}, {})
    assert captured["source_pairs"] == {
        edge: [(2, 3), (4, 5)] for edge in pair_map
    }
    assert all(type(value) is int for pairs in captured["source_pairs"].values()
               for pair in pairs for value in pair)


def test_network_screen_rejects_noninteger_source_pair_arrays(monkeypatch):
    pair_map = {
        edge: np.asarray([[2.0, 3.0]], dtype="<f8")
        for edge in (*prep.NETWORK_EDGES, *prep.SVC_PORTS)
    }
    monkeypatch.setattr(
        prep, "_runtime_network_source_pairs", lambda _edges, _separate: pair_map)
    with pytest.raises(ValueError, match="source collar face pairs are malformed"):
        prep.run_all_target_network_screen({"surfaces": {}}, {}, {})


def test_common_field_displacement_uses_ordered_float32_map_products_and_sums():
    local = np.asarray([[16_777_216.0, -4.0, 2.0]], dtype="<f4")
    basis = np.zeros((1, 7, 4), dtype="<f4")
    basis[0, 0, :3] = [1.0, 2.0, -3.0]
    basis[0, 1, :3] = [-16_777_216.0, 1.0, 4.0]
    basis[0, 2, :3] = [0.25, -2.0, 1.0]
    coordinates = np.asarray([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype="<f4")
    output = prep._common_field_displace_vertices(local, basis, coordinates)
    expected = local.copy()
    for index in range(7):
        product = np.asarray(basis[:, index, :3] * coordinates[index], dtype="<f4")
        expected = np.asarray(expected + product, dtype="<f4")
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == np.dtype("<f4")


def test_common_field_displacement_rejects_nonfinite_or_wrong_record_width():
    local = np.zeros((1, 3), dtype="<f4")
    basis = np.zeros((1, 7, 4), dtype="<f4")
    coordinates = np.zeros(7, dtype="<f4")
    with pytest.raises(ValueError, match="malformed"):
        prep._common_field_displace_vertices(local, basis[:, :, :3], coordinates)
    basis[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="malformed"):
        prep._common_field_displace_vertices(local, basis, coordinates)


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
    current["passive_tissue_parent_bindings"]["anatomy_surfaces"][-1][
        "vertices"][0][0][0] = 1
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


def _ulnar_source_proof_fixture():
    pairs_by_edge = {
        (479, 487): np.asarray([[0, 0], [1, 1]], dtype="<i8"),
        (481, 489): np.asarray([[0, 1], [1, 0]], dtype="<i8"),
    }
    rows = {}
    source_map = {}
    links = []
    for parent, child in prep.ULNAR_SOURCE_LINKS:
        for sid, body in ((parent, 91), (child, 92)):
            vertices6 = np.asarray([
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ], dtype="<f4")
            faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<u4")
            rows[sid] = {
                "vertices6": vertices6,
                "faces": faces,
                "body_index": body,
            }
            source_map[str(sid)] = {
                "source_member": f"FJ{sid}",
                "source_sha256": f"{sid:064x}",
                "name": f"vessel {sid}",
                "provider": "BP3D",
                "body_index": body,
                "layer": 2,
                "vascular_compartment_id": 3 if sid in (479, 487) else 4,
                "source_owner_metadata": {"concept_id": f"FMA{sid}"},
            }
    for parent, child in prep.ULNAR_SOURCE_LINKS:
        pairs = pairs_by_edge[(parent, child)]
        face_hashes = {
            str(sid): prep._sha_bytes(rows[sid]["faces"].tobytes())
            for sid in (parent, child)
        }
        position_hashes = {str(sid): f"{sid + 1000:064x}"
                           for sid in (parent, child)}
        manifold_inputs = []
        for _stage in range(2):
            for sid in (parent, child):
                manifold_inputs.append({
                    "surface_identity": [51011, sid],
                    "input_diagnostics": {
                        "identity": [51011, sid],
                        "source_face_sha256": face_hashes[str(sid)],
                        "source_position_sha256": position_hashes[str(sid)],
                    },
                })
        links.append({
            "link_id": prep.ULNAR_SOURCE_LINKS[(parent, child)],
            "parent": [51011, parent],
            "child": [51011, child],
            "passed": True,
            "status": "SOURCE_RIGID_JUNCTION_CONTINUITY_SCREEN_PASSED_UNADMITTED",
            "current_capture_used": False,
            "source_geometry_frame": "registered NHRIGID source rest",
            "source_interface_derivation": {
                "source_face_pairs": pairs.tolist(),
                "source_face_pairs_sha256": prep._sha_bytes(pairs.tobytes()),
                "connected_contact_component_count": 1,
                "single_connected_source_contact_required": True,
            },
            "unexempted_source_pair_audit": {
                "identities": [[51011, parent], [51011, child]],
                "triangle_pair_count": len(pairs),
                "unexpected_triangle_pairs": pairs.tolist(),
            },
            "existing_per_port_owner_trace": {"manifold_inputs": manifold_inputs},
            "port_solid_and_calibre_proof": {
                "passed": True,
                "source_geometry_mutated": False,
                "calibre_gate": {"passed": True},
            },
        })
    data = {
        "schema": "numi.human.brachiocephalic-ulnar-source-proof.v1",
        "status": "SOURCE_RIGID_ULNAR_JUNCTIONS_PASSED_UNADMITTED",
        "passed": True,
        "requested_link_count": 2,
        "evaluated_link_count": 2,
        "first_failed_link": None,
        "source_candidate_admitted": False,
        "native_or_gpu_work_performed": False,
        "source_rigid_calibre_is_self_reference": True,
        "current_capture_or_motion_admission": False,
        "preflight_path": str(prep.ULNAR_SOURCE_PREFLIGHT.resolve()),
        "preflight_sha256": prep.ULNAR_SOURCE_PREFLIGHT_SHA256,
        "links": links,
    }
    return data, rows, source_map


def test_ulnar_source_proof_reuses_only_exact_connected_witness_pairs():
    data, rows, source_map = _ulnar_source_proof_fixture()
    pair = prep._ulnar_source_witness_pairs(
        data, 479, 487, rows, source_map, rows, source_map)
    assert pair["pair_count"] == 2
    assert pair["pairs"].tolist() == [[0, 0], [1, 1]]
    assert pair["source_rigid_calibre_is_self_reference"] is True
    assert "current capture" in pair["evidence_class"]


def test_ulnar_source_proof_rejects_changed_source_xyz_or_owner_identity():
    data, proof_rows, proof_map = _ulnar_source_proof_fixture()
    current_rows = {sid: {**row, "vertices6": row["vertices6"].copy()}
                    for sid, row in proof_rows.items()}
    current_map = {sid: dict(row) for sid, row in proof_map.items()}
    current_rows[487]["vertices6"][0, 0] = np.nextafter(
        np.float32(0.0), np.float32(1.0), dtype=np.float32)
    with pytest.raises(ValueError, match="exact local XYZ/face ancestry"):
        prep._ulnar_source_witness_pairs(
            data, 479, 487, proof_rows, proof_map, current_rows, current_map)

    data, proof_rows, proof_map = _ulnar_source_proof_fixture()
    current_map = {sid: dict(row) for sid, row in proof_map.items()}
    current_map["489"]["vascular_compartment_id"] = 3
    with pytest.raises(ValueError, match="owner identity"):
        prep._ulnar_source_witness_pairs(
            data, 481, 489, proof_rows, proof_map, proof_rows, current_map)


def test_ulnar_source_proof_rejects_link_status_and_pair_hash_drift():
    data, rows, source_map = _ulnar_source_proof_fixture()
    data["links"][0]["passed"] = False
    with pytest.raises(ValueError, match="unrecognized or failed edge"):
        prep._ulnar_source_witness_pairs(
            data, 479, 487, rows, source_map, rows, source_map)

    data, rows, source_map = _ulnar_source_proof_fixture()
    data["links"][0]["source_interface_derivation"]["source_face_pairs"][0][0] = 1
    with pytest.raises(ValueError, match="face-pair bytes/count"):
        prep._ulnar_source_witness_pairs(
            data, 479, 487, rows, source_map, rows, source_map)



def test_runtime_network_source_pairs_require_all_fifteen_exact_edges():
    edge_evidence = {
        f"{first[1]}->{second[1]}": {
            "pairs": np.asarray([[0, 0]], dtype="<i8"),
            "evidence_status": "PINNED_SOURCE_COLLAR",
        }
        for first, second in prep.NETWORK_EDGES
    }
    svc = {
        f"{first[1]}->{second[1]}": {"pairs": np.asarray([[0, 0]], dtype="<i8")}
        for first, second in prep.SVC_PORTS
    }
    pairs = prep._runtime_network_source_pairs(
        edge_evidence, {"svc_bridge_ports": svc})
    assert len(pairs) == 15
    assert prep.POTENTIAL_NONNETWORK_OVERLAP not in pairs
    del edge_evidence["479->487"]
    with pytest.raises(ValueError, match="exact 13 declared links"):
        prep._runtime_network_source_pairs(
            edge_evidence, {"svc_bridge_ports": svc})


def test_parent_motion_inventory_binds_native_terms_without_claiming_pose_reconstruction():
    receipt = {"passive_tissue_parent_bindings": {
        "source_payloads": [{"sha256": "a" * 64}, {"sha256": "b" * 64}],
        "anatomy_surfaces": [
            {"stable_id": 475, "vertices": [
                [[0, 20, 3, 0.25], [1, 91, 7, 0.75]],
                [[1, 91, 9, 1.0]],
            ]},
        ],
    }}
    result = prep._parent_motion_owner_inventory(receipt, {475, 477})
    assert result["rows"]["475"]["mode"] == (
        "existing_native_passive_anatomy_parent_influences")
    assert result["rows"]["475"]["used_source_payload_indices"] == [0, 1]
    assert result["rows"]["477"]["mode"] == "existing_native_rigid_body_binding"
    assert result["parent_pose_reconstruction_performed"] is False
    assert "source rigid-only" not in result["scope_limit"]


def test_parent_motion_inventory_rejects_nonpartitioned_or_out_of_range_terms():
    receipt = {"passive_tissue_parent_bindings": {
        "source_payloads": [{"sha256": "a" * 64}],
        "anatomy_surfaces": [
            {"stable_id": 475, "vertices": [[[0, 20, 3, 0.5]]]},
        ],
    }}
    with pytest.raises(ValueError, match="do not sum to one"):
        prep._parent_motion_owner_inventory(receipt, {475})
    receipt["passive_tissue_parent_bindings"]["anatomy_surfaces"][0]["vertices"] = [
        [[2, 20, 3, 1.0]]
    ]
    with pytest.raises(ValueError, match="invalid parent motion term"):
        prep._parent_motion_owner_inventory(receipt, {475})



def test_owner_composition_remaps_source_indices_by_payload_hash(tmp_path, monkeypatch):
    old, current, carrier, digest = _owner_composition_fixture(tmp_path)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER", carrier)
    monkeypatch.setattr(prep, "OWNER_COMPOSITION_CARRIER_SHA256", digest)
    bindings = current["passive_tissue_parent_bindings"]
    original_hashes = [row["sha256"] for row in bindings["source_payloads"]]
    remapped = list(reversed(bindings["source_payloads"]))
    new_index = {row["sha256"]: index for index, row in enumerate(remapped)}
    for row in bindings["anatomy_surfaces"]:
        for terms in row["vertices"]:
            for term in terms:
                old_hash = original_hashes[term[0]]
                term[0] = new_index[old_hash]
    bindings["source_payloads"] = remapped

    result = prep._owner_composition_delta(old, current)
    assert result["appended_source_index"] == 0
    assert result["source_payload_index_remap_by_sha256"]["0"] == 7
    assert result["source_payload_identity_set_preserved_plus_one_carrier"] is True
    assert result["prefix_source_payloads_exactly_preserved"] is False
    assert result["owner_rows_changed_only"] == [475, 477]



def test_exact_emitted_candidate_rows_and_owner_terms_are_separately_verified(tmp_path):
    rows = {
        475: {
            "vertices6": np.zeros((2, 6), dtype="<f4"),
            "faces": np.asarray([[0, 1, 0]], dtype="<u4"),
            "record": (20, 0, 2, 0, 3, 475, 2, 0),
        },
    }
    result = prep._compare_payload_rows_exact(
        rows, {475: {**rows[475], "vertices6": rows[475]["vertices6"].copy()}},
        [rows[475]["record"]], [rows[475]["record"]])
    assert result["status"] == "ALL_ABI5_ROWS_BYTE_EXACT_CAPTURED_SOURCE_CANDIDATE"

    _old, current, _carrier, _digest = _owner_composition_fixture(tmp_path)
    owner = prep._captured_candidate_owner_binding_delta(current, current)
    assert owner["status"] == "INVOKED_CAPTURE_AND_SOURCE_RECEIPT_OWNER_MAPS_EXACT"
    altered = copy.deepcopy(current)
    altered["provenance"]["source_id_map"]["475"]["source"] = "changed"
    with pytest.raises(ValueError, match="source identities differ"):
        prep._captured_candidate_owner_binding_delta(altered, current)


def test_explicit_current_capture_bundle_uses_verified_candidate_and_restores_pins(
    tmp_path, monkeypatch
):
    payload = tmp_path / "resting-thorax.nhanatomy"
    payload.write_bytes(b"ABI5 source payload")
    anatomy_receipt = {
        "payload": {
            "path": str(payload.resolve()),
            "sha256": prep._sha_file(payload),
        }
    }
    anatomy_receipt_path = tmp_path / "resting-anatomy-receipt.json"
    anatomy_receipt_path.write_text(json.dumps(anatomy_receipt))
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"accepted pack")
    capture_receipt = tmp_path / "step-0.receipt.json"
    capture_receipt.write_text('{"physical_endpoint":"accepted"}')
    inputs = {
        "capture_pack_path": str(pack.resolve()),
        "capture_pack_sha256": prep._sha_file(pack),
        "capture_receipt_path": str(capture_receipt.resolve()),
        "capture_receipt_sha256": prep._sha_file(capture_receipt),
        "anatomy_receipt_path": str(anatomy_receipt_path.resolve()),
        "anatomy_receipt_sha256": prep._sha_file(anatomy_receipt_path),
        "anatomy_payload_sha256": prep._sha_file(payload),
    }
    gate = prep.capture_gate
    names = (
        "CAPTURE_PACK", "CAPTURE_PACK_SHA256", "CAPTURE_RECEIPT",
        "CAPTURE_RECEIPT_SHA256", "ANATOMY_RECEIPT",
        "ANATOMY_RECEIPT_SHA256", "PREPARED_SUCCESSOR_RECEIPT",
        "PREPARED_SUCCESSOR_RECEIPT_SHA256", "ANATOMY_PAYLOAD_SHA256",
    )
    before = {name: getattr(gate, name) for name in names}

    def load_exact_candidate():
        assert gate.CAPTURE_PACK == pack.resolve()
        assert gate.CAPTURE_PACK_SHA256 == inputs["capture_pack_sha256"]
        assert gate.CAPTURE_RECEIPT == capture_receipt.resolve()
        assert gate.ANATOMY_RECEIPT == anatomy_receipt_path.resolve()
        assert gate.PREPARED_SUCCESSOR_RECEIPT == anatomy_receipt_path.resolve()
        return {
            "anatomy_receipt": anatomy_receipt,
            "payload_path": payload.resolve(),
            "capture_identity": {
                "pack_sha256": inputs["capture_pack_sha256"],
                "capture_receipt_sha256": inputs["capture_receipt_sha256"],
                "accepted_step": 0,
            },
        }

    monkeypatch.setattr(gate, "_load_current_inputs", load_exact_candidate)
    context = prep._load_current_capture_context(inputs)

    assert context["anatomy_receipt"] is anatomy_receipt
    assert context["explicit_capture_input_bundle"]["anatomy_payload_sha256"] == (
        inputs["anatomy_payload_sha256"])
    assert context["capture_source_sha256"] == {
        "capture_pack": inputs["capture_pack_sha256"],
        "capture_receipt": inputs["capture_receipt_sha256"],
        "anatomy_receipt": inputs["anatomy_receipt_sha256"],
        "anatomy_payload": inputs["anatomy_payload_sha256"],
    }
    assert {name: getattr(gate, name) for name in names} == before


def test_explicit_current_capture_bundle_rejects_hash_or_payload_drift(tmp_path):
    payload = tmp_path / "resting-thorax.nhanatomy"
    payload.write_bytes(b"source A")
    receipt = tmp_path / "anatomy-receipt.json"
    receipt.write_text(json.dumps({
        "payload": {"path": str(payload.resolve()), "sha256": prep._sha_file(payload)}
    }))
    pack = tmp_path / "step-0.mrvpack"
    pack.write_bytes(b"pack")
    capture_receipt = tmp_path / "step-0.receipt.json"
    capture_receipt.write_bytes(b"receipt")
    bundle = {
        "capture_pack_path": str(pack.resolve()),
        "capture_pack_sha256": prep._sha_file(pack),
        "capture_receipt_path": str(capture_receipt.resolve()),
        "capture_receipt_sha256": prep._sha_file(capture_receipt),
        "anatomy_receipt_path": str(receipt.resolve()),
        "anatomy_receipt_sha256": prep._sha_file(receipt),
        "anatomy_payload_sha256": prep._sha_file(payload),
    }

    altered_hash = dict(bundle)
    altered_hash["capture_pack_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        prep._validated_capture_input_bundle(altered_hash)

    wrong_payload = dict(bundle)
    wrong_payload["anatomy_payload_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="does not bind"):
        prep._validated_capture_input_bundle(wrong_payload)


def test_network_screen_passes_verified_continuity_and_progress_callback(monkeypatch):
    pair_map = {
        edge: np.asarray([[2, 3]], dtype="<i8")
        for edge in (*prep.NETWORK_EDGES, *prep.SVC_PORTS)
    }
    monkeypatch.setattr(
        prep, "_runtime_network_source_pairs", lambda _edges, _separate: pair_map)
    continuity = {"status": "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED"}
    progress = lambda _stage: None
    captured = {}

    source_inventory = {"candidate_surface_count": 17}

    def oracle(_verifier, surfaces, edges, source_pairs, *,
               verified_network_continuity=None,
               verified_network_source_inventory=None, progress_callback=None):
        captured.update({
            "surfaces": surfaces,
            "edges": edges,
            "source_pairs": source_pairs,
            "continuity": verified_network_continuity,
            "source_inventory": verified_network_source_inventory,
            "progress": progress_callback,
        })
        return {"passed": True}

    monkeypatch.setattr(prep.geometry_gate,
                        "audit_network_rows_against_all_targets", oracle)
    result = prep.run_all_target_network_screen(
        {"surfaces": {"bound": True}}, {}, {},
        verified_network_continuity=continuity,
        verified_network_source_inventory=source_inventory,
        progress_callback=progress)
    assert result["passed"] is True
    assert captured["continuity"] is continuity
    assert captured["source_inventory"] is source_inventory
    assert captured["progress"] is progress
    assert captured["edges"] == (*prep.NETWORK_EDGES, *prep.SVC_PORTS)

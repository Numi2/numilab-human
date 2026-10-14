import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest

from numilab_human import resting_intercostal_mesh_arrangement as arrangement
from numilab_human.resting_anatomy_interface_patch import encode_payload


def _row(stable_id, faces):
    vertices = np.asarray([
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
    ], dtype="<f4")
    return {
        "record": (20, 0, 4, 0, len(faces) * 3, stable_id, 1, 0),
        "body_index": 20,
        "layer": 1,
        "flags": 0,
        "vertices6": vertices,
        "faces": np.asarray(faces, dtype="<u4"),
    }


def _pair(value):
    fraction = Fraction(value)
    return [str(fraction.numerator), str(fraction.denominator)]


def _candidate():
    points = [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [Fraction(1, 2), 0, 0],
    ]
    maps = [
        [[1, 313, 0, "1", "1"]],
        [[1, 313, 1, "1", "1"]],
        [[1, 313, 2, "1", "1"]],
        [[1, 313, 3, "1", "1"]],
        [[1, 313, 0, "1", "2"], [1, 313, 1, "1", "2"]],
    ]
    raw = json.dumps({
        "faces": {"313": [[0, 1, 2], [0, 2, 3], [0, 3, 1], [0, 4, 2]]},
        "face_ancestry": {"313": [
            {"input_index": 0, "origin_kind": 0, "parent_face_index": 0,
             "parent_stable_id": 313, "partition_role": 0},
            {"input_index": 1, "origin_kind": 0, "parent_face_index": 1,
             "parent_stable_id": 313, "partition_role": 0},
            {"input_index": 2, "origin_kind": 0, "parent_face_index": 2,
             "parent_stable_id": 313, "partition_role": 0},
            {"input_index": 0, "origin_kind": 1, "parent_face_index": 0,
             "parent_stable_id": 313, "partition_role": 0},
        ]},
        "parent_maps": {"313": maps},
        "points_exact": {"313": [[_pair(c) for c in point] for point in points]},
    }, separators=(",", ":")).encode()
    return {
        "source_payload_sha256": arrangement.REPAIR_PARENT_PAYLOAD_SHA256,
        "source_payload_index": 1,
        "candidate_raw": raw,
        "candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "lineage_sha256": "b" * 64,
        "report_sha256": "c" * 64,
    }


def _repair_report():
    return {"surfaces": {"313": {
        "source_face_count": 4,
        "retained_face_count": 3,
        "removed_source_face_ids": [3],
        "source_coordinates_modified": False,
        "new_vertices": 0,
        "new_faces": 0,
    }}}


def _candidate_with_null_slot():
    value = _candidate()
    candidate = json.loads(value["candidate_raw"])
    candidate["faces"]["313"][3] = None
    value["candidate_raw"] = json.dumps(candidate, separators=(",", ":")).encode()
    value["candidate_sha256"] = hashlib.sha256(value["candidate_raw"]).hexdigest()
    return value, candidate


def test_route110_receipt_binds_offline_only_status_and_exact_rows():
    pins = {"npz": "a", "report": "b", "runner": "c", "checkpoint": "d"}
    route_data = {"retained_status": "ERROR_EXACT_P6_AUDIT", "records": [{"stable_id": 525}]}
    declaration = {
        "candidate_npz_sha256": "a",
        "candidate_report_sha256": "b",
        "candidate_runner_sha256": "c",
        "target_preparation_checkpoint_sha256": "d",
        "retained_candidate_status": "ERROR_EXACT_P6_AUDIT",
        "exact_candidate_checks": 0,
        "native_or_gpu_run": False,
        "records": [{"stable_id": 525}],
    }
    arrangement._validate_route110_source_declaration(declaration, route_data, pins)

    for key, value in (("native_or_gpu_run", True), ("native_or_gpu_run", None),
                       ("records", []), ("exact_candidate_checks", 1)):
        mutated = dict(declaration)
        mutated[key] = value
        with pytest.raises(ValueError, match="route110 source/report/code pins"):
            arrangement._validate_route110_source_declaration(mutated, route_data, pins)


def test_null_retirement_ledger_reclassifies_complete_slot_with_frozen_parentage():
    value, candidate = _candidate_with_null_slot()
    lineage = {"adds_physical_mass_or_state": False}
    ledger = arrangement.build_null_face_retirement_ledger(
        stable_id=313, candidate=candidate, candidate_sha256=value["candidate_sha256"],
        lineage=lineage, lineage_sha256=value["lineage_sha256"],
        report_sha256=value["report_sha256"])

    assert ledger["candidate_geometry_changed"] is False
    assert ledger["candidate_admission"] is False
    assert ledger["null_face_slots"] == [3]
    retirement = ledger["retirements"][0]
    assert retirement["source_ancestry"] == {
        "input_index": 0, "origin_kind": 1, "parent_face_index": 0,
        "parent_stable_id": 313, "partition_role": 0,
    }
    assert retirement["effective_retired_ancestry"] == retirement["source_ancestry"]
    assert retirement["classification"] == "fully_retired_local_arrangement"
    assert arrangement.validate_null_face_retirement_ledger(
        stable_id=313, candidate=candidate, candidate_sha256=value["candidate_sha256"],
        lineage=lineage, lineage_sha256=value["lineage_sha256"],
        report_sha256=value["report_sha256"], ledger=ledger) == {3: retirement}


@pytest.mark.parametrize("damage", ["missing", "wrong_ancestry", "extra_slot"])
def test_null_retirement_ledger_rejects_incomplete_or_unbound_records(damage):
    value, candidate = _candidate_with_null_slot()
    lineage = {"adds_physical_mass_or_state": False}
    ledger = arrangement.build_null_face_retirement_ledger(
        stable_id=313, candidate=candidate, candidate_sha256=value["candidate_sha256"],
        lineage=lineage, lineage_sha256=value["lineage_sha256"],
        report_sha256=value["report_sha256"])
    if damage == "missing":
        ledger["retirements"] = []
    elif damage == "wrong_ancestry":
        ledger["retirements"][0]["effective_retired_ancestry"]["parent_face_index"] = 2
    else:
        ledger["null_face_slots"].append(2)
    with pytest.raises(ValueError, match="ledger is incomplete"):
        arrangement.validate_null_face_retirement_ledger(
            stable_id=313, candidate=candidate, candidate_sha256=value["candidate_sha256"],
            lineage=lineage, lineage_sha256=value["lineage_sha256"],
            report_sha256=value["report_sha256"], ledger=ledger)


def test_local_composer_rejects_null_face_without_retirement_declaration():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    value, _candidate_data = _candidate_with_null_slot()
    with pytest.raises(ValueError, match="null face slots lack a validated retirement ledger"):
        arrangement.compose_intercostal_arrangement_row(
            stable_id=313, source_row=source, parent_row=parent,
            arrangement=value, repair_report=_repair_report())


def _append_unreferenced_candidate_vertex(value, point, terms):
    candidate = json.loads(value["candidate_raw"])
    candidate["points_exact"]["313"].append([_pair(component) for component in point])
    candidate["parent_maps"]["313"].append(terms)
    value["candidate_raw"] = json.dumps(candidate, separators=(",", ":")).encode()
    value["candidate_sha256"] = hashlib.sha256(value["candidate_raw"]).hexdigest()
    return value


def test_composer_allows_unreferenced_exact_source_prefix_slot():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    value = _append_unreferenced_candidate_vertex(
        _candidate(), [0, 0, 0], [[1, 313, 0, "1", "1"]])
    _composed, summary = arrangement.compose_intercostal_arrangement_row(
        stable_id=313, source_row=source, parent_row=parent,
        arrangement=value, repair_report=_repair_report())
    assert summary["unreferenced_candidate_vertex_count"] == 1
    assert summary["unreferenced_source_prefix_vertex_count"] == 1
    assert summary["unreferenced_documented_operation_retirement_count"] == 0


def test_composer_rejects_unreferenced_generated_vertex_without_operation():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    value = _append_unreferenced_candidate_vertex(
        _candidate(), [Fraction(1, 2), 0, 0],
        [[1, 313, 0, "1", "2"], [1, 313, 1, "1", "2"]])
    with pytest.raises(ValueError, match="lacks an executed local-retirement operation"):
        arrangement.compose_intercostal_arrangement_row(
            stable_id=313, source_row=source, parent_row=parent,
            arrangement=value, repair_report=_repair_report())


def test_local_composer_preserves_original_vertex_prefix_and_retained_faces():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    composed, summary = arrangement.compose_intercostal_arrangement_row(
        stable_id=313, source_row=source, parent_row=parent,
        arrangement=_candidate(), repair_report=_repair_report())

    np.testing.assert_array_equal(composed["vertices6"][:4], source["vertices6"])
    np.testing.assert_array_equal(composed["faces"][:3], parent["faces"])
    np.testing.assert_array_equal(composed["faces"][3], [0, 4, 2])
    assert len(composed["vertices6"]) == 5
    assert summary["new_appended_vertex_count"] == 1
    assert summary["retained_n27_source_face_count"] == 3
    assert summary["patch_parent_face_ids_in_N27_namespace"] == [0]
    assert summary["physical_mass_or_state_added"] is False


def test_local_composer_rejects_child_support_outside_declared_028_parent_face():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    value = _candidate()
    candidate = json.loads(value["candidate_raw"])
    candidate["parent_maps"]["313"][4] = [[1, 313, 2, "1", "2"], [1, 313, 3, "1", "2"]]
    candidate["points_exact"]["313"][4] = [_pair(Fraction(1, 2)), _pair(Fraction(1, 2)), _pair(0)]
    value["candidate_raw"] = json.dumps(candidate, separators=(",", ":")).encode()
    value["candidate_sha256"] = hashlib.sha256(value["candidate_raw"]).hexdigest()
    with pytest.raises(ValueError, match="point 4 is not canonical F32"):
        arrangement.compose_intercostal_arrangement_row(
            stable_id=313, source_row=source, parent_row=parent,
            arrangement=value, repair_report=_repair_report())


def test_local_composer_rejects_wrong_028_face_namespace():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    value = _candidate()
    candidate = json.loads(value["candidate_raw"])
    candidate["face_ancestry"]["313"][3]["parent_face_index"] = 3
    candidate["face_ancestry"]["313"][3]["input_index"] = 3
    value["candidate_raw"] = json.dumps(candidate, separators=(",", ":")).encode()
    value["candidate_sha256"] = hashlib.sha256(value["candidate_raw"]).hexdigest()
    with pytest.raises(ValueError, match="ancestry"):
        arrangement.compose_intercostal_arrangement_row(
            stable_id=313, source_row=source, parent_row=parent,
            arrangement=value, repair_report=_repair_report())


def test_local_composer_requires_original_raw_vertex_prefix_to_match_028_parent():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    parent["vertices6"][0, 0] = np.float32(0.01)
    with pytest.raises(ValueError, match="changed original vertices6 prefix"):
        arrangement.compose_intercostal_arrangement_row(
            stable_id=313, source_row=source, parent_row=parent,
            arrangement=_candidate(), repair_report=_repair_report())


def test_existing_abi5_encoder_rebases_offsets_and_preserves_unselected_rows():
    original = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    unrelated = _row(10, [[0, 1, 2], [0, 2, 3]])
    header = (b"NHANAT1\0", 5, 2, 8, 18, 0x12345678, b"x" * 32)
    raw = encode_payload(header, {313: original, 10: unrelated})
    repaired_parent = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    replacement, _summary = arrangement.compose_intercostal_arrangement_row(
        stable_id=313,
        source_row=original,
        parent_row=repaired_parent,
        arrangement=_candidate(),
        repair_report=_repair_report(),
    )

    output, proof = arrangement.encode_payload_with_row_replacements(
        raw, {313: replacement})
    decoded_header, _records, rows = __import__(
        "numilab_human.resting_pleura_proxy", fromlist=["_parse_payload"]
    )._parse_payload(output)

    assert decoded_header[2] == 2
    assert decoded_header[3] == 9
    assert decoded_header[5:] == header[5:]
    np.testing.assert_array_equal(rows[10]["vertices6"], unrelated["vertices6"])
    np.testing.assert_array_equal(rows[10]["faces"], unrelated["faces"])
    np.testing.assert_array_equal(rows[313]["vertices6"][:4], original["vertices6"])
    assert proof["replaced_stable_ids"] == [313]
    assert proof["unmodified_rows_byte_exact"] == 1
    assert proof["global_offsets_rebased_by_existing_abi5_encoder"] is True


def _f32_word(value):
    return int(np.asarray([value], dtype="<f4").view("<u4")[0])


def test_exact_rational_f32_rounding_is_ties_to_even_without_double_rounding():
    tie_even_low = Fraction(1) + Fraction(1, 1 << 24)
    tie_even_high = Fraction(1) + Fraction(3, 1 << 24)
    above_tie_below_f64_half_ulp = Fraction(1) + Fraction(1, 1 << 24) + Fraction(1, 1 << 54)

    assert _f32_word(arrangement._round_fraction_to_f32(tie_even_low)) == 0x3F800000
    assert _f32_word(arrangement._round_fraction_to_f32(tie_even_high)) == 0x3F800002
    assert _f32_word(arrangement._round_fraction_to_f32(above_tie_below_f64_half_ulp)) == 0x3F800001
    assert _f32_word(np.float32(float(above_tie_below_f64_half_ulp))) == 0x3F800000


def test_captured_world_parent_maps_emit_local_abi5_coordinates():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    value = _candidate()
    candidate = json.loads(value["candidate_raw"])
    translation = (Fraction(10), Fraction(-20), Fraction(30))
    local_exact = candidate["points_exact"]["313"]
    candidate["points_exact"]["313"] = [
        [_pair(Fraction(int(point[axis][0]), int(point[axis][1])) + translation[axis]) for axis in range(3)]
        for point in local_exact
    ]
    value["candidate_raw"] = json.dumps(candidate, separators=(",", ":")).encode()
    value["candidate_sha256"] = hashlib.sha256(value["candidate_raw"]).hexdigest()
    captured = np.asarray(source["vertices6"][:, :3] + np.asarray(translation, dtype="<f4"), dtype="<f4")

    points, maps = arrangement._read_row_maps(
        stable_id=313, source_vertices6=source["vertices6"],
        points_exact=candidate["points_exact"]["313"],
        parent_maps=candidate["parent_maps"]["313"], source_payload_index=1,
        captured_source_xyz=captured,
    )

    np.testing.assert_array_equal(points[:4], source["vertices6"][:, :3])
    np.testing.assert_array_equal(points[4], np.asarray([0.5, 0.0, 0.0], dtype="<f4"))
    assert len(maps) == 5


def test_capture_world_point_must_match_exact_positive_parent_map():
    source = _row(313, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
    value = _candidate()
    captured = np.asarray(source["vertices6"][:, :3] + np.asarray([10, 0, 0], dtype="<f4"), dtype="<f4")
    with pytest.raises(ValueError, match="captured frame"):
        arrangement._read_row_maps(
            stable_id=313, source_vertices6=source["vertices6"],
            points_exact=json.loads(value["candidate_raw"])["points_exact"]["313"],
            parent_maps=json.loads(value["candidate_raw"])["parent_maps"]["313"],
            source_payload_index=1, captured_source_xyz=captured,
        )



def _contraction_replay_inputs():
    point = {vertex: [[str(vertex), "1"], ["0", "1"], ["0", "1"]]
             for vertex in (2052, 49441, 49442, 2056, 2059)}
    terms = {vertex: [[1, 313, vertex, "1", "1"]]
             for vertex in (2052, 49441, 49442, 2056, 2059)}
    ancestry = {
        "input_index": 4114, "origin_kind": 1, "parent_face_index": 4114,
        "parent_stable_id": 313, "partition_role": 0,
    }
    row = lambda: {"points_exact": dict(point), "parent_maps": dict(terms)}
    return {
        "pre_face": [2052, 49441, 2056],
        "post_face": [2052, 2059, 2056],
        "current_face": [2052, 2059, 2056],
        "pre_ancestry": ancestry,
        "post_ancestry": ancestry,
        "current_ancestry": ancestry,
        "operation": {
            "contracted_vertices": [49441, 49442],
            "max_displacement_m": 0.00019755848423334504,
            "target_vertex": 2059,
        },
        "sidecar_entry": {
            "new_face_slot": arrangement.CONTRACTION_313_SLOT,
            "paired_witness_faces": [4122, 4124],
            "source_ancestry": {
                "input_index": 4114, "origin_kind": 0, "parent_face_index": 4114,
                "parent_stable_id": 313, "partition_role": 0,
            },
            "source_face_slot": 4114,
        },
        "pre_row": row(),
        "post_row": row(),
        "current_row": row(),
    }


def test_313_contraction_support_exception_replays_exact_frozen_face_slot():
    proof = arrangement._validate_contraction_face_replay(**_contraction_replay_inputs())
    assert proof["face_slot"] == 98694
    assert proof["source_face_slot"] == 4114
    assert proof["paired_witness_faces"] == [4122, 4124]
    assert proof["executed_local_contraction_only_for_this_support_exception"] is True


def _contraction_group_replay_inputs():
    ancestry = {
        "input_index": 4114, "origin_kind": 1, "parent_face_index": 4114,
        "parent_stable_id": 313, "partition_role": 0,
    }
    source_ancestry = {**ancestry, "origin_kind": 0}
    specs = (
        (98694, [2052, 49441, 2056], [2052, 2059, 2056]),
        (98695, [2052, 49442, 49441], None),
        (98696, [2055, 49442, 2052], [2055, 2059, 2052]),
        (98697, [2055, 49441, 49442], None),
    )
    pre_row = {"faces": [None] * 98698, "points_exact": {}, "parent_maps": {}}
    post_row = {"faces": [None] * 98698, "points_exact": {}, "parent_maps": {}}
    current_row = {"faces": [None] * 98698, "points_exact": {}, "parent_maps": {}}
    pre_ancestry = [None] * 98698
    post_ancestry = [None] * 98698
    current_ancestry = [None] * 98698
    sidecar = []
    for slot, pre_face, post_face in specs:
        pre_row["faces"][slot] = pre_face
        post_row["faces"][slot] = post_face
        current_row["faces"][slot] = post_face
        pre_ancestry[slot] = dict(ancestry)
        post_ancestry[slot] = dict(ancestry)
        current_ancestry[slot] = dict(ancestry)
        sidecar.append({
            "new_face_slot": slot,
            "paired_witness_faces": [4122, 4124],
            "source_ancestry": dict(source_ancestry),
            "source_face_slot": 4114,
        })
    for vertex in (2052, 2055, 2056, 2059, 49441, 49442):
        point = [[str(vertex), "1"], ["0", "1"], ["0", "1"]]
        terms = [[1, 313, vertex, "1", "1"]]
        for row in (pre_row, post_row, current_row):
            row["points_exact"][vertex] = point
            row["parent_maps"][vertex] = terms
    local_lineage = {"retired": [
        {"slot": 98695, "face": [2052, 2059, 2059], "kind": "exact_zero_area",
         "ancestry": dict(ancestry)},
        {"slot": 98697, "face": [2055, 2059, 2059], "kind": "exact_zero_area",
         "ancestry": dict(ancestry)},
    ]}
    return {
        "pre_row": pre_row,
        "post_row": post_row,
        "current_row": current_row,
        "pre_ancestry": pre_ancestry,
        "post_ancestry": post_ancestry,
        "current_ancestry": current_ancestry,
        "operation": {
            "contracted_vertices": [49441, 49442],
            "max_displacement_m": 0.00019755848423334504,
            "target_vertex": 2059,
        },
        "sidecar_entries": sidecar,
        "local_lineage": local_lineage,
    }


def test_313_contraction_group_replays_both_live_faces_and_retired_twins():
    proof = arrangement._validate_contraction_face_group_replay(
        **_contraction_group_replay_inputs())
    assert set(proof) == {98694, 98696}
    assert proof[98694]["post_face"] == [2052, 2059, 2056]
    assert proof[98696]["post_face"] == [2055, 2059, 2052]
    assert proof[98696]["patch_face_slots"] == [98694, 98695, 98696, 98697]
    assert proof[98696]["retired_zero_area_face_slots"] == [98695, 98697]


@pytest.mark.parametrize("damage", ["sibling_winding", "missing_retirement", "wrong_witness"])
def test_313_contraction_group_rejects_unpinned_sibling_details(damage):
    inputs = _contraction_group_replay_inputs()
    if damage == "sibling_winding":
        inputs["current_row"]["faces"][98696] = [2055, 2052, 2059]
    elif damage == "missing_retirement":
        inputs["local_lineage"]["retired"].pop()
    else:
        inputs["sidecar_entries"][2]["paired_witness_faces"] = [4122, 4123]
    with pytest.raises(ValueError, match="313 contraction"):
        arrangement._validate_contraction_face_group_replay(**inputs)


def test_sidecar_source_patch_support_is_exact_and_pinned_to_ancestry():
    vertices = np.zeros((6, 6), dtype="<f4")
    vertices[:, 3:] = np.asarray([0, 0, 1], dtype="<f4")
    vertices[:6, :3] = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [0, 0, 1], [2, 2, 2],
    ], dtype="<f4")
    faces = np.zeros((4125, 3), dtype="<u4")
    faces[4114] = [0, 1, 2]
    faces[4122] = [3, 4, 2]
    faces[4124] = [1, 3, 4]
    origin = {
        "input_index": 4114, "origin_kind": 1, "parent_face_index": 4114,
        "parent_stable_id": 313, "partition_role": 0,
    }
    entry = {
        "new_face_slot": 9000,
        "source_face_slot": 4114,
        "paired_witness_faces": [4122, 4124],
        "source_ancestry": {**origin, "origin_kind": 0},
    }
    proof = arrangement._validate_sidecar_source_patch_support(
        stable_id=313, face_slot=9000, origin=origin, sidecar_entry=entry,
        source_vertices6=vertices, source_faces=faces, support_vertex_ids={3, 4})
    assert proof["source_patch_face_indices_028_namespace"] == [4114, 4122, 4124]
    assert proof["support_is_exact_float32_coordinate_quotient_of_declared_sidecar_patch"]
    bad = {**entry, "source_face_slot": 4115}
    with pytest.raises(ValueError, match="sidecar does not bind"):
        arrangement._validate_sidecar_source_patch_support(
            stable_id=313, face_slot=9000, origin=origin, sidecar_entry=bad,
            source_vertices6=vertices, source_faces=faces, support_vertex_ids={3, 4})


def test_parent_face_support_accepts_only_bitwise_float32_coordinate_aliases():
    vertices = np.zeros((5, 6), dtype="<f4")
    vertices[:, 3:] = np.asarray([0, 0, 1], dtype="<f4")
    vertices[:5, :3] = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [2, 2, 2],
    ], dtype="<f4")
    proof = arrangement._exact_source_face_coordinate_aliases(
        source_vertices6=vertices,
        source_vertex_ids={3},
        source_face_corner_ids={0, 1, 2},
    )
    assert proof["source_face_corner_coordinate_aliases"] == {"3": [1]}
    assert arrangement._exact_source_face_coordinate_aliases(
        source_vertices6=vertices,
        source_vertex_ids={4},
        source_face_corner_ids={0, 1, 2},
    ) is None


def test_313_contraction_source_patch_uses_exact_coordinate_quotient_only():
    vertices = np.zeros((6, 6), dtype="<f4")
    vertices[:, 3:] = np.asarray([0, 0, 1], dtype="<f4")
    vertices[:6, :3] = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [0, 0, 1], [2, 2, 2],
    ], dtype="<f4")
    faces = np.zeros((4125, 3), dtype="<u4")
    faces[4114] = [0, 1, 2]
    faces[4122] = [3, 4, 2]
    faces[4124] = [1, 3, 4]
    proof = {
        "kind": "pinned_313_local_contraction_patch_replay",
        "source_patch_face_indices_028_namespace": [4114, 4122, 4124],
    }
    accepted = arrangement._validate_313_contraction_source_patch_support(
        source_vertices6=vertices, source_faces=faces,
        support_vertex_ids={3, 4}, proof=proof)
    assert accepted["support_source_patch_corner_aliases"]["3"] == [1, 3]
    with pytest.raises(ValueError, match="escapes its exact sidecar source patch"):
        arrangement._validate_313_contraction_source_patch_support(
            source_vertices6=vertices, source_faces=faces,
            support_vertex_ids={5}, proof=proof)


@pytest.mark.parametrize("damage", ["different_target", "different_winding", "different_sidecar"])
def test_313_contraction_replay_rejects_any_other_face_or_ancestry(damage):
    inputs = _contraction_replay_inputs()
    if damage == "different_target":
        inputs["operation"]["target_vertex"] = 2058
    elif damage == "different_winding":
        inputs["post_face"] = [2052, 2056, 2059]
        inputs["current_face"] = [2052, 2056, 2059]
    else:
        inputs["sidecar_entry"]["source_face_slot"] = 4115
    with pytest.raises(ValueError, match="313 contraction"):
        arrangement._validate_contraction_face_replay(**inputs)


def _tip_retessellation_fixture():
    baseline_faces = [[0, 1, 3], [1, 2, 3], [2, 0, 3]]
    baseline_ancestry = [
        {"input_index": slot, "origin_kind": 0, "parent_face_index": slot,
         "parent_stable_id": 317, "partition_role": 0}
        for slot in range(3)
    ]
    source_ancestry = {
        "input_index": 0, "local_three_face_tip_retessellation": True,
        "origin_kind": 1, "parent_face_index": 0,
        "parent_source_face_indices": [0, 1, 2],
        "parent_stable_id": 317, "partition_role": 0,
    }
    patch = {
        "new_face": [0, 1, 2], "new_slot": 3, "old_tip": 3,
        "original_vertex": 0, "retired_slots": [0, 1, 2],
        "signed_geometric_volume_change_m3": 1e-12,
        "source_face_ancestry": source_ancestry,
        "unchanged_oriented_boundary_edges": [[0, 1], [1, 2], [2, 0]],
    }
    retired = [
        {"ancestry": dict(baseline_ancestry[slot]), "face": list(baseline_faces[slot]),
         "slot": slot}
        for slot in range(3)
    ]
    operations = {"patches": [patch], "retired": retired}
    executed_faces = [None, None, None, [0, 1, 2]]
    executed_ancestry = [*baseline_ancestry, source_ancestry]
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    maps = [[[1, 317, index, "1", "1"]] for index in range(4)]
    return {
        "baseline_faces": baseline_faces,
        "baseline_ancestry": baseline_ancestry,
        "executed_faces": executed_faces,
        "executed_ancestry": executed_ancestry,
        "baseline_points": points,
        "executed_points": [list(point) for point in points],
        "baseline_maps": maps,
        "executed_maps": [[list(term) for term in row] for row in maps],
        "operations": operations,
        "expected_signatures": (
            (3, (0, 1, 2), 3, (0, 1, 2), (0, 1, 2), ((0, 1), (1, 2), (2, 0))),
        ),
    }


def test_tip_retessellation_replays_closed_oriented_fan_and_fixed_ring():
    proof = arrangement._validate_tip_retessellation_transition(
        **_tip_retessellation_fixture())
    assert proof[3]["fixed_ring_vertex_ids"] == [0, 1, 2]
    assert proof[3]["source_point_arrays_unchanged"] is True
    assert proof[3]["retired_face_slots"] == [0, 1, 2]


@pytest.mark.parametrize("damage", ["cap_slot", "ring_orientation", "ring_point", "parent_terms"])
def test_tip_retessellation_rejects_unpinned_topology_or_geometry(damage):
    fixture = _tip_retessellation_fixture()
    if damage == "cap_slot":
        fixture["operations"]["patches"][0]["new_slot"] = 4
    elif damage == "ring_orientation":
        fixture["operations"]["patches"][0]["unchanged_oriented_boundary_edges"] = [
            [1, 0], [0, 2], [2, 1]]
    elif damage == "ring_point":
        fixture["executed_points"][0][0] = 0.01
    else:
        fixture["executed_maps"][1][0][3] = "2"
    with pytest.raises(ValueError, match="tip|317"):
        arrangement._validate_tip_retessellation_transition(**fixture)


def test_tip_retessellation_binds_positive_support_to_frozen_ring_not_one_triangle():
    fixture = _tip_retessellation_fixture()
    # This positive source parent lies outside the three retired face corners.
    # The frozen retessellation still authorizes it because the exact ring map
    # is unchanged from the executed candidate and is covered by the replay hash.
    fixture["baseline_maps"][2] = [[1, 317, 9, "1", "1"]]
    fixture["executed_maps"] = [[list(term) for term in row]
                                for row in fixture["baseline_maps"]]
    replay = arrangement._validate_tip_retessellation_transition(**fixture)[3]
    patch = fixture["operations"]["patches"][0]
    parent_maps = [
        ((0, Fraction(1)),), ((1, Fraction(1)),), ((9, Fraction(1)),),
        ((3, Fraction(1)),),
    ]
    source_faces = np.asarray([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    proof = arrangement._validate_tip_retessellation_support(
        patch=patch, raw_slot=3, face=[0, 1, 2],
        origin=patch["source_face_ancestry"], parent_maps=parent_maps,
        raw_points=fixture["baseline_points"],
        raw_parent_maps=fixture["baseline_maps"], source_faces=source_faces,
        source_vertex_count=10, replay_proof=replay, stable_id=317)
    assert proof["positive_parent_support_ids"] == [0, 1, 9]
    assert proof["positive_parent_support_need_not_be_single_triangle_corner_subset"] is True
    changed_maps = [[list(term) for term in row] for row in fixture["baseline_maps"]]
    changed_maps[2] = [[1, 317, 8, "1", "1"]]
    changed_parent_maps = [
        ((0, Fraction(1)),), ((1, Fraction(1)),), ((8, Fraction(1)),),
        ((3, Fraction(1)),),
    ]
    with pytest.raises(ValueError, match="ring point or positive-parent terms"):
        arrangement._validate_tip_retessellation_support(
            patch=patch, raw_slot=3, face=[0, 1, 2],
            origin=patch["source_face_ancestry"], parent_maps=changed_parent_maps,
            raw_points=fixture["baseline_points"], raw_parent_maps=changed_maps,
            source_faces=source_faces, source_vertex_count=10,
            replay_proof=replay, stable_id=317)



def _fan_reclassification_replay_fixture():
    ancestry = [
        {"input_index": 0, "origin_kind": 0, "parent_face_index": 0,
         "parent_stable_id": 313, "partition_role": 0},
        {"input_index": 1, "origin_kind": 0, "parent_face_index": 1,
         "parent_stable_id": 313, "partition_role": 0},
    ]
    current_ancestry = [dict(row) for row in ancestry]
    current_ancestry[1].update({
        "origin_kind": 1,
        "local_vertex_fan_split": True,
        "parent_source_face_indices": [1],
    })
    executed = {
        "faces": [[0, 1, 2], [0, 2, 3]],
        "face_ancestry": ancestry,
        "points_exact": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        "parent_maps": [[[0, 1]], [[1, 1]], [[2, 1]], [[3, 1]]],
    }
    current = {
        "faces": [list(face) for face in executed["faces"]],
        "face_ancestry": current_ancestry,
        "points_exact": [list(point) for point in executed["points_exact"]],
        "parent_maps": [[list(term) for term in row] for row in executed["parent_maps"]],
    }
    lineage = {
        "modified_original_facet_slots_reclassified_as_generated_source_children": [1],
        "repair_operations": {
            "vertex_fan_splits": [{"new_vertex": 4, "faces": [1, 4]}],
        },
    }
    return executed, current, lineage


def test_current_candidate_replay_accepts_only_pinned_fan_child_ancestry():
    executed, current, lineage = _fan_reclassification_replay_fixture()
    proof = arrangement._validate_current_candidate_replay(
        executed_row=executed, current_row=current, lineage=lineage, stable_id=313)
    assert proof["ancestry_reclassified_source_child_slots"] == 1
    assert proof["no_point_or_parent_map_geometry_changed"] is True


@pytest.mark.parametrize("damage", ["uncovered_slot", "multi_parent"])
def test_current_candidate_replay_rejects_unpinned_fan_child_ancestry(damage):
    executed, current, lineage = _fan_reclassification_replay_fixture()
    if damage == "uncovered_slot":
        lineage["repair_operations"]["vertex_fan_splits"][0]["faces"] = [4]
    else:
        current["face_ancestry"][1]["parent_source_face_indices"] = [1, 2]
    with pytest.raises(ValueError, match="313|fan-split|source parent|ancestry"):
        arrangement._validate_current_candidate_replay(
            executed_row=executed, current_row=current, lineage=lineage, stable_id=313)


def test_capture_reference_allows_new_decoder_context_but_keeps_geometry_identity():
    frozen = {
        "capture_manifest_sha256": "a" * 64,
        "pack_sha256": "b" * 64,
        "accepted_step": 0,
        "accepted_time_s": 0,
        "accepted_body20_pose": {"position_m": [1, 2, 3]},
        "rows": {"313": {"captured_world_parent_xyz_f32_sha256": "c" * 64}},
        "capture_loader_source_path": "/old/checkout/resting_anatomy_verify.py",
        "capture_loader_source_sha256": "d" * 64,
    }
    replay = {
        **frozen,
        "capture_loader_source_path": "/current/checkout/resting_anatomy_verify.py",
        "capture_loader_source_sha256": "e" * 64,
    }

    assert arrangement._capture_reference_identity(replay) == \
        arrangement._capture_reference_identity(frozen)

    replay["pack_sha256"] = "f" * 64
    assert arrangement._capture_reference_identity(replay) != \
        arrangement._capture_reference_identity(frozen)


def test_capture_reference_rejects_row_or_pose_drift_with_same_decoder_context():
    frozen = {
        "pack_sha256": "b" * 64,
        "accepted_body20_pose": {"position_m": [1, 2, 3]},
        "rows": {"313": {"captured_world_parent_xyz_f32_sha256": "c" * 64}},
        "capture_loader_source_path": "/checkout/resting_anatomy_verify.py",
        "capture_loader_source_sha256": "d" * 64,
    }
    changed_row = json.loads(json.dumps(frozen))
    changed_row["rows"]["313"]["captured_world_parent_xyz_f32_sha256"] = "e" * 64
    changed_pose = json.loads(json.dumps(frozen))
    changed_pose["accepted_body20_pose"]["position_m"][0] = 1.001

    assert arrangement._capture_reference_identity(changed_row) != \
        arrangement._capture_reference_identity(frozen)
    assert arrangement._capture_reference_identity(changed_pose) != \
        arrangement._capture_reference_identity(frozen)


def test_historical_capture_loader_is_bound_by_owner_relative_git_identity():
    relocated = "/retired/checkouts/human/src/numilab_human/resting_anatomy_verify.py"
    historical_sha = "00cf51d56f6152b56659eb5bc5569fe1d564d9ad6b6a7d79f2b754bf06c753ef"
    commit = arrangement._historical_capture_loader_commit(relocated, historical_sha)
    assert len(commit) == 40

    with pytest.raises(ValueError, match="wrong repository owner"):
        arrangement._historical_capture_loader_commit(
            "/retired/checkouts/human/src/numilab_lab/resting_anatomy_verify.py",
            historical_sha)
    with pytest.raises(ValueError, match="SHA is not present"):
        arrangement._historical_capture_loader_commit(relocated, "f" * 64)

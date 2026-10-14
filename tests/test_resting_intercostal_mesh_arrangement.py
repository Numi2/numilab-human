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

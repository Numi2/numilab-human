import numpy as np
import pytest

from numilab_human.cardiac_cavity_intersections import _audit_pair
from numilab_human.common_atlas_skin_clearance import (
    _baseline_self_pair_table_sha256,
    _exact_surface_records,
    _face_index_sha256,
    _float32_xyz_sha256,
    audit_incremental_skin_self_intersections,
)
from numilab_human.model import ImportError


def _full_self(vertices, faces):
    records = _exact_surface_records(np.asarray(vertices, dtype="<f4"), np.asarray(faces, dtype=np.int64))
    return _audit_pair(records, records, same_surface=True)


def _incremental(base, candidate, faces, baseline_audit=None, **pins):
    base = np.asarray(base, dtype="<f4")
    candidate = np.asarray(candidate, dtype="<f4")
    faces = np.asarray(faces, dtype=np.int64)
    audit = baseline_audit if baseline_audit is not None else _full_self(base, faces)
    return audit_incremental_skin_self_intersections(
        baseline_world_positions=base,
        candidate_world_positions=candidate,
        baseline_faces=faces,
        candidate_faces=faces,
        baseline_self_audit=audit,
        expected_baseline_world_f32_sha256=pins.get("baseline_sha", _float32_xyz_sha256(base)),
        expected_face_index_sha256=pins.get("faces_sha", _face_index_sha256(faces)),
        expected_baseline_self_pair_table_sha256=(
            pins["pairs_sha"] if "pairs_sha" in pins
            else _baseline_self_pair_table_sha256(audit, len(faces))
        ),
    )


def _pairs(audit):
    return {tuple(pair) for pair in audit["triangle_pairs"]}


def test_changed_higher_face_row_finds_intersection_with_lower_unchanged_face():
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.2, .1, 2], [.8, .1, 2], [.4, .8, 2],
    ], dtype="<f4")
    candidate = base.copy()
    candidate[3:6] = np.array([[.3, .1, -1], [.3, .8, 1], [.3, .5, -1]], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    result = _incremental(base, candidate, faces)
    full = _full_self(candidate, faces)

    assert result["changed_skin_face_rows"] == [1]
    assert result["triangle_pairs"] == [[0, 1]]
    assert result["triangle_pairs"] == full["triangle_pairs"]


def test_changed_changed_pair_is_canonicalized_and_checked_once():
    base = np.array([
        [0, 0, 2], [1, 0, 2], [0, 1, 2],
        [4, 0, 0], [4, 1, 0], [4, 0, 1],
    ], dtype="<f4")
    candidate = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.3, .1, -1], [.3, .8, 1], [.3, .5, -1],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    result = _incremental(base, candidate, faces)
    full = _full_self(candidate, faces)

    assert result["changed_skin_face_rows"] == [0, 1]
    assert result["fresh_changed_face_aabb_raw_occurrences_before_dedup"] == 2
    assert result["fresh_changed_face_aabb_candidate_pairs"] == 1
    assert result["triangle_pairs"] == full["triangle_pairs"] == [[0, 1]]


@pytest.mark.parametrize(
    ("name", "base", "candidate", "faces"),
    [
        (
            "edge",
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype="<f4"),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 1]], dtype="<f4"),
            np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64),
        ),
        (
            "vertex",
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 1], [0, -1, 1]], dtype="<f4"),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 1], [0, -2, 1]], dtype="<f4"),
            np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64),
        ),
    ],
)
def test_allowed_shared_edge_and_vertex_semantics_match_full_owner(name, base, candidate, faces):
    result = _incremental(base, candidate, faces)
    full = _full_self(candidate, faces)

    assert result["triangle_pairs"] == full["triangle_pairs"] == []
    assert result["fresh_changed_face_allowed_shared_vertex_or_edge_pair_count"] == 1


def test_one_bit_vertex_change_uses_original_global_face_rows():
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 2], [1, 0, 2], [0, 1, 2],
    ], dtype="<f4")
    candidate = base.copy()
    candidate[4, 0] = np.nextafter(candidate[4, 0], np.float32(np.inf))
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    result = _incremental(base, candidate, faces)
    full = _full_self(candidate, faces)

    assert result["changed_skin_vertex_rows"] == [4]
    assert result["changed_skin_face_rows"] == [1]
    assert result["triangle_pairs"] == full["triangle_pairs"] == []


def test_duplicate_face_with_three_shared_ids_remains_an_unallowed_pair():
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    candidate = base.copy()
    candidate[2, 1] = np.float32(1.25)
    faces = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    baseline = _full_self(base, faces)

    result = _incremental(base, candidate, faces, baseline)
    full = _full_self(candidate, faces)

    assert baseline["triangle_pairs"] == [[0, 1]]
    assert result["triangle_pairs"] == full["triangle_pairs"] == [[0, 1]]


def test_unmodified_baseline_intersection_survives_unrelated_vertex_change():
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.2, .1, -1], [.8, .1, 1], [.4, .8, -1],
        [10, 0, 0], [11, 0, 0], [10, 1, 0],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    candidate = base.copy()
    candidate[8, 1] = np.float32(1.25)
    baseline = _full_self(base, faces)

    result = _incremental(base, candidate, faces, baseline)
    full = _full_self(candidate, faces)

    assert baseline["triangle_pairs"] == [[0, 1]]
    assert result["triangle_pairs"] == full["triangle_pairs"] == [[0, 1]]
    assert result["reused_baseline_exact_pair_count"] == 1
    assert result["fresh_changed_face_exact_pair_count"] == 0
    assert result["unchanged_exact_pair_rows"] == [[0, 1]]
    assert result["aabb_work_scope"].startswith("fresh changed skin faces")


def test_no_change_reuses_complete_baseline_pair_set_with_no_fresh_work():
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.2, .1, -1], [.8, .1, 1], [.4, .8, -1],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    baseline = _full_self(base, faces)

    result = _incremental(base, base.copy(), faces, baseline)

    assert result["changed_skin_vertex_rows"] == []
    assert result["changed_skin_face_rows"] == []
    assert result["triangle_pairs"] == baseline["triangle_pairs"]
    assert result["reused_baseline_exact_pair_count"] == baseline["count"]
    assert result["fresh_changed_face_exact_pair_count"] == 0
    assert result["fresh_changed_face_aabb_candidate_pairs"] == 0
    assert result["fresh_changed_face_allowed_shared_vertex_or_edge_pair_count"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: {**row, "count": row["count"] + 1}, "count, order, or uniqueness"),
        (lambda row: {**row, "triangle_pairs": [[0, 99]], "count": 1}, "outside coverage"),
        (lambda row: {**row, "triangle_pairs": [[1, 0]], "count": 1}, "noncanonical or outside"),
        (lambda row: {**row, "triangle_pairs": row["triangle_pairs"] * 2, "count": row["count"] * 2},
         "count, order, or uniqueness"),
    ],
)
def test_malformed_baseline_self_pairs_fail_closed(mutate, message):
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.2, .1, -1], [.8, .1, 1], [.4, .8, -1],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    baseline = _full_self(base, faces)
    malformed = mutate(baseline)
    expected_hash = _baseline_self_pair_table_sha256(baseline, len(faces))

    with pytest.raises(ImportError, match=message):
        _incremental(base, base.copy(), faces, malformed, pairs_sha=expected_hash)


def test_nonfinite_coordinates_and_changed_topology_fail_closed():
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    baseline = _full_self(base, faces)
    bad = base.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ImportError, match="non-finite"):
        _incremental(base, bad, faces, baseline)

    changed_faces = faces[:, ::-1].copy()
    with pytest.raises(ImportError, match="changed face topology"):
        audit_incremental_skin_self_intersections(
            baseline_world_positions=base,
            candidate_world_positions=base,
            baseline_faces=faces,
            candidate_faces=changed_faces,
            baseline_self_audit=baseline,
            expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
            expected_face_index_sha256=_face_index_sha256(faces),
            expected_baseline_self_pair_table_sha256=_baseline_self_pair_table_sha256(baseline, len(faces)),
        )


@pytest.mark.parametrize(
    ("pin", "message"),
    [
        ("baseline_sha", "baseline world identity"),
        ("faces_sha", "face identity"),
        ("pairs_sha", "baseline exact-pair table"),
    ],
)
def test_incremental_self_audit_rejects_stale_baseline_pins(pin, message):
    base = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [.2, .1, -1], [.8, .1, 1], [.4, .8, -1],
    ], dtype="<f4")
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    baseline = _full_self(base, faces)
    with pytest.raises(ImportError, match=message):
        _incremental(base, base.copy(), faces, baseline, **{pin: "0" * 64})


def test_incremental_self_audit_rejects_degenerate_candidate_face():
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    candidate = base.copy()
    candidate[2] = [2, 0, 0]
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    baseline = _full_self(base, faces)

    with pytest.raises(ImportError, match="exactly degenerate triangle"):
        _incremental(base, candidate, faces, baseline)

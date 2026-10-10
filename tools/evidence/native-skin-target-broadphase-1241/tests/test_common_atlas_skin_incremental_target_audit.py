import numpy as np
import pytest

from numilab_human import common_atlas_skin_clearance as c
from numilab_human.common_atlas_skin_clearance import (
    _exact_surface_records,
    _face_index_sha256,
    _float32_xyz_sha256,
    _target_geometry_f32_sha256,
    _baseline_target_pair_table_sha256,
    _target_intersection_audit,
    audit_incremental_skin_target_intersections,
)
from numilab_human.model import ImportError


def _vertical_triangle(center_x, z_shift=0.0):
    return np.array([
        [center_x - 0.2, -0.2, -1.0 + z_shift],
        [center_x + 0.2, -0.2, -1.0 + z_shift],
        [center_x, 0.2, 1.0 + z_shift],
    ], dtype="<f4")


def _fixture():
    # Changed rows 2 and 4 are deliberately sparse.
    base = np.vstack([
        _vertical_triangle(0.0),
        _vertical_triangle(10.0),
        _vertical_triangle(0.0, 2.0),
        _vertical_triangle(20.0, 2.0),
        _vertical_triangle(0.0),
    ])
    faces = np.arange(15, dtype=np.int64).reshape(5, 3)
    targets = np.array([
        [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0],
        [8.0, -2.0, 0.0], [12.0, -2.0, 0.0], [10.0, 2.0, 0.0],
    ], dtype="<f4")
    target_faces = {
        (51004, 1): np.array([[0, 1, 2]], dtype=np.int64),
        (51005, 2): np.array([[3, 4, 5]], dtype=np.int64),
    }
    baseline = _target_intersection_audit(_exact_surface_records(base, faces), target_faces, targets)
    candidate = base.copy()
    candidate[6:9] = _vertical_triangle(0.0)
    candidate[12:15] = _vertical_triangle(0.0, 2.0)
    return base, candidate, faces, target_faces, targets, baseline


def _incremental(base, candidate, faces, target_faces, targets, audits, validated_cache=None):
    return audit_incremental_skin_target_intersections(
        baseline_world_positions=base,
        candidate_world_positions=candidate,
        baseline_faces=faces,
        candidate_faces=faces,
        baseline_target_audits=audits,
        target_faces_by_key=target_faces,
        target_positions=targets,
        expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
        expected_face_index_sha256=_face_index_sha256(faces),
        expected_target_geometry_f32_sha256=_target_geometry_f32_sha256(targets, target_faces),
        expected_baseline_target_audits_sha256=_baseline_target_pair_table_sha256(audits),
        _validated_target_geometry_cache=validated_cache,
    )


def _pairs(audit):
    return {tuple(pair) for pair in audit["triangle_pairs"]}


def test_incremental_exact_pairs_equal_full_scan_with_sparse_changed_rows():
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    result = _incremental(base, candidate, faces, target_faces, targets, baseline)
    full = _target_intersection_audit(_exact_surface_records(candidate.astype("<f4"), faces), target_faces, targets)

    assert result["changed_skin_face_rows"] == [2, 4]
    assert result["unchanged_skin_face_count"] == 3
    assert result["reused_baseline_exact_pair_count"] == 2
    assert result["fresh_changed_face_exact_pair_count"] == 1
    assert result["fresh_changed_face_aabb_candidate_pairs"] >= 1
    assert result["target_audits"].keys() == full.keys()
    for key in full:
        assert _pairs(result["target_audits"][key]) == _pairs(full[key])
        assert result["target_audits"][key]["count"] == full[key]["count"]
        assert result["target_audits"][key]["aabb_candidate_pairs"] == result["target_audits"][key]["fresh_changed_face_aabb_candidate_pairs"]
        assert result["target_audits"][key]["aabb_work_scope"] == "changed_skin_faces_only"
    assert result["pair_changes_by_target"]["51004:1"] == {
        "added": [[2, 0]], "removed": [[4, 0]], "unchanged": [[0, 0]]
    }
    assert result["pair_changes_by_target"]["51005:2"] == {
        "added": [], "removed": [], "unchanged": [[1, 0]]
    }


def test_float32_quantization_reuses_all_faces():
    base, _, faces, target_faces, targets, baseline = _fixture()
    candidate = base.astype(np.float64)
    candidate[0, 0] += 1.0e-12
    assert _float32_xyz_sha256(candidate) == _float32_xyz_sha256(base)

    result = _incremental(base, candidate, faces, target_faces, targets, baseline)
    full = _target_intersection_audit(_exact_surface_records(candidate.astype("<f4"), faces), target_faces, targets)

    assert result["changed_skin_face_rows"] == []
    assert result["fresh_changed_face_exact_pair_count"] == 0
    assert result["fresh_changed_face_aabb_candidate_pairs"] == 0
    for key in full:
        assert _pairs(result["target_audits"][key]) == _pairs(full[key])


def test_float32_bit_comparison_detects_signed_zero_change():
    base, _, faces, target_faces, targets, baseline = _fixture()
    candidate = base.copy()
    zero_index = tuple(np.argwhere((base == 0.0) & ~np.signbit(base))[0])
    candidate[zero_index] = -0.0
    assert base[zero_index] == candidate[zero_index]
    assert base[zero_index].tobytes() != candidate[zero_index].tobytes()

    result = _incremental(base, candidate, faces, target_faces, targets, baseline)
    assert 0 in result["changed_skin_face_rows"]


def test_incremental_audit_fails_closed_on_baseline_identity_or_topology_mismatch():
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    kwargs = dict(
        baseline_world_positions=base,
        candidate_world_positions=candidate,
        baseline_faces=faces,
        candidate_faces=faces,
        baseline_target_audits=baseline,
        target_faces_by_key=target_faces,
        target_positions=targets,
        expected_baseline_target_audits_sha256=_baseline_target_pair_table_sha256(baseline),
    )
    with pytest.raises(ImportError, match="baseline world identity"):
        audit_incremental_skin_target_intersections(
            **kwargs, expected_baseline_world_f32_sha256="0" * 64,
            expected_face_index_sha256=_face_index_sha256(faces),
            expected_target_geometry_f32_sha256=_target_geometry_f32_sha256(targets, target_faces),
        )

    changed_faces = faces.copy()
    changed_faces[0] = changed_faces[0, ::-1]
    with pytest.raises(ImportError, match="topology or row identity"):
        audit_incremental_skin_target_intersections(
            **{**kwargs, "candidate_faces": changed_faces},
            expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
            expected_face_index_sha256=_face_index_sha256(faces),
            expected_target_geometry_f32_sha256=_target_geometry_f32_sha256(targets, target_faces),
        )

    with pytest.raises(ImportError, match="baseline face identity"):
        audit_incremental_skin_target_intersections(
            **kwargs,
            expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
            expected_face_index_sha256="0" * 64,
            expected_target_geometry_f32_sha256=_target_geometry_f32_sha256(targets, target_faces),
        )
    with pytest.raises(ImportError, match="target geometry does not match"):
        audit_incremental_skin_target_intersections(
            **kwargs,
            expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
            expected_face_index_sha256=_face_index_sha256(faces),
            expected_target_geometry_f32_sha256="0" * 64,
        )


def test_incremental_audit_rejects_missing_target_coverage_and_invalid_baseline_rows():
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    with pytest.raises(ImportError, match="baseline target coverage"):
        _incremental(base, candidate, faces, {next(iter(target_faces)): next(iter(target_faces.values()))},
                     targets, baseline)

    invalid = {key: dict(row) for key, row in baseline.items()}
    key = next(iter(invalid))
    invalid[key] = {"triangle_pairs": [[999, 0]], "count": 1}
    with pytest.raises(ImportError, match="outside coverage"):
        _incremental(base, candidate, faces, target_faces, targets, invalid)


def test_incremental_audit_rejects_altered_well_formed_pair_table_with_old_pin():
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    key = next(key for key, row in baseline.items() if row["triangle_pairs"])
    stale_pin = _baseline_target_pair_table_sha256(baseline)
    altered = {target: dict(row) for target, row in baseline.items()}
    pairs = [list(pair) for pair in altered[key]["triangle_pairs"]]
    original = pairs[0]
    replacement_face_row = (original[0] + 1) % len(faces)
    replacement = [replacement_face_row, original[1]]
    if replacement in pairs or replacement == original:
        replacement = [(replacement_face_row + 1) % len(faces), original[1]]
    pairs[0] = replacement
    altered[key]["triangle_pairs"] = pairs
    assert altered[key]["count"] == len(pairs)
    assert len({tuple(pair) for pair in pairs}) == len(pairs)
    with pytest.raises(ImportError, match="exact-pair table does not match"):
        audit_incremental_skin_target_intersections(
            baseline_world_positions=base,
            candidate_world_positions=candidate,
            baseline_faces=faces,
            candidate_faces=faces,
            baseline_target_audits=altered,
            target_faces_by_key=target_faces,
            target_positions=targets,
            expected_baseline_world_f32_sha256=_float32_xyz_sha256(base),
            expected_face_index_sha256=_face_index_sha256(faces),
            expected_target_geometry_f32_sha256=_target_geometry_f32_sha256(targets, target_faces),
            expected_baseline_target_audits_sha256=stale_pin,
        )


def test_incremental_audit_rejects_degenerate_candidate_face():
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    candidate[6:9] = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    ], dtype="<f4")
    with pytest.raises(ImportError, match="exactly degenerate triangle"):
        _incremental(base, candidate, faces, target_faces, targets, baseline)


def test_fit_owned_target_validation_cache_preserves_closed_aabb_and_original_face_rows(monkeypatch):
    skin = np.array([
        [-0.2, -0.2, -1.0], [0.2, -0.2, -1.0], [0.0, 0.2, 1.0],
    ], dtype="<f4")
    skin_faces = np.array([[0, 1, 2]], dtype=np.int64)
    targets = np.array([
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
        [20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [20.0, 1.0, 0.0],
        [0.2, -0.2, -1.0], [0.2, 0.2, -1.0], [0.2, -0.2, 1.0],
    ], dtype="<f4")
    target_faces = {(51004, 1): np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)}
    skin_records = _exact_surface_records(skin, skin_faces)
    validation_cache = set()

    # The first full audit validates every target face exactly before allowing
    # the cache key to be reused by the later candidate broadphase.
    baseline = _target_intersection_audit(
        skin_records, target_faces, targets,
        _validated_target_geometry_cache=validation_cache,
    )
    assert len(validation_cache) == 1
    assert baseline["51004:1"]["triangle_pairs"] == [[0, 2]]
    assert baseline["51004:1"]["aabb_candidate_pairs"] == 1

    equivalent_f64_targets = targets.astype(np.float64)
    assert _target_geometry_f32_sha256(equivalent_f64_targets, target_faces) == _target_geometry_f32_sha256(targets, target_faces)
    non_float32_targets = equivalent_f64_targets.copy()
    non_float32_targets[6, 0] += 1.0e-10
    assert _target_geometry_f32_sha256(non_float32_targets, target_faces) == _target_geometry_f32_sha256(targets, target_faces)
    with pytest.raises(ImportError, match='not exact Float32'):
        _target_intersection_audit(
            skin_records, target_faces, non_float32_targets,
            _validated_target_geometry_cache=validation_cache,
        )

    observed_exact_record_rows = []
    original_exact = c._exact_surface_records

    def record_rows(vertices, faces):
        observed_exact_record_rows.append(len(faces))
        return original_exact(vertices, faces)

    monkeypatch.setattr(c, "_exact_surface_records", record_rows)
    fast = _target_intersection_audit(
        skin_records, target_faces, equivalent_f64_targets,
        _validated_target_geometry_cache=validation_cache,
    )
    full = _target_intersection_audit(skin_records, target_faces, equivalent_f64_targets)
    assert fast == full
    assert fast["51004:1"]["triangle_pairs"] == [[0, 2]]
    assert fast["51004:1"]["count"] == 1
    assert fast["51004:1"]["aabb_candidate_pairs"] == 1
    # Cache-hit broadphase exact-converts only original target row 2. The
    # uncached reference that follows converts all three rows.
    assert observed_exact_record_rows == [1, 3]


def test_target_validation_cache_miss_still_rejects_far_degenerate_face():
    skin = np.array([
        [-0.2, -0.2, -1.0], [0.2, -0.2, -1.0], [0.0, 0.2, 1.0],
    ], dtype="<f4")
    skin_records = _exact_surface_records(skin, np.array([[0, 1, 2]], dtype=np.int64))
    targets = np.array([
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
        [20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [20.0, 1.0, 0.0],
        [0.2, -0.2, -1.0], [0.2, 0.2, -1.0], [0.2, -0.2, 1.0],
    ], dtype="<f4")
    target_faces = {(51004, 1): np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)}
    cache = set()
    _target_intersection_audit(skin_records, target_faces, targets, _validated_target_geometry_cache=cache)
    changed_targets = targets.copy()
    changed_targets[0:3] = np.array([[30.0, 0.0, 0.0], [31.0, 0.0, 0.0], [32.0, 0.0, 0.0]], dtype="<f4")
    with pytest.raises(ImportError, match="exactly degenerate triangle"):
        _target_intersection_audit(
            skin_records, target_faces, changed_targets,
            _validated_target_geometry_cache=cache,
        )



def test_incremental_fit_owned_validation_cache_matches_legacy_audit(monkeypatch):
    base, candidate, faces, target_faces, targets, baseline = _fixture()
    cache = set()
    observed = []
    original_exact = c._exact_surface_records

    def record_rows(vertices, rows):
        observed.append(int(len(rows)))
        return original_exact(vertices, rows)

    monkeypatch.setattr(c, "_exact_surface_records", record_rows)
    first = _incremental(base, candidate, faces, target_faces, targets, baseline, cache)
    first_calls = list(observed)
    assert len(cache) == 1
    observed.clear()
    second = _incremental(base, candidate, faces, target_faces, targets, baseline, cache)
    second_calls = list(observed)
    legacy = _incremental(base, candidate, faces, target_faces, targets, baseline)

    assert first == second == legacy
    # Calls 0/1 validate the full baseline and candidate skin; call 2 builds
    # changed skin records. The first target digest miss exact-validates both
    # target rows; the second cached call only converts the overlapping row.
    assert first_calls[:3] == [5, 5, 2]
    assert sum(first_calls[3:]) == 2
    assert second_calls[:3] == [5, 5, 2]
    assert sum(second_calls[3:]) == 1

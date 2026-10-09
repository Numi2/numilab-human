import numpy as np
import pytest

from numilab_human.skin_surface_binding import reweight_registered_bone_proximity


def _field():
    # Owners: torso, pelvis, right femur, left femur, unrelated.
    weights = np.asarray([
        [0.10, 0.30, 0.45, 0.05, 0.10],  # right thigh, strong support
        [0.10, 0.30, 0.45, 0.05, 0.10],  # right transition
        [0.10, 0.30, 0.45, 0.05, 0.10],  # protected support point
        [0.10, 0.30, 0.05, 0.45, 0.10],  # left thigh
        [0.10, 0.30, 0.05, 0.45, 0.10],  # no proximity support
    ], dtype=np.float64)
    distances = {
        128: np.asarray([0.14, 0.10, 0.14, 0.15, 0.04]),
        131: np.asarray([0.04, 0.07, 0.04, 0.20, 0.30]),
        145: np.asarray([0.20, 0.16, 0.20, 0.05, 0.30]),
    }
    return weights, distances


def test_reweight_is_compact_smooth_partition_preserving_other_rows_and_torso():
    weights, distances = _field()
    candidate, report = reweight_registered_bone_proximity(
        weights, [20, 128, 131, 145, 77], distances,
        pelvis_body_id=128, femur_body_ids=(131, 145),
        minimum_femur_distance_m=0.02, maximum_femur_distance_m=0.10,
        pelvis_to_femur_ratio_start=1.25, pelvis_to_femur_ratio_full=2.5,
        transfer_fraction=0.75, protected_vertex_ids=(4,),
    )
    assert not np.shares_memory(candidate, weights)
    assert report["changed_vertex_count"] == 4
    assert report["total_transferred_weight"] > 0
    assert candidate[0, 1] < weights[0, 1] and candidate[0, 2] > weights[0, 2]
    assert candidate[3, 1] < weights[3, 1] and candidate[3, 3] > weights[3, 3]
    assert np.array_equal(candidate[4], weights[4])
    assert np.array_equal(candidate[:, 0], weights[:, 0])
    assert np.array_equal(candidate[:, 4], weights[:, 4])
    assert np.all(candidate >= 0.0)
    assert np.max(np.abs(candidate.sum(axis=1) - 1.0)) < 1.0e-12
    assert candidate[1, 2] - weights[1, 2] < candidate[0, 2] - weights[0, 2]


def test_reweight_rejects_malformed_distances_and_owner_ids():
    weights, distances = _field()
    bad = dict(distances)
    bad[128] = np.asarray([0.1])
    with pytest.raises(ValueError, match="distances"):
        reweight_registered_bone_proximity(
            weights, [20, 128, 131, 145, 77], bad,
            pelvis_body_id=128, femur_body_ids=(131, 145),
        )
    with pytest.raises(ValueError, match="unique"):
        reweight_registered_bone_proximity(
            weights, [20, 128, 131, 131, 77], distances,
            pelvis_body_id=128, femur_body_ids=(131, 145),
        )



def test_reweight_rejects_pelvis_as_femur_and_active_protected_vertices():
    weights, distances = _field()
    with pytest.raises(ValueError, match="separate from pelvis"):
        reweight_registered_bone_proximity(
            weights, [20, 128, 131, 145, 77], distances,
            pelvis_body_id=128, femur_body_ids=(128, 145),
        )
    with pytest.raises(ValueError, match="overlap active thigh support"):
        reweight_registered_bone_proximity(
            weights, [20, 128, 131, 145, 77], distances,
            pelvis_body_id=128, femur_body_ids=(131, 145),
            minimum_femur_distance_m=0.02, maximum_femur_distance_m=0.10,
            pelvis_to_femur_ratio_start=1.25, pelvis_to_femur_ratio_full=2.5,
            protected_vertex_ids=(0,),
        )


def test_reweight_bilateral_smooth_union_is_symmetric_and_bounded():
    # The first row is the right-side reflection of the second. Both femurs
    # receive the same mid-ramp support, so the smooth union is 1-(1-.5)^2=.75.
    weights = np.asarray([
        [0.10, 0.30, 0.30, 0.20, 0.10],
        [0.10, 0.30, 0.20, 0.30, 0.10],
    ], dtype=np.float64)
    distances = {
        128: np.asarray([0.20, 0.20]),
        131: np.asarray([0.06, 0.30]),
        145: np.asarray([0.30, 0.06]),
    }
    candidate, report = reweight_registered_bone_proximity(
        weights, [20, 128, 131, 145, 77], distances,
        pelvis_body_id=128, femur_body_ids=(131, 145),
        minimum_femur_distance_m=0.02, maximum_femur_distance_m=0.10,
        pelvis_to_femur_ratio_start=1.5, pelvis_to_femur_ratio_full=3.0,
        femur_weight_start=0.02, femur_weight_full=0.12,
        pelvis_weight_start=0.02, pelvis_weight_full=0.10,
        transfer_fraction=1.0,
    )
    # Only the near-side support is active for each reflected row in this
    # fixture, and the smooth mid-distance gate gives 0.5 transfer support.
    expected_transfer = 0.30 * 0.5
    assert candidate[0, 1] == pytest.approx(0.30 - expected_transfer)
    assert candidate[1, 1] == pytest.approx(0.30 - expected_transfer)
    assert candidate[0, 2] - weights[0, 2] == pytest.approx(candidate[1, 3] - weights[1, 3])
    assert candidate[0, 3] - weights[0, 3] == pytest.approx(candidate[1, 2] - weights[1, 2])
    assert np.array_equal(candidate[:, 0], weights[:, 0])
    assert np.array_equal(candidate[:, 4], weights[:, 4])
    assert np.all(candidate >= 0.0)
    assert np.max(np.abs(candidate.sum(axis=1) - 1.0)) < 1.0e-12
    assert report["bilateral_support_combination"].startswith("smooth bounded union")


def test_reweight_ramp_endpoints_and_duplicate_source_rows_are_exact():
    # Rows 0 and 1 represent duplicate-coordinate source vertices: identical
    # input rows and distances must stay exactly identical after reweighting.
    weights = np.asarray([
        [0.10, 0.30, 0.40, 0.00, 0.20],
        [0.10, 0.30, 0.40, 0.00, 0.20],
        [0.10, 0.30, 0.40, 0.00, 0.20],
        [0.10, 0.02, 0.40, 0.00, 0.48],
        [0.10, 0.30, 0.02, 0.00, 0.58],
        [0.10, 0.30, 0.40, 0.00, 0.20],
    ], dtype=np.float64)
    distances = {
        128: np.asarray([0.20, 0.20, 0.20, 0.20, 0.20, 0.06]),
        131: np.asarray([0.02, 0.02, 0.10, 0.02, 0.02, 0.04]),
        145: np.asarray([0.30, 0.30, 0.30, 0.30, 0.30, 0.30]),
    }
    candidate, report = reweight_registered_bone_proximity(
        weights, [20, 128, 131, 145, 77], distances,
        pelvis_body_id=128, femur_body_ids=(131, 145),
        minimum_femur_distance_m=0.02, maximum_femur_distance_m=0.10,
        pelvis_to_femur_ratio_start=1.5, pelvis_to_femur_ratio_full=3.0,
        femur_weight_start=0.02, femur_weight_full=0.12,
        pelvis_weight_start=0.02, pelvis_weight_full=0.10,
        transfer_fraction=1.0,
    )
    assert np.array_equal(candidate[0], candidate[1])
    assert candidate[0, 1] == pytest.approx(0.0)  # full-support endpoint
    assert np.array_equal(candidate[2], weights[2])  # zero-support endpoint
    assert np.array_equal(candidate[3], weights[3])  # pelvis-weight ramp start
    assert np.array_equal(candidate[4], weights[4])  # femur-weight ramp start
    assert np.array_equal(candidate[5], weights[5])  # ratio-ramp start
    assert report["changed_vertex_count"] == 2

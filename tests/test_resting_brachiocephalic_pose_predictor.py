import numpy as np
import pytest

from numilab_human.resting_brachiocephalic_pose_predictor import (
    predict_brachiocephalic_world_positions,
)


def _ancestry(*, ids=(0, 1, 2), weights=(0.2, 0.3, 0.5), scale=1.0,
              exact=False, copied=None, distal=False):
    return {
        "svc_local_vertices": list(ids),
        "barycentric_weights": list(weights),
        "coefficient_scale": scale,
        "exact_source_coordinate_copy": exact,
        "copied_svc_local_vertex": copied,
        "distal_zero_field_vertex": distal,
    }


def test_projects_captured_svc_motion_with_fade_and_exact_source_copy():
    svc_rigid = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    svc_native = svc_rigid + np.asarray([[0, 0, 1], [0, 2, 0], [3, 0, 0]], dtype="<f4")
    bridge = np.asarray([[1, 0, 0], [4, 5, 6], [7, 8, 9]], dtype="<f4")
    ancestry = [
        _ancestry(ids=(0, 1, 2), weights=(0, 1, 0), scale=1.0,
                  exact=True, copied=1),
        _ancestry(weights=(0.2, 0.3, 0.5), scale=0.5),
        _ancestry(weights=(0.2, 0.3, 0.5), scale=0.0, distal=True),
    ]

    result = predict_brachiocephalic_world_positions(
        bridge, svc_native, svc_rigid, ancestry)
    positions = result["positions_world_m"]

    np.testing.assert_array_equal(positions[0], svc_native[1])
    expected_delta = 0.5 * (
        0.2 * (svc_native[0].astype(np.float64) - svc_rigid[0])
        + 0.3 * (svc_native[1].astype(np.float64) - svc_rigid[1])
        + 0.5 * (svc_native[2].astype(np.float64) - svc_rigid[2]))
    np.testing.assert_allclose(positions[1], bridge[1] + expected_delta, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(positions[2], bridge[2])
    assert result["status"] == "OFFLINE_CAPTURED_SVC_DISPLACEMENT_PREDICTION_ONLY"
    assert result["native_confirmation"] == "required"
    assert result["native_field_error_bound"] == "not_available"
    assert result["float32_interpolation_roundoff_bound_max_m"] >= result["float32_interpolation_roundoff_observed_max_m"]


def test_rejects_non_float32_or_incompatible_pose_surfaces():
    svc = np.zeros((3, 3), dtype="<f4")
    ancestry = [_ancestry()] * 1
    with pytest.raises(ValueError, match="little-endian Float32"):
        predict_brachiocephalic_world_positions(
            np.zeros((1, 3), dtype=np.float64), svc, svc, ancestry)
    with pytest.raises(ValueError, match="identical nonempty shape"):
        predict_brachiocephalic_world_positions(
            np.zeros((1, 3), dtype="<f4"), svc, svc[:2], ancestry)


@pytest.mark.parametrize("record", [
    _ancestry(ids=(0, 1, 3)),
    _ancestry(weights=(-0.1, 0.4, 0.7)),
    _ancestry(weights=(0.1, 0.1, 0.1)),
    _ancestry(scale=1.1),
    _ancestry(scale=0.2, exact=True, copied=0),
    _ancestry(scale=0.1, distal=True),
])
def test_rejects_unbound_or_inconsistent_ancestry(record):
    svc = np.zeros((3, 3), dtype="<f4")
    with pytest.raises(ValueError):
        predict_brachiocephalic_world_positions(
            np.zeros((1, 3), dtype="<f4"), svc, svc, [record])


def test_does_not_mutate_captured_or_source_inputs():
    svc_rigid = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    svc_native = svc_rigid + np.asarray([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]], dtype="<f4")
    bridge = np.asarray([[0.25, 0.25, 0.25]], dtype="<f4")
    before = tuple(x.tobytes() for x in (bridge, svc_native, svc_rigid))
    predict_brachiocephalic_world_positions(
        bridge, svc_native, svc_rigid, [_ancestry()])
    assert tuple(x.tobytes() for x in (bridge, svc_native, svc_rigid)) == before


def test_zero_fade_canonicalizes_negative_captured_residual_to_positive_zero():
    svc_rigid = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    svc_native = svc_rigid + np.asarray([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], dtype="<f4")
    bridge = np.asarray([[2, 3, 4]], dtype="<f4")
    ancestry = [_ancestry(scale=0.0, distal=True)]
    result = predict_brachiocephalic_world_positions(bridge, svc_native, svc_rigid, ancestry)
    assert np.all(result["projected_bridge_displacement_world_m"].view("<u4") == 0)
    np.testing.assert_array_equal(result["positions_world_m"], bridge)

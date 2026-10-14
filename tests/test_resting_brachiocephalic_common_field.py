import numpy as np
import pytest

from numilab_human.resting_brachiocephalic_common_field import (
    _compile_records,
    quintic_smoothstep,
)


def test_quintic_fade_has_zero_endpoint_derivatives_and_monotone_values():
    samples = np.linspace(0.0, 1.0, 101)
    values = quintic_smoothstep(samples, np)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) >= 0.0)
    assert np.isclose(values[50], 0.5)
    assert np.isclose(values[1], 6e-10 - 15e-8 + 10e-6)


def test_attachment_interpolates_svc_then_fades_to_existing_zero_field():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    svc_faces = np.array([[0, 1, 2]], dtype=np.int64)
    base = np.zeros((3, 7, 4), dtype="<f4")
    base[:, :, :3] = 0.25
    bridge_xyz = np.array([[0, 0, 0], [0.25, 0.25, 0.1], [0.5, 0.5, 0]],
                          dtype=np.float32)
    d_svc = np.array([0.0, 0.5, 1.0])
    d_distal = np.array([1.0, 0.5, 0.0])

    records, ancestry = _compile_records(
        bridge_xyz, d_svc, d_distal, svc_xyz, svc_faces, base, 1.0, np)

    assert records.shape == (3, 28)
    assert records[0].tobytes() == base[0].tobytes()
    assert np.allclose(records[1].reshape(7, 4)[:, :3], 0.125)
    assert np.all(records[1].reshape(7, 4)[:, 3] == 0.0)
    assert np.array_equal(records[2], np.zeros(28, dtype="<f4"))
    assert ancestry[0]["exact_source_coordinate_copy"] is True
    assert ancestry[0]["coefficient_scale"] == 1.0
    assert ancestry[1]["coefficient_scale"] == 0.5
    assert ancestry[2]["distal_zero_field_vertex"] is True


def test_projection_fails_closed_outside_route_derived_bound():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    svc_faces = np.array([[0, 1, 2]], dtype=np.int64)
    coefficients = np.ones((3, 7, 4), dtype="<f4")
    with pytest.raises(ValueError, match="exceeds fixed source-route bound"):
        _compile_records(
            np.array([[0.2, 0.2, 0.1]], dtype=np.float32),
            np.array([0.5]), np.array([0.5]),
            svc_xyz, svc_faces, coefficients, 0.05, np)


def test_overlapping_exact_svc_coordinates_must_have_identical_map_records():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                       dtype=np.float32)
    svc_faces = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
    coefficients = np.zeros((4, 7, 4), dtype="<f4")
    coefficients[3, 0, 0] = 1.0
    with pytest.raises(ValueError, match="duplicate SVC coordinates have unequal coefficients"):
        _compile_records(
            np.array([[0, 0, 0]], dtype=np.float32),
            np.array([0.0]), np.array([1.0]),
            svc_xyz, svc_faces, coefficients, 1.0, np)


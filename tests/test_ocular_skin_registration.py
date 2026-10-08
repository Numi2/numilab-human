import numpy as np
import pytest

from numilab_human.ocular_skin_registration import inferred_boundary_projection_field


def _grid():
    points = np.array([
        [float(x), float(y), 0.0]
        for y in range(5)
        for x in range(5)
    ], dtype=np.float64)
    faces = []
    for y in range(4):
        for x in range(4):
            a = y * 5 + x
            b, c, d = a + 1, a + 5, a + 6
            faces.extend(((a, b, c), (b, d, c)))
    # A disconnected component verifies the compact support boundary.
    offset = len(points)
    points = np.vstack((points, [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0],
                                 [30.0, 1.0, 0.0]]))
    faces.append((offset, offset + 1, offset + 2))
    return points, np.asarray(faces, dtype=np.int64)


def test_projection_field_keeps_exact_ring_targets_and_outside_anchors():
    points, faces = _grid()
    before_points, before_faces = points.copy(), faces.copy()
    ring_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    demands = np.array([0.4, 0.8, 0.6, 0.0, 0.0], dtype=np.float64) / 1000.0
    delta, report = inferred_boundary_projection_field(
        points, faces,
        [{
            "name": "left-test",
            "vertex_ids": ring_ids,
            "center_world_m": [2.0, 2.0, -1.0],
            "radial_displacement_m": demands,
        }],
    )
    radial = points[ring_ids] - np.array([2.0, 2.0, -1.0])
    expected = radial / np.linalg.norm(radial, axis=1)[:, None] * demands[:, None]
    np.testing.assert_allclose(delta[ring_ids], expected, atol=1e-15, rtol=0.0)
    np.testing.assert_array_equal(delta[ring_ids[3:]], np.zeros((2, 3)))
    np.testing.assert_array_equal(delta[-3:], np.zeros((3, 3)))
    np.testing.assert_array_equal(points, before_points)
    np.testing.assert_array_equal(faces, before_faces)
    assert report["rings"]["left-test"]["positive_projection_count"] == 3
    assert report["rings"]["left-test"]["preserved_anchor_count"] == 2
    assert np.count_nonzero(np.linalg.norm(delta, axis=1)) > 3


def test_projection_field_rejects_shared_boundary_vertex():
    points, faces = _grid()
    ring = {
        "name": "one",
        "vertex_ids": [0, 1, 2],
        "center_world_m": [0.0, 0.0, -1.0],
        "radial_displacement_m": [0.001, 0.001, 0.001],
    }
    with pytest.raises(ValueError, match="reuse a boundary vertex"):
        inferred_boundary_projection_field(
            points, faces, [ring, {**ring, "name": "two"}],
        )

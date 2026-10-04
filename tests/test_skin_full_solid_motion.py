import numpy as np
import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.skin_full_solid_motion import harmonic_extend


def test_harmonic_extension_preserves_fixed_weights_and_solves_inner_vertex():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    fixed_ids = np.array([0, 1, 2])
    fixed_weights = np.array([[1, 0], [0, 1], [0, 1]], dtype=float)
    weights, metrics = harmonic_extend(vertices, faces, fixed_ids, fixed_weights)
    np.testing.assert_array_equal(weights[fixed_ids], fixed_weights)
    np.testing.assert_allclose(weights[3], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert metrics["graph_component_count"] == 1
    assert metrics["extended_inner_and_connector_vertex_count"] == 1


def test_harmonic_extension_fails_closed_on_disconnected_surface_graph():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [4, 0, 0], [5, 0, 0], [4, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    fixed_ids = np.array([0])
    fixed_weights = np.array([[1.0]])
    with pytest.raises(HumanImportError, match="not connected"):
        harmonic_extend(vertices, faces, fixed_ids, fixed_weights)

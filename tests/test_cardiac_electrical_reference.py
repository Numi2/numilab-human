"""Analytic finite-element checks for the passive cardiac reference owner."""

import numpy as np

from numilab_human.cardiac_electrical_reference import _gradients, _operator, _region


def test_nonorthogonal_tetrahedron_has_exact_linear_gradient_and_neumann_balance():
    points = np.array([[0., 0., 0.], [1., 0., 0.],
                       [1., 1., 0.], [0., 0., 1.]])
    cells = np.array([[0, 1, 2, 3]], dtype=np.uint32)
    gradients, volumes, closure = _gradients(points, cells)
    assert np.allclose(gradients[0], [[-1., 0., -1.],
                                          [1., -1., 0.],
                                          [0., 1., 0.],
                                          [0., 0., 1.]])
    assert np.allclose((points[1:]-points[0]) @ gradients[0, 1:].T,
                       np.eye(3))
    field = points[:, 0]
    residual, energy, grad = _operator(gradients, volumes, cells, 4, field)
    assert np.allclose(grad[0], [1., 0., 0.])
    assert np.isclose(volumes[0], 1/6)
    assert np.isclose(energy, 1/12)
    assert np.isclose(residual.sum(), 0)
    assert closure < 1e-14


def test_shared_geometric_nodes_keep_distinct_region_reference_states():
    points = np.array([[0., 0., 0.], [1., 0., 0.],
                       [0., 1., 0.], [0., 0., 1.]])
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint32)
    labels = np.array([1, 2], dtype=np.uint32)
    fibre = np.tile([1., 0., 0.], (2, 1))
    sheet = np.tile([0., 1., 0.], (2, 1))
    first_nodes, first = _region(1, points, cells, labels, fibre, sheet, 1, 1/6)
    second_nodes, second = _region(2, points, cells, labels, fibre, sheet, 1, 1/6)
    assert np.array_equal(first_nodes, second_nodes)
    assert first['electrical_dof_count'] == second['electrical_dof_count'] == 4
    assert first['bitwise_replay'] and second['bitwise_replay']
    assert first['relative_charge_error'] <= 1e-12
    assert second['relative_charge_error'] <= 1e-12

import numpy as np
import pytest
from scipy.sparse import diags
from numilab_human.skin_surface_binding import solve_source_weight_field


def chain():
    adjacency = diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5)).tocsr()
    confidence = np.array([2., 0., 0., 0., 3.])
    rhs = np.zeros((5, 2))
    rhs[0, 0], rhs[-1, 1] = confidence[0], confidence[-1]
    return adjacency, confidence, rhs


def test_dirichlet_anchor_ownership_and_linear_interpolation():
    adjacency, confidence, rhs = chain()
    weights, metrics = solve_source_weight_field(
        adjacency, confidence, rhs, seed_condition='dirichlet')
    expected = np.column_stack((np.linspace(1., 0., 5), np.linspace(0., 1., 5)))
    np.testing.assert_allclose(weights, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(weights[[0, 4]], [[1., 0.], [0., 1.]])
    assert metrics['maximum_fixed_seed_weight_error'] == 0
    assert metrics['relative_solve_residual'] < 1e-14


def test_projection_gap_owner_shares_are_preserved_at_shared_anchor():
    adjacency, confidence, rhs = chain()
    rhs[0] = [0.5, 1.5]
    weights, _ = solve_source_weight_field(
        adjacency, confidence, rhs, seed_condition='dirichlet')
    np.testing.assert_array_equal(weights[0], [0.25, 0.75])
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-15, rtol=0)


def test_screened_default_keeps_soft_anchors_and_original_equations():
    adjacency, confidence, rhs = chain()
    default, _ = solve_source_weight_field(adjacency, confidence, rhs)
    explicit, _ = solve_source_weight_field(
        adjacency, confidence, rhs, seed_condition='screened')
    np.testing.assert_array_equal(default, explicit)
    assert 0 < default[0, 0] < 1
    assert 0 < default[-1, 1] < 1
    operator = np.diag(np.asarray(adjacency.sum(axis=1)).ravel() + confidence) - adjacency.toarray()
    np.testing.assert_allclose(operator @ default, rhs, atol=1e-15, rtol=0)


@pytest.mark.parametrize('condition', ['unknown', 'Dirichlet'])
def test_unknown_seed_condition_is_rejected(condition):
    with pytest.raises(ValueError, match='unknown'):
        solve_source_weight_field(*chain(), seed_condition=condition)


def test_dirichlet_rejects_unanchored_and_disconnected_domains():
    adjacency, confidence, rhs = chain()
    with pytest.raises(ValueError, match='anchors'):
        solve_source_weight_field(adjacency, confidence * 0, rhs * 0,
                                  seed_condition='dirichlet')
    adjacency = adjacency.tolil()
    adjacency[1, 2] = adjacency[2, 1] = 0
    with pytest.raises(ValueError, match='connected'):
        solve_source_weight_field(adjacency.tocsr(), confidence, rhs,
                                  seed_condition='dirichlet')

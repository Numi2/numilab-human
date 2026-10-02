"""Exact nearest-vertex parity for source-surface registration."""

import numpy as np
import pytest

from numilab_human.upper_limb_registration import _nearest, _nearest_brute_force


def test_spatial_index_matches_original_exhaustive_nearest_results():
    rng = np.random.default_rng(20261002)
    source = rng.normal(size=(97, 3))
    target = rng.normal(size=(131, 3))

    indexed_indices, indexed_squared = _nearest(source, target, np)
    reference_indices, reference_squared = _nearest_brute_force(source, target, np)

    np.testing.assert_array_equal(indexed_indices, reference_indices)
    np.testing.assert_array_equal(indexed_squared, reference_squared)


def test_symmetric_and_duplicate_ties_keep_first_source_index():
    source = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.asarray([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    indices, squared = _nearest(source, target, np)

    # The origin is equidistant from target rows 0 and 1; np.argmin chose 0.
    # The second query hits duplicated coordinates and keeps their first row.
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(squared, [1.0, 1.0])


@pytest.mark.parametrize(
    "source,target",
    [
        (np.empty((0, 3)), np.ones((1, 3))),
        (np.ones((1, 3)), np.empty((0, 3))),
        (np.ones((1, 2)), np.ones((1, 3))),
        (np.asarray([[np.nan, 0.0, 0.0]]), np.ones((1, 3))),
    ],
)
def test_nearest_rejects_empty_malformed_and_nonfinite_surfaces(source, target):
    with pytest.raises(RuntimeError, match="nearest-neighbour|empty surface"):
        _nearest(source, target, np)

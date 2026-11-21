from unittest import TestCase, main

import numpy as np
from numpy.testing import assert_allclose

from exact_preload_segments import determinant_coefficients, first_volume_limit


class ExactSegmentTests(TestCase):
    def test_matrix_row_expansion(self):
        rng = np.random.default_rng(539)
        f = np.eye(3) + .1 * rng.normal(size=(38, 3, 3))
        delta = rng.normal(size=f.shape)
        actual = determinant_coefficients(f, delta)
        points = np.linspace(0., 1., 11)
        powers = points[:, None] ** np.arange(4)[None, :]
        expected = np.linalg.det(f + points[:, None, None, None] * delta)
        assert_allclose(powers @ actual.T, expected, rtol=1.e-12, atol=1.e-13)

    def test_interior_excursion_with_safe_endpoint(self):
        # The endpoint appears unchanged but the middle of the step expands.
        assert_allclose(first_volume_limit(np.array([[1., 12., -12., 0.]]), .35, 2.5),
                        (1. - np.sqrt(.5)) / 2., rtol=1.e-12)

    def test_earliest_of_two_limits(self):
        coefficients = np.array([[1., 0., 0., 2.], [.5, -.6, 0., 0.], [1., 0., 0., 0.]])
        assert_allclose(first_volume_limit(coefficients, .35, 2.5), .25, rtol=1.e-12)

    def test_no_limit_with_small_cubic_term(self):
        assert first_volume_limit(np.array([[1., 1.e-9, -1.e-7, 1.e-21]]), .35, 2.5) == 1.

    def test_false_endpoint_bound_overestimates_early_change(self):
        # J(1) implies a .35 step using linear extrapolation. The first true
        # crossing occurs later because the initial volume derivative is zero.
        assert_allclose(first_volume_limit(np.array([[1., 0., 0., 4.]]), .35, 2.5),
                        np.cbrt(.375), rtol=1.e-12)


if __name__ == "__main__":
    main()

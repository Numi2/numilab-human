import itertools

import numpy as np
import pytest

from numilab_human.zanatomy import _closest_point_on_triangle


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_obtuse_triangle_projection_stays_on_the_nearest_edge(order, scale):
    # This point lies beyond AC's endpoint but projects into BC. Treating the
    # infinite AC line as the edge underestimates clearance by a factor of five.
    offset = np.array([0.2, -0.1, 1.4])
    triangle = scale * np.array([[0., 0., 0.], [3., 0., 0.], [1., 1., 0.]]) + offset
    point = scale * np.array([1.4, 1.6, 0.]) + offset
    expected = scale * np.array([1.08, 0.96, 0.]) + offset
    closest, normal = _closest_point_on_triangle(
        point.tolist(), *triangle[list(order)].tolist()
    )
    np.testing.assert_allclose(closest, expected, rtol=0, atol=1e-12)
    assert np.linalg.norm(point - closest) == pytest.approx(
        scale * np.sqrt(0.512), rel=1e-10
    )
    assert np.linalg.norm(normal) == pytest.approx(1.0)

"""Check the oriented derivative of an overlap at a moving material vertex."""

import numpy as np
from manifold3d import Manifold

from fit_native_neighbor_solids import overlap, solid


def test_clipped_tetrahedron():
    points = np.vstack((np.zeros(3), np.eye(3)))
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    owner = Manifold.reserve_ids(1)
    # x >= 0.25 clips off a smaller tetrahedron with edge length 0.75.
    obstacle = Manifold.cube((2, 3, 3)).translate((0.25, -1, -1))
    empty = Manifold.cube().translate((3, 3, 3))

    def evaluate(x):
        return overlap(x, faces, solid(x, faces, owner), owner, [[obstacle], [empty]])

    values, gradients = evaluate(points)
    np.testing.assert_allclose(values, [0.75**3 / 6, 0], atol=1e-12)
    np.testing.assert_array_equal(gradients[1], np.zeros_like(points))
    for node, coordinate in np.ndindex(points.shape):
        perturbation = np.zeros_like(points)
        perturbation[node, coordinate] = 1e-5
        upper = evaluate(points + perturbation)[0]
        lower = evaluate(points - perturbation)[0]
        np.testing.assert_allclose(
            (upper - lower) / 2e-5, gradients[:, node, coordinate], atol=1e-9
        )


if __name__ == "__main__":
    test_clipped_tetrahedron()

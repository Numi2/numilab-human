from unittest import TestCase, main

import numpy as np
from numpy.testing import assert_allclose

from cavity import determinant, oriented_boundary
from map_cavity import overlap, transfer


class CavityTests(TestCase):
    def test_shared_face_contributes_zero_volume(self):
        pos = np.vstack((np.zeros(3), np.eye(3)))
        reflection = pos.copy()
        reflection[3, 2] *= -1.
        assert overlap(pos, reflection) == 0.
        assert_allclose(overlap(pos, pos), 1./6., rtol=1.e-12)

    def test_convex_cavity_flip_is_oriented_and_conservative(self):
        x = np.array(((0.,0.,0.), (1.,0.,0.), (0.,1.,0.), (.2,.3,1.), (.4,.1,-1.)))
        first = np.array([[0,1,2,3],[0,2,1,4]])
        second = np.array([[0,1,3,4],[1,2,3,4],[2,0,3,4]])
        negatives = determinant(x, second) < 0.
        second[negatives] = second[negatives][:, [0,2,1,3]]
        self.assertEqual(oriented_boundary(first), oriented_boundary(second))
        transition = transfer(x, first, second)
        assert_allclose(transition.sum(axis=1), 1., rtol=1.e-13)
        assert_allclose((determinant(x, first)/6.) @ transition, determinant(x, second)/6., rtol=1.e-12)
        self.assertTrue((transition > 0.).all())


if __name__ == "__main__":
    main()

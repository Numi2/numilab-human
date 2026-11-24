"""Regression for duplicate and partially occluded posterior tissue columns."""
import unittest
import numpy as np
from reconstruct_gluteal_fat import signed_area,visible_parts


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.uv=np.array([[0.,0.],[1.,0.],[0.,1.]])
        self.lower=np.column_stack((self.uv,np.zeros(3)))
        self.upper=np.column_stack((self.uv,np.full(3,3.)))

    def area(self, shields):
        out=visible_parts(self.uv,self.lower,self.upper,shields)
        return sum(abs(signed_area(part)) for part in out)

    def test_nearer_surface_blocks_whole_target(self):
        shield=self.upper.copy()
        shield[:,2]=1.
        self.assertAlmostEqual(self.area([shield]),0.)

    def test_crossing_depth_planes(self):
        shield=self.upper.copy()
        shield[:,2]=[2.,4.,2.]
        self.assertAlmostEqual(self.area([shield]),.125)
        self.assertAlmostEqual(self.area([shield,shield]),.125)

    def test_surface_outside_material_interval(self):
        for height in [-1.,4.]:
            shield=self.lower.copy()
            shield[:,2]=height
            self.assertAlmostEqual(self.area([shield]),.5)


if __name__=='__main__':unittest.main()

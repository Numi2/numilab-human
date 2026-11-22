import unittest
import numpy as np
from measure_bed_tissue_path import hits, closed, union_length

class BedTissuePathTest(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]],dtype=float)
        self.f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])

    def test_closed_cortical_and_muscle_ray(self):
        self.assertTrue(closed(self.v,self.f))
        r=hits(self.v,self.f,{0:np.array([.2,.3]),1:np.array([1.,1.]),2:np.array([0.,.3])},True)
        np.testing.assert_allclose(r[0],[0,.5]);self.assertNotIn(1,r);self.assertIsNone(r[2])

    def test_wrong_winding_and_open_banks(self):
        self.assertFalse(closed(self.v,self.f[:-1]))
        wrong=self.f.copy();wrong[0]=wrong[0,::-1]
        self.assertFalse(closed(self.v,wrong))
        self.assertIsNone(hits(self.v,wrong,{0:np.array([.2,.3])},True)[0])
        seam_v=np.concatenate([self.v,self.v[:1]])
        seam_f=self.f.copy();seam_f[0,0]=4
        self.assertTrue(closed(seam_v,seam_f))

    def test_partial_overlap_counted_once(self):
        self.assertAlmostEqual(union_length([(1.,2.),(.5,1.5),(4.,4.5)]),2.)
        self.assertEqual(union_length([]),0.)

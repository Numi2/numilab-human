import unittest
import numpy as np
from relax_bonded_interface import Patch
from screen_release_forces import element_blocks,summarize

class ReleaseForcesTest(unittest.TestCase):
    def test_force_and_full_diagonal_match_local_tangent(self):
        x=np.array([[0.,0.,0.],[2.,0.,0.],[0.,.8,0.],[0.,0.,1.1],[.2,.1,-.7]])
        cells=np.array([[0,1,2,3],[0,2,1,4]])
        prop=np.array([[6.,13.],[1.,4.]])
        y=x.copy();y[0]+=[.06,.02,-.05];y[1]+=[.1,.04,0.]
        g,H,J,vol=element_blocks(x[cells],y[cells],prop)
        patch=Patch(x,cells,np.arange(5),np.zeros(5),np.array([15,16]),prop)
        _,grad,j,dense=patch.evaluate(y,True)
        ref=np.zeros((5,3));diagonal=np.zeros((5,3,3))
        for k in range(4):
            np.add.at(ref,cells[:,k],g[:,k]);np.add.at(diagonal,cells[:,k],H[:,k])
        np.testing.assert_allclose(ref.ravel(),grad,atol=1e-13)
        np.testing.assert_allclose(diagonal,dense.reshape(5,3,5,3)[np.arange(5),:,np.arange(5),:],atol=1e-13)
        np.testing.assert_allclose(J,j,rtol=1e-12);np.testing.assert_allclose(vol,patch.volume,rtol=1e-12)
        np.testing.assert_allclose(ref.sum(0),np.zeros(3),atol=1e-13)

    def test_stable_witness_ranks(self):
        out=summarize(np.array([2.,.1,2.,.3]),np.array([2,1]))
        self.assertEqual(out['witness_rank'],[2,4])
        self.assertEqual(out['counts_at_or_above_one'],2)

import unittest
import numpy as np
from audit_cavity_response import elastic
from fit_connected_fat_material import gradient,response
from torso_cell_geometry import geom

class MaterialPatchTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,0.,.5],[2.,1.,.5]])
        self.t=np.array([[0,1,2,3],[1,4,2,3]])
        self.mu=2200.;self.lam=6100.

    def test_simultaneous_move_is_affine_in_signed_volume(self):
        nodes=np.array([0,2,3]);quantum=np.array([.4,.25,1.]);weight=np.array([.03,-.09,.015])
        volume=geom(self.x,self.t)[0]
        for axis in range(3):
            x=self.x.copy();x[nodes,axis]+=quantum*weight
            actual=geom(x,self.t)[0]-volume
            np.testing.assert_allclose(gradient(self.x,self.t,axis,nodes,quantum)@weight,actual,atol=3e-16)

    def test_translation_has_no_internal_work(self):
        E,F=response(self.x,self.x+np.array([2.,.125,-.25]),self.t,self.mu,self.lam)
        self.assertEqual(E,0.)
        np.testing.assert_array_equal(F,np.zeros_like(F))

    def test_load_and_energy_match_finite_strain_owner(self):
        rng=np.random.default_rng(12)
        y=self.x+rng.normal(scale=.012,size=self.x.shape)
        E,F=response(self.x,y,self.t,self.mu,self.lam)
        owner,force,_=elastic(self.x,y,self.t,np.arange(len(y)),self.mu,self.lam)
        np.testing.assert_allclose(E,owner,rtol=2e-12)
        np.testing.assert_allclose(F,force,rtol=2e-12,atol=1e-12)
        direction=rng.normal(size=y.shape);h=1e-6
        ep=response(self.x,y+h*direction,self.t,self.mu,self.lam)[0]
        em=response(self.x,y-h*direction,self.t,self.mu,self.lam)[0]
        np.testing.assert_allclose((ep-em)/(2*h),-np.sum(F*direction),rtol=1e-7)

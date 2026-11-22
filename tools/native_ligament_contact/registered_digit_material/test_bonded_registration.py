import unittest
import numpy as np
from relax_bonded_interface import Patch
from fit_deep_interface_clearance import inequalities,plane
from locate_tissue_components import contains
from audit_tissue_connectivity import components

class TestBondedPatch(unittest.TestCase):
    def test_energy_gradient_and_tangent(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-0.7]])
        cells=np.array([[0,1,2,3],[0,2,1,4]])
        rows=np.arange(3)
        p=Patch(x,cells,rows,np.array([.01,.02,.03]),np.array([15,16]),np.array([[6.,13.],[1.,4.]]))
        y=x.copy()
        y[0]=[.04,-.06,.05]
        y[2]+=[.2,.05,-.03]
        rng=np.random.default_rng(6231)
        direction=rng.normal(size=(3,3))
        dy=np.zeros_like(y)
        dy[rows]=direction
        eps=2e-6
        e1,g1,_,_=p.evaluate(y+eps*dy)
        e0,g0,_,_=p.evaluate(y-eps*dy)
        _,g,_,H=p.evaluate(y,True)
        self.assertAlmostEqual((e1-e0)/(2*eps),g@direction.ravel(),places=8)
        np.testing.assert_allclose((g1-g0)/(2*eps),H@direction.ravel(),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(H,H.T,atol=1e-14)
        swapped=y.copy()
        swapped[3,2]=-1.
        self.assertTrue(np.isinf(p.evaluate(swapped)[0]))

    def test_initial_volume_acceleration_gradient(self):
        from bonded_volume_acceleration import volume_rate
        x=np.array([[.0,.0,.0],[1.,.0,.0],[.0,1.,.0],[.0,.0,1.],[.0,.0,-.7]])
        cells=np.array([[0,1,2,3],[0,2,1,4]])
        mass=np.array([.8,.2,.3,.14,.1])
        p=Patch(x,cells,np.arange(5),mass,np.array([15,16]),np.array([[6.,13.],[1.,4.]]))
        y=x.copy();y[0]=[.03,-.08,.07];y[2]+=[.02,.04,.08]
        B=np.random.default_rng(4509).normal(size=(15,3))
        ref=6*p.volume
        def rate(q,jac=False):
            _,g,_,H=p.evaluate(q,jac)
            return volume_rate(q,-g.reshape(-1,3)/mass[:,None],cells,cells,ref,mass,B,H)
        eps=1.e-6
        numerical=np.stack([(rate(y+eps*v.reshape(5,3))-rate(y-eps*v.reshape(5,3)))/(2*eps) for v in B.T],axis=1)
        np.testing.assert_allclose(rate(y,True),numerical,rtol=3e-8,atol=3e-8)

    def test_exact_simplex_boundary(self):
        c=[(0,0,0),(7,0,0),(0,7,0),(0,0,7)]
        self.assertEqual(contains(c,(1,2,3)),2)
        self.assertEqual(contains(c,(1,2,4)),1)
        self.assertEqual(contains(c,(1,2,5)),0)

    def test_node_connectivity(self):
        t=np.array([[0,1,2,3],[3,4,5,6],[7,8,9,10]])
        labels=components(t,11)
        self.assertEqual(labels[0],labels[6])
        self.assertNotEqual(labels[0],labels[7])

    def test_separator_with_shared_vertex(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]])
        pairs=inequalities([0,1,2],[0,3,4])
        n=plane(x,pairs)
        self.assertGreater(min(n@(x[v]-x[u]) for u,v in pairs),0)
        with self.assertRaises(ValueError):
            inequalities([0,1,2],[0,1,3])

if __name__=='__main__':unittest.main()

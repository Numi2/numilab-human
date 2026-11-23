"""Compare the sparse port certificate with a complete 3D material solve."""
import unittest
import numpy as np
from scipy.linalg import eigh
from adaptive_contact import response
from local_contact_inverse import patch_stiffness, finite_difference

class ContactCertificate(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.],[3.,2.,2.]])
        self.t=np.array([[0,1,2,3],[1,2,3,4]])
        self.role=np.array([15,16])
        self.props=np.array([[30.,50.],[10.,20.]])
        self.m=np.array([1.,2.,1.,2.,1.,.025])
        self.fixed=np.array([False,False,False,True,False,False])
        self.ports=np.array([0,1])
    def test_bound_and_reciprocity(self):
        dt=.1
        rec,_,_,Cs,_,_=response(self.x,self.t,self.role,self.props,self.m,self.fixed,self.ports,self.ports,dt)
        K=patch_stiffness(self.x,self.t,self.props[self.role-15])
        self.assertLess(finite_difference(self.x,self.t,self.props[self.role-15],K),3e-6)
        A=K.toarray()+np.diag(np.repeat(self.m,3)/dt**2)
        free=np.flatnonzero(np.repeat(~self.fixed,3))
        ids=np.searchsorted(free,3*self.ports+2)
        truth=np.linalg.inv(A[np.ix_(free,free)])[np.ix_(ids,ids)]
        errors=eigh(truth-Cs,Cs,eigvals_only=True)
        self.assertGreater(errors.min(),-1e-12)
        self.assertGreater(errors.max(),0)
        self.assertLessEqual(errors.max(),rec['upper_relative_energy_error']+1e-12)
        active=np.array([0,1,2,4])
        full,_,_,Cd,_,_=response(self.x,self.t,self.role,self.props,self.m,self.fixed,self.ports,active,dt)
        np.testing.assert_allclose(Cd,truth,rtol=1e-12,atol=1e-14)
        self.assertLess(full['upper_relative_energy_error'],1e-24)
    def test_reference_tangent_rigid_motion(self):
        K=patch_stiffness(self.x,self.t,self.props[self.role-15])
        for axis in np.eye(3):
            u=np.cross(axis,self.x)+axis
            np.testing.assert_allclose(K@u.ravel(),0,atol=5e-13)
if __name__=='__main__':
    unittest.main()

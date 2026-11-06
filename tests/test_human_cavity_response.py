"""Reference-law tests used only for offline mesh preparation diagnostics."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location("response",Path(__file__).resolve().parents[1]/"tools/screen_human_cavity_response.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
P=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
T=np.array([[0,1,2,3]])
def test_identity_has_zero_response():
 f,e,v=m.response(P,T,np.zeros_like(P),2.,10.)
 np.testing.assert_array_equal(f,0);np.testing.assert_array_equal(e,0);np.testing.assert_allclose(v,1/6)
def test_energy_gradient_matches_internal_force():
 rng=np.random.default_rng(7);u=rng.normal(size=(4,3))*.02;f,e,v=m.response(P,T,u,2.,10.)
 for i in range(4):
  for j in range(3):
   du=np.zeros_like(P);du[i,j]=1e-6
   hi=m.response(P,T,u+du,2.,10.)[1].sum();lo=m.response(P,T,u-du,2.,10.)[1].sum()
   np.testing.assert_allclose((hi-lo)/2e-6,-f[0,i,j],rtol=2e-7,atol=1e-10)
def test_rigid_motion_and_affine_subdivision():
 theta=.4;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 f,e,_=m.response(P,T,P@R.T-P+[.2,-.1,.3],2.,10.)
 np.testing.assert_allclose(f,0,atol=1e-15);np.testing.assert_allclose(e,0,atol=1e-15)
 p=np.vstack([P,P.mean(axis=0)]);t=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
 u=p@np.array([[.01,.02,0],[0,-.01,0],[0,0,.03]]).T
 f0,e0,v0=m.response(p,T,u,2.,10.);f1,e1,v1=m.response(p,t,u,2.,10.)
 np.testing.assert_allclose(e0.sum(),e1.sum(),rtol=1e-12)
 g0=m.scatter(T,f0,5);g1=m.scatter(t,f1,5)
 np.testing.assert_allclose(g0,g1,atol=1e-14)
def test_inversion_is_rejected():
 u=np.zeros_like(P);u[3,2]=-2
 with pytest.raises(ValueError):m.response(P,T,u,2.,10.)

@pytest.mark.parametrize("signal,error",[(0.,0.),(-1.,0.),(float("nan"),0.),(1.,float("nan")),(1.,-1.)])
def test_unobservable_or_invalid_force_calibration_rejects(signal,error):
 with pytest.raises(ValueError):
  m.calibrated_force_error(dict(native_free_force_l1_N=signal,analytical_minus_native_free_force_l1_N=error))

def test_force_calibration_uses_measured_signal():
 assert m.calibrated_force_error(dict(native_free_force_l1_N=2.,analytical_minus_native_free_force_l1_N=.01))==.005

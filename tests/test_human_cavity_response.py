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


def test_response_budget_rejects_cumulative_force_and_preserves_state():
 b=m.RegionalResponseBudget({7:10.},{7:2.})
 p,f=b.propose({7:.06},7,.04);assert not f;b.commit(p)
 p,f=b.propose({7:.05},7,.01);assert p is None and f[0]["metric"]=="free_force_l1_bound_N"
 assert b.force_used[7]==.06 and b.energy_used[7]==.04

def test_response_budget_uses_absolute_energy_without_cancellation():
 b=m.RegionalResponseBudget({7:10.},{7:2.})
 p,f=b.propose({7:0.},7,.06);assert not f;b.commit(p)
 p,f=b.propose({7:0.},7,-.05);assert p is None and f[0]["metric"]=="energy_absolute_change_bound_J"

def test_response_budget_unexcited_region_only_allows_zero_change():
 b=m.RegionalResponseBudget({7:0.},{7:0.})
 p,f=b.propose({7:0.},7,0.);assert p is not None and not f
 assert b.propose({7:1e-30},7,0.)[0] is None
 assert b.propose({7:0.},7,1e-30)[0] is None

@pytest.mark.parametrize("change",[float("nan"),float("inf"),-1.])
def test_response_budget_rejects_invalid_force_change(change):
 b=m.RegionalResponseBudget({7:1.},{7:1.})
 with pytest.raises(ValueError):b.propose({7:change},7,0.)

def test_force_budget_groups_actual_node_owners_without_vector_cancellation():
 nodes=np.array([2,0,1]);delta=np.array([[3.,0,0],[-3.,0,0],[8.,0,0]])
 regions=np.array([9,7,9]);free=np.array([True,False,True])
 assert m.regional_force_change(nodes,delta,regions,free)=={9:6.}


def test_negative_reference_energy_is_not_normalized_into_a_valid_budget():
 with pytest.raises(ValueError):m.RegionalResponseBudget({1:1.},{1:-1.})


def test_force_budget_reference_excludes_prescribed_reaction_rows():
 force=np.array([[3.,4,0],[1000.,0,0],[0,0,2.]])
 reference=m.regional_force_change(np.arange(3),force,np.array([7,7,9]),np.array([True,False,False]))
 assert reference=={7:5.}
 budget=m.RegionalResponseBudget(reference,{7:1.})
 assert budget.propose({7:.051},7,0)[0] is None


def test_native_geometry_certificate_keeps_exact_failure_witnesses_and_rejects_bad_domains(tmp_path,monkeypatch):
 import json
 from types import SimpleNamespace
 checker=tmp_path/"checker";checker.write_bytes(b"pinned executable")
 proposal=tmp_path/"proposal.npz";proposal.write_bytes(b"pinned proposal")
 cert=dict(source_sha256="a"*64,proposal_sha256=m.sha(proposal),cavities=3,failures=[dict(cavity=1,reason="cavity boundary is not a closed sphere")],failure_count=1,native_steps=0,mass_qualified=False)
 monkeypatch.setattr(m.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stdout="skin_fat_cavity_geometry="+json.dumps(cert),stderr=""))
 record,rejected=m.native_geometry_check(checker,m.sha(checker),tmp_path/"source.npz",proposal,"a"*64,m.sha(proposal),3)
 assert rejected=={1} and record["certificate"]["failures"][0]["reason"]=="cavity boundary is not a closed sphere"
 cert["failures"][0]["cavity"]=3
 with pytest.raises(ValueError,match="domain"):m.native_geometry_check(checker,m.sha(checker),tmp_path/"source.npz",proposal,"a"*64,m.sha(proposal),3)


def test_native_geometry_checker_hash_is_mandatory_before_execution(tmp_path,monkeypatch):
 checker=tmp_path/"checker";checker.write_bytes(b"changed executable")
 monkeypatch.setattr(m.subprocess,"run",lambda *a,**k:pytest.fail("unverified checker executed"))
 with pytest.raises(ValueError,match="SHA mismatch"):
  m.native_geometry_check(checker,"f"*64,tmp_path/"source",tmp_path/"proposal","a"*64,"b"*64,1)

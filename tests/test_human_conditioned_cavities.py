"""Numerical geometry and native cavity-format regressions."""
from pathlib import Path
import importlib.util,itertools,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location("cavities",Path(__file__).resolve().parents[1]/"tools/build_human_conditioned_cavities.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_centered_condition_is_invariant():
 p=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
 t=np.array(list(itertools.permutations(range(4))))
 np.testing.assert_allclose(m.spectral_condition(p,t),1,rtol=1e-13)
 np.testing.assert_allclose(m.spectral_condition(p*3+[2,5,-8],t),1,rtol=1e-13)

def test_flat_shape_is_detected_despite_finite_positive_volume():
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[.3,.3,1e-7]])
 c=m.spectral_condition(p,[[0,1,2,3]])[0]
 assert c>1e6
 with pytest.raises(ValueError):m.spectral_condition(p*np.nan,[[0,1,2,3]])

def test_native_csr_repeats_the_complete_cavity_ancestry():
 patches=[dict(old_rows=[2,3,4],new_cells=[[0,1,2,4],[0,1,3,4]]),dict(old_rows=[7,8,9,10],new_cells=[[5,6,7,8]]*4)]
 z=m.cavity_payload("a"*64,12,patches)
 assert z["retained_input_cell_rows"].tolist()==[0,1,5,6,11]
 assert z["replacement_tet_offsets"].tolist()==[0,2,6]
 off=z["replacement_parent_offsets"];rows=z["replacement_parent_cell_rows"]
 for i in range(6):assert rows[off[i]:off[i+1]].tolist()==(patches[0]["old_rows"] if i<2 else patches[1]["old_rows"])

def test_overlapping_source_cavities_are_rejected():
 with pytest.raises(ValueError):m.cavity_payload("a"*64,5,[dict(old_rows=[1,2],new_cells=[[0,1,2,3]]),dict(old_rows=[2,3],new_cells=[[0,1,2,3]])])

def test_mass_proof_is_bound_to_mesh_and_prescribed_target():
 from copy import deepcopy
 parity=dict(status="PASS_INDEPENDENT_RECORDED_FLOW_TARGET_PARITY",target=dict(sha256="mass"),mass_receipt=dict(sha256="receipt"),prescribed_cells=4,maximum_relative_mass_difference=1.0000048050639287e-10)
 receipt=dict(status="PASS_SOURCE_ONLY_REPLAY",candidate=dict(child_tetrahedron_count=4,mass_transport_envelope=dict(input_sha256=dict(candidate="volume"))))
 m.validate_mass_proof(parity,receipt,"volume","mass","receipt",4)
 bad=deepcopy(receipt);bad["candidate"]["mass_transport_envelope"]["input_sha256"]["candidate"]="other-volume"
 with pytest.raises(ValueError):m.validate_mass_proof(parity,bad,"volume","mass","receipt",4)
 bad=deepcopy(parity);bad["mass_receipt"]["sha256"]="other-receipt"
 with pytest.raises(ValueError):m.validate_mass_proof(bad,receipt,"volume","mass","receipt",4)
 bad=deepcopy(parity);bad["maximum_relative_mass_difference"]=1e-9
 with pytest.raises(ValueError):m.validate_mass_proof(bad,receipt,"volume","mass","receipt",4)


def star_fixture():
 import rebuild_human_interface_caps as cap
 import simplify_human_skin_fat_boundary as mesh
 p=np.array([[0.,0,-1],[-1,0,0],[0,1,2**-16],[1,0,0],[0,-1,2**-16]],np.float32)
 t=cap.positive_cells(p,[[1,2,3,4],[0,1,2,3],[0,1,3,4]])
 return p,t,np.tile([16,23,23],(3,1)),mesh.incidence(t,len(p))

def test_star_cavity_removes_flat_cell_with_exact_boundary_and_no_new_coordinates():
 import simplify_human_skin_fat_boundary as mesh
 p,t,labels,inc=star_fixture();rows,new=m.grow_star_cavity(p,t,labels,0,0,inc)
 assert rows.tolist()==[0,1,2] and len(new)==2
 assert mesh.unchanged_cavity_boundary(t[rows],new)
 assert np.array_equal(np.unique(t[rows]),np.unique(new))
 assert mesh.measure(p,new)[0].min()>0
 assert m.spectral_condition(p,new).max()<m.spectral_condition(p,t).max()/1000

def test_star_growth_stops_at_material_region_or_capacity_boundary():
 p,t,labels,inc=star_fixture();labels[1,0]=15
 with pytest.raises(ValueError,match='material or region'):m.grow_star_cavity(p,t,labels,0,0,inc)
 p,t,labels,inc=star_fixture()
 with pytest.raises(ValueError,match='capacity'):m.grow_star_cavity(p,t,labels,0,0,inc,maximum_cells=1)
 with pytest.raises(ValueError,match='domain'):m.grow_star_cavity(p,t,labels,0,7,inc)


def test_native_parent_capacity_is_checked_before_emitting_a_payload():
 patch=dict(old_rows=list(range(32)),new_cells=[[0,1,2,3],[0,1,2,4]])
 out=m.cavity_payload("a"*64,40,[patch])
 assert np.diff(out["replacement_parent_offsets"]).tolist()==[32,32]
 patch["old_rows"].append(32)
 with pytest.raises(ValueError,match="ancestry capacity"):m.cavity_payload("a"*64,40,[patch])


def test_native_replacement_capacity_and_empty_cavities_fail_early():
 with pytest.raises(ValueError,match="count capacity"):m.cavity_payload("a"*64,4,[])
 for n in (1,65):
  with pytest.raises(ValueError,match="replacement cell capacity"):
   m.cavity_payload("a"*64,4,[dict(old_rows=[0,1],new_cells=[[0,1,2,3]]*n)])


def test_star_anchor_search_budget_is_explicit_and_bounded():
 p,t,labels,inc=star_fixture();surface=np.ones(len(p),bool)
 assert m.star_anchors(p,t,labels,0,inc,surface,count=8)==[0]
 assert m.star_anchors(p,t,labels,0,inc,surface,count=32)==[0]
 for count in [0,33]:
  with pytest.raises(ValueError,match='anchor count'):m.star_anchors(p,t,labels,0,inc,surface,count=count)

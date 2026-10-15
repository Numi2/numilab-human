"""Exact contact preparation regressions, independent of native admission."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "tools" / "repair_resting_skin_contacts.py"
spec = importlib.util.spec_from_file_location("skin_contact_authoring", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

def test_meter_coordinates_preserve_dimensionless_contact_weights():
    triangle = [(F(0),F(0),F(0)), (F(1,1000),F(0),F(0)), (F(0),F(2,1000),F(0))]
    weights = tool.barycentric(triangle, (F(1,4000),F(1,1000),F(0)))
    np.testing.assert_array_equal(weights, [.25,.25,.5])
    displacement_m = np.array([[0.,0.,0.],[.004,0.,0.],[0.,0.,0.]])
    assert float(weights @ displacement_m @ np.array([1.,0.,0.])) == .001

def test_tiny_nonzero_face_uses_exact_dominant_axis():
    e = F(1,10**20)
    triangle = [(F(1),F(2),F(3)), (F(1),F(2)+e,F(3)), (F(1),F(2),F(3)+e)]
    weights = tool.barycentric(triangle, (F(1),F(2)+e/4,F(3)+e/4))
    np.testing.assert_array_equal(weights, [.5,.25,.25])

def test_contact_edge_has_exact_zero_weight():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(0),F(1),F(0))]
    np.testing.assert_array_equal(tool.barycentric(triangle,(F(1,4),F(3,4),F(0))),[0.,.25,.75])

def test_outside_contact_is_rejected():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(0),F(1),F(0))]
    with pytest.raises(ValueError, match="outside"):
        tool.barycentric(triangle,(F(1),F(1),F(0)))

def test_exact_degenerate_face_is_rejected():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(2),F(0),F(0))]
    with pytest.raises(ValueError, match="degenerate"):
        tool.barycentric(triangle,(F(1),F(0),F(0)))

def test_projected_area_gradient_matches_each_vertex_perturbation():
    triangle=np.array([[0.,0.,0.],[.001,0.,0.],[0.,.002,0.]])
    normal=np.array([0.,0.,1.])
    gradient=tool.projected_area_gradient(triangle,normal)
    original=float(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])@normal)
    for vertex in range(3):
        for axis in range(3):
            changed=triangle.copy()
            changed[vertex,axis]+=1e-5
            actual=float(np.cross(changed[1]-changed[0],changed[2]-changed[0])@normal)
            assert actual-original == pytest.approx(gradient[vertex,axis]*1e-5,abs=1e-21)
    np.testing.assert_allclose(gradient.sum(0),0.,atol=1e-18)

@pytest.mark.parametrize("direction,expected", [([1.,0.,0.],.7),([-1.,0.,0.],.1),([0.,1.,0.],.7),([0.,0.,-1.],.1)])
def test_exact_oriented_exit_handles_signed_and_zero_ray_components(direction,expected):
    positions=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype="<f4")
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int64)
    surface=tool.v.prepare_surface((51004,12),positions,faces,source_vertex_ids=np.arange(4,dtype=np.int64))
    distance,_=tool.exact_forward_exit(surface,surface.meter_index(),np.array([.1,.1,.1]),direction)
    assert distance == pytest.approx(expected,abs=1e-14)

def test_contact_corner_ray_can_leave_without_positive_hit():
    positions=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype="<f4")
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int64)
    surface=tool.v.prepare_surface((51004,12),positions,faces,source_vertex_ids=np.arange(4,dtype=np.int64))
    index=surface.meter_index()
    point=[F(0),F(1,4),F(3,4)]
    # Enters the x=0 half-space, immediately exits through x+y+z=1.
    assert tool.contact_forward_exit(surface,index,point,[1.,0.,1.]) == (0.,None)
    with pytest.raises(ValueError,match="non-boundary"):
        tool.contact_forward_exit(surface,index,[2.,2.,2.],[1.,0.,0.])

def test_self_contact_constraint_preserves_reference_side_in_meters():
    base=np.array([[0.,0.,0.],[0.,.01,0.],[0.,0.,.01],
                   [.02,0.,0.],[.02,.01,0.],[.02,0.,.01]])
    faces=np.array([[0,1,2],[3,4,5]])
    bary=np.array([.5,.25,.25])
    a,wa,b,wb,normal,bound=tool.self_gap_constraint(base,faces,0,bary,1,bary,.0001)
    np.testing.assert_array_equal(normal,[-1.,0.,0.])
    assert bound == pytest.approx(-.0199)
    # Initially separated reference faces satisfy the inequality; moving both
    # witness points to x=.01 produces the exact witnessed collision and fails.
    assert 0. >= bound
    da=np.array([.01,0.,0.]);db=np.array([-.01,0.,0.])
    assert (da-db)@normal < bound
    assert a==0 and b==1
    np.testing.assert_array_equal(wa,bary)
    np.testing.assert_array_equal(wb,bary)

def test_self_contact_without_reference_gap_is_not_given_a_separator():
    base=np.array([[0.,0.,0.],[0.,.01,0.],[0.,0.,.01]])
    faces=np.array([[0,1,2],[0,1,2]])
    with pytest.raises(ValueError,match="unambiguous separation"):
        tool.self_gap_constraint(base,faces,0,np.array([1.,0.,0.]),
                                 1,np.array([1.,0.,0.]),.0001)


def test_complete_face_support_rejects_collision_hidden_from_vertex_witness():
    base=np.array([[0.,0.,0.],[0.,.01,0.],[0.,0.,.01],
                   [.02,0.,0.],[.02,.01,0.],[.02,0.,.01]],dtype="<f4")
    faces=np.array([[0,1,2],[3,4,5]])
    rows=tool.face_support_constraints(base,faces,0,1,.0005)
    assert len(rows)==9
    displacement=np.zeros_like(base);displacement[1,0]=.03
    # A constraint at vertex0 sees no motion; the complete face constraint
    # catches vertex1 crossing the other triangle's plane.
    bary=np.array([1.,0.,0.])
    _,_,_,_,n,bound=tool.self_gap_constraint(base,faces,0,bary,1,bary,.0001)
    assert float((displacement[0]-displacement[3])@n)>=bound
    violated=[]
    for a,wa,b,wb,n,bound in rows:
        assert 0.>=bound
        violated.append(float((wa@displacement[faces[a]]-wb@displacement[faces[b]])@n)<bound)
    assert any(violated)

def test_complete_face_support_handles_coplanar_disjoint_triangles():
    base=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],
                   [.02,0.,0.],[.03,0.,0.],[.02,.01,0.]],dtype="<f4")
    rows=tool.face_support_constraints(base,np.array([[0,1,2],[3,4,5]]),0,1,.0005)
    assert len(rows)==9
    assert all(0.>=r[-1] for r in rows)
    assert all(abs(r[-2][2])<1e-15 for r in rows)

def test_complete_face_support_refuses_reference_intersection():
    base=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]],dtype="<f4")
    with pytest.raises(ValueError,match="complete separating"):
        tool.face_support_constraints(base,np.array([[0,1,2],[0,1,2]]),0,1,.0005)


def test_limb_bone_exit_uses_retained_material_branch_without_dropping_owners():
    owners=np.array([41,20,42,43],dtype=np.uint32)
    torso=np.array([.04,.9,.03,.03])
    arm=np.array([.35,.3,.25,.1])
    before=torso.copy()
    direction,record=tool.compartment_direction([-1.,0.,0.],torso,owners,20)
    np.testing.assert_array_equal(direction,[1.,0.,0.])
    assert record["region"]=="torso_skin_excluding_limb_bone"
    np.testing.assert_array_equal(torso,before)
    direction,record=tool.compartment_direction([-1.,0.,0.],arm,owners,20)
    np.testing.assert_array_equal(direction,[-1.,0.,0.])
    assert record["region"]=="arm_skin_enclosing_limb_bone"

def test_compartment_direction_must_resolve_unique_actual_owner():
    with pytest.raises(ValueError,match="absent"):
        tool.compartment_direction([1.,0.,0.],[.5,.5],np.array([41,42]),20)
    with pytest.raises(ValueError,match="invalid"):
        tool.compartment_direction([1.,0.,0.],[.9,.9],np.array([20,41]),20)

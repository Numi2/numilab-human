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


@pytest.mark.parametrize("faces",[
    np.array([[0,1,2],[0,3,4]]),
    np.array([[0,1,2],[0,2,3]])])
def test_complete_face_support_keeps_shared_simplex_without_positive_gap(faces):
    base=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],
                   [-.01,0.,0.],[0.,-.01,0.]],dtype="<f4")
    rows=tool.face_support_constraints(base,faces,0,1,.0005)
    assert len(rows)==9
    assert all(0.>=row[-1] for row in rows)
    assert any(row[-1]==0. for row in rows)
    shared=set(faces[0])&set(faces[1])
    for _,wa,_,wb,normal,bound in rows:
        i,j=int(np.argmax(wa)),int(np.argmax(wb))
        if faces[0,i]==faces[1,j]:
            assert faces[0,i] in shared and bound==0.
    # Folding an unshared first corner over the second face violates support.
    delta=np.zeros_like(base)
    delta[faces[0,1]]=-.1*rows[0][-2]
    assert any(float((wa@delta[faces[a]]-wb@delta[faces[b]])@normal)<bound
               for a,wa,b,wb,normal,bound in rows)

def test_unrelated_reference_touch_does_not_get_shared_simplex_permission():
    base=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],
                   [0.,0.,0.],[-.01,0.,0.],[0.,-.01,0.]],dtype="<f4")
    with pytest.raises(ValueError,match="complete separating"):
        tool.face_support_constraints(base,np.array([[0,1,2],[3,4,5]]),0,1,.0005)


def test_complete_face_support_preserves_non_axis_aligned_shared_edge_exactly():
    a=np.array([.1234567,.2345678,.3456789],dtype="<f4")
    b=(a.astype(float)+[.0037456,.0012345,.0065432]).astype("<f4")
    c=(a.astype(float)+[.00765,.00432,.00987]).astype("<f4")
    d=(a.astype(float)-[.00765,.00432,.00987]).astype("<f4")
    base=np.array([a,b,c,d],dtype="<f4")
    rows=tool.face_support_constraints(base,np.array([[0,1,2],[1,0,3]]),0,1,.0005)
    assert len(rows)==9 and all(row[-1]<=0 for row in rows)
    for _,wa,_,wb,axis,bound in rows:
        # Shared edge endpoints must remain exactly on a supporting plane;
        # geometric tolerance cannot stand in for that exact incidence.
        if int(np.argmax(wa))<2 and int(np.argmax(wb))<2:
            assert bound==0.


def test_current_contact_inventory_retires_old_separators_but_self_stage_keeps_them():
    old=[("old",)];new=[("current",)]
    assert tool.update_contact_constraints(old,new,self_failed=False,policy="current")==new
    assert tool.update_contact_constraints(old,[],self_failed=True,policy="current")==old
    assert tool.update_contact_constraints(old,new,self_failed=False,policy="accumulate")==old+new
    assert tool.update_contact_constraints(old,[],self_failed=False,policy="current")==[]
    with pytest.raises(ValueError,match="self-only"):
        tool.update_contact_constraints(old,new,self_failed=True,policy="current")


def test_local_source_update_retains_prior_repairs_and_unused_allocation():
    source=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]],dtype="<f4")
    prior=source.copy();prior[1]+=.25;prior[3]-=.5
    used=np.array([0,1,2]);active=np.array([2])
    affine=np.tile(np.eye(3),(3,1,1))
    request=np.array([[.125,.25,.5]])
    candidate=tool.source_points_with_local_displacement(prior,source,used,active,affine,request)
    np.testing.assert_array_equal(candidate[[0,1,3]],prior[[0,1,3]])
    np.testing.assert_array_equal(candidate[2],source[2]+request[0])


def test_edge_variation_retains_fixed_neighbor_displacement_offset():
    delta=np.array([[.001,.002,.003],[.004,.005,.006]])
    np.testing.assert_array_equal(tool.fixed_edge_difference(delta,{0:0},0,1),-delta[1])
    np.testing.assert_array_equal(tool.fixed_edge_difference(delta,{1:0},0,1),delta[0])
    np.testing.assert_array_equal(tool.fixed_edge_difference(delta,{0:0,1:1},0,1),np.zeros(3))
    # With vertex0 variable and vertex1 fixed at .004 m, absolute edge
    # variation is zero at variable=.004, not at variable=0.
    constant=tool.fixed_edge_difference(delta,{0:0},0,1)[0]
    assert abs(.004+constant)==0.
    assert abs(0.+constant)==.004


def test_complete_skin_face_support_clears_every_corner_past_crossed_bone():
    # A face straddles a facet at z=0. A contact-point-only displacement can
    # leave its low corner behind. The complete support must clear that corner.
    base=np.array([[0.,0.,-.004],[.01,0.,.002],[0.,.01,.003]])
    faces=np.array([[0,1,2]])
    bone=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]])
    constraints=tool.bone_face_support_constraints(base,faces,0,bone,[0.,0.,7.],.001)
    moved=base.copy()
    for face,bary,normal,bound in constraints:
        corner=int(np.argmax(bary));moved[corner]+=max(0.,bound)*normal
    assert np.min(moved[:,2]) == pytest.approx(.001)
    assert np.min(moved[:,2])-np.max(bone[:,2]) >= .001-1.e-15
    assert len(constraints)==3


def test_bone_face_support_is_translation_invariant_and_dimensionally_meters():
    base=np.array([[0.,0.,-.004],[.01,0.,.002],[0.,.01,.003]])
    bone=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]])
    first=tool.bone_face_support_constraints(base,[[0,1,2]],0,bone,[0.,0.,1.],.0005)
    shift=np.array([.17,-.4,.08])
    second=tool.bone_face_support_constraints(base+shift,[[0,1,2]],0,bone+shift,[0.,0.,1.],.0005)
    np.testing.assert_allclose([r[3] for r in first],[.0045,-.0015,-.0025],atol=1.e-17)
    np.testing.assert_allclose([r[3] for r in first],[r[3] for r in second],atol=1.e-17)


@pytest.mark.parametrize("direction,margin",[([0.,0.,0.],.001),([0.,0.,float('nan')],.001),([0.,0.,1.],0.),([0.,0.,1.],float('inf'))])
def test_bone_face_support_rejects_invalid_construction(direction,margin):
    with pytest.raises(ValueError,match="invalid bone face-support"):
        tool.bone_face_support_constraints(np.eye(3),[[0,1,2]],0,np.eye(3),direction,margin)

def test_explicit_bone_targets_preserve_retained_default_and_declared_order():
    assert tool.selected_bone_targets(None) == (10,12,14,16)
    assert tool.selected_bone_targets([3,11,13,15,17]) == (3,11,13,15,17)


@pytest.mark.parametrize("values", [[],[13,13],[0],[-1],[True],[2**32],[13.0]])
def test_explicit_bone_targets_reject_ambiguous_or_invalid_identity(values):
    with pytest.raises(ValueError,match="distinct positive UInt32"):
        tool.selected_bone_targets(values)

def test_separate_clear_reference_recovers_crossed_authoring_base():
    reference=np.array([[0,0,0],[0,.01,0],[0,0,.01],
                        [.02,0,0],[.02,.01,0],[.02,0,.01]],dtype="<f4")
    faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64)
    base=reference.copy()
    base[1,0]=.03
    rows=tool.rebase_self_contact_constraints(
        tool.face_support_constraints(reference,faces,0,1,.0005),reference,base,faces)
    assert any(0.<bound for *_,bound in rows)
    # Returning to the clear reference satisfies all rebased inequalities.
    recovery=reference.astype(float)-base.astype(float)
    assert all(float((wa@recovery[faces[a]]-wb@recovery[faces[b]])@n)>=bound-1e-15
               for a,wa,b,wb,n,bound in rows)
    # Collapsing both faces onto x=.01 remains disallowed.
    collapsed=reference.astype(float).copy();collapsed[:,0]=.01
    delta=collapsed-base
    assert any(float((wa@delta[faces[a]]-wb@delta[faces[b]])@n)<bound
               for a,wa,b,wb,n,bound in rows)


def test_reference_rebase_is_invariant_to_common_base_translation():
    reference=np.array([[0,0,0],[0,1,0],[0,0,1],[2,0,0],[2,1,0],[2,0,1]],dtype="<f4")
    faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64)
    rows=tool.face_support_constraints(reference,faces,0,1,.0005)
    base=reference.astype(float);base[1,0]=3.
    first=tool.rebase_self_contact_constraints(rows,reference,base,faces)
    translated=tool.rebase_self_contact_constraints(rows,reference,base+[8.,-4.,2.],faces)
    np.testing.assert_allclose([r[-1] for r in first],[r[-1] for r in translated],atol=1e-15)


@pytest.mark.parametrize("mutation",["faces","gids","float64","nonfinite"])
def test_separate_self_reference_rejects_identity_or_emission_drift(mutation):
    points=np.array([[0,0,0],[0,1,0],[0,0,1]],dtype="<f4")
    faces=np.array([[0,1,2]],dtype=np.int64)
    gids=np.array([10,20,30],dtype=np.uint64)
    reference=dict(capture_positions_m=points.copy(),capture_faces=faces.copy(),
                   capture_global_vertex_ids=gids.copy())
    if mutation=="faces":reference["capture_faces"][0]=[1,0,2]
    if mutation=="gids":reference["capture_global_vertex_ids"][0]=11
    if mutation=="float64":reference["capture_positions_m"]=points.astype(float)
    if mutation=="nonfinite":reference["capture_positions_m"][0,0]=np.nan
    with pytest.raises(ValueError,match="reference"):
        tool.validated_self_contact_reference(reference,points,faces,gids)


def test_separate_self_reference_preserves_shared_simplex_zero_bound():
    reference=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],dtype="<f4")
    faces=np.array([[0,1,2],[0,3,4]],dtype=np.int64)
    base=reference.copy();base[2,1]=-1.
    rows=tool.rebase_self_contact_constraints(
        tool.face_support_constraints(reference,faces,0,1,.0005),reference,base,faces)
    shared_rows=[r for r in rows if faces[r[0],int(np.argmax(r[1]))]==faces[r[2],int(np.argmax(r[3]))]]
    assert shared_rows and all(row[-1]==0. for row in shared_rows)

def test_adaptive_support_clears_actual_crossing_without_returning_to_old_axis():
    reference=np.array([[0,0,0],[.01,0,0],[0,.01,0],
                        [.02,0,.02],[.03,0,.02],[.02,.01,.02]],dtype="<f4")
    faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64)
    current=reference.copy()
    current[3:]=np.array([[.009,0,-.01],[.019,0,.01],[.009,.01,.01]],dtype="<f4")
    old=tool.face_support_constraints(reference,faces,0,1,.0005)
    new=tool.face_support_constraints(reference,faces,0,1,.0005,current_points=current)
    old=tool.rebase_self_contact_constraints(old,reference,current,faces)
    new=tool.rebase_self_contact_constraints(new,reference,current,faces)
    old_need=max(r[-1]/np.linalg.norm(r[-2]) for r in old)
    new_need=max(r[-1]/np.linalg.norm(r[-2]) for r in new)
    assert new_need < old_need and new_need < .002
    # A 1.1 mm lateral separation resolves the actual crossing along the
    # source-consistent side; it does not reverse the reference ordering.
    delta=np.zeros_like(current);delta[3:,0]=.0011
    assert all(float((wa@delta[faces[a]]-wb@delta[faces[b]])@n)>=bound
               for a,wa,b,wb,n,bound in new)
    assert any(float((wa@delta[faces[a]]-wb@delta[faces[b]])@n)<bound
               for a,wa,b,wb,n,bound in old)


def test_adaptive_support_does_not_waive_invalid_reference():
    ref=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype="<f4")
    faces=np.array([[0,1,2],[0,1,2]],dtype=np.int64)
    with pytest.raises(ValueError,match="separating"):
        tool.face_support_constraints(ref,faces,0,1,.0005,current_points=ref)
    with pytest.raises(ValueError,match="current"):
        tool.face_support_constraints(ref,np.array([[0,1,2],[0,2,1]]),0,1,.0005,
                                      current_points=np.array([[np.nan,0,0]]))


def test_interior_area_target_survives_float32_world_emission():
    # Shrinking this valid world-space triangle to exactly the strict LP area
    # bound rounds below it in Float32; an interior construction target passes.
    triangle=np.array([[.1,-.6,.1],[.101,-.6,.1],[.1,-.5981,.1]],dtype="<f4")
    normal=np.array([0.,0.,1.])
    area=lambda x: float(np.cross(x[1].astype(float)-x[0],x[2].astype(float)-x[0])@normal)
    original=area(triangle)
    def emitted(ratio):
        candidate=triangle.astype(float)
        candidate[2]=candidate[0]+ratio*(candidate[2]-candidate[0])
        return candidate.astype("<f4")
    strict=.2
    assert area(emitted(strict)) < strict*original
    target=tool.orientation_construction_ratio(strict,.01)
    assert area(emitted(target)) >= strict*original
    # No reserve preserves the old construction; disabling the area constraint
    # does not implicitly enable one.
    assert tool.orientation_construction_ratio(strict,0.) == strict
    assert tool.orientation_construction_ratio(0.,.01) == 0.


@pytest.mark.parametrize("ratio,reserve",[(.2,-.1),(.2,1.),(.2,float("nan")),(1.,.01)])
def test_invalid_orientation_reserve_is_rejected(ratio,reserve):
    with pytest.raises(ValueError,match="orientation"):
        tool.orientation_construction_ratio(ratio,reserve)

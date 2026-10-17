from fractions import Fraction as F
import numpy as np
import pytest
from numilab_human.resting_local_boolean_clearance import offset_new_boolean_cap_vertices

def args():
    p=[(F(0),F(0),F(0)),(F(1),F(0),F(0)),(F(0),F(1),F(0))]
    return dict(vertices_m=p,triangles=np.asarray([[0,1,2]],dtype="u4"),
                output_face_ancestry=[{"source_face":1}],source_face_count=1,
                obstacle_vertices_m=p,obstacle_triangles=np.asarray([[0,1,2]],dtype="u4"),
                inferred_obstacle_cap_mask=np.asarray([False]))

def test_cap_moves_outward_with_round_once_clearance():
    a=args();p,r=offset_new_boolean_cap_vertices(**a)
    assert np.all(p[:,2]>0)
    assert r["offset_vertex_count"]==3
    assert all(x["minimum_emitted_target_normal_projection_m"]>=r["gap_m"]*.5 for x in r["operations"])
    assert r["native_admission"] is False

def test_source_only_vertices_stay_exact():
    a=args();a["output_face_ancestry"]=[{"source_face":0}]
    p,r=offset_new_boolean_cap_vertices(**a)
    assert p.tobytes()==np.asarray(a["vertices_m"],dtype="<f4").tobytes()
    assert r["offset_vertex_count"]==0

def test_artificial_cap_may_not_enter_anatomy():
    a=args();a["inferred_obstacle_cap_mask"][:]=True
    with pytest.raises(ValueError,match="artificial clipping cap"):offset_new_boolean_cap_vertices(**a)

def test_exact_ancestry_range_is_required():
    a=args();a["output_face_ancestry"]=[{"source_face":2}]
    with pytest.raises(ValueError,match="out of range"):offset_new_boolean_cap_vertices(**a)

def test_displacement_is_bounded():
    a=args()
    with pytest.raises(ValueError,match="configuration"):
        offset_new_boolean_cap_vertices(**a,gap_m=.001,maximum_offset_m=.0001)


def test_shared_cap_vertex_clears_two_perpendicular_target_faces():
    a=args()
    a["obstacle_vertices_m"] += [(F(0),F(0),F(1))]
    a["obstacle_triangles"]=np.asarray([[0,1,2],[0,3,1]],dtype="u4")
    a["inferred_obstacle_cap_mask"]=np.asarray([False,False])
    a["triangles"]=np.asarray([[0,1,2],[0,2,1]],dtype="u4")
    a["output_face_ancestry"]=[{"source_face":1},{"source_face":2}]
    p,r=offset_new_boolean_cap_vertices(**a)
    assert np.all(p[:3,1]-np.asarray(a["vertices_m"],dtype="f4")[:3,1]>0)
    assert np.all(p[:3,2]>0)
    assert p[3].tobytes()==np.asarray(a["vertices_m"][3],dtype="<f4").tobytes()
    assert all(len(op["obstacle_face_ids"])==2 for op in r["operations"])


def test_opposed_incident_target_normals_require_another_construction():
    a=args()
    a["obstacle_triangles"]=np.asarray([[0,1,2],[0,2,1]],dtype="u4")
    a["inferred_obstacle_cap_mask"]=np.asarray([False,False])
    a["triangles"]=np.asarray([[0,1,2],[0,2,1]],dtype="u4")
    a["output_face_ancestry"]=[{"source_face":1},{"source_face":2}]
    with pytest.raises(ValueError,match="opposed target normals"):
        offset_new_boolean_cap_vertices(**a)

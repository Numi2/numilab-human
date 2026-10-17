from fractions import Fraction as F
import itertools
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from numilab_human.resting_closed_obstacle_clip import clip_closed_mesh_to_aabb
from numilab_human import cardiac_cavity_intersections as ci

def cube(lo,hi,reverse=False):
    p=np.asarray(list(itertools.product([lo,hi],repeat=3)),dtype="float32")
    h=ConvexHull(p);f=h.simplices.copy()
    for k,tri in enumerate(f):
        t=p[tri].astype(float)
        if np.cross(t[1]-t[0],t[2]-t[0])@h.equations[k,:3]<0:f[k]=tri[::-1]
    if reverse:f=f[:,::-1]
    return p,f

def check(mesh,expected):
    records=ci._records(mesh["vertices_m"],mesh["triangles"])
    audit=ci._audit_pair(records,records,same_surface=True)
    assert audit["count"]==0
    edges={}
    vol=F(0)
    for f in mesh["triangles"]:
        a,b,c=[mesh["vertices_m"][i] for i in f]
        vol+=sum(a[k]*(b[(k+1)%3]*c[(k+2)%3]-b[(k+2)%3]*c[(k+1)%3]) for k in range(3))/6
        for a,b in zip(f,f[1:]+f[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    assert all(len(x)==2 and x[0]==x[1][::-1] for x in edges.values())
    assert vol==expected
    assert len(mesh["output_face_parent_ids"])==len(mesh["triangles"])
    assert all(x and all(type(i)is int and i>=0 for i in x) for x in mesh["output_face_parent_ids"])
    assert len(mesh["inferred_clipping_cap_mask"])==len(mesh["triangles"])

def test_single_plane_clip_and_full_source_support():
    p,f=cube(-2,2)
    mesh=clip_closed_mesh_to_aabb(p,f,[F(0),F(-3),F(-3)],[F(3),F(3),F(3)])
    check(mesh,F(32))
    assert any(mesh["inferred_clipping_cap_mask"])
    assert all(i<len(f) for x in mesh["output_face_parent_ids"] for i in x)

def test_six_planes_and_exact_dyadic_vertices():
    p,f=cube(-2,2)
    mesh=clip_closed_mesh_to_aabb(p,f,[F(-1,2)]*3,[F(1,2)]*3)
    check(mesh,F(1))
    assert all(F(-1,2)<=mesh["vertices_m"][i][k]<=F(1,2) for face in mesh["triangles"] for i in face for k in range(3))

def test_nested_cap_loops_preserve_a_real_cavity():
    p,f=cube(-2,2);q,g=cube(-1,1,True)
    mesh=clip_closed_mesh_to_aabb(np.r_[p,q],np.r_[f,g+len(p)],[F(0),F(-3),F(-3)],[F(3),F(3),F(3)])
    check(mesh,F(28))

def test_disjoint_closed_components_remain_closed():
    p,f=cube(-2,-1);q,g=cube(1,2)
    mesh=clip_closed_mesh_to_aabb(np.r_[p,q],np.r_[f,g+len(p)],[F(-3)]*3,[F(3)]*3)
    check(mesh,F(2))
    assert not any(mesh["inferred_clipping_cap_mask"])

def test_open_source_rejected_without_fabricated_caps():
    p,f=cube(-2,2)
    with pytest.raises(ValueError,match="unexplained open boundary"):
        clip_closed_mesh_to_aabb(p,f[:-1],[F(-3)]*3,[F(3)]*3)

def test_empty_material_rejected():
    p,f=cube(-2,2)
    with pytest.raises(ValueError,match="empty"):
        clip_closed_mesh_to_aabb(p,f,[F(3)]*3,[F(4)]*3)


def test_preexisting_hole_on_clip_plane_cannot_be_filled():
    p,f=cube(-2,2)
    # Remove every triangle on the x=-2 face, exactly the first clip plane.
    kept=f[~np.all(p[f][:,:,0] == -2,axis=1)]
    with pytest.raises(ValueError,match="unexplained open boundary"):
        clip_closed_mesh_to_aabb(p,kept,[F(-2),F(-3),F(-3)],[F(3)]*3)

def test_zero_area_input_rejected():
    p,f=cube(-2,2)
    f=f.copy();f[0,2]=f[0,1]
    with pytest.raises(ValueError,match="degenerate input"):
        clip_closed_mesh_to_aabb(p,f,[F(-3)]*3,[F(3)]*3)

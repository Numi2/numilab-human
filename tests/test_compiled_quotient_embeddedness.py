"""Spatially identical compiled vertices must be distinguished from gaps."""
from numilab_human.compiled_quotient_embeddedness import classify_quotient
from numilab_human.whole_body_embeddedness import classify_surface


def test_exact_weld_closes_index_seam_without_moving_source_support():
    vertices=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),(0.,0.,2.),(0.,0.,0.)]
    faces=[[4,2,1],[0,1,3],[1,2,3],[2,0,3]]
    indexed=classify_surface(vertices,faces)
    assert not indexed['topology']['closed_oriented_manifold_candidate']
    quotient=classify_quotient(vertices,faces)
    assert quotient['status']=='closed_embedded_source_candidate'
    assert quotient['compiled_exact_duplicate_vertex_count']==1
    assert quotient['source_face_support_unchanged']
    assert not quotient['new_points_added']


def test_one_bit_gap_is_not_welded():
    from math import nextafter
    vertices=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),(0.,0.,2.),(nextafter(0.,1.),0.,0.)]
    faces=[[4,2,1],[0,1,3],[1,2,3],[2,0,3]]
    quotient=classify_quotient(vertices,faces)
    assert quotient['compiled_exact_duplicate_vertex_count']==0
    assert not quotient['topology']['closed_oriented_manifold_candidate']


def test_exact_weld_cannot_rescue_degenerate_triangle():
    vertices=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),(0.,0.,2.),(0.,0.,0.)]
    faces=[[0,2,1],[0,1,3],[1,2,3],[2,0,3],[0,4,1]]
    quotient=classify_quotient(vertices,faces)
    assert quotient['status']=='exact_degenerate_face'
    assert not quotient['closed_embedded_surface_candidate']

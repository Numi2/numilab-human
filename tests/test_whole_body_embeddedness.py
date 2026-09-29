"""Adversarial compiled-mesh cases for the complete source census."""
from numilab_human.whole_body_embeddedness import classify_surface


def tetra(origin=(0.,0.,0.)):
    x,y,z=origin
    points=[(x,y,z),(x+2.,y,z),(x,y+2.,z),(x,y,z+2.)]
    faces=[[0,2,1],[0,1,3],[1,2,3],[2,0,3]]
    return points,faces


def test_closed_embedded_tetrahedron():
    points,faces=tetra()
    result=classify_surface(points,faces)
    assert result['status']=='closed_embedded_source_candidate'
    assert result['exact_intersection_pairs']==0
    assert result['closed_embedded_surface_candidate']


def test_closed_components_can_still_intersect():
    first,faces=tetra()
    second,more=tetra((.5,.5,.5))
    points=first+second
    result=classify_surface(points,faces+[[i+4 for i in f] for f in more])
    assert result['topology']['closed_oriented_manifold_candidate']
    assert result['status']=='closed_self_intersecting'
    assert result['exact_intersection_pairs']>0
    assert not result['closed_embedded_surface_candidate']


def test_open_and_degenerate_sources_are_never_admitted():
    points,faces=tetra()
    opened=classify_surface(points,faces[:-1])
    assert opened['status']=='open_or_nonmanifold'
    assert opened['exact_intersection_pairs']==0
    assert not opened['closed_embedded_surface_candidate']
    degenerate=classify_surface(points,faces+[[0,1,1]])
    assert degenerate['status']=='exact_degenerate_face'
    assert degenerate['self_intersection']=='not_checked_exact_degenerate_face'
    assert degenerate['exact_intersection_pairs'] is None

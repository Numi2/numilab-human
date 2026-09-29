"""Exact-chain and native-coordinate adversarial cases for topology candidates."""
from fractions import Fraction
import random

import pytest

from numilab_human import surface_topology_repair as repair
from numilab_human import surface_topology_audit as audit
from numilab_human import cardiac_cavity_intersections as intersections


def tetrahedron_with_split_seam():
    points=[(0,0,0),(2,0,0),(0,2,0),(0,0,2),(1,0,0)]
    keys=[tuple(map(Fraction,p)) for p in points]
    # The first long edge B-A must be partitioned at M; the adjacent
    # triangular side is already split, and no cap is geometrically needed.
    faces=[(0,2,1),(0,4,3),(4,1,3),(1,2,3),(2,0,3)]
    return points,keys,faces


def test_exact_seam_and_opposite_faces_preserve_oriented_source_chain():
    points,keys,faces=tetrahedron_with_split_seam()
    faces += [(0,2,1),(1,2,0),(0,4,1)]  # opposite duplicate and zero-area face
    result=repair.repair(points,keys,faces)
    assert result['after']['closed_oriented_manifold_candidate']
    assert result['conforming_seam_subdivisions']
    assert result['cancelled_opposite_source_face_pairs']
    assert result['removed_exact_zero_area_source_faces']
    assert audit.verify_derivation(keys,faces,result['triangles'],result)['oriented_source_chain_identity']
    corrupted={**result,'source_face_ids':result['source_face_ids'].copy()}
    corrupted['source_face_ids'][0]=3
    with pytest.raises(Exception,match='face|partition|plane'):
        audit.verify_derivation(keys,faces,result['triangles'],corrupted)


def test_noncollinear_opening_is_not_filled():
    points,keys,faces=tetrahedron_with_split_seam()
    result=repair.repair(points,keys,faces[:-1])
    assert not result['after']['closed_oriented_manifold_candidate']
    assert result['hole_capping'] is False


def test_exact_aabb_tree_matches_existing_complete_sweep_predicate():
    for seed in range(15):
        rng=random.Random(seed)
        vertices=[(rng.randrange(-5,6),rng.randrange(-5,6),rng.randrange(-5,6)) for _ in range(12)]
        faces=[]
        while len(faces)<12:
            f=rng.sample(range(len(vertices)),3)
            p=[vertices[i] for i in f]
            n=intersections._cross(intersections._sub(p[1],p[0]),intersections._sub(p[2],p[0]))
            if any(n): faces.append(f)
        expected=intersections._audit_pair(intersections._records(vertices,faces),
                                           intersections._records(vertices,faces),same_surface=True)
        observed=audit.exact_embedding(vertices,faces)
        for key in ['triangle_pairs','count','aabb_candidate_pairs','allowed_shared_vertex_or_edge_pairs']:
            assert observed[key]==expected[key]


def test_exact_one_bit_coordinate_change_is_visible_to_predicate():
    # No tolerance may erase a real gap between nonadjacent surfaces.
    above=float.fromhex('0x1.0000000000001p+0')
    vertices=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),
              (0.5,0.5,above),(1.5,0.5,above),(0.5,1.5,above)]
    clean=audit.exact_embedding(vertices,[[0,1,2],[3,4,5]])
    assert clean['count']==0
    vertices[3]=(0.5,0.5,0.)
    vertices[4]=(1.5,0.5,0.)
    vertices[5]=(0.5,1.5,0.)
    touching=audit.exact_embedding(vertices,[[0,1,2],[3,4,5]])
    assert touching['count']==1

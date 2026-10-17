"""Exact mixed coplanar/transverse arrangements, independent of native admission."""
from fractions import Fraction as F
import pytest
from numilab_human import cardiac_cavity_partition as p
from numilab_human.model import ImportError as HumanImportError

FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def source(offset=(0, 0, 0), reverse=False):
    first = [(0,0,0), (2,0,0), (0,2,0), (0,0,2)]
    second = [tuple(a+b for a,b in zip(point, offset)) for point in first]
    faces2 = [(a,c,b) for a,b,c in FACES] if reverse else FACES
    return {"vertices": first+second,
            "triangles": FACES+[tuple(i+4 for i in f) for f in faces2],
            "source_sha256": "a"*64}


def volume(result):
    mesh = result["output_mesh"]
    v = mesh["vertices_m"]
    return sum(p.predicates._dot(v[a], p.predicates._cross(v[b], v[c]))
               for a,b,c in mesh["triangles"])/6


def test_exact_duplicate_shell_is_emitted_once_with_both_parents():
    result = p.construct_positive_winding_self_union(source(), allow_coplanar_overlap=True)
    assert volume(result) == F(4,3)
    assert result["output_topology"]["closed_oriented_2_manifold"]
    assert result["output_self_intersection_audit"]["count"] == 0
    assert all(len(row["source_faces"]) == 2 for row in result["output_face_ancestry"])
    assert all(row["reversed_from_sources"] == [False, False]
               for row in result["output_face_ancestry"])


def test_partial_coplanar_and_transverse_union_in_one_pass():
    result = p.construct_positive_winding_self_union(source((1,0,0)), allow_coplanar_overlap=True)
    assert volume(result) == F(5,2)  # two 4/3 tetrahedra minus 1/6 overlap
    assert result["output_topology"]["component_count"] == 1
    assert result["output_self_intersection_audit"]["count"] == 0
    assert any(len(row["source_faces"]) == 2 for row in result["output_face_ancestry"])
    assert {index for row in result["output_face_ancestry"]
            for index in row["source_faces"]} <= set(range(8))


def test_default_still_rejects_coplanar_overlap():
    with pytest.raises(HumanImportError, match="coplanar self-intersection"):
        p.construct_positive_winding_self_union(source((1,0,0)))


def test_opposed_identical_shells_have_no_positive_boundary():
    with pytest.raises(HumanImportError, match="no boundary"):
        p.construct_positive_winding_self_union(source(reverse=True), allow_coplanar_overlap=True)


@pytest.mark.parametrize("offset,expected", [((0,1,0),F(5,2)), ((0,0,1),F(5,2)),
                                            ((3,0,0),F(8,3))])
def test_other_axes_and_disjoint_components(offset, expected):
    result = p.construct_positive_winding_self_union(source(offset), allow_coplanar_overlap=True)
    assert volume(result) == expected
    assert result["output_self_intersection_audit"]["count"] == 0


def test_face_order_and_cyclic_corner_changes_preserve_boundary():
    original = source((1,0,0))
    reference = p.construct_positive_winding_self_union(original, allow_coplanar_overlap=True)
    changed = dict(original)
    changed["triangles"] = [(b,c,a) for a,b,c in reversed(original["triangles"])]
    result = p.construct_positive_winding_self_union(changed, allow_coplanar_overlap=True)
    assert volume(result) == volume(reference)
    assert result["output_topology"] == reference["output_topology"]
    assert result["output_self_intersection_audit"]["count"] == 0


def test_three_coincident_layers_retain_complete_parent_support():
    original = source()
    original["vertices"] += original["vertices"][:4]
    original["triangles"] += [tuple(i+8 for i in face) for face in FACES]
    result = p.construct_positive_winding_self_union(original, allow_coplanar_overlap=True)
    assert volume(result) == F(4,3)
    assert all(len(row["source_faces"]) == 3 for row in result["output_face_ancestry"])


def test_opposed_shared_interface_is_removed_with_transverse_cuts():
    # Reflection across x=0: exactly shared full face, separate interiors.
    original = source()
    original["vertices"][5] = (-2,0,0)
    original["triangles"][4:] = [(a,c,b) for a,b,c in original["triangles"][4:]]
    result = p.construct_positive_winding_self_union(original, allow_coplanar_overlap=True)
    assert volume(result) == F(8,3)
    assert result["output_topology"]["component_count"] == 1


@pytest.mark.parametrize("value", [1, None, "true"])
def test_option_requires_boolean(value):
    with pytest.raises(HumanImportError, match="option must be boolean"):
        p.construct_positive_winding_self_union(source(), allow_coplanar_overlap=value)


def test_singular_point_touch_is_not_admitted():
    with pytest.raises(HumanImportError):
        p.construct_positive_winding_self_union(source((1,1,0)), allow_coplanar_overlap=True)


def test_mixed_oriented_layers_retain_all_support_and_reversal_flags():
    original = source()
    original["vertices"] += original["vertices"][:4]
    original["triangles"] += [(a+8,c+8,b+8) for a,b,c in FACES]
    result = p.construct_positive_winding_self_union(original, allow_coplanar_overlap=True)
    assert volume(result) == F(4,3)
    assert all(len(row["source_faces"]) == 3 and row["reversed_from_sources"] == [False,False,True]
               for row in result["output_face_ancestry"])

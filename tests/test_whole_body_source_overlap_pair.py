import pytest

from numilab_human.model import ImportError
from numilab_human.whole_body_source_overlap_pair import read_obj_triangles


def test_obj_quad_is_fanned_in_source_order():
    points, faces = read_obj_triangles(
        b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    )

    assert points.tolist() == [[0., 0., 0.], [1., 0., 0.],
                               [1., 1., 0.], [0., 1., 0.]]
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_obj_negative_vertex_indices_fail_closed():
    with pytest.raises(ImportError, match="non-positive OBJ index"):
        read_obj_triangles(
            b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n"
        )

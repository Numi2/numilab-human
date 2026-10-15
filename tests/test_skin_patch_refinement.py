import struct

import numpy as np
import pytest

from numilab_human import model
from numilab_human.skin_patch_refinement import refine_skin_patch
from numilab_human.skin_source_payload_preflight import decode_payload


def fixture():
    # Closed tetrahedron plus a deliberately unused source slot.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 8, 7]], dtype="<f4"
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    v = np.zeros((5, 14), dtype="<u4")
    f = v.view("<f4")
    f[:, :3] = points
    f[:, 3:6] = [0, 0, 1]
    v[:, 6:10] = [0, 1, 2, 3]
    f[:, 10:14] = [1, 0, 0, 0]
    weights = np.zeros((5, 6), dtype="<f4")
    weights[np.arange(5), np.arange(5)] = 1
    bindings = b"".join(
        struct.pack("<I8f", i, 0, 0, 0, 0, 0, 0, 1, 1) for i in range(6)
    )
    raw = struct.pack("<8s5I32s", b"NHSKIN1\0", 5, 6, 5, 12, 123, b"x" * 32)
    return raw + bindings + v.tobytes() + faces.tobytes() + weights.tobytes()


M = np.eye(4, dtype=float).tolist()


@pytest.mark.parametrize(
    "selection,expected_vertices,expected_faces",
    [([0], 8, 10), ([0, 1], 10, 14), ([0, 1, 2, 3], 11, 16)],
)
def test_conforming_closed_patch_preserves_winding_and_unused_source_record(
    selection, expected_vertices, expected_faces
):
    raw = fixture()
    out, p = refine_skin_patch(raw, selection, global_source_matrix=M)
    a, b = decode_payload(raw), decode_payload(out)
    f = b["indices"].reshape(-1, 3)
    edges = {}
    for face in f:
        for x, y in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(x), int(y))))
            edges.setdefault(key, []).append((int(x), int(y)))
    assert all(
        len(directions) == 2 and directions[0] == directions[1][::-1]
        for directions in edges.values()
    )
    assert b["vertex_count"] - 1 - len(edges) + len(f) == 2
    assert b["vertex_count"] == expected_vertices and len(f) == expected_faces
    assert np.array_equal(a["vertices_u"][4], b["vertices_u"][4])
    assert all(x == "1" for x in p["exact_parent_oriented_area_ratios"])

    # Independently check total signed enclosed volume, not only edge counts.
    def volume(data):
        t = data["vertices_f"][:, :3].astype(float)[data["indices"].reshape(-1, 3)]
        return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6

    assert volume(a) == volume(b)
    assert not p["native_admission"] and not p["anatomical_admission"]


def test_birth_weights_retain_every_positive_parent_support_and_existing_rows():
    raw = fixture()
    out, p = refine_skin_patch(raw, [0], global_source_matrix=M)
    a, b = decode_payload(raw), decode_payload(out)
    assert np.array_equal(a["full_weights"], b["full_weights"][:5])
    for birth in p["appended_vertex_births"]:
        i = birth["vertex_id"]
        x, y = birth["parent_edge_vertex_ids"]
        assert np.array_equal(
            b["full_weights"][i], (a["full_weights"][x] + a["full_weights"][y]) * 0.5
        )
        assert np.array_equal(
            b["full_weights"][i] > 0,
            (a["full_weights"][x] > 0) | (a["full_weights"][y] > 0),
        )
    assert out[:16] == raw[:16] and out[24 : 60 + 36 * 6] == raw[24 : 60 + 36 * 6]


@pytest.mark.parametrize("selection", [[], [-1], [4], [0, 0], [0.5]])
def test_invalid_source_face_selection_fails_closed(selection):
    with pytest.raises(model.ImportError):
        refine_skin_patch(fixture(), selection, global_source_matrix=M)


def test_repeated_refinement_keeps_large_integer_material_ids_exact():
    from numilab_human.skin_patch_refinement import appended_material_vertex_ids
    ids=np.array([2**53+17,2**53+21],dtype=np.int64)
    first=appended_material_vertex_ids(ids,2)
    second=appended_material_vertex_ids(first,3)
    np.testing.assert_array_equal(second,
        [2**53+17,2**53+21,2**53+22,2**53+23,2**53+24,2**53+25,2**53+26])

def test_material_vertex_identity_allocation_rejects_overflow_and_aliases():
    from numilab_human.skin_patch_refinement import appended_material_vertex_ids
    for ids,count in [(np.array([np.iinfo(np.int64).max]),1),
                      (np.array([1,1],dtype=np.int64),1),
                      (np.array([1.]),1),
                      (np.array([-1],dtype=np.int64),1),
                      (np.array([1],dtype=np.int64),True)]:
        with pytest.raises(model.ImportError):
            appended_material_vertex_ids(ids,count)

from __future__ import annotations

import numpy as np

from numilab_human.right_choroid_laterality_repair import (
    _current_897_compact_component,
    _current_897_coordinate_quotient,
    _current_897_source_face_order,
)
from numilab_human.cardiac_cavity_geometry import analyze_topology


def _tetra_points(offset: float = 0.0) -> np.ndarray:
    return np.asarray([
        [offset + 0.0, 0.0, 0.0],
        [offset + 1.0, 0.0, 0.0],
        [offset + 0.0, 1.0, 0.0],
        [offset + 0.0, 0.0, 1.0],
    ], dtype="<f4")


def _outward_tetra_faces(vertex_offset: int = 0) -> np.ndarray:
    return np.asarray([
        [vertex_offset + 0, vertex_offset + 2, vertex_offset + 1],
        [vertex_offset + 0, vertex_offset + 1, vertex_offset + 3],
        [vertex_offset + 0, vertex_offset + 3, vertex_offset + 2],
        [vertex_offset + 1, vertex_offset + 2, vertex_offset + 3],
    ], dtype=np.int64)


def test_current_897_coordinate_quotient_welds_only_exact_packed_positions():
    points = _tetra_points()
    faces = _outward_tetra_faces()
    # Duplicate each triangle's vertex records exactly as raw OBJ/NHA seams do.
    dup_points = points[faces.reshape(-1)]
    vertices6 = np.column_stack((dup_points, np.tile([0.0, 0.0, 1.0], (12, 1)))).astype("<f4")
    dup_faces = np.arange(12, dtype=np.int64).reshape(-1, 3)

    quotient = _current_897_coordinate_quotient(vertices6, dup_faces)
    assert quotient["identified_vertex_count"] == 8
    assert quotient["unreferenced_quotient_vertex_count"] == 0
    assert quotient["topology"]["closed_oriented_manifold_candidate"] is True
    assert quotient["topology"]["face_component_count"] == 1
    assert quotient["topology"]["euler_characteristic"] == 2

    perturbed = vertices6.copy()
    perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.float32(np.inf))
    near = _current_897_coordinate_quotient(perturbed, dup_faces)
    assert near["identified_vertex_count"] == 7  # no distance tolerance


def test_current_897_face_excision_preserves_vertex_table_and_local_indices():
    a = _tetra_points(0.0)
    b = _tetra_points(3.0)
    points = np.concatenate((a, b), axis=0)
    normals = np.tile(np.asarray([[0.1234567, 0.0, 0.9876543]], dtype="<f4"), (8, 1))
    vertices6 = np.concatenate((points, normals), axis=1).astype("<f4")
    faces = np.concatenate((_outward_tetra_faces(0), _outward_tetra_faces(4)), axis=0)

    candidate = _current_897_compact_component(vertices6, faces, [4, 5, 6, 7])
    assert candidate["vertices6"].tobytes() == vertices6.tobytes()
    assert candidate["retained_parent_vertex_ids"] == list(range(8))
    assert candidate["retained_parent_face_ids"] == [0, 1, 2, 3]
    np.testing.assert_array_equal(candidate["faces"], faces[:4])
    assert analyze_topology(candidate["vertices6"][:, :3].tolist(),
                            candidate["faces"].tolist())["face_component_count"] == 1


def test_current_897_raw_face_ancestry_maps_after_exact_cleanup():
    removed = [
        18447, 18448, 19953, 19965, 20401, 20402,
        23509, 23511, 24119, 24120, 28870, 28871,
    ]
    source_order = _current_897_source_face_order(28872, removed)
    parent_positions = {source_id: parent_id
                        for parent_id, source_id in enumerate(source_order)}
    assert [parent_positions[source_id] for source_id in range(28862, 28870)] == list(range(28852, 28860))
    assert len(source_order) == 28860


def test_lung_composition_retains_unrelated_rows_and_rejects_eye_change():
    import copy
    import pytest
    from numilab_human.right_choroid_laterality_repair import _verify_disjoint_lung_rows
    from numilab_human import model as human
    row = {"body_index": 20, "layer": 1, "flags": 0,
           "vertices6": np.column_stack((_tetra_points(), np.zeros((4, 3)))).astype("<f4"),
           "faces": _outward_tetra_faces()}
    reference = {305: copy.deepcopy(row), 382: copy.deepcopy(row)}
    parent = copy.deepcopy(reference)
    parent[305]["vertices6"][0, 0] += np.float32(1e-6)
    assert _verify_disjoint_lung_rows(reference, parent) == [305]
    assert reference[305]["vertices6"][0, 0] == 0
    parent[382]["faces"] = parent[382]["faces"][:-1]
    with pytest.raises(human.ImportError, match="non-lung row"):
        _verify_disjoint_lung_rows(reference, parent)


def test_lung_composition_rejects_changed_identity_and_owner():
    import copy
    import pytest
    from numilab_human.right_choroid_laterality_repair import _verify_disjoint_lung_rows
    from numilab_human import model as human
    row = {"body_index": 20, "layer": 1, "flags": 0,
           "vertices6": np.column_stack((_tetra_points(), np.zeros((4, 3)))).astype("<f4"),
           "faces": _outward_tetra_faces()}
    with pytest.raises(human.ImportError, match="identity set"):
        _verify_disjoint_lung_rows({305: row}, {306: row})
    parent = copy.deepcopy(row)
    parent["body_index"] = 23
    with pytest.raises(human.ImportError, match="owner fields"):
        _verify_disjoint_lung_rows({305: row}, {305: parent})

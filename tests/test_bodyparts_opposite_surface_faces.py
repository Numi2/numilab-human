from numilab_human.model import _bodyparts_cancel_opposite_surface_faces


def test_opposite_source_faces_cancel_without_changing_retained_support() -> None:
    vertices = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0), (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
    ]
    shell = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    faces = [*shell, (4, 5, 6), (6, 5, 4)]
    result_vertices, result_faces, audit = _bodyparts_cancel_opposite_surface_faces(
        vertices, faces, "test-source"
    )
    assert audit["cancelled_opposite_face_pairs"] == [[4, 5]]
    assert audit["source_triangle_count"] == 6
    assert audit["retained_triangle_count"] == 4
    assert audit["source_coordinate_support_preserved"]
    assert audit["removed_unique_visual_support_triangle_count"] == 0
    assert not audit["new_faces_added"] and not audit["vertices_moved"]
    assert result_vertices == vertices[:4]
    assert result_faces == shell


def test_same_winding_and_displaced_faces_remain_for_audit() -> None:
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    faces = [(0, 1, 2), (1, 2, 0), (0, 3, 1)]
    result_vertices, result_faces, audit = _bodyparts_cancel_opposite_surface_faces(
        vertices, faces, "test-source"
    )
    assert result_vertices == vertices
    assert result_faces == faces
    assert audit["cancelled_opposite_face_pairs"] == []


def test_cancelled_double_sided_sheet_reports_lost_visual_support() -> None:
    vertices = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, 0.0, 3.0), (2.0, 0.0, 3.0), (0.0, 2.0, 3.0),
    ]
    shell = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    result_vertices, result_faces, audit = _bodyparts_cancel_opposite_surface_faces(
        vertices, [*shell, (4, 5, 6), (6, 5, 4)], "test-source"
    )
    assert result_vertices == vertices[:4] and result_faces == shell
    assert audit["oriented_source_chain_preserved"]
    assert not audit["source_coordinate_support_preserved"]
    assert audit["removed_unique_visual_support_triangle_count"] == 1
    assert audit["removed_visual_support_area_mm2"] == 2.0

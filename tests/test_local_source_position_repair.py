import numpy as np
import pytest

from numilab_human.resting_intercostal_mesh_arrangement import replace_row_source_positions


def row_fixture():
    # Two closed solids, an indexed seam alias, and one unused allocation.
    points = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [3, 0, 0], [4, 0, 0], [3, 1, 0], [3, 0, 1],
        [0, 0, 0], [9, 9, 9],
    ], dtype="<f4")
    tetra = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4")
    first = tetra.copy()
    first[0, 0] = 8
    # Authored normals deliberately differ from a full-row regeneration.
    normals = np.tile(np.asarray([0, 0, 1], dtype="<f4"), (len(points), 1))
    vertices6 = np.column_stack((points, normals)).astype("<f4")
    return dict(body_index=20, layer=1, flags=0,
                vertices6=vertices6, faces=np.concatenate((first, tetra + 4)))


def test_local_repair_preserves_distant_hard_normals_unused_and_row_identity():
    row = row_fixture()
    original = row["vertices6"].copy()
    positions = original[:, :3].copy()
    positions[[0, 8], 0] = np.float32(.01)
    updated, proof = replace_row_source_positions(
        row, positions, changed_source_vertex_slots=[8, 0],
    )
    assert row["vertices6"].tobytes() == original.tobytes()
    assert updated["vertices6"][4:].tobytes() != original[4:].tobytes()  # alias8 changed
    assert updated["vertices6"][[4, 5, 6, 7, 9]].tobytes() == original[[4, 5, 6, 7, 9]].tobytes()
    assert updated["faces"] is row["faces"]
    assert (updated["body_index"], updated["layer"], updated["flags"]) == (20, 1, 0)
    assert proof["changed_source_vertex_slots"] == [0, 8]
    assert proof["unaffected_vertex_records_byte_exact"]
    assert proof["exact_coordinate_seams_preserved"]
    assert np.allclose(np.linalg.norm(updated["vertices6"][:4, 3:], axis=1), 1)


def test_repair_rejects_split_source_seam():
    row = row_fixture()
    positions = row["vertices6"][:, :3].copy()
    positions[0, 0] = np.float32(.01)
    with pytest.raises(ValueError, match="split an exact source-coordinate seam"):
        replace_row_source_positions(row, positions, changed_source_vertex_slots=[0])


@pytest.mark.parametrize("declared", [[0], [0, 8, 8], [True, 8], [0, 8, 10]])
def test_repair_rejects_incomplete_or_invalid_declaration(declared):
    row = row_fixture()
    positions = row["vertices6"][:, :3].copy()
    positions[[0, 8], 0] = np.float32(.01)
    with pytest.raises(ValueError):
        replace_row_source_positions(row, positions, changed_source_vertex_slots=declared)


def test_repair_rejects_precision_conversion_and_unused_motion():
    row = row_fixture()
    with pytest.raises(ValueError, match="exact Float32"):
        replace_row_source_positions(row, row["vertices6"][:, :3].astype(float),
                                     changed_source_vertex_slots=[0])
    positions = row["vertices6"][:, :3].copy()
    positions[9, 0] += np.float32(1)
    with pytest.raises(ValueError, match="unreferenced"):
        replace_row_source_positions(row, positions, changed_source_vertex_slots=[9])


def test_repair_rejects_degenerate_face():
    row = row_fixture()
    positions = row["vertices6"][:, :3].copy()
    positions[[0, 8]] = positions[1]
    with pytest.raises(ValueError, match="degenerate incident face"):
        replace_row_source_positions(row, positions, changed_source_vertex_slots=[0, 8])

def test_repair_accepts_existing_nha_parser_int64_face_view_without_conversion():
    row = row_fixture()
    row["faces"] = row["faces"].astype("<i8")
    positions = row["vertices6"][:, :3].copy()
    positions[[0, 8], 0] = np.float32(.01)
    updated, _ = replace_row_source_positions(
        row, positions, changed_source_vertex_slots=[0, 8],
    )
    assert updated["faces"] is row["faces"]

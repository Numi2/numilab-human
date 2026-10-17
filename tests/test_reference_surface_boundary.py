from fractions import Fraction as Q
import numpy as np
import pytest
from numilab_human import passive_attachment_composition as pc
from numilab_human.reference_surface_boundary import emit_source_face_boundary
from numilab_human.resting_anatomy_verify import prepare_surface

FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4")


def fixture():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    ii = np.full((4, 4), 0xffffffff, dtype="<u4")
    ii[:, 0] = [0, 0, 1, 0]
    ww = np.zeros((4, 4), dtype="<f4"); ww[:, 0] = 1
    bindings = [dict(core=k, translation=np.zeros(3, dtype="<f4"),
                     quaternion=np.array([0, 0, 0, 1], dtype="<f4"), scale=np.float32(1))
                for k in range(2)]
    row = dict(stable_id=147, positions=p, normals=np.tile(np.array([0, 0, 1], dtype="<f4"), (4, 1)),
               binding_indices=ii, weights=ww, bindings=bindings, faces=FACES,
               vertex_count=4, binding_count=2)
    poses = {0: np.array([.5, -.5, .25, 0, 0, 0, 1], dtype="<f4"),
             1: np.array([-.25, .25, -.125, 0, 0, 0, 1], dtype="<f4")}
    return row, poses


def boundary(row, poses):
    world = pc._native_blended_rest_positions(row, row["positions"], poses)
    points = [tuple(Q.from_float(float(x)) for x in p) for p in world]
    points.append(tuple((x+y)/2 for x, y in zip(points[0], points[2])))
    return dict(output_mesh=dict(vertices_m=points,
                                triangles=[[0, 4, 1], [4, 2, 1], FACES[1].tolist(), [0, 3, 4], [4, 3, 2], FACES[3].tolist()]),
                output_face_ancestry=[dict(source_face=i) for i in [0, 0, 1, 2, 2, 3]])


def test_exact_birth_preserves_allocations_and_owner_interpolation():
    row, poses = fixture()
    result = emit_source_face_boundary(row, poses, boundary(row, poses))
    patch = result["patch_arrays"]
    np.testing.assert_array_equal(patch["vertices6"][:4, :3], row["positions"])
    np.testing.assert_array_equal(patch["binding_indices"][:4], row["binding_indices"])
    np.testing.assert_array_equal(patch["weights"][:4], row["weights"])
    np.testing.assert_array_equal(patch["binding_indices"][4, :2], [0, 1])
    np.testing.assert_array_equal(patch["weights"][4], [.5, .5, 0, 0])
    np.testing.assert_array_equal(result["world_positions_f32"][4],
                                  np.array([.125, .375, .0625], dtype="<f4"))
    surface = prepare_surface((51005, 147), result["world_positions_f32"], patch["faces"])
    assert surface.closed and surface.self_audit()["count"] == 0
    assert result["ancestry"]["positive_birth_ancestry"][0]["parent_source_faces"] == [0, 2]
    assert result["ancestry"]["anatomical_admission"] is False
    assert result["ancestry"]["native_admission"] is False


@pytest.mark.parametrize("mutation,match", [
    ("off_source", "source face"),
    ("reverse", "reverses source orientation"),
    ("area", "duplicates source face area"),
    ("identity", "source face identity"),
    ("untyped", "exact rational metres"),
])
def test_rejects_invalid_source_ancestry_before_inversion(mutation, match):
    row, poses = fixture(); value = boundary(row, poses)
    if mutation == "off_source":
        value["output_mesh"]["vertices_m"][4] = (Q(2), Q(2), Q(2))
    elif mutation == "reverse":
        value["output_mesh"]["triangles"][0] = [0, 1, 4]
    elif mutation == "area":
        value["output_mesh"]["triangles"].append(FACES[0].tolist())
        value["output_face_ancestry"].append(dict(source_face=0))
    elif mutation == "identity":
        value["output_face_ancestry"][0]["source_face"] = True
    elif mutation == "untyped":
        value["output_mesh"]["vertices_m"][4] = (.125, .375, .0625)
    with pytest.raises(ValueError, match=match):
        emit_source_face_boundary(row, poses, value)


def test_never_truncates_attachment_coefficients():
    row, poses = fixture()
    row["bindings"] = [dict(core=k, translation=np.zeros(3, dtype="<f4"),
                           quaternion=np.array([0, 0, 0, 1], dtype="<f4"), scale=np.float32(1))
                       for k in range(5)]
    row["binding_count"] = 5
    row["binding_indices"][:] = [0, 1, 2, 3]
    row["binding_indices"][2] = [1, 2, 3, 4]
    row["weights"][:] = np.float32(.25)
    poses = {k: np.array([0, 0, 0, 0, 0, 0, 1], dtype="<f4") for k in range(5)}
    with pytest.raises(ValueError, match="never truncated"):
        emit_source_face_boundary(row, poses, boundary(row, poses))


def test_coincident_source_owner_conflict_is_not_silently_welded():
    row, poses = fixture()
    poses[1] = poses[0].copy()
    value = boundary(row, poses)
    row["positions"] = np.concatenate([row["positions"], row["positions"][:1]])
    row["normals"] = np.concatenate([row["normals"], row["normals"][:1]])
    row["binding_indices"] = np.concatenate([row["binding_indices"], row["binding_indices"][2:3]])
    row["weights"] = np.concatenate([row["weights"], row["weights"][:1]])
    row["vertex_count"] = 5
    with pytest.raises(ValueError, match="different attachment owners"):
        emit_source_face_boundary(row, poses, value)


def test_declared_inferred_cover_keeps_distinct_source_provenance():
    row, poses = fixture()
    poses[1] = poses[0].copy()
    world = pc._native_blended_rest_positions(row, row["positions"], poses)
    points = [tuple(Q.from_float(float(x)) for x in p) for p in world]
    center, radius = points[0], Q(1, 20000)
    cover_points = [tuple(center[a] + radius*signs[a] for a in range(3))
                    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    cover_faces = [[4, 5, 6], [4, 7, 5], [4, 6, 7], [5, 7, 6]]
    # A constructor may select any source boundary subset. This fixture tests
    # explicit inferred-parent emission; anatomical selection is a separate gate.
    value = dict(output_mesh=dict(vertices_m=points+cover_points, triangles=cover_faces),
                 output_face_ancestry=[dict(source_face=i) for i in range(4, 8)])
    with pytest.raises(ValueError, match="source face identity"):
        emit_source_face_boundary(row, poses, value)
    result = emit_source_face_boundary(row, poses, value, inferred_covers=[
        dict(exact_center=center, coordinate_radius_m=.00005)])
    np.testing.assert_array_equal(result["patch_arrays"]["vertices6"][:4, :3], row["positions"])
    proof = result["ancestry"]["inferred_repair_material"][0]
    assert proof["original_source_vertex_ancestry"] == 0
    assert proof["source_measurement"] is False
    assert proof["first_augmented_source_face"] == 4


@pytest.mark.parametrize("cover,match", [
    (dict(exact_center=(Q(99), Q(0), Q(0)), coordinate_radius_m=.00005), "source vertex ancestry"),
    (dict(exact_center=(Q(1, 2), Q(-1, 2), Q(1, 4)), coordinate_radius_m=.001), "construction bound"),
])
def test_inferred_cover_must_bind_existing_source_and_declared_size(cover, match):
    row, poses = fixture()
    with pytest.raises(ValueError, match=match):
        emit_source_face_boundary(row, poses, boundary(row, poses), inferred_covers=[cover])

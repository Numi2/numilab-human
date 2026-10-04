import numpy as np

from numilab_human.skin_full_solid_source_seeded_motion import source_seeded_extend


def test_source_seeded_extension_preserves_dirichlet_rows_and_uses_inner_seeds():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.1],
    ])
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    fixed_ids = np.array([0, 1, 2])
    fixed = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    point = vertices[3]+[0.0, 0.0, 0.001]
    bones = [
        {"binding_index": 0, "centroid_world_m": point,
         "sample_world_m": point[None, :],
         "sample_source_vertex_ids": np.array([4]), "diameter_bound_m": 2.0},
        {"binding_index": 1, "centroid_world_m": point,
         "sample_world_m": point[None, :],
         "sample_source_vertex_ids": np.array([5]), "diameter_bound_m": 2.0},
    ]

    candidate, metrics = source_seeded_extend(
        vertices, faces, fixed_ids, fixed, bones,
    )

    assert np.array_equal(candidate[fixed_ids], fixed)
    assert np.all(candidate >= 0.0)
    assert np.allclose(candidate.sum(axis=1), 1.0, atol=1e-12)
    assert metrics["screened_unknown_seed_vertex_count"] == 1
    assert metrics["screened_unknown_seed_body_indices"] == [0, 1]

import csv

import numpy as np

from numilab_human.common_atlas_skin_clearance import (
    _EXPECTED_SURFACE_COUNTS,
    _apply_captured_world_delta,
    _bounded_direction_projection,
    _candidate_outward_winding,
    _load_target_inventory,
    _smooth_vertex_directions,
    _triangle_normal_translation_to_separate,
)
from numilab_human.model import ImportError



def test_zero_world_delta_preserves_accepted_float32_vertex_bits():
    captured = np.array([[0.12345679, -0.25, 1.0000001], [0.5, 0.75, -0.875]], dtype="<f4").astype(np.float64)
    delta = np.array([[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0]], dtype=np.float64)

    candidate = _apply_captured_world_delta(captured, delta)

    assert candidate[0].astype("<f4").tobytes() == captured[0].astype("<f4").tobytes()
    assert np.array_equal(candidate[0], captured[0])
    assert candidate[1, 0] > captured[1, 0]

def test_finite_triangle_footprint_uses_first_separating_translation():
    skin = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = np.array([[0.1, 0.1, -1.0], [0.1, 0.2, 1.0], [-10.0, 0.1, 10.0]])
    plane_projection = float(target[:, 2].max())

    translation = _triangle_normal_translation_to_separate(skin, target, np.array([0.0, 0.0, 1.0]))

    assert 1.08 < translation < 1.10
    assert plane_projection - translation > 8.8


def test_coplanar_overlapping_triangles_need_only_positive_offset_after_first_exit():
    skin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.1, 0.2, 0.0]])

    translation = _triangle_normal_translation_to_separate(skin, target, np.array([0.0, 0.0, 1.0]))

    assert translation == 0.0
    assert translation + 0.25e-3 > translation


def test_near_parallel_intersection_has_finite_small_translation():
    skin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[0.2, 0.2, -1.0e-8], [0.4, 0.2, 1.0e-8], [0.2, 0.6, -1.0e-8]])

    translation = _triangle_normal_translation_to_separate(skin, target, np.array([0.0, 0.0, 1.0]))

    assert np.isfinite(translation)
    assert 0.0 <= translation <= 1.0e-7


def test_parallel_zero_direction_axis_fails_closed_for_disjoint_input():
    skin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0]])

    with np.testing.assert_raises_regex(ImportError, "already disjoint|zero-direction"):
        _triangle_normal_translation_to_separate(skin, target, np.array([0.0, 0.0, 1.0]))



def test_limited_direction_smoothing_reduces_local_turn_without_reversing_outward_normals():
    half = float(np.sqrt(0.5))
    normals = np.array([[0.866025403784, 0.5, 0.0], [0.866025403784, 0.5, 0.0], [0.5, 0.866025403784, 0.0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    smoothed = _smooth_vertex_directions(normals, faces)

    assert np.allclose(np.linalg.norm(smoothed, axis=1), 1.0)
    assert np.all(np.einsum("ij,ij->i", smoothed, normals) > 0.0)
    assert np.linalg.norm(smoothed[0] - smoothed[2]) < np.linalg.norm(normals[0] - normals[2])
    outward_face_normal = np.array([half, half, 0.0])
    assert float((smoothed @ outward_face_normal).min()) > 0.8


def test_direction_smoothing_fails_closed_for_empty_mesh():
    with np.testing.assert_raises_regex(ImportError, "invalid input"):
        _smooth_vertex_directions(np.array([[1.0, 0.0, 0.0]]), np.empty((0, 3), dtype=np.int64))

def test_inventory_keeps_eye_hits_as_monitors_outside_nonocular_clearance(tmp_path):
    rows = []
    for semantic, population in _EXPECTED_SURFACE_COUNTS.items():
        for index in range(population):
            stable_id = (381 + index if semantic == 51010 and index < 17 else 10000 + index)
            pair_count = 0
            if semantic == 51004 and index == 0:
                pair_count = 4448
            if semantic == 51010 and index < 17:
                pair_count = 260 if index < 16 else 255
            rows.append({
                "second_semantic": semantic,
                "second_stable_id": stable_id,
                "source_owner_or_label": "fixture",
                "face_count_skin": 0,
                "face_count_other": 0,
                "aabb_candidate_pairs": pair_count,
                "intersecting_triangle_pairs": pair_count,
                "audit_complete": "True",
            })
    path = tmp_path / "inventory.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    all_keys, clearance_keys, counts, populations = _load_target_inventory(path)

    assert len(all_keys) == 859
    assert len(clearance_keys) == 842
    assert len(all_keys - clearance_keys) == 17
    assert sum(counts.values()) == 8863
    assert sum(counts[key] for key in clearance_keys) == 4448
    assert sum(counts[key] for key in all_keys - clearance_keys) == 4415
    assert populations == _EXPECTED_SURFACE_COUNTS


def _closed_cube():
    vertices = np.array([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return vertices, faces


def test_rotated_candidate_normal_uses_current_full_shell_winding():
    vertices, faces = _closed_cube()
    angle = np.deg2rad(63.6)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    candidate = (vertices @ rotation.T).astype("<f4").astype(np.float64)
    base_triangles = vertices[faces]
    candidate_triangles = candidate[faces]
    base_normals = np.cross(
        base_triangles[:, 1] - base_triangles[:, 0],
        base_triangles[:, 2] - base_triangles[:, 0],
    )
    candidate_normals = np.cross(
        candidate_triangles[:, 1] - candidate_triangles[:, 0],
        candidate_triangles[:, 2] - candidate_triangles[:, 0],
    )
    dots = np.einsum("ij,ij->i", base_normals, candidate_normals) / (
        np.linalg.norm(base_normals, axis=1) * np.linalg.norm(candidate_normals, axis=1)
    )
    face = int(np.argmin(np.abs(dots - np.cos(angle))))
    assert 0.0 < dots[face] < 0.75

    normal = candidate_normals[face] / np.linalg.norm(candidate_normals[face])
    result = _candidate_outward_winding(candidate_triangles[face].mean(axis=0), normal, candidate_triangles)
    projections = _bounded_direction_projection(normal[None, :], normal)

    assert result["sample_offset_mm"] == 0.5
    assert result["absolute_inside_minus_outside_contrast"] > 0.5
    assert projections.min() == 1.0


def test_candidate_winding_rejects_ambiguous_open_shell_face():
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    with np.testing.assert_raises_regex(ImportError, "ambiguous or inward"):
        _candidate_outward_winding(triangle[0].mean(axis=0), np.array([0.0, 0.0, 1.0]), triangle)


def test_outward_but_ill_conditioned_displacement_direction_is_rejected():
    angle = np.deg2rad(63.6)
    current_normal = np.array([np.sin(angle), 0.0, np.cos(angle)])
    assert 0.0 < current_normal[2] < 0.5
    with np.testing.assert_raises_regex(ImportError, "maximum_normal_conversion=2"):
        _bounded_direction_projection(np.array([[0.0, 0.0, 1.0]]), current_normal)

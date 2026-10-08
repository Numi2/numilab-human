import csv

import numpy as np

from numilab_human.common_atlas_skin_clearance import (
    _EXPECTED_SURFACE_COUNTS,
    _load_target_inventory,
    _smooth_vertex_directions,
    _triangle_normal_translation_to_separate,
)
from numilab_human.model import ImportError


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



def test_one_ring_direction_smoothing_reduces_local_turn_without_reversing_outward_normals():
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


def test_one_ring_direction_smoothing_fails_closed_for_empty_mesh():
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

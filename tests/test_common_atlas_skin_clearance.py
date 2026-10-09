import csv

import numpy as np

from numilab_human.common_atlas_skin_clearance import (
    _EXPECTED_SURFACE_COUNTS,
    _apply_captured_world_delta,
    _area_weighted_vertex_normals,
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


def test_current_candidate_area_weighted_normals_follow_local_surface_geometry():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    normals = _area_weighted_vertex_normals(vertices, faces)

    expected_shared = np.array([2.0, 0.0, 1.0]) / np.sqrt(5.0)
    assert np.allclose(normals[0], expected_shared)
    assert np.allclose(normals[1], [0.0, 0.0, 1.0])
    assert np.allclose(normals[3], [1.0, 0.0, 0.0])

    curved = vertices.copy()
    curved[3] = [0.5, 0.0, 1.5]
    current_normals = _area_weighted_vertex_normals(curved, faces)
    assert np.isfinite(current_normals).all()
    assert not np.allclose(current_normals[0], normals[0])


def test_current_candidate_area_weighted_normals_reject_degenerate_mesh():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with np.testing.assert_raises_regex(ImportError, "area-weighted vertex normals are degenerate"):
        _area_weighted_vertex_normals(vertices, faces)



def test_direction_projection_accepts_exact_twofold_conversion_bound():
    direction = np.array([[np.sqrt(0.75), 0.0, 0.5]])
    projection = _bounded_direction_projection(direction, np.array([0.0, 0.0, 1.0]))
    assert projection[0] == 0.5


def test_shared_source_direction_pulls_back_and_averages_pose_normals():
    from numilab_human.common_atlas_skin_clearance import _shared_source_directions_from_pose_normals

    source = np.array([0.3, -0.4, 0.866025403784], dtype=np.float64)
    source /= np.linalg.norm(source)
    angle1, angle2 = np.deg2rad([73.0, -41.0])
    rotations = [
        np.eye(3),
        np.array([[np.cos(angle1), -np.sin(angle1), 0.0], [np.sin(angle1), np.cos(angle1), 0.0], [0.0, 0.0, 1.0]]),
        np.array([[np.cos(angle2), -np.sin(angle2), 0.0], [np.sin(angle2), np.cos(angle2), 0.0], [0.0, 0.0, 1.0]]),
    ]
    scales = [np.diag([1.0, 0.8, 1.2]), np.diag([1.1, 0.9, 1.0]), np.diag([0.9, 1.2, 1.1])]
    maps = np.asarray([rotation @ scale for rotation, scale in zip(rotations, scales)])[:, None, :, :]
    normals = np.einsum("pij,j->pi", maps[:, 0], source)[:, None, :]
    normals /= np.linalg.norm(normals, axis=2)[:, :, None]

    shared, mapped, alignment, metrics = _shared_source_directions_from_pose_normals(maps, normals)

    assert np.allclose(shared[0], source, atol=1.0e-12)
    assert np.allclose(alignment, 1.0, atol=1.0e-12)
    assert np.allclose(mapped[:, 0], normals[:, 0], atol=1.0e-12)
    assert metrics["minimum_pullback_resultant_coherence"] > 1.0 - 1.0e-12


def test_shared_source_direction_rejects_pose_cancellation_and_bad_jacobian():
    from numilab_human.common_atlas_skin_clearance import _shared_source_directions_from_pose_normals

    cancelling_maps = np.repeat(np.eye(3)[None, None, :, :], 2, axis=0)
    cancelling_normals = np.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]])
    with np.testing.assert_raises_regex(ImportError, "cancel"):
        _shared_source_directions_from_pose_normals(cancelling_maps, cancelling_normals)

    ill_conditioned = np.diag([2.1, 1.0, 1.0])[None, None, :, :]
    normal = np.array([[[1.0, 0.0, 0.0]]])
    with np.testing.assert_raises_regex(ImportError, "ill-conditioned"):
        _shared_source_directions_from_pose_normals(ill_conditioned, normal)


def test_source_scalar_demand_uses_actual_world_motion_per_source_metre():
    from numilab_human.common_atlas_skin_clearance import _source_scalar_for_normal_demand

    jacobian = np.diag([2.0, 1.0, 0.5])
    source_direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    source_distance, metrics = _source_scalar_for_normal_demand(
        jacobian, source_direction, normal, 0.25e-3,
    )

    assert source_distance == 0.125e-3
    assert metrics["world_normal_motion_per_source_m"] == 2.0
    assert metrics["world_direction_normal_projection"] == 1.0

    oblique = np.array([0.49, np.sqrt(1.0 - 0.49**2), 0.0])
    with np.testing.assert_raises_regex(ImportError, "boundedly conditioned"):
        _source_scalar_for_normal_demand(np.eye(3), oblique, normal, 1.0e-3)


def test_shared_source_seed_demand_takes_worst_pose_and_preserves_source_units():
    from numilab_human.common_atlas_skin_clearance import _shared_source_seed_demands

    maps = np.asarray([
        np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        np.repeat((0.5 * np.eye(3))[None, :, :], 3, axis=0),
    ])
    source = np.repeat(np.array([[1.0, 0.0, 0.0]]), 3, axis=0)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    normals = np.repeat(np.array([[[1.0, 0.0, 0.0]]]), 2, axis=0)
    demands = np.array([[0.001], [0.003]])

    seed, metrics = _shared_source_seed_demands(maps, source, faces, normals, demands)

    assert np.array_equal(seed, np.array([0.006, 0.006, 0.006]))
    assert metrics["active_face_count"] == 1
    assert metrics["minimum_active_projection"] == 1.0
    assert metrics["maximum_source_seed_demand_m"] == 0.006


def test_shared_source_seed_demand_rejects_inward_or_unbounded_active_face():
    from numilab_human.common_atlas_skin_clearance import _shared_source_seed_demands

    faces = np.array([[0, 1, 2]], dtype=np.int64)
    source = np.repeat(np.array([[1.0, 0.0, 0.0]]), 3, axis=0)
    map_batch = np.repeat(np.eye(3)[None, :, :], 3, axis=0)[None, ...]
    inward = np.array([[[-1.0, 0.0, 0.0]]])
    demand = np.array([[0.001]])
    with np.testing.assert_raises_regex(ImportError, "boundedly conditioned"):
        _shared_source_seed_demands(map_batch, source, faces, inward, demand)



def test_multi_pose_geodesic_metric_uses_maximum_world_edge_length():
    from numilab_human.common_atlas_skin_clearance import _maximum_pose_edge_lengths

    positions = np.array([
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]],
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.01, 0.0]],
    ])
    edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)

    lengths = _maximum_pose_edge_lengths(positions, edges)

    assert np.allclose(lengths, [0.03, 0.02, np.sqrt(0.03**2 + 0.01**2)])

def test_shared_multipose_clearance_uses_one_source_field_and_exact_all_pose_rescans():
    from numilab_human.common_atlas_skin_clearance import (
        _audit_pair,
        _exact_surface_records,
        derive_shared_multipose_inferred_clearance,
    )

    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    maps = np.repeat(np.eye(3)[None, None, :, :], 6, axis=0).reshape(2, 3, 3, 3)
    maps[1, :, 2, 2] = 2.0
    captured = np.stack([source.astype(np.float64), source.astype(np.float64)])
    targets = {
        (0, "1:1"): np.array([[0.3, -0.1, -0.2], [0.3, 0.8, 0.2], [0.3, 0.1, -0.2]]),
        (0, "2:1"): np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]]),
        (1, "1:1"): np.array([[0.3, -0.1, -0.4], [0.3, 0.8, 0.4], [0.3, 0.1, -0.4]]),
        (1, "2:1"): np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]]),
    }
    targets = {key: value.astype("<f4").astype(np.float64) for key, value in targets.items()}

    scan_calls = []
    def scan(pose, skin_world):
        scan_calls.append(pose)
        skin_records = _exact_surface_records(skin_world, faces)
        rows = {}
        for key in ("1:1", "2:1"):
            target_records = _exact_surface_records(targets[(pose, key)], np.array([[0, 1, 2]]))
            pair_result = _audit_pair(skin_records, target_records, same_surface=False)
            rows[key] = {
                "triangle_pairs": pair_result["triangle_pairs"],
                "count": pair_result["count"],
            }
        return rows

    baseline = [scan(pose, captured[pose]) for pose in range(2)]
    def target_triangle(pose, key, row):
        assert row == 0
        return targets[(pose, key)]

    trial_events = []
    scan_progress_events = []
    result, report = derive_shared_multipose_inferred_clearance(
        source_positions=source,
        faces=faces,
        jacobians_by_pose=maps,
        accepted_skin_world_by_pose=captured,
        baseline_target_audits_by_pose=baseline,
        baseline_skin_self_pairs_by_pose=[0, 0],
        source_outward_face_signs=np.array([1], dtype=np.int8),
        scan_candidate_targets=scan,
        target_triangle_by_row=target_triangle,
        all_target_keys={"1:1", "2:1"},
        ocular_monitor_keys={"2:1"},
        fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
        bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        max_iterations=3,
        progress_callback=trial_events.append,
        scan_progress_callback=scan_progress_events.append,
    )

    assert report["final_nonocular_pair_count_by_pose"] == [0, 0]
    assert report["final_ocular_pair_count_by_pose"] == [0, 0]
    assert result[0, 2] > 0.15
    source_delta = result.astype(np.float64) - source.astype(np.float64)
    mapped_delta = np.einsum("pnij,nj->pni", maps, source_delta)
    assert np.allclose(mapped_delta[1, :, 2], 2.0 * mapped_delta[0, :, 2], atol=1.0e-7)
    assert scan_calls == [0, 1, 0, 1]
    assert trial_events[-1]["status"] == "accepted"
    assert trial_events[-1]["source_positions_f32"].shape == (3, 3)
    assert len(trial_events[-1]["source_positions_f32_sha256"]) == 64
    assert trial_events[-1]["nonocular_pair_counts_before_by_pose"] == [1, 1]
    assert trial_events[-1]["nonocular_pair_counts_candidate_by_pose"] == [0, 0]
    assert len(trial_events[-1]["target_audits_by_pose"]) == 2
    assert trial_events[-1]["target_audits_by_pose"][0]["1:1"]["triangle_pairs"] == []
    assert trial_events[-1]["target_audits_by_pose"][1]["1:1"]["count"] == 0
    assert [event["event"] for event in scan_progress_events] == [
        "trial_started", "pose_audit_complete", "pose_audit_complete",
    ]
    assert scan_progress_events[0]["source_positions_f32"].shape == source.shape
    assert scan_progress_events[1]["pose_index"] == 0
    assert scan_progress_events[2]["pose_index"] == 1
    assert scan_progress_events[2]["target_audit"]["1:1"]["count"] == 0
    assert report["qualification"]["native_replay"] == "pending"

    import hashlib

    resume_sha = hashlib.sha256(np.asarray(result, dtype="<f4").tobytes()).hexdigest()
    resumed, resume_report = derive_shared_multipose_inferred_clearance(
        source_positions=source,
        faces=faces,
        jacobians_by_pose=maps,
        accepted_skin_world_by_pose=captured,
        baseline_target_audits_by_pose=baseline,
        baseline_skin_self_pairs_by_pose=[0, 0],
        source_outward_face_signs=np.array([1], dtype=np.int8),
        scan_candidate_targets=scan,
        target_triangle_by_row=target_triangle,
        all_target_keys={"1:1", "2:1"},
        ocular_monitor_keys={"2:1"},
        fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
        bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        max_iterations=1,
        resume_source_positions=result,
        resume_target_audits_by_pose=trial_events[-1]["target_audits_by_pose"],
        resume_provenance={
            "source_positions_path": "resume-fixture-source.npy",
            "source_positions_f32_sha256": resume_sha,
            "target_audits_path": "resume-fixture-audits.json",
            "target_audits_sha256": "a" * 64,
        },
    )
    assert np.array_equal(resumed.astype("<f4"), result.astype("<f4"))
    assert resume_report["initial_nonocular_pair_count_by_pose"] == [1, 1]
    assert resume_report["starting_nonocular_pair_count_by_pose"] == [0, 0]
    assert resume_report["iterations"] == []
    assert resume_report["resume_start"]["status"] == (
        "caller-hash-bound-accepted-candidate-revalidated-against-immutable-baseline-gates"
    )


def test_shared_multipose_clearance_refuses_to_move_fixed_support_seeds():
    from numilab_human.common_atlas_skin_clearance import (
        _audit_pair,
        _exact_surface_records,
        derive_shared_multipose_inferred_clearance,
    )

    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    maps = np.repeat(np.eye(3)[None, None, :, :], 6, axis=0).reshape(2, 3, 3, 3)
    captured = np.stack([source.astype(np.float64), source.astype(np.float64)])
    target = np.array([[0.3, -0.1, -0.2], [0.3, 0.8, 0.2], [0.3, 0.1, -0.2]]).astype("<f4").astype(np.float64)
    far = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]]).astype("<f4").astype(np.float64)

    def scan(pose, skin_world):
        skin = _exact_surface_records(skin_world, faces)
        rows = {}
        for key, other in (("1:1", target), ("2:1", far)):
            pair = _audit_pair(skin, _exact_surface_records(other, faces), same_surface=False)
            rows[key] = {"triangle_pairs": pair["triangle_pairs"], "count": pair["count"]}
        return rows

    baseline = [scan(pose, captured[pose]) for pose in range(2)]
    with np.testing.assert_raises_regex(ImportError, "no movable correction field"):
        derive_shared_multipose_inferred_clearance(
            source_positions=source, faces=faces, jacobians_by_pose=maps,
            accepted_skin_world_by_pose=captured, baseline_target_audits_by_pose=baseline,
            baseline_skin_self_pairs_by_pose=[0, 0], source_outward_face_signs=np.array([1], dtype=np.int8),
            scan_candidate_targets=scan,
            target_triangle_by_row=lambda pose, key, row: target if key == "1:1" else far,
            all_target_keys={"1:1", "2:1"}, ocular_monitor_keys={"2:1"},
            fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
            preserved_source_anchor_vertex_ids=np.array([0, 1, 2], dtype=np.int64),
            bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            max_iterations=1,
        )



def test_source_face_outward_basis_propagates_by_shared_edge_topology():
    from numilab_human.common_atlas_skin_clearance import _propagate_source_face_orientation

    consistent = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    assert np.array_equal(_propagate_source_face_orientation(consistent, np.array([0])), [1, 1])

    one_reversed = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    assert np.array_equal(_propagate_source_face_orientation(one_reversed, np.array([0])), [1, -1])


def test_source_face_outward_basis_rejects_unseeded_components_and_nonmanifold_edges():
    from numilab_human.common_atlas_skin_clearance import _propagate_source_face_orientation

    disconnected = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    with np.testing.assert_raises_regex(ImportError, "disconnected face component"):
        _propagate_source_face_orientation(disconnected, np.array([0]))

    nonmanifold = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    with np.testing.assert_raises_regex(ImportError, "incident faces"):
        _propagate_source_face_orientation(nonmanifold, np.array([0]))


def test_shared_multipose_clearance_recomputes_candidate_forward_map_each_trial():
    from numilab_human.common_atlas_skin_clearance import (
        _audit_pair,
        _exact_surface_records,
        derive_shared_multipose_inferred_clearance,
    )

    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    captured = np.stack([source.astype(np.float64), source.astype(np.float64)])
    targets = {
        (0, "1:1"): np.array([[0.3, -0.1, -0.2], [0.3, 0.8, 0.2], [0.3, 0.1, -0.2]]),
        (0, "2:1"): np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]]),
        (1, "1:1"): np.array([[0.3, -0.1, -0.4], [0.3, 0.8, 0.4], [0.3, 0.1, -0.4]]),
        (1, "2:1"): np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]]),
    }
    targets = {key: value.astype("<f4").astype(np.float64) for key, value in targets.items()}
    forward_calls = []

    def forward(full_source):
        points = np.asarray(full_source, dtype="<f4").astype(np.float64)
        forward_calls.append(points.copy())
        world = np.repeat(points[None, :, :], 2, axis=0)
        maps = np.repeat(np.eye(3)[None, None, :, :], 2 * len(points), axis=0).reshape(2, len(points), 3, 3)
        for pose, curvature in enumerate((0.2, 0.4)):
            world[pose, :, 2] = points[:, 2] + curvature * points[:, 2] ** 2
            maps[pose, :, 2, 2] = 1.0 + 2.0 * curvature * points[:, 2]
        return {
            "world_positions_by_pose": world.astype("<f4"),
            "jacobians_by_pose": maps,
            "diagnostics": {"maximum_candidate_source_z_m": float(points[:, 2].max(initial=0.0))},
        }

    scan_calls = []

    def scan(pose, skin_world):
        scan_calls.append(pose)
        skin_records = _exact_surface_records(skin_world, faces)
        rows = {}
        for key in ("1:1", "2:1"):
            target_records = _exact_surface_records(targets[(pose, key)], faces)
            pair_result = _audit_pair(skin_records, target_records, same_surface=False)
            rows[key] = {"triangle_pairs": pair_result["triangle_pairs"], "count": pair_result["count"]}
        return rows

    baseline = [scan(pose, captured[pose]) for pose in range(2)]

    def target_triangle(pose, key, row):
        assert row == 0
        return targets[(pose, key)]

    result, report = derive_shared_multipose_inferred_clearance(
        source_positions=source,
        faces=faces,
        jacobians_by_pose=None,
        accepted_skin_world_by_pose=captured,
        baseline_target_audits_by_pose=baseline,
        baseline_skin_self_pairs_by_pose=[0, 0],
        source_outward_face_signs=np.array([1], dtype=np.int8),
        scan_candidate_targets=scan,
        target_triangle_by_row=target_triangle,
        all_target_keys={"1:1", "2:1"},
        ocular_monitor_keys={"2:1"},
        fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
        bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        max_iterations=3,
        candidate_forward=forward,
    )

    assert report["final_nonocular_pair_count_by_pose"] == [0, 0]
    assert report["candidate_forward_model"]["mode"] == "recomputed_candidate_callback"
    assert report["candidate_forward_model"]["baseline_replay_max_vertex_error_m"] == 0.0
    assert result[0, 2] > 0.15
    assert len(forward_calls) >= 2
    assert any(float(points[:, 2].max()) > 0.0 for points in forward_calls[1:])
    assert report["candidate_forward_model"]["final_diagnostics"]["maximum_candidate_source_z_m"] > 0.0
    assert scan_calls[:2] == [0, 1]


def test_resume_candidate_is_reaudited_for_exact_skin_self_intersections():
    import hashlib
    from numilab_human.common_atlas_skin_clearance import (
        _audit_pair,
        _exact_surface_records,
        derive_shared_multipose_inferred_clearance,
    )

    source = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.2, 0.2, 1.0], [0.8, 0.2, 1.0], [0.2, 0.8, 1.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    maps = np.repeat(np.eye(3)[None, None, :, :], 12, axis=0).reshape(2, 6, 3, 3)
    captured = np.stack([source.astype(np.float64), source.astype(np.float64)])
    target = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]])

    def scan(pose, skin_world):
        skin_records = _exact_surface_records(skin_world, faces)
        target_records = _exact_surface_records(target, np.array([[0, 1, 2]], dtype=np.int64))
        audit = _audit_pair(skin_records, target_records, same_surface=False)
        row = {
            "triangle_pairs": audit["triangle_pairs"],
            "count": int(audit["count"]),
            "aabb_candidate_pairs": int(audit["aabb_candidate_pairs"]),
            "degenerate_face_rows": [],
        }
        return {"1:1": row, "2:1": dict(row)}

    baseline = [scan(pose, captured[pose]) for pose in range(2)]
    assert all(row["1:1"]["count"] == 0 for row in baseline)
    resume = source.copy()
    resume[3:, 2] = 0.0
    resumed_records = _exact_surface_records(resume.astype(np.float64), faces)
    assert _audit_pair(resumed_records, resumed_records, same_surface=True)["count"] == 1
    resume_sha = hashlib.sha256(np.asarray(resume, dtype="<f4").tobytes()).hexdigest()

    with np.testing.assert_raises_regex(ImportError, "resume candidate has 1 exact skin self-intersections"):
        derive_shared_multipose_inferred_clearance(
            source_positions=source,
            faces=faces,
            jacobians_by_pose=maps,
            accepted_skin_world_by_pose=captured,
            baseline_target_audits_by_pose=baseline,
            baseline_skin_self_pairs_by_pose=[0, 0],
            source_outward_face_signs=np.array([1, 1], dtype=np.int8),
            scan_candidate_targets=scan,
            target_triangle_by_row=lambda pose, key, row: target,
            all_target_keys={"1:1", "2:1"},
            ocular_monitor_keys={"2:1"},
            fixed_source_vertex_ids=np.empty(0, dtype=np.int64),
            bed_plane_origins_by_pose=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            bed_plane_normals_by_pose=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            max_iterations=1,
            resume_source_positions=resume,
            resume_target_audits_by_pose=baseline,
            resume_provenance={
                "source_positions_path": "overlapping-resume-source.npy",
                "source_positions_f32_sha256": resume_sha,
                "target_audits_path": "overlapping-resume-audits.json",
                "target_audits_sha256": "b" * 64,
            },
        )


def test_collateral_fold_taper_smooths_nonseed_motion_and_preserves_required_seed():
    from scipy.sparse import csr_matrix
    from numilab_human.common_atlas_skin_clearance import _smooth_collateral_displacement_near_faces

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    edge_rows = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge_rows = np.unique(np.sort(edge_rows, axis=1), axis=0)
    rows = np.concatenate((edge_rows[:, 0], edge_rows[:, 1]))
    cols = np.concatenate((edge_rows[:, 1], edge_rows[:, 0]))
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(4, 4))

    displacement = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.1, 0.0],
    ])
    raw_triangle = (points + displacement)[faces[0]]
    raw_normal = np.cross(raw_triangle[1] - raw_triangle[0], raw_triangle[2] - raw_triangle[0])
    assert raw_normal[2] < 0.0

    regularized, report = _smooth_collateral_displacement_near_faces(
        displacement,
        graph,
        faces,
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    candidate_triangle = (points + regularized)[faces[0]]
    candidate_normal = np.cross(
        candidate_triangle[1] - candidate_triangle[0],
        candidate_triangle[2] - candidate_triangle[0],
    )
    assert candidate_normal[2] > 0.0
    assert np.array_equal(regularized[3], displacement[3])
    assert np.array_equal(regularized[[0, 1, 2]], np.zeros((3, 3)))
    assert report["zero_increment_fold_core_vertex_count"] == 3
    assert report["fold_core_increment_exactly_zero"] is True
    assert report["required_seed_values_bitwise_preserved"] is True
    assert report["folded_face_rows"] == [0]
    assert report["maximum_increment_change_m"] > 0.0

    with np.testing.assert_raises_regex(ImportError, "touches a required active correction seed"):
        _smooth_collateral_displacement_near_faces(
            displacement,
            graph,
            faces,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

def test_active_face_conditioning_preserves_all_admissible_directions_exactly():
    from numilab_human.common_atlas_skin_clearance import _condition_shared_source_directions

    maps = np.tile(np.eye(3), (2, 4, 1, 1))
    source = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.0], [1.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    normals = np.tile([0.0, 0.0, 1.0], (2, 1, 1))
    original = source.copy()
    result, report = _condition_shared_source_directions(maps, source, faces, normals)
    assert np.array_equal(result, original)
    assert np.array_equal(source, original)
    assert report["conditioned_vertex_count"] == 0
    assert report["unchanged_admission_projection"] == 0.5


def test_active_face_conditioning_resolves_all_incident_faces_and_pose_scales():
    from numilab_human.common_atlas_skin_clearance import (
        _condition_shared_source_directions, _shared_source_seed_demands,
    )

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second_map = rotation @ np.diag([0.9, 1.1, 1.2])
    maps = np.stack([np.tile(np.eye(3), (5, 1, 1)), np.tile(second_map, (5, 1, 1))])
    # Vertex 0 is shared by both active faces. Other vertices already satisfy
    # their own face constraints and must not change.
    source = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    normals = np.array([[[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]],
                        [[0.0, 0.0, 1.0], [-0.6, 0.0, 0.8]]])
    demands = np.array([[0.001, 0.002], [0.003, 0.004]])
    result, report = _condition_shared_source_directions(maps, source, faces, normals)
    seed, verified = _shared_source_seed_demands(maps, result, faces, normals, demands)

    assert np.array_equal(result[1:], source[1:])
    assert report["conditioned_compact_vertex_ids"] == [0]
    assert np.isclose(np.linalg.norm(result[0]), 1.0, atol=1.0e-12)
    assert verified["minimum_active_projection"] >= 0.5
    # Check actual metre displacement, independently of the seed owner.
    for pose in range(2):
        for face_index, face in enumerate(faces):
            for vertex in face:
                moved = maps[pose, vertex] @ (result[vertex] * seed[vertex])
                assert np.dot(moved, normals[pose, face_index]) >= demands[pose, face_index] - 1.0e-12


def test_active_face_conditioning_rejects_opposite_pose_cones():
    from numilab_human.common_atlas_skin_clearance import _condition_shared_source_directions

    maps = np.tile(np.eye(3), (2, 3, 1, 1))
    source = np.tile([0.0, 0.0, 1.0], (3, 1))
    faces = np.array([[0, 1, 2]])
    normals = np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]])
    with np.testing.assert_raises_regex(ImportError, "no nonzero converged direction|unchanged active-face bound"):
        _condition_shared_source_directions(maps, source, faces, normals)


def test_active_face_conditioning_handles_empty_faces_and_rejects_zero_normals():
    from numilab_human.common_atlas_skin_clearance import _condition_shared_source_directions

    maps = np.tile(np.eye(3), (2, 3, 1, 1))
    source = np.tile([0.0, 0.0, 1.0], (3, 1))
    result, report = _condition_shared_source_directions(
        maps, source, np.empty((0, 3), dtype=int), np.empty((2, 0, 3)),
    )
    assert np.array_equal(result, source)
    assert report["minimum_active_projection_after"] is None
    with np.testing.assert_raises_regex(ImportError, "zero direction or normal"):
        _condition_shared_source_directions(maps, source, np.array([[0, 1, 2]]), np.zeros((2, 1, 3)))

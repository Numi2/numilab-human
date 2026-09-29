"""Analytical invariants for geometry metrics, independent of skin authoring."""
import numpy as np
import pytest

from numilab_human.skin_geometry_diagnostics import (
    source_seam_diagnostics, source_surface_topology, surface_face_diagnostics,
)


@pytest.mark.parametrize('rotation', [np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])])
def test_surface_stretch_is_analytic_and_rigid_frame_invariant(rotation):
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    deformed = (source * [2., 3., 1.]) @ rotation.T + [11., -3., 4.]
    normals = np.tile([0., 0., 1.], (3, 1)) @ rotation.T
    report = surface_face_diagnostics(source, [[0, 1, 2]], deformed, normals)
    assert report['minimum_surface_stretch'] == pytest.approx(2.)
    assert report['maximum_surface_stretch'] == pytest.approx(3.)
    assert report['minimum_area_ratio'] == pytest.approx(6.)
    assert report['native_surface_area_m2'] == pytest.approx(3.)
    assert report['faces_opposed_to_rendered_vertex_normals'] == 0


def test_collapsed_native_triangle_is_exposed_without_dividing_by_its_area():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    native = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    report = surface_face_diagnostics(source, [[0, 1, 2]], native, np.tile([0., 0., 1.], (3, 1)))
    assert report['native_collapsed_face_count'] == 1
    assert report['minimum_area_ratio'] == 0
    assert report['minimum_surface_stretch'] == 0
    assert report['faces_area_below_0_1x'] == 1


def test_degenerate_source_triangle_is_reported_instead_of_claiming_a_surface_map():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    report = surface_face_diagnostics(source, [[0, 1, 2]], source, np.tile([0., 0., 1.], (3, 1)))
    assert report['source_degenerate_face_count'] == 1
    assert report['minimum_surface_stretch'] is None
    assert report['minimum_area_ratio'] is None


def test_near_points_are_not_seams_and_exact_seam_separation_is_measured():
    source = np.array([[0., 0., 0.], [0., 0., 0.], [1e-12, 0., 0.]])
    native = np.array([[0., 0., 0.], [0., .003, 0.], [8., 0., 0.]])
    report = source_seam_diagnostics(source, native)
    assert report['exact_coincident_vertex_group_count'] == 1
    assert report['exact_coincident_redundant_vertex_count'] == 1
    assert report['maximum_native_seam_gap_m'] == pytest.approx(.003)
    assert report['worst_source_seam_pairs'][0]['source_skin_vertex_ids'] == [0, 1]


def test_seam_gap_uses_every_pair_in_a_multi_vertex_group():
    source = np.zeros((3, 3))
    native = np.array([[0., 0., 0.], [-.003, 0., 0.], [.003, 0., 0.]])
    report = source_seam_diagnostics(source, native)
    assert report['source_seam_pair_count'] == 3
    assert report['maximum_native_seam_gap_m'] == pytest.approx(.006)
    assert report['worst_source_seam_pairs'][0]['source_skin_vertex_ids'] == [1, 2]


@pytest.mark.parametrize('closed', [False, True])
def test_topological_candidate_never_becomes_volume_or_physics_admission(closed):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] if closed else [[0, 1, 2]]
    report = source_surface_topology(points if closed else points[:3], faces)
    assert report['exact_coordinate_quotient']['closed_oriented_manifold_candidate'] is closed
    assert report['exact_coordinate_quotient']['boundary_edge_count'] == (0 if closed else 3)
    assert report['closed_volume_admitted'] is False
    assert report['self_intersection_status'] == 'not_checked'


def test_normal_opposition_is_a_separate_visual_diagnostic():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    report = surface_face_diagnostics(points, [[0, 1, 2]], points, np.tile([0., 0., -1.], (3, 1)))
    assert report['minimum_area_ratio'] == 1
    assert report['faces_opposed_to_rendered_vertex_normals'] == 1

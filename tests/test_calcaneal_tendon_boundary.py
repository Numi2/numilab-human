"""Keep a visual tendon insertion source-bound and free of invented faces."""

import copy

import pytest

from numilab_human.model import (
    ImportError as HumanImportError,
    _bodyparts_locked_tendon_boundary,
    _bodyparts_project_tendon_boundary_harmonic,
)


def test_named_distal_boundary_moves_with_source_topology_unchanged():
    vertices = [[0., 0., .002], [.01, 0., .002],
                [0., .01, .002], [.01, .01, .002]]
    faces = [(0, 1, 2), (1, 3, 2)]
    attenuation = [0., 0., 1., 1.]
    bone_vertices = [[-.01, -.01, 0.], [.02, -.01, 0.], [0., .02, 0.]]
    bone_faces = [(0, 1, 2)]
    original = copy.deepcopy((vertices, faces, attenuation, bone_vertices, bone_faces))

    fitted, receipt = _bodyparts_project_tendon_boundary_harmonic(
        vertices, faces, attenuation, bone_vertices, bone_faces, "synthetic_tendon",
        radius_m=.005,
    )
    repeated, repeated_receipt = _bodyparts_project_tendon_boundary_harmonic(
        vertices, faces, attenuation, bone_vertices, bone_faces, "synthetic_tendon",
        radius_m=.005,
    )

    assert (vertices, faces, attenuation, bone_vertices, bone_faces) == original
    assert (fitted, receipt) == (repeated, repeated_receipt)
    assert receipt["source_boundary_vertex_count"] == 2
    assert receipt["generated_triangle_count"] == 0
    assert receipt["source_indices_preserved"] is True
    assert fitted[0][2] == pytest.approx(-.00035)
    assert fitted[1][2] == pytest.approx(-.00035)
    assert fitted[2:] == vertices[2:]


def test_missing_distal_cap_and_invalid_projection_fail_closed():
    faces = [(0, 1, 2)]
    with pytest.raises(HumanImportError, match="no named distal source-cap boundary"):
        _bodyparts_locked_tendon_boundary(faces, [1., 1., 1.], 3, "missing")
    with pytest.raises(HumanImportError, match="invalid source triangle"):
        _bodyparts_locked_tendon_boundary([(0, 1, 3)], [0., 0., 0.], 3, "bad")
    with pytest.raises(HumanImportError, match="invalid boundary-projection inputs"):
        _bodyparts_project_tendon_boundary_harmonic(
            [[0., 0., .002], [.01, 0., .002], [0., .01, .002]], faces,
            [0., 0., 1.], [[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]],
            faces, "bad", inset_m=.002,
        )

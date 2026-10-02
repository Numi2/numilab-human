import pytest

from numilab_human.cardiac_cavity_intersections import _records
from numilab_human.muscle_surface_volume_disjointness import (
    MuscleDisjointnessError,
    _component_domain_status,
    _component_meshes,
    _member_pair_status,
    _surface_pair_status,
    _write_immutable,
)


TETRA_FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


def _mesh(vertices):
    scaled = [tuple(round(value * 1000) for value in point) for point in vertices]
    return {"vertices": scaled, "records": _records(scaled, TETRA_FACES)}


def _member_mesh(shells):
    vertices = []
    faces = []
    for shell in shells:
        offset = len(vertices)
        vertices.extend(
            tuple(round(value * 1000) for value in point) for point in shell
        )
        faces.extend([[offset + index for index in face] for face in TETRA_FACES])
    components, _ = _component_meshes(vertices, faces)
    return {
        "vertices": vertices,
        "faces": faces,
        "records": _records(vertices, faces),
        "components": components,
    }


def _translated_tetrahedron(x_offset):
    return [
        (x + x_offset, y, z) for x, y, z in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]


def test_exact_pair_aabb_separation_proves_disjoint_domains():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(2, 0, 0), (3, 0, 0), (2, 1, 0), (2, 0, 1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "strictly_disjoint_aabbs"
    assert result["intersection_pairs"] == 0


def test_exact_pair_reports_crossing_surface_intersection():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0.2, 0, 0), (1.2, 0, 0), (0.2, 1, 0), (0.2, 0, 1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "surface_intersection"
    assert result["intersection_pairs"] > 0


def test_exact_pair_fails_closed_on_coplanar_boundary_contact():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "surface_intersection"
    assert result["intersection_pairs"] > 0


def test_exact_pair_distinguishes_containment_without_surface_crossing():
    first = _mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    second = _mesh([(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)])

    result = _surface_pair_status(first, second)

    assert result["status"] == "nested_closed_domains"
    assert result["intersection_pairs"] == 0


def test_disconnected_closed_components_can_form_an_unambiguous_union():
    member = _member_mesh([_translated_tetrahedron(0), _translated_tetrahedron(6)])

    status, pair_counts, unresolved = _component_domain_status(member["components"])

    assert status == "disjoint_closed_component_union"
    assert pair_counts == {"strictly_disjoint_aabbs": 1}
    assert unresolved == []


def test_pair_audit_handles_unions_with_overlapping_global_bounds():
    first = _member_mesh([_translated_tetrahedron(0), _translated_tetrahedron(6)])
    second = _member_mesh([_translated_tetrahedron(3), _translated_tetrahedron(9)])

    result = _member_pair_status(first, second)

    assert result["status"] == "separate_closed_domains"
    assert result["intersection_pairs"] == 0
    assert result["containment"]["component_pairs_checked"] == 0


def test_nested_shell_components_are_withheld_from_domain_union():
    outer = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    inner = [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)]
    member = _member_mesh([outer, inner])

    status, pair_counts, unresolved = _component_domain_status(member["components"])

    assert status == "component_union_unresolved"
    assert pair_counts == {"nested_closed_domains": 1}
    assert unresolved[0]["status"] == "nested_closed_domains"


def test_immutable_receipt_writer_rejects_symlink_output(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve", encoding="utf-8")
    output = tmp_path / "receipt.json"
    output.symlink_to(target)

    with pytest.raises(MuscleDisjointnessError, match="symlink"):
        _write_immutable(output, {"schema": "test"})

    assert target.read_text(encoding="utf-8") == "preserve"

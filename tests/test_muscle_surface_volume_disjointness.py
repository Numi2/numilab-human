import pytest

from numilab_human.cardiac_cavity_intersections import _records
from numilab_human.muscle_surface_volume_disjointness import (
    MuscleDisjointnessError,
    _component_domain_status,
    _component_meshes,
    _isolated_candidate_subset,
    _member_pair_status,
    _surface_pair_status,
    _write_immutable,
)


TETRA_FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
REVERSED_TETRA_FACES = [[a, c, b] for a, b, c in TETRA_FACES]


def _mesh(vertices):
    scaled = [tuple(round(value * 1000) for value in point) for point in vertices]
    return {"vertices": scaled, "records": _records(scaled, TETRA_FACES)}


def _member_mesh(shells, shell_faces=None):
    vertices = []
    faces = []
    shell_faces = shell_faces or [TETRA_FACES] * len(shells)
    for shell, local_faces in zip(shells, shell_faces, strict=True):
        offset = len(vertices)
        vertices.extend(
            tuple(round(value * 1000) for value in point) for point in shell
        )
        faces.extend([[offset + index for index in face] for face in local_faces])
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


def test_isolated_subset_keeps_only_members_separated_from_the_full_pair_universe():
    surfaces = [
        {
            "stable_id": stable_id,
            "member_id": f"FJ{stable_id}",
            "closed_embedded_candidate": True,
            "compiled_geometry_sha256": f"{stable_id:064x}",
            "triangle_count": 4,
            "source_face_component_count": 1,
            "component_domain_status": "single_closed_component",
        }
        for stable_id in (1, 2, 3, 4)
    ]
    pairs = [
        {
            "first_stable_id": 1,
            "second_stable_id": 2,
            "status": "strictly_disjoint_aabbs",
        },
        {
            "first_stable_id": 1,
            "second_stable_id": 3,
            "status": "separate_closed_domains",
        },
        {
            "first_stable_id": 1,
            "second_stable_id": 4,
            "status": "strictly_disjoint_aabbs",
        },
        {
            "first_stable_id": 2,
            "second_stable_id": 3,
            "status": "separate_closed_domains",
        },
        {
            "first_stable_id": 2,
            "second_stable_id": 4,
            "status": "strictly_disjoint_aabbs",
        },
        {"first_stable_id": 3, "second_stable_id": 4, "status": "surface_intersection"},
    ]

    result = _isolated_candidate_subset(surfaces, pairs)

    assert [row["stable_id"] for row in result] == [1, 2]
    assert [row["member_id"] for row in result] == ["FJ1", "FJ2"]


def test_isolated_subset_fails_closed_on_an_incomplete_pair_matrix():
    surfaces = [
        {
            "stable_id": stable_id,
            "member_id": f"FJ{stable_id}",
            "closed_embedded_candidate": True,
            "compiled_geometry_sha256": f"{stable_id:064x}",
            "triangle_count": 4,
            "source_face_component_count": 1,
            "component_domain_status": "single_closed_component",
        }
        for stable_id in (1, 2, 3)
    ]
    pairs = [
        {
            "first_stable_id": 1,
            "second_stable_id": 2,
            "status": "strictly_disjoint_aabbs",
        },
        {
            "first_stable_id": 1,
            "second_stable_id": 3,
            "status": "separate_closed_domains",
        },
    ]

    with pytest.raises(MuscleDisjointnessError, match="complete pairwise"):
        _isolated_candidate_subset(surfaces, pairs)


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

    status, pair_counts, unresolved, shells = _component_domain_status(
        member["components"]
    )

    assert status == "disjoint_closed_component_union"
    assert pair_counts == {"strictly_disjoint_aabbs": 1}
    assert unresolved == []
    assert [row["role"] for row in shells] == ["solid_boundary", "solid_boundary"]


def test_pair_audit_handles_unions_with_overlapping_global_bounds():
    first = _member_mesh([_translated_tetrahedron(0), _translated_tetrahedron(6)])
    second = _member_mesh([_translated_tetrahedron(3), _translated_tetrahedron(9)])

    result = _member_pair_status(first, second)

    assert result["status"] == "separate_closed_domains"
    assert result["intersection_pairs"] == 0
    assert result["containment"]["component_pairs_checked"] == 0


def test_domain_inside_an_oriented_cavity_is_not_classified_as_nested():
    outer = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cavity = [
        (0.1, 0.1, 0.1),
        (0.2, 0.1, 0.1),
        (0.1, 0.2, 0.1),
        (0.1, 0.1, 0.2),
    ]
    first = _member_mesh([outer, cavity], [TETRA_FACES, REVERSED_TETRA_FACES])
    second = _member_mesh(
        [
            [
                (0.12, 0.12, 0.12),
                (0.14, 0.12, 0.12),
                (0.12, 0.14, 0.12),
                (0.12, 0.12, 0.14),
            ]
        ]
    )

    result = _member_pair_status(first, second)

    assert result["status"] == "separate_closed_domains"
    assert result["intersection_pairs"] == 0


def test_nested_oppositely_oriented_shell_is_admitted_as_a_cavity():
    outer = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    inner = [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)]
    member = _member_mesh([outer, inner], [TETRA_FACES, REVERSED_TETRA_FACES])

    status, pair_counts, unresolved, shells = _component_domain_status(
        member["components"]
    )

    assert status == "closed_shell_domain_with_cavities"
    assert pair_counts == {"nested_closed_domains": 1}
    assert unresolved == []
    assert [(row["role"], row["containment_depth"]) for row in shells] == [
        ("solid_boundary", 0),
        ("cavity_boundary", 1),
    ]


def test_nested_same_orientation_shell_is_withheld():
    outer = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    inner = [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)]
    member = _member_mesh([outer, inner])

    status, pair_counts, unresolved, shells = _component_domain_status(
        member["components"]
    )

    assert status == "component_shell_domain_unresolved"
    assert pair_counts == {"nested_closed_domains": 1}
    assert unresolved[0]["status"] == "shell_orientation_mismatch"
    assert shells == []


def test_immutable_receipt_writer_rejects_symlink_output(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("preserve", encoding="utf-8")
    output = tmp_path / "receipt.json"
    output.symlink_to(target)

    with pytest.raises(MuscleDisjointnessError, match="symlink"):
        _write_immutable(output, {"schema": "test"})

    assert target.read_text(encoding="utf-8") == "preserve"

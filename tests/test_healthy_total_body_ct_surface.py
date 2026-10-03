from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from numilab_human.healthy_total_body_ct_surface import (
    PLAN_SCHEMA,
    _nonmanifold_vertex_count,
    _read_trial_plan,
    _write_binary_ply_gzip,
    build_voxel_boundary_mesh,
)
from numilab_human.model import ImportError


def _reference_nonmanifold_vertex_count(triangles, vertex_count: int) -> int:
    links = [[] for _ in range(vertex_count)]
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    invalid = 0
    for link_edges in links:
        adjacency = {}
        for left, right in link_edges:
            adjacency.setdefault(left, []).append(right)
            adjacency.setdefault(right, []).append(left)
        if not adjacency or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            invalid += 1
            continue
        first = next(iter(adjacency))
        visited, stack = {first}, [first]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        invalid += len(visited) != len(adjacency)
    return invalid


def test_single_anisotropic_reflected_voxel_has_exact_closed_boundary() -> None:
    affine = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, -4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    vertices, triangles, metrics = build_voxel_boundary_mesh(
        np.ones((1, 1, 1), dtype=np.bool_),
        affine,
        voxel_offset_ijk=(4, 5, 6),
    )

    assert vertices.shape == (8, 3)
    assert triangles.shape == (12, 3)
    assert metrics["occupied_voxel_count"] == 1
    assert metrics["exposed_voxel_face_count"] == 6
    assert metrics["surface_area_mm2"] == pytest.approx(52.0)
    assert metrics["signed_volume_mm3"] == pytest.approx(24.0)
    assert metrics["edge_incidence"]["all_edges_have_two_incident_triangles"]
    assert metrics["nonmanifold_vertex_count"] == 0
    assert metrics["closed_two_manifold"]
    assert vertices.mean(axis=0) == pytest.approx([18.0, 35.0, 6.0])


def test_face_adjacent_voxels_form_one_exact_rectangular_boundary() -> None:
    affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    vertices, triangles, metrics = build_voxel_boundary_mesh(
        np.ones((2, 1, 1), dtype=np.bool_),
        affine,
    )

    assert vertices.shape == (12, 3)
    assert triangles.shape == (20, 3)
    assert metrics["exposed_voxel_face_count"] == 10
    assert metrics["surface_area_mm2"] == pytest.approx(10.0)
    assert metrics["signed_volume_mm3"] == pytest.approx(2.0)
    assert metrics["closed_two_manifold"]


def test_diagonal_voxel_touch_is_retained_as_nonmanifold_vertex_evidence() -> None:
    mask = np.zeros((2, 2, 2), dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    _, _, metrics = build_voxel_boundary_mesh(mask, affine)

    assert metrics["occupied_voxel_count"] == 2
    assert metrics["exposed_voxel_face_count"] == 12
    assert metrics["signed_volume_mm3"] == pytest.approx(2.0)
    assert metrics["edge_incidence"]["all_edges_have_two_incident_triangles"]
    assert metrics["nonmanifold_vertex_count"] == 1
    assert not metrics["closed_two_manifold"]


def test_edge_touch_reports_four_face_edge_without_topology_repair() -> None:
    mask = np.zeros((2, 2, 2), dtype=np.bool_)
    mask[0, 0, 0] = True
    mask[1, 1, 0] = True
    affine = np.eye(4).tolist()

    _, _, metrics = build_voxel_boundary_mesh(mask, affine)

    assert metrics["edge_incidence"]["incidence_histogram"].get("4", 0) > 0
    assert metrics["nonmanifold_vertex_count"] > 0
    assert not metrics["closed_two_manifold"]


def test_vectorized_vertex_link_audit_matches_reference_for_all_two_cube_masks() -> (
    None
):
    affine = np.eye(4).tolist()
    for pattern in range(1, 1 << 8):
        mask = np.array(
            [(pattern >> bit) & 1 for bit in range(8)], dtype=np.bool_
        ).reshape(2, 2, 2)
        vertices, triangles, _ = build_voxel_boundary_mesh(mask, affine)
        expected = _reference_nonmanifold_vertex_count(triangles, len(vertices))
        assert _nonmanifold_vertex_count(triangles, len(vertices), np) == expected


def test_contact_split_preserves_coordinates_and_closes_all_two_cube_masks() -> None:
    affine = np.eye(4).tolist()
    for pattern in range(1, 1 << 8):
        mask = np.array(
            [(pattern >> bit) & 1 for bit in range(8)],
            dtype=np.bool_,
        ).reshape(2, 2, 2)
        raw_vertices, raw_triangles, raw_metrics = build_voxel_boundary_mesh(
            mask, affine
        )
        split_vertices, split_triangles, split_metrics = build_voxel_boundary_mesh(
            mask,
            affine,
            split_contact_topology=True,
        )

        assert np.array_equal(
            raw_vertices[raw_triangles], split_vertices[split_triangles]
        )
        assert split_metrics["signed_volume_mm3"] == raw_metrics["signed_volume_mm3"]
        assert split_metrics["surface_area_mm2"] == raw_metrics["surface_area_mm2"]
        assert split_metrics["closed_two_manifold"]
        assert split_metrics["edge_incidence"]["all_edges_have_two_incident_triangles"]
        assert split_metrics["nonmanifold_vertex_count"] == 0
        assert split_metrics["topology_split"]["triangle_coordinates_preserved"]
        assert (
            split_metrics["topology_split"]["alternative_contact_resolution_count"] == 0
        )


def test_contact_split_uniquely_resolves_a_three_layer_diagonal_saddle() -> None:
    mask = np.zeros((3, 2, 2), dtype=np.bool_)
    mask[0, :, :] = True
    mask[2, :, :] = True
    mask[1, 0, 0] = True
    mask[1, 1, 1] = True
    affine = np.eye(4).tolist()

    raw_vertices, raw_triangles, raw_metrics = build_voxel_boundary_mesh(mask, affine)
    split_vertices, split_triangles, split_metrics = build_voxel_boundary_mesh(
        mask,
        affine,
        split_contact_topology=True,
    )

    assert not raw_metrics["closed_two_manifold"]
    assert raw_metrics["edge_incidence"]["incidence_histogram"].get("4") == 1
    assert split_metrics["closed_two_manifold"]
    assert split_metrics["edge_incidence"]["all_edges_have_two_incident_triangles"]
    assert split_metrics["nonmanifold_vertex_count"] == 0
    assert np.array_equal(raw_vertices[raw_triangles], split_vertices[split_triangles])
    assert split_metrics["signed_volume_mm3"] == raw_metrics["signed_volume_mm3"]
    assert split_metrics["surface_area_mm2"] == raw_metrics["surface_area_mm2"]
    assert split_metrics["topology_split"]["alternative_contact_resolution_count"] == 1
    assert (
        split_metrics["topology_split"]["alternative_contact_resolution"][
            "unique_closed_two_manifold_solution_count"
        ]
        == 1
    )


def test_empty_label_is_rejected_and_compressed_ply_is_deterministic(tmp_path) -> None:
    affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ImportError, match="nonempty 3-D"):
        build_voxel_boundary_mesh(np.zeros((2, 2, 2), dtype=np.bool_), affine)

    vertices, triangles, _ = build_voxel_boundary_mesh(
        np.ones((1, 1, 1), dtype=np.bool_),
        affine,
    )
    first, second = tmp_path / "first.ply.gz", tmp_path / "second.ply.gz"
    _write_binary_ply_gzip(first, vertices, triangles, np)
    _write_binary_ply_gzip(second, vertices, triangles, np)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rb") as mesh:
        header = mesh.read(512).split(b"end_header\n", 1)[0]
    assert b"format binary_little_endian 1.0" in header
    assert b"property double x" in header


def test_trial_plan_binds_exact_archive_and_compiler_sources(tmp_path) -> None:
    compiler_sources = {"surface.py": "a" * 64, "reader.py": "b" * 64}
    plan = {
        "schema": PLAN_SCHEMA,
        "input_intake_receipt_sha256": "c" * 64,
        "source_archive_sha256": "d" * 64,
        "scan_id": "001",
        "label_ids": [2, 5],
        "compiler_sources_sha256": compiler_sources,
        "runtime": {
            "python_version": "3.11.12",
            "numpy_version": "2.2.0",
            "scipy_version": "1.17.1",
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    parsed, digest = _read_trial_plan(
        path,
        receipt_sha256="c" * 64,
        source_archive_sha256="d" * 64,
        scan_id="001",
        label_ids=[5, 2],
        compiler_sources_sha256=compiler_sources,
        runtime=plan["runtime"],
    )
    assert parsed == plan
    assert len(digest) == 64
    with pytest.raises(ImportError, match="source, scan, labels, and compiler"):
        _read_trial_plan(
            path,
            receipt_sha256="c" * 64,
            source_archive_sha256="e" * 64,
            scan_id="001",
            label_ids=[2, 5],
            compiler_sources_sha256=compiler_sources,
            runtime=plan["runtime"],
        )

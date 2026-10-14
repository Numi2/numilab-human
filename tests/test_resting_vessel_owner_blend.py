import numpy as np
import pytest

from numilab_human.resting_vessel_owner_blend import (
    _retire_isolated_source_face,
    apply_parent_surface_updates,
    apply_route110_bridge_row,
    geodesic_owner_partition,
)


def _strip(with_island=False):
    vertices = np.asarray(
        [[x, y, 0.0] for x in range(4) for y in (0.0, 1.0)],
        dtype=np.float32,
    )
    faces = []
    for x in range(3):
        a = 2 * x
        faces.extend(((a, a + 2, a + 1), (a + 1, a + 2, a + 3)))
    if with_island:
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    [[6.0, 0.0, 0.0], [6.1, 0.0, 0.0], [6.0, 0.1, 0.0]],
                    dtype=np.float32,
                ),
            )
        )
        faces.append((8, 9, 10))
    return vertices, np.asarray(faces, dtype=np.int64)


def test_geodesic_owner_partition_has_exact_ports_and_positive_interior():
    vertices, faces = _strip()
    weights, report = geodesic_owner_partition(vertices, faces, {0, 1}, {6, 7})
    assert np.array_equal(weights[[0, 1]], [[1, 0], [1, 0]])
    assert np.array_equal(weights[[6, 7]], [[0, 1], [0, 1]])
    assert np.all(weights[2:6] > 0)
    assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-7
    assert report["weights_are_inferred_not_measured"] is True


def test_geodesic_owner_partition_rejects_overlapping_or_disconnected_ports():
    vertices, faces = _strip(with_island=True)
    with pytest.raises(ValueError, match="empty or overlapping"):
        geodesic_owner_partition(vertices, faces, {0}, {0})
    with pytest.raises(ValueError, match="unmapped disconnected"):
        geodesic_owner_partition(vertices, faces, {0, 1}, {6, 7})


def test_geodesic_owner_partition_preserves_explicit_distal_only_remainder():
    vertices, faces = _strip(with_island=True)
    weights, report = geodesic_owner_partition(
        vertices, faces, {0, 1}, {6, 7}, isolated_distal_remainder={8, 9, 10}
    )
    assert np.array_equal(weights[[8, 9, 10]], np.tile([0, 1], (3, 1)))
    assert report["isolated_distal_remainder_vertex_count"] == 3
    assert report["isolated_distal_remainder_face_count"] == 1


def test_scoped_source_face_retirement_keeps_vertex_prefix_and_other_faces():
    vertices, faces = _strip()
    row = {
        "record": (91, 1938892, len(vertices), 10268397, faces.size, 477, 2, 0),
        "vertices6": np.column_stack(
            (vertices, np.tile([0.0, 0.0, 1.0], (len(vertices), 1)))
        ).astype("<f4"),
        "faces": faces.astype("<u4"),
    }
    target = np.asarray(row["faces"][2], dtype=np.uint32)
    replacement, report = _retire_isolated_source_face(row, 2, target)
    assert replacement["vertices6"].tobytes() == row["vertices6"].tobytes()
    assert (
        replacement["faces"].tobytes()
        == np.concatenate((row["faces"][:2], row["faces"][3:])).tobytes()
    )
    assert replacement["record"][4] == replacement["faces"].size
    assert report["removed_face_count"] == 1
    assert report["output_vertex_count"] == report["input_vertex_count"]


def test_parent_binding_update_preserves_existing_sources_and_unrelated_maps():
    prior_surfaces = [{"name": f"tissue-{i}", "weight": i} for i in range(10)]
    prior_map = {
        "stable_id": 477,
        "vertices": [[[4, 2, 0, 1.0]]],
        "retained": "provenance",
    }
    base = {
        "anatomy_payload_sha256": "a" * 64,
        "child_payload_sha256": "b" * 64,
        "source_payloads": [
            {"path": f"assets/old-{i}.nhtissue", "sha256": str(i) * 64}
            for i in range(5)
        ],
        "surfaces": prior_surfaces,
        "anatomy_surfaces": [
            {"stable_id": 3, "vertices": [[[0, 1, 0, 1.0]]]},
            prior_map,
        ],
    }
    maps = {
        475: [[[5, 1, 0, 0.5], [5, 2, 0, 0.5]]],
        477: [[[5, 3, 0, 0.5], [5, 4, 0, 0.5]]],
    }
    updated, digest = apply_parent_surface_updates(
        base,
        {
            "carrier_source_path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
            "carrier_payload_sha256": "c" * 64,
            "parent_surface_updates": maps,
        },
        "d" * 64,
    )
    assert updated["anatomy_payload_sha256"] == "d" * 64
    assert updated["child_payload_sha256"] == base["child_payload_sha256"]
    assert updated["source_payloads"][:5] == base["source_payloads"]
    assert updated["source_payloads"][5] == {
        "path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
        "sha256": "c" * 64,
    }
    assert updated["surfaces"] == prior_surfaces
    rows = {row["stable_id"]: row for row in updated["anatomy_surfaces"]}
    assert rows[3] == base["anatomy_surfaces"][0]
    assert rows[477]["retained"] == "provenance"
    assert rows[477]["vertices"] == maps[477]
    assert rows[475] == {"stable_id": 475, "vertices": maps[475]}
    assert len(digest) == 64


def test_brachiocephalic_wrapper_forwards_owner_binding():
    from pathlib import Path
    from unittest.mock import patch

    from numilab_human import resting_major_vessels

    binding = {
        "n27_receipt_path": Path("N27.json"),
        "port_report_path": Path("086.json"),
    }
    with patch.object(
        resting_major_vessels, "append_viscera", return_value={"ok": True}
    ) as append:
        result = resting_major_vessels.append_brachiocephalic_veins(
            base_receipt=Path("base.json"),
            output=Path("out"),
            sources_path=Path("sources"),
            registration_path=Path("registration.json"),
            rigid_path=Path("rigid.nhrigid"),
            quotient_npz_path=Path("quotient.npz"),
            quotient_report_path=Path("quotient.json"),
            vessel_owner_blend_binding=binding,
        )
    assert result == {"ok": True}
    assert append.call_args.kwargs["vessel_owner_blend_binding"] == binding



def test_route110_bridge_row_transfer_requires_exact_compiler_input():
    before = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype="<f4",
    )
    after = before + np.asarray([0.0, 0.0, 0.001], dtype="<f4")
    normals = np.tile(np.asarray([0.0, 0.0, 1.0], dtype="<f4"), (3, 1))
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    compiled = {
        "record": (20, 0, 3, 0, 3, 525, 2, 0),
        "vertices6": np.column_stack((before, normals)).astype("<f4"),
        "faces": faces,
    }
    route = {
        "before_local_m": before,
        "after_local_m": after,
        "normals_after": normals,
        "faces": faces,
    }
    transferred = apply_route110_bridge_row(compiled, route)
    assert transferred["vertices6"][:, :3].tobytes() == after.tobytes()
    assert transferred["vertices6"][:, 3:].tobytes() == normals.tobytes()
    assert transferred["faces"].tobytes() == faces.tobytes()
    wrong = {**compiled, "vertices6": compiled["vertices6"].copy()}
    wrong["vertices6"][0, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError, match="existing ABI5 compiler output"):
        apply_route110_bridge_row(wrong, route)

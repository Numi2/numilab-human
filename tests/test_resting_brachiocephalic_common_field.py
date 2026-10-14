import copy
import hashlib

import numpy as np
import pytest

from numilab_human.resting_brachiocephalic_common_field import (
    _compile_records,
    _source_code_pins,
    _validate_common_field_descriptor,
    _validated_bridge_source_geometry,
    quintic_smoothstep,
)


def test_source_preparation_pins_existing_compiler_and_map_owners():
    pins = _source_code_pins()
    assert {
        "resting_brachiocephalic_common_field.py",
        "resting_brachiocephalic_source007.py",
        "resting_major_vessels.py",
        "resting_passive_viscera.py",
        "resting_vessel_owner_blend.py",
        "resting_intercostal_mesh_arrangement.py",
        "resting_anatomy_interface_patch.py",
        "resting_pleura_proxy.py",
    } == set(pins)
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in pins.values())


def test_common_field_descriptor_only_allows_anatomy_rebind_and_sidecar_paths():
    base = {
        "schema": "numi.human.cardiac_common_field.v1",
        "anatomy_payload_sha256": "a" * 64,
        "coordinate_order": ["RA", "RV", "LA", "LV"],
        "map": {"path": "base-map.bin", "sha256": "b" * 64, "record_count": 620718},
        "polynomials": {"path": "base-volumes.bin", "sha256": "c" * 64},
        "domain_boxes": {"path": "base-domains.bin", "sha256": "d" * 64},
        "passive_attachment_ranges": [{"stable_id": 10, "first_vertex": 100, "vertex_count": 318}],
        "source_reference_volumes_m3": [0.1, 0.2],
    }
    child = copy.deepcopy(base)
    child["anatomy_payload_sha256"] = "e" * 64
    child["map"]["path"] = "relocated-map.bin"
    child["polynomials"]["path"] = "relocated-volumes.bin"
    child["domain_boxes"]["path"] = "relocated-domains.bin"
    _validate_common_field_descriptor(child, base)

    for key, value in (
        ("coordinate_order", ["RV", "RA", "LA", "LV"]),
        ("passive_attachment_ranges", [{"stable_id": 10, "first_vertex": 101, "vertex_count": 318}]),
        ("source_reference_volumes_m3", [0.1, 0.3]),
    ):
        mutated = copy.deepcopy(child)
        mutated[key] = value
        with pytest.raises(ValueError, match="differs from pinned N27"):
            _validate_common_field_descriptor(mutated, base)

    mutated = copy.deepcopy(child)
    mutated["map"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from pinned N27"):
        _validate_common_field_descriptor(mutated, base)


def test_left_bridge_geometry_binds_route110_after_hash_and_exact_face_ancestry():
    from numilab_human.resting_vessel_owner_blend import ROUTE110_PINS

    old_xyz = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    new_xyz = old_xyz.copy()
    new_xyz[1, 0] = np.float32(1.25)
    normals = np.tile(np.asarray([0, 0, 1], dtype="<f4"), (3, 1))
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    digest = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
    base_xyz_sha = digest(old_xyz)
    after_xyz_sha = digest(new_xyz)
    faces_sha = digest(faces)
    bridge = {"stable_id": 525, "member": "FJ3482",
              "xyz": base_xyz_sha, "faces": faces_sha}
    pointer = {
        "stable_id": 525, "source_member": "FJ3482",
        "candidate_npz_sha256": ROUTE110_PINS["npz"],
        "candidate_report_sha256": ROUTE110_PINS["report"],
        "candidate_runner_sha256": ROUTE110_PINS["runner"],
        "target_preparation_checkpoint_sha256": ROUTE110_PINS["checkpoint"],
        "after_xyz_sha256": after_xyz_sha, "faces_sha256": faces_sha,
        "xyz_changed": True,
    }
    route = {
        "schema": "numi.human.route110.registered-reference-row-transfer.v1",
        "candidate_npz_sha256": ROUTE110_PINS["npz"],
        "candidate_report_sha256": ROUTE110_PINS["report"],
        "candidate_runner_sha256": ROUTE110_PINS["runner"],
        "target_preparation_checkpoint_sha256": ROUTE110_PINS["checkpoint"],
        "native_or_gpu_run": False,
        "bridge_compiler_binding": {
            "stable_id": 525, "source_member": "FJ3482",
            "existing_compiler_before_xyz_sha256": base_xyz_sha,
            "existing_compiler_faces_sha256": faces_sha,
            "compiler_output_matches_route110_before_exact": True,
        },
    }
    receipt = {"provenance": {
        "passive_brachiocephalic_vein_extension": {"route110_reference_field": route},
        "source_id_map": {"525": {"repair": {"route110_reference_field": pointer}}},
    }}
    row = {"vertices6": np.column_stack((new_xyz, normals)), "faces": faces}
    _, _, proof = _validated_bridge_source_geometry(row, bridge, receipt)
    assert proof["route110_source_row_transfer_verified"] is True
    assert proof["baseline_060_xyz_sha256"] == base_xyz_sha
    assert proof["current_xyz_sha256"] == after_xyz_sha

    broken = copy.deepcopy(receipt)
    broken["provenance"]["source_id_map"]["525"]["repair"][
        "route110_reference_field"]["after_xyz_sha256"] = base_xyz_sha
    with pytest.raises(ValueError, match="current geometry differs from exact route110"):
        _validated_bridge_source_geometry(row, bridge, broken)


def test_right_bridge_geometry_remains_bound_to_exact_060_source_row():
    xyz = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    normals = np.tile(np.asarray([0, 0, 1], dtype="<f4"), (3, 1))
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    digest = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
    bridge = {"stable_id": 526, "member": "FJ3583",
              "xyz": digest(xyz), "faces": digest(faces)}
    row = {"vertices6": np.column_stack((xyz, normals)), "faces": faces}
    _, _, proof = _validated_bridge_source_geometry(row, bridge, {})
    assert proof["geometry_owner"] == "unchanged exact source-060 ABI5 compilation"
    mutated = {"vertices6": np.column_stack((xyz + np.float32(0.1), normals)),
               "faces": faces}
    with pytest.raises(ValueError, match="candidate source bridge geometry differs"):
        _validated_bridge_source_geometry(mutated, bridge, {})


def test_quintic_fade_has_zero_endpoint_derivatives_and_monotone_values():
    samples = np.linspace(0.0, 1.0, 101)
    values = quintic_smoothstep(samples, np)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) >= 0.0)
    assert np.isclose(values[50], 0.5)
    assert np.isclose(values[1], 6e-10 - 15e-8 + 10e-6)


def test_attachment_interpolates_svc_then_fades_to_existing_zero_field():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    svc_faces = np.array([[0, 1, 2]], dtype=np.int64)
    base = np.zeros((3, 7, 4), dtype="<f4")
    base[:, :, :3] = 0.25
    bridge_xyz = np.array([[0, 0, 0], [0.25, 0.25, 0.1], [0.5, 0.5, 0]],
                          dtype=np.float32)
    d_svc = np.array([0.0, 0.5, 1.0])
    d_distal = np.array([1.0, 0.5, 0.0])

    records, ancestry = _compile_records(
        bridge_xyz, d_svc, d_distal, svc_xyz, svc_faces, base, 1.0, np)

    assert records.shape == (3, 28)
    assert np.all(records.view("<u4").reshape(3, 7, 4)[:, :, 3] == 0)
    assert records[0].tobytes() == base[0].tobytes()
    assert np.allclose(records[1].reshape(7, 4)[:, :3], 0.125)
    assert np.all(records[1].reshape(7, 4)[:, 3] == 0.0)
    assert np.array_equal(records[2], np.zeros(28, dtype="<f4"))
    assert ancestry[0]["exact_source_coordinate_copy"] is True
    assert ancestry[0]["coefficient_scale"] == 1.0
    assert ancestry[1]["coefficient_scale"] == 0.5
    assert ancestry[2]["distal_zero_field_vertex"] is True


def test_projection_fails_closed_outside_route_derived_bound():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    svc_faces = np.array([[0, 1, 2]], dtype=np.int64)
    coefficients = np.ones((3, 7, 4), dtype="<f4")
    coefficients[:, :, 3] = 0.0
    with pytest.raises(ValueError, match="exceeds fixed source-route bound"):
        _compile_records(
            np.array([[0.2, 0.2, 0.1]], dtype=np.float32),
            np.array([0.5]), np.array([0.5]),
            svc_xyz, svc_faces, coefficients, 0.05, np)


def test_negative_zero_in_existing_svc_w_lane_is_rejected():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    svc_faces = np.array([[0, 1, 2]], dtype=np.int64)
    coefficients = np.zeros((3, 7, 4), dtype="<f4")
    coefficients[0, 0, 3] = np.float32(-0.0)
    with pytest.raises(ValueError, match="canonical positive zero"):
        _compile_records(
            np.array([[0.2, 0.2, 0.1]], dtype=np.float32),
            np.array([0.5]), np.array([0.5]),
            svc_xyz, svc_faces, coefficients, 1.0, np)


def test_overlapping_exact_svc_coordinates_must_have_identical_map_records():
    svc_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                       dtype=np.float32)
    svc_faces = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
    coefficients = np.zeros((4, 7, 4), dtype="<f4")
    coefficients[3, 0, 0] = 1.0
    with pytest.raises(ValueError, match="duplicate SVC coordinates have unequal coefficients"):
        _compile_records(
            np.array([[0, 0, 0]], dtype=np.float32),
            np.array([0.0]), np.array([1.0]),
            svc_xyz, svc_faces, coefficients, 1.0, np)


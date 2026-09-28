"""Source-frame admission regressions, distinct from loaded mechanics."""
import hashlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from numilab_human import model as human
from numilab_human.lower_limb_source_registration import _source_frame_check
from numilab_human.upper_limb_pose_audit import PoseAuditError


def _frame_fixture(name, angle, scale):
    c, s = math.cos(angle), math.sin(angle)
    desired = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    # Core inertial axes can be very different from the atlas/world axes.
    owner = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    matrix = np.eye(4)
    matrix[:3, :3] = owner.T @ desired * .0010076 * scale
    anchor = {"source": {"member_id": "source-fixture"}, "target": {"name": name},
              "registration": {"source_obj_mm_to_core_inertial_body_m": matrix.tolist()}}
    common = {"global_source_mm_to_myosim_world_m": np.diag([.0010076, .0010076, .0010076, 1.]).tolist()}
    return _source_frame_check(anchor, common, owner.tolist())


@pytest.mark.parametrize("scale", [.93, .9695, 1., 1.07])
@pytest.mark.parametrize("angle", [0., .12, .249])
def test_valid_long_bone_variation_and_different_inertial_axes_are_preserved(scale, angle):
    result = _frame_fixture("tibia_l", angle, scale)
    assert result["passed"]
    assert result["atlas_relative_rotation_angle_rad"] == pytest.approx(angle, abs=1e-7)
    assert result["atlas_relative_uniform_scale"] == pytest.approx(scale)


@pytest.mark.parametrize("name,angle,scale", [
    ("patella_l", math.pi, 1.), ("patella_r", .8001, 1.),
    ("patella_l", .1, .98), ("tibia_r", .2501, 1.),
    ("tibia_l", .1, .9299), ("tibia_l", .1, 1.0701),
    ("toes_l", .1, 1.001),
])
def test_total_orientation_and_scale_cannot_be_reset_by_a_fresh_identity_fit(name, angle, scale):
    assert not _frame_fixture(name, angle, scale)["passed"]


def test_common_frame_comes_from_pinned_vertices_and_runtime_not_editable_frame_metadata(tmp_path):
    archive = tmp_path / "atlas.zip"
    archive.write_bytes(b"source atlas fixture")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    anchors, bodies, vertices_by_obj = [], {}, {}
    expected = np.array([[0., -.00103, 0., .3], [.00103, 0., 0., -.2], [0., 0., .00103, 1.], [0., 0., 0., 1.]])
    for i, spec in enumerate(human._BODYPARTS_MYOSIM_FIT_BONE_ANCHORS):
        member, name = spec["member_id"], spec["myosim_body"]
        obj = member.encode()
        center = np.array([10. * i, 3. * i * i, 50. * (i % 5)])
        vertices_by_obj[obj] = (center + np.array([[-1., 0., 0.], [0., 1., 0.], [1., -1., 0.]])).tolist()
        anchors.append({"source": {"member_id": member, "archive_sha256": archive_sha,
                                   "member_sha256": hashlib.sha256(obj).hexdigest(),
                                   "vertex_centroid_mm": [1e9, 1e9, 1e9]}, "target": {"name": name}})
        bodies[name] = (i, {"position_world_m": (expected[:3, :3] @ center + expected[:3, 3]).tolist()})
    registration = {"anchors": anchors, "coordinate_system": {"global_source_mm_to_myosim_world_m": np.eye(4).tolist()}}
    def member_reader(sources, hierarchy, member):
        return archive, member, member.encode()
    with patch.object(human, "_bodyparts_obj_member", side_effect=member_reader), \
         patch.object(human, "_bodyparts_obj_triangles", side_effect=lambda obj, member: (vertices_by_obj[obj], [[0, 1, 2]])):
        result = human._bodyparts_source_common_frame(tmp_path, registration, bodies)
        assert np.asarray(result["global_source_mm_to_myosim_world_m"]) == pytest.approx(expected)
        anchors[0]["source"]["member_sha256"] = "0" * 64
        with pytest.raises(human.ImportError, match="source drift.*FJ3393"):
            human._bodyparts_source_common_frame(tmp_path, registration, bodies)


def test_actual_flipped_patella_is_rejected_even_when_every_interface_passes(tmp_path):
    from numilab_human.lower_limb_pose_audit import audit_lower_limb_poses
    source_path = os.environ.get("NUMILAB_HUMAN_MOTION_REPAIRED")
    sources = os.environ.get("NUMILAB_HUMAN_MOTION_SOURCES")
    artifact = os.environ.get("NUMILAB_HUMAN_MOTION_ARTIFACT")
    if not all((source_path, sources, artifact)):
        pytest.skip("exact source-geometry corruption inputs were not supplied")
    registration = json.loads(Path(source_path).read_text())
    _, runtime_bodies = human._bodyparts_runtime_bindings(registration, Path(artifact))
    for anchor in registration["anchors"]:
        name = anchor["target"]["name"]
        if name not in {"patella_l", "patella_r"}:
            continue
        source = anchor["source"]
        _, member, obj = human._bodyparts_obj_member(Path(sources), source["hierarchy"], source["member_id"])
        vertices, _ = human._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(anchor["registration"]["source_obj_mm_to_core_inertial_body_m"])
        owner_rotation = np.asarray(runtime_bodies[name][1]["rotation_world"])
        flip = owner_rotation.T @ np.diag([-1., 1., -1.]) @ owner_rotation
        center = matrix[:3, :3] @ np.mean(vertices, axis=0) + matrix[:3, 3]
        matrix[:3, :3] = flip @ matrix[:3, :3]
        matrix[:3, 3] = center + flip @ (matrix[:3, 3] - center)
        anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = matrix.tolist()
    corrupted_path = tmp_path / "flipped.registration.json"
    corrupted_path.write_text(json.dumps(registration))
    with pytest.raises(PoseAuditError) as caught:
        audit_lower_limb_poses(sources=Path(sources), registration_path=corrupted_path, artifact=Path(artifact))
    result = caught.value.result
    assert all(i["passed"] for pose in result["poses"] for i in pose["continuity"])
    bad = [i for i in result["source_geometry_checks"] if not i["passed"]]
    assert {i["myosim_body"] for i in bad} == {"patella_l", "patella_r"}
    assert all(i["held_out_surface_metrics"]["p90_m"] <= i["maximum_held_out_p90_m"] for i in bad)
    # The same exact corruption must fail before a compiled payload is written.
    anatomy = {"source_id": registration["source"]["bodyparts"]["id"],
               "version": registration["source"]["bodyparts"]["version"],
               "archives": registration["source"]["bodyparts"]["archives"]}
    with pytest.raises(human.ImportError, match="atlas orientation/scale failed.*FJ3275"):
        human.bodyparts_myosim_bone_visual_payload(Path(sources), anatomy, corrupted_path, tmp_path,
                                                 artifact=Path(artifact))
    assert not (tmp_path / "bodyparts3d-myosim-major-bones.nhbones").exists()

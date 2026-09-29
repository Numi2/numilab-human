"""Execute axial geometry checks on the selected native runtime and source pack."""
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from numilab_human import model as human
from numilab_human.lower_limb_pose_audit import POSE_SUITE
from numilab_human.upper_limb_pose_audit import (
    _pose_qpos, _pose_joint_range_context, _projected_joint_range_checks, _joint_equality_program_checks,
)
from numilab_human.upper_limb_registration import _interface_patch_metrics, _minimum_gap, _rotation_xyzw


@pytest.fixture(scope="module")
def inputs():
    names = ["NUMILAB_HUMAN_NATIVE_VISUAL_PROBE", "NUMILAB_HUMAN_NATIVE_BONE_PAYLOAD",
             "NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT", "NUMILAB_HUMAN_MOTION_SOURCES",
             "NUMILAB_HUMAN_MOTION_REPAIRED"]
    if any(not os.environ.get(name) for name in names):
        pytest.skip("native axial regression requires exact source and runtime inputs")
    probe, bones, artifact, sources, registration_path = [Path(os.environ[name]).resolve() for name in names]
    from myo_sim.build.compose import build_model
    from numilab_human.myosim_export import export_fullbody
    registration = json.loads(registration_path.read_text())
    _, runtime = human._bodyparts_runtime_bindings(registration, artifact)
    source_bodies = {int(b["id"]): b for b in export_fullbody(sources)["bodies"]}
    transitions = human._NUMI_HUMAN_AXIAL_CONTINUITY_TRANSITIONS
    members = {m for _, first, second in transitions for m in (first, second)}
    stable_ids = {s["member_id"]: i for i, s in enumerate(human._BODYPARTS_MYOSIM_BONE_ANCHORS, 1)}
    geometry = {}
    for anchor in registration["anchors"]:
        source = anchor["source"]
        if source["member_id"] not in members:
            continue
        _, member, obj = human._bodyparts_obj_member(sources, source["hierarchy"], source["member_id"])
        vertices, _ = human._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(anchor["registration"]["source_obj_mm_to_core_inertial_body_m"])
        body = source_bodies[anchor["target"]["source_body_id"]]
        core = np.asarray(vertices) @ matrix[:3, :3].T + matrix[:3, 3]
        local = core @ _rotation_xyzw(body["inertial_quaternion_body_xyzw"], np).T + body["inertial_position_body_m"]
        geometry[source["member_id"]] = (body["id"], local)
    return probe, bones.read_bytes(), artifact, runtime, build_model("myofullbody"), geometry, stable_ids


def run_probe(inputs, tmp_path, raw=None, pose=(), equality_raw=None, rigid_raw=None):
    probe, valid, artifact, *_ = inputs
    path = tmp_path / "candidate.nhbones"
    path.write_bytes(valid if raw is None else raw)
    equality = artifact / "myosim-fullbody-joint-equalities.nheq"
    if equality_raw is not None:
        equality = tmp_path / "candidate.nheq"
        equality.write_bytes(equality_raw)
    rigid = artifact / "myosim-fullbody-core-reference.nhrigid"
    if rigid_raw is not None:
        rigid = tmp_path / "candidate.nhrigid"
        rigid.write_bytes(rigid_raw)
    command = [str(probe), str(rigid),
               str(artifact / "myosim-fullbody-muscle-reference.nhmyo"), str(path),
               str(tmp_path / "views"), "--dimension", "512",
               "--joint-equality-payload", str(equality)]
    for q, value in pose:
        command += ["--pose-q", str(q), str(value)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (tmp_path / "stdout").write_text(result.stdout)
    (tmp_path / "stderr").write_text(result.stderr)
    (tmp_path / "exit.code").write_text(str(result.returncode) + "\n")
    return result


def rigid_source_frames(inputs):
    raw = bytearray((inputs[2] / "myosim-fullbody-core-reference.nhrigid").read_bytes())
    header = struct.unpack_from("<8s10I32s", raw)
    assert header[0] == b"NHRIGID2" and header[4:8] == (157, 156, 129, 128)
    joint_offset = 80 + 96 + 48 + 160 * header[4]
    source_map_offset = joint_offset + 144 * header[5] + 64 * header[7] + 4 * (header[6] + header[7])
    source_map = struct.unpack_from(f"<{header[3]}I", raw, source_map_offset)
    # Join the existing source/Core map rather than treating the knee's virtual
    # rotation carrier as the physical tibia owner.
    record = source_map.index(136)
    pose_offset = source_map_offset + 4 * header[3] + 28 * record
    joint = joint_offset + 144 * 133
    assert struct.unpack_from("<8I", raw, joint)[1:6] == (134, 0, 0, 106, 1)
    return raw, joint, pose_offset, record


@pytest.mark.parametrize("corruption,field", [
    ("knee_anchor", "source_COM_position_m"),
    ("knee_frame_rotation", "source_COM_position_m"),
    ("source_position_witness", "source_COM_position_m"),
    ("source_orientation_witness", "source_orientation_unit_witness_m"),
    ("zero_source_orientation", "source_orientation_norm_error"),
    ("nonunit_source_orientation", "source_orientation_norm_error"),
])
def test_source_witnesses_join_executing_rest_frames_before_geometry(inputs, tmp_path, corruption, field):
    raw, joint, pose, record = rigid_source_frames(inputs)
    if corruption == "knee_anchor":
        # All stored source poses, source identities and topology are unchanged.
        struct.pack_into("<f", raw, joint + 80, .020)
    elif corruption == "knee_frame_rotation":
        angle = .020
        struct.pack_into("<4f", raw, joint + 128, 0., 0., np.sin(angle / 2), np.cos(angle / 2))
    elif corruption == "source_position_witness":
        struct.pack_into("<f", raw, pose, struct.unpack_from("<f", raw, pose)[0] + .020)
    else:
        quaternion = struct.unpack_from("<4f", raw, pose + 12)
        if corruption == "source_orientation_witness":
            # A unit quaternion can still falsely describe this source frame.
            angle = .020
            x, y, z, w = quaternion
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            candidate = (c*x - s*y, c*y + s*x, c*z + s*w, c*w - s*z)
        else:
            factor = 0. if corruption == "zero_source_orientation" else 1.0021
            candidate = tuple(factor * value for value in quaternion)
        struct.pack_into("<4f", raw, pose + 12, *candidate)
    result = run_probe(inputs, tmp_path, rigid_raw=raw)
    assert result.returncode != 0
    assert "NHRIGID source rest-frame binding failed" in result.stderr
    assert f"field={field} source_record_index={record} body_index=136 inbound_joint_index=135" in result.stderr
    assert f"rigid_sha256={hashlib.sha256(raw).hexdigest()}" in result.stderr
    failure = dict(part.split("=", 1) for part in result.stderr.split('failed: ', 1)[1].rstrip('\n"').split())
    assert float(failure["measured"]) > float(failure["allowed"])
    if field == "source_orientation_norm_error":
        assert float(failure["allowed"]) == pytest.approx(float(np.float32(.002)), abs=1e-12)
        assert failure["tolerance_basis"] == "existing_native_geometry_quaternion_norm_admission"
    else:
        assert float(failure["allowed"]) == 1e-6
        assert failure["tolerance_basis"] == (
            "existing_native_rest_reconstruction_tolerance_m" if field == "source_COM_position_m" else
            "existing_native_rest_reconstruction_tolerance_on_1m_orientation_witness")
    assert "bone_orientation_semantics=" not in result.stdout
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("target,factor", [
    ("source_witness", -1.), ("source_witness", 1.00199), ("source_witness", .99801),
    ("joint_frame", -1.), ("joint_frame", 1.0000049), ("joint_frame", .9999951),
], ids=["source_antipodal", "source_near_upper", "source_near_lower",
        "joint_antipodal", "joint_near_upper", "joint_near_lower"])
def test_equivalent_rigid_frame_orientations_preserve_source_geometry(inputs, tmp_path, factor, target):
    raw, joint, pose, _ = rigid_source_frames(inputs)
    offset = pose + 12 if target == "source_witness" else joint + 128
    original = struct.unpack_from("<4f", raw, offset)
    # Witness geometry uses the native geometry norm admission (0.002);
    # executing Core joint frames retain their tighter squared-norm bound (1e-5).
    struct.pack_into("<4f", raw, offset, *(factor * value for value in original))
    result = run_probe(inputs, tmp_path, pose=((7, -.4), (8, .1), (9, .2)), rigid_raw=raw)
    assert result.returncode == 0, result.stderr
    check = next(dict(part.split("=", 1) for part in line.split())
                 for line in result.stdout.splitlines() if line.startswith("rigid_source_rest_frames="))
    assert check["rigid_source_rest_frames"] == "passed" and check["source_body_count"] == "103"
    assert check["rigid_sha256"] == hashlib.sha256(raw).hexdigest()
    assert float(check["maximum_COM_position_residual_m"]) <= 1e-6
    assert float(check["maximum_orientation_unit_witness_residual_m"]) <= 1e-6
    assert_source_geometry(inputs, result, ((7, -.4), (8, .1), (9, .2)))


def measurements(result):
    return [dict(part.split("=", 1) for part in line.split())
            for line in result.stdout.splitlines() if line.startswith("axial_bone_interface=")]


def assert_source_geometry(inputs, result, pose):
    import mujoco
    _, _, _, _, model, geometry, stable_ids = inputs
    data = mujoco.MjData(model)
    q, count, _ = _pose_qpos(model, pose, mujoco, np)
    assert count == 51
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    world = {member: vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
             for member, (body, vertices) in geometry.items()}
    checks = measurements(result)
    transitions = human._NUMI_HUMAN_AXIAL_CONTINUITY_TRANSITIONS
    assert len(checks) == len(transitions) == 27
    for check, (name, first, second) in zip(checks, transitions, strict=True):
        assert check["axial_bone_interface"] == name and check["passed"] == "true"
        assert int(check["first_stable_id"]) == stable_ids[first]
        assert int(check["second_stable_id"]) == stable_ids[second]
        assert float(check["allowed_gap_m"]) == human._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M
        assert float(check["allowed_patch_p90_m"]) == .010
        assert float(check["minimum_gap_m"]) == pytest.approx(_minimum_gap(world[first], world[second], np)[0], abs=1e-6)
        expected_patch = _interface_patch_metrics(world[first], world[second], np)["bidirectional_p90_m"]
        assert float(check["patch_p90_m"]) == pytest.approx(expected_patch, abs=1e-6)


@pytest.mark.parametrize("pose", [(), ((7, -.4),), ((8, .15),), ((9, .25),),
    POSE_SUITE[-1][1] + ((7, -.4), (8, .1), (9, .2))],
    ids=["neutral", "flexion", "lateral_bend", "rotation", "coupled_crouch"])
def test_executed_axial_interfaces_match_independent_source_poses(inputs, tmp_path, pose):
    import mujoco
    result = run_probe(inputs, tmp_path, pose=pose)
    if (106, .75) in pose:
        # The source polynomial exceeds both enabled rotation2 limits here.
        # Keep the difficult pose and all 27 geometry comparisons below.
        assert result.returncode != 0
        assert "projected native position range failed" in result.stderr
        assert "failed_range_count=2" in result.stdout
        ranges = [dict(part.split("=", 1) for part in line.split())
                  for line in result.stdout.splitlines() if line.startswith("pose_joint_range=failed ")]
        assert {int(r["q_index"]) for r in ranges} == {107, 121}
        # Independently evaluate the actual serialized coefficients, rather
        # than the higher-precision source model used by the geometry oracle.
        equality_bytes = (inputs[2] / "myosim-fullbody-joint-equalities.nheq").read_bytes()
        equalities = {row[0]: row for offset in range(80, len(equality_bytes), 96)
                      for row in [struct.unpack_from("<4I20f", equality_bytes, offset)]}
        rigid_bytes = (inputs[2] / "myosim-fullbody-core-reference.nhrigid").read_bytes()
        header = struct.unpack_from("<8s10I32s", rigid_bytes)
        dof_offset = 80 + 96 + 48 + 160 * header[4] + 144 * header[5]
        for r in ranges:
            assert r["unit"] == "rad" and float(r["allowed_violation"]) == 1e-9
            row = equalities[int(r["q_index"])]
            delta = dict(pose)[row[2]] - row[5]
            target = row[4] + sum(row[6 + degree] * delta ** degree for degree in range(5))
            value = float(np.float32(target))
            lower, upper = struct.unpack_from("<2f", rigid_bytes, dof_offset + 64 * row[1] + 32)
            assert float(r["native_fp32_value"]) == pytest.approx(value, abs=1e-12)
            assert float(r["violation"]) == pytest.approx(max(0., lower - value, value - upper), abs=1e-12)
        assert len(list((tmp_path / "views").glob("*.png"))) == 4
        assert "myosim_articulated_bodyparts_bone_visual=ok" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
    assert "axial_bone_continuity=passed transition_count=27" in result.stdout
    assert_source_geometry(inputs, result, pose)


def scaled_cervical_orientation(inputs, factor):
    raw = bytearray(inputs[1])
    offset = 60 + (inputs[-1]["FJ3164"] - 1) * 60 + 36
    original = struct.unpack_from("<4f", raw, offset)
    assert original == (-.5, .5, .5, .5)
    struct.pack_into("<4f", raw, offset, *(v * factor for v in original))
    return raw, original, struct.unpack_from("<4f", raw, offset)


@pytest.mark.parametrize("factor", [-1., 1. + 2.**-10, -(1. + 2.**-10), 1.00199, .99801],
                         ids=["antipodal", "scaled", "scaled_antipodal", "near_upper", "near_lower"])
@pytest.mark.parametrize("pose", [((7, -.4), (8, .1), (9, .2)),
                                  POSE_SUITE[-1][1] + ((7, -.4), (8, .1), (9, .2))],
                         ids=["coupled_torso", "coupled_crouch"])
def test_equivalent_bone_orientations_match_executed_source_geometry(inputs, tmp_path, factor, pose):
    raw, original, candidate = scaled_cervical_orientation(inputs, factor)
    # The owning source reader defines these as the same orientation. The old
    # native geometry path deformed scaled inputs, then failed visual admission.
    np.testing.assert_allclose(human._myosim_matrix_from_quaternion_xyzw(candidate),
                               human._myosim_matrix_from_quaternion_xyzw(original),
                               rtol=0., atol=np.finfo(float).eps)
    result = run_probe(inputs, tmp_path, raw, pose=pose)
    if (106, .75) in pose:
        assert result.returncode != 0 and "projected native position range failed" in result.stderr
        assert "failed_range_count=2" in result.stdout
    else:
        assert result.returncode == 0, result.stderr
    assert_source_geometry(inputs, result, pose)
    assert len(list((tmp_path / "views").glob("*.png"))) == 4
    preparation = next(dict(part.split("=", 1) for part in line.split())
                       for line in result.stdout.splitlines() if line.startswith("bone_orientation_semantics="))
    assert preparation["bone_orientation_semantics"] == "unit_normalized_once_at_load"
    assert int(preparation["bone_count"]) == 185
    assert float(preparation["norm_admission_tolerance"]) == pytest.approx(float(np.float32(.002)), abs=1e-12)
    assert preparation["bone_sha256"] == human.sha256(tmp_path / "candidate.nhbones")


@pytest.mark.parametrize("factor", [0., 1.0021, .9979, float("nan"), float("inf")],
                         ids=["zero", "above_admission", "below_admission", "nan", "infinite"])
def test_invalid_bone_orientation_retains_structure_tolerance_and_identity(inputs, tmp_path, factor):
    raw, _, _ = scaled_cervical_orientation(inputs, factor)
    result = run_probe(inputs, tmp_path, raw)
    assert result.returncode != 0
    line = next(line for line in result.stderr.splitlines() if "BodyParts3D bone stable_id=" in line)
    fields = line.split("BodyParts3D bone ", 1)[1].rstrip('"')
    check = dict(part.split("=", 1) for part in fields.split())
    assert int(check["stable_id"]) == inputs[-1]["FJ3164"]
    assert int(check["body_index"]) == 26
    norm, tolerance = float(check["orientation_norm"]), float(check["allowed_absolute_norm_error"])
    assert tolerance == pytest.approx(float(np.float32(.002)), abs=1e-12)
    assert not np.isfinite(norm) or abs(norm - 1.) > tolerance
    assert check["bone_sha256"] == human.sha256(tmp_path / "candidate.nhbones")
    assert not measurements(result) and not (tmp_path / "views").exists()


@pytest.fixture(scope="module")
def joint_ranges(inputs):
    import mujoco
    artifact, model = inputs[2], inputs[4]
    registration = json.loads(Path(os.environ["NUMILAB_HUMAN_MOTION_REPAIRED"]).read_text())
    reference, _ = human._bodyparts_runtime_bindings(registration, artifact)
    return _pose_joint_range_context(artifact, reference, model, mujoco)


@pytest.mark.parametrize("pose_name,expected", [
    ("neutral", set()),
    ("bilateral_knee_flexion", {"knee_angle_translation2_l"}),
    ("bilateral_deep_crouch", {"knee_angle_translation2_l"}),
    ("bilateral_functional_crouch", {
        "knee_angle_translation2_l", "knee_angle_rotation2_r", "knee_angle_rotation2_l",
    }),
])
def test_source_projection_conflicts_are_distinct_from_consumed_limits(inputs, joint_ranges, pose_name, expected):
    import mujoco
    q, _, _ = _pose_qpos(inputs[4], dict(POSE_SUITE)[pose_name], mujoco, np)
    checks = _projected_joint_range_checks(q, joint_ranges, np)
    assert len(checks) == 122
    failed = {c["source_joint_name"]: c for c in checks if not c["passed"]}
    assert set(failed) == expected
    if "knee_angle_translation2_l" in failed:
        left = failed["knee_angle_translation2_l"]
        assert left["unit"] == "m" and left["q_index"] == 118
        assert not left["native_position_limit_enabled"]
        assert left["native_position_range"] is None and left["native_position_range_passed"]
        assert left["source_range_violation"] > .003
    for name in expected - {"knee_angle_translation2_l"}:
        assert failed[name]["native_position_limit_enabled"]
        assert not failed[name]["native_position_range_passed"]
        assert failed[name]["native_range_violation"] == pytest.approx(.00010107457637786865, abs=1e-12)


@pytest.mark.parametrize("constant,valid", [(.10, False), (-.00025, True), (.00025, True)])
def test_dependent_coordinate_corruption_and_valid_variation(inputs, joint_ranges, tmp_path, constant, valid):
    import hashlib
    import mujoco
    model = inputs[4]
    equality_index = next(i for i in range(model.neq)
                          if int(model.jnt_qposadr[int(model.eq_obj1id[i])]) == 107)
    saved = model.eq_data[equality_index, :5].copy()
    try:
        model.eq_data[equality_index, :5] = [constant, 0., 0., 0., 0.]
        q, _, _ = _pose_qpos(model, ((7, -.4),), mujoco, np)
        check = next(c for c in _projected_joint_range_checks(q, joint_ranges, np) if c["q_index"] == 107)
        assert check["passed"] is valid
    finally:
        model.eq_data[equality_index, :5] = saved
    raw = bytearray((inputs[2] / "myosim-fullbody-joint-equalities.nheq").read_bytes())
    assert raw[:8] == b"NHEQ1\0\0\0"
    record = next(offset for offset in range(80, len(raw), 96)
                  if struct.unpack_from("<I", raw, offset)[0] == 107)
    struct.pack_into("<5f", raw, record + 24, constant, 0., 0., 0., 0.)
    result = run_probe(inputs, tmp_path, pose=((7, -.4),), equality_raw=raw)
    assert (result.returncode == 0) is valid, result.stderr
    assert len(list((tmp_path / "views").glob("*.png"))) == 4
    assert hashlib.sha256(raw).hexdigest() in result.stdout
    if not valid:
        assert "pose_joint_range=failed q_index=107 v_index=106 core_joint_index=134" in result.stdout
        assert "projected native position range failed" in result.stderr
        assert "diagnostic_views=" in result.stderr and "rigid_sha256=" in result.stderr
        assert "myosim_articulated_bodyparts_bone_visual=ok" not in result.stdout


def test_source_input_range_tolerance_is_unchanged(inputs):
    import mujoco
    joint = next(i for i in range(inputs[4].njnt) if int(inputs[4].jnt_qposadr[i]) == 107)
    upper = float(inputs[4].jnt_range[joint, 1])
    with pytest.raises(RuntimeError, match="exceeds its source range"):
        _pose_qpos(inputs[4], ((107, upper + 5e-12),), mujoco, np)


@pytest.fixture(scope="module")
def equality_source(inputs):
    from numilab_human.myosim_export import export_fullbody
    return export_fullbody(Path(os.environ["NUMILAB_HUMAN_MOTION_SOURCES"]))


def copy_reference_artifact(inputs, output):
    import shutil
    output.mkdir()
    manifest = json.loads((inputs[2] / "myosim-fullbody-reference.manifest.json").read_text())
    for descriptor in manifest["payloads"].values():
        source = inputs[2] / descriptor["file"]
        if source.is_file():
            shutil.copy2(source, output / descriptor["file"])
    return manifest


@pytest.mark.parametrize("role,field", [
    ("joint_equalities", "polynomial"),
    ("joint_equalities", "coordinate_owner"),
    ("joint_equalities_source_compliance", "reference"),
    ("joint_equalities_source_compliance", "solref"),
    ("joint_equalities_source_compliance", "inverse_weight"),
])
def test_changed_native_equality_law_cannot_hide_behind_updated_sidecar(
    inputs, joint_ranges, equality_source, tmp_path, role, field,
):
    import hashlib
    artifact = tmp_path / "artifact"
    manifest = copy_reference_artifact(inputs, artifact)
    descriptor = manifest["payloads"][role]
    path = artifact / descriptor["file"]
    raw = bytearray(path.read_bytes())
    stride = 96 if role == "joint_equalities" else 112
    offset = next(i for i in range(80, len(raw), stride) if struct.unpack_from("<I", raw, i)[0] == 107)
    if field == "coordinate_owner":
        struct.pack_into("<2I", raw, offset + 8, 7, 6)  # Valid independent torso coordinate, wrong owner.
    else:
        relative = {"polynomial": 24, "reference": 16, "solref": 48, "inverse_weight": 96}[field]
        value = .02 if field in ("polynomial", "reference") else .04 if field == "solref" else 2.0
        struct.pack_into("<f", raw, offset + relative, value)
    path.write_bytes(raw)
    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path = artifact / "myosim-fullbody-reference.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reference = {"manifest": {"file": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}}
    checks = _joint_equality_program_checks(artifact, reference, equality_source, joint_ranges)
    failed = [c for c in checks if not c["passed"]]
    assert len(failed) == 1 and failed[0]["payload_role"] == role
    assert failed[0]["declared_identity_matches"] and failed[0]["source_header_matches"]
    assert failed[0]["source_byte_mismatch_count"] > 0
    assert failed[0]["maximum_allowed_byte_mismatch_count"] == 0
    assert failed[0]["actual_sha256"] != failed[0]["expected_source_sha256"]
    assert len(failed[0]["affected_equalities"]) == 1
    assert failed[0]["affected_equalities"][0]["dependent_name"] == "knee_angle_rotation2_r"
    if field == "polynomial":
        candidate_inputs = (inputs[0], inputs[1], artifact, *inputs[3:])
        result = run_probe(candidate_inputs, tmp_path, pose=((7, -.4),))
        assert result.returncode == 0, result.stderr  # Numeric range admission alone is insufficient.
        assert "pose_joint_ranges=passed" in result.stdout
        assert descriptor["sha256"] in result.stdout


def test_pinned_equality_programs_match_without_rejecting_registered_variation(inputs, joint_ranges, equality_source):
    import hashlib
    path = inputs[2] / "myosim-fullbody-reference.manifest.json"
    reference = {"manifest": {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    checks = _joint_equality_program_checks(inputs[2], reference, equality_source, joint_ranges)
    assert {c["payload_abi"] for c in checks} == {1, 2}
    assert all(c["passed"] and c["record_count"] == 51 for c in checks)
    # These checks consume source programs, not registration placement. The
    # existing small cervical/lumbar variation tests still execute below.


@pytest.mark.parametrize("corruption", ["disable_native_limit", "widen_native_range"])
def test_source_range_claim_cannot_override_consumed_limit_bytes(inputs, tmp_path, corruption):
    import hashlib
    import mujoco
    manifest = json.loads((inputs[2] / "myosim-fullbody-reference.manifest.json").read_text())
    raw = bytearray((inputs[2] / manifest["payloads"]["rigid"]["file"]).read_bytes())
    header = struct.unpack_from("<8s10I32s", raw)
    dofs = 80 + 96 + 48 + 160 * header[4] + 144 * header[5]
    offset = dofs + 64 * 106
    if corruption == "disable_native_limit":
        struct.pack_into("<I", raw, offset + 20, 0)
    else:
        struct.pack_into("<f", raw, offset + 36, .10)
    (tmp_path / manifest["payloads"]["rigid"]["file"]).write_bytes(raw)
    manifest["payloads"]["rigid"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "myosim-fullbody-reference.manifest.json"
    path.write_text(json.dumps(manifest))
    reference = {"manifest": {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    with pytest.raises(RuntimeError, match="joint range binding drifted: knee_angle_rotation2_r"):
        _pose_joint_range_context(tmp_path, reference, inputs[4], mujoco)


def displaced_bone(inputs, amount, isolated_witnesses=False, *,
                   member="FJ3165", body_name="lumbar4",
                   neighbours=(("FJ3162", "lumbar3"), ("FJ3168", "lumbar5"))):
    _, valid, _, runtime, _, _, stable_ids = inputs
    raw = bytearray(valid)
    header_size = struct.calcsize("<8s5I32s")
    _, abi, count, *_ = struct.unpack_from("<8s5I32s", raw)
    assert abi == 3 and count == 185
    records = {struct.unpack_from("<I", raw, header_size + i*60 + 20)[0]: header_size + i*60
               for i in range(count)}
    offset = records[stable_ids[member]]
    translation = np.asarray(struct.unpack_from("<3f", raw, offset + 24))
    translation += np.asarray(runtime[body_name][1]["rotation_world"]).T @ np.asarray([0., 0., amount])
    struct.pack_into("<3f", raw, offset + 24, *translation)
    if isolated_witnesses:
        vertex_start = header_size + count*60
        first_vertex = struct.unpack_from("<I", raw, offset + 4)[0]
        moved_rotation = np.asarray(runtime[body_name][1]["rotation_world"])
        moved_origin = np.asarray(runtime[body_name][1]["position_world_m"])
        bone_values = struct.unpack_from("<8f", raw, offset + 24)
        bone_rotation = _rotation_xyzw(bone_values[3:7], np)
        for i, (neighbour_member, name) in enumerate(neighbours):
            neighbour = records[stable_ids[neighbour_member]]
            index = struct.unpack_from("<I", raw, neighbour + 4)[0]
            vertex = np.asarray(struct.unpack_from("<3f", raw, vertex_start + index*24))
            values = struct.unpack_from("<8f", raw, neighbour + 24)
            point = np.asarray(values[:3]) + values[7] * (_rotation_xyzw(values[3:7], np) @ vertex)
            world = np.asarray(runtime[name][1]["rotation_world"]) @ point + runtime[name][1]["position_world_m"]
            local = bone_rotation.T @ (moved_rotation.T @ (world - moved_origin) - translation) / bone_values[7]
            struct.pack_into("<3f", raw, vertex_start + (first_vertex + i)*24, *local)
    return raw


@pytest.mark.parametrize("isolated_witnesses", [False, True], ids=["disconnected", "isolated_touching_vertices"])
def test_displaced_executing_bone_is_rejected_despite_valid_owner_binding(inputs, tmp_path, isolated_witnesses):
    result = run_probe(inputs, tmp_path, displaced_bone(inputs, .040, isolated_witnesses))
    assert result.returncode != 0
    assert "executed axial continuity failed" in result.stderr
    failed = [c for c in measurements(result) if c["passed"] == "false"]
    assert len(failed) == 1
    assert "lumbar" in failed[0]["axial_bone_interface"]
    if isolated_witnesses:
        assert float(failed[0]["minimum_gap_m"]) < .008
        assert float(failed[0]["patch_p90_m"]) > .010
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("isolated_witnesses", [False, True], ids=["disconnected", "isolated_touching_vertices"])
def test_displaced_cervical_compound_member_is_rejected(inputs, tmp_path, isolated_witnesses):
    raw = displaced_bone(inputs, .040, isolated_witnesses, member="FJ3164",
                         body_name="cervical_spine",
                         neighbours=(("FJ3161", "cervical_spine"), ("FJ3167", "cervical_spine")))
    result = run_probe(inputs, tmp_path, raw)
    assert result.returncode != 0
    assert "executed axial continuity failed" in result.stderr
    failed = [c for c in measurements(result) if c["passed"] == "false"]
    assert len(failed) == 1
    assert failed[0]["axial_bone_interface"] == "cervical3_to_cervical4"
    assert int(failed[0]["second_stable_id"]) == inputs[-1]["FJ3164"]
    assert float(failed[0]["allowed_gap_m"]) == .008
    assert float(failed[0]["allowed_patch_p90_m"]) == .010
    assert f"bone_sha256={human.sha256(tmp_path / 'candidate.nhbones')}" in result.stderr
    if isolated_witnesses:
        assert float(failed[0]["minimum_gap_m"]) < .008
        assert float(failed[0]["patch_p90_m"]) > .010
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("amount", [-.00025, .00025])
@pytest.mark.parametrize("member,body_name", [("FJ3165", "lumbar4"), ("FJ3164", "cervical_spine")],
                         ids=["lumbar", "cervical"])
def test_small_valid_registration_variation_is_preserved(inputs, tmp_path, amount, member, body_name):
    result = run_probe(inputs, tmp_path, displaced_bone(inputs, amount, member=member, body_name=body_name))
    assert result.returncode == 0, result.stderr
    assert "axial_bone_continuity=passed" in result.stdout


def test_source_compiler_preserves_geometry_and_rejects_disconnected_cervical_registration(inputs, tmp_path):
    sources = Path(os.environ["NUMILAB_HUMAN_MOTION_SOURCES"])
    registration_path = Path(os.environ["NUMILAB_HUMAN_MOTION_REPAIRED"])
    anatomy = human.parse_bodyparts3d(sources, Path(human.__file__).resolve().parents[2] /
                                    "config/anatomy-classification.v1.json")
    _, valid, artifact, runtime, *_ = inputs
    compiled = tmp_path / "valid"
    manifest = human.bodyparts_myosim_bone_visual_payload(
        sources, anatomy, registration_path, compiled, artifact=artifact)
    assert (compiled / manifest["payload"]["file"]).read_bytes() == valid
    transitions = manifest["axial_continuity"]["transitions"]
    assert len(transitions) == 27
    assert sum(t["name"].startswith("cervical") for t in transitions) == 7
    assert all(t["minimum_vertex_gap_m"] <= .008 for t in transitions)
    assert manifest["axial_continuity"]["independent_articulation_count"] == 0

    registration = json.loads(registration_path.read_text())
    anchor = next(a for a in registration["anchors"] if a["source"]["member_id"] == "FJ3164")
    matrix = np.asarray(anchor["registration"]["source_obj_mm_to_core_inertial_body_m"])
    matrix[:3, 3] += np.asarray(runtime["cervical_spine"][1]["rotation_world"]).T @ [0., 0., .040]
    anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = matrix.tolist()
    invalid_registration = tmp_path / "disconnected-cervical4.registration.json"
    invalid_registration.write_text(json.dumps(registration))
    rejected = tmp_path / "rejected"
    with pytest.raises(human.ImportError, match="cervical3_to_cervical4.*axial continuity gate"):
        human.bodyparts_myosim_bone_visual_payload(
            sources, anatomy, invalid_registration, rejected, artifact=artifact)
    assert not rejected.exists()

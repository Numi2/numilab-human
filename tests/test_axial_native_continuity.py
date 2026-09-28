"""Execute axial geometry checks on the selected native runtime and source pack."""
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from numilab_human import model as human
from numilab_human.lower_limb_pose_audit import POSE_SUITE
from numilab_human.upper_limb_pose_audit import _pose_qpos
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


def run_probe(inputs, tmp_path, raw=None, pose=()):
    probe, valid, artifact, *_ = inputs
    path = tmp_path / "candidate.nhbones"
    path.write_bytes(valid if raw is None else raw)
    command = [str(probe), str(artifact / "myosim-fullbody-core-reference.nhrigid"),
               str(artifact / "myosim-fullbody-muscle-reference.nhmyo"), str(path),
               str(tmp_path / "views"), "--dimension", "512",
               "--joint-equality-payload", str(artifact / "myosim-fullbody-joint-equalities.nheq")]
    for q, value in pose:
        command += ["--pose-q", str(q), str(value)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def measurements(result):
    return [dict(part.split("=", 1) for part in line.split())
            for line in result.stdout.splitlines() if line.startswith("axial_bone_interface=")]


@pytest.mark.parametrize("pose", [(), ((7, -.4),), ((8, .15),), ((9, .25),),
    POSE_SUITE[-1][1] + ((7, -.4), (8, .1), (9, .2))],
    ids=["neutral", "flexion", "lateral_bend", "rotation", "coupled_crouch"])
def test_executed_axial_interfaces_match_independent_source_poses(inputs, tmp_path, pose):
    import mujoco
    result = run_probe(inputs, tmp_path, pose=pose)
    assert result.returncode == 0, result.stderr
    assert "axial_bone_continuity=passed transition_count=21" in result.stdout
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
    assert len(checks) == len(transitions) == 21
    for check, (name, first, second) in zip(checks, transitions, strict=True):
        assert check["axial_bone_interface"] == name and check["passed"] == "true"
        assert int(check["first_stable_id"]) == stable_ids[first]
        assert int(check["second_stable_id"]) == stable_ids[second]
        assert float(check["allowed_gap_m"]) == human._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M
        assert float(check["allowed_patch_p90_m"]) == .010
        assert float(check["minimum_gap_m"]) == pytest.approx(_minimum_gap(world[first], world[second], np)[0], abs=1e-6)
        expected_patch = _interface_patch_metrics(world[first], world[second], np)["bidirectional_p90_m"]
        assert float(check["patch_p90_m"]) == pytest.approx(expected_patch, abs=1e-6)


def displaced_lumbar4(inputs, amount, isolated_witnesses=False):
    _, valid, _, runtime, _, _, stable_ids = inputs
    raw = bytearray(valid)
    header_size = struct.calcsize("<8s5I32s")
    _, abi, count, *_ = struct.unpack_from("<8s5I32s", raw)
    assert abi == 3 and count == 185
    records = {struct.unpack_from("<I", raw, header_size + i*60 + 20)[0]: header_size + i*60
               for i in range(count)}
    offset = records[stable_ids["FJ3165"]]
    translation = np.asarray(struct.unpack_from("<3f", raw, offset + 24))
    translation += np.asarray(runtime["lumbar4"][1]["rotation_world"]).T @ np.asarray([0., 0., amount])
    struct.pack_into("<3f", raw, offset + 24, *translation)
    if isolated_witnesses:
        vertex_start = header_size + count*60
        first_vertex = struct.unpack_from("<I", raw, offset + 4)[0]
        moved_rotation = np.asarray(runtime["lumbar4"][1]["rotation_world"])
        moved_origin = np.asarray(runtime["lumbar4"][1]["position_world_m"])
        bone_values = struct.unpack_from("<8f", raw, offset + 24)
        bone_rotation = _rotation_xyzw(bone_values[3:7], np)
        for i, (member, name) in enumerate([("FJ3162", "lumbar3"), ("FJ3168", "lumbar5")]):
            neighbour = records[stable_ids[member]]
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
    result = run_probe(inputs, tmp_path, displaced_lumbar4(inputs, .040, isolated_witnesses))
    assert result.returncode != 0
    assert "executed axial continuity failed" in result.stderr
    failed = [c for c in measurements(result) if c["passed"] == "false"]
    assert len(failed) == 1
    assert "lumbar" in failed[0]["axial_bone_interface"]
    if isolated_witnesses:
        assert float(failed[0]["minimum_gap_m"]) < .008
        assert float(failed[0]["patch_p90_m"]) > .010
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("amount", [-.00025, .00025])
def test_small_valid_registration_variation_is_preserved(inputs, tmp_path, amount):
    result = run_probe(inputs, tmp_path, displaced_lumbar4(inputs, amount))
    assert result.returncode == 0, result.stderr
    assert "axial_bone_continuity=passed" in result.stdout

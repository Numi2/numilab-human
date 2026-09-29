"""Native organ placement and adversarial geometry/ownership checks."""
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from numilab_human import model as human
from numilab_human.torso_anatomy_audit import audit_torso_anatomy


@pytest.fixture(scope="module")
def organ_inputs(tmp_path_factory):
    names = ["NUMILAB_HUMAN_NATIVE_VISUAL_PROBE", "NUMILAB_HUMAN_NATIVE_BONE_PAYLOAD",
             "NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT", "NUMILAB_HUMAN_MOTION_SOURCES",
             "NUMILAB_HUMAN_MOTION_REPAIRED"]
    if any(not os.environ.get(name) for name in names):
        pytest.skip("organ source audit needs selected native/source inputs")
    probe, bones, artifact, sources, registration = [Path(os.environ[name]).resolve() for name in names]
    root = tmp_path_factory.mktemp("source-organs")
    anatomy = human.parse_bodyparts3d(sources, human.REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    human.bodyparts_myosim_torso_anatomy_visual_payload(sources, anatomy, registration, artifact, root / "payload")
    payload = root / "payload/bodyparts3d-myosim-torso-anatomy.nhanatomy"
    return probe, bones, sources, artifact, registration, payload, root


@pytest.fixture(scope="module", params=[None, (), ((7, -.4), (8, .1), (9, .2))],
                ids=["raw_source_rest", "projected_neutral", "coupled_torso"])
def native_organs(organ_inputs, request):
    probe, bones, sources, artifact, registration, payload, root = organ_inputs
    pose = request.param
    output = root / request.node.callspec.id if hasattr(request.node, "callspec") else root / str(request.param_index)
    command = [str(probe), str(artifact / "myosim-fullbody-core-reference.nhrigid"),
               str(artifact / "myosim-fullbody-muscle-reference.nhmyo"), str(bones), str(output),
               "--dimension", "512", "--torso-anatomy-payload", str(payload)]
    if pose is not None:
        command += ["--joint-equality-payload", str(artifact / "myosim-fullbody-joint-equalities.nheq")]
    for q, value in pose or ():
        command += ["--pose-q", str(q), str(value)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    output.mkdir(exist_ok=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "stdout").write_text(result.stdout)
    (output / "stderr").write_text(result.stderr)
    (output / "exit.code").write_text(str(result.returncode) + "\n")
    assert result.returncode == 0, result.stderr
    pack = next(output.glob("*.mrvpack"))
    poses = next(output.glob("*.torso-anatomy-poses.json"))
    return sources, artifact, registration, payload, pack, poses, pose


def test_native_organs_preserve_every_source_vertex_and_topology(native_organs):
    report = audit_torso_anatomy(*native_organs)
    assert report["passed"] and report["surface_count"] == 24
    assert all(row["topology_exact"] for row in report["rows"])
    assert {row["member_id"] for row in report["rows"] if row["label"] == "liver"} == {
        "FJ2816", "FJ2818", "FJ2819", "FJ2820", "FJ2821", "FJ2822", "FJ2409", "FJ2823", "FJ2824"}
    native_organs[4].with_name("source-geometry-audit.json").write_text(json.dumps(report, indent=2) + "\n")


def _mutate_pack(path, output, kind, mutate):
    raw = bytearray(path.read_bytes())
    _, _, count, directory, _ = struct.unpack_from("<8sIIQQ", raw)
    for i in range(count):
        entry = directory + 72 * i
        section_kind, _, offset, size, *_ = struct.unpack_from("<IIQQQII32s", raw, entry)
        if section_kind == kind:
            mutate(raw, offset, size)
            raw[entry + 40:entry + 72] = hashlib.sha256(raw[offset:offset + size]).digest()
            output.write_bytes(raw)
            return
    raise AssertionError("test pack section missing")


@pytest.mark.parametrize("corruption", ["native_owner", "source_topology", "native_displacement", "native_normal", "native_pose_shift"])
def test_organ_source_oracle_rejects_rehashed_or_valid_metadata_corruption(native_organs, tmp_path, corruption):
    from numilab_human.torso_anatomy_audit import _pack_sections
    import numpy as np
    args = list(native_organs)
    sections = _pack_sections(args[4])
    primitives = np.frombuffer(sections[4][0], dtype="<u4").reshape(-1, 16)
    primitive_index = int(np.flatnonzero(primitives[:, 4] == 51010)[0])
    primitive = primitives[primitive_index]
    pack = tmp_path / "rehash.mrvpack"
    if corruption == "native_owner":
        _mutate_pack(args[4], pack, 4, lambda raw, offset, _: struct.pack_into("<I", raw, offset + 64 * primitive_index + 24, 7))
        args[4] = pack
        with pytest.raises(ValueError, match="native surface semantic/owner"):
            audit_torso_anatomy(*args)
    elif corruption == "source_topology":
        def swap(raw, offset, size):
            i = offset + 4 * int(primitive[0])
            a, b = struct.unpack_from("<2I", raw, i)
            struct.pack_into("<2I", raw, i, b, a)
        _mutate_pack(args[4], pack, 3, swap)
        args[4] = pack
        with pytest.raises(ValueError, match="native source topology"):
            audit_torso_anatomy(*args)
    elif corruption in {"native_displacement", "native_normal"}:
        native_indices = np.frombuffer(sections[3][0], dtype="<u4")
        vertex = int(native_indices[int(primitive[0])])
        def shift(raw, offset, size):
            i = offset + 80 * vertex + (16 if corruption == "native_normal" else 0)
            struct.pack_into("<f", raw, i, struct.unpack_from("<f", raw, i)[0] + .005)
        _mutate_pack(args[4], pack, 2, shift)
        args[4] = pack
        report = audit_torso_anatomy(*args)
        assert not report["passed"]
        if corruption == "native_displacement":
            assert report["maximum_native_pose_world_error_m"] > .0049
        else:
            assert max(row["source_normal_direction_error"] for row in report["rows"]) > .0049
    else:
        snapshot = json.loads(args[5].read_text())
        snapshot["bodies"][0]["position_world_m"][0] += .020
        poses = tmp_path / "shifted-poses.json"
        poses.write_text(json.dumps(snapshot))
        args[5] = poses
        report = audit_torso_anatomy(*args)
        assert not report["passed"] and report["maximum_native_pose_world_error_m"] > .0199


def test_organ_oracle_rejects_member_provenance_drift_even_with_unchanged_geometry(native_organs, tmp_path):
    args = list(native_organs)
    payload = tmp_path / args[3].name
    payload.write_bytes(args[3].read_bytes())
    manifest = json.loads(args[3].with_name("bodyparts3d-myosim-torso-anatomy.manifest.json").read_text())
    manifest["source"]["surfaces"][0]["member_sha256"] = "0" * 64
    payload.with_name("bodyparts3d-myosim-torso-anatomy.manifest.json").write_text(json.dumps(manifest))
    args[3] = payload
    with pytest.raises(ValueError, match="torso source member hash/identity"):
        audit_torso_anatomy(*args)


def test_native_oracle_rejects_forged_closed_ventricular_topology(native_organs, tmp_path):
    args = list(native_organs)
    payload = tmp_path / args[3].name
    payload.write_bytes(args[3].read_bytes())
    manifest = json.loads(args[3].with_name("bodyparts3d-myosim-torso-anatomy.manifest.json").read_text())
    wall = next(s for s in manifest["source"]["surfaces"] if s["member_id"] == "FJ2428")
    wall["source_family_topology"]["exact_coordinate_quotient"]["closed_oriented_manifold_candidate"] = True
    payload.with_name("bodyparts3d-myosim-torso-anatomy.manifest.json").write_text(json.dumps(manifest))
    args[3] = payload
    with pytest.raises(ValueError, match="torso source topology diagnostic"):
        audit_torso_anatomy(*args)

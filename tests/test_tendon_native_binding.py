"""Execute tendon admission against the selected native anatomy runtime."""
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from numilab_human import model as human
from numilab_human.lower_limb_pose_audit import POSE_SUITE


@pytest.fixture(scope="module")
def inputs():
    names = ["NUMILAB_HUMAN_NATIVE_VISUAL_PROBE", "NUMILAB_HUMAN_NATIVE_BONE_PAYLOAD",
             "NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT", "NUMILAB_HUMAN_NATIVE_TENDON_PAYLOAD"]
    if any(not os.environ.get(name) for name in names):
        pytest.skip("native tendon regression requires exact runtime and paired anatomy inputs")
    probe, bones, artifact, tendon = [Path(os.environ[name]).resolve() for name in names]
    return probe, bones, artifact, tendon


def run_probe(inputs, root, *, bone_bytes=None, muscle_bytes=None, tendon=None, pose=(), omit_bones=False):
    probe, bones, artifact, original_tendon = inputs
    root.mkdir(exist_ok=True)
    bone = root / "candidate.nhbones"
    muscle = root / "candidate.nhmyo"
    bone.write_bytes(bones.read_bytes() if bone_bytes is None else bone_bytes)
    muscle.write_bytes((artifact / "myosim-fullbody-muscle-reference.nhmyo").read_bytes()
                       if muscle_bytes is None else muscle_bytes)
    command = [str(probe), str(artifact / "myosim-fullbody-core-reference.nhrigid"), str(muscle)]
    if not omit_bones:
        command.append(str(bone))
    command += [str(root / "views"), "--dimension", "512", "--tendon-payload",
                str(original_tendon if tendon is None else tendon), "--joint-equality-payload",
                str(artifact / "myosim-fullbody-joint-equalities.nheq")]
    for q, value in pose:
        command += ["--pose-q", str(q), str(value)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def identity_line(result):
    return dict(part.split("=", 1) for part in result.stdout.splitlines()[0].split())


@pytest.mark.parametrize("pose", [(), ((9, .25),),
    POSE_SUITE[-1][1] + ((7, -.4), (8, .1), (9, .2))],
    ids=["neutral", "rotation", "coupled_crouch"])
def test_paired_anatomy_executes_with_all_point_and_distributed_attachments(inputs, tmp_path, pose):
    result = run_probe(inputs, tmp_path, pose=pose)
    assert result.returncode == 0, result.stderr
    check = identity_line(result)
    assert check["tendon_endpoints"] == "832"
    assert check["tendon_point_bindings"] == "190"
    assert check["tendon_envelope_bindings"] == "642"
    assert check["tendon_migrated_envelope_bindings"] == "18"
    assert check["tendon_geometry_identity_verified"] == "true"
    assert check["tendon_muscle_payload_sha256"] == hashlib.sha256(
        (inputs[2] / "myosim-fullbody-muscle-reference.nhmyo").read_bytes()).hexdigest()
    assert check["tendon_bone_payload_sha256"] == hashlib.sha256(inputs[1].read_bytes()).hexdigest()
    assert "axial_bone_continuity=passed" in result.stdout
    assert len(list((tmp_path / "views").glob("*.png"))) == 4


def displaced_patella(inputs, displacement):
    raw = bytearray(inputs[1].read_bytes())
    header = struct.unpack_from("<8s5I32s", raw)
    assert header[1] == 3
    manifest = json.loads((inputs[2] / "myosim-fullbody-reference.manifest.json").read_text())
    owner = next(row["core_body_index"] for row in manifest["core_tree"]["source_body_records"]
                 if row["name"] == "patella_l")
    offsets = [60 + 60*i for i in range(header[2]) if struct.unpack_from("<I", raw, 60 + 60*i)[0] == owner]
    assert len(offsets) == 1
    offset = offsets[0] + 24
    struct.pack_into("<f", raw, offset, struct.unpack_from("<f", raw, offset)[0] + displacement)
    return raw


def test_displaced_patella_with_stale_tendon_geometry_rejected_before_calibration(inputs, tmp_path):
    raw = displaced_patella(inputs, .010)
    # Preserve the source archive, registration fingerprint, mesh counts and
    # source/Core owners: those checks alone accepted the old corruption.
    assert raw[:60] == inputs[1].read_bytes()[:60]
    result = run_probe(inputs, tmp_path, bone_bytes=raw)
    assert result.returncode != 0
    assert "NHTENDON bone payload identity mismatch" in result.stderr
    assert "expected_sha256=" + hashlib.sha256(inputs[1].read_bytes()).hexdigest() in result.stderr
    assert "consumed_sha256=" + hashlib.sha256(raw).hexdigest() in result.stderr
    assert "tendon_max_reference_path_delta_m=" not in result.stdout
    assert not (tmp_path / "views").exists()


def changed_quadriceps_architecture(inputs):
    raw = bytearray((inputs[2] / "myosim-fullbody-muscle-reference.nhmyo").read_bytes())
    header = struct.unpack_from("<8s9I32s", raw)
    assert header[1] == 2 and header[8] == header[3] and header[9] == 32
    offset = 76 + 16*header[4] + 64*header[5] + 16*header[6] + 164*header[3] + 32*413
    struct.pack_into("<f", raw, offset, 1.01 * struct.unpack_from("<f", raw, offset)[0])
    assert raw[:76] == (inputs[2] / "myosim-fullbody-muscle-reference.nhmyo").read_bytes()[:76]
    return raw


def test_changed_quadriceps_architecture_with_stale_tendon_program_rejected(inputs, tmp_path):
    raw = changed_quadriceps_architecture(inputs)
    result = run_probe(inputs, tmp_path, muscle_bytes=raw)
    assert result.returncode != 0
    assert "NHTENDON muscle payload identity mismatch" in result.stderr
    assert "consumed_sha256=" + hashlib.sha256(raw).hexdigest() in result.stderr
    assert "tendon_max_reference_path_delta_m=" not in result.stdout
    assert not (tmp_path / "views").exists()


def test_missing_compiled_geometry_rejected_before_calibration_or_dynamics(inputs, tmp_path):
    result = run_probe(inputs, tmp_path, omit_bones=True)
    assert result.returncode != 0
    assert "NHTENDON2/3 requires its compiled bone positional payload before calibration or dynamics" in result.stderr
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("offset", [32, 36], ids=["bone_count", "registration_fingerprint"])
def test_geometry_declarations_cannot_override_consumed_bone_header(inputs, tmp_path, offset):
    raw = bytearray(inputs[3].read_bytes())
    value = struct.unpack_from("<I", raw, offset)[0]
    struct.pack_into("<I", raw, offset, value + 1)
    tendon = tmp_path / "mismatched-registration.nhtendon"
    tendon.write_bytes(raw)
    result = run_probe(inputs, tmp_path, tendon=tendon)
    assert result.returncode != 0
    assert "NHTENDON bone registration dimensions or fingerprint differ" in result.stderr
    assert not (tmp_path / "views").exists()


def test_historical_source_point_program_remains_readable_without_geometry_claim(inputs, tmp_path):
    output = tmp_path / "point-program"
    manifest = human.numi_human_tendon_endpoint_payload(inputs[2], output)
    result = run_probe(inputs, tmp_path, tendon=output / manifest["payload"]["file"], omit_bones=True)
    assert result.returncode == 0, result.stderr
    check = identity_line(result)
    assert check["tendon_endpoints"] == check["tendon_point_bindings"] == "832"
    assert check["tendon_geometry_identity_verified"] == "false"
    assert check["tendon_bone_payload_sha256"] == "unavailable"


def test_legacy_muscle_bytes_are_bound_without_hashing_synthesized_architectures(inputs, tmp_path):
    artifact = tmp_path / "legacy-source"
    artifact.mkdir()
    raw = (inputs[2] / "myosim-fullbody-muscle-reference.nhmyo").read_bytes()
    header = list(struct.unpack_from("<8s9I32s", raw))
    header[0], header[1], header[8], header[9] = b"NHMYO1\0\0", 1, 0, 0
    legacy = struct.pack("<8s9I32s", *header) + raw[76:-32*header[3]]
    manifest = json.loads((inputs[2] / "myosim-fullbody-reference.manifest.json").read_text())
    descriptor = manifest["payloads"]["muscles"]
    descriptor["sha256"] = hashlib.sha256(legacy).hexdigest()
    descriptor["bytes"] = len(legacy)
    (artifact / descriptor["file"]).write_bytes(legacy)
    (artifact / "myosim-fullbody-reference.manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "point-program"
    compiled = human.numi_human_tendon_endpoint_payload(artifact, output)
    result = run_probe(inputs, tmp_path / "run", muscle_bytes=legacy,
                       tendon=output / compiled["payload"]["file"], omit_bones=True)
    assert result.returncode == 0, result.stderr
    assert identity_line(result)["tendon_muscle_payload_sha256"] == descriptor["sha256"]
    assert identity_line(result)["tendon_geometry_identity_verified"] == "false"


def test_recompiled_submillimeter_variation_and_legacy_envelopes_remain_valid(inputs, tmp_path):
    bones = tmp_path / "bones"
    bones.mkdir()
    original_manifest = inputs[1].with_suffix(".manifest.json")
    manifest = json.loads(original_manifest.read_text())
    raw = displaced_patella(inputs, .00025)
    path = bones / manifest["payload"]["file"]
    path.write_bytes(raw)
    manifest["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
    (bones / original_manifest.name).write_text(json.dumps(manifest))
    # Recompile the existing owning format against the varied geometry. Its
    # authored source endpoints and all valid point dispositions stay fixed.
    output = tmp_path / "compiled"
    compiled = human.numi_human_tendon_attachment_envelope_payload(inputs[2], bones, output)
    result = run_probe(inputs, tmp_path / "run", bone_bytes=raw,
                       tendon=output / compiled["payload"]["file"])
    assert result.returncode == 0, result.stderr
    check = identity_line(result)
    assert check["tendon_payload"] == "NHTENDON2"
    assert check["tendon_endpoints"] == "832"
    assert check["tendon_point_bindings"] == str(compiled["coverage"]["source_site_point_fallback_count"])
    assert check["tendon_geometry_identity_verified"] == "true"
    assert check["tendon_bone_payload_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def reference_probe():
    path = os.environ.get("NUMILAB_HUMAN_NATIVE_REFERENCE_PROBE")
    if not path:
        pytest.skip("native reference regression requires an explicitly selected binary")
    return Path(path).resolve()


def test_reference_probe_executes_all_transfers_without_claiming_loaded_geometry(inputs, reference_probe):
    result = subprocess.run([str(reference_probe),
        str(inputs[2] / "myosim-fullbody-core-reference.nhrigid"),
        str(inputs[2] / "myosim-fullbody-muscle-reference.nhmyo"), str(inputs[3]), "--metal"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "tendon_geometry_identity_verified=false" in result.stdout
    assert "metal_geometry=paired_high_low" in result.stdout
    assert "metal_tendon_transfers=832" in result.stdout
    assert "metal_tendon_envelope_transfers=642" in result.stdout
    assert "metal_tendon_replay_byte_identical=true" in result.stdout


def test_reference_probe_rejects_stale_architecture_before_force_evaluation(inputs, reference_probe, tmp_path):
    raw = changed_quadriceps_architecture(inputs)
    path = tmp_path / "changed-quadriceps.nhmyo"
    path.write_bytes(raw)
    result = subprocess.run([str(reference_probe),
        str(inputs[2] / "myosim-fullbody-core-reference.nhrigid"), str(path), str(inputs[3]), "--metal"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "NHTENDON muscle payload identity mismatch" in result.stderr
    assert "consumed_sha256=" + hashlib.sha256(raw).hexdigest() in result.stderr
    assert "myosim_core_reference PASS" not in result.stdout

"""Compiled identities must survive reordered rows and distinct body/mesh IDs."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from numilab_human import model
from numilab_human import resting_anatomy_verify as v


def test_compiled_bone_ids_are_not_body_indices_or_guessed_names():
    names = [f"carrier_{i}" for i in range(44)]
    names[34], names[41], names[42], names[43] = "scapula_r", "humerus_r", "ulna_r", "radius_r"
    table = {43: [dict(stable_id=16, member_id="FJ3349")],
             34: [dict(stable_id=10, member_id="FJ3384")],
             42: [dict(stable_id=14, member_id="FJ3391")],
             41: [dict(stable_id=12, member_id="FJ3368")]}
    result = v.compiled_bone_identity_map(table, names)
    assert result[12] == dict(typed_identity=[51004, 12], core_body_index=41,
                             core_body_name="humerus_r", laterality="right",
                             source_member_id="FJ3368")
    assert result[16]["core_body_name"] == "radius_r"
    assert result[14]["core_body_name"] == "ulna_r"


@pytest.mark.parametrize("table,names", [
    ({True: [dict(stable_id=1, member_id="FJ1")]}, ["pelvis", "torso"]),
    ({1: [dict(stable_id=1, member_id="FJ1")]}, ["pelvis"]),
    ({0: [dict(stable_id=1, member_id="FJ1"), dict(stable_id=1, member_id="FJ2")]}, ["pelvis"]),
    ({0: [dict(stable_id=True, member_id="FJ1")]}, ["pelvis"]),
    ({0: [dict(stable_id=1, member_id=None)]}, ["pelvis"]),
    ({0: [dict(stable_id=1, member_id="FJ1")]}, ["torso", "torso"]),
])
def test_inconsistent_compiled_identity_is_rejected(table, names):
    with pytest.raises(ValueError):
        v.compiled_bone_identity_map(table, names)


def fixture_files(tmp_path, monkeypatch):
    runtime = tmp_path/"runtime.json"
    runtime.write_text(json.dumps(dict(source=dict(archive_sha256="a"*64),
                                      core_tree=dict(body_order=["pelvis", "humerus_l"]))))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=tmp_path/"bodyparts3d-myosim-major-bones.manifest.json"
    manifest.write_text(json.dumps(dict(source=dict(runtime_reference=dict(
        manifest=dict(sha256=digest(runtime)))))))
    payload=tmp_path/"fixture.nhbones"
    payload.write_bytes(b"owner-reader-fixture")
    def existing_reader(folder, source_sha):
        assert folder == tmp_path and source_sha == "a"*64
        return {1:[dict(stable_id=12, member_id="FJ2")]}, dict(sha256=digest(payload)), payload
    monkeypatch.setattr(model, "_numi_human_bone_envelope_surfaces", existing_reader)
    return runtime, payload, digest


def test_bone_identity_uses_existing_owner_and_pins_actual_capture_input(tmp_path, monkeypatch):
    runtime,payload,digest=fixture_files(tmp_path,monkeypatch)
    capture=tmp_path/"accepted-geometry"/"step-0.mrvpack"
    capture.parent.mkdir()
    invocation=tmp_path/"invocation.json"
    invocation.write_text(json.dumps(dict(argv=["native","rigid","muscles",str(payload)])))
    ids,proof=v.load_compiled_bone_identities(tmp_path,runtime,capture=capture)
    assert ids[12]["laterality"] == "left"
    assert proof["native_invocation"]["sha256"] == digest(invocation)


def test_capture_cannot_bind_different_bones(tmp_path,monkeypatch):
    runtime,payload,_=fixture_files(tmp_path,monkeypatch)
    wrong=tmp_path/"wrong.nhbones"
    wrong.write_bytes(b"other")
    (tmp_path/"invocation.json").write_text(json.dumps(dict(argv=["native","r","m",str(wrong)])))
    with pytest.raises(ValueError,match="differs from native"):
        v.load_compiled_bone_identities(tmp_path,runtime,capture=tmp_path/"accepted-geometry"/"step-0.mrvpack")


def test_runtime_identity_mismatch_stops_before_owner_reader(tmp_path,monkeypatch):
    runtime,_,_=fixture_files(tmp_path,monkeypatch)
    runtime.write_text("{}")
    monkeypatch.setattr(model,"_numi_human_bone_envelope_surfaces",
                        lambda *_:pytest.fail("reader must not run before identity check"))
    with pytest.raises(ValueError,match="manifest hash differs"):
        v.load_compiled_bone_identities(tmp_path,runtime)


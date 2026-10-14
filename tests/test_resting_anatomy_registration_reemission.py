import hashlib
import json
import struct

import numpy as np
import pytest

from numilab_human import resting_anatomy as owner


def prepare(tmp_path, monkeypatch):
    anchors = [{"source": {"member_id": "FJ1"}, "target": {"core_body_index": 20},
                "registration": {"fit": 1}}]
    old = {"coordinate_system": {"global": [1]}, "source": {"bodyparts": {"id": "bp"}},
           "anchors": anchors}
    new = json.loads(json.dumps(old))
    new["anchors"][0]["registration"]["fit"] = 2
    a, b = tmp_path/"old.json", tmp_path/"new.json"
    a.write_text(json.dumps(old)); b.write_text(json.dumps(new))
    fp = int(hashlib.sha256(a.read_bytes()).hexdigest()[:8], 16)
    v = np.array([[0,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,0,1]], dtype="<f4")
    raw = (owner.HEADER.pack(b"NHANAT1\0",5,1,3,3,fp,b"x"*32)
           + owner.RECORD.pack(20,0,3,0,3,313,1,0)
           + v.tobytes() + struct.pack("<3I",0,1,2))
    p = tmp_path/"base.nhanatomy";p.write_bytes(raw)
    context = {"rigid":{"sha256":"verified-rigid"}, "source_archive_sha256":(b"x"*32).hex()}
    monkeypatch.setattr(owner.human, "_bodyparts_runtime_bindings",
                        lambda doc, artifact: (context, {"torso":(20,{"rest":1})}))
    return p,a,b,tmp_path/"out.nhanatomy"


def test_reemission_retains_every_owner_and_geometry_byte(tmp_path, monkeypatch):
    p,a,b,out=prepare(tmp_path,monkeypatch)
    proof=owner.reemit_core_local_registration(p,a,b,tmp_path,out)
    assert out.read_bytes()[owner.HEADER.size:] == p.read_bytes()[owner.HEADER.size:]
    assert proof["anatomical_and_native_admission"] is False
    assert owner.HEADER.unpack_from(out.read_bytes())[5] == int(hashlib.sha256(b.read_bytes()).hexdigest()[:8],16)


@pytest.mark.parametrize("field",["source","target","coordinate_system"])
def test_reject_anatomical_or_global_identity_change(tmp_path, monkeypatch,field):
    p,a,b,out=prepare(tmp_path,monkeypatch)
    d=json.loads(b.read_text())
    if field=="coordinate_system":d[field]={"global":[2]}
    else:d["anchors"][0][field]={"changed":True}
    b.write_text(json.dumps(d))
    with pytest.raises(ValueError):owner.reemit_core_local_registration(p,a,b,tmp_path,out)
    assert not out.exists()


def test_reject_wrong_source_registration_or_body(tmp_path, monkeypatch):
    p,a,b,out=prepare(tmp_path,monkeypatch)
    raw=bytearray(p.read_bytes());struct.pack_into("<I",raw,24,123);p.write_bytes(raw)
    with pytest.raises(ValueError,match="registration or source"):owner.reemit_core_local_registration(p,a,b,tmp_path,out)
    assert not out.exists()


def test_reject_actual_rigid_drift(tmp_path, monkeypatch):
    p,a,b,out=prepare(tmp_path,monkeypatch)
    def frames(doc,artifact):
        return {"rigid":{"sha256":str(doc["anchors"][0]["registration"]["fit"])},
                "source_archive_sha256":(b"x"*32).hex()}, {"torso":(20,{})}
    monkeypatch.setattr(owner.human,"_bodyparts_runtime_bindings",frames)
    with pytest.raises(ValueError,match="rigid source frame"):owner.reemit_core_local_registration(p,a,b,tmp_path,out)
    assert not out.exists()

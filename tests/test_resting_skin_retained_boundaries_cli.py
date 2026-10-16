"""The shared entrypoint records aperture ancestry without exempting geometry."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("skin_boundary_cli",
    Path(__file__).resolve().parents[1]/"tools/verify_resting_anatomy.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

def fixture(tmp_path):
    faces=np.array([[0,1,2],[3,4,5]],dtype=np.uint32)
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]],dtype=np.float32)
    surfaces={(51007,1):(xyz,faces,None,np.arange(6,dtype=np.uint64)+100)}
    paths={}
    for name,data in {
        "pack":b"capture","skin":b"skin-source",
        "receipt":json.dumps(dict(skin_source_mapping=dict(source_vertex_count=6,pack_first_vertex=100))).encode(),
        "provenance":json.dumps(dict(retained_anatomical_interfaces=[
            dict(interface="ocular surface opening; passive eyeball interface",source_vertex_ids=x)
            for x in [[0,1,2],[3,4,5]]])).encode()
    }.items():
        p=tmp_path/name;p.write_bytes(data);paths[name]=(p,hashlib.sha256(data).hexdigest())
    args=["--pack",str(paths["pack"][0]),"--pack-sha256",paths["pack"][1],
          "--capture-receipt",str(paths["receipt"][0]),"--receipt-sha256",paths["receipt"][1],
          "--skin-source","=".join(map(str,paths["skin"])),
          "--skin-boundary-provenance","=".join(map(str,paths["provenance"])),
          "--changed","51007:1","--output",str(tmp_path/"out.json")]
    return args,surfaces,dict(indices=faces.ravel(),vertex_count=6)

def test_aperture_classification_does_not_turn_a_failed_geometry_gate_into_a_pass(tmp_path):
    args,surfaces,decoded=fixture(tmp_path)
    with patch.object(cli.verify,"load_capture",return_value=(surfaces,{})),          patch("numilab_human.skin_source_payload_preflight.decode_payload",return_value=decoded),          patch.object(cli.verify,"audit_changed_surfaces",return_value=dict(
             passed=False,failed_stage="self",rows=[dict(passed=False)])) as audit:
        assert cli.main(args)==1
    r=json.loads((tmp_path/"out.json").read_text())
    assert r["skin_boundary_classification"]["verification"]["unexplained_boundary_edge_count"]==0
    assert not r["skin_boundary_classification"]["predicate_exemptions_granted"]
    assert r["failed_stage"]=="self" and not r["passed"]
    assert not audit.call_args.kwargs["intended_enclosures"]

def test_source_provenance_pin_mismatch_stops_before_geometry(tmp_path):
    args,surfaces,decoded=fixture(tmp_path)
    i=args.index("--skin-boundary-provenance")+1
    args[i]=args[i].rsplit("=",1)[0]+"="+"0"*64
    with patch.object(cli.verify,"load_capture",return_value=(surfaces,{})),          patch.object(cli.verify,"audit_changed_surfaces") as audit:
        assert cli.main(args)==1
    assert not audit.called
    r=json.loads((tmp_path/"out.json").read_text())
    assert "SHA256 mismatch" in r["error"] and r["failed_stage"]=="preparation"

@pytest.mark.parametrize("option",["--skin-source","--skin-boundary-provenance"])
def test_incomplete_classification_inputs_never_pass_preparation(tmp_path,option):
    args,surfaces,_=fixture(tmp_path);i=args.index(option);del args[i:i+2];args+=["--preflight-only"]
    with patch.object(cli.verify,"load_capture",return_value=(surfaces,{})):
        assert cli.main(args)==1
    assert not json.loads((tmp_path/"out.json").read_text())["passed"]

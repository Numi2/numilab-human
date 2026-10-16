"""Collection retains failed anatomy gates without changing their predicates."""
from unittest.mock import patch
import numpy as np
import pytest
from numilab_human import resting_anatomy_verify as v
from numilab_human.resting_shared_partition_verify import audit_with_shared_partitions

F=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.uint32)
P=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
def surface(key,scale=1.,offset=0.):
    return v.prepare_surface(key,(P*np.float32(scale)+np.float32(offset)).astype("<f4"),F)
def bad_surface():
    return (np.concatenate((P[:3],P[:3])),np.array([[0,1,2],[3,4,5]],dtype=np.uint32),None,np.arange(6))

def test_collection_retains_all_failures_and_later_safe_pairs():
    first=surface((51011,7))
    nested=surface((51011,8),.2,.1)
    far=surface((51011,10),1.,5.)
    bad=(51010,313)
    surfaces={first.key:first,nested.key:nested,bad:bad_surface(),far.key:far}
    targets=[nested.key,bad,far.key]
    strict=v.audit_changed_surfaces(surfaces,[first.key],targets=targets)
    progress=[]
    result=v.audit_changed_surfaces(surfaces,[first.key],targets=targets,
                                    collect_failures=True,progress=progress.append)
    assert not strict["passed"] and not result["passed"]
    assert result["rows"][:len(strict["rows"])]==strict["rows"]
    assert result["rows"]==progress
    assert result["collection_complete"] is True and result["failure_count"]==2
    assert [row["stage"] for row in result["rows"]]==["self","nearby_pair","target_preparation","nearby_pair"]
    assert result["rows"][-1]["passed"] is True and result["rows"][-1]["aabb_disjoint"]
    assert result["failure_row_indices"]==[1,2]

def test_collection_keeps_self_failure_and_checks_other_changed_rows():
    a=(51005,88);b=(51011,10)
    points=np.concatenate((P,P+np.float32(.2)))
    faces=np.concatenate((F,F+4))
    folded=v.prepare_surface(a,points,faces)
    assert folded.closed and folded.self_audit()["count"]>0
    far=surface(b,offset=5.)
    result=v.audit_changed_surfaces({a:folded,b:far},[a,b],targets=[],
                                    collect_failures=True)
    assert not result["passed"] and result["failure_count"]==1
    assert len(result["rows"])==2 and result["rows"][-1]["passed"]
    assert result["failed_stage"]=="self"

def test_collection_records_changed_preparation_error_and_continues():
    a=(51010,313);b=(51011,10);good=surface(b,offset=5.)
    surfaces={a:bad_surface(),b:good}
    with pytest.raises(ValueError):
        v.audit_changed_surfaces(surfaces,[a,b],targets=[])
    result=v.audit_changed_surfaces(surfaces,[a,b],targets=[],collect_failures=True)
    assert not result["passed"] and result["rows"][0]["failed_identity"]==list(a)
    assert result["rows"][1]["stage"]=="self" and result["rows"][1]["passed"]

def test_collection_success_matches_strict_predicate_rows_and_requires_boolean():
    a=surface((51011,7));b=surface((51011,10),offset=5.)
    surfaces={a.key:a,b.key:b}
    strict=v.audit_changed_surfaces(surfaces,[a.key])
    collected=v.audit_changed_surfaces(surfaces,[a.key],collect_failures=True)
    assert strict["rows"]==collected["rows"] and collected["passed"]
    assert collected["failure_count"]==0 and collected["failure_row_indices"]==[]
    for value in (1,[],None,"yes"):
        with pytest.raises(TypeError,match="boolean"):
            v.audit_changed_surfaces(surfaces,[a.key],collect_failures=value)

def test_invalid_certificate_never_exempts_pair_and_does_not_hide_later_region():
    a=surface((51005,132));b=surface((51005,134),.2,.1)
    far=surface((51011,10),offset=5.)
    surfaces={a.key:a,b.key:b,far.key:far};progress=[]
    cert=[(a.key,b.key,"not-read","0"*64)]
    with patch("numilab_human.resting_shared_partition_verify.certify_shared_muscle_partition",
               side_effect=ValueError("source coordinate mismatch")):
        with pytest.raises(ValueError,match="coordinate mismatch"):
            audit_with_shared_partitions(surfaces,[a.key,far.key],partitions=cert)
        result=audit_with_shared_partitions(surfaces,[a.key,far.key],partitions=cert,
                    collect_failures=True,progress=progress.append)
    assert not result["passed"] and result["collection_complete"]
    assert result["shared_partition_certificates"]==[]
    assert result["rows"]==progress and result["rows"][0]["stage"]=="shared_partition"
    assert any(row.get("identities")==[list(a.key),list(b.key)] and not row["passed"]
               for row in result["rows"])
    assert result["per_changed_surface"][-1]["audit"]["passed"]

def test_bad_pair_declaration_is_retained_and_later_target_is_checked():
    a=surface((51011,7));b=surface((51011,8),.2,.1);far=surface((51011,10),offset=5.)
    surfaces={s.key:s for s in (a,b,far)}
    invalid={(a.key,b.key):[{"not_a_proof":True}]}
    with pytest.raises(ValueError):
        v.audit_changed_surfaces(surfaces,[a.key],targets=[b.key,far.key],
                    source_bound_connected_overlap_interfaces=invalid)
    result=v.audit_changed_surfaces(surfaces,[a.key],targets=[b.key,far.key],
                    source_bound_connected_overlap_interfaces=invalid,collect_failures=True)
    assert not result["passed"] and result["rows"][1]["stage"]=="pair_validation"
    assert result["rows"][-1]["passed"] and result["rows"][-1]["aabb_disjoint"]


def test_cli_collects_all_failures_and_returns_nonzero(tmp_path):
    import hashlib,json,importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location("resting_collection_cli",
        Path(__file__).resolve().parents[1]/"tools/verify_resting_anatomy.py")
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    pack=tmp_path/"pack";receipt=tmp_path/"receipt";output=tmp_path/"result.json"
    pack.write_bytes(b"capture");receipt.write_bytes(b"receipt")
    a=surface((51011,7));b=surface((51011,8),.2,.1);far=surface((51011,10),offset=5.)
    surfaces={a.key:a,b.key:b,far.key:far}
    args=["--pack",str(pack),"--pack-sha256",hashlib.sha256(pack.read_bytes()).hexdigest(),
          "--capture-receipt",str(receipt),"--receipt-sha256",hashlib.sha256(receipt.read_bytes()).hexdigest(),
          "--changed","51011:7","--target","51011:8","--target","51011:10",
          "--collect-failures","--output",str(output)]
    with patch.object(cli.verify,"load_capture",return_value=(surfaces,{})):
        assert cli.main(args)==1
    report=json.loads(output.read_text())
    assert not report["passed"] and report["collection_complete"]
    assert not report["native_admission"] and not report["anatomical_admission"]
    assert report["rows"][-1]["passed"] and report["failure_count"]==1

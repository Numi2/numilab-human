#!/usr/bin/env python3
"""Independent current-source promoter for saved-operation rebase 809.

It authenticates the saved 806 journal and current state ancestry, checks that
the frozen 809 builder's legacy delegate pin is intact, then delegates source,
topology, cavity, quality and readback validation to a versioned 796 derivative
using the recursive 807 source verifier. It writes nothing before both gates pass.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
BUILDER=HERE/"rebase_saved_partial_ops_onto_current_f32_809.py"
BUILDER_SHA="f81a7bdd6b0b26b7c2a8546b7e879c36774d94f08e09691fbb40de5977fa82a5"
# The frozen 809 builder writes this legacy delegate into its journal. Verify it
# as producer provenance, then use the new 796/807 derivative for promotion.
BUILDER_PROMOTER=BASE/"quality-partial-cavity-684-root"/"independent_promoter_796_795_root.py"
BUILDER_PROMOTER_SHA="320fcbb6f1eb8147c67a27474c4b23100e92bca060122b3dffc37aa851bbcfec"
PROMOTER=HERE/"independent_promoter_796_807.py"
PROMOTER_SHA="5a6e2df084a097448172171cca05e1b0f6e281d64302c5592987ca9630c1a7ce"
SOURCE_806_SHA="2d6b25746661c9b98b29b5ebc930f1c1262892bca320829d1ec4a9af8133e44c"
REBASE_SCHEMA="h706.partial-quality-steiner-current-point-rebase.v1"
PASS_ROLE="PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE_BATCH"
STATE_WRAPPER="current_verified_state_20261031"

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()
def pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}
def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
def loadjson(path):
    return json.loads(Path(path).read_text())
def _verify_pin(row,label):
    if not isinstance(row,dict) or not row.get("path") or not row.get("sha256"):
        raise ValueError(label+" pin missing")
    p=Path(row["path"])
    if not p.is_file() or sha(p)!=row["sha256"]:
        raise ValueError(label+" pin mismatch")
    return p.resolve()
def _unwrap(raw):
    if STATE_WRAPPER in raw:return raw[STATE_WRAPPER]
    return raw
def _authenticated_chain(state_path,state,target_pin):
    target_path=Path(target_pin["path"]).resolve()
    if sha(target_path)!=target_pin["sha256"]:
        raise ValueError("saved 806 source-state ancestor pin changed")
    chain=[pin(state_path)];seen={Path(state_path).resolve()};current=state
    for _ in range(8):
        link=current.get("source_state")
        if not isinstance(link,dict):raise ValueError("current verified-state ancestry ended early")
        p=_verify_pin(link,"verified-state ancestor")
        if p in seen:raise ValueError("verified-state ancestry cycle")
        seen.add(p);normalized={"path":str(p),"sha256":sha(p)};chain.append(normalized)
        if p==target_path and normalized["sha256"]==target_pin["sha256"]:return chain
        current=_unwrap(loadjson(p))
    raise ValueError("verified-state ancestry exceeds bounded depth")

if sha(BUILDER)!=BUILDER_SHA:raise RuntimeError("809 builder pin mismatch")
if sha(BUILDER_PROMOTER)!=BUILDER_PROMOTER_SHA:raise RuntimeError("frozen 809 builder delegate pin mismatch")
if sha(PROMOTER)!=PROMOTER_SHA:raise RuntimeError("796/807 promoter derivative pin mismatch")
P=imp("rebase809_delegate796_807",PROMOTER)

def validate_rebase_envelope(state_path,journal_path,journal=None):
    state_path=Path(state_path).resolve();journal_path=Path(journal_path).resolve()
    state_raw=loadjson(state_path);state=_unwrap(state_raw)
    if journal is None:journal=loadjson(journal_path)
    manifest=journal.get("rebase_manifest")
    if not isinstance(manifest,dict) or manifest.get("schema")!=REBASE_SCHEMA:
        raise ValueError("journal lacks the typed current-point rebase manifest")
    if manifest.get("status")!="PASS_REBASE_VERIFIED_TO_CURRENT_F32":
        raise ValueError("rebase manifest is not passing")
    if manifest.get("physical_admission") is not False or int(manifest.get("native_steps",-1))!=0:
        raise ValueError("rebase manifest overstates physical/native admission")
    if manifest.get("current_coordinates_used") is not True or manifest.get("failed_operations_pruned") is not True:
        raise ValueError("rebase manifest lacks current-coordinate replay/pruning declaration")
    if manifest.get("operation_reoptimization")!="NOT_RUN":
        raise ValueError("rebase manifest misstates the saved-operation rebase")
    if manifest.get("current_state_sha256")!=sha(state_path):
        raise ValueError("rebase manifest current state SHA mismatch")
    current_pin=manifest.get("current_state")
    _verify_pin(current_pin,"rebase current state")
    if Path(current_pin["path"]).resolve()!=state_path:
        raise ValueError("rebase current-state path differs from supplied state")
    if journal.get("source_state")!=current_pin:
        raise ValueError("rebased journal source state differs from current state pin")
    saved_pin=manifest.get("saved_journal")
    saved_path=_verify_pin(saved_pin,"saved 806 journal")
    if sha(saved_path)!=SOURCE_806_SHA:
        raise ValueError("rebase manifest saved journal is not pinned 806")
    if saved_path==journal_path:
        raise ValueError("rebase journal cannot alias its saved input journal")
    old=loadjson(saved_path)
    if old.get("status")!="PASS_WITH_PARTIAL_NATIVE_DEFECTS" or old.get("native_thresholds_unchanged") is not True:
        raise ValueError("saved 806 source journal is not an accepted source-only result")
    if manifest.get("saved_journal_source_state")!=old.get("source_state"):
        raise ValueError("rebase manifest saved-source ancestry mismatch")
    expected_chain=_authenticated_chain(state_path,state,old["source_state"])
    if manifest.get("source_ancestry_chain")!=expected_chain:
        raise ValueError("rebase manifest does not bind the exact current-to-799 state ancestry")
    builder_pin=manifest.get("rebase_builder")
    if _verify_pin(builder_pin,"809 builder")!=BUILDER.resolve() or builder_pin["sha256"]!=BUILDER_SHA:
        raise ValueError("rebase manifest builder pin mismatch")
    delegate_pin=manifest.get("independent_promoter")
    if (_verify_pin(delegate_pin,"frozen 809 builder delegate")!=BUILDER_PROMOTER.resolve()
            or delegate_pin["sha256"]!=BUILDER_PROMOTER_SHA):
        raise ValueError("rebase manifest does not preserve the frozen 809 builder delegate provenance")
    if journal.get("producer",{}).get("path")!=str((BASE/"quality-partial-cavity-684-root"/"batch_interior_steiner_partial_quality_795_root.py").resolve()):
        raise ValueError("rebased journal does not retain pinned 795 producer path")
    if journal.get("producer",{}).get("sha256")!="4e35df53981c2fedc0f03b268138ebf74378a09e0c55d1f7a64b8e09c813f3bf":
        raise ValueError("rebased journal does not retain pinned 795 producer SHA")
    if journal.get("native_thresholds_unchanged") is not True or journal.get("physical_admission") is not False or int(journal.get("native_steps",-1))!=0:
        raise ValueError("rebased journal overstates native/physical status")
    results=journal.get("role_results")
    rebases=manifest.get("role_rebases")
    if not isinstance(results,list) or not isinstance(rebases,list):
        raise ValueError("rebase journal lacks role results/crosswalk")
    result_by={x.get("role"):x for x in results}
    rebase_by={x.get("role"):x for x in rebases}
    expected={r for r in ("Skin","Fat") if r+"_current" in state}
    if set(result_by)!=expected or set(rebase_by)!=expected:
        raise ValueError("rebased journal role coverage differs from current state")
    role_checks=[]
    for role in sorted(expected):
        rr=result_by[role];rb=rebase_by[role];entry=state[role+"_current"]
        if rr.get("input")!={"path":entry.get("path"),"sha256":entry.get("sha256")}:
            raise ValueError(role+" result is not bound to current state role input")
        if rb.get("current_parent")!=rr.get("input"):
            raise ValueError(role+" current parent differs from role input")
        old_role=next((x for x in old.get("role_results",[]) if x.get("role")==role),None)
        if old_role is None or rb.get("source_806_result")!=pin(saved_path):
            raise ValueError(role+" role does not bind to the saved 806 result")
        if rb.get("source_806_output")!=old_role.get("output"):
            raise ValueError(role+" role saved-output pin differs from 806 journal")
        if rb.get("old_806_operation_count")!=len(old_role.get("accepted_operations",[])):
            raise ValueError(role+" saved operation count differs")
        if rr.get("status")==PASS_ROLE:
            if rb.get("status")!="PASS_REBASED_ON_CURRENT_F32":
                raise ValueError(role+" role rebase did not pass")
            for key in ("rebased_output","independent_readback"):
                result_key={"rebased_output":"output","independent_readback":"independent_readback"}[key]
                actual=rr.get(result_key)
                if rb.get(key)!=actual:
                    raise ValueError(role+" role rebase crosswalk mismatch: "+key)
                _verify_pin(actual,role+" "+key)
            if int(rb.get("replayed_operation_count",-1))!=len(rr.get("accepted_operations",[])):
                raise ValueError(role+" operation count differs between rebase and role journal")
            if int(rb.get("rejected_operation_count",-1))!=len(rr.get("rebase_rejections",[])):
                raise ValueError(role+" rejected-operation count differs between rebase and role journal")
            readback=loadjson(rr["independent_readback"]["path"])
            if readback.get("status")!="PASS_INDEPENDENT_STEINER_CAVITY_READBACK":
                raise ValueError(role+" current-point independent readback status is not passing")
            role_checks.append({"role":role,"status":"PASS_REBASED_ON_CURRENT_F32",
                "accepted_operations":len(rr["accepted_operations"]),
                "pruned_operations":len(rr.get("rebase_rejections",[])),
                "current_input":rr["input"],"current_output":rr["output"],
                "independent_readback":rr["independent_readback"]})
        elif rr.get("status") in P.NOOP_STATUSES:
            if rb.get("status")!="PASS_NO_OP_REBASED_CURRENT_F32":
                raise ValueError(role+" no-op role rebase status mismatch")
            if "output" in rr or "accepted_operations" in rr or "rebased_output" in rb or "independent_readback" in rb:
                raise ValueError(role+" no-op unexpectedly carries a proposal")
            if int(rr.get("native_bad_before",-1))!=int(rr.get("native_bad_after",-2)):
                raise ValueError(role+" no-op changed native bad count")
            if float(rr.get("log_deficit_before",-1))!=float(rr.get("log_deficit_after",-2)):
                raise ValueError(role+" no-op changed log deficit")
            if int(rb.get("replayed_operation_count",-1))!=0 or int(rb.get("rejected_operation_count",-1))!=int(rb.get("old_806_operation_count",-2)):
                raise ValueError(role+" no-op pruning counts mismatch")
            role_checks.append({"role":role,"status":"PASS_NO_OP_REBASED_CURRENT_F32",
                "accepted_operations":0,"pruned_operations":rb["rejected_operation_count"],
                "current_input":rr["input"]})
        else:
            raise ValueError(role+" role has unsupported rebase result status")
    return {"status":"PASS_REBASE_ENVELOPE","schema":REBASE_SCHEMA,
            "current_state":pin(state_path),"saved_journal":pin(saved_path),"rebased_journal":pin(journal_path),
            "builder":pin(BUILDER),"builder_delegate":pin(BUILDER_PROMOTER),
            "active_delegate":pin(PROMOTER),"roles":role_checks,
            "physical_admission":False,"native_steps":0}

def promote(state_path,journal_path,output_state,receipt_path):
    start=time.monotonic()
    state_path=Path(state_path).resolve();journal_path=Path(journal_path).resolve()
    output_state=Path(output_state).resolve();receipt_path=Path(receipt_path).resolve()
    if output_state==receipt_path:raise ValueError("state and receipt must be separate")
    if output_state.exists() or receipt_path.exists():raise FileExistsError("refusing to overwrite promotion outputs")
    output_state.parent.mkdir(parents=True,exist_ok=True);receipt_path.parent.mkdir(parents=True,exist_ok=True)
    if os.stat(output_state.parent).st_dev!=os.stat(receipt_path.parent).st_dev:
        raise ValueError("state and receipt targets must be on the same filesystem")
    journal=loadjson(journal_path)
    envelope=validate_rebase_envelope(state_path,journal_path,journal)
    stage=Path(tempfile.mkdtemp(prefix=".rebase-promote-808-",dir=output_state.parent))
    try:
        tmp_state=stage/"verified-state.json";tmp_receipt=stage/"receipt.json"
        base_receipt=P.verify_and_promote(state_path,journal_path,tmp_state,tmp_receipt)
        raw=loadjson(tmp_state);nested=_unwrap(raw)
        nested["quality_steiner_810_rebase_verification"]={
            "schema":REBASE_SCHEMA,"status":"PASS_REBASE_AND_796_INDEPENDENT_PROMOTION",
            "rebase_envelope":envelope,"physical_admission":False,"native_steps":0,
            "mass_mapping":"NOT_RUN"}
        if STATE_WRAPPER in raw:raw[STATE_WRAPPER]=nested
        else:raw=nested
        tmp_state.write_text(json.dumps(raw,indent=2,sort_keys=True)+"\n")
        receipt=loadjson(tmp_receipt)
        receipt["verified_state"]={"path":str(output_state),"sha256":sha(tmp_state)}
        receipt["rebase_verification"]={
            "schema":REBASE_SCHEMA,"status":"PASS_REBASE_AND_796_807_INDEPENDENT_PROMOTION",
            "rebase_envelope":envelope,"promoter":pin(Path(__file__)),
            "builder_delegate":pin(BUILDER_PROMOTER),"active_delegate":pin(PROMOTER),
            "elapsed_s":float(time.monotonic()-start)}
        tmp_receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        # Stage both validated outputs before publishing either; target names are new.
        os.replace(tmp_state,output_state)
        os.replace(tmp_receipt,receipt_path)
        return {"status":receipt["status"],"verified_state":pin(output_state),
                "receipt":pin(receipt_path),"rebase_envelope":envelope,
                "physical_admission":False,"native_steps":0}
    finally:
        shutil.rmtree(stage,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--journal",type=Path,required=True)
    ap.add_argument("--output-state",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(promote(a.state,a.journal,a.output_state,a.receipt),sort_keys=True))
if __name__=="__main__":main()


#!/usr/bin/env python3
"""Disk-backed independent verifier/promoter for complete-star lexicographic proposals.

Verified-state output is written only after the 791 source-evidence chain and
796 independent full-role replay both pass. No geometry, mass mapping, or
physical state is admitted here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRODUCER=HERE/"batch_complete_vertex_star_lex_quality_milp_795_root.py"
REPLAY=HERE/"independent_replay_complete_vertex_star_lex_quality_796_root.py"
SOURCE_EVIDENCE_PATH=BASE/"verify_current_source_evidence_791_root.py"
EXPECTED_PRODUCER_SHA="38aedcd643ab059f215366a511e501787043f0abaa0b3e57c93dae8e04dbfa74"
EXPECTED_REPLAY_SHA="805f76520f9bdd2437c7473ed8e251eac282021a820d35b0680ff70ebe82cb6e"
EXPECTED_SOURCE_EVIDENCE_SHA="b4da6ac3509767e11f3df6bd712a9908adf6b9bd0c1efc2122b16ec15df04ed9"
EXPECTED_SOLVER_SHA="20eae93eff31493702eda00adfd80cd812e31f673af6ee8b469444f7602fc251"
EXPECTED_SCHEMA="h706.complete-vertex-star-lexicographic-quality-milp-batch.v1"
PASS_STATUS="PASS_COMPLETE_VERTEX_STAR_LEXICOGRAPHIC_QUALITY_BATCH"
NOOP_STATUSES={
    "NO_NATIVE_BAD_CELLS",
    "NO_VALID_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_PROGRESS",
    "NO_STRICT_GLOBAL_LEXICOGRAPHIC_QUALITY_GAIN",
    "SKIPPED_GLOBAL_TIME_BUDGET",
}
STATE_WRAPPER="current_verified_state_20261031"

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()

def pin(path):
    p=Path(path)
    return {"path":str(p),"sha256":sha(p)}

def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned helper: "+str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

for path,expected,label in (
    (PRODUCER,EXPECTED_PRODUCER_SHA,"795 producer"),
    (REPLAY,EXPECTED_REPLAY_SHA,"796 independent replay"),
    (SOURCE_EVIDENCE_PATH,EXPECTED_SOURCE_EVIDENCE_SHA,"791 source proof"),
):
    if not path.is_file() or sha(path)!=expected:
        raise ValueError("pinned "+label+" is missing or changed")
REPLAY_MOD=imp("complete_star_lex_replay_for797",REPLAY)
SOURCE_MOD=imp("complete_star_lex_source_proof791_for797",SOURCE_EVIDENCE_PATH)

def unwrap(raw):
    if STATE_WRAPPER in raw:
        return raw[STATE_WRAPPER],True
    return raw,False

def check_pin(record,label,expected_path=None):
    if not isinstance(record,dict) or "path" not in record or "sha256" not in record:
        raise ValueError(label+" pin is malformed")
    path=Path(str(record["path"]))
    if not path.is_file() or sha(path)!=str(record["sha256"]):
        raise ValueError(label+" file pin mismatch")
    if expected_path is not None and path.resolve()!=Path(expected_path).resolve():
        raise ValueError(label+" path mismatch")
    return path

def _report_roles(report):
    rows=report.get("role_results")
    if not isinstance(rows,list):
        raise ValueError("795 report lacks role_results")
    result={}
    for row in rows:
        role=str(row.get("role",""))
        if role not in ("Skin","Fat") or role in result:
            raise ValueError("795 report has unknown or duplicate role")
        result[role]=row
    return result

def verify_and_promote(state_path,journal_path,output_path,receipt_path,replay_dir=None):
    started=time.monotonic()
    state_path=Path(state_path).resolve()
    journal_path=Path(journal_path).resolve()
    output_path=Path(output_path).resolve()
    receipt_path=Path(receipt_path).resolve()
    if output_path==receipt_path:
        raise ValueError("verified-state and receipt paths must differ")
    raw=json.loads(state_path.read_text())
    parent,wrapped=unwrap(raw)
    if not isinstance(parent,dict):
        raise ValueError("input verified-state is not an object")
    state_pin=pin(state_path)
    journal_pin=pin(journal_path)
    source_proof=SOURCE_MOD.verify_current_source_evidence(state_path,parent)
    report=json.loads(journal_path.read_text())
    if report.get("schema")!=EXPECTED_SCHEMA:
        raise ValueError("795 journal schema mismatch")
    report_state=report.get("source_state",{})
    if Path(str(report_state.get("path",""))).resolve()!=state_path:
        raise ValueError("795 journal source state path mismatch")
    if report_state.get("sha256")!=state_pin["sha256"]:
        raise ValueError("795 journal source state digest mismatch")
    if report.get("native_thresholds_unchanged") is not True:
        raise ValueError("795 journal does not preserve native thresholds")
    if report.get("quality_progress_policy")!="strict_bad_count_drop_or_equal_count_with_752_robust_log_deficit_drop":
        raise ValueError("795 journal lexicographic policy mismatch")
    producer=check_pin(report.get("producer"),"producer",PRODUCER)
    if sha(producer)!=EXPECTED_PRODUCER_SHA:
        raise ValueError("795 producer digest mismatch")
    solver=check_pin(report.get("solver"),"solver")
    if sha(solver)!=EXPECTED_SOLVER_SHA:
        raise ValueError("795 exact solver digest mismatch")
    helper=check_pin(report.get("source_evidence_helper"),"source-evidence helper",SOURCE_EVIDENCE_PATH)
    if report.get("source_evidence") is None or report["source_evidence"].get("status") not in (
            source_proof.get("status"),"PASS_INHERITED_CURRENT_SOURCE_EVIDENCE"):
        raise ValueError("795 source proof does not match current 791 evidence")
    output_state=check_pin(report.get("output_state"),"raw proposal state")
    raw_output_state=json.loads(output_state.read_text())
    proposed_state,_=unwrap(raw_output_state)
    role_results=_report_roles(report)
    expected_roles={r for r in ("Skin","Fat") if r+"_current" in parent}
    if set(role_results)!=expected_roles:
        raise ValueError("795 role coverage differs from input state")
    for role in expected_roles:
        key=role+"_current"
        proposed_entry=proposed_state.get(key,{})
        result_entry=role_results[role]
        if result_entry.get("status")==PASS_STATUS:
            proposed_output=result_entry.get("output",{})
            if proposed_entry.get("path")!=proposed_output.get("path") or proposed_entry.get("sha256")!=proposed_output.get("sha256"):
                raise ValueError(role+" raw proposal state does not bind its journal candidate")
        elif proposed_entry.get("path")!=parent[key].get("path") or proposed_entry.get("sha256")!=parent[key].get("sha256"):
            raise ValueError(role+" no-op raw proposal state changed its input candidate")
    replay_dir=Path(replay_dir) if replay_dir is not None else journal_path.parent/"replay"
    replay_dir.mkdir(parents=True,exist_ok=True)
    updated=dict(parent)
    updated["source_state"]=state_pin
    role_receipts=[]
    progress_roles=[]
    for role in sorted(expected_roles):
        entry=parent[role+"_current"]
        rr=role_results[role]
        status=str(rr.get("status",""))
        if status=="SKIPPED_GLOBAL_TIME_BUDGET":
            if "output" in rr:
                raise ValueError(role+" skipped role unexpectedly has a candidate output")
            role_receipts.append({
                "role":role,"status":"PASS_INHERITED_SOURCE_BUDGET_SKIP",
                "input":{"path":str(entry["path"]),"sha256":entry["sha256"]},
                "reason":status,
            })
            continue
        input_pin=rr.get("input",{})
        if Path(str(input_pin.get("path",""))).resolve()!=Path(str(entry.get("path",""))).resolve():
            raise ValueError(role+" journal input candidate path mismatch")
        if input_pin.get("sha256")!=entry.get("sha256") or sha(entry["path"])!=entry["sha256"]:
            raise ValueError(role+" journal input candidate digest mismatch")
        if status==PASS_STATUS:
            output=check_pin(rr.get("output"),role+" proposal")
            if rr.get("state_input_sha256")!=state_pin["sha256"]:
                raise ValueError(role+" proposal is bound to another state")
            map_path=replay_dir/(role.lower()+"-boundary-row-map.npz")
            if map_path.exists():
                raise FileExistsError("refusing to overwrite replay map: "+str(map_path))
            replay_receipt=REPLAY_MOD.verify(
                state_path,journal_path,role,map_path,allow_synthetic_fixture=False)
            if replay_receipt.get("status")!="PASS_INDEPENDENT_COMPLETE_VERTEX_STAR_LEXICOGRAPHIC_QUALITY_REPLAY":
                raise ValueError(role+" independent 796 replay did not pass")
            if replay_receipt["output_candidate"]["sha256"]!=sha(output):
                raise ValueError(role+" replay output pin differs from journal")
            old_bad=int(replay_receipt["whole_input_native_bad_cells"])
            new_bad=int(replay_receipt["whole_output_native_bad_cells"])
            old_score=float(replay_receipt["whole_role_log_deficit_before"])
            new_score=float(replay_receipt["whole_role_log_deficit_after"])
            if new_bad>old_bad or (new_bad==old_bad and
                    old_score-new_score<=max(1.0e-8,1.0e-6*old_score)):
                raise ValueError(role+" independent whole-role lexicographic gate failed")
            updated[role+"_current"]={
                **entry,
                "path":str(output),
                "sha256":sha(output),
                "nodes":int(replay_receipt["output_points"]),
                "cells":int(replay_receipt["output_cells"]),
                "native_quality_failures":new_bad,
                "nonpositive":0,
                "oriented_volume_target_boundary_equal":True,
                "target_boundary_match":True,
            }
            progress_roles.append(role)
            role_receipts.append({
                "role":role,"status":"PASS_EXTERNAL_INDEPENDENT_LEXICOGRAPHIC_REPLAY",
                "proposal":pin(output),"replay":replay_receipt,
                "native_bad_before":old_bad,"native_bad_after":new_bad,
                "log_deficit_before":old_score,"log_deficit_after":new_score,
                "log_deficit_gain":old_score-new_score,
            })
        elif status in NOOP_STATUSES:
            if "output" in rr:
                raise ValueError(role+" no-op unexpectedly has a candidate output")
            declared=int(entry.get("native_quality_failures",-1))
            before=int(rr.get("native_bad_before",rr.get("bad_before",declared)))
            after=int(rr.get("native_bad_after",rr.get("bad_after",before)))
            if before!=declared or after>before:
                raise ValueError(role+" no-op native count does not match the parent")
            role_receipts.append({
                "role":role,"status":"PASS_INHERITED_SOURCE_NOOP",
                "input":{"path":str(entry["path"]),"sha256":entry["sha256"]},
                "native_bad_before":before,"native_bad_after":after,
                "reason":status,
            })
        else:
            raise ValueError(role+" has unsupported 795 role status: "+status)
    if not progress_roles:
        raise ValueError("795 journal contains no independently replayed lexicographic progress")
    updated["physical_admission"]=False
    updated["native_admission"]=False
    updated["native_steps"]=0
    updated["mass_mapping"]="NOT_RUN"
    if "independent_replay" in updated:
        updated["quality_complete_vertex_star_795_parent_independent_replay"]=updated.pop("independent_replay")
    if "exact_context" in updated:
        key="quality_complete_vertex_star_795_parent_exact_context"
        if key in updated:
            key="quality_complete_vertex_star_795_parent_exact_context_prior"
        updated[key]=updated.pop("exact_context")
    promotion={
        "schema":"h706.independent-complete-vertex-star-lexicographic-promoter.v1",
        "status":"PASS_EXTERNAL_INDEPENDENT_LEXICOGRAPHIC_REPLAY",
        "source_state":state_pin,
        "proposal_journal":journal_pin,
        "producer":pin(PRODUCER),
        "solver":pin(solver),
        "independent_replay_owner":pin(REPLAY),
        "source_evidence_helper":pin(helper),
        "source_proof":source_proof,
        "progress_roles":progress_roles,
        "role_receipts":role_receipts,
        "native_thresholds_unchanged":True,
        "native_admission":False,"physical_admission":False,"native_steps":0,
        "mass_mapping":"NOT_RUN",
    }
    updated["quality_complete_vertex_star_795_promotion"]=promotion
    final_state=dict(raw)
    if wrapped:
        final_state[STATE_WRAPPER]=updated
    else:
        final_state=updated
    for path in (output_path,receipt_path):
        if path.exists():
            raise FileExistsError("refusing to overwrite "+str(path))
    proposal_state_path=output_path.with_name(output_path.stem+"-source-check.json")
    if proposal_state_path.exists():
        raise FileExistsError("refusing to overwrite "+str(proposal_state_path))
    proposal_state_path.parent.mkdir(parents=True,exist_ok=True)
    proposal_state_path.write_text(json.dumps(final_state,indent=2,sort_keys=True)+"\n")
    promoted_proof=SOURCE_MOD.verify_current_source_evidence(proposal_state_path)
    if not str(promoted_proof.get("status","")).startswith("PASS_"):
        raise ValueError("791 did not authenticate promoted state source ancestry")
    promotion["promoted_state_source_proof"]=promoted_proof
    updated["quality_complete_vertex_star_795_promotion"]=promotion
    final_state=dict(raw)
    if wrapped:
        final_state[STATE_WRAPPER]=updated
    else:
        final_state=updated
    # Verify the exact bytes to be promoted through the same source ancestry gate.
    temp_state=output_path.with_name(output_path.name+".sourcecheck")
    if temp_state.exists():
        raise FileExistsError("refusing to overwrite "+str(temp_state))
    temp_state.write_text(json.dumps(final_state,indent=2,sort_keys=True)+"\n")
    final_proof=SOURCE_MOD.verify_current_source_evidence(temp_state)
    if not str(final_proof.get("status","")).startswith("PASS_"):
        raise ValueError("791 did not authenticate final verified-state source ancestry")
    os.replace(temp_state,output_path)
    receipt={
        "schema":"h706.independent-complete-vertex-star-lexicographic-promoter.v1",
        "status":"PASS_EXTERNAL_INDEPENDENT_LEXICOGRAPHIC_REPLAY",
        "source_state":state_pin,"proposal_journal":journal_pin,
        "raw_proposal_state":pin(output_state),
        "verified_state":pin(output_path),
        "source_proof_before":source_proof,
        "source_proof_after":final_proof,
        "producer":pin(PRODUCER),"solver":pin(solver),
        "independent_replay_owner":pin(REPLAY),
        "source_evidence_helper":pin(helper),
        "role_receipts":role_receipts,"progress_roles":progress_roles,
        "native_admission":False,"physical_admission":False,"native_steps":0,
        "mass_mapping":"NOT_RUN","elapsed_s":float(time.monotonic()-started),
    }
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--journal",type=Path,required=True)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--receipt",type=Path)
    ap.add_argument("--replay-directory",type=Path)
    args=ap.parse_args()
    output=args.output or args.journal.parent/"verified-state.json"
    receipt=args.receipt or args.journal.parent/"independent-promoter-797-report.json"
    result=verify_and_promote(args.state,args.journal,output,receipt,args.replay_directory)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()

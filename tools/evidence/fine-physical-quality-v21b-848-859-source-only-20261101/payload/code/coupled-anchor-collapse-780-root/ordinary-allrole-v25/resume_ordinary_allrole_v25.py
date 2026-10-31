#!/usr/bin/env python3
"""Resume a saved v25 proposal through 625 rollback, 775 audit, and independent replay."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BUILDER=HERE/"repair_coupled_quality_points_722_v25_ordinary_allrole_local.py"
BUILDER_SHA="cf17d106e83450af02dbd87b1c6c1730b1a5519e170c359f0b0525b7e8392742"
REPLAY=HERE/"verify_ordinary_allrole_v25_independent_replay_v2.py"
REPLAY_SHA="a241f6684814464fbef90340f40c40bd7c941e9a0a14a8de4858057b0b224c4f"
V24_PATH=ROOT/"continue_coupled_quality_points_v24c_ordinary_outer_o_local.py"
V24_SHA="f69ab97f020436c5f0afa997a770ad291bad9ee63942522d0ea0b3c49230074a"
AUDITOR=ROOT/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
AUDITOR_SHA="82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
PREPARER=ROOT/"bounded625-local-cavity-repair/prepare_bounded_rollback_625_inner805.py"
PREPARER_SHA="fe99ede9eff7449e558b7f935d791dee1c3384c5159b2e9d8ef2b10e509ab1d7"
FINALIZER=ROOT/"bounded625-local-cavity-repair/finalize_v7_rollback_805.py"
FINALIZER_SHA="562c778b73d90ddbcf7d7aa511f6f8bbc6d79cd19773bd19151685a499010813"
ZERO_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return {"path":str(p),"sha256":sha(p)}
def checked(p,h,label):
    got=sha(p)
    if got!=h:raise ValueError(label+" SHA mismatch: "+got+" != "+h)
def imported_v24():
    checked(V24_PATH,V24_SHA,"v24 owner")
    spec=importlib.util.spec_from_file_location("v24_owner_for_allrole25_resume",V24_PATH)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import v24 owner")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
V=imported_v24()
def run(label,cmd,logdir):
    logdir.mkdir(parents=True,exist_ok=True);t=time.monotonic()
    p=subprocess.run(cmd,capture_output=True,text=True,check=False)
    (logdir/(label+".stdout.log")).write_text(p.stdout)
    (logdir/(label+".stderr.log")).write_text(p.stderr)
    (logdir/(label+".command.json")).write_text(json.dumps({"command":cmd,"returncode":p.returncode,"elapsed_s":time.monotonic()-t},indent=2)+"\n")
    if p.returncode:raise RuntimeError(label+" exited "+str(p.returncode)+": "+(p.stderr or p.stdout)[-4000:])
def audit(config,out,logs):
    run("audit775",[sys.executable,str(AUDITOR),"--config",str(config),"--out",str(out)],logs)
    report=out/"report.json"
    if not report.is_file():raise RuntimeError("775 auditor emitted no report")
    doc=json.loads(report.read_text())
    V.validate_report_config_pin(report,config.resolve())
    against=doc.get("changed_vs_complete_retained_OID_both_walls_all185",{})
    selfp=doc.get("changed_OID_joint_self_pairs",{})
    n=int(against.get("count",-1));s=int(selfp.get("count",-1))
    if n<0 or s<0:raise ValueError("775 report lacks complete context/self counts")
    return report,{"status":doc.get("status"),"context_pairs":n,"joint_self_pairs":s},doc.get("status")==ZERO_STATUS and n==0 and s==0
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path,required=True)
    ap.add_argument("--audit-config",type=Path,required=True)
    ap.add_argument("--proposal-audit-report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    for p,h,label in ((BUILDER,BUILDER_SHA,"v25 producer"),(REPLAY,REPLAY_SHA,"v25 independent replay"),
        (AUDITOR,AUDITOR_SHA,"775 auditor"),(PREPARER,PREPARER_SHA,"625 preparer"),
        (FINALIZER,FINALIZER_SHA,"625 finalizer")):
        checked(p,h,label)
    parentp=a.parent_state.resolve();propp=a.proposal_state.resolve();cfgp=a.audit_config.resolve();auditp=a.proposal_audit_report.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    parent=V.load_state(parentp);proposal=V.load_state(propp)
    V.validate_proposal_parent_pin(proposal,parentp)
    source=V.verify_source_input(parentp)
    if not str(source.get("status","PASS_")).startswith("PASS_"):raise ValueError("807 source proof failed for parent")
    meta=proposal.get("coupled_point_repair",{})
    if meta.get("ordinary_all_role_local_repair") is not True or meta.get("ordinary_all_role_policy")!="baseline-bad volume corners plus one direct source-triangle ring over O/I/D; current/historical actual Fat back, all scope, rims and registered routes remain fixed":
        raise ValueError("saved proposal lacks exact ordinary-all-role v25 policy")
    cap=float(meta.get("maximum_displacement_m",0.0))
    if not 0<cap<=5e-4:raise ValueError("saved proposal has invalid trust cap")
    checked(BUILDER,BUILDER_SHA,"proposal producer")
    producer=meta.get("producer",{})
    if producer.get("sha256")!=BUILDER_SHA or Path(producer.get("path","" )).resolve()!=BUILDER.resolve():
        raise ValueError("saved proposal producer pin differs from v25 builder")
    cfg=json.loads(cfgp.read_text())
    for role in ("O","I","D"):
        b=cfg.get("baseline_candidates",{}).get(role,{})
        c=cfg.get("candidate_surfaces",{}).get(role,{})
        if b.get("path")!=parent["source_surfaces"][role]["path"] or b.get("sha256")!=parent["source_surfaces"][role]["sha256"]:
            raise ValueError(role+" audit baseline differs from parent")
        if c.get("path")!=proposal["source_surfaces"][role]["path"] or c.get("sha256")!=proposal["source_surfaces"][role]["sha256"]:
            raise ValueError(role+" audit candidate differs from saved proposal")
    V.validate_report_config_pin(auditp,cfgp)
    failed=json.loads(auditp.read_text())
    if failed.get("status")==ZERO_STATUS:raise ValueError("saved report is already clean; use direct independent replay")
    if not str(failed.get("status","")).startswith("FAIL"):
        raise ValueError("saved 775 report is not an explicit failed witness report")
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    rollback=out/"witness-component-rollback625"
    run("prepare625",[sys.executable,str(PREPARER),"--state",str(propp),"--config",str(cfgp),"--audit",str(auditp),"--output",str(rollback)],out/"rollback-preparation-logs")
    rolled=rollback/"state.json";rcfg=rollback/"context-config.json";plan=rollback/"rollback-plan.json";prep=rollback/"rollback-preparation-receipt.json"
    if not all(p.is_file() for p in (rolled,rcfg,plan,prep)):raise RuntimeError("625 preparation outputs incomplete")
    final_report,final_summary,clean=audit(rcfg,out/"rollback-exact-775",out/"rollback-audit-logs")
    if not clean:raise RuntimeError("775 still reports source crossings after whole-component rollback")
    finalized=out/"rollback-finalized"
    run("finalize625",[sys.executable,str(FINALIZER),"--state",str(rolled),"--baseline-state",str(parentp),
        "--proposal-state",str(propp),"--rollback-receipt",str(prep),"--audit-report",str(final_report),
        "--audit-config",str(rcfg),"--output-dir",str(finalized)],out/"rollback-finalizer-logs")
    selected=finalized/"state.json"
    if not selected.is_file():raise RuntimeError("625 finalizer did not emit state.json")
    replay=out/"independent-allrole-replay.json"
    run("independent-replay-v2",[sys.executable,str(REPLAY),"--candidate-state",str(selected),
        "--parent-state",str(parentp),"--audit-report",str(final_report),"--audit-config",str(rcfg),
        "--proposal-state",str(propp),"--rollback-plan",str(plan),"--output-report",str(replay)],out/"independent-replay-logs")
    replay_doc=json.loads(replay.read_text())
    if replay_doc.get("status")!="PASS_775_SOURCE_CONTEXT_VOLUME_QUALITY_AND_V25_FREE_SET":raise RuntimeError("independent replay failed")
    base_replay=replay.with_name(replay.stem+"-base-volume-replay.json")
    final_state=out/"finalized-current-state.json"
    V.finalize_verified_state(selected,final_report,base_replay,final_state)
    final=json.loads(final_state.read_text());final["independent_allrole_local_replay"]=pin(replay)
    final["physical_admission"]=False;final["native_steps"]=0
    final["mass_reconciliation"]={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False}
    final_state.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    final_source=V.verify_source_input(final_state)
    receipt={"schema":"h706.ordinary-allrole-v25-resume-rollback.v1","status":"PASS_SAVED_PROPOSAL_625_ROLLBACK_775_AND_INDEPENDENT_REPLAY",
        "parent_state":pin(parentp),"saved_proposal_state":pin(propp),"proposal_config":pin(cfgp),
        "failed_775_report":pin(auditp),"failed_775_summary":{"status":failed.get("status"),
            "context_pairs":failed.get("changed_vs_complete_retained_OID_both_walls_all185",{}).get("count"),
            "joint_self_pairs":failed.get("changed_OID_joint_self_pairs",{}).get("count")},
        "rollback_preparation_receipt":pin(prep),"rollback_plan":pin(plan),
        "passing_775_report":pin(final_report),"passing_775_summary":final_summary,
        "independent_replay":pin(replay),"finalized_state":pin(final_state),"final_807_source_proof_status":final_source.get("status"),
        "implementation_pins":{"producer":pin(BUILDER),"independent_replay":pin(REPLAY),"775_auditor":pin(AUDITOR),
            "625_preparer":pin(PREPARER),"625_finalizer":pin(FINALIZER),"v24_finalizer":pin(V24_PATH)},
        "native_quality_threshold_unchanged":True,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    outreceipt=out/"resume-receipt.json";outreceipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"finalized_state":receipt["finalized_state"],"receipt":pin(outreceipt)},sort_keys=True))
if __name__=="__main__":main()

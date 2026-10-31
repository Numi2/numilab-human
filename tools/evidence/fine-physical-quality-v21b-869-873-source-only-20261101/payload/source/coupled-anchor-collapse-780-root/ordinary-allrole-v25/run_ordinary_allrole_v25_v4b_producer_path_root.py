#!/usr/bin/env python3
"""Bounded v25 all-role LP -> 775 audit -> 625 rollback -> independent replay."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2]
BASE_BUILDER=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25/repair_coupled_quality_points_722_v25_ordinary_allrole_local.py"
BASE_BUILDER_SHA="cf17d106e83450af02dbd87b1c6c1730b1a5519e170c359f0b0525b7e8392742"
BASE_REPLAY=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25/verify_ordinary_allrole_v25_independent_replay_v2.py"
BASE_REPLAY_SHA="e38d383229087ff6b65fb8d741262f4e692f6f2216df127b0c95eff706fe037d"
BLOCK_REPLAY=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25f-blocklocal/verify_ordinary_allrole_v25f_independent_replay.py"
BLOCK_REPLAY_SHA="04bf2d6b69dd3ce715f0084c4b4168ac93662d39190986462e5bdbcce79d8a8f"
V24_PATH=ROOT/"continue_coupled_quality_points_v24c_ordinary_outer_o_local.py"
V24_SHA="f69ab97f020436c5f0afa997a770ad291bad9ee63942522d0ea0b3c49230074a"
AUDITOR=ROOT/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
AUDITOR_SHA="82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
PREPARER=ROOT/"bounded625-local-cavity-repair/prepare_bounded_rollback_625_inner805.py"
PREPARER_SHA="fe99ede9eff7449e558b7f935d791dee1c3384c5159b2e9d8ef2b10e509ab1d7"
ROLLBACK_FINALIZER=ROOT/"bounded625-local-cavity-repair/finalize_v7_rollback_805.py"
ROLLBACK_FINALIZER_SHA="562c778b73d90ddbcf7d7aa511f6f8bbc6d79cd19773bd19151685a499010813"
ZERO_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def pin(path):
    p=Path(path).resolve();return {"path":str(p),"sha256":sha(p)}
def checked(path,digest,label):
    if sha(path)!=digest:raise ValueError(label+" source SHA mismatch")
def import_v24():
    checked(V24_PATH,V24_SHA,"v24 replay/finalizer")
    spec=importlib.util.spec_from_file_location("v24_owner_for_allrole25_runner",V24_PATH)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import v24 owner")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
V=import_v24()

def run(label,cmd,logdir):
    logdir.mkdir(parents=True,exist_ok=True)
    t=time.monotonic();proc=subprocess.run(cmd,capture_output=True,text=True,check=False)
    (logdir/(label+".stdout.log")).write_text(proc.stdout)
    (logdir/(label+".stderr.log")).write_text(proc.stderr)
    (logdir/(label+".command.json")).write_text(json.dumps({"command":cmd,"returncode":proc.returncode,
        "elapsed_s":time.monotonic()-t},indent=2)+"\n")
    if proc.returncode:
        raise RuntimeError(label+" exited "+str(proc.returncode)+": "+(proc.stderr or proc.stdout)[-4000:])
def audit(config,out,logs,label):
    run(label,[sys.executable,str(AUDITOR),"--config",str(config),"--out",str(out)],logs)
    report=out/"report.json"
    if not report.is_file():raise RuntimeError("775 auditor emitted no report")
    doc=json.loads(report.read_text())
    if doc.get("config",{}).get("path")!=str(config.resolve()) or doc["config"].get("sha256")!=sha(config):
        raise ValueError("775 report/config pin mismatch")
    against=doc.get("changed_vs_complete_retained_OID_both_walls_all185",{})
    selfp=doc.get("changed_OID_joint_self_pairs",{})
    na=int(against.get("count",-1));ns=int(selfp.get("count",-1))
    if na<0 or ns<0:raise ValueError("775 report lacks full context/self counts")
    return report,{"status":doc.get("status"),"context_pairs":na,"joint_self_pairs":ns},(
      doc.get("status")==ZERO_STATUS and na==0 and ns==0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--rounds",type=int,default=1)
    ap.add_argument("--trust-cap-m",type=float,default=.0005)
    ap.add_argument("--seconds",type=float,default=30.)
    ap.add_argument("--objective-mode",choices=("legacy","role_equal","role_equal_nearpass"),default="role_equal_nearpass")
    ap.add_argument("--witness-report",type=Path,action="append",default=[])
    ap.add_argument("--block-local-quality",action="store_true")
    ap.add_argument("--continue-after-stalled",dest="early_stop_stalled",action="store_false",default=True)
    args=ap.parse_args()
    if not 1<=args.rounds<=2:raise ValueError("v25 is bounded to one or two rounds")
    if not 0<args.trust_cap_m<=.0005 or not 0<args.seconds<=30:raise ValueError("ordinary v25 cap/time limit exceeded")
    builder=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25f-blocklocal/repair_coupled_quality_points_722_v25f_ordinary_blocklocal.py" if args.block_local_quality else BASE_BUILDER
    builder_sha="e05d5662d64edd97d25c5a0e42e00a58ba06ef2850b3afc6fd9b23a3111fc376" if args.block_local_quality else BASE_BUILDER_SHA
    replay_path=BLOCK_REPLAY if args.block_local_quality else BASE_REPLAY
    replay_sha=BLOCK_REPLAY_SHA if args.block_local_quality else BASE_REPLAY_SHA
    for path,digest,label in ((builder,builder_sha,"builder"),(replay_path,replay_sha,"replay"),
      (AUDITOR,AUDITOR_SHA,"775 auditor"),(PREPARER,PREPARER_SHA,"625 rollback preparer"),
      (ROLLBACK_FINALIZER,ROLLBACK_FINALIZER_SHA,"625 finalizer")):checked(path,digest,label)
    parent=args.state.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    source_proof=V.verify_source_input(parent)
    if not str(source_proof.get("status","")).startswith("PASS_"):raise ValueError("parent source 807 proof failed")
    out.mkdir(parents=True)
    round_receipts=[]
    witness_reports=[Path(p).resolve() for p in args.witness_report]
    early_stop=None
    for n in range(1,args.rounds+1):
        used_witnesses=[pin(p) for p in witness_reports]
        rd=out/f"round-{n:02d}";rd.mkdir()
        proposal=rd/"proposal"
        cmd=[sys.executable,str(builder),"--state",str(parent),"--output",str(proposal),
             "--trust-cap-m",str(args.trust_cap_m),"--seconds",str(args.seconds),
             "--ordinary-all-role-local-repair","--objective-mode",args.objective_mode]
        if args.block_local_quality:cmd.append("--ordinary-block-local-quality")
        for wr in witness_reports:cmd.extend(["--witness-report",str(wr)])
        run("builder-v25e-block" if args.block_local_quality else "builder-v25",cmd,rd/"builder-logs")
        proposal_state=proposal/"state.json";config=proposal/"context-config.json"
        if not proposal_state.is_file():
            diagnostics=proposal/"lp-diagnostics.json"
            no_proposal={"round":n,"status":"NO_STATE_EMITTED","parent_state":pin(parent),
                "diagnostics":pin(diagnostics) if diagnostics.is_file() else None,
                "builder_logs":str((rd/"builder-logs").resolve())}
            round_receipts.append(no_proposal)
            early_stop={"status":"EARLY_STOP_NO_PROPOSAL","round":n,
                "reason":"builder completed without a candidate state; parent is preserved"}
            break
        if not config.is_file():raise RuntimeError("builder state emitted without context config")
        st=V.load_state(proposal_state);cfg=json.loads(config.read_text())
        policy=st.get("coupled_point_repair",{})
        if policy.get("ordinary_all_role_local_repair") is not True or policy.get("maximum_displacement_m")!=args.trust_cap_m:
            raise ValueError("builder state does not match ordinary all-role policy/cap")
        if args.block_local_quality and policy.get("ordinary_block_local_quality_fit") is not True:
            raise ValueError("block-local builder did not record its selected fitting mode")
        changed_roles=cfg.get("changed_roles",[])
        if not changed_roles or len(changed_roles)!=len(set(changed_roles)) or not set(changed_roles).issubset({"O","I","D"}):
            raise ValueError("builder changed_roles must be a nonempty unique O/I/D subset")
        for role in cfg["changed_roles"]:
            e=cfg["bounded_point_repair"][role]
            if float(e.get("maximum_displacement_m",-1))!=args.trust_cap_m:
                raise ValueError(role+" trust cap differs from runner")
            if any(k in e for k in ("inner_back_layer_repair","ordinary_outer_o_local_repair")):
                raise ValueError(role+" incorrectly carries specialized boundary policy")
        proposal_report,proposal_summary,proposal_clean=audit(config,rd/"proposal-exact-775",rd/"proposal-audit-logs","audit775")
        selected_state=proposal_state;selected_config=config;selected_report=proposal_report
        rollback_plan=None;proposal_for_replay=None
        rollback_receipt=None
        if not proposal_clean:
            rollback=rd/"witness-component-rollback"
            run("rollback625",[sys.executable,str(PREPARER),"--state",str(proposal_state),
                "--config",str(config),"--audit",str(proposal_report),"--output",str(rollback)],
                rd/"rollback-logs")
            rs=rollback/"state.json";rc=rollback/"context-config.json";rollback_plan=rollback/"rollback-plan.json"
            prep=rollback/"rollback-preparation-receipt.json"
            if not all(x.is_file() for x in (rs,rc,rollback_plan,prep)):
                raise RuntimeError("625 rollback preparation outputs are incomplete")
            selected_report,rollback_summary,rollback_clean=audit(rc,rd/"rollback-exact-775",
                rd/"rollback-audit-logs","audit775")
            if not rollback_clean:raise RuntimeError("775 still has witnesses after 625 whole-component rollback")
            finalized=rd/"rollback-finalized"
            run("rollback-finalize625",[sys.executable,str(ROLLBACK_FINALIZER),"--state",str(rs),
                "--baseline-state",str(parent),"--proposal-state",str(proposal_state),
                "--rollback-receipt",str(prep),"--audit-report",str(selected_report),
                "--audit-config",str(rc),"--output-dir",str(finalized)],rd/"rollback-finalizer-logs")
            selected_state=finalized/"state.json";selected_config=rc;proposal_for_replay=proposal_state
            rollback_receipt=prep
        replay_report=rd/"independent-allrole-replay.json"
        replay_cmd=[sys.executable,str(replay_path),"--candidate-state",str(selected_state),
            "--parent-state",str(parent),"--audit-report",str(selected_report),
            "--audit-config",str(selected_config),"--output-report",str(replay_report)]
        if rollback_plan is not None:
            replay_cmd.extend(["--proposal-state",str(proposal_for_replay),"--rollback-plan",str(rollback_plan)])
        run("independent-v25-replay",replay_cmd,rd/"independent-replay-logs")
        replay_doc=json.loads(replay_report.read_text())
        if replay_doc.get("status")!="PASS_775_SOURCE_CONTEXT_VOLUME_QUALITY_AND_V25_FREE_SET":
            raise RuntimeError("v25 independent replay did not pass")
        base_replay=replay_report.with_name(replay_report.stem+"-base-volume-replay.json")
        finalized_state=rd/"finalized-current-state.json"
        V.finalize_verified_state(selected_state,selected_report,base_replay,finalized_state)
        final=json.loads(finalized_state.read_text())
        final.update({"independent_allrole_local_replay":pin(replay_report),
            "physical_admission":False,"native_steps":0,
            "mass_reconciliation":{"status":"PENDING_VOLUME_CHANGE_RECONCILIATION",
                "mass_conservation_claimed":False}})
        finalized_state.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
        final_source=V.verify_source_input(finalized_state)
        fresh_failed_report=None
        if not proposal_clean and n<args.rounds:
            resolved=proposal_report.resolve()
            if resolved not in witness_reports:
                witness_reports.append(resolved)
                fresh_failed_report=pin(resolved)
        round_doc={"round":n,"parent_state":pin(parent),"proposal_state":pin(proposal_state),
            "proposal_audit":pin(proposal_report),"proposal_audit_summary":proposal_summary,
            "rollback_applied":rollback_plan is not None,
            "rollback_preparation":pin(rollback_receipt) if rollback_receipt else None,
            "final_audit":pin(selected_report),"final_audit_summary":rollback_summary if rollback_plan else proposal_summary,
            "independent_replay":pin(replay_report),"final_state":pin(finalized_state),
            "cumulative_witness_reports_used":used_witnesses,
            "fresh_failed_report_carried_forward":fresh_failed_report,
            "block_local_quality":bool(args.block_local_quality),
            "final_807_source_proof_status":final_source.get("status"),
            "native_quality_failures":{"Skin":final["Skin_current"].get("native_quality_failures"),
                "Fat":final["Fat_current"].get("native_quality_failures")},
            "surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
        (rd/"round-receipt.json").write_text(json.dumps(round_doc,indent=2,sort_keys=True)+"\n")
        round_receipts.append(round_doc)
        before_state=V.load_state(parent);after_state=V.load_state(finalized_state)
        before_bad=sum(int(before_state[r+"_current"].get("native_quality_failures",0)) for r in ("Skin","Fat"))
        after_bad=sum(int(after_state[r+"_current"].get("native_quality_failures",0)) for r in ("Skin","Fat"))
        diag_path=proposal/"lp-diagnostics.json"
        diag=json.loads(diag_path.read_text()) if diag_path.is_file() else {}
        strict_score=bool(diag.get("strict_bad_score_progress",False))
        parent=finalized_state
        if args.early_stop_stalled and n<args.rounds and fresh_failed_report is None and after_bad>=before_bad and not strict_score:
            early_stop={"status":"EARLY_STOP_STALLED","round":n,"before_native_bad":before_bad,
                "after_native_bad":after_bad,"strict_bad_score_progress":strict_score,
                "reason":"round produced no count decrease or recorded strict score progress"}
            break
    manifest={"schema":"h706.ordinary-allrole-v4-cumulative-witness-run.v1",
       "status":"PASS_SOURCE_ONLY_WITH_775_AND_INDEPENDENT_REPLAY" if any(x.get("final_state") for x in round_receipts) else (early_stop or {}).get("status","NO_CANDIDATE"),
       "early_stop":early_stop,"cumulative_witness_reports":[pin(p) for p in witness_reports],
       "implementation_pins":{"builder":pin(builder),"independent_replay":pin(replay_path),"775_auditor":pin(AUDITOR),
          "rollback625_preparer":pin(PREPARER),"rollback625_finalizer":pin(ROLLBACK_FINALIZER),"v24_finalizer":pin(V24_PATH)},
       "rounds":round_receipts,"final_state":pin(parent),"physical_admission":False,"native_steps":0,
       "mass_reconciliation":"PENDING","native_quality_thresholds_unchanged":True}
    path=out/"run-receipt.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":manifest["status"],"final_state":manifest["final_state"],
        "receipt":pin(path),"rounds":len(round_receipts)},sort_keys=True))

if __name__=="__main__":main()

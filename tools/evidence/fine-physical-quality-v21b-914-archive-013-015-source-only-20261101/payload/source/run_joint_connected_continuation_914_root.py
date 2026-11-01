from pathlib import Path
import argparse,subprocess,sys,json,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
def counts(s):return {r:int(s[r+"_current"]["native_quality_failures"]) for r in ("Skin","Fat")}
a=argparse.ArgumentParser();a.add_argument("--state",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--rounds",type=int,default=3);args=a.parse_args()
root=args.output.resolve();root.mkdir(exist_ok=False);parent=args.state.resolve()
builder=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25m-back-rowfix/repair_coupled_quality_points_722_v25m_back_rowfix.py"
ver=builder.with_name("verify_ordinary_allrole_v25n_back_lineage_alias.py");audit=H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py"
for p,sha in ((builder,"81b05b9d941d73771d81e31f5597877342cf3a80bff7e1a61e9f2b81cec2bf42"),(ver,"ed25a5ecb0ceadee0595e917d75b2373d9aff19ae8e5c33c74068442cd4ed2bf"),(audit,"0f41e4114eeb56032d4bb6a6c39484d2e8dd0de52ff977c7211a800273101234")):assert pin(p)["sha256"]==sha
witnesses=[H/p for p in ["all-quality-847-root/round-01/proposal-exact-805/report.json","all-quality-850-root/round-01/proposal-exact-805/report.json","all-quality-858-audit775-root/proposal-exact-775/report.json","all-quality-866-audit805b-root/proposal-exact-805/report.json","all-quality-867-audit805b-root/proposal-exact-805/report.json","all-quality-871-root/round-01/proposal-exact-775/report.json","all-quality-873-root/round-01/proposal-exact-775/report.json","all-quality-875-root/round-01/proposal-exact-805/report.json","all-quality-883-audit805-root/proposal-exact-805/report.json","all-quality-886-audit805-root/proposal-exact-805/report.json","all-quality-894-audit805-root/proposal-exact-805/report.json","all-quality-895-connected-root/round-02-attempt-01/audit/proposal-exact-805/report.json","all-quality-913-qualified-root/proposal-exact-805/report.json"]]
def run(label,cmd,d):
 d.mkdir(exist_ok=True,parents=True);t=time.monotonic()
 with (d/(label+".stdout.log")).open("w") as f,(d/(label+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
 write(d/(label+".command.json"),{"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t})
 if r.returncode:raise RuntimeError(label+" failed: "+(d/(label+".stderr.log")).read_text()[-2500:])
accepted=[]
for i in range(1,args.rounds+1):
 if (root/"STOP_AFTER_CURRENT_ROUND").exists():break
 base=json.loads(parent.read_text());bc=counts(base)
 if sum(bc.values())==0:break
 passing=False
 for attempt in range(1,4):
  rd=root/("round-%02d-attempt-%02d"%(i,attempt));rd.mkdir();proposal=rd/"proposal"
  cmd=[sys.executable,str(builder),"--state",str(parent),"--output",str(proposal),"--trust-cap-m",".00005","--seconds","30","--patch-seconds","1.5","--node-fallbacks-per-patch","8","--objective-mode","role_equal_nearpass","--unified-inner-back-ordinary-local-repair","--ordinary-block-local-quality"]
  for w in witnesses:cmd+=["--witness-report",str(w)]
  run("builder",cmd,rd/"builder-logs");sp=proposal/"state.json";s=json.loads(sp.read_text());sc=counts(s)
  diag=json.loads((proposal/"lp-diagnostics.json").read_text());q0=float(diag["bad_score_before"]);q1=float(diag["bad_score_after"])
  assert sc["Skin"]<=bc["Skin"] and sc["Fat"]<=bc["Fat"]
  if not (sum(sc.values())<sum(bc.values()) or (sum(sc.values())==sum(bc.values()) and q1<q0-max(1e-8,1e-6*q0))):
   print("NO_JOINT_COUNT_OR_SCORE_GAIN",i,attempt,bc,sc,q0,q1,flush=True);break
  print("JOINT_PROPOSAL",i,attempt,bc,sc,q0,q1,flush=True)
  ar=rd/"audit/proposal-exact-805/report.json";cfg=proposal/"context-config.json"
  run("audit805",[sys.executable,str(audit),"--config",str(cfg),"--out",str(ar.parent)],rd/"audit/proposal-audit-logs")
  ad=json.loads(ar.read_text());print("PROPOSAL",i,attempt,sc,ad["status"],flush=True)
  if ad["status"]!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":witnesses.append(ar);continue
  out=rd/"qualified";out.mkdir();dest=out/"verified-state.json"
  s.update(exact_context=pin(ar),surface_crossings=0,source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
  write(dest,s);proof=out/"independent-unified-n-replay.json"
  run("verifier",[sys.executable,str(ver),"--candidate-state",str(dest),"--parent-state",str(parent),"--audit-report",str(ar),"--audit-config",str(cfg),"--output-report",str(proof)],out/"verification-logs")
  replay=json.loads(proof.read_text());assert replay["status"]=="PASS_INDEPENDENT_SOURCE_CONTEXT_VOLUME_QUALITY_AND_UNIFIED_MASK"
  receipt={"status":"PASS_INDEPENDENT_UNIFIED_MASK_NATIVE_CELL_VOLUME_AND_805_807_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"raw_proposal":pin(sp),"independent_replay":pin(proof),"exact_audit":pin(ar),"counts":sc,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
  write(out/"receipt.json",receipt);accepted.append(receipt);write(root/"accepted.json",receipt)
  run("checkpoint",[sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],out/"checkpoint-logs")
  parent=dest;passing=True;print("QUALIFIED",i,sc,str(dest),flush=True);break
 if not passing:break
write(root/"run-receipt.json",{"status":"BOUNDED_CONNECTED_QUALITY_CONTINUATION_FINISHED","accepted":accepted,"current_state":pin(parent),"counts":counts(json.loads(parent.read_text())),"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"})


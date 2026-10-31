from pathlib import Path
import json,sys,subprocess,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-868-root";out.mkdir(exist_ok=False)
parent=H/"all-quality-859-qualified-root/quality-working-state.json"
builder=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25d/repair_coupled_quality_points_722_v25d_inner_back_remainder_blocklocal.py"
ver=builder.with_name("verify_targeted_inner_back_remainder_v25d.py")
audit=H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py"
for p,h in [(builder,"299653fb0979d60a9e3d506cfa31750c2f97e786b2e9e8010d9f1aa399e9bd72"),(ver,"0668e57e914a9a17e2d7f7fcbcd091af9e89d44208aba2c54572bd8dabcc6651")]:
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
witnesses=[H/p for p in ["all-quality-847-root/round-01/proposal-exact-805/report.json","all-quality-850-root/round-01/proposal-exact-805/report.json","all-quality-858-audit775-root/proposal-exact-775/report.json","all-quality-866-audit805b-root/proposal-exact-805/report.json","all-quality-867-audit805b-root/proposal-exact-805/report.json"]]
def run(label,cmd,d):
 d.mkdir(exist_ok=True,parents=True);t=time.monotonic()
 with (d/(label+".stdout.log")).open("w") as f,(d/(label+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
 (d/(label+".command.json")).write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n")
 if r.returncode:raise RuntimeError(label+" failed "+str(r.returncode))
for i in range(1,4):
 rd=out/("round-%02d"%i);rd.mkdir();prop=rd/"proposal";cfg=prop/"context-config.json"
 cmd=[sys.executable,str(builder),"--state",str(parent),"--output",str(prop),"--inner-back-remainder-repair","--block-local-target-repair","--trust-cap-m","0.00005","--seconds","30","--patch-seconds","1.5","--node-fallbacks-per-patch","8","--objective-mode","role_equal_nearpass"]
 for p in witnesses:cmd.extend(["--witness-report",str(p)])
 run("builder",cmd,rd/"builder-logs")
 ps=prop/"state.json";doc=json.loads(ps.read_text());counts={r:doc[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")}
 ar=rd/"proposal-exact-805/report.json"
 run("audit805",[sys.executable,str(audit),"--config",str(cfg),"--out",str(ar.parent)],rd/"proposal-audit-logs")
 ad=json.loads(ar.read_text());n=ad["changed_vs_complete_retained_OID_both_walls_all185"]["count"];s=ad["changed_OID_joint_self_pairs"]["count"]
 print("ROUND",i,"NATIVE",counts,"CONTEXT",n,"SELF",s,flush=True)
 if ad["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":
  run("independent-v25d",[sys.executable,str(ver),"--parent-state",str(parent),"--proposal-state",str(ps),"--proposal-config",str(cfg),"--proposal-audit",str(ar),"--output",str(rd/"qualified")],rd/"replay-logs")
  final=rd/"qualified/verified-state.json"
  (out/"accepted.json").write_text(json.dumps({"state":str(final),"sha256":hashlib.sha256(final.read_bytes()).hexdigest(),"counts":counts,"physical_admission":False,"native_steps":0},indent=2)+"\n")
  print("QUALIFIED",str(final),counts,flush=True);break
 witnesses.append(ar)
else:print("BOUNDED_SOURCE_WITNESS_ROUNDS_EXHAUSTED",flush=True)


from pathlib import Path
import subprocess,json,sys,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
o=H/"all-quality-888-audit805-root";o.mkdir(exist_ok=False)
logs=o/"proposal-audit-logs";logs.mkdir()
cmd=[sys.executable,str(H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py"),"--config",str(H/"all-quality-888-unified-root/context-config.json"),"--out",str(o/"proposal-exact-805")]
t=time.monotonic()
with (logs/"audit805.stdout.log").open("w") as f,(logs/"audit805.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(logs/"audit805.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n")
print("RETURN",r.returncode,flush=True)
if r.returncode==0:
 a=json.loads((o/"proposal-exact-805/report.json").read_text());print(a["status"],a["changed_vs_complete_retained_OID_both_walls_all185"]["count"],a["changed_OID_joint_self_pairs"]["count"],flush=True)
raise SystemExit(r.returncode)

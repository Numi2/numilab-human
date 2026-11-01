from pathlib import Path
import sys,json,hashlib,subprocess,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");root=H/"all-quality-928-b5-backtracking-root"
root.mkdir()
owner=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v26b5-cowdiff-f32backtrack-250um/repair_coupled_quality_points_722_v26b5_cowdiff_f32backtrack_250um.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="53071ffde5f51eb65a2090d1e60931f552f0647fa8d5c4940af941d525fab946"
cmd=json.loads((H/"all-quality-922-joint-250um-root/builder.command.json").read_text())["command"]
cmd[1]=str(owner)
cmd[cmd.index("--state")+1]=str(H/"all-quality-927-fixed-cavity-batch-root/qualified/verified-state.json")
cmd[cmd.index("--output")+1]=str(root/"proposal")
cmd[cmd.index("--objective-mode")+1]="role_equal"
cmd+=["--witness-report",str(H/"all-quality-922-joint-250um-root/qualified/proposal-exact-806/report.json")]
t=time.monotonic()
with (root/"builder.stdout.log").open("w") as f,(root/"builder.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(root/"builder.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2))
if r.returncode:print((root/"builder.stderr.log").read_text()[-6000:]);raise SystemExit(r.returncode)
s=json.loads((root/"proposal/state.json").read_text());print("PROPOSAL",{r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")},flush=True)


from pathlib import Path
import sys,json,hashlib,subprocess,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");root=H/"all-quality-943-expanded-joint-root"
root.mkdir()
owner=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v26c-expanded-joint-stars/repair_coupled_quality_points_722_v26c_expanded_joint_stars.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="07522e0c7337d894222e89c65458ce1e356fa335c4a0d81e0c3cbd6067377ddc"
cmd=json.loads((H/"all-quality-922-joint-250um-root/builder.command.json").read_text())["command"]
cmd[1]=str(owner)
cmd[cmd.index("--state")+1]=str(H/"all-quality-941-b5-backtracking-root/qualified/verified-state.json")
cmd[cmd.index("--output")+1]=str(root/"proposal")
cmd[cmd.index("--objective-mode")+1]="role_equal"
cmd+=["--witness-report",str(H/"all-quality-922-joint-250um-root/qualified/proposal-exact-806/report.json")]
t=time.monotonic()
with (root/"builder.stdout.log").open("w") as f,(root/"builder.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(root/"builder.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2))
if r.returncode:print((root/"builder.stderr.log").read_text()[-6000:]);raise SystemExit(r.returncode)
s=json.loads((root/"proposal/state.json").read_text());print("PROPOSAL",{r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")},flush=True)

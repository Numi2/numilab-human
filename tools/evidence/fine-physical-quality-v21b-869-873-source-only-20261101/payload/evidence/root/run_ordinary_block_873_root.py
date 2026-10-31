from pathlib import Path
import subprocess,sys,hashlib,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");runner=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25g-blocklocal/run_ordinary_allrole_v25_v4h_rollback_namespace_root.py"
assert hashlib.sha256(runner.read_bytes()).hexdigest()=="696190bddec26e42391ad0b7041ca169e3de4e97a17b6fef1f0aae3f07d6fb53"
cmd=[sys.executable,str(runner),"--state",str(H/"all-quality-871-confirmation-root/verified-state.json"),"--output",str(H/"all-quality-873-root"),"--rounds","2","--trust-cap-m","0.0005","--seconds","30","--objective-mode","role_equal_nearpass","--block-local-quality"]
for p in ["all-quality-847-root/round-01/proposal-exact-805/report.json","all-quality-850-root/round-01/proposal-exact-805/report.json","all-quality-858-audit775-root/proposal-exact-775/report.json","all-quality-866-audit805b-root/proposal-exact-805/report.json","all-quality-867-audit805b-root/proposal-exact-805/report.json","all-quality-871-root/round-01/proposal-exact-775/report.json"]:
 cmd.extend(["--witness-report",str(H/p)])
print(json.dumps({"command":cmd}),flush=True);raise SystemExit(subprocess.run(cmd).returncode)


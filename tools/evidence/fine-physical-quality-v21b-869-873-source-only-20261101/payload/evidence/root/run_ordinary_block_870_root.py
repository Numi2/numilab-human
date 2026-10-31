from pathlib import Path
import subprocess,sys,hashlib,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");runner=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25/run_ordinary_allrole_v25_v4_cumulative.py"
assert hashlib.sha256(runner.read_bytes()).hexdigest()=="4a15f7c1df26b3a31b48354b5a960c30fb9ecbc3bb287ce4ea257524fc16b637"
cmd=[sys.executable,str(runner),"--state",str(H/"all-quality-868-root/round-01/qualified-witness-aware-868/verified-state.json"),"--output",str(H/"all-quality-870-root"),"--rounds","2","--trust-cap-m","0.0005","--seconds","30","--objective-mode","role_equal_nearpass","--block-local-quality"]
for p in ["all-quality-847-root/round-01/proposal-exact-805/report.json","all-quality-850-root/round-01/proposal-exact-805/report.json","all-quality-858-audit775-root/proposal-exact-775/report.json","all-quality-866-audit805b-root/proposal-exact-805/report.json","all-quality-867-audit805b-root/proposal-exact-805/report.json"]:
 cmd.extend(["--witness-report",str(H/p)])
print(json.dumps({"command":cmd}),flush=True);raise SystemExit(subprocess.run(cmd).returncode)


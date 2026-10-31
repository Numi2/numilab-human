from pathlib import Path
import sys,subprocess,hashlib,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
owner=H/"bad-cluster-fixed-flip-batch-865-nativepreflight/run_fixed_flip_batch_state_865.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="2a890f52d2762a109f0a5e2de91f5cd022006e9b8f4c6aafd75c2e8e5bc189dd"
parent=H/"all-quality-868-root/round-01/qualified-witness-aware-868/verified-state.json"
assert hashlib.sha256(parent.read_bytes()).hexdigest()=="ef4735001c7e19656cf62aacab881359fe830f2169571475a683b3b68fcb1943"
cmd=[sys.executable,str(owner),"--state",str(parent),"--output",str(H/"all-quality-869-root"),"--max-candidate-evaluations","20000","--max-operations","256"]
print(json.dumps({"command":cmd}),flush=True);raise SystemExit(subprocess.run(cmd).returncode)


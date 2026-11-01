from pathlib import Path
import sys,subprocess,hashlib,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
owner=H/"quality-partial-cavity-684-root/paired_connected_boundary_patch_batch_871_replay_localids_staged_root.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="8faaf716c17bac710a0ceed0ea836bd10c63b11148e21a953eb3b2a0758be21e"
parent=H/"all-quality-877-qualified-root/verified-state.json"
assert hashlib.sha256(parent.read_bytes()).hexdigest()=="9f0a23efdf00bc802daf26d796c28d35cbf26409094eedb055a4b221a6466535"
cmd=[sys.executable,str(owner),"--state",str(parent),"--role","I","--output",str(H/"all-quality-878-root"),"--maximum-edge-m",".1","--batch-seconds","120","--solve-seconds","3","--max-patches","32","--maximum-old-cells","64","--maximum-points","24","--inner-back-topology-only","--fixed-coordinate-topology-only"]
print(json.dumps({"command":cmd}),flush=True);raise SystemExit(subprocess.run(cmd).returncode)


from pathlib import Path
import sys,subprocess,hashlib,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
owner=H/"quality-partial-cavity-684-root/paired_connected_boundary_patch_batch_865_adaptive_edge_hausdorff_root.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="fed52fae890a7b2bd749419204140cf10ade3895fe8d4de3139b3d0339bec657"
parent=H/"all-quality-871-confirmation-root/verified-state.json"
assert hashlib.sha256(parent.read_bytes()).hexdigest()=="cd242f3d140baac57aae676095a33ff4dba1b05ae1a448ded314d8e46fb4578b"
cmd=[sys.executable,str(owner),"--state",str(parent),"--role","D","--output",str(H/"all-quality-872-root"),"--maximum-edge-m",".1","--batch-seconds","120","--solve-seconds","3","--max-patches","32","--maximum-old-cells","64","--maximum-points","24","--inner-back-topology-only"]
print(json.dumps({"command":cmd}),flush=True);raise SystemExit(subprocess.run(cmd).returncode)


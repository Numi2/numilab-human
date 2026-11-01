import runpy,json,hashlib
from pathlib import Path
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
r=json.loads((H/"all-quality-939-b5-backtracking-root/qualified/receipt.json").read_text());assert r["status"].startswith("PASS_INDEPENDENT_");assert hashlib.sha256(Path(r["state"]["path"]).read_bytes()).hexdigest()==r["state"]["sha256"]
runpy.run_path(str(H/"run_fixed_boundary_940_root.py"),run_name="__main__")

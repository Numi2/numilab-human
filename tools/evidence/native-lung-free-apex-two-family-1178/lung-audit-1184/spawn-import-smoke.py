#!/usr/bin/env python3
"""Spawn-import regression for the standalone wrapper's canonical 1171 core import."""
import hashlib, json, sys, concurrent.futures
from pathlib import Path
ev=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-standalone-short-preflight-1184")
core_dir=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-late-pose-audit-runner-1171")
expected="37bd87c0ac870e59141422db5f6cd8be9b16452155472e61bd84840bc8f8583a"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(core_dir/"audit_lung_cycle_1159.py")==expected
sys.path.insert(0,str(core_dir))
import audit_lung_cycle_1159 as core
with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
 result=pool.submit(core.worker_pose,{"base":str(ev/"definitely-missing-base.py")},0).result()
assert result["status"]=="failed" and result["error"].startswith("ValueError: pinned input mismatch:"), result
out={"status":"PASS_spawn_imported_worker_then_reached_expected_missing-base","worker_result":result,"core_path":str(Path(core.__file__).resolve()),"core_sha256":sha(core.__file__),"worker_module":core.worker_pose.__module__}
(ev/"spawn-import-smoke.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,sort_keys=True))

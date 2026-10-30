#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=[sys.executable,"-m","unittest","discover","-s",str(root),"-p","test_capacity_gate.py"]
r=subprocess.run(cmd,capture_output=True,text=True)
receipt={"argv":cmd,"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr,
 "test_file_sha256":sha(root/"test_capacity_gate.py"),"capacity_gate_sha256":sha(root/"capacity_gate.py"),
 "physical_admission":False,"runtime":0,"accepted_steps":0}
p=root/"capacity-tests.json"
p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+chr(10))
print(json.dumps(receipt,sort_keys=True))
print("receipt_sha256",sha(p))
raise SystemExit(r.returncode)

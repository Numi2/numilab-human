#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=[sys.executable,"-m","unittest","discover","-s",str(root),"-p","test_adapter.py"]
r=subprocess.run(cmd,capture_output=True,text=True)
receipt={"argv":cmd,"exit_code":r.returncode,"stdout":r.stdout,"stderr":r.stderr,
 "adapter_source_sha256":sha(root/"tetgen-217-role-adapter.py"),
 "adapter_config_sha256":sha(root/"adapter-config.json"),
 "test_source_sha256":sha(root/"test_adapter.py"),
 "native217_source004_sha256":"46799a2d63fd20bc8a191f319f1f56244610a5e719983a7ce44cc5222659b6bd",
 "tetgen054_binary_sha256":"597b6025f1622bcaeb2a9bc9ac3b2813f520d6fb0a09a64acc6485a3474cffb1",
 "region_seed_helper_056_sha256":"0aea6dbf689705cfaae9b0ee72cf76f7459b7ebce665af97c84726509cab2d14",
 "physical_admission":False,"runtime":0,"accepted_steps":0}
out=root/"adapter-tests.json"
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+chr(10))
print(json.dumps(receipt,sort_keys=True))
print("receipt_sha256",sha(out))
raise SystemExit(r.returncode)

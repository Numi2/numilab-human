from pathlib import Path
import json,subprocess,hashlib,time
O=Path(__file__).resolve().parent;p=json.loads((O/'coupon-plan.json').read_text());start=time.time()
for path,digest in p['input_pins'].items():
 with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,path
try:
 with (O/'coupon.stdout.log').open('w') as a,(O/'coupon.stderr.log').open('w') as b:r=subprocess.run(p['argv'],stdout=a,stderr=b,timeout=150)
 code=r.returncode;guard=None
except subprocess.TimeoutExpired:code=124;guard='150-second test bound'
(O/'coupon-execution.json').write_text(json.dumps({'exit_code':code,'wall_s':time.time()-start,'guard_stop':guard,'scope':'GPU canonical material regression only, no Human or speed qualification'},indent=2)+'\n')
print('COUPON_EXIT',code,flush=True)
raise SystemExit(code)

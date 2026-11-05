"""Resume only pending receipts after the current compiler stages pass."""
from pathlib import Path
import hashlib,json,subprocess,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
O=W/'compiler-flat-cap-batch-364'
C=W/'flat-cap-batch-rebuild-364/proposal/compiler'
oldC=W/'witness-cluster-rebuild-356/proposal/compiler'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
middle=O/'compiler-execution-middle.json'
resume=O/'compiler-execution-resume-367.json'
m=json.loads(middle.read_text());r=json.loads(resume.read_text())
assert len(m['stages'])==9 and m['stages'][-1]['guard_stop']=='disk'
assert len(r['stages'])==3
assert [s['name'] for s in r['stages']]==['mass-allocation','source-qualification','mass-admission-prepare']
complete=m['stages'][:8]+r['stages']
assert all(s['exit_code']==0 and s['guard_stop'] is None for s in complete)
execution=O/'compiler-execution-resumed-middle-367.json'
put(execution,dict(stages=complete,retained_initial_execution=dict(path=str(middle),sha256=sha(middle)),retry_execution=dict(path=str(resume),sha256=sha(resume)),failed_attempt_retained=True,physical_admission=False,native_steps=0))
old=oldC/'mass-admission/build_current_fractional_mass_receipt_1061.py'
new=C/'mass-admission/build_current_fractional_mass_receipt_1061.py'
assert sha(old)=='bfa80ce6ae1c412da3629a432de8af880ea775e2bb87d2301eb0829968519114'
oldhash='ded74fd22cab271775ad409ef1e9f90485bb39b3fd82126625a4c5f47319040e'
newhash=sha(C/'mass-support/receiver-rebind.npz')
assert old.read_text().count(oldhash)==1
expected=old.read_text().replace(str(oldC),str(C)).replace(oldhash,newhash)
assert new.read_text()==expected,'Generated receipt algorithm changed beyond authenticated receiver-map binding'
p=json.loads((O/'compiler-plan.json').read_text())
begin=next(i for i,s in enumerate(p['stages']) if s['name'].startswith('mass-receipt'))
p['stages']=p['stages'][begin:]
assert len(p['stages'])==4
for s in p['stages']:s['source_sha256']=sha(s['argv'][1])
p['resume_from']={'path':str(execution),'sha256':sha(execution)}
p['generated_script_identity_rebind']={'old_sha256':sha(old),'new_sha256':sha(new),'only_delta':'Current receiver-map path and SHA; exact normalized code parity checked','receiver_map_sha256':newhash}
put(O/'compiler-plan-final.json',p)
template=(W/'compiler-cluster-rebuild-356/run_guarded-003.py').read_text()
runner=template.replace("O=W/'compiler-cluster-rebuild-356'","O=W/'compiler-flat-cap-batch-364'").replace('compiler-plan-003.json','compiler-plan-final.json').replace('compiler-execution-003.json','compiler-execution-final.json')
(O/'run_final.py').write_text(runner)
r=subprocess.run([sys.executable,str(O/'run_final.py')])
sys.exit(r.returncode)

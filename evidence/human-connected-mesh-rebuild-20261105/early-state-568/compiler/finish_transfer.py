from pathlib import Path
import hashlib,json,subprocess,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=W/'early-state-compiler-570'
oldC=W/'flat-cap-batch-rebuild-364/proposal/compiler';C=W/'early-state-screen-568/proposal/compiler'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for file,count in [('compiler-execution.json',3),('compiler-execution-middle.json',10),('compiler-execution-mass.json',3)]:
 r=json.loads((O/file).read_text());assert len(r['stages'])==count and all(s['exit_code']==0 and s['guard_stop'] is None for s in r['stages']),file
old=oldC/'mass-admission/build_current_fractional_mass_receipt_1061.py';new=C/'mass-admission/build_current_fractional_mass_receipt_1061.py'
expected=old.read_text().replace(str(oldC),str(C)).replace(sha(oldC/'mass-support/receiver-rebind.npz'),sha(C/'mass-support/receiver-rebind.npz'))
assert new.read_text()==expected,'generated mass receipt changed beyond actual bindings'
p=json.loads((W/'compiler-flat-cap-batch-364/compiler-plan-final.json').read_text())
for s in p['stages']:
 s['argv']=[x.replace(str(oldC.parent),str(C.parent)).replace(str(W/'flat-cap-native-365'),str(W/'early-state-plan-571')) for x in s['argv']]
 s['source_sha256']=sha(s['argv'][1])
p['source_root']=str(C.parent);p['native_plan_output']=str(W/'early-state-plan-571');p['generated_script_identity_rebind']=dict(old_sha256=sha(old),new_sha256=sha(new),only_delta='Current compiler path and authenticated receiver-map SHA; exact text parity checked')
p.pop('resume_from',None)
(O/'plan-final.json').write_text(json.dumps(p,indent=2)+'\n')
t=(O/'run_guarded.py').read_text().replace("O/'plan.json'","O/'plan-final.json'").replace('compiler-execution.json','compiler-execution-final.json').replace('free>=14*2**30','free>=13*2**30').replace('need14GiB before bounded local proposal','need13GiB for remaining bounded export; retain12GiB live guard')
(O/'run_final.py').write_text(t)
sys.exit(subprocess.run([sys.executable,str(O/'run_final.py')]).returncode)

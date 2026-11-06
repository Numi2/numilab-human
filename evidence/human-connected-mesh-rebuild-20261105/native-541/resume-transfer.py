from pathlib import Path
import json,hashlib,subprocess,sys
O=Path(__file__).resolve().parent
failure=O/'compiler-execution-middle-disk-failure.json'
prior=json.loads(failure.read_text())
assert len(prior['stages'])==7 and all(s['exit_code']==0 and s['guard_stop'] is None for s in prior['stages'][:6])
code=subprocess.run([sys.executable,str(O/'run_middle_resume_guarded.py')]).returncode
if code:raise SystemExit(code)
newpath=O/'compiler-execution-middle-resume.json';new=json.loads(newpath.read_text());assert len(new['stages'])==2 and all(s['exit_code']==0 and s['guard_stop'] is None for s in new['stages'])
merged=dict(stages=prior['stages'][:6]+new['stages'],physical_admission=False,native_steps=0,resumed_execution=dict(interruption_path=str(failure),interruption_sha256=hashlib.sha256(failure.read_bytes()).hexdigest(),resume_path=str(newpath),resume_sha256=hashlib.sha256(newpath.read_bytes()).hexdigest()))
(O/'compiler-execution-middle.json').write_text(json.dumps(merged,indent=2,sort_keys=True)+'\n')
print('PREPARATION_RESUMED_ALL_EIGHT_MIDDLE_STAGES_PASS',flush=True)

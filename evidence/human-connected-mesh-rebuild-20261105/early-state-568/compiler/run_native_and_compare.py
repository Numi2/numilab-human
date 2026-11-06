from pathlib import Path
import json,sys,subprocess,time,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=Path(__file__).resolve().parent
start=time.time()
while True:
 p=O/'physical-transfer-pipeline.json';r=json.loads(p.read_text()) if p.exists() else []
 assert not any(x['exit_code'] for x in r),r
 a=W/'rejected541-compiler-recovery-574/recovery-index.json';ar=json.loads(a.read_text()) if a.exists() else []
 if len(r)==4 and len(ar)==3:break
 assert time.time()-start<3600,'Preparation/recovery did not finish within bounded session'
 time.sleep(5)
assert shutil.disk_usage(W).free>=15*2**30,'Existing native preflight15GiB remains mandatory'
results=[]
for name,path in [('bind-native',O/'prepare_native572.py'),('native4',W/'early-state-native-572/run_native_plan.py'),('compare',W/'early-state-comparison-575/run.py')]:
 print('START',name,flush=True);t=time.time()
 with (O/('pipeline-'+name+'.log')).open('w') as f:r=subprocess.run([sys.executable,str(path)],stdout=f,stderr=subprocess.STDOUT,cwd=path.parent)
 results.append(dict(stage=name,exit_code=r.returncode,wall_s=time.time()-t));(O/'native-pipeline.json').write_text(json.dumps(results,indent=2)+'\n');print('END',results[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
s=json.loads((W/'early-state-comparison-575/gate-summary.json').read_text())
print(json.dumps({'force_failures':len(s['force_failures']),'strain_failures':len(s['strain_failures']),'qualification':'NOT_PROMOTED' if s['force_failures'] or s['strain_failures'] else 'SHORT_INITIALIZATION_ONLY_NOT_MOTION_QUALIFICATION'}),flush=True)

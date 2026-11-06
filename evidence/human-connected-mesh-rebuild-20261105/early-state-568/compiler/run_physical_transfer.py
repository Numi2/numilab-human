from pathlib import Path
import json,subprocess,sys,time,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=Path(__file__).resolve().parent
archive=W/'closed-mass-witness-recovery-573/human-closed495-541-mass-witnesses-573/human-closed495-541-mass-witnesses-573-verification.json'
assert archive.exists(),'Verified recovery must finish before physical transfer'
a=json.loads(archive.read_text());assert a['status']=='PASS_INDEPENDENT_DOWNLOAD_ALL_MEMBERS_AND_RESTORE'
assert shutil.disk_usage(W).free>=14*2**30,'Existing prefix preparation preflight14GiB; explicit mass-stage headroom gate and12GiB live guard remain'
for name in ['compiler-execution.json']:
 r=json.loads((O/name).read_text());assert len(r['stages'])==3 and all(x['exit_code']==0 and x['guard_stop'] is None for x in r['stages'])
results=[]
for name,script in [('physical-transfer','run_middle.py'),('exact-byte-compression','compress_closed.py'),('conservative-mass','run_mass.py'),('finalize','finish_transfer.py')]:
 print('START',name,flush=True);start=time.time()
 with (O/('pipeline-'+name+'.log')).open('w') as f:r=subprocess.run([sys.executable,str(O/script)],stdout=f,stderr=subprocess.STDOUT,cwd=O)
 results.append(dict(stage=name,exit_code=r.returncode,wall_s=time.time()-start));(O/'physical-transfer-pipeline.json').write_text(json.dumps(results,indent=2)+'\n')
 print('END',results[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
print('PHYSICAL_TRANSFER_COMPLETED_NATIVE_PENDING',flush=True)

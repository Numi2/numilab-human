from pathlib import Path
import subprocess,json,hashlib,time,py_compile
P=Path(__file__).resolve().parent
ROOT=P.parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
checks=[]
for n,args in [('test_d_boundary_retirement_501.py',['--evidence-root',str(ROOT)]),('test_o_boundary_retirement_433.py',[]),('test_operational_reference_map_433.py',[])]:
 cmd=[PY,str(P/'source'/n),*args]
 start=time.time();r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (P/'evidence'/('root-'+n+'.stdout')).write_text(r.stdout)
 checks.append({'argv':cmd,'exit_code':r.returncode,'seconds':time.time()-start,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest()})
 if r.returncode:raise SystemExit(r.returncode)
for n in ['export_refined_coarse_registered_volume_433.py','test_d_boundary_retirement_501.py']:
 py_compile.compile(str(P/'source'/n),doraise=True)
(P/'evidence/root-checks.json').write_text(json.dumps({'checks':checks,'assertions_enabled':__debug__,'physical_admission':False,'runtime_created':0,'accepted_steps':0},indent=2)+'\n')
print(json.dumps({'root_checks':'PASS','physical_admission':False,'runtime_created':0,'accepted_steps':0}))

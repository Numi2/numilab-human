import json,hashlib,subprocess,time,sys
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent.parent
H=P/'fine-physical-volume-706'
S=P/'native-physical-parent-mass-330/native-source-identity-control-003'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
manifest=json.loads((D/'preparation-manifest.json').read_text())
compile=json.loads((D/'compile-receipt.json').read_text())
assert manifest['status']=='PASS_SOURCE_ONLY_FAT_CONNECTED_REGION_SUPPORT_PREPARATION'
assert compile['status']=='PASS_SOURCE_ONLY_DRIVER_COMPILE'
assert sha(Path(compile['binary_path']))==compile['binary_sha256']
assert all(sha(Path(p))==v['sha256'] for p,v in manifest['inputs'].items())
assert sha(D/'semantic-support.bin')==manifest['binary_sha256']
assert all(sha(Path(d['path']))==d['sha256'] for d in compile['dependency_trace'])
out=D/'fat-source-only-transfer-003.json'
assert not out.exists()
frame='0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a'
argv=[compile['binary_path'],str(S/'compiled-parent-cell-mass.csv'),str(S/'compiled-parent-cell-mass.csv.labels.csv'),str(H/'combined-child-handoff-142/fat.node'),str(H/'combined-child-handoff-142/fat.ele'),str(D/'semantic-support.bin'),frame,str(out)]
pre={'status':'PASS_SOURCE_ONLY_FAT_PRELAUNCH','physical_admission':False,'native_steps':0,'argv':argv,'compile_receipt_sha256':sha(D/'compile-receipt.json'),'support_manifest_sha256':sha(D/'preparation-manifest.json'),'allocation_solver':'deterministic_dinic_feasibility','cost_optimality_certified':False}
(D/'prelaunch.json').write_text(json.dumps(pre,indent=2,sort_keys=True)+'\n')
start=time.time()
with open(D/'run.stdout.txt','wb') as o,open(D/'run.stderr.txt','wb') as e:
 p=subprocess.Popen([sys.executable,str(P/'run_geometry_serial.py'),*argv],stdout=o,stderr=e,cwd=D)
 (D/'run-status.json').write_text(json.dumps({'status':'RUNNING_SOURCE_ONLY_FAT_MASS_ALLOCATION','pid':p.pid,'start_unix':start,'physical_admission':False,'native_steps':0})+'\n')
 rc=p.wait()
r={'status':'RUN_COMPLETE_SOURCE_ONLY_FAT_MASS_ALLOCATION' if rc==0 else 'RUN_EXIT_NONZERO_SOURCE_ONLY_FAT_MASS_ALLOCATION','exit_code':rc,'elapsed_s':time.time()-start,'prelaunch_sha256':sha(D/'prelaunch.json'),'result_path':str(out),'result_sha256':sha(out) if out.exists() else None,'stdout_sha256':sha(D/'run.stdout.txt'),'stderr_sha256':sha(D/'run.stderr.txt'),'physical_admission':False,'native_steps':0}
(D/'run-status.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps(r,indent=2,sort_keys=True))
raise SystemExit(rc)

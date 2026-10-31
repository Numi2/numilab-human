"""Replay current Skin chain and retry the existing 2-to-3 owner on its changed topology."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'current-skin-quality-373-root';out.mkdir(exist_ok=False)
r=json.loads((H/'connected-skin-quality-370-root/skin-run001/receipt.json').read_text())
checks=[];started=time.monotonic()
def call(tag,cmd):
 dest=out/tag
 full=[PY,str(H/cmd[0])]+list(map(str,cmd[1:]))+['--output',str(dest)]
 with (out/(tag+'.log')).open('w') as f:
  p=subprocess.run(full,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{tag} failed: {p.returncode}')
 receipt=dest/'receipt.json'
 if not receipt.exists():receipt=dest/'report.json'
 rr=json.loads(receipt.read_text())
 entry={'tag':tag,'command':full,'receipt':str(receipt),'receipt_sha256':sha(receipt),'status':rr['status']}
 checks.append(entry)
 print(json.dumps(entry),flush=True)
 (out/'progress.json').write_text(json.dumps(checks,indent=2)+'\n')
 return rr
for stage in r['history']:
 d=Path(stage['report']).parent
 kind=stage['phase'].split('-')[-1]
 if kind=='retire':
  call('verify-370-retire',['verify_interior_node_retirement_326_root.py','--input',stage['input'],'--proposal',stage['candidate'],'--report',stage['report']])
 elif kind=='condition':
  call('verify-370-condition',['verify_inferred_coordinate_replay_330_root.py','--input',stage['input'],'--proposal',stage['candidate'],'--report',stage['report'],'--registration-map',H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv','--source-node',H/'combined-child-handoff-142/skin.node'])
 elif kind=='kernel':
  call('verify-370-kernel',['verify_cached_kernel_chain_338_root.py','--input',stage['input'],'--step',d])
current=Path(r['final_candidate']);before=r['native_quality_failures_after']
assert sha(current)==r['final_candidate_sha256']
for k in range(4):
 tag=f'face-flips-{k:03d}'
 rr=call(tag,['fixed_boundary_face_flips_npz_313_root.py','--input-npz',current,'--seconds','30','--max-operations','10000'])
 if rr['passing_operations']==0:break
 candidate=out/tag/'fixed-boundary-volume-proposal.npz'
 call('verify-'+tag,['verify_face_flips_generic_374_root.py','--input',current,'--proposal',candidate])
 current=candidate
summary={'status':'PASS_SKIN_CHAIN_AND_BOUNDED_FACE_FLIP_REPLAY','physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__),'checks':checks,'final_candidate':str(current),'final_candidate_sha256':sha(current),'quality_before':before,'quality_after':rr['bad_cells_after'],'elapsed_s':time.monotonic()-started}
(out/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)

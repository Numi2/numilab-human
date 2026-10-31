"""Matched construction controls and bounded optional paired-I net-quality trial."""
from pathlib import Path
import json,hashlib,subprocess,numpy as np,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-quality-exchange-466-root';out.mkdir(exist_ok=False)
pins={
 'Skin':(H/'skin-interior-LP-455-root/run001/conditioned-volume-proposal.npz','cdeccad0b3aa6b5e6171dce2f62691a08b62dbf136950d1abb140d2c1004ddf3'),
 'Fat':(H/'fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/fat-current-typed-target-rebound.npz','72c851692580a639f89a4916bc4bf5c38331be9b6def89322eccf02ee66ea6ec'),
 'I':(H/'paired-I-interface-399-root/iteration002/typed365/candidate-I-typed-current.npz','fec15d6b59e9094952845cbeaaf418b91a4b2f5d0405d1c58b7e3e2e752ec406')}
for p,pin in pins.values():assert sha(p)==pin
roster=H/'d-shortedge-quotient-414-next-001-preflight-attempt004/allowed-D-edge-map.npz';assert sha(roster)=='439a1df81c608ebb153992d884cf8356e507cd228e20346e97a2ef46cce0aa95'
comp0=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/comp0-full-edge-star-exclusion.npz';assert sha(comp0)=='bfb88500119d48ff09c3af5358a405302ea29f152252d7b8dccbd6b4275ac893'
comp1=H/'d-shortedge-quotient-414-next-001-input-attempt004/comp1-current-output-protection.npz';assert sha(comp1)=='2d578f66c9c4b62787995dc83de3f214edb0adae3002bf7dc38d5bad298deae4'
with np.load(pins['Fat'][0]) as z:p=z['points_world_m'];t=z['tetrahedra']
with np.load(roster) as z:rows=np.unique(z['D_orientation_affected_cell_rows'])
protected=set(map(int,t[rows].ravel()))
with np.load(comp0) as z:protected.update(map(int,z['comp0_full_protected_point_rows']))
with np.load(comp1) as z:protected.update(map(int,z['global_point_rows']))
extra=out/'protected-comp0-comp1-all-next-D-orientation-stars.npz'
np.savez_compressed(extra,points_world_m=p,affected_volume_point_rows=np.asarray(sorted(protected),np.int64),all_next_D_candidate_cell_rows=rows)
commands=[];start=time.monotonic()
def run(owner,name,maximum,exchange=False):
 args=[PY,str(H/owner),'--output',str(out/name),'--maximum-edge-m','.001','--maximum-operations',str(maximum),'--seconds','45','--protect-fat-points',str(extra),'--protect-fat-points-sha256',sha(extra)]
 for role,(path,pin) in pins.items():args+=['--'+role.lower(),str(path),'--'+role.lower()+'-sha256',pin]
 if exchange:args+=['--allow-construction-quality-exchange']
 commands.append(args)
 with (out/(name+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 return json.loads((out/name/'report.json').read_text())
a=run('repair_paired_I_interfaces_indexed_384_root.py','strict384-control',32)
b=run('repair_paired_I_quality_exchange_462_root.py','strict462-control',32)
assert a['passing_operations']==b['passing_operations']==32
assert a['operations']==b['operations']
for ka,kb in [(a['candidate_I'],b['candidate_I'])]+[(x['output'],y['output']) for x,y in zip(a['roles'],b['roles'])]:
 with np.load(ka) as x,np.load(kb) as y:
  assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
control={'status':'PASS_379_STRICT_DEFAULT_PAIRED_I_EXACT_ARRAY_PARITY','operations':32,'physical_admission':False,'native_steps':0,'reference_report_sha256':sha(out/'strict384-control/report.json'),'candidate_report_sha256':sha(out/'strict462-control/report.json')}
(out/'strict-default-parity.json').write_text(json.dumps(control,indent=2)+'\n');print(json.dumps(control),flush=True)
trial=run('repair_paired_I_quality_exchange_462_root.py','exchange462',2048,True)
entry={'status':'BOUNDED_PAIRED_I_EXCHANGE_CONSTRUCTED_REQUIRES_TYPED_AND_FULL_CONTEXT_REPLAY','physical_admission':False,'native_steps':0,'raw_report':str(out/'exchange462/report.json'),'raw_report_sha256':sha(out/'exchange462/report.json'),'operations':trial['passing_operations'],'quality':[{k:r[k] for k in ['role','bad_before','bad_after']} for r in trial['roles']],'construction_new_retained_quality_failures':sum(x['new_retained_quality_failures'] for r in trial['roles'] for x in r['operations']),'protected_next_D_cell_count':len(rows),'protected_point_count':len(protected),'commands':commands,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps({k:v for k,v in entry.items() if k!='commands'}),flush=True)

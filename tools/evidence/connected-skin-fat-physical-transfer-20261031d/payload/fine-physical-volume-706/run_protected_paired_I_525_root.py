
"""Protected paired-interface batch using current authenticated geometry and existing owners."""
from pathlib import Path
import json,hashlib,subprocess,time,numpy as np
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-current-protected-525-root';out.mkdir(exist_ok=False);start=time.monotonic();commands=[]
def call(tag,owner,args,check=True):
 cmd=[PY,str(H/owner),*map(str,args)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if check:assert p.returncode==0,(tag,p.returncode)
 return p.returncode
# Default behavior parity against the already qualified legacy input, not current inferred-node motion.
old=json.loads((H/'paired-I-quality-exchange-466-root/receipt.json').read_text())['commands'][1]
for owner,name in [('repair_paired_I_quality_exchange_462_root.py','strict462'),('repair_paired_I_current_coordinates_520_root.py','strict520')]:
 argv=list(map(str,old[2:]));argv[argv.index('--output')+1]=str(out/name);argv[argv.index('--maximum-operations')+1]='4';argv[argv.index('--seconds')+1]='1200'
 call(name,owner,argv)
a=json.loads((out/'strict462/report.json').read_text());b=json.loads((out/'strict520/report.json').read_text())
assert a['operations']==b['operations'] and a['passing_operations']==b['passing_operations']==4
for x,y in [(a['candidate_I'],b['candidate_I'])]+[(x['output'],y['output']) for x,y in zip(a['roles'],b['roles'])]:
 with np.load(x) as z,np.load(y) as w:assert set(z.files)==set(w.files) and all(np.array_equal(z[k],w[k]) for k in z.files)
(out/'strict-default-parity.json').write_text(json.dumps({'status':'PASS_DEFAULT_STRICT_EXACT_ARRAY_PARITY','operations':4,'reference_report_sha256':sha(out/'strict462/report.json'),'candidate_report_sha256':sha(out/'strict520/report.json'),'physical_admission':False,'native_steps':0},indent=2)+'\n')
state=json.loads((H/'checkpoint-current523-root.json').read_text())['current_verified_state_20261031']
base=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight'
cfg=json.loads((base/'run-input-500.json').read_text());prot=cfg['protection_sidecar'];pf=Path(prot['path'])
assert sha(pf)==prot['sha256']=='aaa7b1d01d354b6ad18a885359738b84f267e31cb60a9794665f43f98980365d'
roster=base/'candidate-orientation-affected-point-roster.npz';assert sha(roster)=='47c03dc1fdfa2e1bf871ff083762b684bc4bbd70230b1dcebbad4e61e807b080'
fat=Path(state['Fat_current']['path']);assert sha(fat)==state['Fat_current']['sha256']=='5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42'
with np.load(fat) as z:points=z['points_world_m']
with np.load(pf) as z:pp=z['protected_volume_point_rows']
with np.load(roster) as z:
 rr=z['Fat_candidate_orientation_affected_point_rows']
 assert np.array_equal(points[rr].view('u4'),z['Fat_candidate_orientation_affected_points_world_m_f32'].view('u4'))
 assert str(z['Fat_current_candidate_sha256'][0])==sha(fat)
protected=np.unique(np.r_[pp,rr]);extra=out/'protected-all-D500-and-comp0-comp1-points.npz'
np.savez_compressed(extra,points_world_m=points,affected_volume_point_rows=protected,D500_affected_point_rows=rr,comp0_comp1_protected_point_rows=pp)
args=['--output',out/'exchange520','--maximum-edge-m','.003','--maximum-operations','2048','--seconds','60','--allow-construction-quality-exchange','--use-certified-current496-prefix','--protect-fat-points',extra,'--protect-fat-points-sha256',sha(extra)]
for role,e in [('Skin',state['Skin_current']),('Fat',state['Fat_current']),('I',state['source_surfaces']['I'])]:
 assert sha(e['path'])==e['sha256'];args+=['--'+role.lower(),e['path'],'--'+role.lower()+'-sha256',e['sha256']]
call('exchange520','repair_paired_I_current_coordinates_520_root.py',args)
rp=out/'exchange520/report.json';r=json.loads(rp.read_text())
result={'status':'NO_CURRENT_PROTECTED_PAIRED_I_PROGRESS','raw_report':str(rp),'raw_report_sha256':sha(rp),'operations':r['passing_operations'],'quality':[{k:v[k] for k in ('role','bad_before','bad_after')} for v in r['roles']]}
if r['passing_operations']:
 call('typed463','bind_and_replay_paired_I_quality_exchange_463_root.py',['--report',rp,'--output',out/'typed463001'])
 tp=out/'typed463001/receipt.json';tr=json.loads(tp.read_text())
 cfg=json.loads((H/'joint-current-source-497-root/config354.json').read_text())
 for role,e in state['source_surfaces'].items():
  cfg['baseline_candidates'][role]=dict(e);cfg['candidate_surfaces'][role]=dict(e)
 cfg['candidate_surfaces']['I']={'path':tr['candidate'],'sha256':tr['candidate_sha256']};cfg['changed_roles']=['I']
 cp=out/'config354.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
 code=call('context354','combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py',['--config',cp,'--out',out/'audit354'],False)
 if code==0:
  ar=out/'audit354/report.json'
  call('skin369','bind_paired_I_skin_target_generic_369_root.py',['--report',rp,'--typed-receipt',tp,'--context',ar,'--output',out/'skin-target369'])
  fc=json.loads((H/'fat-comp0-connected-warning-repair-412/run-033-target-rebind-480/result-002/report.json').read_text())['inputs']
  rf=next(x for x in r['roles'] if x['role']=='Fat')
  fc['base_candidate']={'path':rf['output'],'sha256':rf['output_sha256']};fc['base_report']={'path':str(rp),'sha256':sha(rp)}
  fc['typed_I']={'path':tr['candidate'],'sha256':tr['candidate_sha256']};fc['typed_I_receipt']={'path':str(tp),'sha256':sha(tp)}
  d=state['source_surfaces']['D'];fc['typed_D']=dict(d)
  canon=json.loads((H/'fat-comp0-connected-warning-repair-412/run-042-current-54381-normalized-rebind/target-normalization-receipt.json').read_text())
  de=canon['role_pins']['D'];fc['typed_D_697_receipt']={'path':de['697_receipt_path'],'sha256':de['697_receipt_sha256']}
  fcp=out/'config397.json';fcp.write_text(json.dumps(fc,indent=2)+'\n')
  call('fat397','rebind_current_fat_target_397_root.py',['--config',fcp,'--output',out/'fat-target397'])
  result.update(status='PASS_PROTECTED_PAIRED_I_TYPED_COMPLETE_CONTEXT_AND_CURRENT_TARGETS',typed_receipt={'path':str(tp),'sha256':sha(tp)},context={'path':str(ar),'sha256':sha(ar)},Skin_target={'path':str(out/'skin-target369/receipt.json'),'sha256':sha(out/'skin-target369/receipt.json')},Fat_target={'path':str(out/'fat-target397/report.json'),'sha256':sha(out/'fat-target397/report.json')})
 else:result.update(status='FAIL_CLOSED_PAIRED_I_EXACT_CONTEXT',context_report=str(out/'audit354/report.json'))
result.update(physical_admission=False,native_steps=0,source330_mass_immutable=True,protected_D500_and_comp0_comp1_points=len(protected),commands=commands,producer_sha256=sha(__file__),elapsed_s=time.monotonic()-start)
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='commands'}),flush=True)

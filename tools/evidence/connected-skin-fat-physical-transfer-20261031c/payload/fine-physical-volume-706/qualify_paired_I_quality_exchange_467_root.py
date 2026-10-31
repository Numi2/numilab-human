"""Full current anatomical context and existing target binding for paired-I construction."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python';root=H/'paired-I-quality-exchange-466-root';out=root/'qualification467001';out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rp=root/'exchange462/report.json';pr=root/'typed463001/receipt.json';r=json.loads(rp.read_text());p=json.loads(pr.read_text())
assert p['status']=='PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY' and p['input_report_sha256']==sha(rp)
start=time.monotonic();commands=[]
def call(name,args):
 cmd=[PY,*map(str,args)];commands.append(cmd)
 with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
cfg=json.loads((H/'joint-D309-I399-context-398-root/config354.json').read_text())
current={'O':(H/'outer-O-witness-driven-sequence-448-root/iteration002/typed434001/candidate-O-typed-current.npz','7a3954dc5b655e6ac0601645a1fcaa809710b89b142bed63e13d477c89c66dc5'),'D':(H/'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips/candidate-D-three-warning-flips-typed.npz','5bcbe4c42618b84399edd9e878c192598efaa0c7c213b84b22ea85be1e40fd40')}
for role,(path,pin) in current.items():
 assert sha(path)==pin
 for key in ('baseline_candidates','candidate_surfaces'):cfg[key][role]={'path':str(path),'sha256':pin}
cfg['baseline_candidates']['I']={'path':r['source_I'],'sha256':r['source_I_sha256']}
cfg['candidate_surfaces']['I']={'path':p['candidate'],'sha256':p['candidate_sha256']};cfg['changed_roles']=['I']
cp=out/'config354.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
call('context354',[H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py','--config',cp,'--out',out/'audit354'])
ar=out/'audit354/report.json';a=json.loads(ar.read_text());assert a['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
call('skin369',[H/'bind_paired_I_skin_target_generic_369_root.py','--report',rp,'--typed-receipt',pr,'--context',ar,'--output',out/'skin-target369'])
old=json.loads((H/'fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/report.json').read_text())
fc=old['inputs'];rr=next(v for v in r['roles'] if v['role']=='Fat')
fc['base_candidate']={'path':rr['output'],'sha256':rr['output_sha256']};fc['base_report']={'path':str(rp),'sha256':sha(rp)}
fc['typed_I']={'path':p['candidate'],'sha256':p['candidate_sha256']};fc['typed_I_receipt']={'path':str(pr),'sha256':sha(pr)}
assert set(fc)==set(('base_candidate','base_report','typed_I','typed_I_receipt','typed_D','typed_D_697_receipt','wall_csv','collar','wall_receipt'))
fcp=out/'config397.json';fcp.write_text(json.dumps(fc,indent=2)+'\n')
call('fat397',[H/'rebind_current_fat_target_397_root.py','--config',fcp,'--output',out/'fat-target397'])
sr=out/'skin-target369/receipt.json';fr=out/'fat-target397/report.json'
proof={'status':'PASS_PAIRED_I_EXCHANGE_TYPED_COMPLETE_CONTEXT_AND_CURRENT_TARGETS','physical_admission':False,'native_steps':0,'typed_receipt':{'path':str(pr),'sha256':sha(pr)},'complete_context':{'path':str(ar),'sha256':sha(ar)},'Skin_target':{'path':str(sr),'sha256':sha(sr)},'Fat_target':{'path':str(fr),'sha256':sha(fr)},'native_quality':{'Skin':1135,'Fat':23997},'construction_new_retained_quality_failures':12,'source330_mass':'IMMUTABLE_COMPLETE_CURRENT_M_OVER_4_REBIND_REQUIRED','commands':commands,'elapsed_s':time.monotonic()-start,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='commands'}),flush=True)

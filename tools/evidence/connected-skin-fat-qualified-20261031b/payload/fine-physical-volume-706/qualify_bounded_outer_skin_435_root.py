"""Qualify bounded outer-O quotient with existing complete-context and Skin-target owners."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
root=H/'outer-O-quotient-433-root';r=json.loads((root/'run001/report.json').read_text())
proof=root/'typed434001/receipt.json';pr=json.loads(proof.read_text())
assert pr['status']=='PASS_BOUNDED_OUTER_O_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY'
assert pr['input_report_sha256']==sha(root/'run001/report.json')
out=root/'qualification435001';out.mkdir(exist_ok=False);start=time.monotonic()
cfg=json.loads((H/'joint-D309-I399-context-398-root/config354.json').read_text())
for role in ('I','D'):cfg['baseline_candidates'][role]=cfg['candidate_surfaces'][role]
dp=H/'fat614-comp1-target-D-refinement-407-004/candidate-D-comp1-current-target-refinement-typed.npz'
assert sha(dp)=='bf0cef3ab74566428cd46f38716233745b0e2dfb21d7d50d8cfe7b2790b82c38'
for key in ('baseline_candidates','candidate_surfaces'):cfg[key]['D']={'path':str(dp),'sha256':sha(dp)}
cfg['candidate_surfaces']['O']={'path':pr['candidate'],'sha256':pr['candidate_sha256']}
assert cfg['baseline_candidates']['O']['sha256']==r['source_O_sha256']
cfg['changed_roles']=['O']
cp=out/'config354.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
def call(tag,args):
 with (out/(tag+'.log')).open('w') as f:subprocess.run([PY,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
call('context354',[H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py','--config',cp,'--out',out/'audit354'])
ctpath=out/'audit354/report.json';ct=json.loads(ctpath.read_text())
assert ct['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
call('skin-target437',[H/'bind_bounded_outer_O_skin_target_437_root.py','--report',root/'run001/report.json','--typed-receipt',proof,'--context',ctpath,'--output',out/'skin-target437'])
target=out/'skin-target437/receipt.json';tr=json.loads(target.read_text())
assert tr['status']=='PASS_EXACT_BOUNDED_O_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY'
res={'status':'PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET','physical_admission':False,'native_steps':0,'raw_report':str(root/'run001/report.json'),'raw_report_sha256':sha(root/'run001/report.json'),'typed_proof':str(proof),'typed_proof_sha256':sha(proof),'context_report':str(ctpath),'context_sha256':sha(ctpath),'target_receipt':str(target),'target_receipt_sha256':sha(target),'candidate':tr['candidate'],'candidate_sha256':tr['candidate_sha256'],'Skin_native_quality_before':r['roles'][0]['bad_before'],'Skin_native_quality_after':r['roles'][0]['bad_after'],'back412_unchanged':True,'source_mass_immutable':True,'mass_mapping':'NOT_ADMITTED_NEW_CELLS_REQUIRE_CONSERVATIVE_REBIND','native_motion':'NOT_RUN','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)

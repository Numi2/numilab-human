"""Bind final combined D309/I399 geometry through existing typed-target owner397."""
from pathlib import Path
import json,hashlib,subprocess,time,numpy as np
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'current-D309-I399-target-406-root';out.mkdir(exist_ok=False)
vroot=H/'combined-D309-I399-volume-404-root'
vr=json.loads((vroot/'report.json').read_text())
ind=vroot/'independent-verification-002.json'
assert sha(ind)=='661b48b3f1eac78b7ad694aca8970854d963f387689e7d2f5c8387fb4d2519ab'
ir=H/'paired-I-interface-399-root/iteration002/typed365/receipt.json';tr=json.loads(ir.read_text())
dr=H/'d-shortedge-quotient-385-run006/actual-697-D-load-003.json';d=json.loads(dr.read_text())
joint=H/'joint-D309-I399-context-398-root/receipt.json';j=json.loads(joint.read_text())
assert j['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
oldr=json.loads((H/'fat614-yy-volume-replay-004/run-014-current-target-rebound-v3/report.json').read_text())
cfg={k:oldr['inputs'][k] for k in ['wall_csv','collar','wall_receipt']}
def pin(p):return {'path':str(p),'sha256':sha(p)}
cfg.update(base_candidate=pin(vr['candidate']),base_report=pin(vroot/'report.json'),typed_I=pin(tr['candidate']),typed_I_receipt=pin(ir),typed_D=pin(d['typed_source_path']),typed_D_697_receipt=pin(dr))
config=out/'pins397.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
cmd=[PY,str(H/'rebind_current_fat_target_397_root.py'),'--config',str(config),'--output',str(out/'target397')]
with (out/'execution.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
rp=out/'target397/report.json';r=json.loads(rp.read_text())
assert r['actual_vs_typed_target']['missing_oriented_records']==127 and r['actual_vs_typed_target']['extra_oriented_records']==197
assert r['flux']['target_minus_actual_det6']==oldr['flux']['target_minus_actual_det6']
with np.load(r['output_candidate']['path']) as z:
 assert np.array_equal(z['desired_boundary_source_group'],np.r_[np.zeros(r['typed_surface_counts']['I_faces'],np.int8),np.ones(r['typed_surface_counts']['D_faces'],np.int8),np.full(90,2,np.int8)])
receipt={'status':'PASS_JOINT_TYPED_CURRENT_TARGET_REBIND_WITH_PRESERVED_RESIDUAL','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','geometry_report':pin(vroot/'report.json'),'independent_volume_replay':pin(ind),'joint_anatomy_context':pin(joint),'target_report':pin(rp),'candidate':r['output_candidate']['path'],'candidate_sha256':r['output_candidate']['sha256'],'native_quality_failures':vr['bad_after'],'points_and_connectivity_unchanged':True,'missing_target_faces':127,'extra_actual_faces':197,'exact_unresolved_flux_unchanged':True,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

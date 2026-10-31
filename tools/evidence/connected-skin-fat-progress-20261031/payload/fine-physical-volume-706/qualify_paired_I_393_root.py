"""Bind and audit the next paired interface through existing365/354/369 owners."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
root=H/'paired-I-interface-392-root';rpath=root/'run001/report.json';r=json.loads(rpath.read_text())
assert r['status']=='PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT' and r['passing_operations']>0
out=root/'qualification393001';out.mkdir(exist_ok=False);start=time.monotonic()
commands=[]
def call(tag,cmd):
 with (out/(tag+'.log')).open('w') as f:subprocess.run([PY,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
 commands.append([PY,*map(str,cmd)])
 print(tag,flush=True)
typed=out/'typed365'
call('bind365',[H/'bind_and_replay_paired_I_generic_365_root.py','--report',rpath,'--output',typed])
pr=json.loads((typed/'receipt.json').read_text())
baseconfig=H/'paired-I-interface-364-root/typed-replay365001/combined-audit-config-354.json'
cfg=json.loads(baseconfig.read_text())
cfg['baseline_candidates']['I']={'path':r['source_I'],'sha256':r['source_I_sha256']}
cfg['candidate_surfaces']['I']={'path':pr['candidate'],'sha256':pr['candidate_sha256']}
assert cfg['candidate_surfaces']['D']['sha256']=='18517e7dc0b1b25d67eea9691ad98bdbcfd07d18b9e1fe45f25b95515ee99ade'
cfg['changed_roles']=['I']
config=out/'combined-config354.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
context=out/'context354'
call('context354',[H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py','--config',config,'--out',context])
ct=json.loads((context/'report.json').read_text());assert ct['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
bind=out/'skin-target369'
call('bind369',[H/'bind_paired_I_skin_target_generic_369_root.py','--report',rpath,'--typed-receipt',typed/'receipt.json','--context',context/'report.json','--output',bind])
br=json.loads((bind/'receipt.json').read_text())
receipt={'status':'PASS_CURRENT_PAIRED_INTERFACE_TYPED_VOLUME_AND_COMPLETE_CONTEXT','physical_admission':False,'native_steps':0,'input_report':str(rpath),'input_report_sha256':sha(rpath),'typed':{'path':str(typed/'receipt.json'),'sha256':sha(typed/'receipt.json')},'context':{'path':str(context/'report.json'),'sha256':sha(context/'report.json')},'skin_target':{'path':str(bind/'receipt.json'),'sha256':sha(bind/'receipt.json')},'roles':[{'role':a['role'],'input_sha256':a['input_sha256'],'output':a['output'],'output_sha256':a['output_sha256'],'bad_before':a['bad_before'],'bad_after':a['bad_after']} for a in r['roles']],'operations':r['passing_operations'],'changed_source_faces':ct['changed_face_total'],'outer_skin_back_source_geometry_unchanged':True,'producer_sha256':sha(__file__),'commands':commands,'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

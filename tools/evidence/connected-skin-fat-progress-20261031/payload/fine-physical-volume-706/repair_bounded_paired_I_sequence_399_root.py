"""Two bounded repeated paired-interface batches through existing384/365/354/369 owners."""
from pathlib import Path
import json,hashlib,subprocess,time,os
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-interface-399-root';out.mkdir(exist_ok=False)
root=H/'paired-I-interface-394-root'
r=json.loads((root/'run001/report.json').read_text())
qpath=root/'qualification395001/receipt.json';qr=json.loads(qpath.read_text())
assert qr['status']=='PASS_CURRENT_PAIRED_INTERFACE_TYPED_VOLUME_AND_COMPLETE_CONTEXT'
tr=json.loads(Path(qr['typed']['path']).read_text());sr=json.loads(Path(qr['skin_target']['path']).read_text())
pins={'skin':(sr['candidate'],sr['candidate_sha256']),'fat':next((x['output'],x['output_sha256']) for x in r['roles'] if x['role']=='Fat'),'i':(tr['candidate'],tr['candidate_sha256'])}
mask=H/'paired-I-interface-392-root/protected-D006-cavity021-and-prior-stars.npz'
assert sha(mask)=='69297a3859b7038732df1eed551ebdda0916d98c1862bfdfc0f07ac14b5bff4b'
basecfg=json.loads((root/'qualification395001/combined-config354.json').read_text())
start=time.monotonic();summaries=[]
def call(tag,cmd,where):
 with (where/(tag+'.log')).open('w') as f:subprocess.run([PY,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(tag,flush=True)
for iteration in range(1,3):
 s=os.statvfs(H)
 if s.f_bavail*s.f_frsize<12*1024**3:print('RESERVE_STOP',flush=True);break
 where=out/('iteration%03d'%iteration);where.mkdir()
 cmd=[H/'repair_paired_I_interfaces_indexed_384_root.py']
 for role,(path,pin) in pins.items():
  assert sha(path)==pin;cmd+=['--'+role,path,'--'+role+'-sha256',pin]
 cmd+=['--protect-fat-points',mask,'--protect-fat-points-sha256',sha(mask),'--output',where/'batch','--maximum-edge-m','0.001','--maximum-operations','4096','--seconds','90']
 call('batch384',cmd,where)
 report=where/'batch/report.json';rr=json.loads(report.read_text())
 if not rr['passing_operations']:break
 typed=where/'typed365'
 call('bind365',[H/'bind_and_replay_paired_I_generic_365_root.py','--report',report,'--output',typed],where)
 pr=json.loads((typed/'receipt.json').read_text())
 cfg=json.loads(json.dumps(basecfg));cfg['baseline_candidates']['I']={'path':rr['source_I'],'sha256':rr['source_I_sha256']}
 cfg['candidate_surfaces']['I']={'path':pr['candidate'],'sha256':pr['candidate_sha256']};cfg['changed_roles']=['I']
 config=where/'combined-config354.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
 context=where/'context354'
 call('context354',[H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py','--config',config,'--out',context],where)
 ct=json.loads((context/'report.json').read_text());assert ct['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
 target=where/'skin-target369'
 call('target369',[H/'bind_paired_I_skin_target_generic_369_root.py','--report',report,'--typed-receipt',typed/'receipt.json','--context',context/'report.json','--output',target],where)
 sr=json.loads((target/'receipt.json').read_text())
 totalgain=sum(x['bad_before']-x['bad_after'] for x in rr['roles'])
 pins={'skin':(sr['candidate'],sr['candidate_sha256']),'fat':next((x['output'],x['output_sha256']) for x in rr['roles'] if x['role']=='Fat'),'i':(pr['candidate'],pr['candidate_sha256'])}
 record={'iteration':iteration,'report':str(report),'report_sha256':sha(report),'typed':{'path':str(typed/'receipt.json'),'sha256':sha(typed/'receipt.json')},'context':{'path':str(context/'report.json'),'sha256':sha(context/'report.json')},'skin_target':{'path':str(target/'receipt.json'),'sha256':sha(target/'receipt.json')},'passing_operations':rr['passing_operations'],'combined_quality_gain':totalgain,'roles':[{k:x[k] for k in ['role','bad_before','bad_after','output','output_sha256']} for x in rr['roles']],'pins':pins,'physical_admission':False,'native_steps':0}
 summaries.append(record)
 final={'status':'PASS_CURRENT_PAIRED_INTERFACE_TYPED_VOLUME_AND_COMPLETE_CONTEXT','physical_admission':False,'native_steps':0,'source_qualification':str(qpath),'source_qualification_sha256':sha(qpath),'iterations':summaries,'final_pins':pins,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start,'outer_skin_and_back_unchanged':True,'mass_mapping':'NOT_RUN'}
 (out/'receipt.json').write_text(json.dumps(final,indent=2)+'\n')
 print(json.dumps(record),flush=True)
 if totalgain<50:break
print('BOUNDED_SEQUENCE_COMPLETE',flush=True)

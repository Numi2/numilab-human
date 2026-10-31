
from pathlib import Path
import json,hashlib,subprocess,time,numpy as np
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'outer-O-quality-exchange-484-root';out.mkdir(exist_ok=False)
started=time.monotonic()
skin=H/'paired-I-quality-exchange-466-root/qualification467001/skin-target369/skin-bound-current-I-target-volume-proposal.npz'
o=H/'outer-O-witness-driven-sequence-448-root/iteration002/typed434001/candidate-O-typed-current.npz'
assert sha(skin)=='980e1d2e69001410de16212781318533ebbbe2b52af8cb3886ec844be344e109'
assert sha(o)=='7a3954dc5b655e6ac0601645a1fcaa809710b89b142bed63e13d477c89c66dc5'
cfg=H/'joint-D414-I466-target-475-root/config354.json';cpin=sha(cfg)
base=[ '--skin',skin,'--skin-sha256',sha(skin),'--o',o,'--o-sha256',sha(o),'--maximum-edge-m','.001']
r=json.loads((H/'outer-O-witness-driven-sequence-448-root/iteration002/run001/report.json').read_text())
ex={x['path']:x['sha256'] for x in r.get('excluded_witness_reports',[])}
if r.get('witness_pruning'):
 x=H/'outer-O-witness-driven-sequence-448-root/iteration002/run001/report.json';ex[str(x)]=sha(x)
for p,pin in ex.items():
 assert sha(p)==pin;base+=['--exclude-witness-report',p,pin]
commands=[]
def call(tag,argv,check=True):
 cmd=[PY,*map(str,argv)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if check:assert r.returncode==0,(tag,r.returncode)
 return r.returncode
try:
 for name,owner in [('reference447','repair_bounded_outer_O_with_witnesses_447_root.py'),('strict478','repair_outer_O_quality_exchange_478_root.py')]:
  call(name,[H/owner,*base,'--output',out/name,'--maximum-operations','8','--seconds','1200'])
 parity=[]
 for filename in ['candidate-O-outer-quotient-raw.npz','skin-outer-O-quotient-volume-proposal.npz']:
  a=out/'reference447'/filename;b=out/'strict478'/filename
  with np.load(a) as x,np.load(b) as y:
   assert x.files==y.files
   for k in x.files:assert np.array_equal(x[k],y[k]),(filename,k)
   parity.append({'filename':filename,'arrays':len(x.files),'all_arrays_bit_identical':True,'reference_sha256':sha(a),'candidate_sha256':sha(b)})
 (out/'default-parity.json').write_text(json.dumps({'status':'PASS_DEFAULT_STRICT_ARRAY_PARITY','records':parity,'source447_sha256':sha(H/'repair_bounded_outer_O_with_witnesses_447_root.py'),'source478_sha256':sha(H/'repair_outer_O_quality_exchange_478_root.py')},indent=2)+'\n')
 root=out/'exchange';root.mkdir(exist_ok=False)
 call('exchange',[H/'repair_outer_O_quality_exchange_478_root.py',*base,'--output',root/'run001','--maximum-operations','1024','--seconds','60','--allow-construction-quality-exchange'])
 rr=json.loads((root/'run001/report.json').read_text())
 if rr['passing_operations']==0:
  result={'status':'NO_CONSTRUCTION_QUALITY_EXCHANGE_PROGRESS','raw_report':str(root/'run001/report.json'),'raw_report_sha256':sha(root/'run001/report.json')}
 else:
  call('typed479',[H/'bind_outer_O_quality_exchange_479_root.py','--report',root/'run001/report.json','--output',root/'typed434001'])
  code=call('context480',[H/'qualify_current_outer_O_480_root.py','--root',root,'--current-context-config',cfg,'--current-context-config-sha256',cpin],check=False)
  failures=[]
  for i in range(3):
   if code==0:break
   audit=root/'qualification435001/audit354/report.json'
   ar=json.loads(audit.read_text());assert ar['status']=='FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
   failures.append({'path':str(audit),'sha256':sha(audit)})
   nxt=out/('witness-pruned%03d'%(i+1))
   code=call('prune%03d'%i,[H/'prune_current_outer_O_481_root.py','--failed-root',root,'--output',nxt,'--current-context-config',cfg,'--current-context-config-sha256',cpin],check=False)
   root=nxt
  assert code==0,('bounded_exact_witness_pruning_exhausted',str(root))
  qp=root/'qualification435001/receipt.json';qr=json.loads(qp.read_text())
  assert qr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
  result={'status':'PASS_CURRENT_OUTER_O_QUALITY_EXCHANGE_FULL_CONTEXT_AND_TARGET','qualification':str(qp),'qualification_sha256':sha(qp),'source_skin':str(skin),'source_skin_sha256':sha(skin),'candidate':qr['candidate'],'candidate_sha256':qr['candidate_sha256'],'bad_before':qr['Skin_native_quality_before'],'bad_after':qr['Skin_native_quality_after'],'failed_contexts':failures}
 result.update(physical_admission=False,native_steps=0,native_motion='NOT_RUN',source330_mass_immutable=True,producer_sha256=sha(__file__),commands=commands,elapsed_s=time.monotonic()-started)
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except Exception as e:
 (out/'failure.json').write_text(json.dumps({'status':'FAIL_CLOSED_OUTER_O_QUALITY_EXCHANGE_TRIAL','error':repr(e),'physical_admission':False,'native_steps':0,'commands':commands},indent=2)+'\n')
 raise


"""Bounded current Skin/O quality batch, exact owners and complete context mandatory."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'outer-O-connected-quality-506-root';out.mkdir(exist_ok=False)
cp=H/'checkpoint-current502-root.json';assert sha(cp)=='73fd3c5bba68c3e82f6d30c573def32a5a60937cdd2e274c87b11a6e8650b275'
state=json.loads(cp.read_text())['current_verified_state_20261031']
skin=state['Skin_current'];o=state['source_surfaces']['O']
for e in (skin,o):assert sha(e['path'])==e['sha256']
cfg=H/'joint-current-source-497-root/config354.json'
assert sha(cfg)=='20f7a2d706c70afd655b0ea1da05b8de34f8d81d47934b023b52f6542464239a'
raw=H/'outer-O-quality-exchange-484-root/exchange/run001/report.json'
r=json.loads(raw.read_text())
ex={x['path']:x['sha256'] for x in r.get('excluded_witness_reports',[])}
base=['--skin',skin['path'],'--skin-sha256',skin['sha256'],'--o',o['path'],'--o-sha256',o['sha256'],'--maximum-edge-m','.003','--allow-construction-quality-exchange']
for p,pin in ex.items():
 assert sha(p)==pin
 base+=['--exclude-witness-report',p,pin]
commands=[];started=time.monotonic()
def call(tag,argv):
 cmd=[PY,*map(str,argv)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 return r.returncode
root=out/'candidate';root.mkdir()
assert call('batch478',[H/'repair_outer_O_quality_exchange_478_root.py',*base,'--output',root/'run001','--maximum-operations','2048','--seconds','60'])==0
rr=json.loads((root/'run001/report.json').read_text())
result={'status':'NO_BOUNDED_O_QUALITY_PROGRESS','raw_report':str(root/'run001/report.json'),'raw_sha256':sha(root/'run001/report.json'),'operations':rr['passing_operations'],'quality':[{k:r[k] for k in ('role','bad_before','bad_after')} for r in rr['roles']]}
if rr['passing_operations']:
 assert call('typed479',[H/'bind_outer_O_quality_exchange_479_root.py','--report',root/'run001/report.json','--output',root/'typed434001'])==0
 code=call('context480',[H/'qualify_current_outer_O_480_root.py','--root',root,'--current-context-config',cfg,'--current-context-config-sha256',sha(cfg)])
 failures=[]
 for i in range(3):
  if code==0:break
  audit=root/'qualification435001/audit354/report.json'
  ar=json.loads(audit.read_text());assert ar['status']=='FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
  failures.append({'path':str(audit),'sha256':sha(audit)})
  nxt=out/('witness-pruned%03d'%(i+1))
  code=call('prune%03d'%i,[H/'prune_current_outer_O_481_root.py','--failed-root',root,'--output',nxt,'--current-context-config',cfg,'--current-context-config-sha256',sha(cfg)])
  root=nxt
 result.update(status='PASS_CONNECTED_O_QUALITY_BATCH' if code==0 else 'FAIL_CLOSED_BOUNDED_CONTEXT_PRUNING_BUDGET',failures=failures,passing_root=str(root))
 if code==0:
  proof=root/'qualification435001/receipt.json';result['qualification']={'path':str(proof),'sha256':sha(proof),'receipt':json.loads(proof.read_text())}
result.update(physical_admission=False,native_steps=0,source330_mass_immutable=True,back412_unchanged=True,producer_sha256=sha(__file__),commands=commands,elapsed_s=time.monotonic()-started)
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('commands','qualification')}),flush=True)

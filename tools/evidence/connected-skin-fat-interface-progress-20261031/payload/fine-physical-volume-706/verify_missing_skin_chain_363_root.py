"""Complete omitted independent checks on the retained productive Skin chain."""
from pathlib import Path
import json,subprocess,hashlib,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
out=H/'current-skin-chain-independent-363-root-run002';out.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=H/'combined-child-handoff-142/skin.node';registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
work=[]
for rel in ['connected-skin-quality-340-root/skin-run001/cycle-00-retire','connected-skin-quality-342-root/skin-run001/cycle-00-retire','connected-skin-quality-356-root/skin-run001/cycle-00-retire']:
 d=H/rel;r=json.loads((d/'report.json').read_text());work.append(('retire',d,[PY,str(H/'verify_interior_node_retirement_326_root.py'),'--input',r['input'],'--proposal',r['output'],'--report',str(d/'report.json')]))
d=H/'connected-skin-quality-340-root/skin-run001/cycle-00-edge'
r=json.loads((d/'receipt.json').read_text())
work.append(('edge',d,[PY,str(H/'verify_fixed_boundary_connectivity_batch_327_root.py'),'--input-npz',r['input_npz'],'--batch',str(d)]))
d=H/'connected-skin-quality-342-root/skin-run001/cycle-00-condition';r=json.loads((d/'report.json').read_text())
work.append(('condition',d,[PY,str(H/'verify_inferred_coordinate_replay_330_root.py'),'--input',r['input'],'--proposal',r['output'],'--report',str(d/'report.json'),'--registration-map',str(registration),'--source-node',str(source)]))
d=H/'cached-joint-witness-297-root/skin-batch002';r=json.loads((d/'report.json').read_text())
work.append(('kernel',d,[PY,str(H/'verify_cached_kernel_chain_338_root.py'),'--input',r['input'],'--step',str(d)]))
checks=[];start=time.monotonic()
for i,(kind,d,cmd) in enumerate(work):
 target=out/('%02d-%s'%(i,kind));full=cmd+['--output',str(target)]
 with (out/('%02d.log'%i)).open('w') as f:subprocess.run(full,stdout=f,stderr=subprocess.STDOUT,check=True)
 rp=target/'receipt.json';r=json.loads(rp.read_text());assert r['status'].startswith('PASS_') and r['physical_admission'] is False and r['native_steps']==0
 entry={'stage':str(d),'kind':kind,'command':full,'verifier_sha256':sha(Path(cmd[1])),'receipt':str(rp),'receipt_sha256':sha(rp)}
 checks.append(entry);print(json.dumps(entry),flush=True)
 (out/'checkpoint.json').write_text(json.dumps({'physical_admission':False,'native_steps':0,'completed_checks':checks},indent=2)+'\n')
r={'status':'PASS_OMITTED_SKIN_CHAIN_INDEPENDENT_CHECKS','physical_admission':False,'native_steps':0,'checks':checks,'producer_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='checks'}),flush=True)


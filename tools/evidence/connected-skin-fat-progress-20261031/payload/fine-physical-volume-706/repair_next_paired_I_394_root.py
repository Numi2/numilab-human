"""Second bounded shared-interface batch on independently qualified392 geometry."""
from pathlib import Path
import json,hashlib,subprocess
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-interface-394-root';out.mkdir(exist_ok=False)
qualified=H/'paired-I-interface-392-root/qualification393001'
receipt=json.loads((qualified/'receipt.json').read_text());assert receipt['status']=='PASS_CURRENT_PAIRED_INTERFACE_TYPED_VOLUME_AND_COMPLETE_CONTEXT'
t=json.loads((qualified/'typed365/receipt.json').read_text());s=json.loads((qualified/'skin-target369/receipt.json').read_text())
r=json.loads((H/'paired-I-interface-392-root/run001/report.json').read_text())
fat=next(x for x in r['roles'] if x['role']=='Fat')
pins={'skin':(s['candidate'],s['candidate_sha256']),'fat':(fat['output'],fat['output_sha256']),'i':(t['candidate'],t['candidate_sha256'])}
cmd=[PY,str(H/'repair_paired_I_interfaces_indexed_384_root.py')]
for name,(path,pin) in pins.items():
 assert sha(path)==pin
 cmd+=['--'+name,path,'--'+name+'-sha256',pin]
mask=H/'paired-I-interface-392-root/protected-D006-cavity021-and-prior-stars.npz'
assert sha(mask)=='69297a3859b7038732df1eed551ebdda0916d98c1862bfdfc0f07ac14b5bff4b'
cmd+=['--protect-fat-points',str(mask),'--protect-fat-points-sha256',sha(mask),'--output',str(out/'run001'),'--maximum-edge-m','0.001','--maximum-operations','4096','--seconds','90']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'execution.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
r=json.loads((out/'run001/report.json').read_text())
print(json.dumps({k:r[k] for k in ['status','passing_operations','elapsed_s','rejections']}),flush=True)
print(json.dumps([{k:a[k] for k in ['role','bad_before','bad_after','output','output_sha256']} for a in r['roles']]),flush=True)

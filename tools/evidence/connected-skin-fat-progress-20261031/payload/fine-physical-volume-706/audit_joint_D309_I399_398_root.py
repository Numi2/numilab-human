"""Recheck final D309 and I399 together through existing complete-context354 owner."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'joint-D309-I399-context-398-root';out.mkdir(exist_ok=False)
r=json.loads((H/'paired-I-interface-399-root/receipt.json').read_text())
cfg=json.loads((H/'paired-I-interface-399-root/iteration002/combined-config354.json').read_text())
ip,pin=r['final_pins']['i'];assert sha(ip)==pin=='fec15d6b59e9094952845cbeaaf418b91a4b2f5d0405d1c58b7e3e2e752ec406'
dp=H/'d-shortedge-quotient-385-run006/typed-rebind-003/candidate-D-typed-current.npz'
assert sha(dp)=='9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb'
oldI=H/'paired-I-interface-377-root/qualification378001/typed365/candidate-I-typed-current.npz'
cfg['baseline_candidates']['I']={'path':str(oldI),'sha256':sha(oldI)}
cfg['candidate_surfaces']['I']={'path':ip,'sha256':pin}
cfg['candidate_surfaces']['D']={'path':str(dp),'sha256':sha(dp)}
cfg['changed_roles']=['D','I']
config=out/'config354.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
cmd=[PY,str(H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py'),'--config',str(config),'--out',str(out/'audit354')]
with (out/'execution.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
ct=json.loads((out/'audit354/report.json').read_text());assert ct['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
receipt={'status':ct['status'],'physical_admission':False,'native_steps':0,'context_report':str(out/'audit354/report.json'),'context_sha256':sha(out/'audit354/report.json'),'config':str(config),'config_sha256':sha(config),'producer_sha256':sha(__file__),'changed_source_faces':ct['changed_face_total']}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

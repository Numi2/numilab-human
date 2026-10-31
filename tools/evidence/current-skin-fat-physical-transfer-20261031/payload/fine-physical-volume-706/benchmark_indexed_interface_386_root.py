"""Matched full-batch replay of indexed repair against the frozen377 result."""
from pathlib import Path
import subprocess,json,hashlib,time,numpy as np
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
rpath=H/'paired-I-interface-377-root/run002/report.json';r=json.loads(rpath.read_text())
assert r['passing_operations']==941
out=H/'matched-indexed-interface-386-root';out.mkdir(exist_ok=False)
cmd=[PY,str(H/'repair_paired_I_interfaces_indexed_384_root.py')]
for role in r['roles']:
 name=role['role'].lower();cmd+=['--'+name,role['input'],'--'+name+'-sha256',role['input_sha256']]
cmd+=['--i',r['source_I'],'--i-sha256',r['source_I_sha256'],'--protect-fat-points',r['extra_protected_fat_points'],'--protect-fat-points-sha256',r['extra_protected_fat_points_sha256'],'--output',str(out/'batch'),'--maximum-edge-m',str(r['maximum_edge_length_m']),'--maximum-operations',str(r['passing_operations']),'--seconds','120']
with (out/'execution.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
newpath=out/'batch/report.json';n=json.loads(newpath.read_text())
assert n['passing_operations']==r['passing_operations']
assert n['operations']==r['operations']
assert n['candidate_I_sha256']==r['candidate_I_sha256']
for a,b in zip(r['roles'],n['roles']):
 assert a['role']==b['role'] and a['input_sha256']==b['input_sha256'] and a['output_sha256']==b['output_sha256']
 assert a['operations']==b['operations'] and a['bad_after']==b['bad_after']
receipt={'status':'PASS_FULL_MATCHED_INDEXED_INTERFACE_BATCH_BYTE_PARITY','physical_admission':False,'native_steps':0,'reference_report':str(rpath),'reference_report_sha256':sha(rpath),'indexed_report':str(newpath),'indexed_report_sha256':sha(newpath),'same941operations_and_order':True,'source_I_npz_byte_identical':True,'Skin_and_Fat_npz_byte_identical':True,'reference_construction_elapsed_s':r['elapsed_s'],'indexed_construction_elapsed_s':n['elapsed_s'],'construction_throughput_speedup':r['elapsed_s']/n['elapsed_s'],'command':cmd,'producer_sha256':sha(__file__),'limitation':'Offline paired-interface construction throughput; not native physics or end-to-end simulation performance.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

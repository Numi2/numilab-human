#!/usr/bin/env python3
"""Reuse exact negative cavity work and continue only bounded productive fans."""
from pathlib import Path
import numpy as np,json,hashlib,subprocess,time,argparse
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--input-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rounds',type=int,default=4);a=ap.parse_args()
a.output.mkdir(exist_ok=False,parents=True);assert sha(a.input)==a.input_sha256
current=a.input;history=[];st=time.monotonic()
for k in range(a.rounds):
 d=a.output/('pass-%03d'%k);log=a.output/('pass-%03d.log'%k)
 with np.load(current) as z:
  cache=z['kernel_fan_no_progress_sha256'];assert cache.dtype==np.uint8 and cache.shape[1]==32
 with log.open('w') as f:subprocess.run([PY,str(H/'remesh_cached_joint_witness_fans_297_root.py'),
  '--input',str(current),'--output',str(d),'--prior-cache',str(current),'--prior-cache-sha256',sha(current),
  '--seconds','25','--maximum-attempts','4000','--maximum-points','150'],stdout=f,stderr=subprocess.STDOUT,check=True)
 r=json.loads((d/'report.json').read_text());assert r['physical_admission'] is False and r['native_steps']==0 and r['nonpositive']==0
 assert r['bad_after']<=r['bad_before'] and r['oriented_boundary_bit_identical'] and r['original_points_bit_identical']
 nxt=Path(r['output']);assert sha(nxt)==r['output_sha256']
 entry={'pass':k,'input':str(current),'input_sha256':sha(current),'candidate':str(nxt),'candidate_sha256':sha(nxt),
  'report_sha256':sha(d/'report.json'),'before':r['bad_before'],'after':r['bad_after'],'operations':r['passing_operations'],'elapsed_s':r['elapsed_s']}
 history.append(entry);current=nxt
 (a.output/'checkpoint.json').write_text(json.dumps({'physical_admission':False,'native_steps':0,'candidate':str(current),'candidate_sha256':sha(current),'history':history},indent=2)+'\n')
 print(json.dumps(entry),flush=True)
 if r['bad_before']-r['bad_after']<8:break
report={'status':'PASS_BOUNDED_CACHED_FAN_CONTINUATION','physical_admission':False,'native_steps':0,'input':str(a.input),'input_sha256':sha(a.input),'final_candidate':str(current),'final_candidate_sha256':sha(current),'before':history[0]['before'],'after':history[-1]['after'],'history':history,'producer_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-st}
(a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)

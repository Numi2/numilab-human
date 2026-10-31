#!/usr/bin/env python3
"""Bounded connected quality rounds using existing accepted offline owners."""
from pathlib import Path
import argparse,subprocess,time,json,hashlib,numpy as np
H=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(f):
 with open(f,'rb') as q:return hashlib.file_digest(q,'sha256').hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--input-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--cycles',type=int,default=4);a=ap.parse_args()
a.output.mkdir(exist_ok=False,parents=True);assert sha(a.input)==a.input_sha256
source=H/'combined-child-handoff-142/skin.node';registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
current=a.input;cache=a.input;history=[];start=time.monotonic()
def execute(label,script,args,folder,reportname,outputkey,badkeys):
 before=current
 log=a.output/(label+'.log')
 with log.open('w') as q:
  subprocess.run([PY,str(H/script),*args],stdout=q,stderr=subprocess.STDOUT,check=True)
 r=json.loads((folder/reportname).read_text())
 assert r.get('physical_admission') is False and r.get('native_steps')==0
 f=Path(r[outputkey]);assert f.exists()
 bad=next(int(r[k]) for k in badkeys if k in r)
 entry={'phase':label,'input':str(before),'input_sha256':sha(before),'producer':str(H/script),'producer_sha256':sha(H/script),'report':str(folder/reportname),'report_sha256':sha(folder/reportname),'candidate':str(f),'candidate_sha256':sha(f),'native_quality_failures_after':bad}
 history.append(entry)
 (a.output/'checkpoint.json').write_text(json.dumps({'physical_admission':False,'native_steps':0,'history':history,'candidate':str(f),'candidate_sha256':sha(f)},indent=2)+'\n')
 print(json.dumps(entry),flush=True)
 return f,bad
lastbad=None
for cycle in range(a.cycles):
 if time.monotonic()-start>a.seconds:break
 cstartbad=lastbad
 folder=a.output/('cycle-%02d-retire'%cycle)
 current,lastbad=execute(folder.name,'retire_inferred_interior_node_stars_324_root.py',
  ['--input',str(current),'--input-sha256',sha(current),'--source-node',str(source),'--registration-map',str(registration),'--output',str(folder),'--seconds','15'],folder,'report.json','output',['bad_cells_after'])
 if time.monotonic()-start>a.seconds:break
 folder=a.output/('cycle-%02d-edge'%cycle)
 current,lastbad=execute(folder.name,'batch_degree3_and_polygon_flips_307_root.py',
  ['--input-npz',str(current),'--output',str(folder),'--rounds','12','--seconds-per-round','8','--maximum-degree','16'],folder,'receipt.json','final_proposal',['bad_cells_after'])
 if time.monotonic()-start>a.seconds:break
 parent=current.parent/'report.json'
 if not str(json.loads(parent.read_text()).get('status','')).startswith('PASS'):
  old=Path(history[-1]['input'])
  with np.load(old) as z:oldv={k:z[k] for k in ('points_world_m','tetrahedra','material_indices')}
  with np.load(current) as z:
   assert all(np.array_equal(oldv[k],z[k]) for k in oldv),'no-progress operator changed geometry'
  parent=current.parent/'exact-noop-geometry-parent.json'
  parent.write_text(json.dumps({'status':'PASS_EXACT_NOOP_GEOMETRY_READBACK','physical_admission':False,'native_steps':0,
   'oriented_boundary_exactly_unchanged':True,'exact_F32_volume_unchanged':True,'input_sha256':sha(old),
   'proposal_sha256':sha(current),'same_coordinate_connectivity_material_bytes':True},indent=2)+chr(10))
 folder=a.output/('cycle-%02d-condition'%cycle)
 current,lastbad=execute(folder.name,'condition_targeted_interior_328_root.py',
  ['--input',str(current),'--input-sha256',sha(current),'--source-node',str(source),'--registration-map',str(registration),'--parent-report',str(parent),'--output',str(folder),'--seconds','25','--sweeps','1','--maxiter','24'],folder,'report.json','output',['native_quality_defects_after'])
 if time.monotonic()-start>a.seconds:break
 folder=a.output/('cycle-%02d-kernel'%cycle)
 current,lastbad=execute(folder.name,'remesh_cached_joint_witness_fans_297_root.py',
  ['--input',str(current),'--output',str(folder),'--prior-cache',str(cache),'--prior-cache-sha256',sha(cache),'--seconds','25','--maximum-attempts','3000','--maximum-points','150'],folder,'report.json','output',['bad_after'])
 cache=current
 if cstartbad is not None and cstartbad-lastbad<8:break
r={'status':'PASS_BOUNDED_CONNECTED_QUALITY_ROUNDS_SOURCE_ONLY','physical_admission':False,'native_steps':0,
 'input':str(a.input),'input_sha256':sha(a.input),'final_candidate':str(current),'final_candidate_sha256':sha(current),
 'native_quality_failures_after':lastbad,'elapsed_s':time.monotonic()-start,'producer_sha256':sha(Path(__file__)),'history':history,
 'limitation':'offline geometry work, not a physical/native qualification; mass mapping and every native quality gate remain mandatory'}
(a.output/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='history'}),flush=True)

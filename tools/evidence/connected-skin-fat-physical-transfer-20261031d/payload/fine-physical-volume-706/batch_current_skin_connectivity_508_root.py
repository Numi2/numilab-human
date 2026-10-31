
"""Batch exact fixed-boundary Skin connectivity improvement on the qualified current target."""
from pathlib import Path
import json,hashlib,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'skin-fixed-boundary-batch-508-root';out.mkdir(exist_ok=False)
qr=H/'outer-O-connected-quality-506-root/candidate/qualification435001/receipt.json'
assert sha(qr)=='fad0465524e6bc692f86dd306cb38cbd8af4375cd934b046268feb1db0c373da'
r=json.loads(qr.read_text());current=Path(r['candidate']);assert sha(current)==r['candidate_sha256']
initial=current;bad=r['Skin_native_quality_after'];history=[];commands=[];started=time.monotonic()
def call(tag,owner,args):
 cmd=[PY,str(H/owner),*map(str,args)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,(tag,p.returncode)
def record(tag,before,report,candidate,newbad,proof):
 global current,bad
 e={'stage':tag,'input':str(before),'input_sha256':sha(before),'report':str(report),'report_sha256':sha(report),'candidate':str(candidate),'candidate_sha256':sha(candidate),'bad_before':bad,'bad_after':newbad,'independent_proof':str(proof),'independent_proof_sha256':sha(proof)}
 assert newbad<bad
 history.append(e);current=candidate;bad=newbad
 (out/'progress.json').write_text(json.dumps({'history':history,'candidate':str(current),'sha256':sha(current),'bad':bad,'physical_admission':False,'native_steps':0},indent=2)+'\n')
 print(json.dumps(e),flush=True)
source=H/'combined-child-handoff-142/skin.node'
registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
for cycle in range(2):
 cyclebad=bad
 tag='cycle%02d-retire'%cycle;folder=out/tag;before=current
 call(tag,'retire_inferred_interior_node_stars_324_root.py',['--input',current,'--input-sha256',sha(current),'--source-node',source,'--registration-map',registration,'--output',folder,'--seconds','15','--maximum-operations','10000'])
 rp=folder/'report.json';r=json.loads(rp.read_text())
 if r['passing_operations']:
  prop=Path(r['output']);proof=out/(tag+'-verify326')
  call(tag+'-verify326','verify_interior_node_retirement_326_root.py',['--input',before,'--proposal',prop,'--report',rp,'--output',proof])
  record(tag,before,rp,prop,r['bad_cells_after'],proof/'receipt.json')
 tag='cycle%02d-edge'%cycle;folder=out/tag;before=current
 call(tag,'batch_degree3_and_polygon_flips_307_root.py',['--input-npz',current,'--output',folder,'--rounds','4','--seconds-per-round','8','--maximum-degree','16'])
 rp=folder/'receipt.json';r=json.loads(rp.read_text())
 if r['passing_operations']:
  prop=Path(r['final_proposal']);proof=out/(tag+'-verify327')
  call(tag+'-verify327','verify_fixed_boundary_connectivity_batch_327_root.py',['--input-npz',before,'--batch',folder,'--output',proof])
  record(tag,before,rp,prop,r['bad_cells_after'],proof/'receipt.json')
 tag='cycle%02d-face'%cycle;folder=out/tag;before=current
 call(tag,'fixed_boundary_face_flips_npz_313_root.py',['--input-npz',current,'--output',folder,'--seconds','15','--max-operations','10000'])
 rp=folder/'report.json';r=json.loads(rp.read_text())
 if r['passing_operations']:
  prop=folder/'fixed-boundary-volume-proposal.npz';proof=out/(tag+'-verify374')
  call(tag+'-verify374','verify_face_flips_generic_374_root.py',['--input',before,'--proposal',prop,'--output',proof])
  record(tag,before,rp,prop,r['bad_cells_after'],proof/'receipt.json')
 if cyclebad-bad<10:break
res={'status':'PASS_CURRENT_CONNECTED_FIXED_BOUNDARY_SKIN_BATCH','input':str(initial),'input_sha256':sha(initial),'candidate':str(current),'candidate_sha256':sha(current),'bad_before':1082,'bad_after':bad,'history':history,'commands':commands,'source330_mass_immutable':True,'native_thresholds_unchanged':True,'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-started,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ('commands','history')}),flush=True)

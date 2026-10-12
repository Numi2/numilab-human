from pathlib import Path
import json,hashlib,sys,importlib.util,gc
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back-forearm-native-014-016-back412-right177';B=A/'back-forearm-native-014-014-013-right177'
# Retained baseline uses the explicit verified scene name below.
B=A/'back-forearm-native-014-013-right177'
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
def req(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
sp=importlib.util.spec_from_file_location('fwd412',H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
out=D/'native-back412-audit-002';out.mkdir()
pins={};pin=lambda p:pins.update({str(p):sha(p)})
for p in (Path(__file__),H,Path(ca.__file__),Path(ci.__file__),D/'baseline/execution.json',D/'baseline/run-declaration.json'):pin(p)
ex=json.loads((D/'baseline/execution.json').read_text());decl=json.loads((D/'baseline/run-declaration.json').read_text())
req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'execution failed')
req(ex['native_argv_matches_prepared_cli_preview'] and ex['native_environment_matches_prepared_cli_preview'],'invocation changed')
for p,v in decl['immutable_assets'].items():req(sha(p)==v,'immutable changed')
r=D/'baseline/native-run';old=B/'baseline/native-run'
for p in (r/'resting-coupled.csv',old/'resting-coupled.csv'):pin(p)
req((r/'resting-coupled.csv').read_bytes()==(old/'resting-coupled.csv').read_bytes(),'physiology changed')
payloads=[x/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue' for x in (D,B)]
for p in payloads:pin(p)
ts=[h.load_tissue(p) for p in payloads];rows=[{int(x[6]):h.row_data(t,int(x[6])) for x in t['records']} for t in ts]
req(set(rows[0])==set(rows[1]),'row inventory changed')
for sid in rows[0]:
 n,o=rows[0][sid],rows[1][sid]
 req(np.array_equal(n['positions'],o['positions']) and n['binding_bytes']==o['binding_bytes'],'source coordinates/bindings changed')
 if sid not in (157,158):req(n['vertex_bytes']==o['vertex_bytes'],'untargeted source vertex data changed')
 if sid not in (157,158):req(np.array_equal(n['faces'],o['faces']),'untargeted source topology changed')
inventory=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
requests,*_=ca._load_target_inventory(inventory);requests=set(requests)|{(51005,s) for s in range(151,165)}
report={'complete':False,'anatomical_admission':False,'scope':'Three cap retriangulations only; residual crossings remain.','physiology_byte_exact_baseline':True,'source_positions_and_bindings_byte_exact':True,'changed_row_normals_recomputed':True,'captures':{},'pins':pins,'wall_seconds':ex['wall_seconds']}
for step in decl['capture_steps']:
 paths=[z/'accepted-geometry'/f'step-{step}.mrvpack' for z in (r,old)]
 receipts=[p.with_suffix('.receipt.json') for p in paths]
 for p in paths+receipts:pin(p)
 rr=[json.loads(p.read_text()) for p in receipts]
 req(all(x['physical_endpoint']=='accepted' and x['pack_file_sha256']==sha(p) for x,p in zip(rr,paths)),'capture invalid')
 req(rr[0]['accepted_body_poses']==rr[1]['accepted_body_poses'],'body poses changed')
 packs=[ca._pack_surfaces(p,requests) for p in paths];xyz,surfs,_=packs[0];ox,os,_=packs[1]
 req(set(surfs)==set(os),'surface inventory changed')
 unchanged=[k for k in surfs if k not in ((51005,157),(51005,158))]
 for k in unchanged:req(np.array_equal(xyz[np.asarray(surfs[k]['faces'])],ox[np.asarray(os[k]['faces'])]),'untargeted rendered surface changed '+str(k))
 checks={}
 for sid in (157,158):
  worlds=[]
  for pp,ss,source in ((xyz,surfs,rows[0][sid]),(ox,os,rows[1][sid])):
   faces=np.asarray(source['faces'],dtype=np.int64);native=np.asarray(ss[(51005,sid)]['faces'])
   diff=native-faces;req(np.all(diff==diff.flat[0]),'face remap differs')
   start=int(diff.flat[0]);worlds.append(np.asarray(pp[start:start+len(source['positions'])],dtype='<f4'))
  req(np.array_equal(worlds[0],worlds[1]),'rendered vertex motion changed')
  f=np.asarray(rows[0][sid]['faces'],dtype=np.int64);pts=[ci.float32_point_lattice_key(tuple(float(x) for x in p)) for p in worlds[0]]
  records=ci._records(pts,f.tolist())
  item={'exact_degenerate_faces':0,'positions_byte_exact':True}
  if step in (0,8000):
   current=ci._audit_pair(records,records,same_surface=True)
   oldrecords=ci._records(pts,np.asarray(rows[1][sid]['faces']).tolist())
   prior=ci._audit_pair(oldrecords,oldrecords,same_surface=True)
   item.update({'raw_self_pairs':current['count'],'baseline_raw_self_pairs':prior['count'],'pairs':current['triangle_pairs']})
   req(current['count']<=prior['count'],'self pairs increased')
  checks[str(sid)]=item
  np.savez_compressed(out/f'step-{step}-row-{sid}.npz',positions=worlds[0],faces=f)
 report['captures'][str(step)]={'unchanged_rendered_surfaces':len(unchanged),'body_poses_exact':True,'rows':checks}
 save(out/'report.json',report);print(json.dumps({'step':step,'rows':checks},default=int),flush=True)
 del packs,xyz,ox,surfs,os,records;gc.collect()
req(all(sha(p)==v for p,v in pins.items()),'audit input changed')
report['complete']=True;save(out/'report.json',report)


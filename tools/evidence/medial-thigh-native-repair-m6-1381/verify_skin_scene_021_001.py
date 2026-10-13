from pathlib import Path
import json,hashlib,sys,gc,struct,mmap
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back412-skin-reference-native-021-003'
B=A/'back-forearm-native-014-018-back412-recovery-16s'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
o=D/'skin-controlled-comparison-001';o.mkdir()
pins={}
def pin(p):pins[str(p)]=sha(p)
decl=json.loads((D/'baseline/run-declaration.json').read_text())
for z in (D,B):
 ex=json.loads((z/'baseline/execution.json').read_text())
 req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'failed run')
 pin(z/'baseline/execution.json');pin(z/'baseline/run-declaration.json')
for p in (Path(__file__),Path(ca.__file__)):pin(p)
for z in (D,B):pin(z/'baseline/native-run/resting-coupled.csv')
req(json.loads((D/'baseline/run-declaration.json').read_text())['owner_cli_preview']['asset_sha256'][json.loads((D/'baseline/run-declaration.json').read_text())['owner_cli_preview']['native_argv'][0]]==json.loads((B/'baseline/run-declaration.json').read_text())['owner_cli_preview']['asset_sha256'][json.loads((B/'baseline/run-declaration.json').read_text())['owner_cli_preview']['native_argv'][0]],'binary differs')
req((D/'baseline/native-run/resting-coupled.csv').read_bytes()==(B/'baseline/native-run/resting-coupled.csv').read_bytes(),'physiology mismatch')
inv=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
pin(inv);requests,*_=ca._load_target_inventory(inv);requests=set(requests)|{(51005,s) for s in range(151,165)}
changed={(51007,1)}
r={'complete':False,'scope':'Same-binary controlled comparison against N18; only the declared skin registration is excluded from unchanged-geometry gate.','physiology_byte_exact':True,'captures':{},'pins':pins}
for step in decl['capture_steps']:
 paths=[z/'baseline/native-run/accepted-geometry'/f'step-{step}.mrvpack' for z in (D,B)]
 for p in paths:pin(p);pin(p.with_suffix('.receipt.json'))
 receipts=[json.loads(p.with_suffix('.receipt.json').read_text()) for p in paths]
 for rec,p in zip(receipts,paths):req(rec['physical_endpoint']=='accepted' and rec['pack_file_sha256']==sha(p),'invalid capture')
 req(receipts[0]['accepted_body_poses']==receipts[1]['accepted_body_poses'],'body motion changed')
 keys=[]
 for p in paths:
  with p.open('rb') as st:
   mm=mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)
  hd=ca._PACK_HEADER.unpack_from(mm,0)
  sect=[ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size) for i in range(hd[2])]
  pr=next(x for x in sect if x[0]==4)
  keys.append({tuple(struct.unpack_from('<2I',mm,pr[2]+i*64+16)) for i in range(pr[4])})
  mm.close()
 shared=requests&keys[0]&keys[1]
 packs=[ca._pack_surfaces(p,shared) for p in paths]
 xyz,surfs,_=packs[0];ox,os,_=packs[1]
 unchanged=(set(surfs)&set(os))-changed
 for k in unchanged:req(np.array_equal(xyz[np.asarray(surfs[k]['faces'])],ox[np.asarray(os[k]['faces'])]),'rendered geometry changed '+str(k))
 r['captures'][str(step)]={'body_poses_exact':True,'unchanged_rendered_surfaces':len(unchanged),'excluded_declared_changed_surfaces':sorted(changed),'new_scene_requested_surfaces_absent_in_reference':sorted((requests&keys[0])-keys[1])}
 (o/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'step':step,'unchanged_surfaces':len(unchanged)}),flush=True)
 del packs,xyz,surfs,ox,os;gc.collect()
req(all(sha(p)==v for p,v in pins.items()),'input changed')
r['complete']=True;(o/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')

from pathlib import Path
import json,hashlib,sys,gc
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back-forearm-native-014-018-back412-recovery-16s'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
out=D/'all-back-capture-audit-001';out.mkdir()
decl=json.loads((D/'baseline/run-declaration.json').read_text())
ex=json.loads((D/'baseline/execution.json').read_text())
req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'native failed')
pins={str(p):sha(p) for p in (Path(__file__),Path(ca.__file__),Path(ci.__file__),D/'baseline/execution.json',D/'baseline/run-declaration.json')}
report={'complete':False,'anatomical_admission':False,'scope':'Exact binary32 nondegeneracy of all six back muscle surfaces in every retained native accepted capture; does not establish intersection clearance.','captures':{},'pins':pins}
for step in decl['capture_steps']:
 p=D/'baseline/native-run/accepted-geometry'/f'step-{step}.mrvpack'
 rp=p.with_suffix('.receipt.json')
 for q in (p,rp):pins[str(q)]=sha(q)
 r=json.loads(rp.read_text());req(r['physical_endpoint']=='accepted' and r['pack_file_sha256']==sha(p),'invalid capture')
 xyz,surfs,_=ca._pack_surfaces(p,{(51005,s) for s in range(153,159)})
 checks={}
 for sid in range(153,159):
  f=np.asarray(surfs[(51005,sid)]['faces'],dtype=np.int64)
  ids,inv=np.unique(f,return_inverse=True)
  pts=[ci.float32_point_lattice_key(tuple(float(v) for v in point)) for point in np.asarray(xyz)[ids]]
  records=ci._records(pts,inv.reshape((-1,3)).tolist())
  checks[str(sid)]={'faces':len(records),'exact_degenerate_faces':0}
 report['captures'][str(step)]=checks
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'step':step,'rows':checks}),flush=True)
 del xyz,surfs,records;gc.collect()
req(all(sha(p)==v for p,v in pins.items()),'input changed')
report['complete']=True
(out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

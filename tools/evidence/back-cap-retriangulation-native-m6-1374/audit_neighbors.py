from pathlib import Path
import json,hashlib,sys,gc
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back-forearm-native-014-016-back412-right177';B=A/'back-forearm-native-014-013-right177'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,passive_attachment_composition as pc
def req(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(x,f):
 ids=np.unique(f);ps=[ci.float32_point_lattice_key(tuple(float(v) for v in p)) for p in x[ids]]
 return ci._records(ps,np.searchsorted(ids,f).tolist())
sources=[]
for scene in (B,D):
 t=pc._read_nhtiss4(scene/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
 sources.append({int(r[6]):np.asarray(pc._row_slices(t,r)['local_faces']) for r in t['records'] if int(r[6]) in (157,158)})
changed={s:np.flatnonzero(np.any(sources[0][s]!=sources[1][s],axis=1)) for s in (157,158)}
inv=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
keys,*_=ca._load_target_inventory(inv);keys=set(keys)|{(51005,s) for s in range(151,165)}
out=D/'changed-face-neighbor-audit-001';out.mkdir();result={'complete':False,'scope':'Changed cap faces against all other inspected anatomical surfaces, exact predicates after Float32 AABB filtering. Same-row self checks are in the main audit.','captures':{},'pins':{str(Path(__file__)):sha(__file__)}}
for step in (0,8000):
 armres=[]
 for index,scene in enumerate((B,D)):
  p=scene/'baseline/native-run/accepted-geometry'/f'step-{step}.mrvpack';result['pins'][str(p)]=sha(p)
  xyz,surfs,_=ca._pack_surfaces(p,keys);hits={}
  for sid in (157,158):
   sf=np.asarray(surfs[(51005,sid)]['faces'])[changed[sid]];tris=xyz[sf];lo=tris.min(axis=1);hi=tris.max(axis=1)
   sr=records(xyz,sf);rr={}
   for key,s in surfs.items():
    if key==(51005,sid):continue
    tf=np.asarray(s['faces']);tri=xyz[tf];tl=tri.min(axis=1);th=tri.max(axis=1)
    mask=np.zeros(len(tf),dtype=bool)
    for l,h in zip(lo,hi):mask|=np.all(th>=l,axis=1)&np.all(tl<=h,axis=1)
    ids=np.flatnonzero(mask)
    if not len(ids):continue
    audit=ci._audit_pair(sr,records(xyz,tf[ids]),same_surface=False)
    if audit['count']:rr[str(key)]={'count':int(audit['count']),'pairs':[[int(changed[sid][i]),int(ids[j])] for i,j in audit['triangle_pairs']]}
   hits[str(sid)]=rr
  armres.append(hits);del xyz,surfs;gc.collect()
 comparisons={}
 for sid in ('157','158'):
  old,new=armres[0][sid],armres[1][sid]
  addedtargets=sorted(set(new)-set(old))
  comparisons[sid]={'baseline_targets':old,'candidate_targets':new,'new_target_surfaces':addedtargets,'baseline_raw_pair_count':sum(v['count'] for v in old.values()),'candidate_raw_pair_count':sum(v['count'] for v in new.values())}
 result['captures'][str(step)]=comparisons
 (out/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'step':step,'comparisons':{s:{k:v for k,v in d.items() if k not in ('baseline_targets','candidate_targets')} for s,d in comparisons.items()}}),flush=True)
req(all(sha(p)==v for p,v in result['pins'].items()),'inputs changed')
result['complete']=True;(out/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


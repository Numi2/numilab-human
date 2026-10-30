"""Offline, positive degree1 source-cell quadrature for centroid-proxy outliers.
No source or child geometry changes; no additional physical state."""
import csv,hashlib,json,sys,struct,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
D=Path(__file__).resolve().parent
P=D.parent.parent;H=P/'fine-physical-volume-706'
sys.path.insert(0,str(H));from batched_reference_tet_projection_149_root import project_pairs
S=P/'native-physical-parent-mass-330/native-source-identity-control-003'
T=S/'compiled-parent-cell-mass.csv'
out=D/'source-quadrature-diagnostic-153-root';out.mkdir(exist_ok=False)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
start=time.monotonic()
manifest=json.loads((D/'preparation-manifest.json').read_text())
assert all(sha(Path(p))==v['sha256'] for p,v in manifest['inputs'].items())
nodes=read(T.with_name(T.name+'.nodes.csv'));tetrows=read(T.with_name(T.name+'.tets.csv'));labs=read(T.with_name(T.name+'.labels.csv'))
px=np.asarray([[float(r['rest_'+a+'_m']) for a in 'xyz'] for r in nodes])
pt=np.asarray([[int(r['parent_node'+str(k)]) for k in range(4)] for r in tetrows])
global_rows=np.asarray([i for i,r in enumerate(tetrows) if int(r['material_index'])==16],dtype=np.uint32)
inverses=np.linalg.inv(np.transpose(px[pt[:,1:]]-px[pt[:,0,None]],(0,2,1)))
raw=np.loadtxt(H/'combined-child-handoff-142/fat.node',skiprows=1,usecols=(0,1,2,3))
ids=raw[:,0].astype(np.int64);cx=raw[:,1:].astype(np.float32).astype(np.float64)
ele=np.loadtxt(H/'combined-child-handoff-142/fat.ele',skiprows=1)
ct=np.searchsorted(ids,ele[:,1:5].astype(np.int64));assert np.array_equal(ids[ct],ele[:,1:5].astype(np.int64))
cc=cx[ct].mean(axis=1)
with np.load(H/'connected-region-mass-support-143-root/fat-connected-region-support.npz',allow_pickle=False) as z:
 ao=z['cell_admissible_region_offsets'];ai=z['cell_admissible_region_ids']
census=json.loads((D/'all-parent-reachability-census.json').read_text())
assert census['parents_without_any_admissible_child_within_bound']==4
R=.019353

from scipy.spatial import cKDTree
alltree=cKDTree(cc);report=[]
for w in census['witnesses']:
 parent=w['fat_parent_row'];g=int(global_rows[parent]);reg=int(labs[g]['anatomical_region_id'])
 pos=np.flatnonzero(ai==reg);allowed=np.unique(np.searchsorted(ao,pos,side='right')-1)
 v=px[pt[g]];centroid=v.mean(axis=0);row={'parent':parent,'global_parent':g,'region':reg,'trials':[],'source_vertices_m':v.tolist()}
 for label,domain in [('authenticated_onehop',allowed),('all_Fat_diagnostic_only',np.arange(len(cc)))]:
  ci=domain[np.all((cc[domain]>=v.min(axis=0)-R)&(cc[domain]<=v.max(axis=0)+R),axis=1)]
  res=project_pairs(cc[ci],np.full(len(ci),g,dtype=np.int64),px,pt,inverses)
  tree=cKDTree(cc[domain]);dd,jj=tree.query(np.vstack([v,centroid]))
  row[label+'_nearest_distances_source_vertices_then_centroid_m']=dd.tolist()
  for alpha in (0.,.05,.1,.2):
   bs=(1-alpha)*res['bary']+alpha*.25
   ds=np.linalg.norm(bs@v-cc[ci],axis=1);take=ds<=R;bb=bs[take]
   if not len(bb):r={'success':False,'message':'no eligible candidate'}
   else:
    lp=linprog(np.square(ds[take]),A_eq=bb.T,b_eq=np.full(4,.25),bounds=(0,None),method='highs')
    r={'success':bool(lp.success),'message':lp.message}
    if lp.success:
     keep=np.flatnonzero(lp.x>1e-12);cw=lp.x[keep];cb=bb[keep]
     cw=np.linalg.lstsq(cb.T,np.full(4,.25),rcond=None)[0]
     r.update(weights=cw.tolist(),barycentric=cb.tolist(),target_rows=ci[take][keep].tolist(),target_distances_m=ds[take][keep].tolist(),moment_error=float(np.max(np.abs(cw@cb-.25))),min_weight=float(cw.min()),min_barycentric=float(cb.min()))
   row['trials'].append({'domain':label,'alpha':alpha,'bbox_candidates':len(ci),'eligible_carrier_sites':len(bb),**r})
 report.append(row)
result={'status':'SOURCE_ONLY_FAT_QUADRATURE_CONSTRUCTION_COUNTERFACTUALS','physical_admission':False,'native_steps':0,'all_Fat_domains_are_diagnostics_not_admitted':True,'rows':report,'elapsed_s':time.monotonic()-start}
(out/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'elapsed_s':result['elapsed_s'],'parents':[{'parent':r['parent'],'domains':[{'domain':t['domain'],'alpha':t['alpha'],'eligible':t['eligible_carrier_sites'],'success':t['success']} for t in r['trials']]} for r in report]},indent=2))

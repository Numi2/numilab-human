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
out=D/'source-quadrature-152-root';out.mkdir(exist_ok=False)
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
carriers={};diagnostics=[]
for w in census['witnesses']:
 parent=w['fat_parent_row'];g=int(global_rows[parent]);assert g==w['parent_tet_row']
 reg=int(labs[g]['anatomical_region_id']);positions=np.flatnonzero(ai==reg)
 allowed=np.unique(np.searchsorted(ao,positions,side='right')-1)
 v=px[pt[g]];centroid=v.mean(axis=0)
 ci=allowed[np.all((cc[allowed]>=v.min(axis=0)-R)&(cc[allowed]<=v.max(axis=0)+R),axis=1)]
 result=project_pairs(cc[ci],np.full(len(ci),g,dtype=np.int64),px,pt,inverses)
 # Move every projected site20% toward cell centroid: strict interior min bary0.05.
 bary=.8*result['bary']+.05
 points=bary@v
 distances=np.linalg.norm(points-cc[ci],axis=1)
 eligible=distances<=R
 ci=ci[eligible];bary=bary[eligible];distances=distances[eligible]
 if not len(ci):raise ValueError('no eligible carrier sites parent'+str(parent))
 lp=linprog(distances**2,A_eq=bary.T,b_eq=np.full(4,.25),bounds=(0,None),method='highs')
 if not lp.success:raise ValueError('no positive firstmoment quadrature parent'+str(parent)+':'+lp.message)
 keep=np.flatnonzero(lp.x>1e-12);weights=lp.x[keep];bs=bary[keep]
 # Re-solve the selected barycentric equality rather than preserve LP feasibility tolerance.
 weights=np.linalg.lstsq(bs.T,np.full(4,.25),rcond=None)[0]
 if not(np.all(weights>0) and np.max(np.abs(weights@bs-.25))<1e-12 and abs(weights.sum()-1)<1e-12):raise ValueError('quadrature closure')
 pts=bs@v;ds=np.linalg.norm(pts-cc[ci[keep]],axis=1)
 assert np.all(ds<=R) and np.min(bs)>=.05-1e-15
 carriers[parent]=[(float(we),bb.tolist(),int(ch),float(dd)) for we,bb,ch,dd in zip(weights,bs,ci[keep],ds)]
 diagnostics.append({'filtered_parent_row':parent,'global_source330_tet_row':g,'material':16,'region':reg,'stable_source_face_id':int(labs[g]['stable_source_face_id']),'original_centroid_nearest_admissible_child_distance_m':w['nearest_distance_m'],'candidate_child_centroids':len(allowed),'bbox_candidates':len(eligible),'eligible_projected_sites':len(ci),'degree':1,'strict_interior_contraction':.2,'carrier_count':len(keep),'carrier_target_child_rows':ci[keep].tolist(),'carrier_target_distance_m':ds.tolist(),'carrier_weights':weights.tolist(),'carrier_barycentric':bs.tolist(),'mass_fraction_sum_error':float(abs(weights.sum()-1)),'maximum_barycentric_first_moment_error':float(np.max(np.abs(weights@bs-.25))),'world_first_moment_error_m':float(np.linalg.norm(weights@pts-centroid)),'minimum_barycentric':float(bs.min())})
offsets=[0];weights=[];bary=[];targets=[];distances=[]
for i in range(len(global_rows)):
 for we,bb,ch,dd in carriers.get(i,[]):
  weights.append(we);bary.append(bb);targets.append(ch);distances.append(dd)
 offsets.append(len(weights))
arrays={'source_parent_tet_rows':np.arange(len(global_rows),dtype=np.uint32),'global_source330_tet_rows':global_rows,'parent_offsets':np.asarray(offsets,dtype=np.uint32),'carrier_weights':np.asarray(weights,dtype=np.float64),'carrier_barycentric':np.asarray(bary,dtype=np.float64),'carrier_target_child_rows':np.asarray(targets,dtype=np.uint32),'carrier_target_distance_m':np.asarray(distances,dtype=np.float64)}
np.savez_compressed(out/'source-quadrature.npz',**arrays)
with open(out/'source-quadrature.bin','wb') as f:
 f.write(b'FSSCARR01');f.write(struct.pack('<QQ',len(global_rows),len(weights)))
 for k in ('source_parent_tet_rows','global_source330_tet_rows','parent_offsets','carrier_weights','carrier_barycentric','carrier_target_child_rows','carrier_target_distance_m'):
  a=arrays[k];f.write(a.astype('<f8' if a.dtype.kind=='f' else '<u4').tobytes())
pins={p:sha(Path(p)) for p in manifest['inputs']}
for p in (T.with_name(T.name+'.nodes.csv'),T.with_name(T.name+'.tets.csv'),D/'semantic-support.bin',D/'all-parent-reachability-census.json',H/'connected-region-mass-support-143-root/independent-verification.json',Path(__file__),H/'batched_reference_tet_projection_149_root.py'):
 pins[str(p)]=sha(p)
receipt={'schema':'numi.human.source-tetra-centroid-carrier-quadrature.v1','status':'PASS_SOURCE_ONLY_POSITIVE_FIRST_MOMENT_QUADRATURE','role':'Fat','physical_admission':False,'native_steps':0,'counts':{'source_parents':len(global_rows),'fallback_parents':len(carriers),'carriers':len(weights)},'coordinate_domain':'source330 referenceWorldPoints = nodes.csv rest_xyz doubles; target child positions widened from exact Float32','method':'Positive degree1 quadrature of each elongated original source tetra. Sumweights1 and weightedbarycentricquartervector preserve original uniform source mass, volume and firstmoment. Each strictly interior carrier is within unchanged19.353mm of an actualsame-material143-admissiblechildcentroid. Originalsourcefiles/geometry/labels/massunchanged. No secondmoment exactness asserted. No flow or nativeadmission asserted.','maximum_transport_distance_m':R,'inputs':pins,'outlier_rows':diagnostics,'outputs':{str(p):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in (out/'source-quadrature.npz',out/'source-quadrature.bin')},'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':receipt['status'],'counts':receipt['counts'],'rows':diagnostics,'elapsed_s':receipt['elapsed_s']},indent=2))

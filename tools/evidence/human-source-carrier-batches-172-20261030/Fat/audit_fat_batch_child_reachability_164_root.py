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
out=D/'child-reachability-164-root';out.mkdir(exist_ok=False)
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
Q=D/'source-quadrature-162-root/source-quadrature.npz'
with np.load(Q,allow_pickle=False) as z:qs={k:z[k] for k in z.files}
positions=[];orig=[];regs=[]
for p,g in enumerate(global_rows):
 a,b=int(qs['parent_offsets'][p]),int(qs['parent_offsets'][p+1])
 xyz=px[pt[g]]
 if a==b:pp=[xyz.mean(axis=0)]
 else:pp=qs['carrier_barycentric'][a:b]@xyz
 positions.extend(pp);orig.extend([p]*len(pp));regs.extend([int(labs[g]['anatomical_region_id'])]*len(pp))
positions=np.asarray(positions);orig=np.asarray(orig);regs=np.asarray(regs)
cell_for_slot=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(ao))
order=np.argsort(ai,kind='stable');ordered_regions=ai[order];child_sorted=cell_for_slot[order]
unique,starts,counts=np.unique(ordered_regions,return_index=True,return_counts=True)
nearest=np.full(len(cc),np.inf);nearest_expanded=np.full(len(cc),-1,dtype=np.int32)
for reg,st,count in zip(unique,starts,counts):
 ps=np.flatnonzero(regs==reg);children=np.unique(child_sorted[st:st+count])
 if not len(ps):continue
 dist,ii=cKDTree(positions[ps]).query(cc[children]);take=dist<nearest[children]
 nearest[children[take]]=dist[take];nearest_expanded[children[take]]=ps[ii[take]]
holes=np.flatnonzero(nearest>R)
with np.load(H/'fat-current-semantic-support-141-root/fat-current-semantic-support.npz',allow_pickle=False) as z:
 cell_off=z['cell_group_offsets'];group_poff=z['cell_group_parent_tet_offsets'];source_parents=z['cell_group_parent_tet_rows'];group_ids=z['cell_group_ids'];tuples=z['group_tuples']
pools=[];poff=[0];pair_children=[]
for c in holes:
 a,b=int(cell_off[c]),int(cell_off[c+1])
 pool=np.unique(source_parents[group_poff[a]:group_poff[b]])
 pools.extend(pool);pair_children.extend([c]*len(pool));poff.append(len(pools))
pools=np.asarray(pools,dtype=np.int64);pair_children=np.asarray(pair_children,dtype=np.int64)
projection=project_pairs(cc[pair_children],pools,px,pt,inverses)
np.savez_compressed(out/'uncovered-child-p1-source-pools.npz',child_rows=holes,nearest_expanded_parent_rows=nearest_expanded[holes],nearest_expanded_distance_m=nearest[holes],parent_pool_offsets=np.asarray(poff,dtype=np.int64),parent_pool_global_rows=pools,parent_pool_child_rows=pair_children,parent_pool_projection_distance_m=projection['distance'],parent_pool_projected_points_m=projection['projected'],parent_pool_projection_barycentric=projection['bary'],child_centroids_m=cc[holes])
witnesses=[]
for index,c in enumerate(holes):
 a,b=poff[index:index+2];q=a+int(np.argmin(projection['distance'][a:b]));near=int(nearest_expanded[c])
 witnesses.append({'child_row':int(c),'nearest_expanded_parent_row':near,'nearest_original_fat_parent_row':int(orig[near]),'nearest_source_global_row':int(global_rows[orig[near]]),'distance_m':float(nearest[c]),'child_centroid_m':cc[c].tolist(),'admissible_regions':ai[ao[c]:ao[c+1]].astype(int).tolist(),'p1_pool_global_parent_rows':pools[a:b].tolist(),'closest_p1_support_global_parent_row':int(pools[q]),'closest_p1_support_distance_m':float(projection['distance'][q])})
r={'status':'SOURCE_ONLY_COMPLETE_FAT_CHILD_REACHABILITY_CENSUS','physical_admission':False,'native_steps':0,'original_source_parents':len(global_rows),'expanded_source_parents':len(positions),'child_cells':len(cc),'uncovered_child_count':len(holes),'maximum_nearest_admissible_distance_m':float(nearest.max()),'maximum_distance_m':R,'p1_parent_pairs':len(pools),'p1_pool_parent_count':len(np.unique(pools)),'witnesses':witnesses,'elapsed_s':time.monotonic()-start}
(out/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps({**{k:v for k,v in r.items() if k!='witnesses'},'first_witnesses':witnesses[:8]},indent=2))

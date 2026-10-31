from pathlib import Path
import csv,hashlib,json,time,resource
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'
H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
OUT=RUN/'attempt-012-centroid-partial-batch'
OUT.mkdir(parents=True,exist_ok=True)
if any(x.name!='source-freeze' for x in OUT.iterdir()):raise RuntimeError('output directory contains prior results')
FREEZE=OUT/'source-freeze'; FREEZE.mkdir(exist_ok=True)
GRAPH=RUN/'attempt-005/lp-input-current406.npz'
GREPORT=RUN/'attempt-005/lp-report.json'
LEDGER=S/'fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/rebased-held-flow-ledger.npz'
PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
SRCROOT=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
TARG=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
TARGREP=H/'current-D309-I399-target-406-root/target397/report.json'
SUPPORT=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
SUPPORTREP=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
RADIUS=0.019353
LOW=850.0
HIGH=950.0
MAX_FLOWS=15000
MAX_PER_PARENT=5
T0=time.time()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def writej(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def log(x):print(f'{time.time()-T0:.2f}s {x}',flush=True)
greport=json.loads(GREPORT.read_text())
if greport.get('status')!='CANDIDATE_GRAPH_INCOMPLETE_SOURCE_ONLY' or greport.get('physical_admission') is not False:raise RuntimeError('unexpected source graph status')
with PARENTCSV.open(newline='') as f: rows=list(csv.DictReader(f))
if len(rows)!=14316 or [int(r['parent_row']) for r in rows]!=list(range(14316)):raise RuntimeError('role parent CSV domain mismatch')
parent_global=np.asarray([int(r['global_source330_row']) for r in rows],dtype=np.int64)
parent_material=np.asarray([int(r['material']) for r in rows],dtype=np.int32)
parent_donor=np.asarray([int(r['geometric_donor']) for r in rows],dtype=np.int32)
parent_region=np.asarray([int(r['region']) for r in rows],dtype=np.int32)
parent_mass=np.asarray([float(r['mass_kg']) for r in rows],dtype=np.float64)
parent_centroid=np.asarray([[float(r[f'source_centroid_{a}_m']) for a in 'xyz'] for r in rows],dtype=np.float64)
if not np.array_equal(parent_global,np.arange(14316,28632,dtype=np.int64)):raise RuntimeError('role-to-source330 crosswalk mismatch')
with np.load(GRAPH,allow_pickle=False) as z:
 site_child=np.asarray(z['child_rows'],dtype=np.int32)
 site_parent=np.asarray(z['parent_role_rows'],dtype=np.int32)
 site_global=np.asarray(z['parent_global_source330_rows'],dtype=np.int64)
 bary=np.asarray(z['barycentric'],dtype=np.float64)
 graph_dist=np.asarray(z['distance_m'],dtype=np.float64)
 site_cap=np.asarray(z['column_mass_scale_kg'],dtype=np.float64)
 q0=np.asarray(z['residual_moments_kg'],dtype=np.float64)
 low0=np.asarray(z['child_low_kg'],dtype=np.float64)
 high0=np.asarray(z['child_high_kg'],dtype=np.float64)
 vol_hybrid=np.asarray(z['child_volume_hybrid_m3'],dtype=np.float64)
 held_child=np.asarray(z['held_current_child_mass_kg'],dtype=np.float64)
 held_scale=np.asarray(z['held_parent_scale'],dtype=np.float64)
with np.load(LEDGER,allow_pickle=False) as z:
 held_corner=np.asarray(z['parent_corner_mass_after_kg'],dtype=np.float64)
 ledger_parent_mass=np.asarray(z['parent_source_mass_kg'],dtype=np.float64)
 ledger_cell_mass=np.asarray(z['current_cell_mass_kg'],dtype=np.float64)
 flow_retained=np.asarray(z['flow_retained'],dtype=np.uint8)
 flow_parent=np.asarray(z['flow_parent_role_rows'],dtype=np.int32)
 flow_child=np.asarray(z['flow_current_child_rows'],dtype=np.int64)
 flow_mass=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 flow_bary=np.asarray(z['flow_source_barycentric'],dtype=np.float64)
 flow_point=np.asarray(z['flow_source_point_world_m'],dtype=np.float64)
 flow_distance=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64)
if not np.array_equal(parent_mass,ledger_parent_mass):raise RuntimeError('source mass CSV differs from held ledger')
corner_tol=5e-13*(parent_mass[:,None]*0.25)
ret=flow_retained.astype(bool)
if np.any((flow_parent[ret]<0)|(flow_parent[ret]>=len(parent_mass))):raise RuntimeError('retained held parent row out of range')
if np.any((flow_child[ret]<0)|(flow_child[ret]>=len(held_child))):raise RuntimeError('retained held child row out of range')
unscaled_held_child=np.bincount(flow_child[ret],weights=flow_mass[ret],minlength=len(ledger_cell_mass))
if not np.array_equal(unscaled_held_child,ledger_cell_mass):raise RuntimeError('unscaled retained ledger does not replay original current child mass')
scaled_held_mass=flow_mass[ret]*held_scale[flow_parent[ret]]
scaled_held_child=np.bincount(flow_child[ret],weights=scaled_held_mass,minlength=len(held_child))
scaled_held_corner=np.column_stack([np.bincount(flow_parent[ret],weights=scaled_held_mass*flow_bary[ret,k],minlength=len(parent_mass)) for k in range(4)])
if not np.array_equal(scaled_held_child,held_child):raise RuntimeError('graph held child mass does not replay from pinned parent scales')
if np.any(np.abs((parent_mass[:,None]*0.25-scaled_held_corner)-q0)>corner_tol):raise RuntimeError('source-corner residual is not authenticated by parent-scaled held ledger')
held_scale_release=float(flow_mass[ret].sum()-scaled_held_mass.sum())
mask=np.all(bary==0.25,axis=1)&(low0[site_child]>0)&(high0[site_child]>0)&(site_cap>0)&(np.min(q0[site_parent],axis=1)>0)
idx=np.flatnonzero(mask)
if not len(idx):raise RuntimeError('no centroid sites cover positive lower-bound rows')
initial_source_cap=4.0*np.min(q0[site_parent[idx]],axis=1)
initial_cap=np.minimum.reduce((site_cap[idx],low0[site_child[idx]],high0[site_child[idx]],initial_source_cap))
ok=initial_cap>0
idx=idx[ok]; initial_cap=initial_cap[ok]
order=np.lexsort((site_parent[idx],site_child[idx],graph_dist[idx],-initial_cap,-low0[site_child[idx]]))
idx=idx[order]
log(f'centroid candidate rows={len(idx)}; running bounded greedy batch')
q=q0.copy(); need=low0.copy(); headroom=high0.copy(); pcount=np.zeros(len(parent_mass),dtype=np.int32)
out_i=[]; out_mass=[]
for si in idx:
 if len(out_i)>=MAX_FLOWS:break
 p=int(site_parent[si]); c=int(site_child[si])
 if pcount[p]>=MAX_PER_PARENT or need[c]<=0.0 or headroom[c]<=0.0:continue
 d=min(float(site_cap[si]),float(need[c]),float(headroom[c]),4.0*float(np.min(q[p])))
 if d<=0.0:continue
 out_i.append(int(si)); out_mass.append(d)
 q[p]-=d*0.25; need[c]-=d; headroom[c]-=d; pcount[p]+=1
if not out_i:raise RuntimeError('bounded centroid greedy batch produced no positive flow')
site_rows=np.asarray(out_i,dtype=np.int64); masses=np.asarray(out_mass,dtype=np.float64)
parents=site_parent[site_rows].astype(np.int32); children=site_child[site_rows].astype(np.int32)
bary_out=bary[site_rows].copy(); distances=graph_dist[site_rows].copy(); globals_out=site_global[site_rows].copy()
points=parent_centroid[parents].copy()
delta_corner=np.zeros_like(q0)
for k in range(4):np.add.at(delta_corner[:,k],parents,masses*bary_out[:,k])
delta_child=np.bincount(children,weights=masses,minlength=len(low0))
if np.any(delta_corner>q0+corner_tol):raise RuntimeError('greedy batch overdraws a source corner')
if np.any(delta_child>high0+1e-15*np.maximum(high0,1e-300)):raise RuntimeError('greedy batch exceeds receiver upper headroom')
if np.any(delta_child>low0+1e-15*np.maximum(low0,1e-300)):raise RuntimeError('greedy batch exceeds receiver lower-target amount')
q_after=q0-delta_corner
need_after=low0-delta_child
head_after=high0-delta_child
npzout=OUT/'partial-centroid-additions.npz'
np.savez_compressed(npzout,source_candidate_site_rows=site_rows.astype(np.int64),parent_role_rows=parents,global_source330_rows=globals_out,child_current_rows=children,flow_mass_kg=masses,source_barycentric=bary_out,source_point_world_m=points,measured_distance_m=distances,parent_corner_delta_kg=delta_corner,parent_corner_residual_after_kg=q_after,child_mass_delta_kg=delta_child,child_lower_residual_after_kg=need_after,child_upper_headroom_after_kg=head_after)
ret_rows=np.flatnonzero(ret)
merged_parent=np.concatenate((flow_parent[ret].astype(np.int32),parents))
merged_global=np.concatenate((parent_global[flow_parent[ret]],globals_out.astype(np.int64)))
merged_child=np.concatenate((flow_child[ret].astype(np.int32),children))
merged_mass=np.concatenate((scaled_held_mass,masses))
merged_bary=np.concatenate((flow_bary[ret],bary_out))
merged_point=np.concatenate((flow_point[ret],points))
merged_distance=np.concatenate((flow_distance[ret],distances))
merged_origin=np.concatenate((np.zeros(len(ret_rows),dtype=np.uint8),np.ones(len(masses),dtype=np.uint8)))
merged_ledger_row=np.concatenate((ret_rows.astype(np.int64),np.full(len(masses),-1,dtype=np.int64)))
merged_child_mass=scaled_held_child+delta_child
merged_corner=np.column_stack([np.bincount(merged_parent,weights=merged_mass*merged_bary[:,k],minlength=len(parent_mass)) for k in range(4)])
if np.any(np.abs(merged_corner-(parent_mass[:,None]*.25-q_after))>corner_tol):raise RuntimeError('merged parent-corner ledger mismatch')
merged_child_replay=np.bincount(merged_child,weights=merged_mass,minlength=len(held_child))
merged_child_tol=64*np.finfo(np.float64).eps*np.maximum(merged_child_mass,np.finfo(np.float64).tiny)
if np.any(np.abs(merged_child_replay-merged_child_mass)>merged_child_tol):raise RuntimeError('merged child-mass ledger mismatch')
mergedout=OUT/'partial-merged-held-plus-additions.npz'
np.savez_compressed(mergedout,flow_parent_role_rows=merged_parent,flow_global_source330_rows=merged_global,flow_current_child_rows=merged_child,flow_mass_kg=merged_mass,flow_source_barycentric=merged_bary,flow_source_point_world_m=merged_point,flow_current_centroid_distance_m=merged_distance,flow_origin_code=merged_origin,base_ledger_flow_rows=merged_ledger_row,parent_source_mass_kg=parent_mass,parent_corner_residual_after_kg=q_after,child_current_mass_kg=merged_child_mass)

tz=np.load(TARG,allow_pickle=False)
tet=np.asarray(tz['tetrahedra'],dtype=np.int64); xyz=np.asarray(tz['points_world_m'],dtype=np.float64); child_mat=np.asarray(tz['material_indices'],dtype=np.int32)
sel_xyz=xyz[tet[children]]
child_centers=sel_xyz.mean(axis=1)
direct_dist=np.linalg.norm(points-child_centers,axis=1)
direct_det=np.einsum('ij,ij->i',sel_xyz[:,1]-sel_xyz[:,0],np.cross(sel_xyz[:,2]-sel_xyz[:,0],sel_xyz[:,3]-sel_xyz[:,0]))
if np.any(direct_dist>RADIUS):raise RuntimeError('recomputed centroid flow exceeds unchanged radius')
if np.any(np.abs(direct_dist-distances)>5e-12):raise RuntimeError('candidate graph centroid distances fail direct recomputation')
if np.any(child_mat[children]!=parent_material[parents]):raise RuntimeError('selected centroid flow changes material')
sz=np.load(SUPPORT,allow_pickle=False)
off=np.asarray(sz['cell_admissible_region_offsets'],dtype=np.int64); ids=np.asarray(sz['cell_admissible_region_ids'],dtype=np.int32)
for p,c in zip(parents,children):
 lo=int(off[c]); hi=int(off[c+1])
 if not np.any(ids[lo:hi]==parent_region[p]):raise RuntimeError(f'one-hop region mismatch parent={int(p)} child={int(c)} region={int(parent_region[p])}')
if not np.array_equal(globals_out,parent_global[parents]):raise RuntimeError('selected source330 global rows mismatch')
if not np.all(bary_out==.25):raise RuntimeError('non-centroid site entered centroid-only proposal')
uniq_child=np.unique(children)
direct_unique=np.zeros(len(uniq_child),dtype=np.float64)
for j,c in enumerate(uniq_child):
 x=xyz[tet[c]]
 direct_unique[j]=abs(float(np.linalg.det(x[1:]-x[0])))/6.0
 if not np.isfinite(direct_unique[j]) or direct_unique[j]<=0:raise RuntimeError('nonpositive direct receiver volume')
if not np.allclose(direct_unique,vol_hybrid[uniq_child],rtol=2e-7,atol=0):raise RuntimeError('hybrid child volume parity outside declared relative audit tolerance')
after_mass=scaled_held_child+delta_child
after_density=after_mass/vol_hybrid
if np.any(after_density>HIGH+5e-10):raise RuntimeError('receiver density exceeds 950 kg/m3')
corner_after=scaled_held_corner+delta_corner
if np.any(np.abs(corner_after-parent_mass[:,None]*0.25)>corner_tol+np.abs(q_after)):raise RuntimeError('partial corner ledger leaves target accounting inconsistent')
inputs=[GRAPH,GREPORT,LEDGER,PARENTCSV,SRCROOT/'compiled-parent-cell-mass.csv.nodes.csv',SRCROOT/'compiled-parent-cell-mass.csv.tets.csv',TARG,TARGREP,SUPPORT,SUPPORTREP,Path(__file__)]
report={'schema':'numi.human.current406-fat-centroid-site-partial-allocation.v1','status':'PASS_SOURCE_ONLY_PARTIAL_CENTROID_SITE_ALLOCATION','physical_admission':False,'native_steps':0,'mass_transport':'PARTIAL_SOURCE_ONLY_NOT_COMPLETE_ALLOCATION','accepted_complete_allocation':False,'cost_optimality_certified':False,'scope':'Bounded greedy partial transfer over already pinned centroid candidate sites. All current source/child constraints are checked for these additions only; remaining source residual and receiver lower deficits are retained. Candidate graph is incomplete and this is not global allocation feasibility or physical admission.','inputs':{str(p):sha(p) for p in inputs},'policy':{'site_type':'exact barycentric centroid [0.25,0.25,0.25,0.25] only','hard_radius_m':RADIUS,'material':'source and current recipient material must match','region':'source region must be listed in verified current 143 one-hop admissible regions','parent_corner_usage':'each flow delta consumes exactly one quarter from each residual parent corner after the already recorded per-parent held-flow scale','receiver_bounds':'addition capped by current low deficit and current 950 kg/m3 headroom','maximum_flow_rows':MAX_FLOWS,'maximum_flows_per_parent':MAX_PER_PARENT,'selection_order':'descending child lower deficit, then descending initial feasible chunk, ascending candidate distance, parent row, child row','no_mass_clipping':True},'held_scale':{'parents_scaled':int(np.count_nonzero(held_scale<1.0)),'minimum_scale':float(held_scale.min(initial=1.0)),'released_mass_kg':held_scale_release},'counts':{'current_parent_rows':int(len(parent_mass)),'current_child_rows':int(len(low0)),'candidate_centroid_rows_positive_initial_cap':int(len(idx)),'added_flow_rows':int(len(masses)),'distinct_parent_rows':int(len(np.unique(parents))),'distinct_child_rows':int(len(uniq_child)),'touched_children_filled_to_850_bound':int(np.count_nonzero(need_after[uniq_child]<=1e-14*np.maximum(low0[uniq_child],1e-300))),'touched_children_still_underfloor':int(np.count_nonzero(need_after[uniq_child]>0))},'mass_kg':{'added':float(masses.sum()),'source_residual_before':float(q0.sum()),'source_residual_after':float(q_after.sum()),'receiver_lower_deficit_before':float(low0.sum()),'receiver_lower_deficit_after':float(need_after.sum()),'receiver_upper_headroom_before':float(high0.sum()),'receiver_upper_headroom_after':float(head_after.sum()),'held_receiver_mass_before':float(held_child.sum()),'held_receiver_mass_after':float(after_mass.sum()),'unscaled_held_receiver_mass_before':float(ledger_cell_mass.sum()),'scaled_held_receiver_mass_before':float(scaled_held_child.sum()),'explicit_held_scale_release_kg':held_scale_release,'merged_retained_plus_additions_kg':float(merged_mass.sum()),'whole_source_parent_mass_unchanged':float(parent_mass.sum())},'checks':{'max_parent_corner_delta_over_q_kg':float(np.max(delta_corner-q0,initial=0.0)),'max_receiver_addition_over_high_kg':float(np.max(delta_child-high0,initial=0.0)),'max_receiver_addition_over_low_kg':float(np.max(delta_child-low0,initial=0.0)),'max_distance_m':float(np.max(direct_dist,initial=0.0)),'max_graph_vs_direct_distance_error_m':float(np.max(np.abs(direct_dist-distances),initial=0.0)),'min_direct_receiver_volume_m3':float(np.min(direct_unique,initial=np.inf)),'max_hybrid_vs_direct_volume_relative_error':float(np.max(np.abs(vol_hybrid[uniq_child]-direct_unique)/direct_unique,initial=0.0)),'min_result_density_kg_m3_touched':float(np.min(after_density[uniq_child],initial=np.inf)),'max_result_density_kg_m3_touched':float(np.max(after_density[uniq_child],initial=0.0)),'parent_corner_max_abs_error_to_M4_kg':float(np.max(np.abs(corner_after-parent_mass[:,None]*.25),initial=0.0)),'independent_onehop_region_check':'PASS'},'outputs':{npzout.name:{'sha256':sha(npzout),'size_bytes':npzout.stat().st_size},mergedout.name:{'sha256':sha(mergedout),'size_bytes':mergedout.stat().st_size}},'elapsed_seconds':time.time()-T0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
writej(OUT/'producer-report.json',report)
print(json.dumps({k:report[k] for k in ['status','counts','mass_kg','checks','elapsed_seconds','peak_rss_bytes']},indent=2))
print('npz_sha256',sha(npzout),'report_sha256',sha(OUT/'producer-report.json'))

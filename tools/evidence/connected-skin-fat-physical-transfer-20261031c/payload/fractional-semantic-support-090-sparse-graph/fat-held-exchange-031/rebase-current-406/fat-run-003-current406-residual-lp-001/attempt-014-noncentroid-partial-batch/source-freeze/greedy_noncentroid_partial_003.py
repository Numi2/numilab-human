#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,time,resource
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'
H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
OUT=RUN/'attempt-014-noncentroid-partial-batch'
if any(x.name!='source-freeze' for x in OUT.iterdir()):raise RuntimeError('output directory contains prior results')
GRAPH=RUN/'attempt-005/lp-input-current406.npz'
GREPORT=RUN/'attempt-005/lp-report.json'
LEDGER=S/'fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/rebased-held-flow-ledger.npz'
PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
SRCROOT=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
NODECSV=SRCROOT/'compiled-parent-cell-mass.csv.nodes.csv'
TETCSV=SRCROOT/'compiled-parent-cell-mass.csv.tets.csv'
TARG=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
TARGREP=H/'current-D309-I399-target-406-root/target397/report.json'
SUPPORT=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
SUPPORTREP=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
BASEOUT=RUN/'attempt-012-centroid-partial-batch'
BASEADD=BASEOUT/'partial-centroid-additions.npz'
BASEMERGED=BASEOUT/'partial-merged-held-plus-additions.npz'
BASEREPORT=BASEOUT/'producer-report.json'
BASEVERIFY=BASEOUT/'independent-replay-002-corrected.json'
OUTMAX=20000
PER_PARENT=5
RADIUS=0.019353
LOW=850.0
HIGH=950.0
T0=time.time()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def log(msg):print(f'{time.time()-T0:.2f}s {msg}',flush=True)
pinned={str(GRAPH):'4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328',str(GREPORT):'d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767',str(LEDGER):'0ee4e4909740e0029a10cb0019a531652b90c882226a0212f6d688f0bbf405fc',str(TARG):'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',str(SUPPORT):'e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526',str(SUPPORTREP):'b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482',str(BASEADD):'be4f6f634b1fd0beb99265708df90163a1f2804167755d8a083d2e4ee0555d32',str(BASEMERGED):'112e669b9ae080a11d9be06eb569dc4cbc9604f23cf2493eba1931c120c4a059',str(BASEREPORT):'01946c3f29b74a3f22a0be83b86ecea7802d153c0c5ac0ab970ff868684913c7',str(BASEVERIFY):'535241a4aab0d9bb9a098c50503f010a0345bd0f1ba403ae196de47f6cc70c5e',str(NODECSV):'7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044',str(TETCSV):'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992'}
for p,h in pinned.items():
 if sha(Path(p))!=h:raise RuntimeError('input hash mismatch '+p)
if json.loads(BASEVERIFY.read_text()).get('status')!='PASS_INDEPENDENT_PARTIAL_CURRENT406_CENTROID_BATCH_REPLAY':raise RuntimeError('centroid base independent proof absent')
with PARENTCSV.open(newline='') as f: pr=list(csv.DictReader(f))
if len(pr)!=14316 or [int(r['parent_row']) for r in pr]!=list(range(14316)):raise RuntimeError('role parent domain mismatch')
pg=np.array([int(r['global_source330_row']) for r in pr],dtype=np.int64)
pmat=np.array([int(r['material']) for r in pr],dtype=np.int32)
pdon=np.array([int(r['geometric_donor']) for r in pr],dtype=np.int32)
preg=np.array([int(r['region']) for r in pr],dtype=np.int32)
pmass=np.array([float(r['mass_kg']) for r in pr],dtype=np.float64)
pcent=np.array([[float(r[f'source_centroid_{a}_m']) for a in 'xyz'] for r in pr],dtype=np.float64)
if not np.array_equal(pg,np.arange(14316,28632,dtype=np.int64)):raise RuntimeError('role parent/global source330 map mismatch')
log('loading source tetra/node coordinates for exact pinned barycentric site reconstruction')
node_rows=np.loadtxt(NODECSV,delimiter=',',skiprows=1,usecols=(0,5,6,7),dtype=np.float64)
tet_rows=np.loadtxt(TETCSV,delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6),dtype=np.float64)
if node_rows.ndim!=2 or tet_rows.ndim!=2 or not np.array_equal(node_rows[:,0].astype(np.int64),np.arange(len(node_rows),dtype=np.int64)):raise RuntimeError('source node row domain mismatch')
if not np.array_equal(tet_rows[:,0].astype(np.int64),np.arange(len(tet_rows),dtype=np.int64)):raise RuntimeError('source tet row domain mismatch')
if not np.array_equal(tet_rows[pg,1].astype(np.int32),pmat) or not np.array_equal(tet_rows[pg,2].astype(np.int32),pdon):raise RuntimeError('source parent material/donor crosswalk mismatch')
pnode=tet_rows[pg,3:7].astype(np.int64)
if np.any(pnode<0) or np.any(pnode>=len(node_rows)):raise RuntimeError('source tet node outside source node domain')
pverts=node_rows[pnode,1:4]
centroid_rebuild=pverts.mean(axis=1)
centroid_delta=np.abs(centroid_rebuild-pcent)
if not np.array_equal(centroid_rebuild,pcent):raise RuntimeError('rest-coordinate source tetra centroids differ from pinned source parent centroids')
with np.load(GRAPH,allow_pickle=False) as z:
 sp=np.asarray(z['parent_role_rows'],dtype=np.int32); sg=np.asarray(z['parent_global_source330_rows'],dtype=np.int64)
 sc=np.asarray(z['child_rows'],dtype=np.int32); sb=np.asarray(z['barycentric'],dtype=np.float64)
 sd=np.asarray(z['distance_m'],dtype=np.float64); scap=np.asarray(z['column_mass_scale_kg'],dtype=np.float64)
with np.load(BASEADD,allow_pickle=False) as z:
 q1=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 low1=np.asarray(z['child_lower_residual_after_kg'],dtype=np.float64)
 high1=np.asarray(z['child_upper_headroom_after_kg'],dtype=np.float64)
 base_add_rows=np.asarray(z['source_candidate_site_rows'],dtype=np.int64)
 base_add_mass=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 base_add_parent=np.asarray(z['parent_role_rows'],dtype=np.int32)
 base_add_child=np.asarray(z['child_current_rows'],dtype=np.int32)
 base_add_corner=np.asarray(z['parent_corner_delta_kg'],dtype=np.float64)
 base_add_child_delta=np.asarray(z['child_mass_delta_kg'],dtype=np.float64)
with np.load(BASEMERGED,allow_pickle=False) as z:
 base_p=np.asarray(z['flow_parent_role_rows'],dtype=np.int32); base_g=np.asarray(z['flow_global_source330_rows'],dtype=np.int64)
 base_c=np.asarray(z['flow_current_child_rows'],dtype=np.int32); base_m=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 base_b=np.asarray(z['flow_source_barycentric'],dtype=np.float64); base_x=np.asarray(z['flow_source_point_world_m'],dtype=np.float64)
 base_d=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64)
 base_origin=np.asarray(z['flow_origin_code'],dtype=np.uint8); base_ledger_row=np.asarray(z['base_ledger_flow_rows'],dtype=np.int64)
 base_child_mass=np.asarray(z['child_current_mass_kg'],dtype=np.float64)
 base_q=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
if not np.array_equal(base_q,q1) or len(base_p)!=754841+len(base_add_mass):raise RuntimeError('base merged candidate row/Q mismatch')
if np.any(~np.isfinite(sb)) or np.any(sb<0) or np.any(np.abs(sb.sum(axis=1)-1)>2e-14):raise RuntimeError('candidate barycentric domain invalid')
noncentroid=~np.all(sb==0.25,axis=1)
pos=sb>0
ratio=np.full_like(sb,np.inf)
np.divide(q1[sp],sb,out=ratio,where=pos)
src_cap=ratio.min(axis=1)
valid=(noncentroid & np.all(np.isfinite(sb),axis=1) & (src_cap>0) & np.isfinite(scap) & (scap>0) &
       (low1[sc]>0) & (high1[sc]>0) & (sg==pg[sp]) & (sd<=RADIUS))
cand=np.flatnonzero(valid)
initial=np.minimum.reduce((scap[cand],low1[sc[cand]],high1[sc[cand]],src_cap[cand]))
keep=initial>0
cand=cand[keep]; initial=initial[keep]
if len(cand)==0:raise RuntimeError('no pinned non-centroid sites retain positive feasible capacity')
order=np.lexsort((cand,sp[cand],sd[cand],-initial,-low1[sc[cand]]))
cand=cand[order]
log(f'non-centroid pinned candidate rows={len(cand)}; selected batch starts')
q=q1.copy(); need=low1.copy(); head=high1.copy(); per_parent=np.zeros(len(pmass),dtype=np.int32)
selected=[]; masses=[]
for r in cand:
 if len(selected)>=OUTMAX:break
 p=int(sp[r]); c=int(sc[r]); b=sb[r]
 if per_parent[p]>=PER_PARENT or need[c]<=0 or head[c]<=0:continue
 allowed=b>0
 if not np.any(allowed):continue
 curr_cap=float(np.min(q[p,allowed]/b[allowed]))
 d=min(float(scap[r]),float(need[c]),float(head[c]),curr_cap)
 if not np.isfinite(d) or d<=0:continue
 selected.append(int(r)); masses.append(d)
 q[p]-=d*b; need[c]-=d; head[c]-=d; per_parent[p]+=1
if not selected:raise RuntimeError('non-centroid batch selected no positive flow')
rows=np.asarray(selected,dtype=np.int64); dm=np.asarray(masses,dtype=np.float64)
ap=sp[rows].astype(np.int32); ag=sg[rows].astype(np.int64); ac=sc[rows].astype(np.int32)
ab=sb[rows].copy(); ad=sd[rows].copy()
ax=np.einsum('ij,ijk->ik',ab,pverts[ap])
dc=np.column_stack([np.bincount(ap,weights=dm*ab[:,k],minlength=len(pmass)) for k in range(4)])
dchild=np.bincount(ac,weights=dm,minlength=len(low1))
ctol=5e-13*(pmass[:,None]*.25)
if np.any(dc>q1+ctol):raise RuntimeError('selected additions overdraw source corner')
if np.any(dchild>low1+1e-15*np.maximum(low1,1e-300)):raise RuntimeError('selected additions exceed residual lower deficit')
if np.any(dchild>high1+1e-15*np.maximum(high1,1e-300)):raise RuntimeError('selected additions exceed upper headroom')
if not np.allclose(q1-dc,q,rtol=0,atol=ctol):raise RuntimeError('incremental parent corner replay mismatch')
target=np.load(TARG,allow_pickle=False)
xyz=np.asarray(target['points_world_m'],dtype=np.float64); tet=np.asarray(target['tetrahedra'],dtype=np.int64); childmat=np.asarray(target['material_indices'],dtype=np.int32)
tx=xyz[tet[ac]]; childcent=tx.mean(axis=1); directdist=np.linalg.norm(ax-childcent,axis=1)
if np.any(directdist>RADIUS):raise RuntimeError('selected source bary point violates original radius')
if not np.allclose(directdist,ad,rtol=0,atol=5e-12):raise RuntimeError('graph/direct non-centroid distance mismatch')
if not np.array_equal(childmat[ac],pmat[ap]):raise RuntimeError('selected recipient material differs from source parent')
support=np.load(SUPPORT,allow_pickle=False); off=np.asarray(support['cell_admissible_region_offsets'],dtype=np.int64); ids=np.asarray(support['cell_admissible_region_ids'],dtype=np.int32)
for p,c in zip(ap,ac):
 if not np.any(ids[off[c]:off[c+1]]==preg[p]):raise RuntimeError('selected source region not allowed by current143 support')
vol=np.abs(np.einsum('ij,ij->i',tx[:,1]-tx[:,0],np.cross(tx[:,2]-tx[:,0],tx[:,3]-tx[:,0])))/6
with np.load(GRAPH,allow_pickle=False) as z: hvol=np.asarray(z['child_volume_hybrid_m3'],dtype=np.float64)
if np.any(vol<=0) or not np.allclose(vol,hvol[ac],rtol=2e-7,atol=0):raise RuntimeError('direct/hybrid child volume mismatch')
after_child=base_child_mass+dchild
after_density=after_child[ac]/vol
if np.any(after_density>HIGH+5e-10):raise RuntimeError('recipient density exceeds950')
merged_parent=np.concatenate((base_p,ap)); merged_global=np.concatenate((base_g,ag)); merged_child=np.concatenate((base_c,ac))
merged_mass=np.concatenate((base_m,dm)); merged_bary=np.concatenate((base_b,ab)); merged_point=np.concatenate((base_x,ax)); merged_dist=np.concatenate((base_d,directdist))
merged_origin=np.concatenate((base_origin,np.full(len(dm),2,dtype=np.uint8)))
merged_ledger=np.concatenate((base_ledger_row,np.full(len(dm),-1,dtype=np.int64)))
merged_child_replay=np.bincount(merged_child,weights=merged_mass,minlength=len(base_child_mass))
mtol=64*np.finfo(float).eps*np.maximum(after_child,np.finfo(float).tiny)
if np.any(np.abs(merged_child_replay-after_child)>mtol):raise RuntimeError('merged child mass mismatch')
merged_corner=np.column_stack([np.bincount(merged_parent,weights=merged_mass*merged_bary[:,k],minlength=len(pmass)) for k in range(4)])
if np.any(np.abs((pmass[:,None]*.25-merged_corner)-q)>ctol):raise RuntimeError('merged source corner residual mismatch')
if abs((merged_mass.sum()+q.sum())-pmass.sum())>5e-13*pmass.sum():raise RuntimeError('mass total does not close with remaining residual')
proposal=OUT/'partial-noncentroid-additions.npz'
np.savez_compressed(proposal,source_candidate_site_rows=rows,parent_role_rows=ap,global_source330_rows=ag,child_current_rows=ac,flow_mass_kg=dm,source_barycentric=ab,source_point_world_m=ax,measured_distance_m=directdist,parent_corner_delta_kg=dc,parent_corner_residual_after_kg=q,child_mass_delta_kg=dchild,child_lower_residual_after_kg=need,child_upper_headroom_after_kg=head)
mergedout=OUT/'partial-merged-held-centroid-noncentroid.npz'
np.savez_compressed(mergedout,flow_parent_role_rows=merged_parent,flow_global_source330_rows=merged_global,flow_current_child_rows=merged_child,flow_mass_kg=merged_mass,flow_source_barycentric=merged_bary,flow_source_point_world_m=merged_point,flow_current_centroid_distance_m=merged_dist,flow_origin_code=merged_origin,base_ledger_flow_rows=merged_ledger,parent_source_mass_kg=pmass,parent_corner_residual_after_kg=q,child_current_mass_kg=after_child)
unique,unique_first=np.unique(ac,return_index=True)
unique_density=after_child[unique]/vol[unique_first]
inputs=[GRAPH,GREPORT,LEDGER,PARENTCSV,NODECSV,TETCSV,TARG,TARGREP,SUPPORT,SUPPORTREP,BASEADD,BASEMERGED,BASEREPORT,BASEVERIFY,Path(__file__)]
report={'schema':'numi.human.current406-fat-noncentroid-site-partial-allocation.v1','status':'PASS_SOURCE_ONLY_PARTIAL_NONCENTROID_SITE_ALLOCATION','physical_admission':False,'native_steps':0,'complete_allocation':False,'scope':'Bounded greedy additions from existing pinned non-centroid barycentric candidate sites on the current406 Fat mesh. No sites are generated or coordinates moved. Source corner moments, current receiver bounds, material, 143 one-hop regions, and original radius remain enforced for additions. Residual source mass and lower-density cells remain; this is not complete allocation or physical admission.','inputs':{str(p):sha(p) for p in inputs},'policy':{'candidate_rows':'non-centroid rows already present in pinned attempt005 graph','source_point':'barycentric combination of four pinned source330 tet vertex coordinates','radius_m':RADIUS,'material':'source and receiver material equal','region':'authenticated current143 one-hop admissibility','source_cap':'min residual corner q_j/barycentric_j over strictly positive barycentric coordinates','receiver_caps':'both remaining lower deficit and upper headroom','max_added_rows':OUTMAX,'max_rows_per_parent':PER_PARENT,'priority':'larger current child lower deficit; then larger feasible per-site mass; then shorter distance; then stable parent/child/site ordering','mass_clipping':False},'counts':{'candidate_noncentroid_rows_positive_capacity':int(len(cand)),'added_flow_rows':int(len(dm)),'distinct_parent_rows':int(len(np.unique(ap))),'distinct_child_rows':int(len(unique)),'positive_barycentric_support_1':int(np.count_nonzero(np.sum(ab>0,axis=1)==1)),'positive_barycentric_support_2':int(np.count_nonzero(np.sum(ab>0,axis=1)==2)),'positive_barycentric_support_3':int(np.count_nonzero(np.sum(ab>0,axis=1)==3)),'positive_barycentric_support_4':int(np.count_nonzero(np.sum(ab>0,axis=1)==4)),'touched_children_still_below_850':int(np.count_nonzero(unique_density<LOW))},'mass_kg':{'added':float(dm.sum()),'residual_corner_mass_before':float(q1.sum()),'residual_corner_mass_after':float(q.sum()),'lower_density_deficit_before':float(low1.sum()),'lower_density_deficit_after':float(need.sum()),'upper_headroom_before':float(high1.sum()),'upper_headroom_after':float(head.sum()),'merged_transferred_mass':float(merged_mass.sum()),'source_parent_mass_total_unchanged':float(pmass.sum())},'checks':{'parent_role_global_source_map':'PASS','source_parent_tet_material_and_donor_crosswalk':'PASS','source_barycentric_sites_inside_tetrahedra':'PASS','actual_source_point_reconstruction':'PASS','direct_current_recipient_centroid_distance':'PASS','hard_radius':'PASS','material':'PASS','current143_onehop_region':'PASS','source_corner_residual_after':'PASS','receiver_lower_and_upper_caps':'PASS','upper_density_all_touched':'PASS','exact_volume_positive_and_hybrid_parity':'PASS','merged_child_and_corner_ledger':'PASS','source_total_plus_residual':'PASS','lower_density_complete':False,'physical_admission':False},'metrics':{'max_source_centroid_reconstruction_error_m':float(np.max(centroid_delta,initial=0.0)),'max_direct_distance_m':float(np.max(directdist,initial=0.0)),'max_graph_distance_error_m':float(np.max(np.abs(directdist-ad),initial=0.0)),'max_volume_relative_error':float(np.max(np.abs(vol-hvol[ac])/vol,initial=0.0)),'min_touched_density_kg_m3':float(np.min(unique_density,initial=np.inf)),'max_touched_density_kg_m3':float(np.max(unique_density,initial=0.0)),'max_corner_residual_replay_error_kg':float(np.max(np.abs((pmass[:,None]*.25-merged_corner)-q),initial=0.0))},'outputs':{proposal.name:{'sha256':sha(proposal),'size_bytes':proposal.stat().st_size},mergedout.name:{'sha256':sha(mergedout),'size_bytes':mergedout.stat().st_size}},'elapsed_seconds':time.time()-T0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
rp=OUT/'producer-report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'counts':report['counts'],'mass_kg':report['mass_kg'],'metrics':report['metrics'],'outputs':report['outputs'],'report_sha256':sha(rp)},indent=2))

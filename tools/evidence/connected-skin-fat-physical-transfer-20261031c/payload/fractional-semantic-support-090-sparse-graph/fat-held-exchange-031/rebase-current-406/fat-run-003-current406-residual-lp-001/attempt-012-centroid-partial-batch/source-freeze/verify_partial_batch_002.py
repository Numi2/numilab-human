
from pathlib import Path
import hashlib,json,csv,time,resource
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'
H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
OUT=RUN/'attempt-012-centroid-partial-batch'
GRAPH=RUN/'attempt-005/lp-input-current406.npz'
GREPORT=RUN/'attempt-005/lp-report.json'
LEDGER=S/'fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/rebased-held-flow-ledger.npz'
LEDGER_REPLAY=S/'fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/independent-ledger-replay-v3.json'
PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
SRCROOT=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
TARG=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
TARGREP=H/'current-D309-I399-target-406-root/target397/report.json'
SUPPORT=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
SUPPORTREP=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
PRODUCER=OUT/'producer-report.json'
ADD=OUT/'partial-centroid-additions.npz'
MERGED=OUT/'partial-merged-held-plus-additions.npz'
DEST=OUT/'independent-replay-001.json'
T0=time.time()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
expected={
 str(GRAPH):'4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328',
 str(GREPORT):'d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767',
 str(LEDGER):'0ee4e4909740e0029a10cb0019a531652b90c882226a0212f6d688f0bbf405fc',
 str(TARG):'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',
 str(SUPPORT):'e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526',
 str(SUPPORTREP):'b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482',
 str(PRODUCER):'01946c3f29b74a3f22a0be83b86ecea7802d153c0c5ac0ab970ff868684913c7',
 str(ADD):'be4f6f634b1fd0beb99265708df90163a1f2804167755d8a083d2e4ee0555d32',
 str(MERGED):'112e669b9ae080a11d9be06eb569dc4cbc9604f23cf2493eba1931c120c4a059'
}
for p,h in expected.items():
 if sha(Path(p))!=h:raise RuntimeError('input/output hash mismatch '+p)
ledger_replay=json.loads(LEDGER_REPLAY.read_text())
if ledger_replay.get('status')!='PASS_INDEPENDENT_CURRENT_HELD_LEDGER_REPLAY_SOURCE_ONLY' or not all(ledger_replay['checks'].values()):raise RuntimeError('base held ledger independent receipt not PASS')
with PARENTCSV.open(newline='') as f: pr=list(csv.DictReader(f))
if len(pr)!=14316 or [int(r['parent_row']) for r in pr]!=list(range(14316)):raise RuntimeError('parent row domain mismatch')
parent_global=np.array([int(r['global_source330_row']) for r in pr],dtype=np.int64)
parent_mat=np.array([int(r['material']) for r in pr],dtype=np.int32)
parent_region=np.array([int(r['region']) for r in pr],dtype=np.int32)
parent_mass=np.array([float(r['mass_kg']) for r in pr],dtype=np.float64)
parent_xyz=np.array([[float(r[f'source_centroid_{a}_m']) for a in 'xyz'] for r in pr],dtype=np.float64)
if not np.array_equal(parent_global,np.arange(14316,28632,dtype=np.int64)):raise RuntimeError('parent/source330 role mapping mismatch')
with np.load(GRAPH,allow_pickle=False) as z:
 sp=np.asarray(z['parent_role_rows'],dtype=np.int32); sg=np.asarray(z['parent_global_source330_rows'],dtype=np.int64)
 sc=np.asarray(z['child_rows'],dtype=np.int32); sb=np.asarray(z['barycentric'],dtype=np.float64)
 sd=np.asarray(z['distance_m'],dtype=np.float64); scap=np.asarray(z['column_mass_scale_kg'],dtype=np.float64)
 q0=np.asarray(z['residual_moments_kg'],dtype=np.float64); low=np.asarray(z['child_low_kg'],dtype=np.float64)
 high=np.asarray(z['child_high_kg'],dtype=np.float64); hybrid_vol=np.asarray(z['child_volume_hybrid_m3'],dtype=np.float64)
 held_child=np.asarray(z['held_current_child_mass_kg'],dtype=np.float64); scale=np.asarray(z['held_parent_scale'],dtype=np.float64)
with np.load(LEDGER,allow_pickle=False) as z:
 lp=np.asarray(z['flow_parent_role_rows'],dtype=np.int32); lc=np.asarray(z['flow_current_child_rows'],dtype=np.int64)
 lm=np.asarray(z['flow_mass_kg'],dtype=np.float64); lb=np.asarray(z['flow_source_barycentric'],dtype=np.float64)
 lx=np.asarray(z['flow_source_point_world_m'],dtype=np.float64); ld=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64)
 retained=np.asarray(z['flow_retained'],dtype=np.uint8).astype(bool)
 raw_child=np.asarray(z['current_cell_mass_kg'],dtype=np.float64)
 raw_pm=np.asarray(z['parent_source_mass_kg'],dtype=np.float64)
if not np.array_equal(parent_mass,raw_pm):raise RuntimeError('source mass domain mismatch')
ridx=np.flatnonzero(retained)
if np.any((lp[ridx]<0)|(lp[ridx]>=len(parent_mass))|(lc[ridx]<0)|(lc[ridx]>=len(held_child))):raise RuntimeError('retained row index outside domain')
raw_replay=np.bincount(lc[ridx],weights=lm[ridx],minlength=len(held_child))
if not np.array_equal(raw_replay,raw_child):raise RuntimeError('original retained ledger flow does not reproduce original child mass')
scaled_mass=lm[ridx]*scale[lp[ridx]]
scaled_replay=np.bincount(lc[ridx],weights=scaled_mass,minlength=len(held_child))
if not np.array_equal(scaled_replay,held_child):raise RuntimeError('scaled retained flow does not reproduce graph baseline')
scaled_corner=np.column_stack([np.bincount(lp[ridx],weights=scaled_mass*lb[ridx,k],minlength=len(parent_mass)) for k in range(4)])
ctol=5e-13*(parent_mass[:,None]*.25)
if np.any(np.abs(parent_mass[:,None]*.25-scaled_corner-q0)>ctol):raise RuntimeError('parent scale/residual source corner mismatch')
with np.load(ADD,allow_pickle=False) as z:
 site_rows=np.asarray(z['source_candidate_site_rows'],dtype=np.int64)
 add_p=np.asarray(z['parent_role_rows'],dtype=np.int32); add_g=np.asarray(z['global_source330_rows'],dtype=np.int64)
 add_c=np.asarray(z['child_current_rows'],dtype=np.int32); add_m=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 add_b=np.asarray(z['source_barycentric'],dtype=np.float64); add_x=np.asarray(z['source_point_world_m'],dtype=np.float64)
 add_d=np.asarray(z['measured_distance_m'],dtype=np.float64); dpc=np.asarray(z['parent_corner_delta_kg'],dtype=np.float64)
 dcc=np.asarray(z['child_mass_delta_kg'],dtype=np.float64); q_after=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 low_after=np.asarray(z['child_lower_residual_after_kg'],dtype=np.float64); high_after=np.asarray(z['child_upper_headroom_after_kg'],dtype=np.float64)
if len(site_rows)!=len(add_m) or len(np.unique(site_rows))!=len(site_rows):raise RuntimeError('candidate site row domain duplicate/mismatch')
if len(add_m)!=9240:raise RuntimeError('unexpected addition flow count')
if not np.array_equal(add_p,sp[site_rows]) or not np.array_equal(add_g,sg[site_rows]) or not np.array_equal(add_c,sc[site_rows]) or not np.array_equal(add_b,sb[site_rows]) or not np.array_equal(add_d,sd[site_rows]):raise RuntimeError('additions do not map exactly to pinned graph rows')
if not np.array_equal(add_g,parent_global[add_p]):raise RuntimeError('added global source330 row mismatch')
if not np.array_equal(add_x,parent_xyz[add_p]) or not np.all(add_b==.25):raise RuntimeError('added centroid position/barycentric mismatch')
if np.any(add_m<=0) or np.any(add_m>scap[site_rows]):raise RuntimeError('nonpositive or over-cap candidate flow')
if np.any((add_m*.25)[:,None]>q0[add_p]+ctol[add_p]):raise RuntimeError('added source corner exceeds residual')
indep_pc=np.column_stack([np.bincount(add_p,weights=add_m*add_b[:,k],minlength=len(parent_mass)) for k in range(4)])
if np.any(np.abs(indep_pc-dpc)>ctol):raise RuntimeError('source-corner delta not independently reaggregated')
add_child=np.bincount(add_c,weights=add_m,minlength=len(held_child))
if np.any(np.abs(add_child-dcc)>64*np.finfo(float).eps*np.maximum(dcc,np.finfo(float).tiny)):raise RuntimeError('added receiver flow not independently reaggregated')
if np.any(add_child>low+1e-15*np.maximum(low,1e-300)) or np.any(add_child>high+1e-15*np.maximum(high,1e-300)):raise RuntimeError('addition violates lower-target or upper-capacity amount')
if not np.allclose(q0-dpc,q_after,rtol=0,atol=ctol):raise RuntimeError('source residual output mismatch')
if np.any(np.abs(low-add_child-low_after)>64*np.finfo(float).eps*np.maximum(low,np.finfo(float).tiny)):raise RuntimeError('receiver low residual output mismatch')
if np.any(np.abs(high-add_child-high_after)>64*np.finfo(float).eps*np.maximum(high,np.finfo(float).tiny)):raise RuntimeError('receiver headroom output mismatch')
producer=json.loads(PRODUCER.read_text())
if producer.get('status')!='PASS_SOURCE_ONLY_PARTIAL_CENTROID_SITE_ALLOCATION' or producer.get('physical_admission') is not False or producer.get('native_steps')!=0:raise RuntimeError('producer source-only status invalid')
if producer.get('inputs',{}).get(str(Path(__file__).parent/'greedy_centroid_partial_v5.py'))!='75a5f3e8edc9429db95bc75bda8e73547cc28b86ae88e2fe48c8f95bfe710ca5':raise RuntimeError('producer does not pin v5 source')
held_release=float(lm[ridx].sum()-scaled_mass.sum())
if abs(float(producer['mass_kg']['explicit_held_scale_release_kg'])-held_release)>1e-18:raise RuntimeError('held scale release receipt mismatch')
with np.load(MERGED,allow_pickle=False) as z:
 mp=np.asarray(z['flow_parent_role_rows'],dtype=np.int32); mg=np.asarray(z['flow_global_source330_rows'],dtype=np.int64)
 mc=np.asarray(z['flow_current_child_rows'],dtype=np.int32); mm=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 mb=np.asarray(z['flow_source_barycentric'],dtype=np.float64); mx=np.asarray(z['flow_source_point_world_m'],dtype=np.float64)
 md=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64); origin=np.asarray(z['flow_origin_code'],dtype=np.uint8)
 base_row=np.asarray(z['base_ledger_flow_rows'],dtype=np.int64); merged_q=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 merged_cm=np.asarray(z['child_current_mass_kg'],dtype=np.float64); merged_pm=np.asarray(z['parent_source_mass_kg'],dtype=np.float64)
nr=len(ridx)
if not np.array_equal(merged_pm,parent_mass) or len(mm)!=nr+len(add_m):raise RuntimeError('merged proposal parent/flow size mismatch')
if not np.array_equal(mp[:nr],lp[ridx]) or not np.array_equal(mg[:nr],parent_global[lp[ridx]]) or not np.array_equal(mc[:nr],lc[ridx].astype(np.int32)) or not np.array_equal(mm[:nr],scaled_mass) or not np.array_equal(mb[:nr],lb[ridx]) or not np.array_equal(mx[:nr],lx[ridx]) or not np.array_equal(md[:nr],ld[ridx]):raise RuntimeError('merged retained flow prefix mismatch')
if not np.all(origin[:nr]==0) or not np.array_equal(base_row[:nr],ridx):raise RuntimeError('merged ledger provenance prefix mismatch')
if not np.array_equal(mp[nr:],add_p) or not np.array_equal(mg[nr:],add_g) or not np.array_equal(mc[nr:],add_c) or not np.array_equal(mm[nr:],add_m) or not np.array_equal(mb[nr:],add_b) or not np.array_equal(mx[nr:],add_x) or not np.array_equal(md[nr:],add_d):raise RuntimeError('merged added-flow suffix mismatch')
if not np.all(origin[nr:]==1) or not np.all(base_row[nr:]==-1):raise RuntimeError('merged new-flow provenance mismatch')
if not np.array_equal(merged_q,q_after):raise RuntimeError('merged residual corner output mismatch')
mcorner=np.column_stack([np.bincount(mp,weights=mm*mb[:,k],minlength=len(parent_mass)) for k in range(4)])
if np.any(np.abs(mcorner-(parent_mass[:,None]*.25-q_after))>ctol):raise RuntimeError('merged ledger fails M/4 residual replay')
mchild=np.bincount(mc,weights=mm,minlength=len(held_child))
mass_tol=64*np.finfo(float).eps*np.maximum(merged_cm,np.finfo(float).tiny)
if np.any(np.abs(mchild-merged_cm)>mass_tol) or not np.array_equal(merged_cm,scaled_replay+add_child):raise RuntimeError('merged child ledger reaggregation mismatch')
if np.any(mp<0) or np.any(mp>=len(parent_mass)) or not np.array_equal(mg,parent_global[mp]):raise RuntimeError('merged parent identity crosswalk mismatch')
if not np.all(np.isfinite(mm)) or np.any(mm<=0) or not np.all(np.isfinite(mb)) or not np.allclose(mb.sum(axis=1),1,rtol=0,atol=2e-15):raise RuntimeError('merged flow positivity/barycentric controls fail')
with np.load(TARG,allow_pickle=False) as z:
 xyz=np.asarray(z['points_world_m'],dtype=np.float64); tet=np.asarray(z['tetrahedra'],dtype=np.int64); mat=np.asarray(z['material_indices'],dtype=np.int32)
with np.load(SUPPORT,allow_pickle=False) as z:
 roff=np.asarray(z['cell_admissible_region_offsets'],dtype=np.int64); rids=np.asarray(z['cell_admissible_region_ids'],dtype=np.int32)
txyz=xyz[tet[add_c]]
cc=txyz.mean(axis=1)
direct_dist=np.linalg.norm(add_x-cc,axis=1)
if np.any(direct_dist>0.019353):raise RuntimeError('replayed centroid point violates original 19.353mm hard radius')
if not np.array_equal(direct_dist,add_d):raise RuntimeError('direct centroid distance differs from pinned graph')
if not np.array_equal(mat[add_c],parent_mat[add_p]):raise RuntimeError('added flow material mismatch')
for p,c in zip(add_p,add_c):
 if not np.any(rids[roff[c]:roff[c+1]]==parent_region[p]):raise RuntimeError('added flow fails authenticated one-hop region permission')
det=np.einsum('ij,ij->i',txyz[:,1]-txyz[:,0],np.cross(txyz[:,2]-txyz[:,0],txyz[:,3]-txyz[:,0]))
direct_vol=np.abs(det)/6.0
if np.any(direct_vol<=0) or not np.allclose(direct_vol,hybrid_vol[add_c],rtol=2e-7,atol=0):raise RuntimeError('direct current recipient volume fails source hybrid parity')
after=scaled_replay+add_child
density=after[add_c]/direct_vol
if np.any(density>950.0+5e-10):raise RuntimeError('merged touched receiver exceeds 950 kg/m3')
added=float(add_m.sum()); residual=float(q_after.sum())
if abs((scaled_mass.sum()+added+residual)-parent_mass.sum())>5e-13*parent_mass.sum():raise RuntimeError('partial source total does not close')
indep={'schema':'numi.human.current406-fat-centroid-batch-independent-replay.v1','status':'PASS_INDEPENDENT_PARTIAL_CURRENT406_CENTROID_BATCH_REPLAY','physical_admission':False,'native_steps':0,'complete_allocation':False,'inputs':{p:sha(Path(p)) for p in [str(GRAPH),str(GREPORT),str(LEDGER),str(LEDGER_REPLAY),str(PARENTCSV),str(TARG),str(TARGREP),str(SUPPORT),str(SUPPORTREP),str(PRODUCER),str(ADD),str(MERGED),str(Path(__file__))]},'counts':{'base_retained_rows':int(nr),'new_centroid_flow_rows':int(len(add_m)),'distinct_added_parents':int(len(np.unique(add_p))),'distinct_added_children':int(len(np.unique(add_c))),'touched_children_still_below_850':int(np.count_nonzero(density<850.0)),'merged_flow_rows':int(len(mm))},'mass_kg':{'unscaled_held':float(lm[ridx].sum()),'scaled_held':float(scaled_mass.sum()),'held_scale_released':held_release,'added':added,'merged_transferred':float(mm.sum()),'source_remaining_residual':residual,'source_total':float(parent_mass.sum()),'receiver_lower_deficit_before':float(low.sum()),'receiver_lower_deficit_after':float(low_after.sum()),'receiver_upper_headroom_after':float(high_after.sum())},'checks':{'base_ledger_independent_replay_pass':'PASS','scaled_held_exact_child_replay':True,'added_site_rows_exactly_pinned':True,'source_global_rows_exact':True,'source_corner_residual_replay':True,'merged_parent_M4_replay_with_remaining_residual':True,'merged_child_mass_replay':True,'added_material_and_onehop_region':True,'added_actual_centroid_radius':True,'added_density_upper_bound':True,'direct_current_volume_parity':True,'source_total_conservation':True,'lower_density_complete':False,'complete_allocation':False},'metrics':{'max_distance_m':float(direct_dist.max(initial=0.0)),'max_volume_relative_parity_error':float(np.max(np.abs(direct_vol-hybrid_vol[add_c])/direct_vol,initial=0.0)),'min_touched_density_kg_m3':float(density.min(initial=np.inf)),'max_touched_density_kg_m3':float(density.max(initial=0.0)),'max_parent_corner_residual_mismatch_kg':float(np.max(np.abs(mcorner-(parent_mass[:,None]*.25-q_after)),initial=0.0))},'elapsed_seconds':time.time()-T0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
DEST.write_text(json.dumps(indep,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':indep['status'],'counts':indep['counts'],'mass_kg':indep['mass_kg'],'metrics':indep['metrics'],'receipt':str(DEST),'receipt_sha256':sha(DEST)},indent=2))

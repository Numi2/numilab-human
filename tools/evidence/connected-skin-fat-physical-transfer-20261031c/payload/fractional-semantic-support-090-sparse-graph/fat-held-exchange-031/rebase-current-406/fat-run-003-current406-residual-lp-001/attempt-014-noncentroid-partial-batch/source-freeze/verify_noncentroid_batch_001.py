#!/usr/bin/env python3
from pathlib import Path
import csv,gzip,hashlib,json,time,resource
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'
H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
BASE012=RUN/'attempt-012-centroid-partial-batch'
OUT=RUN/'attempt-014-noncentroid-partial-batch'
GRAPH=RUN/'attempt-005/lp-input-current406.npz'
GREPORT=RUN/'attempt-005/lp-report.json'
LEDGER=S/'fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/rebased-held-flow-ledger.npz'
PARENTCSV=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
SRC=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
NODECSV=SRC/'compiled-parent-cell-mass.csv.nodes.csv'
TETCSV=SRC/'compiled-parent-cell-mass.csv.tets.csv'
OWNER_GZ=SRC/'source/NumiHumanPhysicalSkinChildMassMapping.hpp.gz'
OWNER_MANIFEST=SRC/'launch-manifest.json'
TARG=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
SUPPORT=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
SUPPORTREP=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
BASEADD=BASE012/'partial-centroid-additions.npz'
BASEMERGED=BASE012/'partial-merged-held-plus-additions.npz'
BASEVERIFY=BASE012/'independent-replay-002-corrected.json'
BASEPRODUCER=BASE012/'producer-report.json'
ADD=OUT/'partial-noncentroid-additions.npz'
MERGED=OUT/'partial-merged-held-centroid-noncentroid.npz'
PRODUCER=OUT/'producer-report.json'
DEST=OUT/'independent-replay-001.json'
T0=time.time()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
expected={
 str(GRAPH):'4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328',
 str(GREPORT):'d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767',
 str(LEDGER):'0ee4e4909740e0029a10cb0019a531652b90c882226a0212f6d688f0bbf405fc',
 str(NODECSV):'7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044',
 str(TETCSV):'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992',
 str(OWNER_GZ):'a6e26dd70101bb5d62ce924147c66f9025671997637f9e4ceb3de5b3beb8d069',
 str(OWNER_MANIFEST):'c4b4654d7836e680292f89c8351e5d821b2099171685aece618a38a26da43306',
 str(TARG):'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',
 str(SUPPORT):'e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526',
 str(SUPPORTREP):'b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482',
 str(BASEADD):'be4f6f634b1fd0beb99265708df90163a1f2804167755d8a083d2e4ee0555d32',
 str(BASEMERGED):'112e669b9ae080a11d9be06eb569dc4cbc9604f23cf2493eba1931c120c4a059',
 str(BASEVERIFY):'535241a4aab0d9bb9a098c50503f010a0345bd0f1ba403ae196de47f6cc70c5e',
 str(BASEPRODUCER):'01946c3f29b74a3f22a0be83b86ecea7802d153c0c5ac0ab970ff868684913c7',
 str(ADD):'da134faf8f980393a142f87a78ed40548cbbd0b38193ae2cb1fc91050214ecf7',
 str(MERGED):'9dabd0e81ada103b9c900c574fb7706d12325222d06db195bd858f11a693d1f0',
 str(PRODUCER):'5ae12ea126a8603a8701327ad391b39c63b7b5e0a9cbf29b7a29f176d04e0c15'
}
for p,h in expected.items():
 if sha_file(Path(p))!=h:raise RuntimeError('pinned input hash mismatch '+p)
br=json.loads(BASEVERIFY.read_text())
if br.get('status')!='PASS_INDEPENDENT_PARTIAL_CURRENT406_CENTROID_BATCH_REPLAY' or br.get('physical_admission') is not False:raise RuntimeError('centroid base independent receipt not PASS')
manifest=json.loads(OWNER_MANIFEST.read_text())
owner_bytes=gzip.open(OWNER_GZ,'rb').read()
owner_uncompressed_sha=hashlib.sha256(owner_bytes).hexdigest()
if owner_uncompressed_sha!=manifest['source_sha256']['NumiHumanPhysicalSkinChildMassMapping.hpp']:raise RuntimeError('rest-coordinate owner source does not match compiler manifest')
owner_text=owner_bytes.decode('utf-8')
for needle in ['row[5u+axis]','referenceWorldPoints.push_back(reference)','const auto& points=next.referenceWorldPoints']:
 if needle not in owner_text:raise RuntimeError('source reference coordinate semantics missing: '+needle)
with PARENTCSV.open(newline='') as f: pr=list(csv.DictReader(f))
if len(pr)!=14316 or [int(r['parent_row']) for r in pr]!=list(range(14316)):raise RuntimeError('source role-row domain mismatch')
pg=np.asarray([int(r['global_source330_row']) for r in pr],dtype=np.int64)
pmat=np.asarray([int(r['material']) for r in pr],dtype=np.int32)
pdon=np.asarray([int(r['geometric_donor']) for r in pr],dtype=np.int32)
preg=np.asarray([int(r['region']) for r in pr],dtype=np.int32)
pmass=np.asarray([float(r['mass_kg']) for r in pr],dtype=np.float64)
pcent=np.asarray([[float(r[f'source_centroid_{a}_m']) for a in 'xyz'] for r in pr],dtype=np.float64)
if not np.array_equal(pg,np.arange(14316,28632,dtype=np.int64)):raise RuntimeError('role/global source330 mapping mismatch')
node=np.loadtxt(NODECSV,delimiter=',',skiprows=1,usecols=(0,5,6,7),dtype=np.float64)
tet=np.loadtxt(TETCSV,delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6),dtype=np.float64)
if not np.array_equal(node[:,0].astype(np.int64),np.arange(len(node),dtype=np.int64)) or not np.array_equal(tet[:,0].astype(np.int64),np.arange(len(tet),dtype=np.int64)):raise RuntimeError('source node/tet row ordering mismatch')
pn=tet[pg,3:7].astype(np.int64)
if not np.array_equal(tet[pg,1].astype(np.int32),pmat) or not np.array_equal(tet[pg,2].astype(np.int32),pdon):raise RuntimeError('source tet labels differ from role parent map')
verts=node[pn,1:4]
if not np.array_equal(verts.mean(axis=1),pcent):raise RuntimeError('reference vertex centroids do not exactly reproduce pinned source centroids')
with np.load(GRAPH,allow_pickle=False) as z:
 gp=np.asarray(z['parent_role_rows'],dtype=np.int32); gg=np.asarray(z['parent_global_source330_rows'],dtype=np.int64)
 gc=np.asarray(z['child_rows'],dtype=np.int32); gb=np.asarray(z['barycentric'],dtype=np.float64)
 gd=np.asarray(z['distance_m'],dtype=np.float64); gcap=np.asarray(z['column_mass_scale_kg'],dtype=np.float64)
 hybrid=np.asarray(z['child_volume_hybrid_m3'],dtype=np.float64)
with np.load(BASEADD,allow_pickle=False) as z:
 q1=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 low1=np.asarray(z['child_lower_residual_after_kg'],dtype=np.float64)
 high1=np.asarray(z['child_upper_headroom_after_kg'],dtype=np.float64)
with np.load(BASEMERGED,allow_pickle=False) as z:
 basep=np.asarray(z['flow_parent_role_rows'],dtype=np.int32); baseg=np.asarray(z['flow_global_source330_rows'],dtype=np.int64)
 basec=np.asarray(z['flow_current_child_rows'],dtype=np.int32); basem=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 baseb=np.asarray(z['flow_source_barycentric'],dtype=np.float64); basex=np.asarray(z['flow_source_point_world_m'],dtype=np.float64)
 based=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64)
 baseorigin=np.asarray(z['flow_origin_code'],dtype=np.uint8); baserow=np.asarray(z['base_ledger_flow_rows'],dtype=np.int64)
 basecm=np.asarray(z['child_current_mass_kg'],dtype=np.float64)
 baseq=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
if not np.array_equal(baseq,q1) or len(basep)!=764081:raise RuntimeError('verified centroid state does not match phase-2 parent')
with np.load(ADD,allow_pickle=False) as z:
 rows=np.asarray(z['source_candidate_site_rows'],dtype=np.int64); p=np.asarray(z['parent_role_rows'],dtype=np.int32)
 g=np.asarray(z['global_source330_rows'],dtype=np.int64); c=np.asarray(z['child_current_rows'],dtype=np.int32)
 m=np.asarray(z['flow_mass_kg'],dtype=np.float64); b=np.asarray(z['source_barycentric'],dtype=np.float64)
 x=np.asarray(z['source_point_world_m'],dtype=np.float64); d=np.asarray(z['measured_distance_m'],dtype=np.float64)
 q2=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 dmom=np.asarray(z['parent_corner_delta_kg'],dtype=np.float64); dchild=np.asarray(z['child_mass_delta_kg'],dtype=np.float64)
 low2=np.asarray(z['child_lower_residual_after_kg'],dtype=np.float64); high2=np.asarray(z['child_upper_headroom_after_kg'],dtype=np.float64)
if len(rows)!=6353 or len(np.unique(rows))!=len(rows):raise RuntimeError('new-site row count/uniqueness mismatch')
if np.any(rows<0)|np.any(rows>=len(gp)):raise RuntimeError('candidate site index out of range')
if not np.array_equal(p,gp[rows]) or not np.array_equal(g,gg[rows]) or not np.array_equal(c,gc[rows]) or not np.array_equal(b,gb[rows]) or not np.array_equal(d,gd[rows]):raise RuntimeError('new flows do not match pinned candidate graph rows')
if np.any(np.all(b==.25,axis=1)):raise RuntimeError('centroid site leaked into non-centroid phase')
if not np.array_equal(g,pg[p]) or not np.array_equal(x,np.einsum('ij,ijk->ik',b,verts[p])):raise RuntimeError('source role/global point reconstruction mismatch')
if np.any(m<=0) or np.any(m>gcap[rows]) or np.any(~np.isfinite(m)):raise RuntimeError('new flow invalid or exceeds per-site cap')
if np.any(b<0) or np.any(np.abs(b.sum(axis=1)-1)>2e-14):raise RuntimeError('new source barycentric site outside simplex')
ctol=5e-13*(pmass[:,None]*.25)
if np.any((m[:,None]*b)>q1[p]+ctol[p]):raise RuntimeError('new flow exceeds remaining source-corner M/4 capacity')
dc=np.column_stack([np.bincount(p,weights=m*b[:,k],minlength=len(pmass)) for k in range(4)])
if np.any(np.abs(dc-dmom)>ctol):raise RuntimeError('new source corner rows do not reaggregate')
if np.any(np.abs((q1-dc)-q2)>ctol):raise RuntimeError('remaining source corners do not replay')
newchild=np.bincount(c,weights=m,minlength=len(low1))
if np.any(np.abs(newchild-dchild)>64*np.finfo(float).eps*np.maximum(dchild,np.finfo(float).tiny)):raise RuntimeError('new child flow rows do not reaggregate')
if np.any(newchild>low1+1e-15*np.maximum(low1,1e-300)) or np.any(newchild>high1+1e-15*np.maximum(high1,1e-300)):raise RuntimeError('new flow exceeds receiver lower amount or upper headroom')
if np.any(np.abs(low1-newchild-low2)>64*np.finfo(float).eps*np.maximum(low1,np.finfo(float).tiny)) or np.any(np.abs(high1-newchild-high2)>64*np.finfo(float).eps*np.maximum(high1,np.finfo(float).tiny)):raise RuntimeError('remaining receiver bounds do not replay')
producer=json.loads(PRODUCER.read_text())
if producer.get('status')!='PASS_SOURCE_ONLY_PARTIAL_NONCENTROID_SITE_ALLOCATION' or producer.get('physical_admission') is not False or producer.get('native_steps')!=0 or producer.get('complete_allocation') is not False:raise RuntimeError('producer status is not source-only partial')
sourcehash=hashlib.sha256(Path(__file__).parent.joinpath('greedy_noncentroid_partial_003.py').read_bytes()).hexdigest()
if producer.get('inputs',{}).get(str(Path(__file__).parent/'greedy_noncentroid_partial_003.py'))!=sourcehash:raise RuntimeError('producer source path/hash mismatch')
with np.load(MERGED,allow_pickle=False) as z:
 mp=np.asarray(z['flow_parent_role_rows'],dtype=np.int32); mg=np.asarray(z['flow_global_source330_rows'],dtype=np.int64)
 mc=np.asarray(z['flow_current_child_rows'],dtype=np.int32); mm=np.asarray(z['flow_mass_kg'],dtype=np.float64)
 mb=np.asarray(z['flow_source_barycentric'],dtype=np.float64); mx=np.asarray(z['flow_source_point_world_m'],dtype=np.float64)
 md=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64); origin=np.asarray(z['flow_origin_code'],dtype=np.uint8)
 brow=np.asarray(z['base_ledger_flow_rows'],dtype=np.int64); mq=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 mcm=np.asarray(z['child_current_mass_kg'],dtype=np.float64); mpm=np.asarray(z['parent_source_mass_kg'],dtype=np.float64)
if not np.array_equal(mpm,pmass) or len(mm)!=len(basem)+len(m):raise RuntimeError('merged output row domain mismatch')
if not np.array_equal(mp[:len(basem)],basep) or not np.array_equal(mg[:len(baseg)],baseg) or not np.array_equal(mc[:len(basec)],basec) or not np.array_equal(mm[:len(basem)],basem) or not np.array_equal(mb[:len(baseb)],baseb) or not np.array_equal(mx[:len(basex)],basex) or not np.array_equal(md[:len(based)],based):raise RuntimeError('merged base prefix changed')
if not np.array_equal(mp[len(basem):],p) or not np.array_equal(mg[len(baseg):],g) or not np.array_equal(mc[len(basec):],c) or not np.array_equal(mm[len(basem):],m) or not np.array_equal(mb[len(baseb):],b) or not np.array_equal(mx[len(basex):],x) or not np.array_equal(md[len(based):],d):raise RuntimeError('merged non-centroid suffix mismatch')
if not np.all(origin[:len(basem)]==baseorigin) or not np.array_equal(brow[:len(basem)],baserow) or not np.all(origin[len(basem):]==2) or not np.all(brow[len(basem):]==-1):raise RuntimeError('merged flow lineage codes mismatch')
if not np.array_equal(mq,q2):raise RuntimeError('merged source residual mismatch')
mcorner=np.column_stack([np.bincount(mp,weights=mm*mb[:,k],minlength=len(pmass)) for k in range(4)])
if np.any(np.abs(mcorner-(pmass[:,None]*.25-q2))>ctol):raise RuntimeError('merged M/4 plus residual ledger mismatch')
mchild=np.bincount(mc,weights=mm,minlength=len(basecm))
if np.any(np.abs(mchild-mcm)>64*np.finfo(float).eps*np.maximum(mcm,np.finfo(float).tiny)):raise RuntimeError('merged child flow ledger mismatch')
if not np.array_equal(mcm,basecm+newchild):raise RuntimeError('merged child mass state mismatch')
target=np.load(TARG,allow_pickle=False); xyz=np.asarray(target['points_world_m'],dtype=np.float64)
childtet=np.asarray(target['tetrahedra'],dtype=np.int64); childmat=np.asarray(target['material_indices'],dtype=np.int32)
tx=xyz[childtet[c]]; center=tx.mean(axis=1); ddirect=np.linalg.norm(x-center,axis=1)
if np.any(ddirect>0.019353) or not np.allclose(ddirect,d,rtol=0,atol=5e-12):raise RuntimeError('direct current receiver centroid distance fails')
if not np.array_equal(childmat[c],pmat[p]):raise RuntimeError('material mismatch')
support=np.load(SUPPORT,allow_pickle=False); ro=np.asarray(support['cell_admissible_region_offsets'],dtype=np.int64); ri=np.asarray(support['cell_admissible_region_ids'],dtype=np.int32)
for pp,cc in zip(p,c):
 if not np.any(ri[ro[cc]:ro[cc+1]]==preg[pp]):raise RuntimeError('one-hop region mismatch')
det=np.einsum('ij,ij->i',tx[:,1]-tx[:,0],np.cross(tx[:,2]-tx[:,0],tx[:,3]-tx[:,0]))
v=np.abs(det)/6
if np.any(v<=0) or not np.allclose(v,hybrid[c],rtol=2e-7,atol=0):raise RuntimeError('direct recipient volume/hybrid parity failure')
density=(basecm+dchild)[c]/v
if np.any(density>950.0+5e-10):raise RuntimeError('upper density bound exceeded')
u,first=np.unique(c,return_index=True); u_density=(basecm+dchild)[u]/v[first]
rem=float(q2.sum()); add=float(m.sum())
if abs((mm.sum()+rem)-pmass.sum())>5e-13*pmass.sum():raise RuntimeError('total source mass plus residual mismatch')
receipt={'schema':'numi.human.current406-fat-noncentroid-batch-independent-replay.v1','status':'PASS_INDEPENDENT_PARTIAL_CURRENT406_NONCENTROID_BATCH_REPLAY','physical_admission':False,'native_steps':0,'complete_allocation':False,'inputs':{str(p):sha_file(p) for p in [GRAPH,GREPORT,LEDGER,PARENTCSV,NODECSV,TETCSV,OWNER_GZ,OWNER_MANIFEST,TARG,SUPPORT,SUPPORTREP,BASEADD,BASEMERGED,BASEPRODUCER,BASEVERIFY,ADD,MERGED,PRODUCER,Path(__file__)]},'source_coordinate_contract':{'owner':'NumiHumanPhysicalSkinChildMassMapping.hpp','owner_uncompressed_sha256':owner_uncompressed_sha,'owner_pin_in_compiler_manifest':manifest['source_sha256']['NumiHumanPhysicalSkinChildMassMapping.hpp'],'coordinate_lane':'source330 referenceWorldPoints reconstructed from compiled-parent-cell-mass.csv.nodes.csv rest_x_m/rest_y_m/rest_z_m columns','parent_reference_centroids_bitwise_replayed':True,'max_centroid_rebuild_error_m':float(np.max(np.abs(verts.mean(axis=1)-pcent),initial=0.0))},'counts':{'base_merged_flow_rows':int(len(basem)),'new_noncentroid_flow_rows':int(len(m)),'distinct_added_source_parents':int(len(np.unique(p))),'distinct_added_children':int(len(u)),'new_sites_by_positive_barycentric_corner_count':{str(k):int(np.count_nonzero(np.sum(b>0,axis=1)==k)) for k in range(1,5)},'touched_children_below_850_direct_volume':int(np.count_nonzero(u_density<850.0)),'merged_flow_rows':int(len(mm))},'mass_kg':{'base_source_corner_residual':float(q1.sum()),'added_noncentroid':add,'source_corner_residual_after':rem,'base_receiver_lower_deficit':float(low1.sum()),'receiver_lower_deficit_after':float(low2.sum()),'receiver_upper_headroom_after':float(high2.sum()),'merged_transferred':float(mm.sum()),'source_parent_total':float(pmass.sum()),'total_with_residual':float(mm.sum()+rem)},'checks':{'centroid_partial_independent_base_pass':True,'source_coordinate_owner_and_rest_lane_pinned':True,'source_tet_material_donor_role_map':True,'candidate_rows_and_barycentric_exact':True,'new_mass_positive_and_site_cap':True,'source_M4_corner_caps_and_residual':True,'receiver_low_and_high_caps':True,'source_point_reconstruction':True,'current_centroid_radius':True,'same_material':True,'143_onehop_region':True,'actual_volume_positive_and_pinned_hybrid_parity':True,'merged_flow_prefix_suffix_lineage':True,'merged_parent_corner_replay':True,'merged_child_mass_replay':True,'source_total_plus_residual':True,'upper_density':True,'lower_density_complete':False,'complete_allocation':False},'metrics':{'max_radius_m':float(ddirect.max(initial=0.0)),'max_candidate_graph_distance_delta_m':float(np.max(np.abs(ddirect-d),initial=0.0)),'max_direct_volume_relative_error':float(np.max(np.abs(v-hybrid[c])/v,initial=0.0)),'min_touched_density_kg_m3':float(np.min(u_density,initial=np.inf)),'max_touched_density_kg_m3':float(np.max(u_density,initial=0.0)),'max_source_corner_residual_error_kg':float(np.max(np.abs(mcorner-(pmass[:,None]*.25-q2)),initial=0.0))},'elapsed_seconds':time.time()-T0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
DEST.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':receipt['status'],'counts':receipt['counts'],'mass_kg':receipt['mass_kg'],'metrics':receipt['metrics'],'receipt':str(DEST),'receipt_sha256':sha_file(DEST)},indent=2))

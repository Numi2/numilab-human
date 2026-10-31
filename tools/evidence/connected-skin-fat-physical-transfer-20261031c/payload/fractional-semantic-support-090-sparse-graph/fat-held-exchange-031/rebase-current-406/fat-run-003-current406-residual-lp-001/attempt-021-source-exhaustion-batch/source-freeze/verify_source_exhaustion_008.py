#!/usr/bin/env python3
from pathlib import Path
import csv,gzip,hashlib,json,time,resource
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'; H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
OUT=RUN/'attempt-021-source-exhaustion-batch'; B12=RUN/'attempt-012-centroid-partial-batch'; B14=RUN/'attempt-014-noncentroid-partial-batch'
G=RUN/'attempt-005/lp-input-current406.npz'; GR=RUN/'attempt-005/lp-report.json'
PARENT=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
SRC=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
NODE=SRC/'compiled-parent-cell-mass.csv.nodes.csv'; TET=SRC/'compiled-parent-cell-mass.csv.tets.csv'
OWNER=SRC/'source/NumiHumanPhysicalSkinChildMassMapping.hpp.gz'; OM=SRC/'launch-manifest.json'
T=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
SUP=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
SUPR=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
A12=B12/'partial-centroid-additions.npz'; M12=B12/'partial-merged-held-plus-additions.npz'; V12=B12/'independent-replay-002-corrected.json'
A14=B14/'partial-noncentroid-additions.npz'; M14=B14/'partial-merged-held-centroid-noncentroid.npz'; V14=B14/'independent-replay-002.json'
A=OUT/'run-001/proposal.npz'; M=OUT/'run-001/merged-held-centroid-noncentroid-exhaustion.npz'; R=OUT/'run-001/producer-report.json'
SOURCE=Path(__file__).with_name('phase3_source_exhaustion_008.py'); DEST=OUT/'independent-replay-008.json'
t0=time.time(); radius=.019353; low=850.; high=950.
# The emitted child point rows are F32; centroid arithmetic widens each vertex before accumulation.
f32x=np.asarray([0.9999994039535522,1.000000238418579,1.000006079673767,1.0000108480453491],dtype=np.float32)
f32_fixture=np.column_stack((f32x,np.zeros(4,dtype=np.float32),np.zeros(4,dtype=np.float32)))
center_f32=f32_fixture.mean(axis=0).astype(np.float64)
center_widened=f32_fixture.astype(np.float64).mean(axis=0)
shift=center_f32-center_widened; shift_norm=float(np.linalg.norm(shift))
if shift_norm<=0:raise RuntimeError('F32 centroid precision fixture failed to distinguish arithmetic')
axis=shift/shift_norm; test_point=center_widened+axis*(radius+.5*shift_norm)
dist_widened=float(np.linalg.norm(test_point-center_widened));dist_f32=float(np.linalg.norm(test_point-center_f32))
if not (dist_widened>radius and dist_f32<radius):raise RuntimeError('F32 centroid radius decision control did not discriminate')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
pins={G:'4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328',GR:'d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767',NODE:'7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044',TET:'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992',OWNER:'a6e26dd70101bb5d62ce924147c66f9025671997637f9e4ceb3de5b3beb8d069',OM:'c4b4654d7836e680292f89c8351e5d821b2099171685aece618a38a26da43306',T:'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',SUP:'e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526',SUPR:'b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482',A12:'be4f6f634b1fd0beb99265708df90163a1f2804167755d8a083d2e4ee0555d32',M12:'112e669b9ae080a11d9be06eb569dc4cbc9604f23cf2493eba1931c120c4a059',V12:'535241a4aab0d9bb9a098c50503f010a0345bd0f1ba403ae196de47f6cc70c5e',A14:'da134faf8f980393a142f87a78ed40548cbbd0b38193ae2cb1fc91050214ecf7',M14:'9dabd0e81ada103b9c900c574fb7706d12325222d06db195bd858f11a693d1f0',V14:'43cda84682df5dffeead36f7a61b33872e2ed9b54009b1b5c6716166900d0417',SOURCE:'b3b964dd2a8bccb872ec3084eda51fc5e98cd29c47e5b103b3a552bfca166f3b'}
for f,h in pins.items():
 if sha(f)!=h:raise RuntimeError('pin mismatch '+str(f))
prod=json.loads(R.read_text())
if prod['status']!='PASS_SOURCE_ONLY_PARTIAL_SOURCE_EXHAUSTION_BATCH' or prod['physical_admission'] is not False or prod['native_steps']!=0 or prod['complete_allocation'] is not False:raise RuntimeError('invalid producer status')
for n,v in prod['outputs'].items():
 f=R.parent/n
 if sha(f)!=v['sha256'] or f.stat().st_size!=v['size_bytes']:raise RuntimeError('producer output pin mismatch')
if json.loads(V12.read_text())['status']!='PASS_INDEPENDENT_PARTIAL_CURRENT406_CENTROID_BATCH_REPLAY' or json.loads(V14.read_text())['status']!='PASS_INDEPENDENT_PARTIAL_CURRENT406_NONCENTROID_BATCH_REPLAY':raise RuntimeError('base independent replay missing')
owner=gzip.open(OWNER,'rb').read(); owner_sha=hashlib.sha256(owner).hexdigest(); om=json.loads(OM.read_text())
if owner_sha!=om['source_sha256']['NumiHumanPhysicalSkinChildMassMapping.hpp']:raise RuntimeError('source coordinate owner mismatch')
for x in ['row[5u+axis]','referenceWorldPoints.push_back(reference)','const auto& points=next.referenceWorldPoints']:
 if x not in owner.decode():raise RuntimeError('source reference lane semantics missing')
with PARENT.open(newline='') as f: rows=list(csv.DictReader(f))
pg=np.array([int(x['global_source330_row']) for x in rows],np.int64); mat=np.array([int(x['material']) for x in rows],np.int32); donor=np.array([int(x['geometric_donor']) for x in rows],np.int32); region=np.array([int(x['region']) for x in rows],np.int32); pm=np.array([float(x['mass_kg']) for x in rows],np.float64); pcent=np.array([[float(x[f'source_centroid_{a}_m']) for a in 'xyz'] for x in rows],np.float64)
if len(rows)!=14316 or not np.array_equal(pg,np.arange(14316,28632)):raise RuntimeError('source parent domain mismatch')
node=np.loadtxt(NODE,delimiter=',',skiprows=1,usecols=(0,5,6,7),dtype=np.float64); tet=np.loadtxt(TET,delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6),dtype=np.float64)
if not np.array_equal(node[:,0].astype(np.int64),np.arange(len(node))) or not np.array_equal(tet[:,0].astype(np.int64),np.arange(len(tet))):raise RuntimeError('source node/tet row order mismatch')
pn=tet[pg,3:7].astype(np.int64); verts=node[pn,1:4]
if not np.array_equal(verts.mean(axis=1),pcent) or not np.array_equal(tet[pg,1].astype(np.int32),mat) or not np.array_equal(tet[pg,2].astype(np.int32),donor):raise RuntimeError('source reference points or labels mismatch')
with np.load(G,allow_pickle=False) as z:
 P=z['parent_role_rows']; GG=z['parent_global_source330_rows']; C=z['child_rows']; B=z['barycentric']; D=z['distance_m']; cap=z['column_mass_scale_kg']; volhy=z['child_volume_hybrid_m3']
if not np.array_equal(GG,pg[P]):raise RuntimeError('graph global-parent map mismatch')
keys=['flow_origin_code','flow_parent_role_rows','flow_global_source330_rows','flow_current_child_rows','flow_mass_kg','flow_source_barycentric','flow_source_point_world_m','flow_current_centroid_distance_m','base_ledger_flow_rows','parent_source_mass_kg','parent_corner_residual_after_kg','child_current_mass_kg']
def loadnpz(f,ks):
 with np.load(f,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in ks}
base12=loadnpz(M12,keys); base=loadnpz(M14,keys)
n12=len(base12['flow_mass_kg'])
for k in keys[:10]:
 if not np.array_equal(base12[k],base[k][:n12]):raise RuntimeError('012 merged ledger is not exact 014 prefix: '+k)
if not np.array_equal(base['parent_source_mass_kg'],pm) or not np.array_equal(base['flow_global_source330_rows'],pg[base['flow_parent_role_rows']]):raise RuntimeError('base source domain changed')
s12=loadnpz(A12,['source_candidate_site_rows','parent_role_rows','global_source330_rows','child_current_rows','source_barycentric','flow_mass_kg'])
s14=loadnpz(A14,['source_candidate_site_rows','parent_role_rows','global_source330_rows','child_current_rows','source_barycentric','flow_mass_kg'])
def stagecheck(code,stage,merged):
 msk=merged['flow_origin_code']==code
 if int(msk.sum())!=len(stage['flow_mass_kg']):raise RuntimeError('stage not counted exactly once')
 for sk,mk in [('parent_role_rows','flow_parent_role_rows'),('global_source330_rows','flow_global_source330_rows'),('child_current_rows','flow_current_child_rows'),('source_barycentric','flow_source_barycentric'),('flow_mass_kg','flow_mass_kg')]:
  if not np.array_equal(stage[sk],merged[mk][msk]):raise RuntimeError('stage merged mismatch '+str(code)+' '+sk)
stagecheck(1,s12,base12);stagecheck(1,s12,base);stagecheck(2,s14,base)
a=loadnpz(A,['source_candidate_site_rows','parent_role_rows','global_source330_rows','child_current_rows','source_barycentric','flow_mass_kg','parent_corner_residual_after_kg','child_lower_residual_after_kg','child_upper_headroom_after_kg'])
rows=a['source_candidate_site_rows']; p=a['parent_role_rows']; g=a['global_source330_rows']; c=a['child_current_rows']; m=a['flow_mass_kg']; b=a['source_barycentric']; qafter=a['parent_corner_residual_after_kg']
if len(rows)!=100000 or len(np.unique(rows))!=len(rows):raise RuntimeError('proposal row uniqueness/count')
if not np.array_equal(p,P[rows]) or not np.array_equal(g,GG[rows]) or not np.array_equal(c,C[rows]) or not np.array_equal(b,B[rows]) or not np.array_equal(g,pg[p]):raise RuntimeError('proposal is not exact graph row subset')
if np.any(m<=0) or np.any(~np.isfinite(m)) or np.any(b<0) or np.any(np.abs(b.sum(axis=1)-1)>2e-14):raise RuntimeError('proposal bary/mass invalid')
used=np.zeros(len(cap));np.add.at(used,s12['source_candidate_site_rows'],s12['flow_mass_kg']);np.add.at(used,s14['source_candidate_site_rows'],s14['flow_mass_kg']);np.add.at(used,rows,m)
if np.any(used>cap+64*np.finfo(float).eps*np.maximum(cap,np.finfo(float).tiny)):raise RuntimeError('merged stage site cap exceeded')
base_corner=np.column_stack([np.bincount(base['flow_parent_role_rows'],weights=base['flow_mass_kg']*base['flow_source_barycentric'][:,k],minlength=len(pm)) for k in range(4)])
qbase=base['parent_corner_residual_after_kg'];tol=np.broadcast_to(5e-13*(pm[:,None]*.25),(len(pm),4)).copy()
if np.any(np.abs(pm[:,None]*.25-base_corner-qbase)>tol):raise RuntimeError('base 014 M/4 state mismatch')
dc=np.column_stack([np.bincount(p,weights=m*b[:,k],minlength=len(pm)) for k in range(4)])
if np.any(dc>qbase+tol) or np.any(np.abs(qbase-dc-qafter)>tol):raise RuntimeError('proposal M/4 corners or residual mismatch')
basechild=np.bincount(base['flow_current_child_rows'],weights=base['flow_mass_kg'],minlength=len(volhy))
if np.any(np.abs(basechild-base['child_current_mass_kg'])>64*np.finfo(float).eps*np.maximum(base['child_current_mass_kg'],np.finfo(float).tiny)):raise RuntimeError('base child mass does not replay')
dchild=np.bincount(c,weights=m,minlength=len(volhy)); childfinal=basechild+dchild
with np.load(T,allow_pickle=False) as z:xyz=np.asarray(z['points_world_m'],dtype=np.float64); childtet=np.asarray(z['tetrahedra'],dtype=np.int64); childmat=np.asarray(z['material_indices'],dtype=np.int32)
with np.load(SUP,allow_pickle=False) as z: ro=z['cell_admissible_region_offsets']; ri=z['cell_admissible_region_ids']
if len(childfinal)!=len(childtet) or len(childfinal)!=len(ro)-1 or len(childfinal)!=len(volhy):raise RuntimeError('child row domain mismatch')
pt=np.einsum('ij,ijk->ik',b,verts[p]); tx=xyz[childtet[c]]; cent=tx.mean(axis=1); dr=np.linalg.norm(pt-cent,axis=1)
if np.any(dr>radius) or not np.allclose(dr,D[rows],rtol=0,atol=5e-12):raise RuntimeError('radius replay fail')
if not np.array_equal(childmat[c],mat[p]):raise RuntimeError('material mismatch')
for pp,cc in zip(p,c):
 if region[pp] not in ri[ro[cc]:ro[cc+1]]:raise RuntimeError('143 region mismatch')
dv=np.abs(np.einsum('ij,ij->i',tx[:,1]-tx[:,0],np.cross(tx[:,2]-tx[:,0],tx[:,3]-tx[:,0])))/6
if np.any(dv<=0) or not np.allclose(dv,volhy[c],rtol=2e-7,atol=0):raise RuntimeError('direct volume/hybrid mismatch')
for k in range(4):
 if np.any(m*b[:,k]>qbase[p,k]+tol[p,k]):raise RuntimeError('M/4 local corner cap fail')
expectedq=pm[:,None]*.25-(base_corner+dc)
if np.any(np.abs(expectedq-qafter)>tol):raise RuntimeError('full source residual mismatch')
needbefore=np.maximum(low*volhy-basechild,0); upperbefore=high*volhy-basechild
if np.any(dchild>upperbefore+64*np.finfo(float).eps*np.maximum(upperbefore,np.finfo(float).tiny)):raise RuntimeError('950 upper capacity exceeded')
needafter=np.maximum(low*volhy-childfinal,0); upperafter=high*volhy-childfinal
if not np.allclose(needafter,a['child_lower_residual_after_kg'],rtol=0,atol=64*np.finfo(float).eps*np.maximum(LOW*volhy,np.finfo(float).tiny)):raise RuntimeError('lower residual mismatch')
if not np.allclose(upperafter,a['child_upper_headroom_after_kg'],rtol=0,atol=64*np.finfo(float).eps*np.maximum(np.maximum(upperbefore,1e-300),1e-300)):raise RuntimeError('upper headroom mismatch')
o=loadnpz(M,keys); nb=len(base['flow_mass_kg'])
for k in keys[:10]:
 if not np.array_equal(o[k][:nb],base[k]):raise RuntimeError('merged prefix changed '+k)
suffix={'flow_parent_role_rows':p,'flow_global_source330_rows':g,'flow_current_child_rows':c,'flow_mass_kg':m,'flow_source_barycentric':b,'flow_source_point_world_m':pt,'flow_current_centroid_distance_m':dr}
for k,v in suffix.items():
 if not np.array_equal(o[k][nb:],v):raise RuntimeError('merged suffix mismatch '+k)
if not np.all(o['flow_origin_code'][nb:]==3) or not np.all(o['base_ledger_flow_rows'][nb:]==-1):raise RuntimeError('merged lineage mismatch')
if not np.array_equal(o['parent_corner_residual_after_kg'],qafter) or not np.array_equal(o['parent_source_mass_kg'],pm) or not np.array_equal(o['child_current_mass_kg'],childfinal):raise RuntimeError('merged residual/source/child mismatch')
volall=np.abs(np.einsum('ij,ij->i',xyz[childtet[:,1]]-xyz[childtet[:,0]],np.cross(xyz[childtet[:,2]]-xyz[childtet[:,0]],xyz[childtet[:,3]]-xyz[childtet[:,0]])))/6
density=childfinal/volall
if np.any(density>high+5e-10):raise RuntimeError('all-current-child 950 upper density fail')
trans=float(o['flow_mass_kg'].sum()); residual=float(qafter.sum()); total=float(pm.sum())
if abs(trans+residual-total)>5e-13*total:raise RuntimeError('source total plus residual failure')
receipt={'schema':'numi.human.current406-fat-source-exhaustion-independent-replay.v1','status':'PASS_INDEPENDENT_PARTIAL_CURRENT406_SOURCE_EXHAUSTION_BATCH_REPLAY','physical_admission':False,'native_steps':0,'complete_allocation':False,'inputs':{str(p):sha(p) for p in [G,GR,PARENT,NODE,TET,OWNER,OM,T,SUP,SUPR,A12,M12,V12,A14,M14,V14,A,M,R,SOURCE]},'counts':{'prior_centroid_rows':len(s12['flow_mass_kg']),'prior_noncentroid_rows':len(s14['flow_mass_kg']),'prior_rows_in_014_merged':nb,'new_rows':len(m),'new_parents':int(len(np.unique(p))),'new_children':int(len(np.unique(c))),'all_children_below_850':int(np.count_nonzero(density<low)),'positive_bary_corner_counts':{str(k):int(np.count_nonzero((b>0).sum(axis=1)==k)) for k in range(1,5)}},'mass_kg':{'prior_source_residual':float(qbase.sum()),'new_flow':float(m.sum()),'source_residual_after':residual,'lower_deficit_before':float(needbefore.sum()),'lower_deficit_after':float(needafter.sum()),'upper_headroom_before':float(upperbefore.sum()),'upper_headroom_after':float(upperafter.sum()),'source_total':total,'all_flow_total':trans,'flow_plus_residual':trans+residual},'checks':{'012_and_014_prior_stages_counted_once':True,'site_capacity_after_prior_and_new_flows':True,'full_source_corner_M4':True,'current143_region_material_radius':True,'current_volume_positive_and_hybrid_parity':True,'float32_widened_centroid_control':True,'full_child_upper_density':True,'source_total_plus_residual':True,'lower_density_complete':False,'source_exhaustion_phase_used':prod['counts']['new_flows_by_stage']['source_exhaustion_mass_kg']>0,'complete_allocation':False},'metrics':{'max_radius_m':float(dr.max(initial=0)),'max_radius_graph_error_m':float(np.max(np.abs(dr-D[rows]),initial=0)),'max_source_corner_residual_error_kg':float(np.max(np.abs(expectedq-qafter),initial=0)),'max_volume_relative_error':float(np.max(np.abs(dv-volhy[c])/dv,initial=0)),'full_min_density_kg_m3':float(np.min(density,initial=np.inf)),'full_max_density_kg_m3':float(np.max(density,initial=0)),'rest_centroid_error_m':float(np.max(np.abs(verts.mean(axis=1)-pcent),initial=0)),'precision_fixture':{'float32_mean_m':center_f32.tolist(),'widened_mean_m':center_widened.tolist(),'centroid_delta_m':shift_norm,'test_distance_widened_m':dist_widened,'test_distance_float32_mean_m':dist_f32}},'elapsed_seconds':time.time()-t0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
DEST.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':receipt['status'],'counts':receipt['counts'],'mass_kg':receipt['mass_kg'],'checks':receipt['checks'],'metrics':receipt['metrics'],'receipt':str(DEST),'sha256':sha(DEST)},indent=2))

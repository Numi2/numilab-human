#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,resource,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
S=ROOT/'fractional-semantic-support-090-sparse-graph'
H=ROOT/'fine-physical-volume-706'
RUN=S/'fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
OUT=RUN/'attempt-021-source-exhaustion-batch'
MAX_NEW_ROWS=100000
RADIUS=0.019353
LOW=850.0
HIGH=950.0
T0=time.time()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def log(s):print(f'{time.time()-T0:.2f}s {s}',flush=True)
def site_cap_for(rows,P,B,Q):
 p=P[rows]; b=B[rows]
 ratio=np.full_like(b,np.inf)
 np.divide(Q[p],b,out=ratio,where=b>0)
 return ratio.min(axis=1)
def select_batch(rows,P,G,C,B,D,global_parent,site_rem,Qin,NeedIn,HeadIn,max_rows):
 rows=np.asarray(rows,dtype=np.int64)
 q=Qin.copy(); need=NeedIn.copy(); head=HeadIn.copy(); rem=site_rem.copy()
 alloc={}
 lower_mass=0.0; source_mass=0.0
 if len(rows)==0:
  return {'rows':np.zeros(0,dtype=np.int64),'parents':np.zeros(0,dtype=np.int32),'global':np.zeros(0,dtype=np.int64),'children':np.zeros(0,dtype=np.int32),'bary':np.zeros((0,4),dtype=np.float64),'mass':np.zeros(0,dtype=np.float64),'q_after':q,'need_after':need,'head_after':head,'lower_mass':0.0,'source_mass':0.0,'site_remaining':rem}
 b=B[rows]; p=P[rows]; c=C[rows]
 src_cap=site_cap_for(rows,P,B,q)
 base_cap=np.minimum.reduce((rem[rows],HeadIn[c],NeedIn[c],src_cap))
 phase1=(NeedIn[c]>0)&(HeadIn[c]>0)&(rem[rows]>0)&(src_cap>0)&(base_cap>0)
 r1=rows[phase1]
 if len(r1):
  c1=C[r1]; p1=P[r1]; sc1=site_cap_for(r1,P,B,q)
  cap1=np.minimum.reduce((rem[r1],need[c1],head[c1],sc1))
  ok=cap1>0; r1=r1[ok]; cap1=cap1[ok]
  order=np.lexsort((r1,C[r1],global_parent[P[r1]],D[r1],-cap1,-need[C[r1]]))
  for rr in r1[order]:
   if len(alloc)>=max_rows:break
   pp=int(P[rr]); cc=int(C[rr]); bb=B[rr]
   active=bb>0
   if rem[rr]<=0 or need[cc]<=0 or head[cc]<=0 or np.any(q[pp,active]<=0):continue
   cap=float(np.min(q[pp,active]/bb[active]))
   amt=min(float(rem[rr]),float(need[cc]),float(head[cc]),cap)
   if amt<=0:continue
   amt=float(np.nextafter(amt,0.0))
   if amt<=0:continue
   rem[rr]-=amt; q[pp]-=amt*bb; need[cc]-=amt; head[cc]-=amt
   alloc[int(rr)]=alloc.get(int(rr),0.0)+amt; lower_mass+=amt
 # Source-exhaustion pass: only after all feasible lower-target sites were visited.
 # It uses receiver upper headroom and orders remaining lower deficits ahead of already-satisfied cells.
 if len(alloc)<max_rows:
  sc2=site_cap_for(rows,P,B,q)
  cap2=np.minimum(rem[rows],np.minimum(head[C[rows]],sc2))
  needkey=need[C[rows]]
  phase2=(rem[rows]>0)&(head[C[rows]]>0)&(sc2>0)&(cap2>0)
  r2=rows[phase2]
  if len(r2):
   c2=C[r2]; p2=P[r2]; cap2=np.minimum(rem[r2],np.minimum(head[c2],site_cap_for(r2,P,B,q)))
   ok=cap2>0; r2=r2[ok]; cap2=cap2[ok]
   tier=(need[C[r2]]<=0).astype(np.int8)
   order=np.lexsort((r2,C[r2],global_parent[P[r2]],D[r2],-cap2,-need[C[r2]],tier))
   for rr in r2[order]:
    if len(alloc)>=max_rows and int(rr) not in alloc:break
    pp=int(P[rr]); cc=int(C[rr]); bb=B[rr]; active=bb>0
    if rem[rr]<=0 or head[cc]<=0 or np.any(q[pp,active]<=0):continue
    cap=float(np.min(q[pp,active]/bb[active]))
    amt=min(float(rem[rr]),float(head[cc]),cap)
    if amt<=0:continue
    amt=float(np.nextafter(amt,0.0))
    if amt<=0:continue
    rem[rr]-=amt; q[pp]-=amt*bb; head[cc]-=amt
    need[cc]=max(0.0,float(need[cc])-amt)
    alloc[int(rr)]=alloc.get(int(rr),0.0)+amt; source_mass+=amt
 outrows=np.asarray(sorted(alloc),dtype=np.int64)
 masses=np.asarray([alloc[int(r)] for r in outrows],dtype=np.float64)
 parents=P[outrows].astype(np.int32); children=C[outrows].astype(np.int32)
 return {'rows':outrows,'parents':parents,'global':global_parent[parents].astype(np.int64),'children':children,'bary':B[outrows].copy(),'mass':masses,'q_after':q,'need_after':need,'head_after':head,'lower_mass':lower_mass,'source_mass':source_mass,'site_remaining':rem}
def serialize_proposal(out,result,report):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 npz=out/'proposal.npz'
 np.savez_compressed(npz,source_candidate_site_rows=result['rows'],parent_role_rows=result['parents'],global_source330_rows=result['global'],child_current_rows=result['children'],source_barycentric=result['bary'],flow_mass_kg=result['mass'],parent_corner_residual_after_kg=result['q_after'],child_lower_residual_after_kg=result['need_after'],child_upper_headroom_after_kg=result['head_after'])
 report=dict(report)
 report['outputs']={npz.name:{'sha256':sha(npz),'size_bytes':npz.stat().st_size}}
 report['counts']['flow_rows']=int(len(result['rows']))
 report['mass_kg']['flow_total']=float(result['mass'].sum())
 report['mass_kg']['parent_corner_residual_after']=float(result['q_after'].sum())
 report['mass_kg']['receiver_lower_deficit_after']=float(result['need_after'].sum())
 report['mass_kg']['receiver_upper_headroom_after']=float(result['head_after'].sum())
 rp=out/'producer-report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 with np.load(npz,allow_pickle=False) as z:
  if z['source_candidate_site_rows'].shape!=(len(result['rows']),) or z['source_barycentric'].shape!=(len(result['rows']),4):raise RuntimeError('serialized proposal shape mismatch')
 if json.loads(rp.read_text()).get('physical_admission') is not False or json.loads(rp.read_text()).get('native_steps')!=0:raise RuntimeError('serializer changed source-only status')
 return npz,rp
def hash_input_paths(paths):
 if any(not isinstance(p,(str,Path)) for p in paths):raise RuntimeError('report input must remain an explicit path, not an opened resource')
 return {str(p):sha(p) for p in paths}
def finalize_report(out,report,outputs):
 out=Path(out);report['outputs']={Path(p).name:{'sha256':sha(p),'size_bytes':Path(p).stat().st_size} for p in outputs}
 rp=out/'producer-report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 reread=json.loads(rp.read_text())
 if reread.get('physical_admission') is not False or reread.get('native_steps')!=0 or reread.get('complete_allocation') is not False:raise RuntimeError('final report status guard failed')
 for name,meta in reread['outputs'].items():
  path=out/name
  if sha(path)!=meta['sha256'] or path.stat().st_size!=meta['size_bytes']:raise RuntimeError('final report output pin replay failed')
 return rp
def account_site_usage(cap,*stages):
 cap=np.asarray(cap,dtype=np.float64);used=np.zeros(len(cap),dtype=np.float64)
 for rows,mass in stages:
  rows=np.asarray(rows,dtype=np.int64);mass=np.asarray(mass,dtype=np.float64)
  if rows.ndim!=1 or mass.shape!=rows.shape or np.any(rows<0) or np.any(rows>=len(cap)) or np.any(~np.isfinite(mass)) or np.any(mass<0):raise RuntimeError('invalid prior site usage rows/masses')
  np.add.at(used,rows,mass)
 if np.any(used>cap+64*np.finfo(float).eps*np.maximum(cap,np.finfo(float).tiny)):raise RuntimeError('prior batches overused a candidate site cap')
 return used,cap-used
def verify_stage_in_merged(origin,stage,merged):
 mask=np.asarray(merged['flow_origin_code'])==int(origin)
 if int(mask.sum())!=len(np.asarray(stage['flow_mass_kg'])):raise RuntimeError(f'prior stage origin {origin} flow count is not represented exactly once')
 for sk,mk in (('parent_role_rows','flow_parent_role_rows'),('global_source330_rows','flow_global_source330_rows'),('child_current_rows','flow_current_child_rows'),('source_barycentric','flow_source_barycentric'),('flow_mass_kg','flow_mass_kg')):
  if not np.array_equal(np.asarray(stage[sk]),np.asarray(merged[mk])[mask]):raise RuntimeError(f'prior stage origin {origin} does not match merged ledger field {sk}')
def run_control():
 gp=np.asarray([4274,6006],dtype=np.int64)
 cent={'source_candidate_site_rows':np.asarray([0,3],dtype=np.int64),'parent_role_rows':np.asarray([0,1],dtype=np.int32),'global_source330_rows':np.asarray([4274,6006],dtype=np.int64),'child_current_rows':np.asarray([27,4],dtype=np.int32),'source_barycentric':np.asarray([[.25,.25,.25,.25],[1,0,0,0]],dtype=np.float64),'flow_mass_kg':np.asarray([.05,.10],dtype=np.float64)}
 non={'source_candidate_site_rows':np.asarray([1,2],dtype=np.int64),'parent_role_rows':np.asarray([1,0],dtype=np.int32),'global_source330_rows':np.asarray([6006,4274],dtype=np.int64),'child_current_rows':np.asarray([11,27],dtype=np.int32),'source_barycentric':np.asarray([[0,.5,.5,0],[0,0,1,0]],dtype=np.float64),'flow_mass_kg':np.asarray([.15,.20],dtype=np.float64)}
 merged={'flow_origin_code':np.asarray([1,1,2,2],dtype=np.uint8),'flow_parent_role_rows':np.concatenate((cent['parent_role_rows'],non['parent_role_rows'])),'flow_global_source330_rows':np.concatenate((cent['global_source330_rows'],non['global_source330_rows'])),'flow_current_child_rows':np.concatenate((cent['child_current_rows'],non['child_current_rows'])),'flow_source_barycentric':np.concatenate((cent['source_barycentric'],non['source_barycentric'])),'flow_mass_kg':np.concatenate((cent['flow_mass_kg'],non['flow_mass_kg']))}
 verify_stage_in_merged(1,cent,merged);verify_stage_in_merged(2,non,merged)
 cap=np.asarray([.30,.40,.50,.60]);used,remain=account_site_usage(cap,(cent['source_candidate_site_rows'],cent['flow_mass_kg']),(non['source_candidate_site_rows'],non['flow_mass_kg']))
 if not np.array_equal(used,np.asarray([.05,.15,.20,.10])) or not np.array_equal(remain,np.asarray([.25,.25,.30,.50])):raise RuntimeError('two-stage site-cap remaining control mismatch')
 bad=dict(merged);bad['flow_mass_kg']=merged['flow_mass_kg'].copy();bad['flow_mass_kg'][0]*=2
 try:verify_stage_in_merged(1,cent,bad)
 except RuntimeError:pass
 else:raise RuntimeError('merged-ledger exact-once tamper control was not rejected')
 p=np.asarray([1,0,1,0],dtype=np.int32);c=np.asarray([27,4,11,27],dtype=np.int32)
 b=np.asarray([[.25,.25,.25,.25],[1,0,0,0],[0,.5,.5,0],[.25,.25,.25,.25]],dtype=np.float64)
 d=np.asarray([.001,.002,.003,.004]);sitecap=np.asarray([.01,.015,.012,.02])
 q=np.asarray([[.005,.006,.007,.008],[.01,.011,.012,.013]])
 need=np.zeros(28);head=np.zeros(28);need[[4,11,27]]=[.002,.001,.003];head[[4,11,27]]=[.04,.02,.03]
 result=select_batch(np.arange(4,dtype=np.int64),p,gp,c,b,d,gp,sitecap,q,need,head,10)
 if len(result['rows'])==0 or not np.array_equal(result['global'],gp[result['parents']]):raise RuntimeError('noncontiguous source/global map allocator control failed')
 control=OUT/'control-001'
 rep={'schema':'numi.human.current406-source-exhaustion-entrypoint-control.v1','status':'PASS_CONTROL_TWO_STAGE_LEDGER_CAPS_AND_SERIALIZATION','physical_admission':False,'native_steps':0,'complete_allocation':False,'scope':'Two-stage exact-once ledger/site-cap join plus positive and zero-flow production serializer controls. Synthetic only.','counts':{'global_parent_ids':[4274,6006],'centroid_stage_rows':2,'noncentroid_stage_rows':2,'merged_origin1_rows':2,'merged_origin2_rows':2,'positive_allocator_rows':int(len(result['rows'])),'site_cap_rows':4,'tamper_rejection':'PASS'},'mass_kg':{'prior_centroid':float(cent['flow_mass_kg'].sum()),'prior_noncentroid':float(non['flow_mass_kg'].sum()),'prior_total_counted_once':float(used.sum()),'remaining_site_capacity':float(remain.sum())}}
 posnpz,posrp=serialize_proposal(control/'positive',result,rep)
 posmerged=control/'positive'/'merged-control.npz';np.savez_compressed(posmerged,flow_mass_kg=result['mass'],parent_rows=result['global'])
 posrp=finalize_report(control/'positive',rep,[posnpz,posmerged])
 with np.load(posnpz,allow_pickle=False) as z:
  if not np.array_equal(z['global_source330_rows'],gp[z['parent_role_rows']]) or len(z['flow_mass_kg'])==0:raise RuntimeError('serialized positive fixture identity/flow mismatch')
 zr=select_batch(np.arange(4,dtype=np.int64),p,gp,c,b,d,gp,sitecap,np.zeros_like(q),np.zeros_like(need),np.zeros_like(head),10)
 zrep={'schema':'numi.human.current406-source-exhaustion-entrypoint-control.v1','status':'PASS_CONTROL_ZERO_FLOW_SERIALIZATION','physical_admission':False,'native_steps':0,'complete_allocation':False,'scope':'Empty feasible-site result is serialized with stable zero-length arrays.','counts':{'candidate_rows':4,'global_parent_ids':[4274,6006]},'mass_kg':{'source_total':0.0,'added':0.0}}
 znpz,zrp=serialize_proposal(control/'zero',zr,zrep)
 zmerged=control/'zero'/'merged-control.npz';np.savez_compressed(zmerged,flow_mass_kg=zr['mass'],parent_rows=zr['global'])
 zrp=finalize_report(control/'zero',zrep,[znpz,zmerged])
 with np.load(znpz,allow_pickle=False) as z:
  if z['flow_mass_kg'].shape!=(0,) or z['source_barycentric'].shape!=(0,4) or z['global_source330_rows'].shape!=(0,):raise RuntimeError('zero-flow serialized shape control failed')
 both={'schema':'numi.human.current406-source-exhaustion-entrypoint-control.v1','status':'PASS_CONTROL_TWO_STAGE_LEDGER_CAPS_AND_SERIALIZATION','physical_admission':False,'native_steps':0,'complete_allocation':False,'controls':{'global_parent_domain':[4274,6006],'centroid_rows':2,'noncentroid_rows':2,'site_usage_counted_exactly_once':True,'site_cap_remaining_kg':remain.tolist(),'tampered_merged_projection_rejected':True,'positive_flow_rows':int(len(result['rows'])),'positive_proposal_sha256':sha(posnpz),'positive_report_sha256':sha(posrp),'zero_flow_rows':0,'zero_proposal_sha256':sha(znpz),'zero_report_sha256':sha(zrp)},'source_sha256':sha(Path(__file__))}
 pth=control/'control-report.json';pth.write_text(json.dumps(both,indent=2,sort_keys=True)+'\n')
 print(json.dumps(both,indent=2));print('control_receipt_sha256',sha(pth))
def run_actual():
 parent=S/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv'
 srcroot=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003'
 nodecsv=srcroot/'compiled-parent-cell-mass.csv.nodes.csv';tetcsv=srcroot/'compiled-parent-cell-mass.csv.tets.csv'
 target=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
 targetrep=H/'current-D309-I399-target-406-root/target397/report.json'
 support=H/'connected-region-mass-support-143-current406-root/fat-connected-region-support.npz'
 supportrep=H/'connected-region-mass-support-143-current406-root/independent-verification.json'
 graph=RUN/'attempt-005/lp-input-current406.npz';greport=RUN/'attempt-005/lp-report.json'
 baseout=RUN/'attempt-014-noncentroid-partial-batch';baseadd=baseout/'partial-noncentroid-additions.npz';basemerged=baseout/'partial-merged-held-centroid-noncentroid.npz';baserep=baseout/'producer-report.json';baseverify=baseout/'independent-replay-002.json';basecentroid=RUN/'attempt-012-centroid-partial-batch/partial-centroid-additions.npz';basecentroidrep=RUN/'attempt-012-centroid-partial-batch/producer-report.json';basecentroidverify=RUN/'attempt-012-centroid-partial-batch/independent-replay-002-corrected.json';basecentroidmerged=RUN/'attempt-012-centroid-partial-batch/partial-merged-held-plus-additions.npz'
 ownergz=srcroot/'source/NumiHumanPhysicalSkinChildMassMapping.hpp.gz';ownerman=srcroot/'launch-manifest.json'
 pinned={graph:'4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328',greport:'d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767',target:'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',support:'e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526',supportrep:'b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482',baseadd:'da134faf8f980393a142f87a78ed40548cbbd0b38193ae2cb1fc91050214ecf7',basemerged:'9dabd0e81ada103b9c900c574fb7706d12325222d06db195bd858f11a693d1f0',baserep:'5ae12ea126a8603a8701327ad391b39c63b7b5e0a9cbf29b7a29f176d04e0c15',baseverify:'43cda84682df5dffeead36f7a61b33872e2ed9b54009b1b5c6716166900d0417',basecentroid:'be4f6f634b1fd0beb99265708df90163a1f2804167755d8a083d2e4ee0555d32',basecentroidrep:'01946c3f29b74a3f22a0be83b86ecea7802d153c0c5ac0ab970ff868684913c7',basecentroidverify:'535241a4aab0d9bb9a098c50503f010a0345bd0f1ba403ae196de47f6cc70c5e',basecentroidmerged:'112e669b9ae080a11d9be06eb569dc4cbc9604f23cf2493eba1931c120c4a059',nodecsv:'7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044',tetcsv:'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992',ownergz:'a6e26dd70101bb5d62ce924147c66f9025671997637f9e4ceb3de5b3beb8d069',ownerman:'c4b4654d7836e680292f89c8351e5d821b2099171685aece618a38a26da43306'}
 for path,h in pinned.items():
  if sha(path)!=h:raise RuntimeError('frozen input pin mismatch '+str(path))
 if json.loads(baseverify.read_text()).get('status')!='PASS_INDEPENDENT_PARTIAL_CURRENT406_NONCENTROID_BATCH_REPLAY':raise RuntimeError('phase-2 independent base verification missing')
 if json.loads(basecentroidverify.read_text()).get('status')!='PASS_INDEPENDENT_PARTIAL_CURRENT406_CENTROID_BATCH_REPLAY':raise RuntimeError('phase-1 independent base verification missing')
 with parent.open(newline='') as f: pr=list(csv.DictReader(f))
 pg=np.asarray([int(r['global_source330_row']) for r in pr],dtype=np.int64);pdon=np.asarray([int(r['geometric_donor']) for r in pr],dtype=np.int32);pmat=np.asarray([int(r['material']) for r in pr],dtype=np.int32);preg=np.asarray([int(r['region']) for r in pr],dtype=np.int32);pmass=np.asarray([float(r['mass_kg']) for r in pr],dtype=np.float64)
 if len(pr)!=14316 or not np.array_equal(pg,np.arange(14316,28632,dtype=np.int64)):raise RuntimeError('role source row domain mismatch')
 node=np.loadtxt(nodecsv,delimiter=',',skiprows=1,usecols=(0,5,6,7),dtype=np.float64)
 tet=np.loadtxt(tetcsv,delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6),dtype=np.float64)
 if not np.array_equal(tet[pg,1].astype(np.int32),pmat) or not np.array_equal(tet[pg,2].astype(np.int32),pdon):raise RuntimeError('source parent material/donor crosswalk mismatch')
 pverts=node[tet[pg,3:7].astype(np.int64),1:4]
 with np.load(basemerged,allow_pickle=False) as z:
  bp=np.asarray(z['flow_parent_role_rows'],dtype=np.int32);bg=np.asarray(z['flow_global_source330_rows'],dtype=np.int64);bc=np.asarray(z['flow_current_child_rows'],dtype=np.int32);bm=np.asarray(z['flow_mass_kg'],dtype=np.float64);bb=np.asarray(z['flow_source_barycentric'],dtype=np.float64);bx=np.asarray(z['flow_source_point_world_m'],dtype=np.float64);bd=np.asarray(z['flow_current_centroid_distance_m'],dtype=np.float64)
  borg=np.asarray(z['flow_origin_code'],dtype=np.uint8);brow=np.asarray(z['base_ledger_flow_rows'],dtype=np.int64);bcm=np.asarray(z['child_current_mass_kg'],dtype=np.float64)
  bq=np.asarray(z['parent_corner_residual_after_kg'],dtype=np.float64)
 if not np.array_equal(bg,pg[bp]):raise RuntimeError('verified base merged ledger source mapping mismatch')
 with np.load(basecentroid,allow_pickle=False) as z:centstage={k:np.asarray(z[k]) for k in ['source_candidate_site_rows','parent_role_rows','global_source330_rows','child_current_rows','source_barycentric','flow_mass_kg']}
 with np.load(baseadd,allow_pickle=False) as z:nonstage={k:np.asarray(z[k]) for k in ['source_candidate_site_rows','parent_role_rows','global_source330_rows','child_current_rows','source_barycentric','flow_mass_kg']}
 prior_merged={'flow_origin_code':borg,'flow_parent_role_rows':bp,'flow_global_source330_rows':bg,'flow_current_child_rows':bc,'flow_source_barycentric':bb,'flow_mass_kg':bm}
 verify_stage_in_merged(1,centstage,prior_merged);verify_stage_in_merged(2,nonstage,prior_merged)
 with np.load(basecentroidmerged,allow_pickle=False) as z: oldmerged={k:np.asarray(z[k]) for k in ['flow_origin_code','flow_parent_role_rows','flow_global_source330_rows','flow_current_child_rows','flow_source_barycentric','flow_mass_kg']}
 verify_stage_in_merged(1,centstage,oldmerged)
 prefix=len(oldmerged['flow_mass_kg'])
 for key in oldmerged:
  if not np.array_equal(oldmerged[key],prior_merged[key][:prefix]):raise RuntimeError('012 merged ledger is not an exact prefix of verified 014 ledger: '+key)
 with np.load(graph,allow_pickle=False) as z:
  P=np.asarray(z['parent_role_rows'],dtype=np.int32);G=np.asarray(z['parent_global_source330_rows'],dtype=np.int64);C=np.asarray(z['child_rows'],dtype=np.int32);B=np.asarray(z['barycentric'],dtype=np.float64)
  D=np.asarray(z['distance_m'],dtype=np.float64);cap=np.asarray(z['column_mass_scale_kg'],dtype=np.float64);hybrid=np.asarray(z['child_volume_hybrid_m3'],dtype=np.float64)
  good_bary=np.all(np.isfinite(B),axis=1)&np.all(B>=0,axis=1)&(np.abs(B.sum(axis=1)-1)<=2e-14)
 if not np.array_equal(G,pg[P]):raise RuntimeError('candidate role/global source map mismatch')
 used,site_rem=account_site_usage(cap,(centstage['source_candidate_site_rows'],centstage['flow_mass_kg']),(nonstage['source_candidate_site_rows'],nonstage['flow_mass_kg']))
 # The verified merged 014 ledger is the only held-mass/corner/receiver state source; stage files authenticate row use only.
 bincorner=np.column_stack([np.bincount(bp,weights=bm*bb[:,k],minlength=len(pmass)) for k in range(4)])
 q_replay=pmass[:,None]*.25-bincorner;q0=bq.copy();qtol_state=5e-13*(pmass[:,None]*.25)
 if np.any(np.abs(q0-q_replay)>qtol_state) or np.any(q0 < -qtol_state):raise RuntimeError('merged ledger source-corner state replay failed')
 merged_child_replay=np.bincount(bc,weights=bm,minlength=len(bcm));childtol=64*np.finfo(float).eps*np.maximum(np.abs(bcm),np.finfo(float).tiny)
 if np.any(np.abs(merged_child_replay-bcm)>childtol):raise RuntimeError('merged ledger child mass state replay failed')
 need0=np.maximum(LOW*hybrid-bcm,0.0);head0=HIGH*hybrid-bcm
 if np.any(head0 < -64*np.finfo(float).eps*np.maximum(HIGH*hybrid,np.finfo(float).tiny)):raise RuntimeError('merged ledger already exceeds receiver upper capacity')
 candidate=np.flatnonzero((site_rem>0)&good_bary&(D<=RADIUS))
 log(f'pinned source sites with remaining mass capacity={len(candidate)}; run lower-first then source-exhaustion batch')
 target_data=np.load(target,allow_pickle=False);xyz=np.asarray(target_data['points_world_m'],dtype=np.float64);childtet=np.asarray(target_data['tetrahedra'],dtype=np.int64);childmat=np.asarray(target_data['material_indices'],dtype=np.int32)
 support_data=np.load(support,allow_pickle=False);off=np.asarray(support_data['cell_admissible_region_offsets'],dtype=np.int64);regions=np.asarray(support_data['cell_admissible_region_ids'],dtype=np.int32)
 # Remove isolated exact row witnesses from the eligible set and replay from the same input state.
 excluded=np.zeros(len(P),dtype=bool);result=None;rejects=[];max_passes=6
 for iteration in range(max_passes):
  use=candidate[~excluded[candidate]]
  result=select_batch(use,P,G,C,B,D,pg,site_rem,q0,need0,head0,MAX_NEW_ROWS)
  rr=result['rows']
  if len(rr)==0:break
  p=result['parents'];c=result['children'];b=result['bary'];x=np.einsum('ij,ijk->ik',b,pverts[p])
  tx=xyz[childtet[c]];center=tx.mean(axis=1);direct=np.linalg.norm(x-center,axis=1)
  ddet=np.abs(np.einsum('ij,ij->i',tx[:,1]-tx[:,0],np.cross(tx[:,2]-tx[:,0],tx[:,3]-tx[:,0])))/6
  bad=(direct>RADIUS)|(~np.isfinite(direct))|(childmat[c]!=pmat[p])|(ddet<=0)|(~np.isclose(ddet,hybrid[c],rtol=2e-7,atol=0))
  region_ok=np.ones(len(rr),dtype=bool)
  for j,(pp,cc) in enumerate(zip(p,c)):
   if not np.any(regions[off[cc]:off[cc+1]]==preg[pp]):region_ok[j]=False
  bad|=~region_ok
  if not np.any(bad):
   result['points']=x;result['direct_distance']=direct;result['volume']=ddet
   break
  for j in np.flatnonzero(bad):
   rejects.append({'site_row':int(rr[j]),'parent_role_row':int(p[j]),'global_source330_row':int(pg[p[j]]),'child_row':int(c[j]),'graph_distance_m':float(D[rr[j]]),'direct_distance_m':float(direct[j]),'reason':'DIRECT_RADIUS_MATERIAL_REGION_OR_VOLUME_REPLAY'})
   excluded[rr[j]]=True
 else:raise RuntimeError('bounded direct-site rejection loop did not converge')
 if result is None or len(result['rows'])==0:raise RuntimeError('no positive legal source-exhaustion flows found')
 rows=result['rows'];parents=result['parents'];globalrows=result['global'];children=result['children'];bary=result['bary'];mass=result['mass'];points=result['points'];direct=result['direct_distance'];qafter=result['q_after'];needafter=result['need_after'];headafter=result['head_after']
 # Independently aggregate new delta and the complete merged ledger.
 delta_corner=np.column_stack([np.bincount(parents,weights=mass*bary[:,k],minlength=len(pmass)) for k in range(4)])
 delta_child=np.bincount(children,weights=mass,minlength=len(need0))
 qtol=5e-13*(pmass[:,None]*.25)
 if np.any(delta_corner>q0+qtol):raise RuntimeError('source M/4 corner cap exceeded')
 if np.any(delta_child>head0+1e-15*np.maximum(head0,1e-300)):raise RuntimeError('receiver 950 upper headroom exceeded')
 if np.any(np.abs(q0-delta_corner-qafter)>qtol):raise RuntimeError('source corner remainder mismatch')
 mergedp=np.concatenate((bp,parents));mergedg=np.concatenate((bg,globalrows));mergedc=np.concatenate((bc,children));mergedm=np.concatenate((bm,mass));mergedb=np.concatenate((bb,bary));mergedx=np.concatenate((bx,points));mergedd=np.concatenate((bd,direct))
 mergedorigin=np.concatenate((borg,np.full(len(mass),3,dtype=np.uint8)));mergedbase=np.concatenate((brow,np.full(len(mass),-1,dtype=np.int64)))
 mergedchild=np.bincount(mergedc,weights=mergedm,minlength=len(bcm))
 expected_child=bcm+delta_child
 mtol=64*np.finfo(float).eps*np.maximum(expected_child,np.finfo(float).tiny)
 if np.any(np.abs(mergedchild-expected_child)>mtol):raise RuntimeError('new merged child mass replay failed')
 mergedcorner=np.column_stack([np.bincount(mergedp,weights=mergedm*mergedb[:,k],minlength=len(pmass)) for k in range(4)])
 if np.any(np.abs((pmass[:,None]*.25-mergedcorner)-qafter)>qtol):raise RuntimeError('merged full source M/4 target + residual mismatch')
 targetvol=np.abs(np.einsum('ij,ij->i',xyz[childtet[children,1]]-xyz[childtet[children,0]],np.cross(xyz[childtet[children,2]]-xyz[childtet[children,0]],xyz[childtet[children,3]]-xyz[childtet[children,0]])))/6
 afterdensity=expected_child[children]/targetvol
 if np.any(afterdensity>HIGH+5e-10):raise RuntimeError('direct post-transfer density exceeds950')
 unique,first=np.unique(children,return_index=True);udensity=expected_child[unique]/targetvol[first]
 lowerused=float(np.minimum(delta_child,need0).sum());extra=float(np.maximum(delta_child-need0,0).sum())
 if abs((mergedm.sum()+qafter.sum())-pmass.sum())>5e-13*pmass.sum():raise RuntimeError('source total + residual conservation failure')
 out=OUT/'run-001';res={'rows':rows,'parents':parents,'global':globalrows,'children':children,'bary':bary,'mass':mass,'q_after':qafter,'need_after':needafter,'head_after':headafter}
 report={'schema':'numi.human.current406-fat-source-exhaustion-partial-allocation.v1','status':'PASS_SOURCE_ONLY_PARTIAL_SOURCE_EXHAUSTION_BATCH','physical_admission':False,'native_steps':0,'complete_allocation':False,'scope':'Two-stage bounded allocation on existing pinned candidate graph sites: first cover receiver lower deficits, then use remaining 950-bound headroom after the lower-priority pass. Source parent corner M/4 budgets, source total, material, current143 one-hop, exact-source barycentric sites, radius, and upper density remain enforced. This is not complete allocation or physical admission.','inputs':hash_input_paths([graph,greport,parent,nodecsv,tetcsv,target,support,supportrep,baseadd,basemerged,baserep,baseverify,basecentroid,basecentroidrep,basecentroidverify,basecentroidmerged,ownergz,ownerman,Path(__file__)]),'policy':{'site_domain':'all pinned attempt005 sites with remaining site capacity after both prior partial batches','phase1':'remaining child lower deficits first; capped by site, source-corner M/4, child low residual and 950 headroom','phase2':'after phase1, allocate additional mass under remaining source-corner and child upper bounds; positive low residuals are ranked first','flow_row_limit':MAX_NEW_ROWS,'site_row_unique':True,'barycentric_point_source_lane':'compiled source330 referenceWorldPoints/rest_x_m/rest_y_m/rest_z_m','source_moment_contract':'each source parent corner remains bounded by original parent_mass/4; residual is kept explicit','radius_m':RADIUS,'material':'same source/recipient material','region':'current143 authenticated one-hop region set','no_mass_clipping':True},'counts':{'remaining_site_rows_before':int(len(candidate)),'prior_centroid_stage_rows':int(len(centstage['flow_mass_kg'])),'prior_noncentroid_stage_rows':int(len(nonstage['flow_mass_kg'])),'merged_origin1_rows':int(np.count_nonzero(borg==1)),'merged_origin2_rows':int(np.count_nonzero(borg==2)),'prior_site_usage_counted_once':True,'invalid_direct_sites_rejected':len(rejects),'new_flow_rows':int(len(mass)),'distinct_source_parents':int(len(np.unique(parents))),'distinct_children':int(len(unique)),'new_flows_by_stage':{'lower_priority_mass_kg':result['lower_mass'],'source_exhaustion_mass_kg':result['source_mass']},'touched_children_below850_direct_volume':int(np.count_nonzero(udensity<LOW))},'mass_kg':{'added_total':float(mass.sum()),'used_to_satisfy_prior_lower_need':lowerused,'above_prior_lower_need':extra,'source_corner_residual_before':float(q0.sum()),'source_corner_residual_after':float(qafter.sum()),'receiver_lower_deficit_before':float(need0.sum()),'receiver_lower_deficit_after':float(needafter.sum()),'receiver_upper_headroom_before':float(head0.sum()),'receiver_upper_headroom_after':float(headafter.sum()),'source_mass_total_unchanged':float(pmass.sum()),'merged_transferred_mass':float(mergedm.sum()),'source_total_plus_residual':float(mergedm.sum()+qafter.sum())},'checks':{'role_parent_global_map':'PASS','source_tet_material_donor_map':'PASS','source_rest_reference_lane_owner_pinned':'PASS','source_point_barycentric_reconstruction':'PASS','onehop_region':'PASS','same_material':'PASS','strict_radius':'PASS','source_corner_budget':'PASS','receiver_upper_headroom':'PASS','positive_direct_volume':'PASS','direct_hybrid_volume_parity':'PASS','merged_child_mass':'PASS','merged_parent_M4_plus_residual':'PASS','source_mass_total_plus_residual':'PASS','upper_density':'PASS','lower_density_complete':False,'complete_allocation':False},'metrics':{'max_direct_distance_m':float(direct.max(initial=0.0)),'max_selected_graph_distance_error_m':float(np.max(np.abs(D[rows]-direct),initial=0.0)),'max_corner_residual_replay_error_kg':float(np.max(np.abs((pmass[:,None]*.25-mergedcorner)-qafter),initial=0.0)),'min_touched_density_kg_m3':float(np.min(udensity,initial=np.inf)),'max_touched_density_kg_m3':float(np.max(udensity,initial=0.0)),'max_volume_relative_error':float(np.max(np.abs(targetvol-hybrid[children])/targetvol,initial=0.0))},'witnessed_rejected_sites':rejects,'elapsed_seconds':time.time()-T0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
 prop,rep=serialize_proposal(out,res,report)
 mergedout=out/'merged-held-centroid-noncentroid-exhaustion.npz'
 np.savez_compressed(mergedout,flow_parent_role_rows=mergedp,flow_global_source330_rows=mergedg,flow_current_child_rows=mergedc,flow_mass_kg=mergedm,flow_source_barycentric=mergedb,flow_source_point_world_m=mergedx,flow_current_centroid_distance_m=mergedd,flow_origin_code=mergedorigin,base_ledger_flow_rows=mergedbase,parent_source_mass_kg=pmass,parent_corner_residual_after_kg=qafter,child_current_mass_kg=expected_child)
 report['inputs']=hash_input_paths([graph,greport,parent,nodecsv,tetcsv,target,support,supportrep,baseadd,basemerged,baserep,baseverify,basecentroid,basecentroidrep,basecentroidverify,basecentroidmerged,ownergz,ownerman,Path(__file__)])
 # Re-write final report only after both output payload hashes are frozen.
 finalize_report(out,report,[prop,mergedout])
 print(json.dumps({'status':report['status'],'counts':report['counts'],'mass_kg':report['mass_kg'],'metrics':report['metrics'],'outputs':report['outputs'],'report_sha256':sha(out/'producer-report.json')},indent=2))
if __name__=='__main__':
 if '--control' in sys.argv:run_control()
 else:run_actual()

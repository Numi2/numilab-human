from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,sys,time,math
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706';OUT=H/'fat-current-support-533/run001'
VOL=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-D500-I528-comp0-54138-005/fat-D500-I528-comp0-kernel-fan-composed-volume-proposal.npz'
OLDVOL=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
OLDSUPPORT=H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz'
OLDRECEIPT=H/'fat-current-semantic-support-406-root-run002/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
FAT_REPORT=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-D500-I528-comp0-54138-005/report.json'
EXPECTED={VOL:'d899b2789c0d4fc91553a5d46b1bb8d7b40e9c818d6301f6b32a3265976e3703',OLDVOL:'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',OLDSUPPORT:'c16f2ff3cd6e30df365ed1418ca385db58cc76403b57359f3effdd1ac3381efc'}
DERIVED_BASE=0x80000000;LOCAL_FACES=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def exact_keys(x):
 x=np.ascontiguousarray(x)
 if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3:raise ValueError('expected F32 Nx3 point key')
 return x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).reshape(-1)
def main():
 start=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in EXPECTED.items():
  if sha(p)!=h:raise ValueError(f'frozen input hash mismatch: {p}')
 oldrec=json.loads(OLDRECEIPT.read_text())
 if oldrec.get('physical_admission') is not False or oldrec.get('status')!='PASS_SOURCE_ONLY_COMPLETE_CURRENT_FAT_SEMANTIC_SUPPORT' or oldrec.get('output',{}).get('sha256')!=sha(OLDSUPPORT):raise ValueError('406 support receipt/output mismatch')
 OUT.mkdir(parents=True)
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 reg=imp('regFat533',OWNER);parent,index,mi,pm=reg.load_parent_current(BASE)
 with np.load(VOL,allow_pickle=False)as z:vol={k:z[k]for k in z.files}
 with np.load(OLDVOL,allow_pickle=False)as z:oldvol={k:z[k]for k in z.files}
 with np.load(OLDSUPPORT,allow_pickle=False)as z:old={k:z[k]for k in z.files}
 pts=np.asarray(vol['points_world_m'],np.float32);tets=np.asarray(vol['tetrahedra'],np.int64);mats=np.asarray(vol['material_indices'],np.int32)
 if pts.shape!=(253712,3) or tets.shape!=(798672,4) or not np.all(mats==16):raise ValueError('Fat d899 geometry shape/material mismatch')
 if not np.array_equal(old['group_tuples'],np.array(sorted({tuple(int(parent[k][r])for k in ('materials','donors','regions'))for r in np.flatnonzero(parent['materials']==16)}),np.int32)):raise ValueError('406 source group tuple domain mismatch')
 oldpts=np.asarray(oldvol['points_world_m'],np.float32);oldkeys=exact_keys(oldpts);curkeys=exact_keys(pts)
 if len(np.unique(oldkeys))!=len(oldkeys)or len(np.unique(curkeys))!=len(curkeys):raise ValueError('duplicate exact F32 node coordinates')
 oord=np.argsort(oldkeys,kind='stable');skeys=oldkeys[oord];pos=np.searchsorted(skeys,curkeys);valid=pos<len(skeys);valid &= skeys[np.minimum(pos,len(skeys)-1)]==curkeys
 current_to_old=np.full(len(pts),-1,np.int64);current_to_old[valid]=oord[pos[valid]]
 n=len(pts);groups=np.asarray(old['group_tuples'],np.int32);node_maps=[{}for _ in range(n)];chosen=np.full(n,-1,np.int32);bary=np.zeros((n,4),np.float64);resid=np.zeros((n,3),np.float64);dist=np.full(n,np.nan,np.float64);origin=np.zeros(n,np.uint8)
 old_go=old['node_group_offsets'];old_gids=old['node_group_ids'];old_po=old['node_group_parent_tet_offsets'];old_pr=old['node_group_parent_tet_rows']
 old_chosen=old['node_parent_chosen_tet_row'];old_bary=old['node_parent_barycentric'];old_resid=old['node_parent_residual_world_m'];old_dist=old['node_parent_registration_distance_m'];old_origin=old['node_origin_kind']
 for i in np.flatnonzero(valid):
  oi=int(current_to_old[i]);chosen[i]=old_chosen[oi];bary[i]=old_bary[oi];resid[i]=old_resid[oi];dist[i]=old_dist[oi];origin[i]=old_origin[oi]
  for s in range(int(old_go[oi]),int(old_go[oi+1])):
   gid=int(old_gids[s]);rows=np.asarray(old_pr[int(old_po[s]):int(old_po[s+1])],np.int32)
   node_maps[i][gid]=rows
 # Query only exact-current points missing from the authenticated 406 cache.
 missing=np.flatnonzero(~valid);query_witness=[]
 for j,i in enumerate(missing):
  q=reg.closest_same_material(parent,index,mi,16,pts[i].astype(np.float64),.02,1.0e-8)
  ties=sorted(set(map(int,q['ties'])))
  if int(q['tet']) not in ties or not ties:raise ValueError(f'new Fat node {i} lacks chosen/tied source parent')
  for tr in ties:
   if int(parent['materials'][tr])!=16:raise ValueError(f'new node {i} source support crosses material')
   gid=tuple(int(parent[k][tr])for k in ('materials','donors','regions'));gidrow=int(np.searchsorted(groups,np.asarray(gid,dtype=np.int32),axis=0)) if False else -1
   try:gidrow=next(k for k,row in enumerate(groups)if tuple(map(int,row))==gid)
   except StopIteration:raise ValueError(f'new node {i} source tuple missing from group registry')
   node_maps[i].setdefault(gidrow,set()).add(tr)
  node_maps[i]={gid:np.asarray(sorted(rows),np.int32)for gid,rows in node_maps[i].items()}
  chosen[i]=int(q['tet']);bary[i]=q['bary'];resid[i]=q['residual'];dist[i]=float(q['distance']);origin[i]=2
  if len(query_witness)<32:query_witness.append({'node_row':int(i),'chosen_parent_row':int(q['tet']),'tie_count':len(ties),'distance_m':float(q['distance'])})
  if j and j%200==0:print(json.dumps({'stage':'query_new_points','done':j,'total':len(missing)}),flush=True)
 for i in np.flatnonzero(valid):node_maps[i]={gid:np.asarray(rows,np.int32)for gid,rows in node_maps[i].items()}
 if any(not s for s in node_maps):raise ValueError('uncovered current Fat node')
 # Build stable-face unions from exact source330 parent rows and retain a fully aligned node-group CSR.
 for i,m in enumerate(node_maps):
  for gid,rows in list(m.items()):
   if not len(rows) or np.any(rows[1:]<=rows[:-1]):raise ValueError(f'node parent support invalid node {i}')
 node_go=[0];node_ids=[];node_po=[0];node_pr=[];node_fo=[0];node_fr=[]
 for m in node_maps:
  for gid in sorted(m):
   rows=m[gid];faces=np.unique(np.asarray(parent['faces'])[rows]).astype(np.uint32)
   node_ids.append(gid);node_pr.extend(rows.tolist());node_po.append(len(node_pr));node_fr.extend(faces.tolist());node_fo.append(len(node_fr))
  node_go.append(len(node_ids))
 # P1 integration, full group/parent/stable-face unions.
 cell_go=[0];cell_ids=[];cell_num=[];cell_den=[];cpo=[0];cpr=[];cfo=[0];cfr=[];selected=np.full(len(tets),-1,np.int32);status=np.full(len(tets),2,np.uint8);denom=np.zeros(len(tets),np.int64);winner=np.zeros(len(tets),np.int64);resolved=[];resolved_gid=[];scalar_face=[];parent_po=[0];parent_rows=[];face_po=[0];face_rows=[]
 for ci,tet in enumerate(tets):
  corners=[node_maps[int(v)]for v in tet];scale=1
  for m in corners:scale=int(np.lcm(scale,len(m)))
  scores=Counter()
  for m in corners:
   share=scale//len(m)
   for gid in m:scores[gid]+=share
  den=4*scale;denom[ci]=den;top=max(scores.values());wins=sorted(g for g,v in scores.items()if v==top)
  if len(wins)==1:
   gid=wins[0];selected[ci]=gid;status[ci]=1;winner[ci]=scores[gid];resolved.append(ci);resolved_gid.append(gid)
   pr=np.unique(np.concatenate([m[gid]for m in corners if gid in m])).astype(np.int32);fr=np.unique(np.asarray(parent['faces'])[pr]).astype(np.uint32)
   if not len(pr)or not len(fr):raise ValueError(f'unique max missing parent/face support child {ci}')
   parent_rows.extend(pr.tolist());parent_po.append(len(parent_rows));face_rows.extend(fr.tolist());face_po.append(len(face_rows));scalar_face.append(int(fr[0])if len(fr)==1 else DERIVED_BASE+gid)
  for gid in sorted(scores):
   cell_ids.append(gid);cell_num.append(scores[gid]);cell_den.append(den)
   pr=np.unique(np.concatenate([m[gid]for m in corners if gid in m])).astype(np.int32);fr=np.unique(np.asarray(parent['faces'])[pr]).astype(np.uint32)
   if not len(pr)or not len(fr):raise ValueError(f'group support missing parent/face child {ci}, group {gid}')
   cpr.extend(pr.tolist());cpo.append(len(cpr));cfr.extend(fr.tolist());cfo.append(len(cfr))
  cell_go.append(len(cell_ids))
  if ci and ci%100000==0:print(json.dumps({'stage':'P1_CELL_SUPPORT','cells_done':ci,'cells_total':len(tets),'group_slots':len(cell_ids)}),flush=True)
 # Authenticate role-specific one-hop region edges through shared source330 faces only.
 st=np.asarray(parent['tets'],np.int64);sm=np.asarray(parent['materials'],np.int32);sr=np.asarray(parent['regions'],np.int32);sf=np.sort(st[:,LOCAL_FACES].reshape(-1,3),axis=1);so=np.repeat(np.arange(len(st),dtype=np.int32),4);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));sf=sf[order];so=so[order];adj=defaultdict(set);proof={}
 for r in np.unique(sr[sm==16]):adj[int(r)].add(int(r))
 for j in np.flatnonzero(np.all(sf[:-1]==sf[1:],axis=1)):
  x,y=int(so[j]),int(so[j+1])
  if sm[x]!=16 or sm[y]!=16:continue
  r,s=int(sr[x]),int(sr[y])
  if r==s:continue
  adj[r].add(s);adj[s].add(r);proof.setdefault(tuple(sorted((r,s))),{'source_parent_tet_rows':[x,y],'shared_source_node_rows':sf[j].tolist()})
 factor=int(sr.max())+1;child=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(np.asarray(cell_go,np.int64)));direct=np.unique(child*factor+groups[np.asarray(cell_ids,np.int32),2]);dc=direct//factor;dr=direct%factor;chunks=[]
 for r in np.unique(dr):
  rows=dc[dr==r];near=np.asarray(sorted(adj[int(r)]),np.int64);chunks.append((rows[:,None]*factor+near[None,:]).reshape(-1))
 expanded=np.unique(np.concatenate(chunks));ec=expanded//factor;er=expanded%factor;lengths=np.bincount(ec,minlength=len(tets));admiss_off=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
 if np.any(lengths==0)or not np.all(np.isin(direct,expanded)):raise ValueError('one-hop support omitted direct Fat group')
 base_arrays={'resolved_child_tet_rows':np.asarray(resolved,np.int32),'resolved_group_ids':np.asarray(resolved_gid,np.int32),'stable_source_face_id':np.asarray(scalar_face,np.uint32),'parent_source_tet_offsets':np.asarray(parent_po,np.int64),'parent_source_tet_rows':np.asarray(parent_rows,np.int32),'stable_source_face_offsets':np.asarray(face_po,np.int64),'stable_source_face_rows':np.asarray(face_rows,np.uint32),'unresolved_child_tet_rows':np.flatnonzero(selected<0).astype(np.int32),'unresolved_reason_codes':np.ones(int(np.count_nonzero(selected<0)),np.int16),'cell_selected_group_ids':selected,'cell_semantic_status':status,'cell_p1_score_denominator':denom,'cell_p1_winner_score_numerator':winner,'cell_group_offsets':np.asarray(cell_go,np.int64),'cell_group_ids':np.asarray(cell_ids,np.int32),'cell_group_p1_score_numerators':np.asarray(cell_num,np.int64),'cell_group_p1_score_denominators':np.asarray(cell_den,np.int64),'cell_group_parent_tet_offsets':np.asarray(cpo,np.int64),'cell_group_parent_tet_rows':np.asarray(cpr,np.int32),'cell_group_stable_face_offsets':np.asarray(cfo,np.int64),'cell_group_stable_face_rows':np.asarray(cfr,np.uint32),'node_group_offsets':np.asarray(node_go,np.int64),'node_group_ids':np.asarray(node_ids,np.int32),'node_group_parent_tet_offsets':np.asarray(node_po,np.int64),'node_group_parent_tet_rows':np.asarray(node_pr,np.int32),'node_origin_kind':origin,'node_parent_registration_distance_m':dist,'group_tuples':groups,'node_group_stable_face_offsets':np.asarray(node_fo,np.int64),'node_group_stable_face_rows':np.asarray(node_fr,np.uint32),'geometry_sha256':np.array(sha(VOL)),'node_support_406_sha256':np.array(sha(OLDSUPPORT))}
 base_path=OUT/'fat-current-semantic-support-d899-p1.npz';np.savez_compressed(base_path,**base_arrays);base_sha=sha(base_path)
 final_arrays=dict(base_arrays);final_arrays['cell_admissible_region_offsets']=admiss_off;final_arrays['cell_admissible_region_ids']=er.astype(np.int32);final_arrays['admissibility_source_support_sha256']=np.array(base_sha);final_arrays['admissibility_max_source_region_hops']=np.array(1,np.int32)
 final_path=OUT/'fat-current-semantic-support-d899-p1-143.npz';np.savez_compressed(final_path,**final_arrays)
 with np.load(final_path,allow_pickle=False)as z:
  if str(z['admissibility_source_support_sha256'].item())!=base_sha:raise ValueError('143 base P1 hash mismatch')
  for k,v in final_arrays.items():
   if not np.array_equal(z[k],v):raise ValueError(f'final support roundtrip mismatch {k}')
 report={'schema':'numi.human.current-fat-p1-semantic-onehop-support.v1','status':'PASS_SOURCE_ONLY_CURRENT_FAT_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{'current_geometry':{'path':str(VOL),'sha256':sha(VOL)},'fat_geometry_parent406':{'path':str(OLDVOL),'sha256':sha(OLDVOL)},'fat_semantic_parent406':{'path':str(OLDSUPPORT),'sha256':sha(OLDSUPPORT)},'fat_semantic_parent406_receipt':{'path':str(OLDRECEIPT),'sha256':sha(OLDRECEIPT)},'source330':{'mass_path':str(BASE),'sha256':sha(BASE),'parent_node_sha256':pm['nodeSHA256'],'parent_labels_sha256':pm['labelsSHA256'],'parent_tets_sha256':pm['tetsSHA256']},'semantic_owner':{'path':str(OWNER),'sha256':sha(OWNER)},'fat_compose_report':{'path':str(FAT_REPORT),'sha256':sha(FAT_REPORT)}},'counts':{'nodes':n,'cells':len(tets),'source_groups':len(groups),'reused_exact_points':int(valid.sum()),'queried_new_points':len(missing),'retired_old_points':int(len(oldpts)-valid.sum()),'P1_group_slots':len(cell_ids),'unique_max_cells':len(resolved),'P1_tie_cells':int(np.count_nonzero(selected<0)),'direct_cell_region_slots':len(direct),'admissible_cell_region_slots':len(er),'max_admissible_regions_per_cell':int(lengths.max()),'same_material_shared_face_region_edges':len(proof)},'query_policy':{'same_material':16,'max_surface_support_distance_m':0.02,'tie_tolerance_m':1e-8,'mass_transport_bound_m':0.019353,'mass_transport_not_evaluated':True},'new_point_query_witnesses_first32':query_witness,'outputs':{'base_p1':{'path':str(base_path),'sha256':base_sha,'size_bytes':base_path.stat().st_size},'p1_plus_authenticated_143':{'path':str(final_path),'sha256':sha(final_path),'size_bytes':final_path.stat().st_size,'admissibility_source_support_sha256':base_sha,'max_source_region_hops':1}},'elapsed_s':time.monotonic()-start,'limitations':['This is the d899 D500/I528 geometry target and must be regenerated if the final current mesh changes.','P1 groups are support preferences; full cell semantic CSR and source donor ledger remain separate.','No mass allocation, native installation or physical admission.']}
 receipt=OUT/'receipt.json';receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'counts':report['counts'],'base_p1_sha256':base_sha,'p1_143_sha256':sha(final_path),'receipt_path':str(receipt),'receipt_sha256':sha(receipt),'elapsed_s':report['elapsed_s']},sort_keys=True),flush=True)
if __name__=='__main__':main()

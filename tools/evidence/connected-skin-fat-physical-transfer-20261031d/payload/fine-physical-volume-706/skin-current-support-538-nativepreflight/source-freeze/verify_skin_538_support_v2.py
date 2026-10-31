from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706';OUT=H/'skin-current-support-538-nativepreflight/run005'
VOL=H/'skin-canonical-fan-chain-538-root/skin-current-O530-I528-fan-canonical-volume-proposal.npz';OLDVOL=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz';CACHE=H/'skin-node-support-cache-531-nativepreflight/run001/skin-node-source330-support.npz'
BASEP1=H/'skin-current-support-538-nativepreflight/run003/skin-current-semantic-support-538-p1.npz';SUPPORT=H/'skin-current-support-538-nativepreflight/run003/skin-current-semantic-support-538-p1-143.npz';RECEIPT=H/'skin-current-support-538-nativepreflight/run003/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv';OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
PIN={VOL:'402ae29e83a8b8122e14518f0f1266f7da096a3f8cbd8827ea6ee6f0ff7d74ea',OLDVOL:'aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe',CACHE:'00148df66d7e928edd3490a285fe028cafaa014f0fd43a057a7f961b32edac59',BASEP1:'e7d7f9b61477a9fe694d35918394e3cec938e6d999e53185aa52227b584dd2f0',SUPPORT:'6289f3e322edad81f20a9fba97f64ce21ac68593dc4315774feaacb4de1b7959'}
LF=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def key(x):
 x=np.ascontiguousarray(x)
 if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3:raise ValueError('point key dtype/shape')
 return x.view(np.dtype((np.void,12))).reshape(-1)
def main():
 t0=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in PIN.items():
  if sha(p)!=h:raise ValueError('pin mismatch '+str(p))
 rec=json.loads(RECEIPT.read_text())
 if rec.get('physical_admission') is not False or rec.get('status')!='PASS_SOURCE_ONLY_CURRENT_SKIN_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT' or rec['outputs']['p1_143']['sha256']!=sha(SUPPORT):raise ValueError('producer receipt mismatch')
 OUT.mkdir()
 with np.load(VOL,allow_pickle=False)as z:v={k:z[k]for k in z.files}
 with np.load(OLDVOL,allow_pickle=False)as z:ov={k:z[k]for k in z.files}
 with np.load(CACHE,allow_pickle=False)as z:c={k:z[k]for k in z.files}
 with np.load(BASEP1,allow_pickle=False)as z:b={k:z[k]for k in z.files}
 with np.load(SUPPORT,allow_pickle=False)as z:a={k:z[k]for k in z.files}
 pts=np.asarray(v['points_world_m'],np.float32);tets=np.asarray(v['tetrahedra'],np.int64);mat=np.asarray(v['material_indices'],np.int32);oldpts=np.asarray(ov['points_world_m'],np.float32)
 if not np.all(mat==15)or str(b['geometry_sha256'].item())!=sha(VOL):raise ValueError('geometry/material binding mismatch')
 ok=key(oldpts);ck=key(pts)
 if len(np.unique(ok))!=len(ok)or len(np.unique(ck))!=len(ck):raise ValueError('duplicate F32 point')
 order=np.argsort(ok,kind='stable');sk=ok[order];pos=np.searchsorted(sk,ck);valid=pos<len(sk);valid &= sk[np.minimum(pos,len(sk)-1)]==ck;oldrow=np.full(len(pts),-1,np.int64);oldrow[valid]=order[pos[valid]]
 if int(valid.sum())!=210268 or int((~valid).sum())!=30:raise ValueError('unexpected prefix/new-node domain')
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'));reg=imp('regSkin538Independent',OWNER);par,index,mi,pm=reg.load_parent_current(BASE)
 groups=np.asarray(b['group_tuples'],np.int32);expected_groups=np.array(sorted({tuple(int(par[k][r])for k in ('materials','donors','regions'))for r in np.flatnonzero(par['materials']==15)}),np.int32)
 if not np.array_equal(groups,expected_groups):raise ValueError('full source group registry mismatch')
 gmap={tuple(map(int,row)):i for i,row in enumerate(groups)};oldgroups=np.asarray(c['group_tuples'],np.int32);oldid_to_new=np.asarray([gmap[tuple(map(int,row))]for row in oldgroups],np.int32)
 maps=[{}for _ in pts];dist=np.full(len(pts),np.nan);origin=np.zeros(len(pts),np.uint8);co=c['node_group_offsets'];ci=c['node_group_ids'];cpo=c['node_group_parent_tet_offsets'];cpr=c['node_group_parent_tet_rows']
 for i in np.flatnonzero(valid):
  o=int(oldrow[i]);dist[i]=c['node_parent_registration_distance_m'][o];origin[i]=c['node_origin_kind'][o]
  for s in range(int(co[o]),int(co[o+1])):maps[i][int(oldid_to_new[int(ci[s])])]=np.asarray(cpr[int(cpo[s]):int(cpo[s+1])],np.int32)
 faces=np.asarray(par['faces'],np.uint32);newrows=np.flatnonzero(~valid);witness=[]
 for i in newrows:
  q=reg.closest_same_material(par,index,mi,15,pts[i].astype(np.float64),.02,1e-8);ties=sorted(set(map(int,q['ties'])))
  if not ties or int(q['tet']) not in ties:raise ValueError(f'new node query missing chosen/tie {i}')
  for row in ties:
   if int(par['materials'][row])!=15:raise ValueError(f'new node crosses material {i}')
   tup=tuple(int(par[k][row])for k in ('materials','donors','regions'));g=gmap.get(tup)
   if g is None:raise ValueError(f'new node tuple absent {i}/{tup}')
   maps[i].setdefault(g,set()).add(row)
  maps[i]={g:np.asarray(sorted(rows),np.int32)for g,rows in maps[i].items()};dist[i]=float(q['distance']);origin[i]=2
  witness.append({'node':int(i),'chosen_parent':int(q['tet']),'tie_rows':ties,'distance_m':float(q['distance'])})
 if any(not m for m in maps):raise ValueError('current node uncovered')
 # Replay per-node tuple parent/face unions and compare complete output CSR.
 ngo=[0];ngi=[];npo=[0];npr=[];nfo=[0];nfr=[]
 for i,m in enumerate(maps):
  if int(a['node_origin_kind'][i])!=int(origin[i])or not np.allclose(a['node_parent_registration_distance_m'][i],dist[i],rtol=0,atol=0,equal_nan=True):raise ValueError(f'node origin/distance mismatch {i}')
  for g in sorted(m):
   rows=np.asarray(m[g],np.int32)
   if not len(rows)or np.any(rows[1:]<=rows[:-1]):raise ValueError(f'node parent rows invalid {i}/{g}')
   if np.any(par['materials'][rows]!=groups[g,0])or np.any(par['donors'][rows]!=groups[g,1])or np.any(par['regions'][rows]!=groups[g,2]):raise ValueError(f'node tuple identity error {i}/{g}')
   ngi.append(g);npr.extend(rows.tolist());npo.append(len(npr));nfr.extend(np.unique(faces[rows]).tolist());nfo.append(len(nfr))
  ngo.append(len(ngi))
 for k,x in [('node_group_offsets',ngo),('node_group_ids',ngi),('node_group_parent_tet_offsets',npo),('node_group_parent_tet_rows',npr),('node_group_stable_face_offsets',nfo),('node_group_stable_face_rows',nfr)]:
  if not np.array_equal(b[k],np.asarray(x,dtype=b[k].dtype)):raise ValueError('node CSR mismatch '+k)
 # Replay exact P1 fractions, all group ties, and parent/face unions.
 cgo=b['cell_group_offsets'];cids=b['cell_group_ids'];nums=b['cell_group_p1_score_numerators'];dens=b['cell_group_p1_score_denominators'];cpof=b['cell_group_parent_tet_offsets'];cprs=b['cell_group_parent_tet_rows'];cfo=b['cell_group_stable_face_offsets'];cfr=b['cell_group_stable_face_rows'];selected=b['cell_selected_group_ids'];status=b['cell_semantic_status'];den=b['cell_p1_score_denominator'];win=b['cell_p1_winner_score_numerator'];resolved=[]
 for ci,tet in enumerate(tets):
  corners=[maps[int(n)]for n in tet];scale=1
  for m in corners:scale=int(np.lcm(scale,len(m)))
  scores=defaultdict(int)
  for m in corners:
   for g in m:scores[g]+=scale//len(m)
  gids=sorted(scores);lo,hi=map(int,cgo[ci:ci+2]);d=4*scale;ns=np.asarray([scores[g]for g in gids],np.int64)
  if gids!=list(map(int,cids[lo:hi]))or not np.array_equal(ns,nums[lo:hi])or not np.all(dens[lo:hi]==d)or int(den[ci])!=d or int(ns.sum())!=d:raise ValueError(f'P1 fraction mismatch cell {ci}')
  top=max(ns);winners=[gids[j]for j,z in enumerate(ns)if z==top];sel=winners[0]if len(winners)==1 else -1
  if int(selected[ci])!=sel or int(status[ci])!=(1 if sel>=0 else 2)or int(win[ci])!=(top if sel>=0 else 0):raise ValueError(f'P1 tie resolution mismatch cell {ci}')
  for j,g in enumerate(gids):
   prows=sorted(set(int(r)for m in corners if g in m for r in m[g]));frows=sorted(set(int(f)for f in faces[np.asarray(prows,np.int64)]));s=lo+j
   if not np.array_equal(cprs[int(cpof[s]):int(cpof[s+1])],np.asarray(prows,np.int32))or not np.array_equal(cfr[int(cfo[s]):int(cfo[s+1])],np.asarray(frows,np.uint32)):raise ValueError(f'P1 ancestry mismatch cell {ci}/{g}')
  if sel>=0:resolved.append(ci)
  if ci and ci%150000==0:print(json.dumps({'stage':'independent-P1','cells':ci,'total':len(tets)}),flush=True)
 if not np.array_equal(b['resolved_child_tet_rows'],np.asarray(resolved,np.int32)):raise ValueError('unique-max row map mismatch')
 # Rebuild exact one-hop adjacency from same-material shared faces in source330.
 st=np.asarray(par['tets'],np.int64);sm=np.asarray(par['materials'],np.int32);sr=np.asarray(par['regions'],np.int32);tri=np.sort(st[:,LF].reshape(-1,3),axis=1);own=np.repeat(np.arange(len(st),dtype=np.int32),4);ix=np.lexsort((tri[:,2],tri[:,1],tri[:,0]));tri=tri[ix];own=own[ix];adj=defaultdict(set);edges=set()
 for r in np.unique(sr[sm==15]):adj[int(r)].add(int(r))
 j=0
 while j<len(tri):
  k=j+1
  while k<len(tri)and np.array_equal(tri[k],tri[j]):k+=1
  if k-j>2:raise ValueError(f'nonmanifold source face {tri[j].tolist()}')
  if k-j==2:
   x,y=int(own[j]),int(own[j+1])
   if sm[x]==15 and sm[y]==15:
    xreg,yreg=int(sr[x]),int(sr[y]);adj[xreg].add(yreg);adj[yreg].add(xreg)
    if xreg!=yreg:edges.add(tuple(sorted((xreg,yreg))))
  j=k
 factor=int(sr.max())+1;cc=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(cgo));direct=np.unique(cc*factor+groups[cids,2]);dc=direct//factor;dr=direct%factor;parts=[]
 for r in np.unique(dr):parts.append((dc[dr==r,None]*factor+np.asarray(sorted(adj[int(r)]),np.int64)[None,:]).reshape(-1))
 expected=np.unique(np.concatenate(parts));ao=a['cell_admissible_region_offsets'];ar=a['cell_admissible_region_ids'];actual=np.unique(np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(ao))*factor+ar)
 if not np.array_equal(actual,expected):raise ValueError(f'143 mismatch missing={len(np.setdiff1d(expected,actual))} extra={len(np.setdiff1d(actual,expected))}')
 report={'schema':'numi.human.current-skin-p1-143-independent-replay.v1','status':'PASS_INDEPENDENT_CURRENT_SKIN_P1_AND_ONE_HOP_REPLAY','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{str(p):sha(p)for p in (VOL,OLDVOL,CACHE,BASEP1,SUPPORT,RECEIPT,OWNER,BASE)},'checks':{'unchanged_prefix_exact_F32_join':True,'new_node_queries_replayed':True,'source_node_group_tuple_parent_and_face_unions':True,'all_P1_integer_fractions_sum_one':True,'full_tied_group_parent_and_face_unions':True,'unique_max_scalar_resolution':True,'same_material_shared_face_adjacency':True,'exact_one_hop_143_closure':True},'counts':{'nodes':len(pts),'cells':len(tets),'reused_exact_prefix':int(valid.sum()),'requeried_new_nodes':len(newrows),'full_source_groups':len(groups),'P1_group_slots':len(cids),'unique_max_cells':len(resolved),'ties':int(np.count_nonzero(selected<0)),'admissible_region_slots':len(ar),'same_material_source_region_edges':len(edges)},'new_node_witnesses':witness,'elapsed_s':time.monotonic()-t0}
 rp=OUT/'independent-replay.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');report['replay_sha256']=sha(rp);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()

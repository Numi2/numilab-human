from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706';OUT=H/'fat-current-support-533/run002'
VOL=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-D500-I528-comp0-54138-005/fat-D500-I528-comp0-kernel-fan-composed-volume-proposal.npz'
OLDVOL=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz';OLDSUPPORT=H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz';OLDRECEIPT=H/'fat-current-semantic-support-406-root-run002/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv';OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
BASE_P1=H/'fat-current-support-533/run001/fat-current-semantic-support-d899-p1.npz';SUPPORT=H/'fat-current-support-533/run001/fat-current-semantic-support-d899-p1-143.npz';RECEIPT=H/'fat-current-support-533/run001/receipt.json'
PIN={VOL:'d899b2789c0d4fc91553a5d46b1bb8d7b40e9c818d6301f6b32a3265976e3703',OLDVOL:'5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211',OLDSUPPORT:'c16f2ff3cd6e30df365ed1418ca385db58cc76403b57359f3effdd1ac3381efc',BASE_P1:'a3c6467ce7f08fda3a7d95d056f91c98de3b55356dbecc217ea3125201c5f00f',SUPPORT:'65c2478e7ebcfec0c99af7eeb083a64ed8a5648f37b831a288af2f1694d6104d'}
LF=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def key(x):
 x=np.ascontiguousarray(x)
 if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3:raise ValueError('bad point key')
 return x.view(np.dtype((np.void,12))).reshape(-1)
def main():
 t0=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in PIN.items():
  if sha(p)!=h:raise ValueError('pin mismatch '+str(p))
 rec=json.loads(RECEIPT.read_text())
 if rec.get('physical_admission') is not False or rec.get('status')!='PASS_SOURCE_ONLY_CURRENT_FAT_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT' or rec['outputs']['p1_plus_authenticated_143']['sha256']!=sha(SUPPORT):raise ValueError('builder receipt mismatch')
 OUT.mkdir()
 with np.load(VOL,allow_pickle=False)as z:v={k:z[k]for k in z.files}
 with np.load(OLDVOL,allow_pickle=False)as z:ov={k:z[k]for k in z.files}
 with np.load(OLDSUPPORT,allow_pickle=False)as z:o={k:z[k]for k in z.files}
 with np.load(BASE_P1,allow_pickle=False)as z:b={k:z[k]for k in z.files}
 with np.load(SUPPORT,allow_pickle=False)as z:a={k:z[k]for k in z.files}
 if str(a['admissibility_source_support_sha256'].item())!=sha(BASE_P1):raise ValueError('bad 143 base binding')
 pts=np.asarray(v['points_world_m']);tets=np.asarray(v['tetrahedra'],np.int64);mats=np.asarray(v['material_indices'],np.int32);oldpts=np.asarray(ov['points_world_m'])
 if not np.all(mats==16) or not np.array_equal(b['geometry_sha256'],np.array(sha(VOL))):raise ValueError('geometry/material pin mismatch')
 ok=key(oldpts);ck=key(pts)
 if len(np.unique(ok))!=len(ok)or len(np.unique(ck))!=len(ck):raise ValueError('duplicate F32 coordinates')
 ord=np.argsort(ok,kind='stable');sk=ok[ord];p=np.searchsorted(sk,ck);valid=p<len(sk);valid &= sk[np.minimum(p,len(sk)-1)]==ck
 oldrow=np.full(len(pts),-1,np.int64);oldrow[valid]=ord[p[valid]]
 if int(valid.sum())!=252938:raise ValueError('unexpected exact-cache overlap')
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'));reg=imp('regFat533Replay',OWNER);par,idx,mi,pm=reg.load_parent_current(BASE)
 groups=np.asarray(b['group_tuples'],np.int32);expected_groups=np.array(sorted({tuple(int(par[k][r])for k in ('materials','donors','regions'))for r in np.flatnonzero(par['materials']==16)}),np.int32)
 if not np.array_equal(groups,expected_groups):raise ValueError('source group registry mismatch')
 group_index={tuple(map(int,row)):i for i,row in enumerate(groups)}
 maps=[{}for _ in pts];ch=np.full(len(pts),-1,np.int32);bb=np.zeros((len(pts),4));rr=np.zeros((len(pts),3));dd=np.full(len(pts),np.nan);origin=np.zeros(len(pts),np.uint8)
 og=o['node_group_offsets'];oi=o['node_group_ids'];op=o['node_group_parent_tet_offsets'];opr=o['node_group_parent_tet_rows']
 for i in np.flatnonzero(valid):
  q=int(oldrow[i]);ch[i]=o['node_parent_chosen_tet_row'][q];bb[i]=o['node_parent_barycentric'][q];rr[i]=o['node_parent_residual_world_m'][q];dd[i]=o['node_parent_registration_distance_m'][q];origin[i]=o['node_origin_kind'][q]
  for s in range(int(og[q]),int(og[q+1])):maps[i][int(oi[s])]=np.asarray(opr[int(op[s]):int(op[s+1])],np.int32)
 newrows=np.flatnonzero(~valid);witness=[]
 for j,i in enumerate(newrows):
  q=reg.closest_same_material(par,idx,mi,16,pts[i].astype(np.float64),.02,1e-8);ties=sorted(set(map(int,q['ties'])))
  if not ties or int(q['tet']) not in ties:raise ValueError(f'new point lacks deterministic tie {i}')
  for tr in ties:
   if int(par['materials'][tr])!=16:raise ValueError(f'new point crosses material {i}')
   tup=tuple(int(par[k][tr])for k in ('materials','donors','regions'));g=group_index.get(tup)
   if g is None:raise ValueError(f'new point group absent {i}')
   maps[i].setdefault(g,set()).add(tr)
  maps[i]={g:np.asarray(sorted(x),np.int32)for g,x in maps[i].items()};ch[i]=int(q['tet']);bb[i]=q['bary'];rr[i]=q['residual'];dd[i]=float(q['distance']);origin[i]=2
  if len(witness)<32:witness.append({'node':int(i),'chosen_parent':int(q['tet']),'ties':ties,'distance_m':float(q['distance'])})
  if j and j%256==0:print(json.dumps({'stage':'new-node-requery','done':j,'total':len(newrows)}),flush=True)
 if any(not m for m in maps):raise ValueError('uncovered current point')
 # Verify source tuple identity and exact node-support CSR against independently rebuilt maps.
 go=[0];ids=[];po=[0];pr=[];fo=[0];fr=[];faces=np.asarray(par['faces'],np.uint32)
 for i,m in enumerate(maps):
  if int(a['node_origin_kind'][i])!=int(origin[i])or int(a['node_parent_chosen_tet_row'][i])!=int(ch[i]):raise ValueError(f'node provenance mismatch {i}')
  if not np.array_equal(a['node_parent_barycentric'][i],bb[i])or not np.array_equal(a['node_parent_residual_world_m'][i],rr[i])or not np.allclose(a['node_parent_registration_distance_m'][i],dd[i],rtol=0,atol=0,equal_nan=True):raise ValueError(f'node projection mismatch {i}')
  for g in sorted(m):
   rows=np.asarray(m[g],np.int32)
   if not len(rows)or np.any(rows[1:]<=rows[:-1]):raise ValueError(f'node support rows invalid {i}/{g}')
   if np.any(par['materials'][rows]!=groups[g,0])or np.any(par['donors'][rows]!=groups[g,1])or np.any(par['regions'][rows]!=groups[g,2]):raise ValueError(f'node tuple mismatch {i}/{g}')
   ids.append(g);pr.extend(rows.tolist());po.append(len(pr));fr.extend(np.unique(faces[rows]).tolist());fo.append(len(fr))
  go.append(len(ids))
 for k,x in [('node_group_offsets',go),('node_group_ids',ids),('node_group_parent_tet_offsets',po),('node_group_parent_tet_rows',pr),('node_group_stable_face_offsets',fo),('node_group_stable_face_rows',fr)]:
  if not np.array_equal(b[k],np.asarray(x,dtype=b[k].dtype)):raise ValueError('node CSR mismatch '+k)
 # Independently replay exact integer P1 fractions, all ties, full parent/face unions.
 cg=b['cell_group_offsets'];ciids=b['cell_group_ids'];cn=b['cell_group_p1_score_numerators'];cdens=b['cell_group_p1_score_denominators'];cpo=b['cell_group_parent_tet_offsets'];cpr=b['cell_group_parent_tet_rows'];cfo=b['cell_group_stable_face_offsets'];cfr=b['cell_group_stable_face_rows'];sel=b['cell_selected_group_ids'];status=b['cell_semantic_status'];dnom=b['cell_p1_score_denominator'];win=b['cell_p1_winner_score_numerator'];resolved=[]
 for ci,tet in enumerate(tets):
  corners=[maps[int(n)]for n in tet];scale=1
  for m in corners:scale=int(np.lcm(scale,len(m)))
  sc=defaultdict(int)
  for m in corners:
   for g in m:sc[g]+=scale//len(m)
  gs=sorted(sc);lo,hi=map(int,cg[ci:ci+2]);den=4*scale;nu=np.asarray([sc[g]for g in gs],np.int64)
  if gs!=list(map(int,ciids[lo:hi]))or not np.array_equal(nu,cn[lo:hi])or not np.all(cdens[lo:hi]==den)or int(dnom[ci])!=den or int(nu.sum())!=den:raise ValueError(f'P1 score mismatch child {ci}')
  top=max(nu);wins=[gs[j]for j,x in enumerate(nu)if x==top];chosen=wins[0]if len(wins)==1 else -1
  if int(sel[ci])!=chosen or int(status[ci])!=(1 if chosen>=0 else 2)or int(win[ci])!=(top if chosen>=0 else 0):raise ValueError(f'P1 tie resolution mismatch child {ci}')
  for j,g in enumerate(gs):
   pp=sorted(set(int(r)for m in corners if g in m for r in m[g]));ff=sorted(set(int(f)for r in pp for f in faces[r]));s=lo+j
   if not np.array_equal(cpr[int(cpo[s]):int(cpo[s+1])],np.asarray(pp,np.int32))or not np.array_equal(cfr[int(cfo[s]):int(cfo[s+1])],np.asarray(ff,np.uint32)):raise ValueError(f'P1 ancestry mismatch child {ci} group {g}')
  if chosen>=0:resolved.append(ci)
  if ci and ci%200000==0:print(json.dumps({'stage':'P1-independent-replay','cells':ci,'total':len(tets)}),flush=True)
 if not np.array_equal(b['resolved_child_tet_rows'],np.asarray(resolved,np.int32)):raise ValueError('unique-max cell row list mismatch')
 # Rebuild same-material source shared-face graph and exact one-hop cell closure.
 st=np.asarray(par['tets'],np.int64);sm=np.asarray(par['materials'],np.int32);sr=np.asarray(par['regions'],np.int32);sf=np.sort(st[:,LF].reshape(-1,3),axis=1);owner=np.repeat(np.arange(len(st),dtype=np.int32),4);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));sf=sf[order];owner=owner[order];adj=defaultdict(set)
 for r in np.unique(sr[sm==16]):adj[int(r)].add(int(r))
 edgepairs=set();k=0
 while k<len(sf):
  j=k+1
  while j<len(sf)and np.array_equal(sf[k],sf[j]):j+=1
  if j-k>2:raise ValueError(f'source face nonmanifold {sf[k].tolist()}')
  if j-k==2:
   x,y=int(owner[k]),int(owner[k+1])
   if sm[x]==16 and sm[y]==16:
    r,s=int(sr[x]),int(sr[y]);adj[r].add(s);adj[s].add(r)
    if r!=s:edgepairs.add(tuple(sorted((r,s))))
  k=j
 factor=int(sr.max())+1;direct=np.unique(np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(cg))*factor+groups[ciids,2]);dc=direct//factor;dr=direct%factor;chunks=[]
 for r in np.unique(dr):chunks.append((dc[dr==r,None]*factor+np.asarray(sorted(adj[int(r)]),np.int64)[None,:]).reshape(-1))
 expected=np.unique(np.concatenate(chunks));ao=a['cell_admissible_region_offsets'];ar=a['cell_admissible_region_ids'];actual=np.unique(np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(ao))*factor+ar)
 if not np.array_equal(actual,expected):raise ValueError(f'143 closure mismatch miss={len(np.setdiff1d(expected,actual))} extra={len(np.setdiff1d(actual,expected))}')
 report={'schema':'numi.human.current-fat-p1-143-independent-replay.v1','status':'PASS_INDEPENDENT_CURRENT_FAT_P1_AND_ONE_HOP_REPLAY','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{str(p):sha(p)for p in (VOL,OLDVOL,OLDSUPPORT,BASE_P1,SUPPORT,RECEIPT,OWNER,BASE)},'checks':{'exact_F32_cache_join':True,'new_points_requeried_with_source_owner':True,'node_group_parent_tuple_and_face_unions':True,'full_cell_P1_integer_scores':True,'all_P1_group_ties_and_parent_face_unions':True,'same_material_shared_face_region_graph':True,'143_one_hop_closure':True},'counts':{'nodes':len(pts),'cells':len(tets),'reused_exact_points':int(valid.sum()),'requeried_points':len(newrows),'P1_group_slots':len(ciids),'unique_max_cells':len(resolved),'tie_cells':int(np.count_nonzero(sel<0)),'admissible_region_slots':len(ar),'same_material_region_edges':len(edgepairs)},'new_point_witnesses_first32':witness,'elapsed_s':time.monotonic()-t0}
 p=OUT/'independent-replay.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');report['replay_receipt_sha256']=sha(p);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()

from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json, sys, time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706'
OUT=H/'skin-current-support-532/run007'
VOL=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz'
CACHE=H/'skin-node-support-cache-531-nativepreflight/run001/skin-node-source330-support.npz'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
BASE_P1=H/'skin-current-support-532/run004/skin-current-semantic-support-531-p1.npz'
SUPPORT=H/'skin-current-support-532/run004/skin-current-semantic-support-531-p1-143.npz'
RECEIPT=H/'skin-current-support-532/run004/receipt.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 start=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir()
 with np.load(VOL,allow_pickle=False)as z:vol={k:z[k]for k in z.files}
 with np.load(CACHE,allow_pickle=False)as z:cache={k:z[k]for k in z.files}
 with np.load(SUPPORT,allow_pickle=False)as z:a={k:z[k]for k in z.files}
 if str(a['admissibility_source_support_sha256'].item())!=sha(BASE_P1):raise ValueError('143 does not bind exact base P1 artifact')
 pts=vol['points_world_m'];tets=vol['tetrahedra'];mats=vol['material_indices'];groups=a['group_tuples'];nnode=len(pts);nc=len(tets)
 if not np.array_equal(pts.view('<u4'),cache['points_world_m_f32'].view('<u4')):raise ValueError('cache geometry bytes differ')
 if len(a['cell_group_offsets'])!=nc+1 or len(a['cell_admissible_region_offsets'])!=nc+1:raise ValueError('child CSR row-domain mismatch')
 # Independent source330 identity replay for cached node-level parent and face ties.
 sys.path.insert(0,str(H/"downstream-003-current-typed-228/native-domain-217-source-005"))
 reg=imp('regSkin532Independent',OWNER);parent,_,_,pm=reg.load_parent_current(BASE)
 g=cache['group_tuples'];ngoff=cache['node_group_offsets'];ngid=cache['node_group_ids'];npof=cache['node_group_parent_tet_offsets'];npr=cache['node_group_parent_tet_rows'];nfof=cache['node_group_stable_face_offsets'];nfr=cache['node_group_stable_face_rows']
 assert len(ngoff)==nnode+1 and len(npof)==len(ngid)+1 and len(nfof)==len(ngid)+1
 for s,gid in enumerate(ngid):
  lo,hi=map(int,npof[s:s+2]);flo,fhi=map(int,nfof[s:s+2]);rows=npr[lo:hi]
  if not len(rows) or np.any(rows[1:]<=rows[:-1]):raise ValueError(f'node parent support not sorted unique at slot {s}')
  if np.any(parent['materials'][rows]!=g[gid,0]) or np.any(parent['donors'][rows]!=g[gid,1]) or np.any(parent['regions'][rows]!=g[gid,2]):raise ValueError(f'node parent semantic tuple mismatch slot {s}')
  faces=np.unique(np.asarray(parent['faces'])[rows]).astype(np.uint32)
  if not np.array_equal(faces,nfr[flo:fhi]):raise ValueError(f'node stable-face union mismatch slot {s}')
 # Replay each cell P1 score, tie class, full group parent CSR and stable-face CSR.
 go=a['cell_group_offsets'];cids=a['cell_group_ids'];nums=a['cell_group_p1_score_numerators'];dens=a['cell_group_p1_score_denominators'];cpo=a['cell_group_parent_tet_offsets'];cpr=a['cell_group_parent_tet_rows'];cfo=a['cell_group_stable_face_offsets'];cfr=a['cell_group_stable_face_rows'];selected=a['cell_selected_group_ids'];status=a['cell_semantic_status'];cd=a['cell_p1_score_denominator'];win=a['cell_p1_winner_score_numerator']
 exp_r=[]; exp_g=[]
 for ci,tet in enumerate(tets):
  node_slots=[]
  for node in tet:
   node_slots.append([(int(ngid[s]),npr[int(npof[s]):int(npof[s+1])],nfr[int(nfof[s]):int(nfof[s+1])]) for s in range(int(ngoff[int(node)]),int(ngoff[int(node)+1]))])
  scale=1
  for slots in node_slots:scale=int(np.lcm(scale,len(slots)))
  score=defaultdict(int)
  for slots in node_slots:
   share=scale//len(slots)
   for gid,_,_ in slots:score[gid]+=share
  gids=sorted(score);lo,hi=map(int,go[ci:ci+2]);
  if gids!=list(map(int,cids[lo:hi])):raise ValueError(f'P1 group IDs differ at child {ci}')
  expected_nums=np.asarray([score[x] for x in gids],dtype=np.int64);den=4*scale
  if not np.array_equal(expected_nums,nums[lo:hi]) or not np.all(dens[lo:hi]==den) or int(cd[ci])!=den or int(expected_nums.sum())!=den:raise ValueError(f'P1 exact fractions fail at child {ci}')
  top=max(expected_nums);wins=[gids[k]for k,x in enumerate(expected_nums)if x==top]
  expect_sel=wins[0] if len(wins)==1 else -1
  if int(selected[ci])!=expect_sel or int(status[ci])!=(1 if expect_sel>=0 else 2) or int(win[ci])!=(top if expect_sel>=0 else 0):raise ValueError(f'P1 scalar resolution/tie mismatch at child {ci}')
  for j,gid in enumerate(gids):
   ps=sorted(set(int(r) for gid2,pr,_ in node_slots if gid2==gid for r in pr));fs=sorted(set(int(f) for gid2,_,fr in node_slots if gid2==gid for f in fr));slot=lo+j
   if not np.array_equal(cpr[int(cpo[slot]):int(cpo[slot+1])],np.asarray(ps,np.int32)):raise ValueError(f'full parent union differs child {ci} group {gid}')
   if not np.array_equal(cfr[int(cfo[slot]):int(cfo[slot+1])],np.asarray(fs,np.uint32)):raise ValueError(f'full stable-face union differs child {ci} group {gid}')
  if ci and ci%150000==0:print(json.dumps({'stage':'independent-P1-replay','children':ci,'total':nc}),flush=True)
 # Independently reconstruct one-hop edges from source330 shared faces, same material only.
 src=np.asarray(parent['tets'],dtype=np.int64);sm=np.asarray(parent['materials'],dtype=np.int32);sr=np.asarray(parent['regions'],dtype=np.int32);local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
 sf=np.sort(src[:,local].reshape(-1,3),axis=1);owner=np.repeat(np.arange(len(src),dtype=np.int32),4);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));sf=sf[order];owner=owner[order]
 adj=defaultdict(set)
 for r in np.unique(sr[sm==15]):adj[int(r)].add(int(r))
 edge_pairs=np.flatnonzero(np.all(sf[:-1]==sf[1:],axis=1))
 for j in edge_pairs:
  x,y=int(owner[j]),int(owner[j+1])
  if sm[x]==15 and sm[y]==15:
   r,s=int(sr[x]),int(sr[y]);adj[r].add(s);adj[s].add(r)
 factor=int(sr.max())+1
 child=np.repeat(np.arange(nc,dtype=np.int64),np.diff(go));direct=np.unique(child*factor+groups[cids,2]);dc=direct//factor;dr=direct%factor;chunks=[]
 for r in np.unique(dr):
  rows=dc[dr==r];near=np.asarray(sorted(adj[int(r)]),np.int64);chunks.append((rows[:,None]*factor+near[None,:]).reshape(-1))
 expected=np.unique(np.concatenate(chunks));actualoff=a['cell_admissible_region_offsets'];actualid=a['cell_admissible_region_ids'];actualchild=np.repeat(np.arange(nc,dtype=np.int64),np.diff(actualoff));actual=np.unique(actualchild*factor+actualid)
 if not np.array_equal(actual,expected):
  miss=np.setdiff1d(expected,actual);extra=np.setdiff1d(actual,expected);raise ValueError(f'143 closure mismatch missing={len(miss)} extra={len(extra)} examples={miss[:8].tolist()}/{extra[:8].tolist()}')
 if not np.all(np.isin(direct,actual)):raise ValueError('143 direct P1 region not preserved')
 report={'schema':'numi.human.current-skin-p1-143-independent-replay.v1','status':'PASS_INDEPENDENT_CURRENT_SKIN_P1_AND_ONE_HOP_REPLAY','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{str(p):sha(p)for p in (VOL,CACHE,BASE_P1,SUPPORT,RECEIPT,OWNER,BASE)},'checks':{'geometry_exact_points':True,'source330_node_parent_group_identity':True,'source330_node_stable_face_union':True,'all_cell_p1_fractions_sum_exactly_one':True,'full_p1_parent_and_face_unions_replayed':True,'full_group_ties_retained':True,'scalar_only_unique_max_resolution':True,'same_material_shared_face_region_graph_replayed':True,'143_exact_one_hop_region_closure':True,'direct_regions_preserved':True},'counts':{'nodes':nnode,'cells':nc,'source_groups':len(groups),'cell_group_slots':len(cids),'resolved_unique_max_cells':int(np.count_nonzero(selected>=0)),'p1_tie_cells':int(np.count_nonzero(selected<0)),'admissible_region_slots':len(actualid),'source_adjacency_pairs':len(edge_pairs),'same_material_skin_region_edges':len(adj)},'elapsed_s':time.monotonic()-start}
 rpath=OUT/'independent-replay.json';rpath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');report['replay_receipt_sha256']=sha(rpath);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()

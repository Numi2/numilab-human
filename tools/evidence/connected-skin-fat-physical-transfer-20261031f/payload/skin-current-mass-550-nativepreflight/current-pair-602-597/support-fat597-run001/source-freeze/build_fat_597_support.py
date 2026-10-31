from pathlib import Path
from collections import Counter, defaultdict
import hashlib, importlib.util, json, sys, time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209'); H=ROOT/'fine-physical-volume-706'
OUT=H/'skin-current-mass-550-nativepreflight/current-pair-602-597/support-fat597-run001'
VOL=H/'current-paired-fixed-boundary-composition-597-root/fat-current-connected-volume-proposal.npz'
VOLREC=H/'current-paired-fixed-boundary-composition-597-root/receipt.json'
OLDVOL=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
OLD=H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz'
OLDREC=H/'fat-current-semantic-support-406-root-run002/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
EXPECTED={VOL:'6677a4bbc53637617cebaf9e74d7b0dfd9c775d0d5e64d7050bf4741f8e35054', OLD:'c16f2ff3cd6e30df365ed1418ca385db58cc76403b57359f3effdd1ac3381efc', OLDREC:'16667c9d91e78a230f41b433093939d05ef9b6da4eca66407041c0d9708123f7'}
DERIVED_BASE=0x80000000; LF=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def key(x):
 x=np.ascontiguousarray(x); return x.view(np.dtype((np.void,12))).reshape(-1)
def main():
 t0=time.monotonic()
 if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(OUT)
 OUT.mkdir(parents=True,exist_ok=True)
 for p,h in EXPECTED.items():
  if sha(p)!=h: raise ValueError('pinned input SHA mismatch '+str(p))
 oldrec=json.loads(OLDREC.read_text())
 if oldrec.get('output',{}).get('sha256')!=sha(OLD): raise ValueError('406 support receipt/output mismatch')
 if oldrec.get('inputs',{}).get(str(OLDVOL))!='5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211': raise ValueError('406 support is not bound to expected geometry')
 with np.load(VOL,allow_pickle=False) as z: v={k:z[k] for k in z.files}
 with np.load(OLDVOL,allow_pickle=False) as z: oldpts=z['points_world_m'].copy()
 with np.load(OLD,allow_pickle=False) as z: c={k:z[k] for k in z.files}
 pts=np.asarray(v['points_world_m'],np.float32); tets=np.asarray(v['tetrahedra'],np.int64); mats=np.asarray(v['material_indices'],np.int32)
 if pts.shape!=(253712,3) or tets.shape!=(789327,4) or not np.all(mats==16): raise ValueError('current Fat597 shape/material mismatch')
 if len(np.unique(key(pts)))!=len(pts) or len(np.unique(key(oldpts)))!=len(oldpts): raise ValueError('duplicate exact F32 points')
 order=np.argsort(key(oldpts),kind='stable'); sk=key(oldpts)[order]; pos=np.searchsorted(sk,key(pts)); valid=pos<len(sk); valid &= sk[np.minimum(pos,len(sk)-1)]==key(pts)
 oldrow=np.full(len(pts),-1,np.int64); oldrow[valid]=order[pos[valid]]
 if int(valid.sum())!=252938: raise ValueError(f'expected 252938 exact cached points, got {int(valid.sum())}')
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 reg=imp('regFat597',OWNER); parent,index,mi,pm=reg.load_parent_current(BASE)
 groups=np.array(sorted({tuple(int(parent[k][r]) for k in ('materials','donors','regions')) for r in np.flatnonzero(parent['materials']==16)}),np.int32)
 if not np.array_equal(c['group_tuples'],groups): raise ValueError('406 cached group registry differs from current source330 role registry')
 group_lookup={tuple(map(int,row)):i for i,row in enumerate(groups)}
 maps=[{} for _ in pts]; dist=np.full(len(pts),np.nan); origin=np.zeros(len(pts),np.uint8)
 co=c['node_group_offsets']; ci=c['node_group_ids']; cpo=c['node_group_parent_tet_offsets']; cpr=c['node_group_parent_tet_rows']
 for i in np.flatnonzero(valid):
  o=int(oldrow[i]); dist[i]=c['node_parent_registration_distance_m'][o]; origin[i]=c['node_origin_kind'][o]
  for s in range(int(co[o]),int(co[o+1])): maps[i][int(ci[s])]=np.asarray(cpr[int(cpo[s]):int(cpo[s+1])],np.int32)
 new=np.flatnonzero(~valid); witnesses=[]
 for j,i0 in enumerate(new):
  i=int(i0); q=reg.closest_same_material(parent,index,mi,16,pts[i].astype(np.float64),.02,1e-8); ties=sorted(set(map(int,q['ties'])))
  if not ties or int(q['tet']) not in ties: raise ValueError(f'new Fat point {i} lacks selected tie')
  for row in ties:
   if int(parent['materials'][row])!=16: raise ValueError(f'new point crosses material at {i}')
   tup=tuple(int(parent[k][row]) for k in ('materials','donors','regions'))
   if tup not in group_lookup: raise ValueError(f'new point source tuple absent at {i}: {tup}')
   maps[i].setdefault(group_lookup[tup],set()).add(row)
  maps[i]={g:np.asarray(sorted(rows),np.int32) for g,rows in maps[i].items()}; dist[i]=float(q['distance']); origin[i]=2
  if len(witnesses)<40: witnesses.append({'node_row':i,'chosen_parent_row':int(q['tet']),'tie_rows':ties,'distance_m':float(q['distance'])})
  if (j+1)%128==0 or j+1==len(new): print(json.dumps({'stage':'230_query_new_points','done':j+1,'total':len(new),'elapsed_s':time.monotonic()-t0}),flush=True)
 sf=np.asarray(parent['faces'],np.uint32); ngo=[0]; ngi=[]; npo=[0]; npr=[]; nfo=[0]; nfr=[]
 for i,m in enumerate(maps):
  if not m: raise ValueError(f'uncovered Fat point {i}')
  for gid in sorted(m):
   rows=np.asarray(m[gid],np.int32); faces=np.unique(sf[rows]); ngi.append(gid); npr.extend(rows.tolist()); npo.append(len(npr)); nfr.extend(faces.tolist()); nfo.append(len(nfr))
  ngo.append(len(ngi))
 ncell=len(tets); cgo=[0]; cids=[]; cnum=[]; cden=[]; cpof=[0]; cprs=[]; cfo=[0]; cfr=[]
 selected=np.full(ncell,-1,np.int32); status=np.full(ncell,2,np.uint8); den=np.zeros(ncell,np.int64); win=np.zeros(ncell,np.int64); resolved=[]; rgids=[]; scalar=[]; upo=[0]; upr=[]; ufo=[0]; ufr=[]
 for ci,tet in enumerate(tets):
  corners=[maps[int(v)] for v in tet]; scale=1
  for m in corners: scale=int(np.lcm(scale,len(m)))
  sc=Counter()
  for m in corners:
   for gid in m: sc[gid]+=scale//len(m)
  d=4*scale; den[ci]=d; top=max(sc.values()); wins=sorted(g for g,x in sc.items() if x==top)
  if len(wins)==1:
   gid=wins[0]; selected[ci]=gid; status[ci]=1; win[ci]=sc[gid]; resolved.append(ci); rgids.append(gid)
   pr=np.unique(np.concatenate([m[gid] for m in corners if gid in m])).astype(np.int32); fr=np.unique(sf[pr]).astype(np.uint32)
   if not len(pr) or not len(fr): raise ValueError(f'P1 unique max lacks ancestry cell {ci}')
   upr.extend(pr.tolist()); upo.append(len(upr)); ufr.extend(fr.tolist()); ufo.append(len(ufr)); scalar.append(int(fr[0]) if len(fr)==1 else DERIVED_BASE+gid)
  for gid in sorted(sc):
   cids.append(gid); cnum.append(sc[gid]); cden.append(d)
   pr=np.unique(np.concatenate([m[gid] for m in corners if gid in m])).astype(np.int32); fr=np.unique(sf[pr]).astype(np.uint32)
   cprs.extend(pr.tolist()); cpof.append(len(cprs)); cfr.extend(fr.tolist()); cfo.append(len(cfr))
  cgo.append(len(cids))
  if ci and ci%100000==0: print(json.dumps({'stage':'p1_cells','done':ci,'total':ncell,'slots':len(cids),'elapsed_s':time.monotonic()-t0}),flush=True)
 st=np.asarray(parent['tets'],np.int64); sm=np.asarray(parent['materials'],np.int32); sr=np.asarray(parent['regions'],np.int32)
 tri=np.sort(st[:,LF].reshape(-1,3),axis=1); owner=np.repeat(np.arange(len(st),dtype=np.int32),4); ix=np.lexsort((tri[:,2],tri[:,1],tri[:,0])); tri=tri[ix]; owner=owner[ix]
 adj=defaultdict(set); edge=set()
 for r in np.unique(sr[sm==16]): adj[int(r)].add(int(r))
 j=0
 while j<len(tri):
  k=j+1
  while k<len(tri) and np.array_equal(tri[k],tri[j]): k+=1
  if k-j>2: raise ValueError(f'nonmanifold source face {tri[j].tolist()}')
  if k-j==2:
   x,y=int(owner[j]),int(owner[j+1])
   if sm[x]==16 and sm[y]==16:
    a,b=int(sr[x]),int(sr[y]); adj[a].add(b); adj[b].add(a)
    if a!=b: edge.add(tuple(sorted((a,b))))
  j=k
 factor=int(sr.max())+1; cc=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(cgo)); gids=np.asarray(cids,np.int32)
 direct=np.unique(cc*factor+groups[gids,2]); dc=direct//factor; dr=direct%factor; pieces=[]
 for r in np.unique(dr): pieces.append((dc[dr==r,None]*factor+np.asarray(sorted(adj[int(r)]),np.int64)[None,:]).reshape(-1))
 expanded=np.unique(np.concatenate(pieces).reshape(-1)); ec=expanded//factor; er=expanded%factor; length=np.bincount(ec,minlength=ncell); aoff=np.r_[0,np.cumsum(length)].astype(np.int64)
 base={'resolved_child_tet_rows':np.asarray(resolved,np.int32),'resolved_group_ids':np.asarray(rgids,np.int32),'stable_source_face_id':np.asarray(scalar,np.uint32),'parent_source_tet_offsets':np.asarray(upo,np.int64),'parent_source_tet_rows':np.asarray(upr,np.int32),'stable_source_face_offsets':np.asarray(ufo,np.int64),'stable_source_face_rows':np.asarray(ufr,np.uint32),'unresolved_child_tet_rows':np.flatnonzero(selected<0).astype(np.int32),'unresolved_reason_codes':np.ones(int(np.count_nonzero(selected<0)),np.int16),'cell_selected_group_ids':selected,'cell_semantic_status':status,'cell_p1_score_denominator':den,'cell_p1_winner_score_numerator':win,'cell_group_offsets':np.asarray(cgo,np.int64),'cell_group_ids':np.asarray(cids,np.int32),'cell_group_p1_score_numerators':np.asarray(cnum,np.int64),'cell_group_p1_score_denominators':np.asarray(cden,np.int64),'cell_group_parent_tet_offsets':np.asarray(cpof,np.int64),'cell_group_parent_tet_rows':np.asarray(cprs,np.int32),'cell_group_stable_face_offsets':np.asarray(cfo,np.int64),'cell_group_stable_face_rows':np.asarray(cfr,np.uint32),'node_group_offsets':np.asarray(ngo,np.int64),'node_group_ids':np.asarray(ngi,np.int32),'node_group_parent_tet_offsets':np.asarray(npo,np.int64),'node_group_parent_tet_rows':np.asarray(npr,np.int32),'node_group_stable_face_offsets':np.asarray(nfo,np.int64),'node_group_stable_face_rows':np.asarray(nfr,np.uint32),'node_origin_kind':origin,'node_parent_registration_distance_m':dist,'group_tuples':groups,'geometry_sha256':np.array(sha(VOL)),'node_support_406_sha256':np.array(sha(OLD))}
 basepath=OUT/'fat-current-semantic-support-597-p1.npz'; np.savez_compressed(basepath,**base); bsha=sha(basepath)
 full=dict(base); full['cell_admissible_region_offsets']=aoff; full['cell_admissible_region_ids']=er.astype(np.int32); full['admissibility_source_support_sha256']=np.array(bsha); full['admissibility_max_source_region_hops']=np.array(1,np.int32)
 fullpath=OUT/'fat-current-semantic-support-597-p1-143.npz'; np.savez_compressed(fullpath,**full)
 if int(np.count_nonzero(length==0)) or not np.all(np.isin(direct,expanded)): raise ValueError('143 region closure incomplete')
 with np.load(fullpath,allow_pickle=False) as z:
  for k,x in full.items():
   if not np.array_equal(z[k],x): raise ValueError('NPZ roundtrip '+k)
 report={'schema':'numi.human.current-fat-p1-semantic-onehop-support.v1','status':'PASS_SOURCE_ONLY_CURRENT_FAT_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{'geometry':{'path':str(VOL),'sha256':sha(VOL)},'geometry_receipt':{'path':str(VOLREC),'sha256':sha(VOLREC)},'prior_geometry_406':{'path':str(OLDVOL),'sha256':sha(OLDVOL)},'prior_support_406':{'path':str(OLD),'sha256':sha(OLD)},'prior_support_receipt_406':{'path':str(OLDREC),'sha256':sha(OLDREC)},'source330_mass':{'path':str(BASE),'sha256':sha(BASE),'node_sha256':pm['nodeSHA256'],'labels_sha256':pm['labelsSHA256'],'tet_sha256':pm['tetsSHA256']},'semantic_owner':{'path':str(OWNER),'sha256':sha(OWNER)}},'counts':{'nodes':len(pts),'cells':ncell,'reused_exact_points':int(valid.sum()),'queried_new_points':len(new),'source_groups':len(groups),'p1_group_slots':len(cids),'unique_max_cells':len(resolved),'tie_cells':int(np.count_nonzero(selected<0)),'direct_region_slots':len(direct),'admissible_region_slots':len(er),'max_regions_per_cell':int(length.max()),'same_material_source_region_edges':len(edge),'unsupported_nodes':0,'unsupported_cells':int(np.count_nonzero(length==0))},'new_point_witnesses':witnesses,'outputs':{'base_p1':{'path':str(basepath),'sha256':bsha},'p1_143':{'path':str(fullpath),'sha256':sha(fullpath),'admissibility_source_support_sha256':bsha}},'limits':{'same_material':16,'node_projection_distance_m_max':.02,'tie_tolerance_m':1e-8,'mass_region_policy':'separate143authenticated1hop','mass_transport_radius_m':.019353,'mass_transport_NOT_evaluated':True},'limitations':['Source-only current Fat597 support; no mass flow, native steps or physical admission.','Exact F32 point matches reuse 406 support; unmatched current coordinates were queried with the pinned 230 owner.','Regenerate if target cell topology or point coordinates change.'],'elapsed_s':time.monotonic()-t0}
 rp=OUT/'receipt.json'; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'counts':report['counts'],'base':bsha,'support':sha(fullpath),'receipt':str(rp),'receipt_sha256':sha(rp),'elapsed_s':report['elapsed_s']},sort_keys=True),flush=True)
if __name__=='__main__': main()

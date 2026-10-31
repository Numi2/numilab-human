from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706';OUT=H/'skin-current-mass-550-nativepreflight/current-pair-602-597/support-skin602-run002/output'
VOL=H/'skin-interior-fixed-boundary-quotient-602-root/skin-interior-quotient-volume-proposal.npz';OLDVOL=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz';CACHE=H/'skin-node-support-cache-531-nativepreflight/run001/skin-node-source330-support.npz';CACHE_REC=H/'skin-node-support-cache-531-nativepreflight/run001/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv';OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py';VOLREC=H/'skin-interior-fixed-boundary-quotient-602-root/receipt.json'
PINS={VOL:'1d6e1070b5da38e4f9e57ad3ad1fe9aa9ae488cacc04ee22df8eeb2da79341f0',OLDVOL:'aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe',CACHE:'00148df66d7e928edd3490a285fe028cafaa014f0fd43a057a7f961b32edac59'}
DERIVED_BASE=0x80000000;LF=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def key(x):
 x=np.ascontiguousarray(x);return x.view(np.dtype((np.void,12))).reshape(-1)
def main():
 t0=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError('input SHA mismatch '+str(p))
 if sha(CACHE_REC)!='e87c472705556f4d956c8f97692a82bb303f2d916a850d3210c3da0a21419124':raise ValueError('cache receipt SHA mismatch')
 OUT.mkdir(parents=True)
 with np.load(VOL,allow_pickle=False)as z:v={k:z[k]for k in z.files}
 with np.load(OLDVOL,allow_pickle=False)as z:ov={k:z[k]for k in z.files}
 with np.load(CACHE,allow_pickle=False)as z:c={k:z[k]for k in z.files}
 pts=np.asarray(v['points_world_m'],np.float32);tets=np.asarray(v['tetrahedra'],np.int64);mats=np.asarray(v['material_indices'],np.int32);oldpts=np.asarray(ov['points_world_m'],np.float32)
 if pts.shape!=(210298,3)or tets.shape!=(729624,4)or not np.all(mats==15):raise ValueError('current Skin602 shape/material mismatch')
 if not np.array_equal(c['points_world_m_f32'].view('<u4'),oldpts.view('<u4')):raise ValueError('cache prefix != Skin531 geometry')
 if len(np.unique(key(pts)))!=len(pts)or len(np.unique(key(oldpts)))!=len(oldpts):raise ValueError('duplicate exact F32 points')
 order=np.argsort(key(oldpts),kind='stable');sk=key(oldpts)[order];pos=np.searchsorted(sk,key(pts));valid=pos<len(sk);valid &= sk[np.minimum(pos,len(sk)-1)]==key(pts);oldrow=np.full(len(pts),-1,np.int64);oldrow[valid]=order[pos[valid]]
 if int(valid.sum())!=210268:raise ValueError(f'expected 210268 unchanged points, got {int(valid.sum())}')
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'));reg=imp('regSkin602',OWNER);parent,index,mi,pm=reg.load_parent_current(BASE);oldgroups=np.asarray(c['group_tuples'],np.int32)
 groups=np.array(sorted({tuple(int(parent[k][r])for k in ('materials','donors','regions'))for r in np.flatnonzero(parent['materials']==15)}),np.int32)
 group_lookup={tuple(map(int,row)):i for i,row in enumerate(groups)}
 try: old_group_remap=np.asarray([group_lookup[tuple(map(int,row))] for row in oldgroups],np.int32)
 except KeyError as e:raise ValueError(f'cached tuple absent from complete source330 role registry: {e}')
 maps=[{}for _ in pts];dist=np.full(len(pts),np.nan);origin=np.zeros(len(pts),np.uint8)
 co=c['node_group_offsets'];ci=c['node_group_ids'];cpo=c['node_group_parent_tet_offsets'];cpr=c['node_group_parent_tet_rows']
 for i in np.flatnonzero(valid):
  o=int(oldrow[i]);dist[i]=c['node_parent_registration_distance_m'][o];origin[i]=c['node_origin_kind'][o]
  for s in range(int(co[o]),int(co[o+1])):maps[i][int(old_group_remap[int(ci[s])])]=np.asarray(cpr[int(cpo[s]):int(cpo[s+1])],np.int32)
 new=np.flatnonzero(~valid);witness=[]
 for j,i in enumerate(new):
  q=reg.closest_same_material(parent,index,mi,15,pts[i].astype(np.float64),.02,1e-8);ties=sorted(set(map(int,q['ties'])))
  if not ties or int(q['tet']) not in ties:raise ValueError(f'new Skin point {i} lacks chosen tie')
  for row in ties:
   tup=tuple(int(parent[k][row])for k in ('materials','donors','regions'));gid=group_lookup.get(tup)
   if gid is None:raise ValueError(f'new point source tuple absent {i}')
   if int(parent['materials'][row])!=15:raise ValueError(f'new point crosses Skin material {i}')
   maps[i].setdefault(gid,set()).add(row)
  maps[i]={g:np.asarray(sorted(x),np.int32)for g,x in maps[i].items()};dist[i]=float(q['distance']);origin[i]=2
  if len(witness)<30:witness.append({'node_row':int(i),'chosen_parent_row':int(q['tet']),'tie_rows':ties,'distance_m':float(q['distance'])})
 # Assemble node support/face ancestry from current source330 rows.
 sf=np.asarray(parent['faces'],np.uint32);ngo=[0];ngi=[];npo=[0];npr=[];nfo=[0];nfr=[]
 for i,m in enumerate(maps):
  if not m:raise ValueError(f'uncovered Skin node {i}')
  for gid in sorted(m):
   rows=np.asarray(m[gid],np.int32);faces=np.unique(sf[rows]);ngi.append(gid);npr.extend(rows.tolist());npo.append(len(npr));nfr.extend(faces.tolist());nfo.append(len(nfr))
  ngo.append(len(ngi))
 # Exact integer P1 support and full tied ancestry.
 ncell=len(tets);cgo=[0];cids=[];cnum=[];cden=[];cpof=[0];cprs=[];cfo=[0];cfr=[];selected=np.full(ncell,-1,np.int32);status=np.full(ncell,2,np.uint8);den=np.zeros(ncell,np.int64);win=np.zeros(ncell,np.int64);resolved=[];rgids=[];scalar=[];upo=[0];upr=[];ufo=[0];ufr=[]
 for ci,tet in enumerate(tets):
  corners=[maps[int(v)]for v in tet];scale=1
  for m in corners:scale=int(np.lcm(scale,len(m)))
  sc=Counter()
  for m in corners:
   for gid in m:sc[gid]+=scale//len(m)
  d=4*scale;den[ci]=d;top=max(sc.values());wins=sorted(g for g,v in sc.items()if v==top)
  if len(wins)==1:
   gid=wins[0];selected[ci]=gid;status[ci]=1;win[ci]=sc[gid];resolved.append(ci);rgids.append(gid)
   pr=np.unique(np.concatenate([m[gid]for m in corners if gid in m])).astype(np.int32);fr=np.unique(sf[pr]).astype(np.uint32)
   if not len(pr)or not len(fr):raise ValueError(f'P1 unique max lacks ancestry cell {ci}')
   upr.extend(pr.tolist());upo.append(len(upr));ufr.extend(fr.tolist());ufo.append(len(ufr));scalar.append(int(fr[0])if len(fr)==1 else DERIVED_BASE+gid)
  for gid in sorted(sc):
   cids.append(gid);cnum.append(sc[gid]);cden.append(d);pr=np.unique(np.concatenate([m[gid]for m in corners if gid in m])).astype(np.int32);fr=np.unique(sf[pr]).astype(np.uint32)
   cprs.extend(pr.tolist());cpof.append(len(cprs));cfr.extend(fr.tolist());cfo.append(len(cfr))
  cgo.append(len(cids))
  if ci and ci%100000==0:print(json.dumps({'stage':'p1_cells','done':ci,'total':ncell,'slots':len(cids)}),flush=True)
 # Same-material, authenticated source330 shared-face one-hop region closure.
 st=np.asarray(parent['tets'],np.int64);sm=np.asarray(parent['materials'],np.int32);sr=np.asarray(parent['regions'],np.int32);tri=np.sort(st[:,LF].reshape(-1,3),axis=1);owner=np.repeat(np.arange(len(st),dtype=np.int32),4);ix=np.lexsort((tri[:,2],tri[:,1],tri[:,0]));tri=tri[ix];owner=owner[ix];adj=defaultdict(set);edge=set()
 for r in np.unique(sr[sm==15]):adj[int(r)].add(int(r))
 j=0
 while j<len(tri):
  k=j+1
  while k<len(tri)and np.array_equal(tri[k],tri[j]):k+=1
  if k-j>2:raise ValueError(f'nonmanifold source face {tri[j].tolist()}')
  if k-j==2:
   x,y=int(owner[j]),int(owner[j+1])
   if sm[x]==15 and sm[y]==15:
    a,b=int(sr[x]),int(sr[y]);adj[a].add(b);adj[b].add(a)
    if a!=b:edge.add(tuple(sorted((a,b))))
  j=k
 factor=int(sr.max())+1;cc=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(cgo));direct=np.unique(cc*factor+groups[np.asarray(cids,np.int32),2]);dc=direct//factor;dr=direct%factor;pieces=[]
 for r in np.unique(dr):pieces.append((dc[dr==r,None]*factor+np.asarray(sorted(adj[int(r)]),np.int64)[None,:]).reshape(-1))
 expanded=np.unique(np.concatenate(pieces));ec=expanded//factor;er=expanded%factor;length=np.bincount(ec,minlength=ncell);aoff=np.r_[0,np.cumsum(length)].astype(np.int64)
 base={'resolved_child_tet_rows':np.asarray(resolved,np.int32),'resolved_group_ids':np.asarray(rgids,np.int32),'stable_source_face_id':np.asarray(scalar,np.uint32),'parent_source_tet_offsets':np.asarray(upo,np.int64),'parent_source_tet_rows':np.asarray(upr,np.int32),'stable_source_face_offsets':np.asarray(ufo,np.int64),'stable_source_face_rows':np.asarray(ufr,np.uint32),'unresolved_child_tet_rows':np.flatnonzero(selected<0).astype(np.int32),'unresolved_reason_codes':np.ones(int(np.count_nonzero(selected<0)),np.int16),'cell_selected_group_ids':selected,'cell_semantic_status':status,'cell_p1_score_denominator':den,'cell_p1_winner_score_numerator':win,'cell_group_offsets':np.asarray(cgo,np.int64),'cell_group_ids':np.asarray(cids,np.int32),'cell_group_p1_score_numerators':np.asarray(cnum,np.int64),'cell_group_p1_score_denominators':np.asarray(cden,np.int64),'cell_group_parent_tet_offsets':np.asarray(cpof,np.int64),'cell_group_parent_tet_rows':np.asarray(cprs,np.int32),'cell_group_stable_face_offsets':np.asarray(cfo,np.int64),'cell_group_stable_face_rows':np.asarray(cfr,np.uint32),'node_group_offsets':np.asarray(ngo,np.int64),'node_group_ids':np.asarray(ngi,np.int32),'node_group_parent_tet_offsets':np.asarray(npo,np.int64),'node_group_parent_tet_rows':np.asarray(npr,np.int32),'node_group_stable_face_offsets':np.asarray(nfo,np.int64),'node_group_stable_face_rows':np.asarray(nfr,np.uint32),'node_origin_kind':origin,'node_parent_registration_distance_m':dist,'group_tuples':groups,'geometry_sha256':np.array(sha(VOL)),'node_support_531_sha256':np.array(sha(CACHE))}
 basepath=OUT/'skin-current-semantic-support-602-p1.npz';np.savez_compressed(basepath,**base);bsha=sha(basepath);full=dict(base);full['cell_admissible_region_offsets']=aoff;full['cell_admissible_region_ids']=er.astype(np.int32);full['admissibility_source_support_sha256']=np.array(bsha);full['admissibility_max_source_region_hops']=np.array(1,np.int32);fullpath=OUT/'skin-current-semantic-support-602-p1-143.npz';np.savez_compressed(fullpath,**full)
 if int(np.count_nonzero(length==0))or not np.all(np.isin(direct,expanded)):raise ValueError('143 region closure incomplete')
 with np.load(fullpath,allow_pickle=False)as z:
  for k,x in full.items():
   if not np.array_equal(z[k],x):raise ValueError('NPZ roundtrip '+k)
 vr=json.loads(VOLREC.read_text());report={'schema':'numi.human.current-skin-p1-semantic-onehop-support.v1','status':'PASS_SOURCE_ONLY_CURRENT_SKIN_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{'geometry':{'path':str(VOL),'sha256':sha(VOL)},'geometry_receipt':{'path':str(VOLREC),'sha256':sha(VOLREC)},'old_geometry_531':{'path':str(OLDVOL),'sha256':sha(OLDVOL)},'old_node_support_531':{'path':str(CACHE),'sha256':sha(CACHE)},'old_node_support_receipt':{'path':str(CACHE_REC),'sha256':sha(CACHE_REC)},'source330_mass':{'path':str(BASE),'sha256':sha(BASE),'node_sha256':pm['nodeSHA256'],'labels_sha256':pm['labelsSHA256'],'tet_sha256':pm['tetsSHA256']},'semantic_owner':{'path':str(OWNER),'sha256':sha(OWNER)}},'counts':{'nodes':len(pts),'cells':ncell,'reused_exact_points':int(valid.sum()),'queried_new_points':len(new),'source_groups':len(groups),'p1_group_slots':len(cids),'unique_max_cells':len(resolved),'tie_cells':int(np.count_nonzero(selected<0)),'direct_region_slots':len(direct),'admissible_region_slots':len(er),'max_regions_per_cell':int(length.max()),'same_material_source_region_edges':len(edge)},'new_point_witnesses':witness,'outputs':{'base_p1':{'path':str(basepath),'sha256':bsha},'p1_143':{'path':str(fullpath),'sha256':sha(fullpath),'admissibility_source_support_sha256':bsha}},'limitations':['Source-only current Skin602 support; no mass flow, native steps or physical admission.','Requires regeneration if current target geometry changes.'],'elapsed_s':time.monotonic()-t0};rp=OUT/'receipt.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'counts':report['counts'],'base':bsha,'support':sha(fullpath),'receipt':str(rp),'receipt_sha256':sha(rp),'elapsed_s':report['elapsed_s']},sort_keys=True),flush=True)
if __name__=='__main__':main()

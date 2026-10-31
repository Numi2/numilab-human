from pathlib import Path
from collections import Counter, defaultdict
import hashlib, importlib.util, json, sys, time
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
H=ROOT/'fine-physical-volume-706'
OUT=H/'skin-current-support-532/run001'
VOL=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz'
VOL_RECEIPT=H/'skin-canonical-outer-O-531-root/receipt.json'
CACHE=H/'skin-node-support-cache-531-nativepreflight/run001/skin-node-source330-support.npz'
CACHE_RECEIPT=H/'skin-node-support-cache-531-nativepreflight/run001/receipt.json'
PARENT=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
EXACT_VOLUME=H/'global-mass-volume-preflight-532/run002/global-density-volume-report.json'
PINS={VOL:'aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe',
      CACHE:'00148df66d7e928edd3490a285fe028cafaa014f0fd43a057a7f961b32edac59',
      EXACT_VOLUME:'b783e88fa0e62d43e4af0cf2604f1fdc3f27070bc8d41b8e718dbceb156cba52'}
DERIVED_BASE=0x80000000
LOCAL_FACES=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],dtype=np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 start=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError(f'input pin mismatch {p}')
 if sha(CACHE_RECEIPT)!='e87c472705556f4d956c8f97692a82bb303f2d916a850d3210c3da0a21419124':
  # Cache receipt is independently checked structurally below; pin it in report.
  raise ValueError("cache receipt pin mismatch")
 OUT.mkdir()
 with np.load(VOL,allow_pickle=False)as z: vol={k:z[k] for k in z.files}
 with np.load(CACHE,allow_pickle=False)as z: cache={k:z[k] for k in z.files}
 pts=np.asarray(vol['points_world_m'],dtype=np.float32);tets=np.asarray(vol['tetrahedra'],dtype=np.int64);mat=np.asarray(vol['material_indices'],dtype=np.int32)
 if pts.shape!=(210268,3) or tets.shape!=(729848,4) or not np.all(mat==15):raise ValueError('current Skin531 geometry shape/material mismatch')
 if not np.array_equal(cache['points_world_m_f32'].view('<u4'),pts.view('<u4')):raise ValueError('node cache point domain/bytes differ from Skin531')
 if cache['group_tuples'].ndim!=2 or cache['group_tuples'].shape[1]!=3 or not np.all(cache['group_tuples'][:,0]==15):raise ValueError('node support is not same-material Skin')
 groups=np.asarray(cache['group_tuples'],dtype=np.int32);ngoff=cache['node_group_offsets'];ngids=cache['node_group_ids'];ngpoff=cache['node_group_parent_tet_offsets'];ngprows=cache['node_group_parent_tet_rows'];ngfoff=cache['node_group_stable_face_offsets'];ngfrows=cache['node_group_stable_face_rows']
 nnode=len(pts);ncell=len(tets);ng=len(groups)
 if len(ngoff)!=nnode+1 or len(ngpoff)!=len(ngids)+1 or len(ngfoff)!=len(ngids)+1:raise ValueError('node support CSR alignment failure')
 # Source owner confirms source330 role domains and stable parent identity.
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 owner=loadmod('regSkin532',OWNER);parent,index,mi,pm=owner.load_parent_current(PARENT)
 if len(parent['materials'])!=28632 or np.any(parent['materials'][np.unique(ngprows)]!=15):raise ValueError('cached Skin parent rows are not material15')
 # Materialized per-node group -> full source parent/stable-face support maps.
 node_maps=[]
 for node in range(nnode):
  m={}
  for s in range(int(ngoff[node]),int(ngoff[node+1])):
   gid=int(ngids[s]);m[gid]=(ngprows[int(ngpoff[s]):int(ngpoff[s+1])],ngfrows[int(ngfoff[s]):int(ngfoff[s+1])])
  if not m:raise ValueError(f'unsupported Skin node {node}')
  node_maps.append(m)
 # P1-integrate all current cells; retain all tied groups and union ancestry.
 cell_go=[0];cell_gid=[];cell_num=[];cell_den=[];cg_po=[0];cg_pr=[];cg_fo=[0];cg_fr=[]
 selected=np.full(ncell,-1,np.int32);status=np.full(ncell,2,np.uint8);denom=np.zeros(ncell,np.int64);winner_num=np.zeros(ncell,np.int64)
 resolved_rows=[];resolved_ids=[];scalar_face=[];parent_po=[0];parent_rows=[];face_po=[0];face_rows=[];reason=Counter();reason_by_row=np.full(ncell,0,np.int16)
 for ci,tet in enumerate(tets):
  corners=[node_maps[int(v)] for v in tet]
  n_groups=[len(x) for x in corners]
  scale=1
  for n in n_groups:scale=np.lcm(scale,n)
  scores=Counter()
  for m in corners:
   share=int(scale//len(m))
   for gid in m:scores[gid]+=share
  d=int(4*scale);denom[ci]=d
  top=max(scores.values());winners=sorted(g for g,v in scores.items() if v==top)
  if len(winners)==1:
   gid=winners[0];selected[ci]=gid;status[ci]=1;winner_num[ci]=int(scores[gid]);resolved_rows.append(ci);resolved_ids.append(gid)
   pr=np.unique(np.concatenate([m[gid][0] for m in corners if gid in m])).astype(np.int32)
   fr=np.unique(np.concatenate([m[gid][1] for m in corners if gid in m])).astype(np.uint32)
   if not len(pr) or not len(fr):raise ValueError(f'P1 winner lacks ancestry at child {ci}')
   parent_rows.extend(pr.tolist());parent_po.append(len(parent_rows));face_rows.extend(fr.tolist());face_po.append(len(face_rows))
   scalar_face.append(int(fr[0]) if len(fr)==1 else DERIVED_BASE+int(gid))
  else:
   reason['integrated_group_tie']+=1;reason_by_row[ci]=1
  for gid in sorted(scores):
   cell_gid.append(int(gid));cell_num.append(int(scores[gid]));cell_den.append(d)
   pr=np.unique(np.concatenate([m[gid][0] for m in corners if gid in m])).astype(np.int32)
   fr=np.unique(np.concatenate([m[gid][1] for m in corners if gid in m])).astype(np.uint32)
   if not len(pr) or not len(fr):raise ValueError(f'group support lacks ancestry at child {ci}, group {gid}')
   cg_pr.extend(pr.tolist());cg_po.append(len(cg_pr));cg_fr.extend(fr.tolist());cg_fo.append(len(cg_fr))
  cell_go.append(len(cell_gid))
  if ci and ci%100000==0:print(json.dumps({'stage':'P1_CELL_SUPPORT','cells_done':ci,'cells_total':ncell,'group_slots':len(cell_gid)}),flush=True)
 # Reuse exact registered source330 topology to authenticate same-material, shared-face one-hop regions.
 t=np.asarray(parent['tets'],dtype=np.int64);materials=np.asarray(parent['materials'],dtype=np.int32);regions=np.asarray(parent['regions'],dtype=np.int32)
 face_rows_src=np.sort(t[:,LOCAL_FACES].reshape(-1,3),axis=1);owners=np.repeat(np.arange(len(t),dtype=np.int32),4)
 order=np.lexsort((face_rows_src[:,2],face_rows_src[:,1],face_rows_src[:,0]));fs=face_rows_src[order];os=owners[order]
 adjacency=defaultdict(set);edgeproof={}
 for r in np.unique(regions[materials==15]):adjacency[int(r)].add(int(r))
 matches=np.flatnonzero(np.all(fs[:-1]==fs[1:],axis=1))
 for j in matches:
  a,b=int(os[j]),int(os[j+1])
  if materials[a]!=15 or materials[b]!=15:continue
  ra,rb=int(regions[a]),int(regions[b])
  if ra==rb:continue
  adjacency[ra].add(rb);adjacency[rb].add(ra)
  edgeproof.setdefault(tuple(sorted((ra,rb))),{'source_parent_tet_rows':[a,b],'shared_source_node_rows':fs[j].tolist()})
 factor=int(regions.max())+1
 child=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(np.asarray(cell_go,dtype=np.int64)))
 direct_ids=np.asarray(cell_gid,dtype=np.int32);direct_regions=groups[direct_ids,2].astype(np.int32)
 direct_codes=np.unique(child*factor+direct_regions)
 direct_child=direct_codes//factor;direct_region=direct_codes%factor
 add=[]
 for region in np.unique(direct_region):
  rows=direct_child[direct_region==region]; neigh=np.asarray(sorted(adjacency[int(region)]),dtype=np.int64)
  add.append((rows[:,None]*factor+neigh[None,:]).reshape(-1))
 expanded=np.unique(np.concatenate(add));admiss_child=expanded//factor;admiss_region=expanded%factor
 lengths=np.bincount(admiss_child,minlength=ncell);admiss_off=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
 if np.any(lengths==0) or not np.all(np.isin(direct_codes,expanded)):raise ValueError('143 region support omitted direct P1 region')
 out=P=OUT/'skin-current-semantic-support-531-p1-143.npz'
 arrays={
  'resolved_child_tet_rows':np.asarray(resolved_rows,np.int32),'resolved_group_ids':np.asarray(resolved_ids,np.int32),'stable_source_face_id':np.asarray(scalar_face,np.uint32),
  'parent_source_tet_offsets':np.asarray(parent_po,np.int64),'parent_source_tet_rows':np.asarray(parent_rows,np.int32),'stable_source_face_offsets':np.asarray(face_po,np.int64),'stable_source_face_rows':np.asarray(face_rows,np.uint32),
  'unresolved_child_tet_rows':np.flatnonzero(selected<0).astype(np.int32),'unresolved_reason_codes':np.ones(int(np.count_nonzero(selected<0)),np.int16),
  'cell_selected_group_ids':selected,'cell_semantic_status':status,'cell_p1_score_denominator':denom,'cell_p1_winner_score_numerator':winner_num,
  'cell_group_offsets':np.asarray(cell_go,np.int64),'cell_group_ids':np.asarray(cell_gid,np.int32),'cell_group_p1_score_numerators':np.asarray(cell_num,np.int64),'cell_group_p1_score_denominators':np.asarray(cell_den,np.int64),
  'cell_group_parent_tet_offsets':np.asarray(cg_po,np.int64),'cell_group_parent_tet_rows':np.asarray(cg_pr,np.int32),'cell_group_stable_face_offsets':np.asarray(cg_fo,np.int64),'cell_group_stable_face_rows':np.asarray(cg_fr,np.uint32),
  'node_group_offsets':ngoff,'node_group_ids':ngids,'node_group_parent_tet_offsets':ngpoff,'node_group_parent_tet_rows':ngprows,'node_origin_kind':cache['node_origin_kind'],'node_parent_registration_distance_m':cache['node_parent_registration_distance_m'],'group_tuples':groups,
  'node_group_stable_face_offsets':ngfoff,'node_group_stable_face_rows':ngfrows,
  'cell_admissible_region_offsets':admiss_off,'cell_admissible_region_ids':admiss_region.astype(np.int32),
  'admissibility_max_source_region_hops':np.array(1,np.int32),'admissibility_source_support_sha256':np.array('P1-embedded-current-support'),
  'geometry_sha256':np.array(sha(VOL)),'node_support_sha256':np.array(sha(CACHE)),'source330_mass_sha256':np.array(sha(PARENT)),
 }
 np.savez_compressed(out,**arrays)
 with np.load(out,allow_pickle=False)as z:
  for k,v in arrays.items():
   if not np.array_equal(z[k],v):raise ValueError(f'roundtrip mismatch {k}')
 # Full compact receipt and direct same-material face-adjacency proofs.
 rpath=OUT/'receipt.json';report={
  'schema':'numi.human.current-skin-p1-semantic-onehop-support.v1','status':'PASS_SOURCE_ONLY_CURRENT_SKIN_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT',
  'physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN',
  'method':'P1-integrated source330 (material,donor,region) groups from exact cached current-node support; all ties and source-parent/stable-face unions retained. One-hop region admissibility adds only same-material source330 tetrahedra sharing an exact source-node face. P1 groups remain preference support; current source semantics and donor ledgers are unchanged.',
  'inputs':{'geometry':{'path':str(VOL),'sha256':sha(VOL)},'geometry_receipt':{'path':str(VOL_RECEIPT),'sha256':sha(VOL_RECEIPT)},'node_support':{'path':str(CACHE),'sha256':sha(CACHE)},'node_support_receipt':{'path':str(CACHE_RECEIPT),'sha256':sha(CACHE_RECEIPT)},'source330_mass':{'path':str(PARENT),'sha256':sha(PARENT)},'semantic_owner':{'path':str(OWNER),'sha256':sha(OWNER)},'exact_global_volume_report':{'path':str(EXACT_VOLUME),'sha256':sha(EXACT_VOLUME)}},
  'counts':{'nodes':nnode,'cells':ncell,'source_groups':ng,'P1_group_slots':len(cell_gid),'cells_with_unique_max_group':len(resolved_rows),'cells_with_P1_tie':int(np.count_nonzero(selected<0)),'direct_cell_region_slots':len(direct_codes),'admissible_cell_region_slots':len(admiss_region),'max_admissible_regions_per_cell':int(lengths.max()),'same_material_shared_face_region_edges':len(edgeproof)},
  'adjacency_edges':[{'regions':list(k),**v}for k,v in sorted(edgeproof.items())],
  'bounds':{'node_parent_registration_m':0.02,'mass_transport_m':0.019353,'density_bounds_kg_m3':[900,1100],'transport_and_density_not_evaluated_here':True},
  'output':{'path':str(out),'sha256':sha(out),'size_bytes':out.stat().st_size},'elapsed_s':time.monotonic()-start,
  'limitations':['Cell support is derived for Skin531 and must be regenerated if current cell topology changes.','No mass allocation, receiver-level feasibility, native installation, or physical admission.']}
 rpath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 report['receipt_sha256']=sha(rpath)
 print(json.dumps({'status':report['status'],'output_path':str(out),'output_sha256':sha(out),'receipt_path':str(rpath),'receipt_sha256':sha(rpath),'counts':report['counts'],'elapsed_s':report['elapsed_s']},sort_keys=True),flush=True)
if __name__=='__main__':main()

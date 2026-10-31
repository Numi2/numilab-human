#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
C=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-006/fat-comp0-conditioned-volume-proposal.npz'
D=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/candidate-D-comp0-480-boundary-split-typed.npz'
C190=H/'joint-D414-I466-target-475-root/fat-target397/fat-current-typed-target-rebound.npz'
S1=H/'d-shortedge-quotient-475-next-001-preflight-100um/transfer-inputs-001/comp1-current-protection.npz'
C_EXPECT='54381f3d6072f95111010b3de0fdd7334f254f5466f5422b6936bf684083ee3e'
D_EXPECT='3ef9353262b4821318c76a6e75f47a22101f62368d6546c3ea7026275bde5e7f'
C190_EXPECT='c1903e60b03aeb3a53592c8ecab020abae59eb8829899dece614d4a946f6fb04'
S1_EXPECT='faed6ff45827e5e6496ec10bd0d8bb750204dfc149ac947878839e39e0afaac0'
OUT=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/protection-004'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def digest(a,dtype):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def key_array(t,m):
 x=np.sort(np.asarray(t,np.int64),axis=1)
 k=np.empty(len(x),dtype=np.dtype([('mat','<i4'),('a','<i8'),('b','<i8'),('c','<i8'),('d','<i8')]))
 k['mat']=np.asarray(m,np.int32);k['a']=x[:,0];k['b']=x[:,1];k['c']=x[:,2];k['d']=x[:,3]
 return k
if OUT.exists():raise FileExistsError(OUT)
for p,w in ((C,C_EXPECT),(D,D_EXPECT),(C190,C190_EXPECT),(S1,S1_EXPECT)):
 if sha(p)!=w:raise ValueError('pinned input hash mismatch '+str(p))
with np.load(S1,allow_pickle=False) as z:
 old_comp1_rows=np.asarray(z['input_cell_rows'],np.int64)
 old_comp1_points=np.asarray(z['global_point_rows'],np.int64)
 old_comp1_output_rows=np.asarray(z['source_comp1_output_rows_to_current'],np.int64)
 with np.load(C190,allow_pickle=False) as c:
  old_t=np.asarray(c['tetrahedra'],np.int64);old_m=np.asarray(c['material_indices'],np.int32);old_p=np.asarray(c['points_world_m'],np.float32)
 if str(z['source_current_volume_sha256'].item())!=C190_EXPECT:raise ValueError('legacy comp1 protection source volume hash mismatch')
with np.load(C,allow_pickle=False) as z:
 p=np.asarray(z['points_world_m'],np.float32);t=np.asarray(z['tetrahedra'],np.int64);mat=np.asarray(z['material_indices'],np.int32)
 comp0_rows=np.asarray(z['comp0_replacement_output_cell_rows'],np.int64)
 comp0_tets=np.asarray(z['comp0_replacement_output_tetrahedra_global_node_rows'],np.int64)
 if not np.array_equal(t[comp0_rows],comp0_tets):raise ValueError('comp0 cell map mismatch')
 comp0_points=np.unique(comp0_tets.ravel())
 if np.any(old_comp1_points>=len(p)) or np.any(old_comp1_rows>=len(old_t)):raise ValueError('comp1 source row out of bounds')
 used_comp1=np.unique(old_t[old_comp1_rows].ravel())
 if not np.array_equal(np.sort(used_comp1),np.sort(old_comp1_points)):raise ValueError('comp1 point list is not the full protected cell-star point domain')
 if not np.array_equal(old_p[used_comp1].view(np.uint32),p[used_comp1].view(np.uint32)):
  raise ValueError('comp1 protected point coordinates changed in current candidate')
 oldkeys=key_array(old_t[old_comp1_rows],old_m[old_comp1_rows])
 current_keys=key_array(t,mat);order=np.argsort(current_keys,order=('mat','a','b','c','d'));sorted_current=current_keys[order]
 ix=np.searchsorted(sorted_current,oldkeys);good=ix<len(sorted_current);good[good]&=sorted_current[ix[good]]==oldkeys[good]
 if not np.all(good) or len(np.unique(ix))!=len(ix):raise ValueError('not all protected comp1 star tet/material keys map uniquely into current candidate')
 comp1_rows_current=order[ix].astype(np.int64)
 outkeys=key_array(old_t[old_comp1_output_rows],old_m[old_comp1_output_rows])
 oi=np.searchsorted(sorted_current,outkeys);og=oi<len(sorted_current);og[og]&=sorted_current[oi[og]]==outkeys[og]
 if not np.all(og) or len(np.unique(oi))!=len(oi):raise ValueError('comp1 protected output rows do not map uniquely')
 comp1_output_rows_current=order[oi].astype(np.int64)
 comp0_cells=np.flatnonzero(np.any(np.isin(t,comp0_points),axis=1)).astype(np.int64)
 overlap=np.intersect1d(comp0_cells,comp1_rows_current)
 prot_cells=np.union1d(comp0_cells,comp1_rows_current).astype(np.int64)
 prot_points=np.unique(t[prot_cells].ravel()).astype(np.int64)
 keys=key_array(t[prot_cells],mat[prot_cells]);keys=keys[np.argsort(keys,order=('mat','a','b','c','d'))]
 with np.load(D,allow_pickle=False) as dz:
  dp=np.asarray(dz['points_world_m'],np.float32)
  if not np.array_equal(dp[120644:120647].view(np.uint32),p[253220:253223].view(np.uint32)):
   raise ValueError('D comp1 three-point crosswalk is not bit-identical')
 if not np.array_equal(np.sort(np.asarray([253220,253221,253222],np.int64)),np.sort(old_comp1_points[np.isin(old_comp1_points,[253220,253221,253222])])):
  raise ValueError('expected D split nodes absent from comp1 protected source points')
OUT.mkdir(parents=True)
side=OUT/'comp0-comp1-protected-stars.npz'
np.savez_compressed(side,protected_current_cell_rows=prot_cells,protected_tet_material_corner_keys=keys,
 comp0_patch_output_cell_rows=comp0_rows,comp0_patch_point_rows=comp0_points,comp0_full_point_star_cell_rows=comp0_cells,
 comp1_source_c190_full_star_cell_rows=old_comp1_rows,comp1_current_full_star_cell_rows=comp1_rows_current,
 comp1_source_c190_protected_point_rows=old_comp1_points,comp1_current_protected_point_rows=old_comp1_points,
 comp1_source_c190_output_rows=old_comp1_output_rows,comp1_current_output_rows=comp1_output_rows_current,
 comp1_three_D_split_point_rows=np.asarray([120644,120645,120646],np.int64),comp1_three_volume_split_point_rows=np.asarray([253220,253221,253222],np.int64),
 comp1_point_coordinates_f32=p[old_comp1_points],protected_volume_point_rows=prot_points)
receipt={'status':'PASS_EXACT_COMP0_COMP1_CURRENT_PROTECTED_STAR_REBASE','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN',
 'current_candidate':str(C),'current_candidate_sha256':sha(C),'current_D_target':str(D),'current_D_target_sha256':sha(D),
 'comp0_patch_cells':int(len(comp0_rows)),'comp0_patch_points':int(len(comp0_points)),'comp0_full_point_star_cells':int(len(comp0_cells)),
 'comp1_legacy_protection':str(S1),'comp1_legacy_protection_sha256':sha(S1),'comp1_source_volume':str(C190),'comp1_source_volume_sha256':sha(C190),
 'comp1_source_full_star_cells':int(len(old_comp1_rows)),'comp1_source_protected_points':int(len(old_comp1_points)),
 'comp1_full_star_unique_material_tet_keys_rebased':int(len(comp1_rows_current)),'all_comp1_keys_mapped_uniquely':True,
 'comp1_comp1_source_point_coordinates_unchanged_on_current_candidate':True,'comp1_comp1_source_point_domain_equals_star_corner_union':True,
 'comp1_source_output_rows':int(len(old_comp1_output_rows)),'comp1_output_rows_rebased_uniquely':int(len(comp1_output_rows_current)),
 'comp1_new_D_point_crosswalk':'D rows120644..120646 ↔ Fat rows253220..253222, Float32 bits equal',
 'comp0_comp1_fullstar_cell_intersection':int(len(overlap)),'union_protected_cells':int(len(prot_cells)),'union_protected_volume_points':int(len(prot_points)),
 'protected_key_sha256':digest(np.asarray([(int(k['mat']),int(k['a']),int(k['b']),int(k['c']),int(k['d'])) for k in keys],dtype='<i8'),'<i8'),
 'sidecar':str(side),'sidecar_sha256':sha(side),'mapping_basis':'comp0 full-star rows are recomputed from explicit current patch points; comp1 full-star and output rows are reidentified from authenticated c190 rows by exact unique (material, sorted tet corners) keys, with all protected comp1 point F32 bytes checked unchanged.'}
rp=OUT/'receipt.json';rp.write_text(json.dumps(receipt,indent=2)+'\n');receipt['receipt_sha256']=sha(rp);print(json.dumps(receipt,indent=2))

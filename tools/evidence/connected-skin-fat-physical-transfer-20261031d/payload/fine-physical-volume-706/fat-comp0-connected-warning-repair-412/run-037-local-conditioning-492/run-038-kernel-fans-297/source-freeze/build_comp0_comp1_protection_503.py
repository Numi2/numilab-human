#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
C=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-006/fat-comp0-conditioned-volume-proposal.npz'
D=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/candidate-D-comp0-480-boundary-split-typed.npz'
EXPECT_C='54381f3d6072f95111010b3de0fdd7334f254f5466f5422b6936bf684083ee3e'
EXPECT_D='3ef9353262b4821318c76a6e75f47a22101f62368d6546c3ea7026275bde5e7f'
OUT=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/protection-003'
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def ah(a,dtype): return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
if OUT.exists(): raise FileExistsError(OUT)
if sha(C)!=EXPECT_C or sha(D)!=EXPECT_D: raise ValueError('source candidate/D target pin mismatch')
with np.load(C,allow_pickle=False) as z:
 p=np.asarray(z['points_world_m'],np.float32);t=np.asarray(z['tetrahedra'],np.int64);mat=np.asarray(z['material_indices'],np.int32)
 comp0_rows=np.asarray(z['comp0_replacement_output_cell_rows'],np.int64)
 comp0_tets=np.asarray(z['comp0_replacement_output_tetrahedra_global_node_rows'],np.int64)
 if not np.array_equal(t[comp0_rows],comp0_tets):raise ValueError('comp0 cell row map mismatch')
 comp0_points=np.unique(comp0_tets.ravel())
 comp1_points=np.asarray([253220,253221,253222],np.int64)
 if np.any(comp1_points>=len(p)):raise ValueError('comp1 point rows out of domain')
 with np.load(D,allow_pickle=False) as dz:
  dp=np.asarray(dz['points_world_m'],np.float32)
  if dp.shape[0] <= 120646 or not np.array_equal(dp[120644:120647].view(np.uint32),p[comp1_points].view(np.uint32)):
   raise ValueError('comp1 volume/D split point crosswalk mismatch')
 comp0_cells=np.flatnonzero(np.any(np.isin(t,comp0_points),axis=1)).astype(np.int64)
 comp1_cells=np.flatnonzero(np.any(np.isin(t,comp1_points),axis=1)).astype(np.int64)
 prot_cells=np.union1d(comp0_cells,comp1_cells).astype(np.int64)
 prot_points=np.unique(t[prot_cells].ravel()).astype(np.int64)
 keys=np.sort(t[prot_cells],axis=1)
 keys=np.c_[mat[prot_cells],keys]
 order=np.lexsort((keys[:,4],keys[:,3],keys[:,2],keys[:,1],keys[:,0]));keys=keys[order]
 OUT.mkdir(parents=True)
 q=OUT/'comp0-comp1-protected-stars.npz'
 np.savez_compressed(q,candidate_output_cell_rows=prot_cells,protected_tet_material_corner_keys=keys,
  comp0_patch_output_cell_rows=comp0_rows,comp0_patch_point_rows=comp0_points,comp0_full_point_star_cell_rows=comp0_cells,
  comp1_split_point_rows=comp1_points,comp1_full_point_star_cell_rows=comp1_cells,protected_volume_point_rows=prot_points,
  comp0_point_coordinates_f32=p[comp0_points],comp1_point_coordinates_f32=p[comp1_points])
receipt={'status':'PASS_HASHED_CURRENT_FAT_COMP0_COMP1_FULLSTAR_EXCLUSIONS','physical_admission':False,'native_steps':0,
 'candidate':str(C),'candidate_sha256':sha(C),'D_target':str(D),'D_target_sha256':sha(D),
 'comp0_patch_cells':int(len(comp0_rows)),'comp0_patch_points':int(len(comp0_points)),'comp0_full_point_star_cells':int(len(comp0_cells)),
 'comp1_D_point_rows':[120644,120645,120646],'comp1_volume_point_rows':comp1_points.tolist(),'comp1_point_crosswalk_f32_bit_identical':True,
 'comp1_full_point_star_cells':int(len(comp1_cells)),'union_protected_cells':int(len(prot_cells)),'union_protected_volume_points':int(len(prot_points)),
 'protected_tet_material_corner_keys_sha256':ah(keys,'<i8'),'sidecar':str(q),'sidecar_sha256':sha(q),
 'mapping_basis':'D480 split point rows 120644..120646 bit-match the three appended comp1 volume nodes; comp0 rows/points are explicit fields on the exact pinned Fat candidate.',
 'source_only':True}
rp=OUT/'receipt.json';rp.write_text(json.dumps(receipt,indent=2)+'\n');receipt['receipt_sha256']=sha(rp)
print(json.dumps(receipt,indent=2))

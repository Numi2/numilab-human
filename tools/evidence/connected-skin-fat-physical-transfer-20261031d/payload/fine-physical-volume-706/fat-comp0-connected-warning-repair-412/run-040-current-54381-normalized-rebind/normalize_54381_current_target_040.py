#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-006/fat-comp0-conditioned-volume-proposal.npz'
OLD_REBIND=H/'fat-comp0-connected-warning-repair-412/run-033-target-rebind-480/result-002/fat-current-typed-target-rebound.npz'
OUT=H/'fat-comp0-connected-warning-repair-412/run-040-current-54381-normalized-rebind'
D=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/candidate-D-comp0-480-boundary-split-typed-lineage-corrected-001.npz'
D_SHA='6451dfe347d42b247ecf9ca9c6baf14dd8f0c0bf7d8aa5bfca81c7fa8b6991e8'
D697=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/actual-697-003/actual-697-D-comp0-split-load.json'
D697_SHA='6b39046f3648d571ceb4772f3cac3fd49c076bba1df4dfd4236129bd73b361e4'
CORR=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/metadata-lineage-correction-receipt.json'
CORR_SHA='146d3c38065b5254f5fb2d9efbbaf35f92664bd3c3bf9ee77eb50fcb5b7e0625'
I=H/'fat-comp0-connected-warning-repair-412/run-031-threeD-oneI-safe-flips/prep-003/candidate-I-warning-flips-typed.npz'
I_SHA='81b285a4e12f11dfd4156bb2545bb86f5ccee900ef6021a11beff36954b00774'
O=H/'outer-O-connected-quality-506-root/candidate/typed434001/candidate-O-typed-current.npz'
O_SHA='0c3353060d3e675f7c907e8b0bbf4317b798b0c041cf9f25294b22bee9e38f0f'
O_RECEIPT=H/'outer-O-connected-quality-506-root/candidate/typed434001/receipt.json'
O_RECEIPT_SHA='5c938e6d0451c70e926974bc520f1acb914971d1b03017130cc5338ee861b653'
I_RECEIPT=H/'fat-comp0-connected-warning-repair-412/run-031-threeD-oneI-safe-flips/prep-003/actual-697-warning-flips-001/actual-697-D475-warning-flips-load-001.json'
# This existing receipt loads D; record I's source and full-context proof separately.
I_CONTEXT=H/'fat-comp0-connected-warning-repair-412/run-031-threeD-oneI-safe-flips/prep-003/report.json'
I_CONTEXT_SHA='a5aec63aa8c55ad9d25788cfb1e32748a7745c81898b5f50e5c13348cb1bb4ea'
AUDIT197=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/exact-audit-197-005/report.json'
AUDIT197_SHA='083c88aaae74455a9c5084f4097cd942bb8031d743e2cb34cd10e269463b42f4'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def f32key(x):return np.ascontiguousarray(np.asarray(x,dtype='<f4')).tobytes()
def unique_map(points):
 m={};dups=set()
 for i,p in enumerate(np.asarray(points,np.float32)):
  k=f32key(p)
  if k in m:dups.add(k)
  else:m[k]=int(i)
 for k in dups:m[k]=-1
 return m
def crosswalk(points,lookup,label):
 out=np.asarray([lookup.get(f32key(p),-1) for p in np.asarray(points,np.float32)],np.int64)
 if np.any(out<0):raise ValueError(f'{label}: {np.count_nonzero(out<0)} source points missing or ambiguous')
 if len(np.unique(out))!=len(out):raise ValueError(f'{label}: non-injective coordinate map')
 return out
def oriented_key(tri):
 x=tuple(map(int,tri));s=tuple(sorted(x));rank={v:i for i,v in enumerate(s)}
 inv=int(rank[x[0]]>rank[x[1]])+int(rank[x[0]]>rank[x[2]])+int(rank[x[1]]>rank[x[2]])
 return (*s,-1 if inv%2 else 1)
def arrhash(a):
 a=np.ascontiguousarray(a)
 return {'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'dtype':a.dtype.str,'shape':list(a.shape),'nbytes':int(a.nbytes)}
if not OUT.exists():OUT.mkdir(parents=True)
else:
 existing=[q for q in OUT.iterdir() if q.resolve()!=Path(__file__).resolve()]
 if existing:raise FileExistsError(OUT)
for p,h in [(BASE,'54381f3d6072f95111010b3de0fdd7334f254f5466f5422b6936bf684083ee3e'),(OLD_REBIND,'1bdff4e0007fdc2ab5867830f20ded3f30c70a1bac5a9ffaa0b66577db7913b2'),(D,D_SHA),(D697,D697_SHA),(CORR,CORR_SHA),(I,I_SHA),(O,O_SHA),(O_RECEIPT,O_RECEIPT_SHA),(I_CONTEXT,I_CONTEXT_SHA),(AUDIT197,AUDIT197_SHA)]:
 if sha(p)!=h:raise ValueError(f'pin mismatch {p}: {sha(p)} != {h}')
a=load(BASE); old=load(OLD_REBIND)
P=np.asarray(a['points_world_m_f32'],np.float32); actual=np.asarray(a['boundary_faces'],np.int64)
proposed=np.asarray(a['proposed_current_desired_boundary_faces'],np.int64)
from collections import Counter
def oriented_key(tri):
 x=tuple(map(int,tri));s=tuple(sorted(x));rank={v:i for i,v in enumerate(s)}
 inv=int(rank[x[0]]>rank[x[1]])+int(rank[x[0]]>rank[x[2]])+int(rank[x[1]]>rank[x[2]])
 return (*s,-1 if inv%2 else 1)
actual_chain=Counter(map(oriented_key,actual));proposed_chain=Counter(map(oriented_key,proposed))
if actual.shape!=proposed.shape or actual_chain!=proposed_chain:raise ValueError('actual boundary oriented chain differs from proposed current desired target')
if not np.array_equal(np.asarray(a['points_world_m'],np.float32),P):raise ValueError('Float32 point views differ before normalization')
# Exact role point crosswalks into the current Fat volume.
lookup=unique_map(P)
d=load(D);ii=load(I);oo=load(O)
Dmap=crosswalk(d['points_world_m_f32'],lookup,'D')
Imap=crosswalk(ii['points_world_m_f32'],lookup,'I')
Omap=crosswalk(oo['points_world_m_f32'],lookup,'O')
if int(Dmap[-1])!=253223:raise ValueError(f'appended D split point maps to {int(Dmap[-1])}, expected volume row253223')
# Copy the registered 90 wall map by exact F32 coordinate identity from the prior rebind.
oldpts=np.asarray(old['points_world_m_f32'],np.float32)
oldwall=np.asarray(old['fat_current_target_rebind_wall_faces_volume_rows'],np.int64)
oldwallcoords=oldpts[oldwall]
wallmap=unique_map(P)
wall=np.asarray([[wallmap.get(f32key(x),-1) for x in tri] for tri in oldwallcoords],np.int64)
if np.any(wall<0):raise ValueError('one or more registered wall vertices are absent/ambiguous in current normalized volume')
if len(np.unique(wall.reshape(-1)))!=len(np.unique(oldwallcoords.reshape(-1))):raise ValueError('wall point map is not injective')
if not np.array_equal(P[wall],oldwallcoords):raise ValueError('wall point coordinates changed during transfer')
# Canonical target field names used by 397 consumers.
for dst,src in [('desired_boundary_faces','proposed_current_desired_boundary_faces'),('desired_boundary_source_group','proposed_current_desired_boundary_source_group'),('desired_boundary_source_rows','proposed_current_desired_boundary_source_rows'),('desired_boundary_source_face_rows','proposed_current_desired_boundary_source_face_rows')]:
 a[dst]=np.array(a[src],copy=True)
# Exact oriented closure between the actual boundary and canonical target.
from collections import Counter
if Counter(map(oriented_key,actual))!=Counter(map(oriented_key,a['desired_boundary_faces'])):raise ValueError('canonical desired target oriented chain does not equal actual Fat boundary')
# Normalize current role and wall mappings. Existing wall collar/source metadata is retained.
a['fat_current_target_rebind_schema']=np.asarray(old['fat_current_target_rebind_schema'])
a['fat_current_target_rebind_D_point_to_volume_rows']=Dmap
a['fat_current_target_rebind_I_point_to_volume_rows']=Imap
a['fat_current_target_rebind_O_point_to_volume_rows']=Omap
a['fat_current_target_rebind_wall_faces_volume_rows']=wall
a['fat_current_target_rebind_wall_faces_collar_point_rows']=np.array(old['fat_current_target_rebind_wall_faces_collar_point_rows'],copy=True)
a['fat_current_target_rebind_wall_face_points_world_m_f32']=np.array(old['fat_current_target_rebind_wall_face_points_world_m_f32'],copy=True)
assets=json.loads(str(np.asarray(old['fat_current_target_rebind_source_asset_sha256_json']).item()))
assets.update({'typed_D':D_SHA,'typed_D_697_receipt':D697_SHA,'typed_D_lineage_metadata_correction_receipt':CORR_SHA,'typed_I':I_SHA,'typed_O':O_SHA,'typed_O_receipt':O_RECEIPT_SHA,'typed_I_context_parent_report':I_CONTEXT_SHA})
a['fat_current_target_rebind_source_asset_sha256_json']=np.asarray([json.dumps(assets,sort_keys=True,separators=(',',':'))])
a['fat_current_target_rebind_typed_D_candidate_path']=np.asarray([str(D)])
a['fat_current_target_rebind_typed_D_candidate_sha256']=np.asarray([D_SHA])
a['fat_current_target_rebind_typed_D_697_receipt_path']=np.asarray([str(D697)])
a['fat_current_target_rebind_typed_D_697_receipt_sha256']=np.asarray([D697_SHA])
a['fat_current_target_rebind_typed_D_lineage_correction_receipt_path']=np.asarray([str(CORR)])
a['fat_current_target_rebind_typed_D_lineage_correction_receipt_sha256']=np.asarray([CORR_SHA])
a['fat_current_target_rebind_typed_I_candidate_path']=np.asarray([str(I)])
a['fat_current_target_rebind_typed_I_candidate_sha256']=np.asarray([I_SHA])
a['fat_current_target_rebind_typed_O_candidate_path']=np.asarray([str(O)])
a['fat_current_target_rebind_typed_O_candidate_sha256']=np.asarray([O_SHA])
a['fat_current_target_rebind_metadata_mode']=np.asarray(['METADATA_ONLY_TARGET_NORMALIZATION; exact current O/I/D crosswalks and corrected D lineage pins; no point/tet/boundary edits'])
NPZ=OUT/'fat-current-typed-target-rebound-54381-lineage-corrected.npz'
np.savez_compressed(NPZ,**a)
# Readback, exact base-array preservation, and points/target pins.
r=load(NPZ)
if not set(load(BASE)).issubset(r):raise ValueError('normalization dropped base candidate arrays')
for k,v in load(BASE).items():
 if r[k].dtype!=v.dtype or r[k].shape!=v.shape or r[k].tobytes()!=v.tobytes():raise ValueError(f'normalization altered original source array {k}')
if not np.array_equal(r['desired_boundary_faces'],proposed):raise ValueError('canonical target differs from proposed target')
if not np.array_equal(r['fat_current_target_rebind_D_point_to_volume_rows'],Dmap) or not np.array_equal(r['fat_current_target_rebind_I_point_to_volume_rows'],Imap) or not np.array_equal(r['fat_current_target_rebind_O_point_to_volume_rows'],Omap):raise ValueError('normalized coordinate maps differ on readback')
if not np.array_equal(r['fat_current_target_rebind_wall_faces_volume_rows'],wall):raise ValueError('wall coordinate map differs on readback')
# Report exact same geometry bytes, target closure, source roles, and D lineage receipt.
report={
 'schema':'h706.fat54381-current-target-metadata-normalization.v1','status':'PASS_METADATA_ONLY_CURRENT_TARGET_NORMALIZATION_AND_EXACT_F32_CROSSWALK',
 'input_candidate_path':str(BASE),'input_candidate_sha256':sha(BASE),'output_candidate_path':str(NPZ),'output_candidate_sha256':sha(NPZ),
 'metadata_only':True,'original_arrays_preserved_count':len(load(BASE)),'original_arrays_changed':[],
 'canonical_target':{'source_field':'proposed_current_desired_boundary_faces','target_field':'desired_boundary_faces','face_count':int(len(proposed)),'source_array_order_byte_identical':bool(np.array_equal(actual,proposed)),'actual_boundary_face_count':int(len(actual)),'oriented_boundary_residual_count':0,'exact_actual_faces_equal_proposed_target':True},
 'role_pins':{'D':{'path':str(D),'sha256':D_SHA,'697_receipt_path':str(D697),'697_receipt_sha256':D697_SHA,'metadata_correction_receipt_path':str(CORR),'metadata_correction_receipt_sha256':CORR_SHA,'point_count':int(len(Dmap)),'mapped_point_count':int(np.count_nonzero(Dmap>=0)),'appended_split_point_surface_row':120647,'appended_split_point_volume_row':int(Dmap[-1])},'I':{'path':str(I),'sha256':I_SHA,'point_count':int(len(Imap)),'mapped_point_count':int(np.count_nonzero(Imap>=0))},'O':{'path':str(O),'sha256':O_SHA,'typed_receipt_path':str(O_RECEIPT),'typed_receipt_sha256':O_RECEIPT_SHA,'point_count':int(len(Omap)),'mapped_point_count':int(np.count_nonzero(Omap>=0))}},
 'wall_mapping':{'source_rebind_path':str(OLD_REBIND),'source_rebind_sha256':sha(OLD_REBIND),'wall_face_count':int(len(wall)),'mapped_vertex_count':int(len(np.unique(wall))),'coordinates_bitwise_equal':True},
 'array_hashes':{k:arrhash(r[k]) for k in ['points_world_m','points_world_m_f32','tetrahedra','material_indices','boundary_faces','proposed_current_desired_boundary_faces','desired_boundary_faces','desired_boundary_source_group','desired_boundary_source_rows','desired_boundary_source_face_rows','fat_current_target_rebind_D_point_to_volume_rows','fat_current_target_rebind_I_point_to_volume_rows','fat_current_target_rebind_O_point_to_volume_rows','fat_current_target_rebind_wall_faces_volume_rows']},
 'admission':{'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','quality_gate':'not rerun by metadata normalization'}
}
RP=OUT/'target-normalization-receipt.json';RP.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'candidate':str(NPZ),'candidate_sha256':sha(NPZ),'receipt':str(RP),'receipt_sha256':sha(RP),'points':len(P),'tets':len(a['tetrahedra']),'target_faces':len(proposed),'D_split_volume_row':int(Dmap[-1]),'D_mapped':len(Dmap),'I_mapped':len(Imap),'O_mapped':len(Omap),'wall_faces':len(wall)},sort_keys=True))

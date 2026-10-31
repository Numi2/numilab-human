#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
SRC=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/result-006/fat-comp0-conditioned-volume-proposal.npz'
EXPECT='54381f3d6072f95111010b3de0fdd7334f254f5466f5422b6936bf684083ee3e'
OUT=H/'fat-comp0-connected-warning-repair-412/run-037-local-conditioning-492/run-038-kernel-fans-297/prep-extract-002'
OWNER=H/'fixed_boundary_volume_flip_198_root.py'
TOPO=H/'remesh_complete_surface_cavities_218_root.py'
def sha(p): return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def loadmod(name,path):
 import importlib.util
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
if __name__=='__main__':
 if OUT.exists(): raise FileExistsError(OUT)
 if sha(SRC)!=EXPECT: raise ValueError('source candidate hash mismatch')
 outdir=OUT/'prepared';outdir.mkdir(parents=True)
 with np.load(SRC,allow_pickle=False) as z:
  p=np.asarray(z['points_world_m'],np.float32)
  if not np.array_equal(p,z['points_world_m_f32']):raise ValueError('noncanonical F32 source point view')
  t=np.asarray(z['tetrahedra'],np.int64)
  mat=np.asarray(z['material_indices'],np.int32)
  rows=np.asarray(z['comp0_replacement_output_cell_rows'],np.int64)
  local_t=np.asarray(z['comp0_replacement_output_tetrahedra_global_node_rows'],np.int64)
  local_m=np.asarray(z['comp0_replacement_output_material_indices'],np.int32)
  desired=np.asarray(z['comp0_refined_selected_shell_faces_global'],np.int64)
  if not np.array_equal(t[rows],local_t) or not np.array_equal(mat[rows],local_m):raise ValueError('local patch does not match full candidate row map')
  if len(t)!=802165 or len(local_t)!=13353 or len(rows)!=len(local_t) or np.any(local_m!=16):raise ValueError('unexpected candidate/patch domain')
  if desired.shape!=(9338,3):raise ValueError('unexpected local refined shell size')
  topo=loadmod('prep_topo218',TOPO)
  bd=topo.m.boundary(local_t)
  def fchain(faces):
   out={}
   for ff in np.asarray(faces,np.int64).reshape(-1,3):
    a,b,c=map(int,ff);key=tuple(sorted((a,b,c)));sgn=1 if ((a>b)+(a>c)+(b>c))%2==0 else -1
    out[key]=out.get(key,0)+sgn
   return {k:v for k,v in out.items() if v}
  if fchain(bd)!=fchain(desired):raise ValueError('current local patch boundary != pinned refined shell')
  payload={'points_world_m':p,'tetrahedra':local_t,'material_indices':local_m,'boundary_faces':bd,'desired_boundary_faces':desired,'solver_infeasible_cavity_sha256':np.empty((0,32),np.uint8)}
  local=outdir/'conditioned-comp0-local-input.npz';np.savez_compressed(local,**payload)
 rep={'status':'PASS_EXACT_LOCAL_INPUT_EXTRACTED','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','source_candidate':str(SRC),'source_candidate_sha256':sha(SRC),'source_parent_cell_rows_count':int(len(rows)),'source_parent_cell_rows_sha256':hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest(),'input_cells':int(len(local_t)),'point_rows':int(len(p)),'material':16,'local_boundary_faces':int(len(desired)),'local_oriented_boundary_matches_source_shell':True,'point_coordinates_unchanged':True,'input_npz':str(local),'input_npz_sha256':sha(local),'preparer':str(Path(__file__)),'preparer_sha256':sha(Path(__file__))}
 (outdir/'receipt.json').write_text(json.dumps(rep,indent=2)+'\n')
 print(json.dumps(rep,indent=2))

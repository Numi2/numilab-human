#!/usr/bin/env python3
"""Read-only exact diagnosis of the retained 484 TetGen -YY failure."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse,hashlib,json,importlib.util,sys,time
import numpy as np
ROOT=Path(r"/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
RUN=ROOT/"fat-comp0-connected-warning-repair-412/run-035-current-target-480-comp0-interior128"
CASE=RUN/"component-0000"
INPUT=ROOT/"fat-comp0-connected-warning-repair-412/run-033-target-rebind-480/result-002/fat-current-typed-target-rebound.npz"
SHELL=ROOT/"fat-comp0-connected-warning-repair-412/run-034-compact-prep-481/preparation-002/connected-cavity-shells.npz"
PREP=ROOT/"fat-comp0-connected-warning-repair-412/run-034-compact-prep-481/preparation-002/report.json"
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
R=load("remesh484owner",ROOT/"remesh_fat614_compact_cavities_484_root.py")
M=R.m.m
J=load("joint272diag",ROOT/"audit_joint_native_apex_272_root.py")
def read_table(path):
 lines=[x.strip().split() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 n=int(lines[0][0]);assert len(lines)-1==n
 return np.asarray([[float(v) for v in row] for row in lines[1:]],np.float64)
def orient_key(f):
 a,b,c=map(int,f)
 return min((a,b,c),(b,c,a),(c,a,b))
def oriented_counter(f):return Counter(orient_key(x) for x in f)
def edge_chain(faces):
 cc=Counter()
 for a,b,c in np.asarray(faces,np.int64):
  for x,y in ((a,b),(b,c),(c,a)):
   k=tuple(sorted((int(x),int(y))));cc[k]+=1 if x<y else -1
 return {e:n for e,n in cc.items() if n}
def flux(ip,faces,global_to_local):
 total=0
 for a,b,c in np.asarray(faces,np.int64):
  x,y,z=[ip[global_to_local[int(i)]] for i in (a,b,c)]
  total+=x[0]*(y[1]*z[2]-y[2]*z[1])-x[1]*(y[0]*z[2]-y[2]*z[0])+x[2]*(y[0]*z[1]-y[1]*z[0])
 return total
def run(outdir):
 outdir.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 pin={"candidate":sha(INPUT),"shell":sha(SHELL),"prep_report":sha(PREP),"remesher":sha(ROOT/"remesh_fat614_compact_cavities_484_root.py"),"node":sha(CASE/"cavity.1.node"),"ele":sha(CASE/"cavity.1.ele"),"face":sha(CASE/"cavity.1.face"),"poly":sha(CASE/"cavity.poly"),"stdout":sha(CASE/"tetgen.stdout.log")}
 if pin["candidate"]!="1bdff4e0007fdc2ab5867830f20ded3f30c70a1bac5a9ffaa0b66577db7913b2" or pin["shell"]!="c5f9889ab26e152144652ae51fde8ce7eff6a673038cc110b3f34f0cd7ba46b8":raise ValueError("frozen parent pin mismatch")
 with np.load(INPUT,allow_pickle=False) as z:v={k:z[k] for k in z.files}
 with np.load(SHELL,allow_pickle=False) as z:
  po=z["point_offsets"];fo=z["face_offsets"];co=z["cell_offsets"]
  grows=np.asarray(z["global_point_rows"][po[0]:po[1]],np.int64)
  lf=np.asarray(z["local_faces"][fo[0]:fo[1]],np.int64)
  rows=np.asarray(z["input_cell_rows"][co[0]:co[1]],np.int64)
 p=np.asarray(v["points_world_m"],np.float32);t=np.asarray(v["tetrahedra"],np.int64)
 node=read_table(CASE/"cavity.1.node");ele=read_table(CASE/"cavity.1.ele");face=read_table(CASE/"cavity.1.face")
 ids=node[:,0].astype(np.int64);xyz64=node[:,1:4]
 if len(np.unique(ids))!=len(ids):raise ValueError("duplicate TetGen node IDs")
 id_to_local={int(k):i for i,k in enumerate(ids)}
 xyz32=xyz64.astype(np.float32)
 def key(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
 input_lookup={key(x):i for i,x in enumerate(p)}
 if len(input_lookup)!=len(p):raise ValueError("duplicate source F32 point coordinates")
 gm=np.full(len(ids),-1,np.int64);new_points=[];new_key={}
 for i,x in enumerate(xyz32):
  k=key(x)
  if k in input_lookup:gm[i]=input_lookup[k]
  elif k in new_key:gm[i]=new_key[k]
  else:
   gm[i]=len(p)+len(new_points);new_key[k]=int(gm[i]);new_points.append(x.copy())
 node_ids=ele[:,1:5].astype(np.int64)
 if any(int(x) not in id_to_local for x in node_ids.ravel()):raise ValueError("tet references unknown output node ID")
 ot=np.asarray([[id_to_local[int(x)] for x in row] for row in node_ids],np.int64)
 if ele.shape[1]<6:raise ValueError("missing material attribute")
 attrs=ele[:,5].astype(np.int64)
 if np.any(attrs!=16):raise ValueError("unexpected TetGen material attribute")
 bp=np.vstack([p,np.asarray(new_points,np.float32).reshape(-1,3)])
 gt=gm[ot]
 if np.any(gt<0):raise ValueError("unmapped output tet vertex")
 det,q=M.measurements(xyz32,ot)
 ob_local=M.boundary(ot);ob_global=gm[ob_local]
 bnode_ids=sorted(set(int(x) for x in face[1:,1:4].astype(np.int64).ravel()))
 extra_boundary=sorted({int(gm[id_to_local[x]]) for x in bnode_ids if key(xyz32[id_to_local[x]]) not in input_lookup})
 if len(extra_boundary)!=1:raise ValueError("boundary Steiner count is not exactly one")
 edge_point=extra_boundary[0]
 if edge_point<len(p):raise ValueError("boundary node unexpectedly aliases source point")
 shell_global=grows[lf]
 retired=[np.array([131938,132166,245998],np.int64),np.array([131938,132165,132166],np.int64)]
 oldkey={tuple(sorted(map(int,f))):i for i,f in enumerate(shell_global)}
 if any(tuple(sorted(map(int,f))) not in oldkey for f in retired):raise ValueError("reported edge faces absent from shell")
 retired_shell_rows=[oldkey[tuple(sorted(map(int,f)))] for f in retired]
 patchfaces=ob_global[np.any(ob_global==edge_point,axis=1)]
 patch_topology=(len(patchfaces)==4 and set(map(int,patchfaces.ravel()))=={131938,132166,245998,132165,edge_point})
 expected_shell=np.asarray([f for i,f in enumerate(shell_global) if i not in set(retired_shell_rows)],np.int64)
 expected_shell=np.vstack([expected_shell,patchfaces])
 out_counter=oriented_counter(ob_global);expected_counter=oriented_counter(expected_shell)
 missing_chain=list((expected_counter-out_counter).elements());extra_chain=list((out_counter-expected_counter).elements())
 oldpatch=shell_global[retired_shell_rows]
 chain_equal=edge_chain(oldpatch)==edge_chain(patchfaces)
 a=bp[131938].astype(float);b=bp[132166].astype(float);x=bp[edge_point].astype(float);d=b-a
 param=float(np.dot(x-a,d)/np.dot(d,d));proj=a+param*d;edge_res=float(np.linalg.norm(x-proj))
 cross=[]
 for j in range(3):
  u=[Fraction.from_float(float(x[k]))-Fraction.from_float(float(a[k])) for k in range(3)]
  w=[Fraction.from_float(float(b[k]))-Fraction.from_float(float(a[k])) for k in range(3)]
  cross.append(u[(j+1)%3]*w[(j+2)%3]-u[(j+2)%3]*w[(j+1)%3])
 exact_collinear=all(z==0 for z in cross)
 plane=[]
 for f in retired:
  aa,bb,cc=bp[f].astype(float);n=np.cross(bb-aa,cc-aa);plane.append(abs(float(np.dot(x-aa,n)/np.linalg.norm(n))))
 patch_score=J.score(bp,patchfaces);apex_points=np.unique(patchfaces);apex_local={int(g):i for i,g in enumerate(apex_points)};ip_apex=M.integer_points(bp[apex_points])
 apex_reasons=[]
 for f in patchfaces:
  reason,gap=J.exact_terms(ip_apex,[apex_local[int(v)] for v in f])
  apex_reasons.append({"face":f.tolist(),"reason":reason,"gap":str(gap) if gap is not None else None})
 olddet,oldq=M.measurements(p,t[rows]);oldbad=int(np.count_nonzero(~M.passes(olddet,oldq)))
 newbad=int(np.count_nonzero(~M.passes(det,q)))
 used_global=np.unique(np.r_[gt.ravel(),ob_global.ravel(),shell_global.ravel(),expected_shell.ravel()])
 loc={int(g):i for i,g in enumerate(used_global)}
 ip=M.integer_points(bp[used_global])
 exact_det=[M.exact_det(ip,[loc[int(x)] for x in tet]) for tet in gt]
 exact_positive=all(x>0 for x in exact_det)
 oldflux=flux(ip,shell_global,loc);newflux=flux(ip,expected_shell,loc);tetflux=sum(exact_det)
 volume_scale=2.0**(-447)/6.0
 report={
  "status":"PASS_LOCAL_TETGEN_MESH_DIAGNOSTIC_SOURCE_ONLY" if exact_positive and not missing_chain and not extra_chain and patch_topology and chain_equal else "FAIL_LOCAL_TETGEN_MESH_DIAGNOSTIC",
  "physical_admission":False,"native_steps":0,"input_path":str(INPUT),"input_sha256":pin["candidate"],
  "prepared_shell_path":str(SHELL),"prepared_shell_sha256":pin["shell"],"remesher_path":str(ROOT/"remesh_fat614_compact_cavities_484_root.py"),"remesher_sha256":pin["remesher"],
  "tetgen_return_code":200,"tetgen_failure":"-YY boundary Steiner rejection; original failure retained","tetgen_artifact_sha256":pin,
  "source_cavity_cell_rows":len(rows),"source_cavity_boundary_faces":len(lf),"source_cavity_points":len(grows),
  "emitted_node_count":len(ids),"emitted_tet_count":len(ot),"emitted_boundary_face_count":len(ob_local),
  "new_output_points_after_F32_map":len(new_points),"new_output_boundary_points":len(extra_boundary),
  "new_output_interior_points":len(set(map(int,gm))-set(range(len(p)))-set(extra_boundary)),
  "single_boundary_point":{"global_volume_row":int(edge_point),"xyz_f32":bp[edge_point].tolist(),"sha256_f32_bytes":hashlib.sha256(bp[edge_point].tobytes()).hexdigest(),
   "source_edge_global_volume_rows":[131938,132166],"source_edge_typed_D_rows":[6570,6798],"source_typed_D_parent_face_rows":[201274,197352],
   "source_parent_triangles_global":[f.tolist() for f in retired],"edge_parameter_F32":param,"inside_parent_segment":bool(0<param<1),
   "perpendicular_residual_m":edge_res,"max_source_face_plane_residual_m":max(plane),"exact_F32_collinear_with_edge":exact_collinear,
   "raw_TetGen_F64_xyz":xyz64[np.flatnonzero(gm==edge_point)[0]].tolist()},
  "boundary_refinement":{"old_parent_face_rows_in_shell":retired_shell_rows,"output_child_face_count":len(patchfaces),"output_child_faces_global":patchfaces.tolist(),
   "four_vertex_edge_split_topology":patch_topology,"oriented_shell_residual_missing":len(missing_chain),"oriented_shell_residual_extra":len(extra_chain),
   "oriented_patch_outer_chain_preserved":chain_equal,"old_patch_faces_global":oldpatch.tolist()},
  "tetrahedra":{"all_F32_determinants_positive":bool(np.all(det>0)),"nonpositive_F32_count":int(np.count_nonzero(det<=0)),
   "all_exact_F32_determinants_positive":bool(exact_positive),"exact_nonpositive_count":int(sum(x<=0 for x in exact_det)),
   "minimum_exact_det6_integer":str(min(exact_det)),"minimum_F32_det6_m3":float(det.min()),
   "native_quality_failures":newbad,"old_component_quality_failures":oldbad,"quality_failure_delta":newbad-oldbad,
   "material_all_16":bool(np.all(attrs==16)),"duplicate_unordered_tetrahedra":int(len(ot)-len(np.unique(np.sort(ot,axis=1),axis=0))),
   "exact_tet_flux_equals_refined_shell_flux":tetflux==newflux,"exact_original_shell_flux_integer":str(oldflux),
   "exact_refined_shell_flux_integer":str(newflux),"exact_tet_flux_integer":str(tetflux),
   "refinement_volume_delta_m3":float((newflux-oldflux)*volume_scale)},
  "new_target_face_joint_apex":{"joint_score_min":float(np.min(patch_score)) if len(patch_score) else None,
   "proven_exact_apex_failures":sum(z["reason"] is not None for z in apex_reasons),"per_face":apex_reasons},
  "limitations":["Read-only diagnosis of TetGen output; no candidate D or Fat mesh was written.",
   "F32 boundary point is off the old source edge and requires an explicit typed D target refinement.",
   "No complete O/I/D/wall/all185 audit, 697 load, whole-volume SAT, P1/mass rebind, or native admission."],
  "elapsed_s":time.monotonic()-tic}
 (outdir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);run(ap.parse_args().output)

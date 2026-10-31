#!/usr/bin/env python3
"""Independent readback validation of 742's 750-style compact canonical output."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def input_unions(z,n):
 direct=np.asarray(z.get("output_cell_to_input_cell",np.arange(n,dtype=np.int64)),np.int64)
 if direct.shape!=(n,):raise ValueError("bad input cell map")
 changed=np.flatnonzero(direct<0)
 off=np.asarray(z.get("replacement_parent_offsets",np.empty(0,np.int64)),np.int64)
 vals=np.asarray(z.get("replacement_parent_cell_rows",np.empty(0,np.int64)),np.int64)
 out=[None]*n
 if len(off)==len(changed)+1 and len(off)>0 and off[0]==0 and off[-1]==len(vals):
  for j,r in enumerate(changed):
   out[int(r)]=tuple(sorted(set(map(int,vals[off[j]:off[j+1]])))) or (int(r),)
 for r in range(n):
  if out[r] is None:
   if int(direct[r])<0:raise ValueError("input ancestry is missing")
   out[r]=(int(direct[r]),)
 return out
def replay(z,ops):
 points=np.asarray(z["points_world_m"],np.float32).copy()
 t=np.asarray(z["tetrahedra"],np.int64).copy();mat=np.asarray(z["material_indices"],np.int32).copy()
 n0=len(t);lin=np.arange(n0,dtype=np.int64);unions=input_unions(z,n0);touched=set()
 point_parents=[];op_offsets=[0];op_rows=[];child_offsets=[0]
 for op in ops:
  a,b=sorted(map(int,op["edge"]))
  star=np.flatnonzero(np.any(t==a,axis=1)&np.any(t==b,axis=1))
  if not np.array_equal(star,np.asarray(op["star_rows"],np.int64)):raise ValueError("full operation star differs")
  x=((points[a].astype(np.float64)+points[b].astype(np.float64))*0.5).astype(np.float32)
  if not np.array_equal(x.view(np.uint32),np.asarray(op["new_point_world_m_f32"],np.float32).view(np.uint32)):
   raise ValueError("operation midpoint differs")
  if int(op["new_point_row"])!=len(points):raise ValueError("operation point row differs")
  parents=sorted(set(int(lin[int(r)]) for r in star))
  point_parents.append(parents)
  op_rows.extend(parents);op_offsets.append(len(op_rows))
  children=np.asarray(op["children_tetrahedra"],np.int64).reshape(-1,4)
  if len(children)!=2*len(star):raise ValueError("operation child shape differs")
  new_point_index=len(points)
  nt=[];nm=[];nl=[];nu=[]
  for i,r0 in enumerate(star):
   r=int(r0);left=t[r].copy();right=t[r].copy()
   left[np.flatnonzero(left==b)[0]]=new_point_index
   right[np.flatnonzero(right==a)[0]]=new_point_index
   if not np.array_equal(children[2*i],left) or not np.array_equal(children[2*i+1],right):
    raise ValueError("operation children differ from midpoint bisection")
   t[r]=left;nt.append(right);nm.append(mat[r]);nl.append(lin[r]);nu.append(unions[r]);touched.add(int(lin[r]))
  t=np.concatenate((t,np.asarray(nt,np.int64).reshape(-1,4)),axis=0)
  mat=np.r_[mat,np.asarray(nm,np.int32)];lin=np.r_[lin,np.asarray(nl,np.int64)];unions.extend(nu)
  points=np.vstack((points,x.reshape(1,3)));child_offsets.append(child_offsets[-1]+len(children))
 return points,t,mat,lin,unions,touched,point_parents,op_offsets,op_rows,child_offsets

def verify(state_path,raw_report_path,role,raw_receipt_path,canonicalization_receipt_path):
 state_path=Path(state_path).resolve();raw_report_path=Path(raw_report_path).resolve()
 raw_receipt_path=Path(raw_receipt_path).resolve();canonicalization_receipt_path=Path(canonicalization_receipt_path).resolve()
 state=json.loads(state_path.read_text());report=json.loads(raw_report_path.read_text())
 raw_receipt=json.loads(raw_receipt_path.read_text());crec=json.loads(canonicalization_receipt_path.read_text())
 if report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0:
  raise ValueError("raw report not source-only")
 rr=report["roles"][role]
 ip=Path(rr["input"]["path"]);rawp=Path(rr["output"]["path"]);logp=Path(rr["operation_log"]["path"])
 if sha(ip)!=rr["input"]["sha256"] or sha(rawp)!=rr["output"]["sha256"] or sha(logp)!=rr["operation_log"]["sha256"]:
  raise ValueError("raw proposal pin mismatch")
 if raw_receipt.get("status")!="PASS_INDEPENDENT_COMPOUND_BISECTION_REPLAY" or raw_receipt.get("role")!=role:
  raise ValueError("raw independent geometry replay did not pass")
 if raw_receipt.get("input_sha256")!=sha(ip) or raw_receipt.get("proposal_sha256")!=sha(rawp) or raw_receipt.get("operation_log_sha256")!=sha(logp):
  raise ValueError("raw replay receipt is not bound to this proposal")
 if crec.get("status")!="PASS_CANONICAL_IMMEDIATE_MAP_NORMALIZATION" or crec.get("role")!=role:
  raise ValueError("canonicalization did not pass")
 cp=Path(crec["canonical_output"]["path"])
 if sha(cp)!=crec["canonical_output"]["sha256"]:raise ValueError("canonical output hash mismatch")
 z=load_npz(ip);raw=load_npz(rawp);out=load_npz(cp);ops=json.loads(logp.read_text())
 points,t,mat,lin,unions,touched,pointparents,opoff,oprows,childoff=replay(z,ops)
 if not np.array_equal(t,np.asarray(raw["tetrahedra"],np.int64)):raise ValueError("raw output geometry replay mismatch")
 if not np.array_equal(mat,np.asarray(raw["material_indices"],np.int32)):raise ValueError("raw material replay mismatch")
 if not np.array_equal(np.asarray(raw["points_world_m"],np.float32).view(np.uint32),points.view(np.uint32)):
  raise ValueError("raw point output replay mismatch")
 n0=len(z["tetrahedra"]);retained=np.asarray([r for r in range(n0) if r not in touched],np.int64)
 parents=sorted(touched);replacement=[];offsets=[0]
 for parent in parents:
  rows=np.flatnonzero(lin==parent)
  if len(rows)==0:raise ValueError("changed input parent has no output children")
  replacement.extend(map(int,rows));offsets.append(len(replacement))
 repl=np.asarray(replacement,np.int64);order=np.r_[retained,repl]
 if len(order)!=len(t) or not np.array_equal(np.sort(order),np.arange(len(t),dtype=np.int64)):
  raise ValueError("canonical cell order is not a permutation")
 exp_t=t[order];exp_m=mat[order]
 if not np.array_equal(np.asarray(out["tetrahedra"],np.int64),exp_t):raise ValueError("canonical tets are not retained-prefix/replacement-suffix")
 if not np.array_equal(np.asarray(out["material_indices"],np.int32),exp_m):raise ValueError("canonical material order mismatch")
 if not np.array_equal(np.asarray(out["points_world_m"],np.float32).view(np.uint32),points.view(np.uint32)):
  raise ValueError("canonical point bytes differ")
 if "points_world_m_f32" in out and not np.array_equal(np.asarray(out["points_world_m_f32"],np.float32).view(np.uint32),points.view(np.uint32)):
  raise ValueError("canonical Float32 point alias differs")
 origin=np.r_[retained,np.full(len(repl),-1,np.int64)]
 inverse=np.full(n0,-1,np.int64);inverse[retained]=np.arange(len(retained),dtype=np.int64)
 for key,expected in (("output_cell_to_input_cell",origin),("input_cell_to_output_cell",inverse),
                      ("retained_input_cell_rows",retained),("replacement_tet_offsets",np.asarray(offsets,np.int64)),
                      ("replacement_tetrahedra",exp_t[len(retained):])):
  if not np.array_equal(np.asarray(out[key],np.int64),np.asarray(expected,np.int64)):raise ValueError("canonical immediate cell map differs: "+key)
 po=[0];pv=[]
 for row in repl:
  pv.extend(sorted(set(map(int,unions[int(row)]))));po.append(len(pv))
 if not np.array_equal(np.asarray(out["replacement_parent_offsets"],np.int64),np.asarray(po,np.int64)):raise ValueError("canonical replacement parent offsets mismatch")
 if not np.array_equal(np.asarray(out["replacement_parent_cell_rows"],np.int64),np.asarray(pv,np.int64)):raise ValueError("canonical replacement parent union mismatch")
 oldabs=np.asarray(z.get("edge_bisection_absolute_output_cell_to_input_cell",
                         z.get("output_cell_to_input_cell",np.arange(n0,dtype=np.int64))),np.int64)
 absolute=np.r_[oldabs[retained].copy(),np.full(len(repl),-1,np.int64)]
 for j,row in enumerate(repl):
  u=tuple(sorted(set(map(int,unions[int(row)]))));absolute[len(retained)+j]=int(u[0]) if len(u)==1 else -1
 if not np.array_equal(np.asarray(out["edge_bisection_absolute_output_cell_to_input_cell"],np.int64),absolute):
  raise ValueError("canonical absolute cell ancestry mismatch")
 # Reconstruct per-operation immediate parents and immediate point-parent CSR.
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_parent_cell_offsets"],np.int64),np.asarray(opoff,np.int64)):
  raise ValueError("operation parent offsets mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_parent_cell_rows"],np.int64),np.asarray(oprows,np.int64)):
  raise ValueError("operation parent rows mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_child_tet_offsets"],np.int64),np.asarray(childoff,np.int64)):
  raise ValueError("operation child offsets mismatch")
 hist=np.asarray([c for op in ops for c in op["children_tetrahedra"]],np.int64).reshape(-1,4)
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_child_tetrahedra"],np.int64),hist):
  raise ValueError("operation child log mismatch")
 npts=len(points);nnew=len(ops);npoint0=len(z["points_world_m"]);identity=np.arange(npoint0,dtype=np.int64);out_to_in=np.r_[identity,np.full(nnew,-1,np.int64)]
 for key,expected in (("input_node_to_output_node",identity),("output_node_to_input_node",out_to_in),
  ("output_point_to_input_point_row",out_to_in),("input_point_to_output_point_row",identity),
  ("source_I_vertex_quotient",out_to_in),("source_input_vertex_quotient",out_to_in),
  ("source_input_node_to_output_node",identity),("quality_milp_input_node_to_output_node",identity),
  ("quality_milp_output_node_to_input_node",out_to_in)):
  if key not in out or not np.array_equal(np.asarray(out[key],np.int64),expected):raise ValueError("750 immediate point map mismatch: "+key)
 if "source_interior_vertex_quotient" in z:
  if np.asarray(z["source_interior_vertex_quotient"]).shape!=(npoint0,):raise ValueError("source interior quotient old domain mismatch")
  expected=np.r_[np.asarray(z["source_interior_vertex_quotient"],np.int64),np.full(nnew,-1,np.int64)]
  if not np.array_equal(np.asarray(out["source_interior_vertex_quotient"],np.int64),expected):raise ValueError("historical interior quotient mismatch")
 poff=np.zeros(npoint0+1,np.int64);prows=[]
 for pp in pointparents:prows.extend(pp);poff=np.r_[poff,len(prows)]
 if not np.array_equal(np.asarray(out["output_point_parent_offsets"],np.int64),poff):raise ValueError("immediate point parent offsets not reset/rebuilt")
 if not np.array_equal(np.asarray(out["output_point_parent_input_cell_rows"],np.int64),np.asarray(prows,np.int64)):
  raise ValueError("immediate point parent rows mismatch")
 oldoff=np.asarray(z.get("output_point_parent_offsets",np.zeros(n0+1,np.int64)),np.int64)
 oldrows=np.asarray(z.get("output_point_parent_input_cell_rows",np.empty(0,np.int64)),np.int64)
 for key,expected in (("steiner_parent_point_csr_input_offsets",oldoff),
                      ("steiner_parent_point_csr_input_rows",oldrows),
                      ("steiner_parent_point_csr_input_output_point_parent_offsets",oldoff),
                      ("steiner_parent_point_csr_input_output_point_parent_input_cell_rows",oldrows)):
  if key not in out or not np.array_equal(np.asarray(out[key],np.int64),expected):raise ValueError("prior point CSR not archived: "+key)
 if str(np.asarray(out["steiner_point_csr_policy"]).reshape(-1)[0])!="RESET_IMMEDIATE_OLD_POINT_PREFIX_AND_ARCHIVE_PRIOR_ABSOLUTE_PARENT":
  raise ValueError("point CSR policy marker mismatch")
 # All authored point lanes are passed through from the raw validated producer.
 mutable={"tetrahedra","material_indices","output_cell_to_input_cell","input_cell_to_output_cell","retained_input_cell_rows",
  "replacement_tetrahedra","replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets",
  "edge_bisection_absolute_output_cell_to_input_cell","edge_bisection_replacement_parent_cell_offsets",
  "edge_bisection_replacement_parent_cell_rows","edge_bisection_replacement_child_tet_offsets",
  "edge_bisection_replacement_child_tetrahedra","edge_bisection_parent_cell_rows",
  "edge_bisection_operation_parent_offsets","edge_bisection_operation_parent_rows","input_node_to_output_node",
  "output_node_to_input_node","output_point_to_input_point_row","input_point_to_output_point_row",
  "source_I_vertex_quotient","source_input_vertex_quotient","source_input_node_to_output_node",
  "quality_milp_input_node_to_output_node","quality_milp_output_node_to_input_node",
  "source_interior_vertex_quotient","output_point_parent_offsets","output_point_parent_input_cell_rows",
  "point_parent_offsets","point_parent_rows","output_point_parent_cell_rows",
  "steiner_parent_point_csr_input_path","steiner_parent_point_csr_input_sha256",
  "steiner_parent_point_csr_input_offsets","steiner_parent_point_csr_input_rows",
  "steiner_parent_point_csr_input_output_point_parent_offsets",
  "steiner_parent_point_csr_input_output_point_parent_input_cell_rows","steiner_point_csr_policy",
  "metadata_finalizer_point_map_status","metadata_finalizer_point_parent_status",
  "edge_bisection_canonical_layout","edge_bisection_raw_candidate_path","edge_bisection_raw_candidate_sha256",
  "edge_bisection_raw_independent_replay_receipt_path","edge_bisection_raw_independent_replay_receipt_sha256"}
 canonicalizer_additions={
  "input_point_to_output_point_row","source_I_vertex_quotient","source_input_vertex_quotient",
  "source_input_node_to_output_node","quality_milp_input_node_to_output_node","quality_milp_output_node_to_input_node",
  "steiner_parent_point_csr_input_path","steiner_parent_point_csr_input_sha256",
  "steiner_parent_point_csr_input_offsets","steiner_parent_point_csr_input_rows",
  "steiner_parent_point_csr_input_output_point_parent_offsets",
  "steiner_parent_point_csr_input_output_point_parent_input_cell_rows","steiner_point_csr_policy",
  "edge_bisection_canonical_layout","edge_bisection_raw_candidate_path","edge_bisection_raw_candidate_sha256",
  "edge_bisection_raw_independent_replay_receipt_path","edge_bisection_raw_independent_replay_receipt_sha256"}
 if set(out)!=set(raw)|(canonicalizer_additions-set(raw)):
  raise ValueError("canonical npz key set changed: unexpected="+repr(sorted(set(out)-set(raw)-canonicalizer_additions))+" missing="+repr(sorted(set(raw)-set(out))))
 for key in raw:
  if key in mutable:continue
  if not np.array_equal(np.asarray(out[key]),np.asarray(raw[key])):raise ValueError("unexpected noncanonical array mutation: "+key)
 # Geometry is a row permutation of an independently exact-positive raw proposal.
 if sha(raw_receipt_path)=="" or raw_receipt.get("exact_total_det6_delta")!="0":
  raise ValueError("raw exact geometry certificate absent")
 return {"status":"PASS_INDEPENDENT_750_STYLE_CANONICAL_READBACK","role":role,
  "input_sha256":sha(ip),"raw_proposal_sha256":sha(rawp),"raw_replay_receipt_sha256":sha(raw_receipt_path),
  "canonical_output_sha256":sha(cp),"input_cells":n0,"output_cells":len(exp_t),
  "retained_cells":len(retained),"replacement_cells":len(repl),"appended_points":nnew,
  "immediate_cell_maps":"750_COMPACT_RETAINED_PREFIX","immediate_point_parent_csr":"750_RESET_AND_ARCHIVE",
  "exact_geometry_inherited_by_row_permutation":True,"physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--state",type=Path,required=True);ap.add_argument("--report",type=Path,required=True)
 ap.add_argument("--role",choices=("Skin_current","Fat_current"),required=True)
 ap.add_argument("--raw-replay",type=Path,required=True);ap.add_argument("--canonicalization",type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args();r=verify(a.state,a.report,a.role,a.raw_replay,a.canonicalization)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+"\n")
 print(json.dumps(r,separators=(",",":")))
 return 0
if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""750-style canonical immediate map normalization for a raw, independently replayed 742 proposal."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OWNER=ROOT/"quality-edge-bisection-683-root"/"quality_edge_bisection_683_helper.py"
SPEC=importlib.util.spec_from_file_location("quality_bisection742_canonical_owner",OWNER)
QB=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=QB;SPEC.loader.exec_module(QB)
M=QB.M

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def resolve(v,state_path):
 p=Path(str(v));return p if p.is_absolute() else (state_path.parent/p).resolve()

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def point_f32(z):
 p=np.asarray(z["points_world_m"],np.float32)
 if p.ndim!=2 or p.shape[1]!=3:raise ValueError("bad point array")
 if "points_world_m_f32" in z and not np.array_equal(p.view(np.uint32),np.asarray(z["points_world_m_f32"],np.float32).view(np.uint32)):
  raise ValueError("Float32 point alias mismatch")
 return p

def input_unions(z,n):
 direct=np.asarray(z.get("output_cell_to_input_cell",np.arange(n,dtype=np.int64)),np.int64)
 if direct.shape!=(n,):raise ValueError("input direct cell map malformed")
 changed=np.flatnonzero(direct<0)
 off=np.asarray(z.get("replacement_parent_offsets",np.empty(0,np.int64)),np.int64)
 vals=np.asarray(z.get("replacement_parent_cell_rows",np.empty(0,np.int64)),np.int64)
 out=[None]*n
 if len(off)==len(changed)+1 and len(off)>0 and off[0]==0 and off[-1]==len(vals):
  for j,r in enumerate(changed):
   u=tuple(sorted(set(map(int,vals[off[j]:off[j+1]]))))
   out[int(r)]=u if u else (int(r),)
 for r in range(n):
  if out[r] is None:
   if int(direct[r])<0:raise ValueError("changed input row missing full ancestry")
   out[r]=(int(direct[r]),)
 return out

def replay_rows(z,ops):
 p0=point_f32(z);t=np.asarray(z["tetrahedra"],np.int64).copy()
 mats=np.asarray(z["material_indices"],np.int32).copy()
 if mats.shape!=(len(t),):raise ValueError("material row mismatch")
 n0=len(t);lineage=np.arange(n0,dtype=np.int64);unions=input_unions(z,n0)
 touched=set();op_parent_offsets=[0];op_parent_rows=[];point_parents=[];child_offsets=[0]
 points=p0.copy()
 for oi,op in enumerate(ops):
  a,b=sorted(map(int,op["edge"]))
  if a<0 or b>=len(points) or a==b:raise ValueError("bad operation edge")
  star=np.flatnonzero(np.any(t==a,axis=1)&np.any(t==b,axis=1))
  if not np.array_equal(star,np.asarray(op["star_rows"],np.int64)):raise ValueError("raw operation edge-star mismatch")
  x=(((points[a].astype(np.float64)+points[b].astype(np.float64))*0.5).astype(np.float32))
  if not np.array_equal(x.view(np.uint32),np.asarray(op["new_point_world_m_f32"],np.float32).view(np.uint32)):
   raise ValueError("raw operation midpoint mismatch")
  if int(op["new_point_row"])!=len(points):raise ValueError("raw operation point row mismatch")
  parents=sorted(set(int(lineage[int(r)]) for r in star))
  if parents!=op.get("point_parent_union"):raise ValueError("raw point-parent lineage mismatch")
  op_rows=sorted(set(parents));op_parent_rows.extend(op_rows);op_parent_offsets.append(len(op_parent_rows))
  point_parents.append(op_rows)
  child=np.asarray(op["children_tetrahedra"],np.int64).reshape(-1,4)
  if len(child)!=2*len(star):raise ValueError("raw operation child count mismatch")
  new_point_index=len(points)
  newt=[];newm=[];newl=[];newu=[]
  for i,r0 in enumerate(star):
   r=int(r0)
   if not np.array_equal(child[2*i],np.where(t[r]==b, new_point_index, t[r])) or not np.array_equal(child[2*i+1],np.where(t[r]==a, new_point_index, t[r])):
    raise ValueError("raw operation child construction mismatch")
   t[r]=child[2*i];newt.append(child[2*i+1]);newm.append(mats[r])
   newl.append(lineage[r]);newu.append(unions[r]);touched.add(int(lineage[r]))
  t=np.concatenate((t,np.asarray(newt,np.int64).reshape(-1,4)),axis=0)
  mats=np.r_[mats,np.asarray(newm,np.int32)]
  lineage=np.r_[lineage,np.asarray(newl,np.int64)]
  unions.extend(newu)
  points=np.vstack((points,x.reshape(1,3)))
  child_offsets.append(child_offsets[-1]+len(child))
 return p0,t,mats,lineage,unions,touched,op_parent_offsets,op_parent_rows,child_offsets,point_parents

def canonicalize(state_path,raw_report_path,role,raw_receipt_path,raw_receipt_sha,output_path,receipt_path):
 state_path=Path(state_path).resolve();raw_report_path=Path(raw_report_path).resolve()
 raw_receipt_path=Path(raw_receipt_path).resolve();output_path=Path(output_path).resolve();receipt_path=Path(receipt_path).resolve()
 state=json.loads(state_path.read_text());report=json.loads(raw_report_path.read_text())
 if report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0 or report.get("mass_transport")!="NOT_RUN":
  raise ValueError("raw report carries physical/mass admission")
 rr=report["roles"][role]
 ip=Path(rr["input"]["path"]);rawpath=Path(rr["output"]["path"]);log=Path(rr["operation_log"]["path"])
 for rec,path in ((rr["input"],ip),(rr["output"],rawpath),(rr["operation_log"],log)):
  if sha(path)!=rec["sha256"]:raise ValueError("raw producer input/output/log pin mismatch")
 if sha(raw_receipt_path)!=raw_receipt_sha:raise ValueError("independent raw replay receipt hash mismatch")
 z=load_npz(ip);raw=load_npz(rawpath);ops=json.loads(log.read_text())
 p0,t,m,lineage,unions,touched,opoff,oprows,childoff,pointparents=replay_rows(z,ops)
 if not np.array_equal(t,np.asarray(raw["tetrahedra"],np.int64)):raise ValueError("raw tets not equal to reconstructed operation log")
 if not np.array_equal(m,np.asarray(raw["material_indices"],np.int32)):raise ValueError("raw material rows mismatch")
 if not np.array_equal(np.asarray(raw["points_world_m"],np.float32).view(np.uint32),
                       np.vstack((p0,np.asarray([op["new_point_world_m_f32"] for op in ops],np.float32))).view(np.uint32)):
  raise ValueError("raw point output mismatch")
 n0=len(z["tetrahedra"]);npoint0=len(p0);new_n=len(ops)
 touched_rows=np.asarray(sorted(touched),np.int64)
 retained=np.asarray([r for r in range(n0) if r not in touched],np.int64)
 blocks=[];block_parent_ids=[];block_offsets=[0]
 for parent in touched_rows:
  rows=np.flatnonzero(lineage==int(parent))
  if len(rows)==0:raise ValueError("changed parent has no final descendants")
  blocks.extend(map(int,rows));block_parent_ids.append(int(parent));block_offsets.append(len(blocks))
 replacement_rows=np.asarray(blocks,np.int64)
 order=np.r_[retained,replacement_rows]
 if len(order)!=len(t) or not np.array_equal(np.sort(order),np.arange(len(t),dtype=np.int64)):
  raise ValueError("retained/replacement partition is not a permutation")
 canonical_t=t[order].copy();canonical_m=m[order].copy()
 out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in raw.items()}
 out["tetrahedra"]=canonical_t;out["material_indices"]=canonical_m
 origin=np.r_[retained,np.full(len(replacement_rows),-1,np.int64)]
 inverse=np.full(n0,-1,np.int64);inverse[retained]=np.arange(len(retained),dtype=np.int64)
 out["output_cell_to_input_cell"]=origin
 out["input_cell_to_output_cell"]=inverse
 out["retained_input_cell_rows"]=retained
 out["replacement_tetrahedra"]=canonical_t[len(retained):].copy()
 po=[0];pv=[]
 for r in replacement_rows:
  pv.extend(sorted(set(map(int,unions[int(r)]))));po.append(len(pv))
 out["replacement_parent_offsets"]=np.asarray(po,np.int64)
 out["replacement_parent_cell_rows"]=np.asarray(pv,np.int64)
 out["replacement_tet_offsets"]=np.asarray(block_offsets,np.int64)
 old_abs=np.asarray(z.get("edge_bisection_absolute_output_cell_to_input_cell",
                          z.get("output_cell_to_input_cell",np.arange(n0,dtype=np.int64))),np.int64)
 if old_abs.shape!=(n0,):raise ValueError("input absolute cell map malformed")
 absolute=np.r_[old_abs[retained].copy(),np.full(len(replacement_rows),-1,np.int64)]
 for j,r in enumerate(replacement_rows):
  u=tuple(sorted(set(map(int,unions[int(r)]))))
  absolute[len(retained)+j]=int(u[0]) if len(u)==1 else -1
 out["edge_bisection_absolute_output_cell_to_input_cell"]=absolute
 # Operation-level parent rows are reset to the immediate input cell namespace.
 out["edge_bisection_replacement_parent_cell_offsets"]=np.asarray(opoff,np.int64)
 out["edge_bisection_replacement_parent_cell_rows"]=np.asarray(oprows,np.int64)
 out["edge_bisection_replacement_child_tet_offsets"]=np.asarray(childoff,np.int64)
 out["edge_bisection_replacement_child_tetrahedra"]=np.asarray(
     [c for op in ops for c in op["children_tetrahedra"]],np.int64).reshape(-1,4)
 out["edge_bisection_parent_cell_rows"]=np.asarray(oprows,np.int64)
 out["edge_bisection_operation_parent_offsets"]=np.asarray(opoff,np.int64)
 out["edge_bisection_operation_parent_rows"]=np.asarray(oprows,np.int64)
 # 750 immediate point namespace: identity old prefix; append-only new points.
 identity=np.arange(npoint0,dtype=np.int64);out_to_in=np.r_[identity,np.full(new_n,-1,np.int64)]
 out["input_node_to_output_node"]=identity.copy()
 out["output_node_to_input_node"]=out_to_in.copy()
 out["output_point_to_input_point_row"]=out_to_in.copy()
 out["input_point_to_output_point_row"]=identity.copy()
 out["source_I_vertex_quotient"]=out_to_in.copy()
 out["source_input_vertex_quotient"]=out_to_in.copy()
 out["source_input_node_to_output_node"]=identity.copy()
 out["quality_milp_input_node_to_output_node"]=identity.copy()
 out["quality_milp_output_node_to_input_node"]=out_to_in.copy()
 if "source_interior_vertex_quotient" in z:
  oldq=np.asarray(z["source_interior_vertex_quotient"],np.int64)
  if oldq.shape!=(npoint0,):raise ValueError("source interior quotient has wrong old domain")
  out["source_interior_vertex_quotient"]=np.r_[oldq,np.full(new_n,-1,np.int64)]
 # Reset immediate point-parent CSR and preserve prior CSR as immutable history.
 oldoff=np.asarray(z.get("output_point_parent_offsets",np.zeros(n0+1,np.int64)),np.int64)
 oldrows=np.asarray(z.get("output_point_parent_input_cell_rows",np.empty(0,np.int64)),np.int64)
 if oldoff.shape!=(npoint0+1,) or oldoff[0]!=0 or oldoff[-1]!=len(oldrows) or np.any(np.diff(oldoff)<0):
  raise ValueError("prior point-parent CSR malformed")
 for key in ("point_parent_offsets",):
  if key in z and not np.array_equal(np.asarray(z[key],np.int64),oldoff):raise ValueError("historical point offset alias mismatch")
 for key in ("point_parent_rows","output_point_parent_cell_rows"):
  if key in z and not np.array_equal(np.asarray(z[key],np.int64),oldrows):raise ValueError("historical point row alias mismatch")
 newoff=[0]*(npoint0+1);newrows=[];cursor=0
 for parents in pointparents:
  newrows.extend(parents);cursor+=len(parents);newoff.append(cursor)
 p_off=np.asarray(newoff,np.int64);p_rows=np.asarray(newrows,np.int64)
 out["output_point_parent_offsets"]=p_off
 out["output_point_parent_input_cell_rows"]=p_rows
 for key in ("point_parent_offsets",):
  if key in z:out[key]=p_off.copy()
 for key in ("point_parent_rows","output_point_parent_cell_rows"):
  if key in z:out[key]=p_rows.copy()
 out["steiner_parent_point_csr_input_path"]=np.asarray([str(ip)])
 out["steiner_parent_point_csr_input_sha256"]=np.asarray([sha(ip)])
 out["steiner_parent_point_csr_input_offsets"]=oldoff.copy()
 out["steiner_parent_point_csr_input_rows"]=oldrows.copy()
 out["steiner_parent_point_csr_input_output_point_parent_offsets"]=oldoff.copy()
 out["steiner_parent_point_csr_input_output_point_parent_input_cell_rows"]=oldrows.copy()
 out["steiner_point_csr_policy"]=np.asarray("RESET_IMMEDIATE_OLD_POINT_PREFIX_AND_ARCHIVE_PRIOR_ABSOLUTE_PARENT")
 if "metadata_finalizer_point_map_status" in z:out["metadata_finalizer_point_map_status"]=np.asarray("APPENDED_EDGE_DERIVED_POINTS_NO_LEGACY_BINDING")
 if "metadata_finalizer_point_parent_status" in z:out["metadata_finalizer_point_parent_status"]=np.asarray("COMPLETE_FULL_EDGE_STAR_PARENT_CELL_CSR")
 out["edge_bisection_canonical_layout"]=np.asarray("numi.h706.750.compacted-replacement-suffix-and-reset-immediate-point-csr.v1")
 out["edge_bisection_raw_candidate_path"]=np.asarray([str(rawpath)])
 out["edge_bisection_raw_candidate_sha256"]=np.asarray([sha(rawpath)])
 out["edge_bisection_raw_independent_replay_receipt_path"]=np.asarray([str(raw_receipt_path)])
 out["edge_bisection_raw_independent_replay_receipt_sha256"]=np.asarray([raw_receipt_sha])
 output_path.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output_path,**out)
 result={"schema":"numi.h706.compound-edge-bisection-742-canonicalization.v1",
  "status":"PASS_CANONICAL_IMMEDIATE_MAP_NORMALIZATION","role":role,
  "input":{"path":str(ip),"sha256":sha(ip)},"raw_proposal":{"path":str(rawpath),"sha256":sha(rawpath)},
  "raw_replay_receipt":{"path":str(raw_receipt_path),"sha256":raw_receipt_sha},
  "canonical_output":{"path":str(output_path),"sha256":sha(output_path)},
  "input_cells":n0,"retained_input_cells":len(retained),"replacement_cells":len(replacement_rows),
  "input_points":len(p0),"appended_points":new_n,"operation_count":len(ops),
  "replacement_parent_rows":len(pv),"immediate_old_point_parent_rows_reset":True,
  "prior_point_parent_csr_archived":True,"physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"}
 receipt_path.parent.mkdir(parents=True,exist_ok=True);receipt_path.write_text(json.dumps(result,indent=2)+"\n")
 return result

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--state",type=Path,required=True);ap.add_argument("--report",type=Path,required=True)
 ap.add_argument("--role",choices=("Skin_current","Fat_current"),required=True)
 ap.add_argument("--raw-replay",type=Path,required=True);ap.add_argument("--raw-replay-sha256",required=True)
 ap.add_argument("--output",type=Path,required=True);ap.add_argument("--receipt",type=Path,required=True)
 a=ap.parse_args();r=canonicalize(a.state,a.report,a.role,a.raw_replay,a.raw_replay_sha256,a.output,a.receipt)
 print(json.dumps(r,separators=(",",":")))
 return 0
if __name__=="__main__":raise SystemExit(main())

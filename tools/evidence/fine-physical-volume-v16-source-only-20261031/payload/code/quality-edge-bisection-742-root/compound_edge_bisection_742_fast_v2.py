#!/usr/bin/env python3
"""Bounded compound interior-edge bisection proposal. Source-only; no admission."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, itertools, json, math, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OWNER=ROOT/"quality-edge-bisection-683-root"/"quality_edge_bisection_683_helper.py"
_spec=importlib.util.spec_from_file_location("quality_bisection742_owner", OWNER)
QB=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=QB;_spec.loader.exec_module(QB)
M=QB.M
V739_PATH=ROOT/"quality-edge-bisection-739-root"/"edge_bisection_739.py"
_spec2=importlib.util.spec_from_file_location("quality_bisection739_support", V739_PATH)
V739=importlib.util.module_from_spec(_spec2);sys.modules[_spec2.name]=V739;_spec2.loader.exec_module(V739)
ROLE_SURFACE_MAPS=V739.ROLE_SURFACE_MAPS
ROLES=("Skin_current","Fat_current")
FACES=((0,1,2),(0,3,1),(0,2,3),(1,3,2))
EDGE_PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
MAX_DEPTH=8
MAX_NEW_CELLS=256
BEAM=24

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def resolve(v,state_path):
 p=Path(str(v));return p if p.is_absolute() else (state_path.parent/p).resolve()

def face_edges(faces):
 out=set()
 for f in np.asarray(faces,np.int64):
  out.update(tuple(sorted((int(f[i]),int(f[j])))) for i,j in ((0,1),(1,2),(2,0)))
 return out

def protected_edge_set(z, role, surfaces):
 out=set()
 for k in ("boundary_faces","desired_boundary_faces"):
  if k in z: out |= face_edges(z[k])
 wk=next((k for k in z if k.endswith("_wall_faces_volume_rows")),None)
 if wk: out |= face_edges(z[wk])
 for sname,mapkey in ROLE_SURFACE_MAPS[role].items():
  if sname not in surfaces or mapkey not in z: continue
  sf=surfaces[sname]
  faces=np.asarray(sf["faces"],np.int64)
  mp=np.asarray(z[mapkey],np.int64)
  for f in faces:
   v=mp[f]
   if np.all(v>=0):
    out.update(tuple(sorted((int(v[i]),int(v[j])))) for i,j in ((0,1),(1,2),(2,0)))
 return out

class EdgeIndex:
 def __init__(self, points, tets):
  self.n=int(len(points));t=np.asarray(tets,np.int64)
  edges=np.concatenate([t[:,p] for p in EDGE_PAIRS],axis=0);edges.sort(axis=1)
  rows=np.tile(np.arange(len(t),dtype=np.int64),6)
  codes=edges[:,0]*np.int64(self.n)+edges[:,1];order=np.argsort(codes,kind="stable")
  self.codes=codes[order];self.rows=rows[order]
 def query(self,a,b):
  a,b=sorted((int(a),int(b)));code=np.int64(a)*np.int64(self.n)+np.int64(b)
  i=int(np.searchsorted(self.codes,code))
  if i==len(self.codes) or self.codes[i]!=code:return np.empty(0,np.int64)
  j=int(np.searchsorted(self.codes,code,side="right"))
  return np.unique(self.rows[i:j])

def edge_star(tets, dynamic_rows, index, edge):
 a,b=map(int,edge);rows=set(map(int,index.query(a,b)))
 for r in dynamic_rows:
  if r < len(tets) and np.any(tets[r]==a) and np.any(tets[r]==b):rows.add(int(r))
 if not rows:return np.empty(0,np.int64)
 rr=np.asarray(sorted(rows),np.int64)
 return rr[np.any(tets[rr]==a,axis=1)&np.any(tets[rr]==b,axis=1)]

def exact_average(a,b):
 mid=((np.asarray(a,np.float32).astype(np.float64)+np.asarray(b,np.float32).astype(np.float64))*0.5).astype(np.float32)
 exact=(np.asarray(a,np.float32).astype(np.float64)+np.asarray(b,np.float32).astype(np.float64))*0.5
 return mid, bool(np.array_equal(mid.astype(np.float64),exact))

def local_metrics(points,tets):
 d,q=M.measurements(points,np.asarray(tets,np.int64))
 good=np.asarray(M.passes(d,q),bool)
 return d,q,good

def score(d,q):
 return float(QB._log_deficit(d,q))

def base_unions(z,n):
 direct=np.asarray(z.get("output_cell_to_input_cell",np.arange(n,dtype=np.int64)),np.int64)
 if direct.shape!=(n,):raise ValueError("output_cell_to_input_cell domain mismatch")
 changed=np.flatnonzero(direct<0)
 off=np.asarray(z.get("replacement_parent_offsets",np.empty(0,np.int64)),np.int64)
 vals=np.asarray(z.get("replacement_parent_cell_rows",np.empty(0,np.int64)),np.int64)
 unions=[None]*n
 if len(off)==len(changed)+1 and len(off)>0 and off[0]==0 and off[-1]==len(vals):
  for i,row in enumerate(changed):
   u=tuple(sorted(set(map(int,vals[off[i]:off[i+1]]))))
   unions[int(row)]=u if u else (int(row),)
 for r in range(n):
  if unions[r] is None:
   v=int(direct[r]);unions[r]=(v,) if v>=0 else (r,)
 return unions

def clone_state(s):
 return {
  "cells":{k:{kk:(vv.copy() if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} for k,v in s["cells"].items()},
  "newpoints":{k:v.copy() for k,v in s["newpoints"].items()},
  "operations":[dict(x) for x in s["operations"]],
  "base_bad":dict(s["base_bad"]),"base_score":dict(s["base_score"]),
  "new_cells":int(s["new_cells"]),"next_cell":int(s["next_cell"]),
  "next_point":int(s["next_point"]),"dynamic":set(s["dynamic"]),
 }

def state_points(base_points,state):
 if not state["newpoints"]:return base_points
 ids=sorted(state["newpoints"])
 if ids!=list(range(len(base_points),len(base_points)+len(ids))):
  raise ValueError("branch point ids are not contiguous")
 return np.vstack((base_points,np.asarray([state["newpoints"][i] for i in ids],np.float32)))

def edge_length(base_points,state,edge):
 a,b=map(int,edge)
 return float(np.linalg.norm(point_at(base_points,state,a).astype(np.float64)-point_at(base_points,state,b).astype(np.float64)))

def commit_patch(points,tets,mats,unions,lineage,depth,dynamic,point_parent_unions,state):
 old_ncells=len(tets); old_npoints=len(points)
 expected_rows=list(range(old_ncells,int(state["next_cell"])))
 if len(expected_rows)!=int(state["new_cells"]):
  raise ValueError("branch appended-cell count is inconsistent")
 new_tets=np.empty((len(expected_rows),4),np.int64)
 new_mats=np.empty(len(expected_rows),np.int32)
 new_unions=[None]*len(expected_rows);new_lineage=np.empty(len(expected_rows),np.int64);new_depth=np.empty(len(expected_rows),np.int16)
 for row,cell in state["cells"].items():
  row=int(row)
  if row<old_ncells:
   tets[row]=cell["tet"];mats[row]=int(cell["mat"]);unions[row]=tuple(cell["union"])
   lineage[row]=int(cell["lineage"]);depth[row]=int(cell["depth"])
  else:
   j=row-old_ncells
   if j<0 or j>=len(expected_rows):raise ValueError("branch appended row out of range")
   new_tets[j]=cell["tet"];new_mats[j]=int(cell["mat"]);new_unions[j]=tuple(cell["union"])
   new_lineage[j]=int(cell["lineage"]);new_depth[j]=int(cell["depth"])
 if any(x is None for x in new_unions):raise ValueError("branch appended rows are not contiguous")
 tets=np.concatenate((tets,new_tets),axis=0);mats=np.concatenate((mats,new_mats))
 unions=list(unions)+new_unions;lineage=np.r_[lineage,new_lineage];depth=np.r_[depth,new_depth]
 point_ids=sorted(state["newpoints"])
 if point_ids!=list(range(old_npoints,old_npoints+len(point_ids))):
  raise ValueError("accepted branch point rows are not contiguous")
 if point_ids:points=np.vstack((points,np.asarray([state["newpoints"][i] for i in point_ids],np.float32)))
 dynamic=set(dynamic)|set(map(int,state["dynamic"]))
 for op in state["operations"]:
  point_parent_unions.append(tuple(sorted(set(map(int,op["point_parent_union"])))))
 return points,tets,mats,unions,lineage,depth,dynamic,point_parent_unions

def point_at(base_points,state,row):
 row=int(row)
 if row<len(base_points):return base_points[row]
 try:return state["newpoints"][row]
 except KeyError as e:raise ValueError("branch point row is not present") from e

def local_geometry(base_points,state,tets,extra=None):
 tt=np.asarray(tets,np.int64)
 if tt.size==0:return np.empty((0,3),np.float32),tt.reshape(-1,4)
 nodes=np.unique(tt.ravel())
 coords=np.empty((len(nodes),3),np.float32)
 extra_id=None if extra is None else int(extra[0])
 extra_xyz=None if extra is None else np.asarray(extra[1],np.float32)
 for i,node in enumerate(nodes):
  row=int(node)
  if extra_id is not None and row==extra_id:coords[i]=extra_xyz
  else:coords[i]=point_at(base_points,state,row)
 return coords,np.searchsorted(nodes,tt)

def branch_metrics(base_points,state,tets,extra=None):
 pp,tt=local_geometry(base_points,state,tets,extra)
 return M.measurements(pp,tt)

def branch_exact_dets(base_points,state,tets,extra=None):
 pp,tt=local_geometry(base_points,state,tets,extra)
 return QB._exact_det_rows(pp,tt)

def patch_stats(base_points,state):
 rows=sorted(state["cells"])
 tt=np.asarray([state["cells"][r]["tet"] for r in rows],np.int64)
 d,q=branch_metrics(base_points,state,tt)
 g=np.asarray(M.passes(d,q),bool)
 bad=int(np.count_nonzero(~g));sc=score(d,q)
 return rows,d,q,g,bad,sc

def add_global_star_cells(base_points, state, global_tets, global_mats, global_unions, global_lineage, global_depth,
                          index, edge, reserved_rows):
 a,b=map(int,edge)
 rr=edge_star(global_tets,state["dynamic"] | set(),index,edge)
 # Include branch-local descendants that carry the edge.
 for row,c in state["cells"].items():
  if np.any(c["tet"]==int(edge[0])) and np.any(c["tet"]==int(edge[1])):rr=np.r_[rr,int(row)]
 rr=np.unique(np.asarray(rr,np.int64))
 if len(rr):
  keep=[]
  for row0 in rr:
   row=int(row0)
   tet=state["cells"][row]["tet"] if row in state["cells"] else global_tets[row]
   if np.any(tet==a) and np.any(tet==b):keep.append(row)
  rr=np.asarray(keep,np.int64)
 if len(rr)==0:return None
 for row0 in rr:
  row=int(row0)
  if row in state["cells"]:continue
  if row in reserved_rows:return None
  if row>=len(global_tets):return None
  state["cells"][row]={"tet":global_tets[row].copy(),"mat":int(global_mats[row]),
      "depth":int(global_depth[row]),"lineage":int(global_lineage[row]),
      "union":tuple(global_unions[row])}
  if row not in state["base_bad"]:
   d,q,g=local_metrics(base_points,global_tets[row:row+1])
   state["base_bad"][row]=bool(not g[0]);state["base_score"][row]=score(d,q)
 return rr

def preview_split(base_points,state,global_tets,global_mats,global_unions,global_lineage,global_depth,
                  index,edge,protected,reserved_rows, max_depth=MAX_DEPTH,max_new_cells=MAX_NEW_CELLS,
                  point_keys=None):
 a,b=sorted(map(int,edge))
 if (a,b) in protected:return None,"protected_surface_edge"
 if max(a,b)>=len(base_points)+len(state["newpoints"]):return None,"point_row_out_of_range"
 work=clone_state(state)
 rr=add_global_star_cells(base_points,work,global_tets,global_mats,global_unions,global_lineage,global_depth,
                          index,(a,b),reserved_rows)
 if rr is None:return None,"incomplete_or_reserved_star"
 cells=[work["cells"][int(r)] for r in rr]
 if not cells:return None,"empty_star"
 if len({int(c["mat"]) for c in cells})!=1:return None,"mixed_material_star"
 if any(int(c["depth"])>=max_depth for c in cells):return None,"depth_cap"
 if work["new_cells"]+len(rr)>max_new_cells:return None,"new_cell_cap"
 x,exact=exact_average(point_at(base_points,work,a),point_at(base_points,work,b))
 if not exact:return None,"midpoint_not_exact_f32"
 xkey=np.ascontiguousarray(x).tobytes()
 if point_keys is not None:
  duplicate=xkey in point_keys or any(np.ascontiguousarray(v).tobytes()==xkey for v in work["newpoints"].values())
 else:
  duplicate=np.any(np.all(base_points.view(np.uint32)==x.view(np.uint32),axis=1)) or any(
      np.array_equal(v.view(np.uint32),x.view(np.uint32)) for v in work["newpoints"].values())
 if duplicate:return None,"duplicate_point"
 mid=int(work["next_point"])
 old=np.asarray([work["cells"][int(r)]["tet"] for r in rr],np.int64)
 children=np.empty((2*len(rr),4),np.int64)
 for i,tet in enumerate(old):
  ia=int(np.flatnonzero(tet==a)[0]);ib=int(np.flatnonzero(tet==b)[0])
  left=tet.copy();right=tet.copy();left[ib]=mid;right[ia]=mid
  children[2*i]=left;children[2*i+1]=right
 nd,nq=branch_metrics(base_points,work,children,extra=(mid,x))
 ng=np.asarray(M.passes(nd,nq),bool)
 if np.any(~np.isfinite(nd)) or np.any(nd<=0):return None,"nonpositive_f32_child"
 exact_old=branch_exact_dets(base_points,work,old);exact_new=branch_exact_dets(base_points,work,children,extra=(mid,x))
 if min(exact_old)<=0 or min(exact_new)<=0:return None,"nonpositive_exact_child"
 if sum(exact_old)!=sum(exact_new):return None,"exact_det6_delta"
 expected_boundary,_=QB._expected_split_chain(old,a,b,mid)
 if M.chain(children)!=expected_boundary:return None,"local_boundary_chain"
 # Build branch state with deterministic output rows and ancestry.
 before_d,before_q=branch_metrics(base_points,work,old);before_good=np.asarray(M.passes(before_d,before_q),bool)
 op_newrow=[];parent_depths=[]
 for i,row0 in enumerate(rr):
  row=int(row0);parent=work["cells"][row];parent_depths.append(int(parent["depth"]))
  c0=children[2*i].copy();c1=children[2*i+1].copy()
  work["cells"][row]={"tet":c0,"mat":int(parent["mat"]),"depth":int(parent["depth"])+1,
                       "lineage":int(parent["lineage"]),"union":tuple(parent["union"])}
  nr=int(work["next_cell"]);work["next_cell"]+=1;op_newrow.append(nr)
  work["cells"][nr]={"tet":c1,"mat":int(parent["mat"]),"depth":int(parent["depth"])+1,
                      "lineage":int(parent["lineage"]),"union":tuple(parent["union"])}
  work["dynamic"].add(row);work["dynamic"].add(nr)
 work["newpoints"][mid]=x.copy();work["next_point"]+=1;work["new_cells"]+=len(rr)
 point_union=tuple(sorted({int(work["cells"][int(r)]["lineage"]) for r in rr}))
 old_bad=int(np.count_nonzero(~before_good));new_bad=int(np.count_nonzero(~ng))
 record={"edge":[a,b],"star_rows":rr.tolist(),"new_point_row":mid,
         "new_point_world_m_f32":x.tolist(),"new_point_bits":[hex(int(v)) for v in x.view(np.uint32)],
         "new_cell_rows":op_newrow,"children_tetrahedra":children.tolist(),
         "parent_unions":[list(work["cells"][int(r)]["union"]) for r in rr],
         "point_parent_union":list(point_union),"old_bad_in_star":old_bad,"new_bad_in_children":new_bad,
         "depth":max(parent_depths)+1,
         "exact_det6_delta":"0","complete_edge_star":True}
 work["operations"].append(record)
 return work,"PASS"

def current_patch_ids(state):
 return set(state["cells"])

def branch_quality(base_points,state):
 rows,d,q,g,bad,sc=patch_stats(base_points,state)
 return rows,d,q,g,bad,sc

def branch_is_improved(base_points,state,cutoff=1e-8):
 rows,d,q,g,bad,sc=branch_quality(base_points,state)
 base_bad=sum(int(v) for v in state["base_bad"].values())
 base_sc=sum(float(v) for v in state["base_score"].values())
 oldgood=set(r for r,v in state["base_bad"].items() if not v)
 lineage_bad={}
 for i,r in enumerate(rows):
  lin=int(state["cells"][int(r)]["lineage"])
  lineage_bad.setdefault(lin,[]).append(bool(g[i]))
 if any(not all(lineage_bad.get(int(r),[])) for r in oldgood):
  return False,bad,sc,base_bad,base_sc
 return (bad<base_bad or (bad==base_bad and sc<base_sc-cutoff)),bad,sc,base_bad,base_sc

def clone_initial_patch(seed_edge,base_points,global_tets,global_mats,global_unions,global_lineage,global_depth,
                        index,reserved_rows,next_cell,next_point,dynamic_rows=None):
 state={"cells":{},"newpoints":{},"operations":[],"base_bad":{},"base_score":{},
        "new_cells":0,"next_cell":int(next_cell),"next_point":int(next_point),
        "dynamic":set() if dynamic_rows is None else set(dynamic_rows)}
 rr=edge_star(global_tets,state["dynamic"],index,seed_edge)
 if len(rr)==0:return None
 for r0 in rr:
  r=int(r0)
  state["cells"][r]={"tet":global_tets[r].copy(),"mat":int(global_mats[r]),"depth":int(global_depth[r]),
                     "lineage":int(global_lineage[r]),"union":tuple(global_unions[r])}
  d,q,g=local_metrics(base_points,global_tets[r:r+1])
  state["base_bad"][r]=bool(not g[0]);state["base_score"][r]=score(d,q)
 return state

def build_output(z,role,points,tets,mats,unions,lineage,depth,base_npoints,base_ntets,
                 touched_base,operations,point_parent_unions,input_path,input_sha):
 out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in z.items()}
 oldp=V739.points_f32(z);oldn=len(oldp);n0=len(z["tetrahedra"]);nout=len(tets)
 out["points_world_m"]=points.copy();out["points_world_m_f32"]=points.copy()
 out["tetrahedra"]=tets.copy();out["material_indices"]=mats.copy()
 old_direct=np.asarray(z.get("output_cell_to_input_cell",np.arange(n0,dtype=np.int64)),np.int64)
 old_abs=np.asarray(z.get("edge_bisection_absolute_output_cell_to_input_cell",old_direct),np.int64)
 if len(old_direct)!=n0 or len(old_abs)!=n0:raise ValueError("input cell map length mismatch")
 direct=np.r_[np.arange(n0,dtype=np.int64),np.full(nout-n0,-1,np.int64)]
 direct[np.asarray(sorted(touched_base),np.int64)]=-1
 out["output_cell_to_input_cell"]=direct
 absolute=np.full(nout,-1,np.int64)
 absolute[:n0]=old_abs
 for row in range(n0,nout):
  pu=unions[row]
  absolute[row]=int(pu[0]) if len(pu)==1 else -1
 for row in touched_base:
  pu=unions[int(row)];absolute[int(row)]=int(pu[0]) if len(pu)==1 else -1
 out["edge_bisection_absolute_output_cell_to_input_cell"]=absolute
 inverse=np.arange(n0,dtype=np.int64);inverse[np.asarray(sorted(touched_base),np.int64)]=-1
 out["input_cell_to_output_cell"]=inverse
 out["retained_input_cell_rows"]=np.flatnonzero(inverse>=0)
 changed=np.flatnonzero(direct<0)
 po=[0];pv=[]
 for row in changed:
  pu=tuple(sorted(set(map(int,unions[int(row)]))))
  pv.extend(pu);po.append(len(pv))
 out["replacement_parent_offsets"]=np.asarray(po,np.int64)
 out["replacement_parent_cell_rows"]=np.asarray(pv,np.int64)
 touched_lineages=sorted(set(map(int,lineage[np.asarray(sorted(touched_base),np.int64)])))
 grouped=[]
 tet_offsets=[0]
 for parent_row in touched_lineages:
  group=np.flatnonzero((lineage==parent_row)&(direct<0))
  grouped.extend(map(int,group));tet_offsets.append(len(grouped))
 out["replacement_tet_offsets"]=np.asarray(tet_offsets,np.int64)
 out["replacement_tetrahedra"]=tets[np.asarray(grouped,np.int64)].copy().reshape(-1,4)
 # Immediate canonical operation CSR; replacement ancestry is the complete parent union above.
 op_parents=[];opoffs=[0];childoffs=[0]
 for op in operations:
  star=sorted(set(map(int,op["star_rows"])))
  op_parents.extend(star);opoffs.append(len(op_parents))
  childoffs.append(childoffs[-1]+2*len(star))
 out["edge_bisection_replacement_parent_cell_offsets"]=np.asarray(opoffs,np.int64)
 out["edge_bisection_replacement_parent_cell_rows"]=np.asarray(op_parents,np.int64)
 out["edge_bisection_replacement_child_tet_offsets"]=np.asarray(childoffs,np.int64)
 out["edge_bisection_replacement_child_tetrahedra"]=np.asarray(
      [c for op in operations for c in op["children_tetrahedra"]],np.int64).reshape(-1,4)
 out["edge_bisection_parent_edge_volume_rows"]=np.asarray([op["edge"] for op in operations],np.int64)
 out["edge_bisection_parent_cell_rows"]=np.asarray(op_parents,np.int64)
 out["edge_bisection_new_point_index"]=np.asarray([op["new_point_row"] for op in operations],np.int64)
 out["edge_bisection_new_point_parameter"]=np.full(len(operations),.5,np.float64)
 out["edge_bisection_new_point_support_residual_m"]=np.zeros(len(operations),np.float64)
 out["output_point_to_input_point_row"]=np.r_[np.arange(oldn,dtype=np.int64),np.full(len(points)-oldn,-1,np.int64)]
 out["input_node_to_output_node"]=np.arange(oldn,dtype=np.int64)
 out["output_node_to_input_node"]=np.r_[np.arange(oldn,dtype=np.int64),np.full(len(points)-oldn,-1,np.int64)]
 out["source_interior_vertex_quotient"]=np.arange(oldn,dtype=np.int64)
 # Preserve all old point-parent rows and append each new point's full star parent union.
 old_off=np.asarray(z.get("output_point_parent_offsets",np.zeros(oldn+1,np.int64)),np.int64)
 old_vals=np.asarray(z.get("output_point_parent_input_cell_rows",np.empty(0,np.int64)),np.int64)
 if len(old_off)!=oldn+1 or old_off[-1]!=len(old_vals):raise ValueError("parent point CSR malformed")
 new_off=list(map(int,old_off));new_vals=list(map(int,old_vals))
 for pu in point_parent_unions:
  new_vals.extend(sorted(set(map(int,pu))));new_off.append(len(new_vals))
 out["output_point_parent_offsets"]=np.asarray(new_off,np.int64)
 out["output_point_parent_input_cell_rows"]=np.asarray(new_vals,np.int64)
 # Carry point-valued authored lanes by the exact same edge midpoint construction.
 point_skip={"points_world_m","points_world_m_f32","output_point_to_input_point_row","input_node_to_output_node",
             "output_node_to_input_node","source_interior_vertex_quotient"}
 point_skip.update(ROLE_SURFACE_MAPS[role].values())
 for key,value in z.items():
  arr=np.asarray(value)
  if key in point_skip or arr.ndim==0 or arr.shape[0]!=oldn or not ("point" in key.lower() or "vertex" in key.lower()):continue
  current=arr.copy();lanes=[]
  for op in operations:
   a,b=map(int,op["edge"])
   lane=V739.derived_point_lane(current,a,b,.5)
   lanes.append(lane)
   current=np.concatenate((current,np.asarray(lane,dtype=arr.dtype).reshape((1,)+arr.shape[1:])),axis=0)
  if lanes:out[key]=current
 # These are interior operations: authored boundary and desired boundary are exact inherited rows.
 out["edge_bisection_schema"]=np.asarray("numi.h706.compound-interior-edge-bisection-742.v1")
 out["edge_bisection_parent_candidate_path"]=np.asarray([str(input_path)])
 out["edge_bisection_parent_candidate_sha256"]=np.asarray([input_sha])
 out["edge_bisection_operation_parent_offsets"]=out["edge_bisection_replacement_parent_cell_offsets"].copy()
 out["edge_bisection_operation_parent_rows"]=out["edge_bisection_replacement_parent_cell_rows"].copy()
 out["edge_bisection_operation_depth"]=np.asarray([op["depth"] for op in operations],np.int32)
 for k in ("current_geometry_parent_path","metadata_finalizer_parent_candidate_path"):
  if k in out:out[k]=np.asarray([str(input_path)])
 for k in ("current_geometry_parent_sha256","metadata_finalizer_parent_candidate_sha256"):
  if k in out:out[k]=np.asarray([input_sha])
 return out

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--seconds",type=float,default=60.);ap.add_argument("--candidate-limit",type=int,default=24)
 ap.add_argument("--patch-limit",type=int,default=4)
 args=ap.parse_args();start=time.monotonic();sp=args.state.resolve();state=json.loads(sp.read_text())
 if state.get("physical_admission") is True or int(state.get("native_steps",0))!=0:raise SystemExit("refuse physical/runtime state")
 od=args.output.resolve();od.mkdir(parents=True,exist_ok=False);deadline=start+max(.1,args.seconds)
 report={"schema":"numi.h706.compound-interior-edge-bisection-742.v1","status":"NO_QUALIFIED_COMPOUND_PATCH",
  "physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN","time_budget_s":args.seconds,
  "input_state":{"path":str(sp),"sha256":sha(sp)},"producer":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},
  "owner":{"path":str(OWNER),"sha256":sha(OWNER)},"roles":{}}
 surfaces={};surface_pins={}
 for sname,rec in state.get("source_surfaces",{}).items():
  p=resolve(rec["path"],sp);h=sha(p)
  if h!=rec.get("sha256"):raise ValueError("surface pin mismatch "+sname)
  surfaces[sname]=V739.load_npz(p);surface_pins[sname]={"path":str(p),"sha256":h}
 output_state=dict(state);changed_roles={}
 for role in ROLES:
  if time.monotonic()>=deadline:break
  rec=state[role];ip=resolve(rec["path"],sp);ih=sha(ip)
  if ih!=rec.get("sha256"):raise ValueError("volume pin mismatch "+role)
  z=V739.load_npz(ip);points=V739.points_f32(z);tets=np.asarray(z["tetrahedra"],np.int64).copy();mats=np.asarray(z["material_indices"],np.int32).copy()
  det,q,good=local_metrics(points,tets)
  if np.any(det<=0):raise ValueError("input has nonpositive cell")
  base_bad=~good;base_score=score(det,q)
  base_ncells=len(tets);base_npoints=len(points)
  unions=base_unions(z,base_ncells);lineage=np.arange(base_ncells,dtype=np.int64);depth=np.zeros(base_ncells,np.int16)
  dynamic=set();point_keys={np.ascontiguousarray(v).tobytes() for v in points}
  protected=protected_edge_set(z,role,surfaces)
  protected |= face_edges(np.asarray(list(M.chain(tets).keys()),np.int64).reshape(-1,3))
  index=EdgeIndex(points,tets)
  point_parent_unions=[];reserved_rows=set();reserved_vertices=set();ops_all=[]
  full_bad=int(np.count_nonzero(base_bad));full_score=base_score
  reject=collections.Counter();accepted_patches=0
  seed_rows=np.flatnonzero(base_bad)
  # Most severe quality first; ties follow stable row order.
  seed_rows=sorted(map(int,seed_rows),key=lambda r:(-float(-math.log(max(min(float(q[r])/1e-5,float(det[r])/(6e-18),1.),1e-300))),r))
  for seed in seed_rows[:max(1,int(args.candidate_limit))]:
   if accepted_patches>=args.patch_limit or time.monotonic()>=deadline:break
   if seed in reserved_rows:continue
   # Candidate edges of bad seed, longest first. This is an interior-only patch seed.
   ted=tets[seed];e0=sorted({tuple(sorted((int(ted[i]),int(ted[j])))) for i,j in EDGE_PAIRS},
                  key=lambda e:(-float(np.linalg.norm(points[e[0]].astype(np.float64)-points[e[1]].astype(np.float64))),e))
   found=None
   for edge in e0:
    if edge in protected:reject["protected_edge_seed"]+=1;continue
    if time.monotonic()>=deadline:break
    # Branch the initial edge on a snapshot; accepted patch remains disjoint.
    local_t=tets;local_m=mats;local_u=unions;local_lin=lineage;local_dep=depth
    st=clone_initial_patch(edge,points,local_t,local_m,local_u,local_lin,local_dep,index,reserved_rows,
                           len(local_t),len(points),dynamic)
    if st is None:reject["empty_seed_star"]+=1;continue
    # Add edge star rows to this patch; retain baseline quality by current rows.
    result,why=preview_split(points,st,local_t,local_m,local_u,local_lin,local_dep,index,edge,protected,reserved_rows,point_keys=point_keys)
    if result is None:reject[why]+=1;continue
    patch_base_rows=set(result["base_bad"])
    patch_base_bad=sum(int(v) for v in result["base_bad"].values())
    patch_base_score=sum(float(v) for v in result["base_score"].values())
    # Greedy bounded compound continuation. Intermediate bad-count increases are allowed.
    for step in range(1,MAX_DEPTH+1):
     improved,pbad,pscore,bbase,sbase=branch_is_improved(points,result)
     if improved:
      found=(result,patch_base_rows,patch_base_bad,patch_base_score);break
     if step>=MAX_DEPTH or result["new_cells"]>=MAX_NEW_CELLS:break
     rows0,dd,qq,gg,cbad,cscore=branch_quality(points,result)
     candidate_edges=set()
     for ii,r0 in enumerate(rows0):
      if gg[ii]:continue
      tet=result["cells"][int(r0)]["tet"]
      candidate_edges.update(tuple(sorted((int(tet[i]),int(tet[j])))) for i,j in EDGE_PAIRS)
     previews=[]
     for e in sorted(candidate_edges,key=lambda e:(-edge_length(points,result,e),e)):
      if time.monotonic()>=deadline:break
      cand,reason=preview_split(points,result,local_t,local_m,local_u,local_lin,local_dep,index,e,protected,reserved_rows,point_keys=point_keys)
      if cand is None:reject[reason]+=1;continue
      imp,nb,ns,bb,bs=branch_is_improved(points,cand)
      if nb>bb+8:continue
      previews.append((nb,ns,cand["new_cells"],e,cand))
     if not previews:break
     previews.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
     result=previews[0][4]
     # Extend the remembered input domain with every complete star row.
     patch_base_rows.update(result["base_bad"])
     patch_base_bad=sum(int(result["base_bad"][r]) for r in patch_base_rows)
     patch_base_score=sum(float(result["base_score"][r]) for r in patch_base_rows)
    if found:break
   if found is None:continue
   patch,patch_base_rows,patch_base_bad,patch_base_score=found
   patch_bad=branch_quality(points,patch)[4];patch_score=branch_quality(points,patch)[5]
   if not (patch_bad<patch_base_bad or (patch_bad==patch_base_bad and patch_score<patch_base_score-1e-8)):continue
   # Ensure selected patch cells/vertices do not overlap any earlier patch.
   touched_rows=set();verts=set()
   for op in patch["operations"]:
    for r0 in op["star_rows"]:
     r=int(r0)
     if r<len(tets):touched_rows.add(r)
     tet=patch["cells"][r]["tet"] if r in patch["cells"] else tets[r]
     verts.update(map(int,tet))
   if touched_rows & reserved_rows or verts & reserved_vertices:reject["overlapping_patch"]+=1;continue
   # Commit the local final state.
   points,tets,mats,unions,lineage,depth,dynamic,point_parent_unions=commit_patch(
       points,tets,mats,unions,lineage,depth,dynamic,point_parent_unions,patch)
   point_keys.update(np.ascontiguousarray(op["new_point_world_m_f32"],dtype=np.float32).tobytes() for op in patch["operations"])
   ops_all.extend(patch["operations"]);reserved_rows|=touched_rows;reserved_vertices|=verts
   accepted_patches+=1
   d,q,g=local_metrics(points,tets);full_bad=int(np.count_nonzero(~g));full_score=score(d,q)
  if not ops_all:
   report["roles"][role]={"status":"NO_QUALIFIED_COMPOUND_PATCH","baseline_cells":base_ncells,
      "baseline_bad_cells":int(np.count_nonzero(base_bad)),"rejections":dict(reject),"operations":[]}
   continue
  # Global final gates: positive exact determinant, total strict gain, old-good descendant preservation.
  d,q,g=local_metrics(points,tets)
  if np.any(d<=0):raise ValueError("compound output has nonpositive cells")
  oldgood=np.flatnonzero(~base_bad);desc_bad={}
  for row,origin in enumerate(lineage):
   desc_bad.setdefault(int(origin),[]).append(bool(g[row]))
  regress=[int(r) for r in oldgood if not all(desc_bad.get(int(r),[]))]
  final_bad=int(np.count_nonzero(~g));final_score=score(d,q)
  if regress:raise ValueError("old-good input descendants regressed: "+str(regress[:10]))
  if not (final_bad<int(np.count_nonzero(base_bad)) or (final_bad==int(np.count_nonzero(base_bad)) and final_score<base_score-1e-8)):
   raise ValueError("compound final quality did not strictly improve")
  exactall=QB._exact_det_rows(points,tets)
  if min(exactall)<=0:raise ValueError("full exact determinant positivity failure")
  exactbase=QB._exact_det_rows(V739.points_f32(z),np.asarray(z["tetrahedra"],np.int64))
  if sum(exactall)!=sum(exactbase):raise ValueError("whole-volume exact det6 changed")
  if M.chain(np.asarray(z["tetrahedra"],np.int64))!=M.chain(tets):
   raise ValueError("whole-volume oriented boundary chain changed")
  # Emit compressed candidate.
  op_path=od/(role.lower()+"-compound-operations.json")
  op_path.write_text(json.dumps(ops_all,indent=2)+"\n")
  outdata=build_output(z,role,points,tets,mats,unions,lineage,depth,base_npoints,base_ncells,
                       touched_rows,ops_all,point_parent_unions,ip,ih)
  opath=od/(role.lower()+"-compound-volume-proposal.npz")
  np.savez_compressed(opath,**outdata);oh=sha(opath)
  nrec=dict(rec);nrec.update({"path":str(opath),"sha256":oh,"nodes":len(points),"cells":len(tets),
      "native_quality_failures":final_bad,"nonpositive":0,"geometry_changed_points":len(ops_all),
      "absolute_geometric_volume_change_m3":0.0,"target_face_count":len(z["desired_boundary_faces"]),
      "missing_target_faces":0,"extra_actual_faces":0,"target_boundary_match":True})
  output_state[role]=nrec;changed_roles[role]={"path":str(opath),"sha256":oh}
  report["roles"][role]={"status":"PASS_SOURCE_ONLY_COMPOUND_INTERIOR_EDGE_BISECTION",
    "input":{"path":str(ip),"sha256":ih},"output":{"path":str(opath),"sha256":oh},
    "operation_log":{"path":str(op_path),"sha256":sha(op_path)},
    "baseline_cells":base_ncells,"output_cells":len(tets),"baseline_nodes":base_npoints,"output_nodes":len(points),
    "baseline_bad_cells":int(np.count_nonzero(base_bad)),"output_bad_cells":final_bad,
    "baseline_log_deficit":base_score,"output_log_deficit":final_score,
    "exact_full_output_positive_count":len(exactall),"full_boundary_faces_bitwise_unchanged":True,
    "old_good_input_descendants_pass":True,"rejections":dict(reject),"operations":ops_all}
 if changed_roles:
  output_state.update({"physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN",
       "boundary_source_binding_status":"INTERIOR_ONLY_BOUNDARY_INHERITANCE_AUDIT_PENDING",
       "source_crossing_evidence_inherited_from_unchanged_surfaces":None,"surface_crossings":None,
       "compound_interior_edge_bisection":{"schema":"numi.h706.compound-interior-edge-bisection-742.v1",
          "status":"PASS_LOCAL_SOURCE_ONLY","operations_by_role":{r:rep.get("operations",[]) for r,rep in report["roles"].items()},
          "physical_admission":False,"native_steps":0,"all185_surface_audit":"NOT_RUN"}})
  stpath=od/"state.json";stpath.write_text(json.dumps(output_state,indent=2)+"\n")
  report["status"]="PASS_SOURCE_ONLY_COMPOUND_INTERIOR_EDGE_BISECTION";report["outputs"]={"state":{"path":str(stpath),"sha256":sha(stpath)},"volumes":changed_roles}
 report["surface_pins"]=surface_pins;report["elapsed_s"]=time.monotonic()-start
 rp=od/"report.json";rp.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"elapsed_s":report["elapsed_s"],"roles":report["roles"],"outputs":report.get("outputs",{})},separators=(",",":")))
 return 0 if changed_roles else 2
if __name__=="__main__":raise SystemExit(main())

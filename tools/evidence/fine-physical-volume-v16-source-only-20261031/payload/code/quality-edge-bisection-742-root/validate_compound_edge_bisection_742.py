#!/usr/bin/env python3
"""Independent replay checks for compound edge-bisection 742 proposals."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OWNER=ROOT/"quality-edge-bisection-683-root"/"quality_edge_bisection_683_helper.py"
SPEC=importlib.util.spec_from_file_location("quality_bisection742_verify_owner",OWNER)
QB=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=QB;SPEC.loader.exec_module(QB)
M=QB.M
V739_PATH=ROOT/"quality-edge-bisection-739-root"/"edge_bisection_739.py"
SPEC2=importlib.util.spec_from_file_location("quality_bisection742_verify_739",V739_PATH)
V=importlib.util.module_from_spec(SPEC2);sys.modules[SPEC2.name]=V;SPEC2.loader.exec_module(V)
ROLES=("Skin_current","Fat_current")
ROLE_SURFACE_MAPS=V.ROLE_SURFACE_MAPS

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def resolve(v,state_path):
 p=Path(str(v));return p if p.is_absolute() else (state_path.parent/p).resolve()

def chain(t):
 return M.chain(np.asarray(t,np.int64))

def base_unions(z,n):
 direct=np.asarray(z.get("output_cell_to_input_cell",np.arange(n,dtype=np.int64)),np.int64)
 if direct.shape!=(n,):raise ValueError("input cell identity map malformed")
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
   v=int(direct[r])
   if v<0:raise ValueError("changed cell missing complete replacement-parent CSR")
   out[r]=(v,)
 return out

def edge_star(tets,a,b):
 t=np.asarray(tets,np.int64)
 mask=np.any(t==int(a),axis=1)&np.any(t==int(b),axis=1)
 return np.flatnonzero(mask)

def exact_midpoint(points,a,b):
 aa=np.asarray(points[int(a)],np.float32).astype(np.float64)
 bb=np.asarray(points[int(b)],np.float32).astype(np.float64)
 x=((aa+bb)*0.5).astype(np.float32)
 return x, bool(np.array_equal(x.astype(np.float64),(aa+bb)*0.5))

def split_children(tets,star,a,b,mid):
 t=np.asarray(tets,np.int64);out=[]
 for row in map(int,star):
  parent=t[row]
  ia=np.flatnonzero(parent==int(a));ib=np.flatnonzero(parent==int(b))
  if len(ia)!=1 or len(ib)!=1:raise ValueError("operation star tet lacks exact edge")
  left=parent.copy();right=parent.copy()
  left[int(ib[0])]=int(mid);right[int(ia[0])]=int(mid)
  out.extend((left,right))
 return np.asarray(out,np.int64).reshape(-1,4)

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def protected_edges_for_role(z,role,surfaces):
 out=set()
 for key in ("boundary_faces","desired_boundary_faces"):
  if key in z:
   for f in np.asarray(z[key],np.int64):
    out.update(tuple(sorted((int(f[i]),int(f[j])))) for i,j in ((0,1),(1,2),(2,0)))
 wall=next((k for k in z if k.endswith("_wall_faces_volume_rows")),None)
 if wall:
  for f in np.asarray(z[wall],np.int64):
   out.update(tuple(sorted((int(f[i]),int(f[j])))) for i,j in ((0,1),(1,2),(2,0)))
 for sname,mapkey in ROLE_SURFACE_MAPS[role].items():
  if sname not in surfaces or mapkey not in z:continue
  sf=surfaces[sname];faces=np.asarray(sf["faces"],np.int64);mapping=np.asarray(z[mapkey],np.int64)
  for f in faces:
   v=mapping[f]
   if np.all(v>=0):
    out.update(tuple(sorted((int(v[i]),int(v[j])))) for i,j in ((0,1),(1,2),(2,0)))
 return out

def verify_one(state_path,report_path,role):
 state_path=Path(state_path).resolve();report_path=Path(report_path).resolve()
 state=json.loads(state_path.read_text());report=json.loads(report_path.read_text())
 if state.get("physical_admission") is True or int(state.get("native_steps",0))!=0:
  raise ValueError("input state claims physical/runtime admission")
 if report.get("schema")!="numi.h706.compound-interior-edge-bisection-742.v1":
  raise ValueError("unexpected report schema")
 if report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0 or report.get("mass_transport")!="NOT_RUN":
  raise ValueError("proposal report contains admission or mass claim")
 if report.get("status")!="PASS_SOURCE_ONLY_COMPOUND_INTERIOR_EDGE_BISECTION":
  raise ValueError("report is not a passing source-only proposal")
 state_pin=report.get("input_state",{})
 if str(Path(state_pin.get("path","")).resolve())!=str(state_path) or state_pin.get("sha256")!=sha(state_path):
  raise ValueError("input state digest mismatch")
 for group in ("producer","owner"):
  pin=report.get(group,{})
  pp=Path(pin.get("path",""))
  if not pp.is_file() or sha(pp)!=pin.get("sha256"):raise ValueError(group+" source digest mismatch")
 if role not in report.get("roles",{}):raise ValueError("role is missing from proposal")
 rr=report["roles"][role]
 if rr.get("status")!="PASS_SOURCE_ONLY_COMPOUND_INTERIOR_EDGE_BISECTION":raise ValueError("role did not pass")
 ip=Path(rr["input"]["path"]);op=Path(rr["output"]["path"]);lp=Path(rr["operation_log"]["path"])
 if sha(ip)!=rr["input"]["sha256"] or sha(op)!=rr["output"]["sha256"] or sha(lp)!=rr["operation_log"]["sha256"]:
  raise ValueError("input/output/log digest mismatch")
 state_rec=state.get(role,{})
 state_ip=resolve(state_rec.get("path",""),state_path)
 if str(state_ip)!=str(ip) or sha(state_ip)!=state_rec.get("sha256"):
  raise ValueError("input role does not match state pin")
 outrec=report.get("outputs",{}).get("state",{})
 if not outrec:raise ValueError("output state pin is missing")
 osp=Path(outrec["path"])
 if sha(osp)!=outrec.get("sha256"):raise ValueError("output state digest mismatch")
 outstate=json.loads(osp.read_text())
 if outstate.get("physical_admission") is not False or int(outstate.get("native_steps",-1))!=0 or outstate.get("mass_transport")!="NOT_RUN":
  raise ValueError("output state contains physical or mass admission")
 if outstate.get(role,{}).get("sha256")!=rr["output"]["sha256"] or str(Path(outstate.get(role,{}).get("path","")))!=str(op):
  raise ValueError("output state role pin mismatch")
 z=load_npz(ip);out=load_npz(op);ops=json.loads(lp.read_text())
 p0=V.points_f32(z);t0=np.asarray(z["tetrahedra"],np.int64);m0=np.asarray(z["material_indices"],np.int32)
 if len(m0)!=len(t0):raise ValueError("input material row mismatch")
 points=p0.copy();tets=t0.copy();mats=m0.copy();unions=base_unions(z,len(t0))
 lineage=np.arange(len(t0),dtype=np.int64);depth=np.zeros(len(t0),np.int16)
 operation_parents=[];operation_offsets=[0];child_offsets=[0]
 touched=set();point_parent_rows=[]
 surface_pins={}
 # The unmodified source surfaces establish the protected edge domain.
 surfaces={}
 for sname,srec in state.get("source_surfaces",{}).items():
  sp=resolve(srec["path"],state_path)
  if sha(sp)!=srec.get("sha256"):raise ValueError("surface pin mismatch: "+sname)
  surfaces[sname]=V.load_npz(sp);surface_pins[sname]={"path":str(sp),"sha256":srec.get("sha256")}
 if report.get("surface_pins",{})!=surface_pins:raise ValueError("report source surface pins mismatch")
 protected=protected_edges_for_role(z,role,surfaces)
 protected |= {tuple(sorted((int(f[i]),int(f[j])))) for f in chain(t0)
               for i,j in ((0,1),(1,2),(2,0))}
 for oi,oprow in enumerate(ops):
  edge=tuple(sorted(map(int,oprow["edge"])))
  a,b=edge
  if a==b or min(edge)<0 or max(edge)>=len(points):raise ValueError("invalid operation edge")
  if edge in protected:raise ValueError("operation bisects protected boundary/surface edge")
  star=edge_star(tets,a,b)
  if not np.array_equal(star,np.asarray(oprow["star_rows"],np.int64)):
   raise ValueError("operation star is incomplete or noncanonical")
  if len(star)==0:raise ValueError("empty operation star")
  if len(np.unique(mats[star]))!=1:raise ValueError("operation crosses material")
  if np.any(depth[star]>=8):raise ValueError("operation exceeds depth limit")
  x,exact=exact_midpoint(points,a,b)
  if not exact:raise ValueError("midpoint is not exact in Float32")
  mid=len(points)
  if int(oprow["new_point_row"])!=mid:raise ValueError("new point row is not append-only")
  if not np.array_equal(x.view(np.uint32),np.asarray(oprow["new_point_world_m_f32"],np.float32).view(np.uint32)):
   raise ValueError("logged midpoint coordinates differ")
  bits=[hex(int(v)) for v in x.view(np.uint32)]
  if oprow.get("new_point_bits")!=bits:raise ValueError("logged midpoint bit pattern differs")
  old=tets[star].copy()
  child=split_children(tets,star,a,b,mid)
  if not np.array_equal(child,np.asarray(oprow["children_tetrahedra"],np.int64)):
   raise ValueError("operation child tets do not replay")
  old_exact=QB._exact_det_rows(points,old);newp=np.vstack((points,x.reshape(1,3)))
  child_exact=QB._exact_det_rows(newp,child)
  if min(old_exact)<=0 or min(child_exact)<=0:raise ValueError("operation has nonpositive exact det")
  if sum(old_exact)!=sum(child_exact):raise ValueError("operation det6 not conserved")
  expected,_=QB._expected_split_chain(old,a,b,mid)
  if chain(child)!=expected:raise ValueError("operation oriented local chain mismatch")
  nd,nq=M.measurements(newp,child)
  if np.any(nd<=0):raise ValueError("operation has nonpositive floating child")
  if not bool(oprow.get("complete_edge_star")):raise ValueError("operation does not attest full star")
  expected_union=[list(unions[int(r)]) for r in star]
  if oprow.get("parent_unions")!=expected_union:raise ValueError("operation parent ancestry changed")
  current_parent_union=sorted({int(lineage[int(r)]) for r in star})
  if oprow.get("point_parent_union")!=current_parent_union:raise ValueError("point parent CSR ancestry mismatch")
  newrows=list(range(len(tets),len(tets)+len(star)))
  if list(map(int,oprow["new_cell_rows"]))!=newrows:raise ValueError("operation append-cell map differs")
  # Preserve each parent's first child in its current row and append its second child.
  app_t=[];app_m=[];app_u=[];app_lin=[];app_dep=[]
  for i,row0 in enumerate(star):
   row=int(row0);parent_union=unions[row];parent_line=int(lineage[row]);parent_depth=int(depth[row])
   tets[row]=child[2*i];app_t.append(child[2*i+1])
   app_m.append(mats[row]);app_u.append(parent_union);app_lin.append(parent_line)
   mats[row]=mats[row];depth[row]=parent_depth+1
   app_dep.append(parent_depth+1);unions[row]=parent_union;lineage[row]=parent_line
   touched.add(row)
  tets=np.concatenate((tets,np.asarray(app_t,np.int64).reshape(-1,4)),axis=0)
  mats=np.r_[mats,np.asarray(app_m,np.int32)]
  unions.extend(app_u);lineage=np.r_[lineage,np.asarray(app_lin,np.int64)]
  depth=np.r_[depth,np.asarray(app_dep,np.int16)]
  points=np.vstack((points,x.reshape(1,3)))
  point_parent_rows.append(current_parent_union)
  operation_parents.extend(map(int,star));operation_offsets.append(len(operation_parents))
  child_offsets.append(child_offsets[-1]+2*len(star))
 # Final volume and exact conservation gates.
 if len(points)!=len(p0)+len(ops):raise ValueError("unexpected output point count")
 if not np.array_equal(V.points_f32(out).view(np.uint32),points.view(np.uint32)):
  raise ValueError("output points differ from operation replay")
 if not np.array_equal(np.asarray(out["tetrahedra"],np.int64),tets):raise ValueError("output tetrahedra differ from replay")
 if not np.array_equal(np.asarray(out["material_indices"],np.int32),mats):raise ValueError("output materials differ from replay")
 ex=QB._exact_det_rows(points,tets);ex0=QB._exact_det_rows(p0,t0)
 if min(ex)<=0:raise ValueError("output has nonpositive exact determinant")
 if sum(ex)!=sum(ex0):raise ValueError("whole exact volume det6 changed")
 if chain(tets)!=chain(t0):raise ValueError("whole oriented boundary chain changed")
 det,q=M.measurements(points,tets);good=np.asarray(M.passes(det,q),bool)
 d0,q0=M.measurements(p0,t0);good0=np.asarray(M.passes(d0,q0),bool)
 oldgood=np.flatnonzero(good0);desc={}
 for i,lin in enumerate(lineage):desc.setdefault(int(lin),[]).append(bool(good[i]))
 regress=[int(i) for i in oldgood if not all(desc.get(int(i),[]))]
 if regress:raise ValueError("old-good descendants regressed: "+repr(regress[:20]))
 out_bad=int(np.count_nonzero(~good));base_bad=int(np.count_nonzero(~good0))
 out_score=QB._log_deficit(det,q);base_score=QB._log_deficit(d0,q0)
 if not (out_bad<base_bad or (out_bad==base_bad and out_score<base_score-1e-8)):
  raise ValueError("output did not strictly improve native quality")
 # Recompute canonical immediate maps.
 direct=np.r_[np.arange(len(t0),dtype=np.int64),np.full(len(tets)-len(t0),-1,np.int64)]
 direct[np.asarray(sorted(touched),np.int64)]=-1
 if not np.array_equal(np.asarray(out["output_cell_to_input_cell"],np.int64),direct):
  raise ValueError("direct cell crosswalk mismatch")
 inverse=np.arange(len(t0),dtype=np.int64)
 touched_input=np.asarray(sorted(r for r in touched if int(r)<len(t0)),np.int64)
 inverse[touched_input]=-1
 if not np.array_equal(np.asarray(out["input_cell_to_output_cell"],np.int64),inverse):raise ValueError("inverse cell crosswalk mismatch")
 if not np.array_equal(np.asarray(out["retained_input_cell_rows"],np.int64),np.flatnonzero(inverse>=0)):
  raise ValueError("retained input rows mismatch")
 old_abs=np.asarray(z.get("edge_bisection_absolute_output_cell_to_input_cell",
                          z.get("output_cell_to_input_cell",np.arange(len(t0)))),np.int64)
 expected_abs=np.r_[old_abs.copy(),np.full(len(tets)-len(t0),-1,np.int64)]
 for row in touched:
  u=unions[int(row)];expected_abs[int(row)]=int(u[0]) if len(u)==1 else -1
 for row in range(len(t0),len(tets)):
  u=unions[row];expected_abs[row]=int(u[0]) if len(u)==1 else -1
 if not np.array_equal(np.asarray(out["edge_bisection_absolute_output_cell_to_input_cell"],np.int64),expected_abs):
  raise ValueError("absolute cell crosswalk mismatch")
 # Full replacement ancestry and final replacement tet grouping.
 changed=np.flatnonzero(direct<0);po=[0];pv=[]
 for row in changed:
  pv.extend(sorted(set(map(int,unions[int(row)]))));po.append(len(pv))
 if not np.array_equal(np.asarray(out["replacement_parent_offsets"],np.int64),np.asarray(po,np.int64)):
  raise ValueError("replacement parent offsets mismatch")
 if not np.array_equal(np.asarray(out["replacement_parent_cell_rows"],np.int64),np.asarray(pv,np.int64)):
  raise ValueError("replacement parent rows mismatch")
 touched_lineages=sorted(set(map(int,lineage[np.asarray(sorted(touched),np.int64)])))
 grouped=[];to=[0]
 for pr in touched_lineages:
  grouped.extend(map(int,np.flatnonzero((lineage==pr)&(direct<0))));to.append(len(grouped))
 if not np.array_equal(np.asarray(out["replacement_tet_offsets"],np.int64),np.asarray(to,np.int64)):
  raise ValueError("replacement tet offsets mismatch")
 if not np.array_equal(np.asarray(out["replacement_tetrahedra"],np.int64),tets[np.asarray(grouped,np.int64)]):
  raise ValueError("replacement final tet rows mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_parent_cell_offsets"],np.int64),np.asarray(operation_offsets,np.int64)):
  raise ValueError("operation parent offsets mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_parent_cell_rows"],np.int64),np.asarray(operation_parents,np.int64)):
  raise ValueError("operation parent rows mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_child_tet_offsets"],np.int64),np.asarray(child_offsets,np.int64)):
  raise ValueError("operation child offsets mismatch")
 logged_children=np.asarray([c for op in ops for c in op["children_tetrahedra"]],np.int64).reshape(-1,4)
 if not np.array_equal(np.asarray(out["edge_bisection_replacement_child_tetrahedra"],np.int64),logged_children):
  raise ValueError("operation child history mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_parent_edge_volume_rows"],np.int64),
                       np.asarray([op["edge"] for op in ops],np.int64)):
  raise ValueError("operation edge rows mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_new_point_index"],np.int64),
                       np.asarray([op["new_point_row"] for op in ops],np.int64)):
  raise ValueError("operation point map mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_new_point_parameter"],np.float64),np.full(len(ops),.5)):
  raise ValueError("operation parameter mismatch")
 # Point crosswalks and complete edge-star parent CSR.
 n0=len(p0);nop=len(ops)
 if not np.array_equal(np.asarray(out["output_point_to_input_point_row"],np.int64),
                       np.r_[np.arange(n0,dtype=np.int64),np.full(nop,-1,np.int64)]):
  raise ValueError("point direct crosswalk mismatch")
 if not np.array_equal(np.asarray(out["input_node_to_output_node"],np.int64),np.arange(n0,dtype=np.int64)):
  raise ValueError("input point crosswalk mismatch")
 if not np.array_equal(np.asarray(out["output_node_to_input_node"],np.int64),
                       np.r_[np.arange(n0,dtype=np.int64),np.full(nop,-1,np.int64)]):
  raise ValueError("output point crosswalk mismatch")
 if "source_interior_vertex_quotient" in out and not np.array_equal(
    np.asarray(out["source_interior_vertex_quotient"],np.int64),np.arange(n0,dtype=np.int64)):
  raise ValueError("immediate point quotient mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_operation_depth"],np.int32),
                       np.asarray([op["depth"] for op in ops],np.int32)):
  raise ValueError("operation depth lane mismatch")
 if not np.array_equal(np.asarray(out["edge_bisection_new_point_support_residual_m"],np.float64),
                       np.zeros(nop,np.float64)):
  raise ValueError("operation support residual lane mismatch")
 inoff=np.asarray(z.get("output_point_parent_offsets",np.zeros(n0+1,np.int64)),np.int64)
 inval=np.asarray(z.get("output_point_parent_input_cell_rows",np.empty(0,np.int64)),np.int64)
 if len(inoff)!=n0+1 or inoff[-1]!=len(inval):raise ValueError("input point-parent CSR malformed")
 outoff=np.asarray(out["output_point_parent_offsets"],np.int64);outval=np.asarray(out["output_point_parent_input_cell_rows"],np.int64)
 if len(outoff)!=n0+nop+1 or outoff[0]!=0 or outoff[-1]!=len(outval):raise ValueError("output point-parent CSR malformed")
 if not np.array_equal(outoff[:n0+1],inoff) or not np.array_equal(outval[:len(inval)],inval):
  raise ValueError("old point-parent rows changed")
 cursor=len(inval)
 for i,parents in enumerate(point_parent_rows):
  lo,hi=map(int,outoff[n0+i:n0+i+2])
  if not np.array_equal(outval[lo:hi],np.asarray(parents,np.int64)):raise ValueError("new point parent row mismatch")
  cursor=hi
 if cursor!=len(outval):raise ValueError("extra point-parent rows")
 # Original coordinate prefix and all role surface maps/faces remain exact.
 if not np.array_equal(points[:n0].view(np.uint32),p0.view(np.uint32)):raise ValueError("old point coordinates changed")
 for key in ("boundary_faces","desired_boundary_faces"):
  if key in z and not np.array_equal(np.asarray(out[key]),np.asarray(z[key])):raise ValueError("boundary face rows changed: "+key)
 for key in ROLE_SURFACE_MAPS[role].values():
  if key in z and not np.array_equal(np.asarray(out[key]),np.asarray(z[key])):raise ValueError("surface point map changed in interior operation")
 # Verify every point-valued authored lane used the same ordered midpoint construction.
 skips={"points_world_m","points_world_m_f32","output_point_to_input_point_row","input_node_to_output_node",
        "output_node_to_input_node","source_interior_vertex_quotient"}
 skips.update(ROLE_SURFACE_MAPS[role].values())
 checked_lanes=0
 for key,val in z.items():
  arr=np.asarray(val)
  if key in skips or arr.ndim==0 or arr.shape[0]!=n0 or not ("point" in key.lower() or "vertex" in key.lower()):continue
  if key not in out:raise ValueError("missing derived point lane "+key)
  cur=arr.copy();made=[]
  for op in ops:
   a,b=map(int,op["edge"]);lane=V.derived_point_lane(cur,a,b,.5);made.append(lane)
   cur=np.concatenate((cur,np.asarray(lane,dtype=arr.dtype).reshape((1,)+arr.shape[1:])),axis=0)
  expected=np.concatenate((arr,np.asarray(made,dtype=arr.dtype).reshape((nop,)+arr.shape[1:])),axis=0)
  if not np.array_equal(np.asarray(out[key]),expected):raise ValueError("derived point lane mismatch "+key)
  checked_lanes+=1
 return {"status":"PASS_INDEPENDENT_COMPOUND_BISECTION_REPLAY","role":role,
         "input_sha256":sha(ip),"proposal_sha256":sha(op),"operation_log_sha256":sha(lp),
         "operation_count":len(ops),"input_cells":len(t0),"output_cells":len(tets),
         "input_points":n0,"output_points":len(points),"baseline_bad_cells":base_bad,
         "output_bad_cells":out_bad,"baseline_log_deficit":base_score,"output_log_deficit":out_score,
         "old_good_descendant_regressions":0,"exact_positive_cells":len(ex),
         "exact_total_det6_delta":"0","whole_oriented_boundary_chain_equal":True,
         "surface_files_rehashed":len(surface_pins),"derived_point_lanes_checked":checked_lanes,
         "physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--state",type=Path,required=True);ap.add_argument("--report",type=Path,required=True)
 ap.add_argument("--role",choices=ROLES,required=True);ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args()
 result=verify_one(a.state,a.report,a.role);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,separators=(",",":")))
 return 0

if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Fixed-boundary exact cavity retet: no source edits, no vertex movement."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE/"postquotient_cavity_remesh_883e_root.py"
PARENT_SHA="b411e7754f22d0d838b0671e08731e020501af6fb0b8bca2e92d855093b49a7d"
def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def ref(p):
 p=Path(p).resolve();return {"path":str(p),"sha256":sha(p)}
def imp(name,path,digest):
 path=Path(path).resolve();got=sha(path)
 if got!=digest:raise RuntimeError(f"{name} SHA mismatch {got}")
 spec=importlib.util.spec_from_file_location(name,path)
 if not spec or not spec.loader:raise RuntimeError("import failed "+str(path))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
E=imp("fixed_boundary_parent_883e",PARENT,PARENT_SHA)
B,ENG,M,T=E.B,E.ENG,E.M,E.T
MAX_CELLS,MAX_POINTS=64,24
def chain(f):return ENG.chain(np.asarray(f,np.int64))
def native_by_mat(p,t,mat):
 d,q=M.measurements(np.asarray(p,np.float32),np.asarray(t,np.int64));g=np.asarray(M.passes(d,q),bool)
 return {str(int(m)):int(np.count_nonzero(~g[np.asarray(mat)==m])) for m in np.unique(mat)},d,q,g
def cavity(cache,edge):
 a,b=map(int,edge);p=cache["points"];t=cache["tets"];mat=cache["mat"]
 if a==b or min(a,b)<0 or max(a,b)>=len(p):raise ValueError("edge outside current point rows")
 seed=np.asarray(B.incident_cells(cache,np.asarray([a,b],np.int64)),np.int64)
 if not len(seed):raise ValueError("edge has no current volume-cell star")
 ms=np.unique(mat[seed])
 if len(ms)!=1:raise ValueError("edge star crosses material")
 material=int(ms[0])
 rows,added,diag=E.P.expand_complete_material_face_ring(cache,seed,np.empty((0,3),np.int64),material)
 rows=np.asarray(rows,np.int64);added=np.asarray(added,np.int64)
 if len(rows)>MAX_CELLS:raise ValueError(f"cell cap 64 exceeded: {len(rows)}")
 oldt=t[rows];oldmat=mat[rows]
 if len(np.unique(oldmat))!=1:raise ValueError("cavity ring crosses material")
 verts=np.unique(oldt)
 if len(verts)>MAX_POINTS:raise ValueError(f"point cap 24 exceeded: {len(verts)}")
 oldb=np.asarray(T.m.boundary(oldt),np.int64)
 closed,_=E.P.checked_closed_boundary(oldb,"fixed-boundary")
 topo=T.manifold_topology(oldb)
 if not closed or topo.get("bad_vertex_links",0) or topo.get("boundary_components")!=1:raise ValueError("current cavity boundary not one closed manifold")
 loc={int(v):i for i,v in enumerate(verts)}
 lb=np.asarray([[loc[int(v)] for v in f] for f in oldb],np.int64)
 return {"edge":(a,b),"seed":seed,"rows":rows,"added":added,"ring":diag,"material":material,
         "oldt":oldt,"oldmat":oldmat,"verts":verts,"oldb":oldb,"lb":lb}
def solve(cache,edge,seconds=3.):
 p=np.asarray(cache["points"],np.float32);c=cavity(cache,edge)
 oe=E.exact_positive(p,c["oldt"])
 if not oe or min(oe)<=0:raise ValueError("old cavity exact F32 sign failure")
 oldsum=sum(oe);lp=p[c["verts"]];flux=int(ENG.exact_boundary_flux(lp,c["lb"]))
 if flux!=oldsum:raise ValueError("old cell det6 does not equal fixed-boundary flux")
 globalflux=int(ENG.exact_boundary_flux(p,cache["target"]))
 rep,nloc=ENG.S.solve(lp,c["lb"],seconds=min(3.,max(.02,float(seconds))),max_points=MAX_POINTS,material=c["material"])
 if nloc is None or rep.get("status")!="PASS_PARTIAL_POSITIVE_CAVITY_MILP":raise ValueError("no positive native-feasible incumbent: "+str(rep.get("status")))
 nloc=np.asarray(nloc,np.int64)
 if nloc.ndim!=2 or nloc.shape[1]!=4 or np.any(nloc<0) or np.any(nloc>=len(c["verts"])):raise ValueError("MILP tet index outside fixed pool")
 newt=c["verts"][nloc];ne=E.exact_positive(p,newt)
 if not ne or min(ne)<=0:raise ValueError("new exact F32 sign failure")
 newsum=sum(ne)
 if newsum!=flux or chain(T.m.boundary(newt))!=chain(c["oldb"]):raise ValueError("new tets fail exact oriented cavity boundary/flux")
 od,oq,og=E.native(p,c["oldt"]);nd,nq,ng=E.native(p,newt)
 oldbad=int(np.count_nonzero(~og));newbad=int(np.count_nonzero(~ng))
 olddef=E.deficit(od,oq);newdef=E.deficit(nd,nq);tol=max(1e-8,1e-6*olddef)
 if newbad>oldbad:raise ValueError("cavity bad count regresses")
 if not(newbad<oldbad or (newbad==oldbad and newdef<olddef-tol)):raise ValueError("no strict count/deficit improvement")
 t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"]);keep=np.ones(len(t),bool);keep[c["rows"]]=False
 ret=np.flatnonzero(keep).astype(np.int64);outt=np.vstack((t[ret],newt));outmat=np.r_[mat[ret],np.full(len(newt),c["material"],dtype=mat.dtype)]
 actual=np.asarray(T.m.boundary(outt),np.int64);target=np.asarray(cache["target"],np.int64)
 if chain(actual)!=chain(target):raise ValueError("whole target boundary changed")
 if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("duplicate full-volume tet")
 allsign=E.exact_positive(p,outt)
 if not allsign or min(allsign)<=0:raise ValueError("whole volume exact sign failure")
 bc,_,_,_=native_by_mat(p,t,mat);ac,_,_,_=native_by_mat(p,outt,outmat)
 if any(ac.get(k,0)>v for k,v in bc.items()):raise ValueError("per-material native bad count regresses")
 score=np.asarray(ENG.J.score(p,target),np.float64)
 return {**c,"newt":newt,"solver":rep,"oldsum":int(oldsum),"newsum":int(newsum),"delta":int(newsum-oldsum),
  "globalflux":globalflux,"oldbad":oldbad,"newbad":newbad,"olddef":olddef,"newdef":newdef,
  "oldq":[float(oq.min()),float(oq.max())],"newq":[float(nq.min()),float(nq.max())],
  "target_min_apex":float(score.min()) if len(score) else None,"bad_before_mat":bc,"bad_after_mat":ac,
  "target":np.asarray(cache["target"],np.int64),"ret":ret,"outt":outt,"outmat":outmat,"actual":actual,"progress":"count" if newbad<oldbad else "deficit_tie"}
def materialize(cache,pr):
 a0=cache["arrays"];nold=len(cache["tets"]);ret=pr["ret"];newt=pr["newt"];nnew=len(newt)
 out={k:np.array(v,copy=True) for k,v in a0.items()}
 special={"tetrahedra","material_indices","boundary_faces","desired_boundary_faces","mapped_current_boundary_faces","proposed_current_desired_boundary_faces",
 "retained_input_cell_rows","output_cell_to_input_cell_rows","output_cell_to_input_cell","input_cell_to_output_cell_rows","input_cell_to_output_cell",
 "replacement_tetrahedra","replacement_parent_offsets","replacement_parent_cell_rows","fixed_boundary_cavity_edge_volume_rows",
 "fixed_boundary_cavity_old_cell_rows","fixed_boundary_cavity_old_boundary_faces","fixed_boundary_cavity_new_tetrahedra",
 "fixed_boundary_cavity_parent_offsets","fixed_boundary_cavity_parent_cell_rows","changed_output_cell_rows","changed_input_cell_rows","affected_input_cell_rows"}
 sent={}
 for k,x in list(out.items()):
  a=np.asarray(x)
  if k in special or a.ndim==0 or a.shape[0]!=nold:continue
  if a.dtype.kind in "iu":fill=-1 if a.dtype.kind=="i" else 0
  elif a.dtype.kind=="b":fill=False
  elif a.dtype.kind in "fc":fill=0
  elif a.dtype.kind in "US":fill=""
  else:raise ValueError("unknown cell metadata dtype "+k)
  out[k]=np.concatenate((a[ret],np.full((nnew,)+a.shape[1:],fill,dtype=a.dtype)),axis=0);sent[k]={"dtype":str(a.dtype),"fill":str(fill)}
 nout=len(pr["outt"]);origin=np.r_[ret,np.full(nnew,-1,np.int64)];inverse=np.full(nold,-1,np.int64);inverse[ret]=np.arange(len(ret))
 changed=np.arange(len(ret),nout,dtype=np.int64);parents=pr["rows"];offs=np.arange(nnew+1,dtype=np.int64)*len(parents);prow=np.tile(parents,nnew).astype(np.int64)
 out.update({"tetrahedra":pr["outt"],"material_indices":pr["outmat"],"boundary_faces":pr["actual"],"desired_boundary_faces":pr["target"],
 "mapped_current_boundary_faces":pr["actual"],"proposed_current_desired_boundary_faces":pr["target"],
 "retained_input_cell_rows":ret,"output_cell_to_input_cell_rows":origin,"output_cell_to_input_cell":origin,
 "input_cell_to_output_cell_rows":inverse,"input_cell_to_output_cell":inverse,
 "input_node_to_output_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"input_node_to_output_node":np.arange(len(cache["points"]),dtype=np.int64),
 "output_node_to_input_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"output_node_to_input_node":np.arange(len(cache["points"]),dtype=np.int64),
 "changed_output_cell_rows":changed,"changed_input_cell_rows":parents,"affected_input_cell_rows":parents,
 "replacement_tetrahedra":newt,"replacement_parent_offsets":offs,"replacement_parent_cell_rows":prow,
 "fixed_boundary_cavity_edge_volume_rows":np.asarray(pr["edge"],np.int64),"fixed_boundary_cavity_old_cell_rows":parents,
 "fixed_boundary_cavity_old_boundary_faces":pr["oldb"],"fixed_boundary_cavity_new_tetrahedra":newt,
 "fixed_boundary_cavity_parent_offsets":offs,"fixed_boundary_cavity_parent_cell_rows":prow,
 "fixed_boundary_cavity_material":np.asarray(pr["material"],np.int32),
 "fixed_boundary_cavity_exact_det6_delta_integer":np.asarray(str(pr["delta"])),
 
 "fixed_boundary_cavity_point_map_identity":np.arange(len(cache["points"]),dtype=np.int64),
 "fixed_boundary_cavity_cell_metadata_sentinels_json":np.asarray(json.dumps(sent,sort_keys=True))})
 return out,sent
def replay(cache,child,j):
 c=cavity(cache,j["edge"]);t=cache["tets"];mat=cache["mat"];p=cache["points"];rows=c["rows"]
 if not np.array_equal(rows,np.asarray(j["cavity_cell_rows"],np.int64)):raise ValueError("edge-star/ring rows differ")
 if not np.array_equal(c["verts"],np.asarray(j["cavity_vertices"],np.int64)):raise ValueError("cavity point pool differs")
 if np.asarray(child["points_world_m"],np.float32).tobytes()!=np.asarray(p,np.float32).tobytes():raise ValueError("point bytes changed")
 ret=np.setdiff1d(np.arange(len(t),dtype=np.int64),rows,assume_unique=True)
 if not np.array_equal(child["tetrahedra"][:len(ret)],t[ret]) or not np.array_equal(child["material_indices"][:len(ret)],mat[ret]):raise ValueError("retained cells/material changed")
 newt=np.asarray(child["tetrahedra"][len(ret):],np.int64)
 if not np.array_equal(newt,np.asarray(j["new_tetrahedra"],np.int64)):raise ValueError("output tet set differs from journal")
 if not np.array_equal(child["desired_boundary_faces"],cache["target"]):raise ValueError("target faces array changed")
 if chain(T.m.boundary(child["tetrahedra"]))!=chain(cache["target"]):raise ValueError("global oriented boundary mismatch")
 if chain(T.m.boundary(newt))!=chain(c["oldb"]):raise ValueError("cavity oriented boundary mismatch")
 if len(np.unique(np.sort(child["tetrahedra"],axis=1),axis=0))!=len(child["tetrahedra"]):raise ValueError("duplicate output tetrahedra")
 oe=E.exact_positive(p,t[rows]);ne=E.exact_positive(p,newt)
 if not oe or not ne or min(oe)<=0 or min(ne)<=0 or sum(oe)!=sum(ne):raise ValueError("exact F32 sign or exact volume fails")
 if sum(oe)!=ENG.exact_boundary_flux(p[c["verts"]],c["lb"]):raise ValueError("old exact flux mismatch")
 if not np.array_equal(child["replacement_parent_offsets"],np.arange(len(newt)+1,dtype=np.int64)*len(rows)):raise ValueError("parent CSR offsets mismatch")
 if not np.array_equal(child["replacement_parent_cell_rows"],np.tile(rows,len(newt))):raise ValueError("full immediate parent union mismatch")
 ob=int(np.count_nonzero(~E.native(p,t[rows])[2]));nb=int(np.count_nonzero(~E.native(p,newt)[2]))
 if nb>ob:raise ValueError("cavity native bad count regression")
 return {"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_REPLAY","cavity_cells":len(rows),"cavity_points":len(c["verts"]),
 "replacement_cells":len(newt),"exact_det6_delta":"0","retained_cells_identical":True,"point_bytes_fixed":True,
 "target_boundary_exact":True,"full_parent_CSR":True,"cavity_bad_before":ob,"cavity_bad_after":nb,
 "physical_admission":False,"native_steps":0}
def run(state_path,role,edge,outdir,fixture=False,seconds=3.):
 out=Path(outdir).resolve()
 if out.exists():raise FileExistsError("fresh output required")
 out.mkdir(parents=True)
 ctx=E.load_context(state_path,out,fixture=fixture,source_role="D");st=ctx["st"];cache=ctx["caches"][role];pr=solve(cache,edge,seconds)
 arrays,sent=materialize(cache,pr);vp=out/(role.lower()+"-fixed-boundary-cavity-volume-proposal.npz");np.savez_compressed(vp,**arrays);vref=ref(vp)
 j={"schema":"h706.fixed-boundary-edge-cavity-retet.journal.v1","role":role,"edge":list(map(int,edge)),
 "input_state":ref(state_path),"parent_volume":dict(st[role+"_current"]),"cavity_cell_rows":pr["rows"].tolist(),
 "seed_edge_star_rows":pr["seed"].tolist(),"one_ring_added_rows":pr["added"].tolist(),"cavity_vertices":pr["verts"].tolist(),
 "cavity_boundary_faces":pr["oldb"].tolist(),"material":int(pr["material"]),"new_tetrahedra":pr["newt"].tolist(),
 "new_cell_parent_union":pr["rows"].tolist(),"parent_union_complete":True,"old_det6":str(pr["oldsum"]),
 "new_det6":str(pr["newsum"]),"exact_det6_delta":str(pr["delta"]),"solver":pr["solver"],"all_points_fixed":True,
 "source_surfaces_unchanged":True,"sentinels":sent,"physical_admission":False,"native_steps":0}
 jp=out/"operation-journal.json";E.write_json(jp,j);ind=replay(cache,E.load_npz(vp),j)
 state=copy.deepcopy(ctx["root"]);ost=state.get("current_verified_state_20261031",state);cur=dict(st[role+"_current"]);cur.update(vref)
 cur.update({"cells":len(arrays["tetrahedra"]),"nodes":len(arrays["points_world_m"]),"native_quality_failures":sum(pr["bad_after_mat"].values()),
 "nonpositive":0,"oriented_volume_target_boundary_equal":True,"boundary_source_binding_status":"PASS_FIXED_BOUNDARY_REPLAY_PENDING_CONTEXT"})
 ost[role+"_current"]=cur
 for k in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay","independent_target_replay","source_evidence","source_audit","native_qualification","mass_mapping_receipt"):ost.pop(k,None)
 ost.update({"source_state":ref(state_path),"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING",
 "fixed_boundary_cavity_remesh":{"status":"PASS_LOCAL_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDIT","role":role,"edge_volume_rows":list(map(int,edge)),
 "all_point_bytes_fixed":True,"source_surfaces_unchanged":True,"exact_target_preserved":True,"native_admission":False}})
 sp=out/"candidate-state-proposal.json";E.write_json(sp,state)
 report={"schema":"h706.fixed-boundary-edge-cavity-retet.report.v1","status":"PASS_LOCAL_INDEPENDENT_REPLAY_PENDING_EXTERNAL_775_805_ALL185",
 "input_state":ref(state_path),"parent_volume":dict(st[role+"_current"]),"producer":ref(Path(__file__)),"volume_candidate":vref,
 "candidate_state":ref(sp),"journal":ref(jp),"independent_replay":ind,"role":role,"edge_volume_rows":list(map(int,edge)),
 "seed_cell_count":len(pr["seed"]),"ring_added_cell_count":len(pr["added"]),"cavity_cell_count":len(pr["rows"]),
 "cavity_point_count":len(pr["verts"]),"replacement_cell_count":len(pr["newt"]),"material":int(pr["material"]),
 "quality":{"bad_before":pr["oldbad"],"bad_after":pr["newbad"],"deficit_before":pr["olddef"],"deficit_after":pr["newdef"],
 "progress":pr["progress"],"q_before":pr["oldq"],"q_after":pr["newq"],"fixed_target_min_joint_apex":pr["target_min_apex"],
 "bad_by_material_before":pr["bad_before_mat"],"bad_by_material_after":pr["bad_after_mat"]},
 "exact_old_det6":str(pr["oldsum"]),"exact_new_det6":str(pr["newsum"]),"exact_det6_delta":str(pr["delta"]),
 "exact_global_flux":str(pr["globalflux"]),"all_points_fixed":True,"source_surfaces_unchanged":True,
 "exact_oriented_target_boundary_preserved":True,"full_parent_CSR_complete":True,"all185":"PENDING",
 "native_steps":0,"physical_admission":False,"mass_mapping":"PENDING"}
 rp=out/"report.json";E.write_json(rp,report);return report
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--state",type=Path,required=True)
 ap.add_argument("--role",choices=("Skin","Fat"),default="Fat");ap.add_argument("--edge-volume",required=True)
 ap.add_argument("--output",type=Path,required=True);ap.add_argument("--solve-seconds",type=float,default=3.);ap.add_argument("--fixture",action="store_true")
 a=ap.parse_args();edge=tuple(map(int,a.edge_volume.split(",")))
 if len(edge)!=2:ap.error("--edge-volume expects a,b")
 r=run(a.state,a.role,edge,a.output,a.fixture,a.solve_seconds)
 print(json.dumps({"status":r["status"],"role":r["role"],"edge":r["edge_volume_rows"],
 "cavity_cells":r["cavity_cell_count"],"cavity_points":r["cavity_point_count"],"replacement_cells":r["replacement_cell_count"],
 "quality":r["quality"],"exact_det6_delta":r["exact_det6_delta"],"report":r["candidate_state"]},indent=2))
if __name__=="__main__":main()

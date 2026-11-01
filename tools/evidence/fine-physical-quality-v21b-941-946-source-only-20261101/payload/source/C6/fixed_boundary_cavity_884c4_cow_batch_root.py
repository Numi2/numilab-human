#!/usr/bin/env python3
"""Batch fixed-boundary edge-star cavity retets for current Skin and/or Fat volumes."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,itertools,json,shutil,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE/"fixed_boundary_cavity_884b_root.py"
PARENT_SHA="8131e4b6198abe5bbcba828ff1a97acc0510700b759602f2901268c825a67f3c"
def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def ref(p):
 p=Path(p).resolve();return {"path":str(p),"sha256":sha(p)}
def imp(name,path,digest):
 path=Path(path).resolve();got=sha(path)
 if got!=digest:raise RuntimeError(f"{name} SHA mismatch {got}")
 spec=importlib.util.spec_from_file_location(name,path)
 if not spec or not spec.loader:raise RuntimeError("import failed "+str(path))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
E=imp("fixed_boundary_parent_884b",PARENT,PARENT_SHA)
B,ENG,M,T=E.B,E.ENG,E.M,E.T
P883=E.E
COW_PATH=HERE.parent/"npz-cow-append-925-root"/"npz_cow_append_925.py"
COW_SHA="4799f33d73a37072498565b6ff9c20d181916f0158e253fe5c7e9056ba2c0270"
COW=imp("npz_cow_append_925_pinned",COW_PATH,COW_SHA)
COW_MIN_FREE_BYTES=64*1024*1024

def cow_changed_payload(parent_arrays,output_arrays):
 parent_keys=set(parent_arrays);output_keys=set(output_arrays)
 missing=parent_keys-output_keys
 if missing:raise ValueError("materializer omitted parent NPZ arrays: "+repr(sorted(missing)[:12]))
 payload={}
 for key,value in output_arrays.items():
  a=np.asarray(value)
  if a.dtype.hasobject:raise TypeError("object array in materialized payload: "+key)
  if key not in parent_arrays:
   payload[key]=a;continue
  b=np.asarray(parent_arrays[key])
  same=(a.dtype==b.dtype and a.shape==b.shape and
        a.flags.f_contiguous==b.flags.f_contiguous and
        a.tobytes(order="A")==b.tobytes(order="A"))
  if not same:payload[key]=a
 return payload
def face_rows(tet):
 a,b,c,d=map(int,tet)
 return ((a,b,c),(a,d,b),(a,c,d),(b,d,c))
def components_and_edges(cache):
 t=np.asarray(cache["tets"],np.int64);p=np.asarray(cache["points"],np.float32)
 bad=np.flatnonzero(np.asarray(cache["bad"],bool)).astype(np.int64)
 if not len(bad):return []
 parent={int(r):int(r) for r in bad}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(int(a));b=find(int(b))
  if a!=b:parent[max(a,b)]=min(a,b)
 owners={}
 for r in bad:
  for f in face_rows(t[int(r)]):
   k=tuple(sorted(f))
   if k in owners:union(int(r),owners[k])
   else:owners[k]=int(r)
 groups={}
 for r in bad:groups.setdefault(find(int(r)),[]).append(int(r))
 out=[]
 for root,rows in groups.items():
  ec={}
  for r in rows:
   for a,b in itertools.combinations(map(int,t[r]),2):
    e=tuple(sorted((a,b)));ec[e]=ec.get(e,0)+1
  edges=sorted(ec,key=lambda e:(-ec[e],float(np.linalg.norm(p[e[0]].astype(np.float64)-p[e[1]].astype(np.float64))),e))
  out.append({"component_root":int(root),"bad_cell_rows":np.asarray(sorted(rows),np.int64),"edges":[{"edge":list(e),"bad_edge_hit_count":int(ec[e]),"length_m":float(np.linalg.norm(p[e[0]].astype(np.float64)-p[e[1]].astype(np.float64)))} for e in edges]})
 out.sort(key=lambda c:(-len(c["bad_cell_rows"]),c["component_root"]))
 return out
def solve_op(cache,edge,seconds):
 c=E.cavity(cache,edge);p=np.asarray(cache["points"],np.float32)
 if len(c["rows"])>64 or len(c["verts"])>24:raise ValueError("64-cell/24-point hard cap")
 od,oq,og=P883.native(p,c["oldt"]);oldbad=int(np.count_nonzero(~og))
 if not oldbad:raise ValueError("cavity contains no current native-bad cells")
 olddef=P883.deficit(od,oq)
 exact_old=P883.exact_positive(p,c["oldt"])
 if not exact_old or min(exact_old)<=0:raise ValueError("old cavity has nonpositive exact F32 cell")
 oldsum=sum(exact_old);oldflux=int(ENG.exact_boundary_flux(p[c["verts"]],c["lb"]))
 if oldsum!=oldflux:raise ValueError("old cavity exact determinant sum differs from fixed oriented boundary flux")
 bq=np.asarray(ENG.J.score(p,c["oldb"]),np.float64);bmin=float(bq.min()) if len(bq) else None
 if bmin is not None and bmin<1e-5:raise ValueError(f"fixed cavity boundary joint-apex score below 1e-5: {bmin:.9g}")
 rep,nloc=ENG.S.solve(p[c["verts"]],c["lb"],seconds=min(3.,max(.02,float(seconds))),max_points=24,material=c["material"])
 if nloc is None or rep.get("status")!="PASS_PARTIAL_POSITIVE_CAVITY_MILP":raise ValueError("no native-feasible positive cavity incumbent: "+str(rep.get("status")))
 nloc=np.asarray(nloc,np.int64)
 if nloc.ndim!=2 or nloc.shape[1]!=4 or np.any(nloc<0) or np.any(nloc>=len(c["verts"])):raise ValueError("MILP output point index outside cavity pool")
 newt=c["verts"][nloc];ne=P883.exact_positive(p,newt)
 if not ne or min(ne)<=0:raise ValueError("new cavity has nonpositive exact F32 cell")
 newsum=sum(ne)
 if newsum!=oldflux or P883.chain(T.m.boundary(newt))!=P883.chain(c["oldb"]):raise ValueError("new cavity does not preserve exact oriented boundary/volume")
 nd,nq,ng=P883.native(p,newt);newbad=int(np.count_nonzero(~ng));newdef=P883.deficit(nd,nq);tol=max(1e-8,1e-6*olddef)
 if newbad>oldbad:raise ValueError("cavity native bad count regresses")
 if not(newbad<oldbad or (newbad==oldbad and newdef<olddef-tol)):raise ValueError("no strict bad-count or log-deficit progress")
 return {**c,"newt":newt,"solver":rep,"oldbad":oldbad,"newbad":newbad,"olddef":olddef,"newdef":newdef,
  "oldq":[float(oq.min()),float(oq.max())],"newq":[float(nq.min()),float(nq.max())],
  "oldsum":int(oldsum),"newsum":int(newsum),"delta":int(newsum-oldsum),"boundary_min_joint_apex":bmin}
def materialize_many(cache,ops):
 a0=cache["arrays"];t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"]);nold=len(t)
 retired=np.unique(np.concatenate([x["rows"] for x in ops])).astype(np.int64);keep=np.ones(nold,bool);keep[retired]=False
 ret=np.flatnonzero(keep).astype(np.int64);newt=np.vstack([x["newt"] for x in ops]);newmat=np.concatenate([np.full(len(x["newt"]),int(x["material"]),dtype=mat.dtype) for x in ops])
 outt=np.vstack((t[ret],newt));outmat=np.r_[mat[ret],newmat];actual=np.asarray(T.m.boundary(outt),np.int64)
 target=np.asarray(cache["target"],np.int64)
 if P883.chain(actual)!=P883.chain(target):raise ValueError("combined volume boundary differs from unchanged full current target")
 if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("combined output contains duplicate tetrahedra")
 parents=[];offs=[0];op_new_offsets=[0];op_cell_offsets=[0]
 for x in ops:
  cr=np.asarray(x["rows"],np.int64)
  for _ in x["newt"]:
   parents.extend(map(int,cr));offs.append(len(parents))
  op_new_offsets.append(op_new_offsets[-1]+len(x["newt"]));op_cell_offsets.append(op_cell_offsets[-1]+len(cr))
 parents=np.asarray(parents,np.int64);offs=np.asarray(offs,np.int64)
 special={"tetrahedra","material_indices","boundary_faces","desired_boundary_faces","mapped_current_boundary_faces","proposed_current_desired_boundary_faces",
  "retained_input_cell_rows","output_cell_to_input_cell_rows","output_cell_to_input_cell","input_cell_to_output_cell_rows","input_cell_to_output_cell",
  "replacement_tetrahedra","replacement_parent_offsets","replacement_parent_cell_rows","changed_output_cell_rows","changed_input_cell_rows","affected_input_cell_rows",
  "fixed_boundary_cavity_operation_edge_rows","fixed_boundary_cavity_operation_cell_offsets","fixed_boundary_cavity_operation_cell_rows",
  "fixed_boundary_cavity_operation_new_tet_offsets","fixed_boundary_cavity_operation_new_tetrahedra",
  "fixed_boundary_cavity_parent_offsets","fixed_boundary_cavity_parent_cell_rows","fixed_boundary_cavity_edge_volume_rows",
  "fixed_boundary_cavity_old_cell_rows","fixed_boundary_cavity_old_boundary_faces","fixed_boundary_cavity_new_tetrahedra"}
 out={k:np.array(v,copy=True) for k,v in a0.items()};sent={}
 for k,x in list(out.items()):
  a=np.asarray(x)
  if k in special or a.ndim==0 or a.shape[0]!=nold:continue
  if a.dtype.kind in "iu":fill=-1 if a.dtype.kind=="i" else 0
  elif a.dtype.kind=="b":fill=False
  elif a.dtype.kind in "fc":fill=0
  elif a.dtype.kind in "US":fill=""
  else:raise ValueError("unknown cell metadata dtype "+k)
  out[k]=np.concatenate((a[ret],np.full((len(newt),)+a.shape[1:],fill,dtype=a.dtype)),axis=0);sent[k]={"dtype":str(a.dtype),"fill":str(fill)}
 origin=np.r_[ret,np.full(len(newt),-1,np.int64)];inverse=np.full(nold,-1,np.int64);inverse[ret]=np.arange(len(ret))
 crflat=np.concatenate([np.asarray(x["rows"],np.int64) for x in ops]);erows=np.asarray([x["edge"] for x in ops],np.int64)
 out.update({"tetrahedra":outt,"material_indices":outmat,"boundary_faces":actual,"desired_boundary_faces":target,
  "mapped_current_boundary_faces":actual,"proposed_current_desired_boundary_faces":target,"retained_input_cell_rows":ret,
  "output_cell_to_input_cell_rows":origin,"output_cell_to_input_cell":origin,"input_cell_to_output_cell_rows":inverse,"input_cell_to_output_cell":inverse,
  "input_node_to_output_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"input_node_to_output_node":np.arange(len(cache["points"]),dtype=np.int64),
  "output_node_to_input_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"output_node_to_input_node":np.arange(len(cache["points"]),dtype=np.int64),
  "changed_output_cell_rows":np.arange(len(ret),len(outt),dtype=np.int64),"changed_input_cell_rows":retired,"affected_input_cell_rows":retired,
  "replacement_tetrahedra":newt,"replacement_parent_offsets":offs,"replacement_parent_cell_rows":parents,
  "fixed_boundary_cavity_operation_edge_rows":erows,"fixed_boundary_cavity_operation_cell_offsets":np.asarray(op_cell_offsets,np.int64),
  "fixed_boundary_cavity_operation_cell_rows":crflat,"fixed_boundary_cavity_operation_new_tet_offsets":np.asarray(op_new_offsets,np.int64),
  "fixed_boundary_cavity_operation_new_tetrahedra":newt,"fixed_boundary_cavity_parent_offsets":offs,
  "fixed_boundary_cavity_parent_cell_rows":parents,"fixed_boundary_cavity_cell_metadata_sentinels_json":np.asarray(json.dumps(sent,sort_keys=True))})
 return out,{"retained_rows":ret,"retired_rows":retired,"new_tets":newt,"parent_offsets":offs,"parent_rows":parents,
             "operation_cell_offsets":np.asarray(op_cell_offsets,np.int64),"operation_new_offsets":np.asarray(op_new_offsets,np.int64),"operation_rows":crflat,"edges":erows,"sentinels":sent}
def independent_replay(cache,child,ops,meta):
 t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"]);p=np.asarray(cache["points"],np.float32);target=np.asarray(cache["target"],np.int64)
 retired=np.asarray(meta["retired_rows"],np.int64);ret=np.asarray(meta["retained_rows"],np.int64)
 seen_cells=set();seen_verts=set();replacement=[];exactdeltas=[];oldbad_sum=newbad_sum=0
 for i,x in enumerate(ops):
  c=E.cavity(cache,x["edge"])
  if not np.array_equal(c["rows"],x["rows"]) or not np.array_equal(c["verts"],x["verts"]):raise ValueError("operation cavity does not reconstruct")
  if seen_cells&set(map(int,c["rows"])) or seen_verts&set(map(int,c["verts"])):raise ValueError("accepted cavities overlap by cell or vertex")
  seen_cells.update(map(int,c["rows"]));seen_verts.update(map(int,c["verts"]))
  olde=P883.exact_positive(p,t[c["rows"]]);ne=P883.exact_positive(p,x["newt"])
  if not olde or not ne or min(olde)<=0 or min(ne)<=0:raise ValueError("operation exact F32 sign failure")
  flux=int(ENG.exact_boundary_flux(p[c["verts"]],c["lb"]))
  if sum(olde)!=sum(ne) or sum(olde)!=flux or P883.chain(T.m.boundary(x["newt"]))!=P883.chain(c["oldb"]):raise ValueError("operation exact flux/boundary mismatch")
  ob=int(np.count_nonzero(~P883.native(p,t[c["rows"]])[2]));nb=int(np.count_nonzero(~P883.native(p,x["newt"])[2]))
  if nb>ob or not(nb<ob or (nb==ob and x["newdef"]<x["olddef"]-max(1e-8,1e-6*x["olddef"]))):raise ValueError("operation native progress/regression mismatch")
  if not np.all(np.asarray(child["material_indices"])[len(ret)+len(replacement):len(ret)+len(replacement)+len(x["newt"]) ]==x["material"]):raise ValueError("operation material mismatch")
  replacement.extend(range(len(x["newt"])))
  oldbad_sum+=ob;newbad_sum+=nb;exactdeltas.append(str(sum(ne)-sum(olde)))
 if not np.array_equal(np.asarray(child["points_world_m"],np.float32).tobytes(),p.tobytes()):raise ValueError("volume point bytes changed")
 if not np.array_equal(child["tetrahedra"][:len(ret)],t[ret]) or not np.array_equal(child["material_indices"][:len(ret)],mat[ret]):raise ValueError("retained tets/material changed")
 if not np.array_equal(child["tetrahedra"][len(ret):],meta["new_tets"]):raise ValueError("replacement tet order differs")
 if not np.array_equal(child["desired_boundary_faces"],target):raise ValueError("global target array changed")
 if P883.chain(T.m.boundary(child["tetrahedra"]))!=P883.chain(target):raise ValueError("combined whole-volume oriented boundary differs from target")
 if len(np.unique(np.sort(child["tetrahedra"],axis=1),axis=0))!=len(child["tetrahedra"]):raise ValueError("duplicate output tets")
 if not np.array_equal(child["replacement_parent_offsets"],meta["parent_offsets"]) or not np.array_equal(child["replacement_parent_cell_rows"],meta["parent_rows"]):raise ValueError("complete immediate parent CSR mismatch")
 # Full native replay is performed once on the combined output.
 _,_,oldcounts=P883.native(p,t);_,_,newcounts=P883.native(p,child["tetrahedra"])
 if int(np.count_nonzero(~newcounts))>int(np.count_nonzero(~oldcounts)):raise ValueError("whole-volume native bad count regresses")
 return {"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_BATCH_REPLAY","operations":len(ops),"cavity_old_cells":len(retired),
  "replacement_cells":len(meta["new_tets"]),"retained_cells_bit_identical":True,"all_point_bytes_fixed":True,
  "full_target_oriented_boundary_exact":True,"full_parent_CSR_complete":True,"local_exact_det6_deltas":exactdeltas,
  "cavity_native_bad_before":oldbad_sum,"cavity_native_bad_after":newbad_sum,
  "full_bad_before":int(np.count_nonzero(~oldcounts)),"full_bad_after":int(np.count_nonzero(~newcounts)),
  "physical_admission":False,"native_steps":0}
def role_batch(state_path,ctx,role,max_seconds=60.,solve_seconds=3.,max_patches=32):
 cache=ctx["caches"][role];components=components_and_edges(cache);start=time.monotonic();attempts=[];ops=[];reserved_cells=set();reserved_verts=set();solves=0;complete=True
 for ci,comp in enumerate(components):
  if len(ops)>=max_patches or time.monotonic()-start>=max_seconds:complete=False;break
  accepted=False
  for ei,edgeinfo in enumerate(comp["edges"]):
   if solves>=max(1,int(max_seconds//max(.1,solve_seconds))) or time.monotonic()-start>=max_seconds:complete=False;break
   e=tuple(map(int,edgeinfo["edge"]))
   try:
    c=E.cavity(cache,e)
    if reserved_cells&set(map(int,c["rows"])) or reserved_verts&set(map(int,c["verts"])):
     attempts.append({"component":ci,"edge":list(e),"status":"REJECTED_OVERLAP_WITH_ACCEPTED_CAVITY"});continue
    solves+=1;pr=solve_op(cache,e,min(solve_seconds,max_seconds-(time.monotonic()-start)))
    pr["component_root"]=comp["component_root"];pr["component_bad_rows"]=comp["bad_cell_rows"].tolist()
    reserved_cells.update(map(int,pr["rows"]));reserved_verts.update(map(int,pr["verts"]));ops.append(pr);accepted=True
    attempts.append({"component":ci,"edge":list(e),"status":"ACCEPTED","native_bad_before":pr["oldbad"],"native_bad_after":pr["newbad"],"cavity_cells":len(pr["rows"]),"cavity_points":len(pr["verts"])})
    break
   except Exception as exc:attempts.append({"component":ci,"edge":list(e),"status":"REJECTED","error_type":type(exc).__name__,"error":str(exc)})
  if not accepted:continue
 return {"role":role,"input_bad_count":int(np.count_nonzero(cache["bad"])),"component_count":len(components),
  "components":[{"root":c["component_root"],"bad_cells":c["bad_cell_rows"].tolist(),"candidate_edge_count":len(c["edges"])} for c in components],
  "ops":ops,"attempts":attempts,"solves":solves,"elapsed_s":time.monotonic()-start,
  "search_complete":bool(complete)}
def run_batch(state_path,outdir,roles=("Skin","Fat"),max_seconds=60.,solve_seconds=3.,max_patches=32,fixture=False):
 out=Path(outdir).resolve()
 if out.exists():raise FileExistsError("fresh output directory required")
 out.mkdir(parents=True)
 ctx=P883.load_context(state_path,out,fixture=fixture,source_role="D")
 results={r:role_batch(state_path,ctx,r,max_seconds,solve_seconds,max_patches) for r in roles}
 accepted={r:x["ops"] for r,x in results.items() if x["ops"]}
 summary={"schema":"h706.fixed-boundary-component-batch.884c.v1","status":"PREPARED_NO_ADMISSIBLE_PATCHES" if not accepted else "IN_MEMORY_BATCH_READY",
  "input_state":ref(state_path),"producer":ref(Path(__file__)),"roles":{r:{k:v for k,v in x.items() if k!="ops"} for r,x in results.items()},
  "accepted_counts":{r:len(ops) for r,ops in accepted.items()},"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
 if not accepted:
  P883.write_json(out/"batch-report.json",summary);return summary
 free=shutil.disk_usage(out).free;summary["free_bytes_before_output"]=free
 summary["minimum_free_bytes_for_output"]=COW_MIN_FREE_BYTES
 summary["storage_mode"]="APFS_COW_APPEND_NO_FULL_COPY_FALLBACK"
 summary["cow_owner"]=ref(COW_PATH)
 if not fixture and free<COW_MIN_FREE_BYTES:
  summary["status"]="IN_MEMORY_GAIN_BLOCKED_COW_DISK_FLOOR";P883.write_json(out/"batch-report.json",summary);return summary
 volume_outputs={};role_meta={};cow_reports={}
 for r,ops in accepted.items():
  cache=ctx["caches"][r];arrays,meta=materialize_many(cache,ops)
  parent_ref=dict(ctx["st"][r+"_current"]);parent_path=Path(parent_ref["path"]).resolve()
  if sha(parent_path)!=str(parent_ref["sha256"]):raise ValueError(r+" parent volume SHA changed before COW write")
  # The solver's cache is built from this hash-checked parent; compare the
  # materialized arrays against those decoded parent arrays to append deltas only.
  payload=cow_changed_payload(cache["arrays"],arrays)
  if not payload:raise ValueError(r+" accepted operations produced an empty COW payload")
  vp=out/(r.lower()+"-fixed-boundary-cavity-batch-proposal.npz")
  cow=COW.write_npz_clone(parent_path,vp,payload)
  if cow["source_sha256"]!=parent_ref["sha256"] or cow["output_sha256"]!=sha(vp):
   raise RuntimeError(r+" COW writer returned inconsistent source/output pins")
  crp=out/(r.lower()+"-cow-write-report.json");P883.write_json(crp,cow)
  cow_reports[r]={"report":ref(crp),"source_sha256":cow["source_sha256"],"output_sha256":cow["output_sha256"],
   "apfs_clonefile_used":cow["apfs_clonefile_used"],"full_copy_fallback":cow["full_copy_fallback"],
   "arrays_bitwise_verified":cow["readback_arrays_bitwise_verified"],
   "unchanged_compressed_members_bitwise_verified":cow["unchanged_compressed_members_bitwise_verified"],
   "replaced_keys":cow["replaced_keys"],"new_keys":cow["new_keys"],"deleted_keys":cow["deleted_keys"]}
  volume_outputs[r]=ref(vp);role_meta[r]=meta
 # Read exact serialized children and independently reconstruct each current-boundary cavity.
 independent={}
 for r,ops in accepted.items():
  child=P883.load_npz(volume_outputs[r]["path"]);independent[r]=independent_replay(ctx["caches"][r],child,ops,role_meta[r])
 state=copy.deepcopy(ctx["root"]);ost=state.get("current_verified_state_20261031",state)
 for r,ops in accepted.items():
  vref=volume_outputs[r];cur=dict(ctx["st"][r+"_current"]);cache=ctx["caches"][r]
  bad_after=int(independent[r]["full_bad_after"]);cur.update(vref);cur.update({"cells":len(cache["tets"])-independent[r]["cavity_old_cells"]+independent[r]["replacement_cells"],
   "nodes":len(cache["points"]),"native_quality_failures":bad_after,"nonpositive":0,"oriented_volume_target_boundary_equal":True,
   "boundary_source_binding_status":"PASS_FIXED_BOUNDARY_BATCH_REPLAY_PENDING_EXTERNAL_CONTEXT"})
  ost[r+"_current"]=cur
 for k in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay","independent_target_replay","source_evidence","source_audit","native_qualification","mass_mapping_receipt"):ost.pop(k,None)
 ost.update({"source_state":ref(state_path),"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING",
  "fixed_boundary_cavity_batch":{"status":"PASS_LOCAL_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS","roles":list(accepted),
  "all_volume_points_fixed":True,"all_source_surfaces_unchanged":True,"exact_current_target_preserved":True,"all185":"PENDING"}})
 sp=out/"candidate-state-proposal.json";P883.write_json(sp,state)
 journal={"schema":"h706.fixed-boundary-component-batch.journal.v1","parent_state":ref(state_path),"roles":list(accepted),
  "volume_candidates":volume_outputs,"operations":{r:[{"edge":list(map(int,x["edge"])),"cell_rows":x["rows"].tolist(),
  "vertices":x["verts"].tolist(),"seed_cell_rows":x["seed"].tolist(),"ring_added_rows":x["added"].tolist(),
  "old_boundary":x["oldb"].tolist(),"new_tets":x["newt"].tolist(),"material":x["material"],"old_bad":x["oldbad"],
  "new_bad":x["newbad"],"old_deficit":x["olddef"],"new_deficit":x["newdef"],"old_det6":str(x["oldsum"]),
  "new_det6":str(x["newsum"]),"exact_delta_det6":str(x["delta"]),"complete_parent_union":x["rows"].tolist(),
  "solver":x["solver"]} for x in ops] for r,ops in accepted.items()},
  "independent_replay":independent,"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
 jp=out/"operation-journal.json";P883.write_json(jp,journal)
 summary.update({"status":"PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS","volume_candidates":volume_outputs,
  "cow_writes":cow_reports,"candidate_state":ref(sp),"journal":ref(jp),"independent_replay":independent,
  "accepted_quality":{"roles":{r:[{"edge":list(map(int,x["edge"])),"bad_before":x["oldbad"],"bad_after":x["newbad"],
    "deficit_before":x["olddef"],"deficit_after":x["newdef"],"cavity_cells":len(x["rows"]),"cavity_points":len(x["verts"])} for x in ops] for r,ops in accepted.items()}},
  "full_OID_walls_all185":"PENDING_EXTERNAL","physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"})
 P883.write_json(out/"batch-report.json",summary);return summary
def synthetic_fixture(root):
 root=Path(root).resolve()
 if root.exists():raise FileExistsError("fixture output already exists")
 root.mkdir(parents=True);inp=root/"inputs";inp.mkdir()
 base=HERE/"run-883e-synthetic-fixture-001"/"inputs";st=json.loads((base/"state.json").read_text())
 p=np.asarray([[0,0,200],[0,0,-200],[1,0,0],[-.5,.8660254,0],[-.5,-.8660254,0]],np.float32)
 raw=np.asarray([[0,1,2,3],[0,1,3,4],[0,1,4,2]],np.int64);t=M.orient(p,raw);det,q=M.measurements(p,t)
 if np.any(det<=0) or np.count_nonzero(M.passes(det,q))!=0:raise AssertionError("each role needs three positive native-bad input tets")
 b=np.asarray(T.m.boundary(t),np.int64)
 arr={"points_world_m":p,"points_world_m_f32":p.copy(),"tetrahedra":t,"material_indices":np.full(3,16,np.int32),
 "boundary_faces":b,"desired_boundary_faces":b.copy(),"mapped_current_boundary_faces":b.copy(),"proposed_current_desired_boundary_faces":b.copy(),
 "retained_input_cell_rows":np.arange(3,dtype=np.int64),"output_cell_to_input_cell_rows":np.arange(3,dtype=np.int64),
 "output_cell_to_input_cell":np.arange(3,dtype=np.int64),"input_cell_to_output_cell_rows":np.arange(3,dtype=np.int64),"input_cell_to_output_cell":np.arange(3,dtype=np.int64),
 "input_node_to_output_node_rows":np.arange(5,dtype=np.int64),"input_node_to_output_node":np.arange(5,dtype=np.int64),
 "output_node_to_input_node_rows":np.arange(5,dtype=np.int64),"output_node_to_input_node":np.arange(5,dtype=np.int64)}
 refs={}
 for role in ("Skin","Fat"):
  vp=inp/(role.lower()+".npz");np.savez_compressed(vp,**arr);refs[role]={"path":str(vp),"sha256":sha(vp),"cells":3,"nodes":5,"native_quality_failures":3}
 st["Skin_current"]=refs["Skin"];st["Fat_current"]=refs["Fat"];sp=inp/"state.json";P883.write_json(sp,st)
 r=run_batch(sp,root/"batch",roles=("Skin","Fat"),max_seconds=60,solve_seconds=3,max_patches=32,fixture=True)
 if r["status"]!="PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS":raise AssertionError("paired synthetic batch did not replay")
 if r["independent_replay"]["Skin"]["full_bad_after"]!=0 or r["independent_replay"]["Fat"]["full_bad_after"]!=0:raise AssertionError("both roles must reach zero fixture failures")
 receipt={"schema":"h706.fixed-boundary-component-batch-fixture.884c.v1","status":"PASS","producer":ref(Path(__file__)),
  "input_state":ref(sp),"batch_report":ref(root/"batch"/"batch-report.json"),"candidate_state":r["candidate_state"],
  "volume_candidates":r["volume_candidates"],"independent_replay":r["independent_replay"],"skin_bad":3,"skin_after":0,"fat_bad":3,"fat_after":0,
  "all_points_fixed":True,"source_surfaces_unchanged":True,"each_cavity_exact_det6_delta":"0",
  "all185":"NOT_RUN","mass_mapping":"NOT_RUN","physical_admission":False,"native_steps":0}
 rp=root/"fixture-receipt.json";P883.write_json(rp,receipt)
 return {"status":"PASS","receipt":str(rp),"independent_replay":r["independent_replay"]}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--state",type=Path);ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--roles",default="Skin,Fat");ap.add_argument("--seconds-per-role",type=float,default=60.)
 ap.add_argument("--solve-seconds",type=float,default=3.);ap.add_argument("--max-patches",type=int,default=32);ap.add_argument("--fixture",action="store_true")
 a=ap.parse_args()
 if a.fixture:r=synthetic_fixture(a.output)
 else:
  if a.state is None:ap.error("--state required unless --fixture")
  roles=tuple(x.strip() for x in a.roles.split(",") if x.strip())
  if not roles or any(x not in ("Skin","Fat") for x in roles):ap.error("--roles must be Skin,Fat and/or Fat/Skin")
  r=run_batch(a.state,a.output,roles,a.seconds_per_role,a.solve_seconds,a.max_patches,False)
 print(json.dumps(r,indent=2));return r
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Batch fixed-boundary edge-star cavity retets for current Skin and/or Fat volumes."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,itertools,json,shutil,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE/"fixed_boundary_cavity_884c5_count_first_root.py"
PARENT_SHA="c21559d5871e0c3c8414bcde592550be8f5a44ad88afdada3d9bb02ff1d79467"
def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def ref(p):
 p=Path(p).resolve();return {"path":str(p),"sha256":sha(p)}
def imp(name,path,digest):
 path=Path(path).resolve();got=sha(path)
 if got!=digest:raise RuntimeError(f"{name} SHA mismatch {got}")
 spec=importlib.util.spec_from_file_location(name,path)
 if not spec or not spec.loader:raise RuntimeError("import failed "+str(path))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
C5=imp("fixed_boundary_parent_884c5",PARENT,PARENT_SHA)
E,B,ENG,M,T=C5.E,C5.B,C5.ENG,C5.M,C5.T
P883=C5.P883
COW_PATH=C5.COW_PATH
COW_SHA=C5.COW_SHA
COW=imp("npz_cow_append_925_pinned",COW_PATH,COW_SHA)
COW_MIN_FREE_BYTES=C5.COW_MIN_FREE_BYTES
MAX_CELLS=64
MAX_POINTS=24
_MILP_CALLS=0
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
_MILP_CALLS=0
def face_connected_rows(tets,rows):
 rows=np.asarray(rows,np.int64)
 if len(rows)<=1:return True
 owners={};adj=[set() for _ in rows];local={int(r):i for i,r in enumerate(rows)}
 for r in rows:
  i=local[int(r)]
  for f in face_rows(np.asarray(tets[int(r)],np.int64)):
   k=tuple(sorted(map(int,f)))
   prior=owners.get(k)
   if prior is None:owners[k]=i
   else:
    adj[i].add(prior);adj[prior].add(i)
 seen={0};stack=[0]
 while stack:
  i=stack.pop()
  for j in adj[i]:
   if j not in seen:seen.add(j);stack.append(j)
 return len(seen)==len(rows)

def cavity_union(cache,edges,cavities=None):
 edges=tuple(tuple(map(int,e)) for e in edges)
 if not 1<=len(edges)<=4:raise ValueError("connected cavity requires 1-4 source edge stars")
 if len(set(edges))!=len(edges):raise ValueError("duplicate source edge in cavity group")
 if cavities is None:cavities=[E.cavity(cache,e) for e in edges]
 if len(cavities)!=len(edges):raise ValueError("edge-star cavity count mismatch")
 rows=np.unique(np.concatenate([np.asarray(c["rows"],np.int64) for c in cavities]))
 if not len(rows):raise ValueError("empty connected edge-star union")
 if len(rows)>MAX_CELLS:raise ValueError("cell cap 64 exceeded: "+str(len(rows)))
 t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"]);p=np.asarray(cache["points"],np.float32)
 seed=np.unique(np.concatenate([np.asarray(c["seed"],np.int64) for c in cavities]))
 materials=np.unique(mat[rows])
 if len(materials)!=1:raise ValueError("multi-edge cavity crosses material")
 if not face_connected_rows(t,rows):raise ValueError("multi-edge cavity is not face-connected")
 oldt=t[rows];oldmat=mat[rows];verts=np.unique(oldt)
 if len(verts)>MAX_POINTS:raise ValueError("point cap 24 exceeded: "+str(len(verts)))
 oldb=np.asarray(T.m.boundary(oldt),np.int64)
 closed,_=P883.P.checked_closed_boundary(oldb,"fixed-boundary-multi-edge")
 topo=T.manifold_topology(oldb)
 if not closed or topo.get("bad_vertex_links",0) or topo.get("boundary_components")!=1:
  raise ValueError("combined cavity boundary not one closed manifold")
 loc={int(v):i for i,v in enumerate(verts)}
 lb=np.asarray([[loc[int(v)] for v in f] for f in oldb],np.int64)
 return {"edge":edges[0],"edges":edges,"seed":seed,"rows":rows,"added":np.setdiff1d(rows,seed,assume_unique=True),
  "ring":{"edge_star_count":len(edges),"face_connected":True},"material":int(materials[0]),"oldt":oldt,"oldmat":oldmat,
  "verts":verts,"oldb":oldb,"lb":lb}

def solve_op(cache,edges,seconds,cavity=None):
 global _MILP_CALLS
 edges=tuple(tuple(map(int,e)) for e in edges)
 c=cavity_union(cache,edges) if cavity is None else cavity
 p=np.asarray(cache["points"],np.float32)
 if len(c["rows"])>MAX_CELLS or len(c["verts"])>MAX_POINTS:raise ValueError("64-cell/24-point hard cap")
 od,oq,og=P883.native(p,c["oldt"]);oldbad=int(np.count_nonzero(~og))
 if not oldbad:raise ValueError("cavity contains no current native-bad cells")
 olddef=P883.deficit(od,oq)
 exact_old=P883.exact_positive(p,c["oldt"])
 if not exact_old or min(exact_old)<=0:raise ValueError("old cavity has nonpositive exact F32 cell")
 oldsum=sum(exact_old);oldflux=int(ENG.exact_boundary_flux(p[c["verts"]],c["lb"]))
 if oldsum!=oldflux:raise ValueError("old cavity exact determinant sum differs from fixed oriented boundary flux")
 bq=np.asarray(ENG.J.score(p,c["oldb"]),np.float64);bmin=float(bq.min()) if len(bq) else None
 if bmin is not None and bmin<1e-5:raise ValueError(f"fixed cavity boundary joint-apex score below 1e-5: {bmin:.9g}")
 _MILP_CALLS+=1
 rep,nloc=ENG.S.solve(p[c["verts"]],c["lb"],seconds=min(3.,max(.02,float(seconds))),max_points=MAX_POINTS,material=c["material"])
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
 if P883.chain(actual)!=P883.chain(target):raise ValueError("combined output boundary does not equal full current target")
 offs=[0];edgeflat=[]
 for x in ops:edgeflat.extend(x["edges"]);offs.append(len(edgeflat))
 edge_group_offsets=np.asarray(offs,np.int64);edge_group_rows=np.asarray(edgeflat,np.int64).reshape((-1,2))
 primary_edges=np.asarray([x["edges"][0] for x in ops],np.int64).reshape((-1,2))
 parent_rows=[];parent_offsets=[0];op_cell_rows=[];op_cell_offsets=[0];op_tet_offsets=[0]
 for x in ops:
  rows=np.asarray(x["rows"],np.int64);op_cell_rows.extend(map(int,rows));op_cell_offsets.append(len(op_cell_rows))
  for _ in x["newt"]:parent_rows.extend(map(int,rows));parent_offsets.append(len(parent_rows))
  op_tet_offsets.append(op_tet_offsets[-1]+len(x["newt"]))
 parents=np.asarray(parent_rows,np.int64);po=np.asarray(parent_offsets,np.int64)
 special={"tetrahedra","material_indices","boundary_faces","desired_boundary_faces","mapped_current_boundary_faces","proposed_current_desired_boundary_faces",
  "retained_input_cell_rows","output_cell_to_input_cell_rows","output_cell_to_input_cell","input_cell_to_output_cell_rows","input_cell_to_output_cell",
  "replacement_tetrahedra","replacement_parent_offsets","replacement_parent_cell_rows","changed_output_cell_rows","changed_input_cell_rows","affected_input_cell_rows",
  "fixed_boundary_cavity_operation_edge_rows","fixed_boundary_cavity_operation_edge_group_offsets","fixed_boundary_cavity_operation_edge_group_rows",
  "fixed_boundary_cavity_operation_cell_offsets","fixed_boundary_cavity_operation_cell_rows","fixed_boundary_cavity_operation_new_tet_offsets","fixed_boundary_cavity_operation_new_tetrahedra",
  "fixed_boundary_cavity_parent_offsets","fixed_boundary_cavity_parent_cell_rows","fixed_boundary_cavity_old_cell_rows","fixed_boundary_cavity_old_boundary_faces"}
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
 crflat=np.concatenate([np.asarray(x["rows"],np.int64) for x in ops]);erows=np.asarray([x["edges"][0] for x in ops],np.int64).reshape((-1,2))
 out.update({"tetrahedra":outt,"material_indices":outmat,"boundary_faces":actual,"desired_boundary_faces":target,
  "mapped_current_boundary_faces":actual,"proposed_current_desired_boundary_faces":target,"retained_input_cell_rows":ret,
  "output_cell_to_input_cell_rows":origin,"output_cell_to_input_cell":origin,"input_cell_to_output_cell_rows":inverse,"input_cell_to_output_cell":inverse,
  "input_node_to_output_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"input_node_to_output_node":np.arange(len(cache["points"]),dtype=np.int64),
  "output_node_to_input_node_rows":np.arange(len(cache["points"]),dtype=np.int64),"output_node_to_input_node":np.arange(len(cache["points"]),dtype=np.int64),
  "changed_output_cell_rows":np.arange(len(ret),len(outt),dtype=np.int64),"changed_input_cell_rows":retired,"affected_input_cell_rows":retired,
  "replacement_tetrahedra":newt,"replacement_parent_offsets":po,"replacement_parent_cell_rows":parents,
  "fixed_boundary_cavity_operation_edge_rows":primary_edges,"fixed_boundary_cavity_operation_edge_group_offsets":edge_group_offsets,
  "fixed_boundary_cavity_operation_edge_group_rows":edge_group_rows,
  "fixed_boundary_cavity_operation_cell_offsets":np.asarray(op_cell_offsets,np.int64),"fixed_boundary_cavity_operation_cell_rows":crflat,
  "fixed_boundary_cavity_operation_new_tet_offsets":np.asarray(op_tet_offsets,np.int64),"fixed_boundary_cavity_operation_new_tetrahedra":newt,
  "fixed_boundary_cavity_parent_offsets":po,"fixed_boundary_cavity_parent_cell_rows":parents,
  "fixed_boundary_cavity_cell_metadata_sentinels_json":np.asarray(json.dumps(sent,sort_keys=True))})
 return out,{"retained_rows":ret,"retired_rows":retired,"new_tets":newt,"parent_offsets":po,"parent_rows":parents,
  "operation_cell_offsets":np.asarray(op_cell_offsets,np.int64),"operation_new_offsets":np.asarray(op_tet_offsets,np.int64),"operation_rows":crflat,
  "edges":primary_edges,"edge_group_offsets":edge_group_offsets,"edge_group_rows":edge_group_rows,"sentinels":sent}
def independent_replay(cache,child,ops,meta):
 t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"]);p=np.asarray(cache["points"],np.float32);target=np.asarray(cache["target"],np.int64)
 retired=np.asarray(meta["retired_rows"],np.int64);ret=np.asarray(meta["retained_rows"],np.int64)
 seen_cells=set();seen_verts=set();exactdeltas=[];oldbad_sum=newbad_sum=0;replacement_offset=0
 for i,x in enumerate(ops):
  c=cavity_union(cache,x["edges"])
  if not np.array_equal(c["rows"],x["rows"]) or not np.array_equal(c["verts"],x["verts"]):raise ValueError("operation multi-edge cavity does not reconstruct")
  if seen_cells&set(map(int,c["rows"])) or seen_verts&set(map(int,c["verts"])):raise ValueError("accepted cavities overlap by cell or vertex")
  seen_cells.update(map(int,c["rows"]));seen_verts.update(map(int,c["verts"]))
  olde=P883.exact_positive(p,t[c["rows"]]);ne=P883.exact_positive(p,x["newt"])
  if not olde or not ne or min(olde)<=0 or min(ne)<=0:raise ValueError("operation exact F32 sign failure")
  flux=int(ENG.exact_boundary_flux(p[c["verts"]],c["lb"]))
  if sum(olde)!=sum(ne) or sum(olde)!=flux or P883.chain(T.m.boundary(x["newt"]))!=P883.chain(c["oldb"]):raise ValueError("operation exact flux/boundary mismatch")
  ob=int(np.count_nonzero(~P883.native(p,t[c["rows"]])[2]));nb=int(np.count_nonzero(~P883.native(p,x["newt"])[2]))
  if nb>ob or not(nb<ob or (nb==ob and x["newdef"]<x["olddef"]-max(1e-8,1e-6*x["olddef"]))):raise ValueError("operation native progress/regression mismatch")
  if not np.all(np.asarray(child["material_indices"])[len(ret)+replacement_offset:len(ret)+replacement_offset+len(x["newt"])]==x["material"]):raise ValueError("operation material mismatch")
  replacement_offset+=len(x["newt"]);oldbad_sum+=ob;newbad_sum+=nb;exactdeltas.append(str(sum(ne)-sum(olde)))
 if not np.array_equal(np.asarray(child["points_world_m"],np.float32).tobytes(),p.tobytes()):raise ValueError("volume point bytes changed")
 if not np.array_equal(child["tetrahedra"][:len(ret)],t[ret]) or not np.array_equal(child["material_indices"][:len(ret)],mat[ret]):raise ValueError("retained tets/material changed")
 if not np.array_equal(child["tetrahedra"][len(ret):],meta["new_tets"]):raise ValueError("replacement tet order differs")
 if not np.array_equal(child["desired_boundary_faces"],target):raise ValueError("global target array changed")
 if P883.chain(T.m.boundary(child["tetrahedra"]))!=P883.chain(target):raise ValueError("combined whole-volume oriented boundary differs from target")
 if len(np.unique(np.sort(child["tetrahedra"],axis=1),axis=0))!=len(child["tetrahedra"]):raise ValueError("duplicate output tets")
 if not np.array_equal(child["replacement_parent_offsets"],meta["parent_offsets"]) or not np.array_equal(child["replacement_parent_cell_rows"],meta["parent_rows"]):raise ValueError("complete immediate parent CSR mismatch")
 if not np.array_equal(child["fixed_boundary_cavity_operation_edge_rows"],meta["edges"]):raise ValueError("primary operation edge rows mismatch")
 if not np.array_equal(child["fixed_boundary_cavity_operation_edge_group_offsets"],meta["edge_group_offsets"]) or not np.array_equal(child["fixed_boundary_cavity_operation_edge_group_rows"],meta["edge_group_rows"]):raise ValueError("multi-edge operation identifier CSR mismatch")
 if not np.array_equal(child["fixed_boundary_cavity_operation_cell_rows"],meta["operation_rows"]):raise ValueError("operation cell rows mismatch")
 if not np.array_equal(child["fixed_boundary_cavity_operation_cell_offsets"],meta["operation_cell_offsets"]):raise ValueError("operation cell offsets mismatch")
 if not np.array_equal(child["fixed_boundary_cavity_operation_new_tet_offsets"],meta["operation_new_offsets"]):raise ValueError("operation tet offsets mismatch")
 _,_,oldcounts=P883.native(p,t);_,_,newcounts=P883.native(p,child["tetrahedra"])
 if int(np.count_nonzero(~newcounts))>int(np.count_nonzero(~oldcounts)):raise ValueError("whole-volume native bad count regresses")
 return {"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_BATCH_REPLAY","operations":len(ops),"cavity_old_cells":len(retired),
  "replacement_cells":len(meta["new_tets"]),"retained_cells_bit_identical":True,"all_point_bytes_fixed":True,
  "full_target_oriented_boundary_exact":True,"full_parent_CSR_complete":True,"multi_edge_identifier_CSR_exact":True,
  "local_exact_det6_deltas":exactdeltas,"cavity_native_bad_before":oldbad_sum,"cavity_native_bad_after":newbad_sum,
  "full_bad_before":int(np.count_nonzero(~oldcounts)),"full_bad_after":int(np.count_nonzero(~newcounts)),"physical_admission":False,"native_steps":0}
def role_batch(state_path,ctx,role,max_seconds=60.,solve_seconds=3.,max_patches=32,max_attempts_per_role=512):
 cache=ctx["caches"][role];components=components_and_edges(cache);start=time.monotonic()
 attempts=[];ops=[];reserved_cells=set();reserved_verts=set();solve_calls=0;edge_trials=0;complete=True;milp_before=_MILP_CALLS
 for ci,comp in enumerate(components):
  if len(ops)>=max_patches or time.monotonic()-start>=max_seconds:complete=False;break
  singles=[];single_candidates=[];single_progress=False;component_interrupted=False
  for ei,edgeinfo in enumerate(comp["edges"]):
   if edge_trials>=max_attempts_per_role or time.monotonic()-start>=max_seconds:
    complete=False;component_interrupted=True;break
   edge_trials+=1;e=tuple(map(int,edgeinfo["edge"]))
   try:
    c=E.cavity(cache,e)
    if reserved_cells&set(map(int,c["rows"])) or reserved_verts&set(map(int,c["verts"])):
     attempts.append({"component":ci,"edge_group":[list(e)],"status":"REJECTED_OVERLAP_WITH_ACCEPTED_CAVITY"});continue
    singles.append((e,c))
    solve_calls+=1;remaining=max(.02,max_seconds-(time.monotonic()-start))
    pr=solve_op(cache,(e,),min(solve_seconds,remaining),cavity_union(cache,(e,),(c,)))
    pr["component_root"]=comp["component_root"];pr["component_bad_rows"]=comp["bad_cell_rows"].tolist()
    gain=int(pr["oldbad"])-int(pr["newbad"]);tie=not gain and pr["newdef"]<pr["olddef"]-max(1e-8,1e-6*pr["olddef"])
    entry={"component":ci,"edge_group":[list(e)],"status":"PASS_PROVISIONAL_COUNT_GAIN" if gain>0 else "PASS_PROVISIONAL_DEFICIT_TIE" if tie else "REJECTED_NO_STRICT_PROGRESS",
      "native_bad_before":pr["oldbad"],"native_bad_after":pr["newbad"],"deficit_before":pr["olddef"],"deficit_after":pr["newdef"],"cavity_cells":len(pr["rows"]),"cavity_points":len(pr["verts"])}
    attempts.append(entry)
    if gain>0 or tie:
     single_progress=True;single_candidates.append(((-gain,pr["newbad"],pr["newdef"],ei) if gain>0 else (pr["newdef"]-pr["olddef"],pr["newdef"],ei),pr,len(attempts)-1,gain>0))
   except Exception as exc:
    attempts.append({"component":ci,"edge_group":[list(e)],"status":"REJECTED","error_type":type(exc).__name__,"error":str(exc)})
  candidates=list(single_candidates)
  multi_group_count=0
  if not single_progress and not component_interrupted and len(singles)>=2:
   by_edge={e:c for e,c in singles}
   # Rank connected groups by how many current bad cells they cover, then number of edge stars.
   groups=[]
   for nedge in (2,3,4):
    for group in itertools.combinations([e for e,_ in singles],nedge):
     rows=np.unique(np.concatenate([by_edge[e]["rows"] for e in group]))
     covered=int(np.count_nonzero(np.asarray(cache["bad"],bool)[rows]))
     groups.append(((-covered,-nedge,len(rows),group),group,rows))
   groups.sort(key=lambda x:x[0])
   for _,group,rows in groups:
    if edge_trials>=max_attempts_per_role or time.monotonic()-start>=max_seconds:
     complete=False;component_interrupted=True;break
    edge_trials+=1;multi_group_count+=1
    try:
     cavs=[by_edge[e] for e in group]
     c=cavity_union(cache,group,cavs)
     if reserved_cells&set(map(int,c["rows"])) or reserved_verts&set(map(int,c["verts"])):
      attempts.append({"component":ci,"edge_group":[list(e) for e in group],"status":"REJECTED_OVERLAP_WITH_ACCEPTED_CAVITY"});continue
     solve_calls+=1;remaining=max(.02,max_seconds-(time.monotonic()-start))
     pr=solve_op(cache,group,min(solve_seconds,remaining),cavity=c)
     pr["component_root"]=comp["component_root"];pr["component_bad_rows"]=comp["bad_cell_rows"].tolist()
     gain=int(pr["oldbad"])-int(pr["newbad"]);tie=not gain and pr["newdef"]<pr["olddef"]-max(1e-8,1e-6*pr["olddef"])
     entry={"component":ci,"edge_group":[list(e) for e in group],"status":"PASS_PROVISIONAL_COUNT_GAIN" if gain>0 else "PASS_PROVISIONAL_DEFICIT_TIE" if tie else "REJECTED_NO_STRICT_PROGRESS",
      "native_bad_before":pr["oldbad"],"native_bad_after":pr["newbad"],"deficit_before":pr["olddef"],"deficit_after":pr["newdef"],"cavity_cells":len(pr["rows"]),"cavity_points":len(pr["verts"])}
     attempts.append(entry)
     if gain>0 or tie:
      candidates.append(((-gain,pr["newbad"],pr["newdef"],multi_group_count) if gain>0 else (pr["newdef"]-pr["olddef"],pr["newdef"],multi_group_count),pr,len(attempts)-1,gain>0))
    except Exception as exc:
     attempts.append({"component":ci,"edge_group":[list(e) for e in group],"status":"REJECTED","error_type":type(exc).__name__,"error":str(exc)})
  chosen=None
  counts=[x for x in candidates if x[3]]
  if counts:chosen=min(counts,key=lambda x:x[0])
  elif candidates:chosen=min(candidates,key=lambda x:x[0])
  if chosen is not None:
   _,pr,ai,is_count=chosen;attempts[ai]["status"]="ACCEPTED";attempts[ai]["selection_reason"]="STRICT_BAD_COUNT_GAIN" if is_count else "DEFICIT_TIE_FALLBACK"
   reserved_cells.update(map(int,pr["rows"]));reserved_verts.update(map(int,pr["verts"]));ops.append(pr)
  if component_interrupted:break
 return {"role":role,"input_bad_count":int(np.count_nonzero(cache["bad"])),"component_count":len(components),
  "components":[{"root":c["component_root"],"bad_cells":c["bad_cell_rows"].tolist(),"candidate_edge_count":len(c["edges"])} for c in components],
  "ops":ops,"attempts":attempts,"solve_calls":solve_calls,"milp_invocations":int(_MILP_CALLS-milp_before),"candidate_edge_attempts":edge_trials,
  "candidate_edge_attempt_limit":int(max_attempts_per_role),"elapsed_s":time.monotonic()-start,"search_complete":bool(complete)}
def run_batch(state_path,outdir,roles=("Skin","Fat"),max_seconds=60.,solve_seconds=3.,max_patches=32,fixture=False,max_attempts_per_role=512):
 out=Path(outdir).resolve()
 if out.exists():raise FileExistsError("fresh output directory required")
 out.mkdir(parents=True)
 ctx=P883.load_context(state_path,out,fixture=fixture,source_role="D")
 results={r:role_batch(state_path,ctx,r,max_seconds,solve_seconds,max_patches,max_attempts_per_role) for r in roles}
 accepted={r:x["ops"] for r,x in results.items() if x["ops"]}
 summary={"schema":"h706.fixed-boundary-connected-cavity-batch.884c6.v1","status":"PREPARED_NO_ADMISSIBLE_PATCHES" if not accepted else "IN_MEMORY_BATCH_READY",
  "input_state":ref(state_path),"producer":ref(Path(__file__)),"roles":{r:{k:v for k,v in x.items() if k!="ops"} for r,x in results.items()},
  "accepted_counts":{r:len(ops) for r,ops in accepted.items()},"candidate_edge_attempt_limit_per_role":int(max_attempts_per_role),
  "physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
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
 journal={"schema":"h706.fixed-boundary-connected-cavity-batch.journal.884c6.v1","parent_state":ref(state_path),"roles":list(accepted),
  "volume_candidates":volume_outputs,"operations":{r:[{"edge":list(map(int,x["edges"][0])),"edge_group":[list(map(int,e)) for e in x["edges"]],"cell_rows":x["rows"].tolist(),
  "vertices":x["verts"].tolist(),"seed_cell_rows":x["seed"].tolist(),"ring_added_rows":x["added"].tolist(),
  "old_boundary":x["oldb"].tolist(),"new_tets":x["newt"].tolist(),"material":x["material"],"old_bad":x["oldbad"],
  "new_bad":x["newbad"],"old_deficit":x["olddef"],"new_deficit":x["newdef"],"old_det6":str(x["oldsum"]),
  "new_det6":str(x["newsum"]),"exact_delta_det6":str(x["delta"]),"complete_parent_union":x["rows"].tolist(),
  "solver":x["solver"]} for x in ops] for r,ops in accepted.items()},
  "independent_replay":independent,"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
 jp=out/"operation-journal.json";P883.write_json(jp,journal)
 summary.update({"status":"PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS","volume_candidates":volume_outputs,
  "cow_writes":cow_reports,"candidate_state":ref(sp),"journal":ref(jp),"independent_replay":independent,
  "accepted_quality":{"roles":{r:[{"edge":list(map(int,x["edges"][0])),"edge_group":[list(map(int,e)) for e in x["edges"]],"bad_before":x["oldbad"],"bad_after":x["newbad"],
    "deficit_before":x["olddef"],"deficit_after":x["newdef"],"cavity_cells":len(x["rows"]),"cavity_points":len(x["verts"])} for x in ops] for r,ops in accepted.items()}},
  "full_OID_walls_all185":"PENDING_EXTERNAL","physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"})
 P883.write_json(out/"batch-report.json",summary);return summary
def count_first_connected_group_controls():
 global components_and_edges,solve_op,cavity_union
 old_components=components_and_edges;old_solve=solve_op;old_union=cavity_union;old_cavity=E.cavity
 try:
  components_and_edges=lambda cache:[{"component_root":0,"bad_cell_rows":np.asarray([2,3],np.int64),"edges":[{"edge":[0,1]},{"edge":[1,2]},{"edge":[2,3]}]}]
  def fake_cavity(cache,edge):return {"edge":tuple(edge),"seed":np.asarray([2],np.int64),"rows":np.asarray([2,3],np.int64),"verts":np.asarray([0,1,2,3],np.int64),"material":16,"oldb":np.empty((0,3),np.int64),"lb":np.empty((0,3),np.int64)}
  E.cavity=fake_cavity
  cavity_union=lambda cache,edges,cavities=None:{"edge":tuple(edges[0]),"edges":tuple(edges),"rows":np.asarray([2,3],np.int64),"verts":np.asarray([0,1,2,3],np.int64),"seed":np.asarray([2],np.int64),"added":np.asarray([3],np.int64),"ring":{},"material":16,"oldt":np.empty((2,4),np.int64),"oldmat":np.asarray([16,16]),"oldb":np.empty((0,3),np.int64),"lb":np.empty((0,3),np.int64)}
  calls=[]
  def fake_solve(cache,edges,seconds,cavity=None):
   calls.append(tuple(edges))
   if len(edges)==1:raise ValueError("synthetic_single_edge_no_strict_progress")
   gain=1 if len(edges)==2 else 2
   return {**cavity,"edges":tuple(edges),"newt":np.zeros((1,4),np.int64),"oldbad":2,"newbad":2-gain,"olddef":2.,"newdef":1. if gain==1 else .5}
  solve_op=fake_solve
  r=role_batch(None,{"caches":{"Skin":{"bad":np.asarray([0,0,1,1],bool)}}},"Skin",60.,3.,32,512)
  if not r["ops"] or len(r["ops"][0]["edges"])<2 or r["ops"][0]["newbad"]>=r["ops"][0]["oldbad"]:
   raise AssertionError("multi-edge fallback did not select a strict count gain")
  if not any(len(e)>1 for e in calls):raise AssertionError("no multi-edge solver call recorded")
  return {"status":"PASS","single_edges_rejected_before_group_fallback":True,"multi_edge_count_gain_selected":True,"selected_edge_count":len(r["ops"][0]["edges"])}
 finally:
  components_and_edges=old_components;solve_op=old_solve;cavity_union=old_union;E.cavity=old_cavity

def synthetic_volume_arrays(points,tets,material=16):
 p=np.asarray(points,np.float32);t=M.orient(p,np.asarray(tets,np.int64));b=np.asarray(T.m.boundary(t),np.int64)
 return {"points_world_m":p,"points_world_m_f32":p.copy(),"tetrahedra":t,"material_indices":np.full(len(t),material,np.int32),
  "boundary_faces":b,"desired_boundary_faces":b.copy(),"mapped_current_boundary_faces":b.copy(),"proposed_current_desired_boundary_faces":b.copy(),
  "retained_input_cell_rows":np.arange(len(t),dtype=np.int64),"output_cell_to_input_cell_rows":np.arange(len(t),dtype=np.int64),
  "output_cell_to_input_cell":np.arange(len(t),dtype=np.int64),"input_cell_to_output_cell_rows":np.arange(len(t),dtype=np.int64),
  "input_cell_to_output_cell":np.arange(len(t),dtype=np.int64),"input_node_to_output_node_rows":np.arange(len(p),dtype=np.int64),
  "input_node_to_output_node":np.arange(len(p),dtype=np.int64),"output_node_to_input_node_rows":np.arange(len(p),dtype=np.int64),
  "output_node_to_input_node":np.arange(len(p),dtype=np.int64)}

def synthetic_fixture(root):
 root=Path(root).resolve()
 if root.exists():raise FileExistsError("fixture output already exists")
 root.mkdir(parents=True);inp=root/"inputs";inp.mkdir()
 base=HERE/"run-883e-synthetic-fixture-001"/"inputs";st=json.loads((base/"state.json").read_text())
 p=np.asarray([[0,0,200],[0,0,-200],[1,0,0],[-.5,.8660254,0],[-.5,-.8660254,0]],np.float32)
 raw=np.asarray([[0,1,2,3],[0,1,3,4],[0,1,4,2]],np.int64)
 arr=synthetic_volume_arrays(p,raw)
 _,_,good=P883.native(p,arr["tetrahedra"])
 if np.any(~good):
  pass
 else:raise AssertionError("fixture requires native-bad positive tetrahedra")
 refs={}
 for role in ("Skin","Fat"):
  vp=inp/(role.lower()+".npz");np.savez_compressed(vp,**arr);refs[role]={"path":str(vp.resolve()),"sha256":sha(vp),"cells":len(raw),"nodes":len(p),"native_quality_failures":int(np.count_nonzero(~good))}
 st["Skin_current"]=refs["Skin"];st["Fat_current"]=refs["Fat"];sp=inp/"state.json";P883.write_json(sp,st)
 controls=count_first_connected_group_controls()
 real_solve=solve_op
 forced={"single_edge_calls":0}
 def fixture_solve(cache,edges,seconds,cavity=None):
  if len(edges)==1:
   forced["single_edge_calls"]+=1
   raise ValueError("FIXTURE_CONTROL_SINGLE_EDGE_NO_STRICT_PROGRESS")
  return real_solve(cache,edges,seconds,cavity)
 globals()["solve_op"]=fixture_solve
 try:
  main_batch=root/"main-batch"
  r=run_batch(sp,main_batch,roles=("Skin","Fat"),max_seconds=30,solve_seconds=3,max_patches=32,fixture=True,max_attempts_per_role=512)
 finally:globals()["solve_op"]=real_solve
 if r.get("status")!="PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS":raise AssertionError("multi-edge synthetic batch did not emit/replay")
 for role in ("Skin","Fat"):
  accepted=r["accepted_quality"]["roles"].get(role,[])
  if not accepted or not any(len(x.get("edge_group",[]))>=2 for x in accepted):raise AssertionError(role+" fixture did not accept a multi-edge group")
  if r["independent_replay"][role]["full_bad_after"]>=r["independent_replay"][role]["full_bad_before"]:raise AssertionError(role+" fixture did not improve native bad count")
 # A separate good-Fat control proves an unaffected volume role remains byte-identical.
 no_op=root/"unaffected-role";ni=no_op/"inputs";ni.mkdir(parents=True)
 no_st=json.loads(sp.read_text());no_st["Skin_current"]=dict(refs["Skin"])
 gp=np.asarray([[0,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]],np.float32)
 gt=synthetic_volume_arrays(gp,np.asarray([[0,1,2,3]],np.int64))
 _,_,gg=P883.native(gp,gt["tetrahedra"])
 if not np.all(gg):raise AssertionError("unaffected-role fixture tetra must be native-good")
 fp=ni/"fat-good.npz";np.savez_compressed(fp,**gt);no_st["Fat_current"]={"path":str(fp.resolve()),"sha256":sha(fp),"cells":1,"nodes":4,"native_quality_failures":0}
 nsp=ni/"state.json";P883.write_json(nsp,no_st);no_batch=no_op/"batch"
 globals()["solve_op"]=fixture_solve
 try:nr=run_batch(nsp,no_batch,roles=("Skin","Fat"),max_seconds=30,solve_seconds=3,max_patches=32,fixture=True,max_attempts_per_role=512)
 finally:globals()["solve_op"]=real_solve
 if nr.get("status")!="PASS_BATCH_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDITS" or nr["accepted_counts"].get("Skin",0)<1 or nr["accepted_counts"].get("Fat",0)!=0:raise AssertionError("unaffected-role fixture did not produce Skin-only batch")
 receipt={"schema":"h706.fixed-boundary-connected-cavity-fixture.884c6.v1","status":"PASS","producer":ref(Path(__file__)),
  "search_controls":controls,"single_edge_solver_rejections_forced_only_to_exercise_fallback":True,"forced_single_calls":forced["single_edge_calls"],
  "main_batch":ref(main_batch/"batch-report.json"),"main_candidate_state":r["candidate_state"],"main_journal":r["journal"],
  "main_multi_edge_ops":{role:[x.get("edge_group") for x in r["accepted_quality"]["roles"][role]] for role in ("Skin","Fat")},
  "unaffected_role_batch":ref(no_batch/"batch-report.json"),"unaffected_role_counts":nr["accepted_counts"],
  "unaffected_Fat_native_bad":0,"unaffected_Fat_volume_written":False,"all185":"NOT_RUN","mass_mapping":"NOT_RUN","physical_admission":False,"native_steps":0}
 P883.write_json(root/"fixture-receipt.json",receipt)
 return {"status":"PASS","receipt":str(root/"fixture-receipt.json"),"forced_single_edge_calls":forced["single_edge_calls"],"multi_edge_ops":receipt["main_multi_edge_ops"],"unaffected_role_counts":nr["accepted_counts"]}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--state",type=Path);ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--roles",default="Skin,Fat");ap.add_argument("--seconds-per-role",type=float,default=60.)
 ap.add_argument("--solve-seconds",type=float,default=3.);ap.add_argument("--max-patches",type=int,default=32)
 ap.add_argument("--max-attempts-per-role",type=int,default=512);ap.add_argument("--fixture",action="store_true")
 a=ap.parse_args()
 if a.fixture:r=synthetic_fixture(a.output)
 else:
  if a.state is None:ap.error("--state required unless --fixture")
  roles=tuple(x.strip() for x in a.roles.split(",") if x.strip())
  if not roles or any(x not in ("Skin","Fat") for x in roles):ap.error("--roles must be Skin,Fat and/or Fat/Skin")
  r=run_batch(a.state,a.output,roles,a.seconds_per_role,a.solve_seconds,a.max_patches,False,a.max_attempts_per_role)
 print(json.dumps(r,indent=2));return r
if __name__=="__main__":main()

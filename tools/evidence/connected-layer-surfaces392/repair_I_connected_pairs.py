#!/usr/bin/env python3
from pathlib import Path
import collections,hashlib,importlib.util,json,sys,time
import numpy as np

PBASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
B=PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367"
PARENT=B/"condition-287-inherited-pair-repair-001"
INPUT_F64=PARENT/"result/candidate-f64.npz"
INPUT_F32=PARENT/"result/candidate-f32-typed.npz"
SELF_REPORT=PARENT/"result/local-pair-query-report.json"
PAIR_ARCHIVE=PARENT/"result/new-face-exact-self-witnesses.npz"
INVENTORY=PARENT/"pair-cluster-inventory-002/inventory.json"
RUN=B/"condition-288-pair-connected-remesh-001-run-002"
OUT=RUN/"result"
REMESH=PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-043-abstract-cycle-coupled-remesh-fixed/remesh.py"
HELPER=PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/remesh_d_witness_stars_rerun022.py"
FLIP=PBASE/"geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/authoring-source-chain/local_float32_edge_flip.py"
PAIR_HELPER=PBASE/"exact_local_pair_witnesses_381.py"
EMITTER=PBASE.parent/"emit_f32.py"
PRODUCER=PBASE/"derived-lineage-producer-260/export_authoring_registered_surface_sidecars_230.py"
DERIVED_ID=120000204
MAX_EXPANSIONS=2

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,str(path))
 if spec is None or spec.loader is None:raise ImportError(f"cannot load {path}")
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def face_normals(P,F):
 T=np.asarray(P,np.float64)[np.asarray(F,np.int64)]
 return np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0])
def edge_map(F):
 em=collections.defaultdict(set)
 for r,(a,b,c) in enumerate(np.asarray(F,np.int64)):
  for u,v in ((a,b),(b,c),(c,a)):em[tuple(sorted((int(u),int(v))))].add(int(r))
 return em
def face_components(rows,em):
 todo=set(map(int,rows));out=[]
 while todo:
  st=min(todo);stack=[st];c=set()
  while stack:
   r=stack.pop()
   if r in c:continue
   c.add(r);todo.discard(r)
   a,b,c0 = None,None,None
   for e,rs in em.items():
    if r in rs:stack.extend(rs-c)
  out.append(c)
 return out
def components_fast(rows,em):
 todo=set(map(int,rows));out=[]
 adj=collections.defaultdict(set)
 for rs in em.values():
  local=rs & todo
  if len(local)>1:
   for r in local:adj[r].update(local-{r})
 while todo:
  st=min(todo);stack=[st];c=set()
  while stack:
   r=stack.pop()
   if r in c:continue
   c.add(r);todo.discard(r);stack.extend(adj[r]-c)
  out.append(c)
 return out
def boundary_check(old_directed,Fnew):
 ed=collections.defaultdict(list)
 for a,b,c in np.asarray(Fnew,np.int64):
  for u,v in ((a,b),(b,c),(c,a)):ed[tuple(sorted((int(u),int(v))))].append((int(u),int(v)))
 if any(len(v)>2 for v in ed.values()):return False,"internal edge incidence >2"
 for v in ed.values():
  if len(v)==2 and v[0]!=(v[1][1],v[1][0]):return False,"internal edge winding inconsistent"
 old=collections.Counter(tuple(map(int,x)) for x in old_directed)
 new=collections.Counter(v[0] for v in ed.values() if len(v)==1)
 return (old==new,None if old==new else "directed boundary changed")
def edge_set(F):
 out=set()
 for a,b,c in np.asarray(F,np.int64):
  out.update((tuple(sorted((int(a),int(b)))),tuple(sorted((int(b),int(c)))),tuple(sorted((int(c),int(a))))))
 return out
def boundary_stats(F):
 E=np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0),axis=1)
 U,C=np.unique(E,axis=0,return_counts=True)
 hist={str(int(k)):int(v) for k,v in collections.Counter(C.tolist()).items()}
 bd=U[C==1];g=collections.defaultdict(list)
 for a,b in bd:g[int(a)].append(int(b));g[int(b)].append(int(a))
 if any(len(v)!=2 for v in g.values()):return {"histogram":hist,"boundary_edges":len(bd),"loop_lengths":[],"bad_boundary_degree":True}
 seen=set();loops=[]
 for st in sorted(g):
  if st in seen:continue
  prev=-1;cur=st;n=0
  while True:
   seen.add(cur);nxt=g[cur][0] if g[cur][0]!=prev else g[cur][1];prev,cur=cur,nxt;n+=1
   if cur==st:break
   if cur in seen:return {"histogram":hist,"boundary_edges":len(bd),"loop_lengths":[],"bad_boundary_degree":False}
  loops.append(n)
 return {"histogram":hist,"boundary_edges":int(len(bd)),"loop_lengths":sorted(loops),"bad_boundary_degree":False}
def ancestry_rows(d):
 off=np.asarray(d["face_ancestry_offsets"],np.int64)
 flat=np.asarray(d["face_ancestry_input_cache_rows"],np.int64)
 return [set(map(int,flat[off[i]:off[i+1]])) for i in range(len(off)-1)]
def expand_rows(rows,F,em):
 out=set(rows);ring=set()
 for r in rows:
  a,b,c=map(int,F[int(r)])
  for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):
   ring.update(em[e])
 ring-=out
 return out|ring,ring
def main():
 t0=time.monotonic()
 required=[INPUT_F64,INPUT_F32,SELF_REPORT,PAIR_ARCHIVE,INVENTORY,REMESH,HELPER,FLIP,PAIR_HELPER,EMITTER,PRODUCER]
 missing=[str(p) for p in required if not p.is_file()]
 if missing:raise FileNotFoundError("missing required inputs: "+", ".join(missing))
 if RUN.exists():raise FileExistsError(RUN)
 RUN.mkdir(parents=True);shutil_path=RUN/"authoring_source.py"
 import shutil;shutil.copy2(__file__,shutil_path)
 OUT.mkdir()
 inputsha=sha(INPUT_F64);typedsha=sha(INPUT_F32);pairsha=sha(PAIR_ARCHIVE);inventorysha=sha(INVENTORY)
 self=json.loads(SELF_REPORT.read_text());inventory=json.loads(INVENTORY.read_text())
 if self.get("status")!="FAIL_EXACT_SELF_PAIRS" or int(self.get("same_surface_exact_triangle_pair_count",-1))!=8:raise ValueError("pinned complete exact self report is not the expected 51-pair failure")
 with np.load(INPUT_F64,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(INPUT_F32,allow_pickle=False) as z:t={k:z[k] for k in z.files}
 with np.load(PAIR_ARCHIVE,allow_pickle=False) as z:pairs=np.asarray(z["triangle_pairs"],np.int64)
 P=np.asarray(d["points_world_m_f64"],np.float64);F=np.asarray(d["faces_f64"],np.int64)
 cap=np.asarray(d["temporary_cap_descendant_face_mask"],bool);open_rows=np.flatnonzero(~cap);Fopen=F[open_rows]
 Y=np.asarray(t["points_world_m"],np.float32);Ft=np.asarray(t["faces"],np.int64)
 if sha(INPUT_F32)!=self.get("candidate_sha256") or len(Ft)!=len(open_rows) or len(pairs)!=8:raise ValueError("exact-self, emitted candidate, and current F64 face domains disagree")
 # Verify emitted open face row order and Float32 point remap exactly.
 with np.load(PARENT/"result/emitted-vertex-maps.npz",allow_pickle=False) as z:emap={k:z[k] for k in z.files}
 vmap=np.asarray(emap["input_vertex_old_to_emitted"],np.int64)
 if not np.array_equal(Ft,vmap[Fopen]):raise ValueError("typed open face order is not the pinned F64 open-face row order")
 if not np.array_equal(Y[vmap],P.astype(np.float32)):raise ValueError("typed points differ from the pinned F64 Float32 cast")
 if pairs.size and (pairs.min()<0 or pairs.max()>=len(Fopen)):raise ValueError("exact pair row index out of open face range")
 seeds=set(map(int,pairs.ravel()))
 if int(inventory.get("self_pair_count",-1))!=len(pairs):raise ValueError("pair inventory/archive count mismatch")
 # All exact pairs must be touched by the selected source faces; this is the
 # proof that a full base self query can be reduced to exact changed-face tests.
 if any(int(a) not in seeds or int(b) not in seeds for a,b in pairs):raise ValueError("an exact pair is not covered by the selected patch set")
 cap_vertices=set(map(int,np.unique(F[np.flatnonzero(cap)])))
 if any(cap_vertices.intersection(map(int,Fopen[r])) for r in seeds):raise ValueError("exact-pair seed patch touches a protected head cap vertex")
 # Read the exact-derived source fields and reserve an unused operation ID.
 schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
 if str(DERIVED_ID) in schema or DERIVED_ID in set(map(int,np.unique(d["face_run_original_id"]))):raise ValueError(f"derived source ID collision {DERIVED_ID}")
 rowcomponents=[set(map(int,x["seed_faces"])) for x in inventory["topological_seed_components"]]
 if set().union(*rowcomponents)!=seeds:raise ValueError("inventory pair-connected groups do not cover exact pair faces")
 em=edge_map(Fopen)
 # Merge witness-connected disks, then add one intact edge ring so interacting
 # components are solved as connected boundary patches.
 pair_adj={int(r):set() for r in seeds}
 for a,b in pairs:
  pair_adj[int(a)].add(int(b));pair_adj[int(b)].add(int(a))
 left=set(seeds);pairgroups=[]
 while left:
  st=min(left);left.remove(st);stack=[st];group={st}
  while stack:
   r=stack.pop()
   for q in pair_adj[r]:
    if q in left:left.remove(q);group.add(q);stack.append(q)
  pairgroups.append(group)
 if {tuple(sorted(x)) for x in pairgroups}!={tuple(sorted(x)) for x in rowcomponents}:raise ValueError("saved pair-connected groups disagree with exact pair archive")
 helpers=loadmod("d043_I_pair_remesh_helpers_367",HELPER)
 remesh=loadmod("d043_I_pair_remesh_367",REMESH)
 pred=remesh.load_exact_owner()
 P32=P.astype(np.float32);lattice=tuple(pred.float32_point_lattice_key(p) for p in P32)
 seed_specs=[]
 for g in pairgroups:
  grown,_=expand_rows(g,Fopen,em)
  seed_specs.append({"rows":grown,"seed_rows":sorted(g),"expanded_rings":1})
 selected=set().union(*(q["rows"] for q in seed_specs))
 merged=components_fast(selected,em)
 seed_specs=[{"rows":set(c),"seed_rows":sorted(set(c)&seeds),"expanded_rings":1} for c in merged]
 seed_specs.sort(key=lambda q:min(q["rows"]))
 attempts=[];patch_product=None;selected=None
 for expansion_round in range(MAX_EXPANSIONS+1):
  selected=set().union(*(q["rows"] for q in seed_specs))
  if any(cap_vertices.intersection(map(int,Fopen[r])) for r in selected):raise ValueError("expanded patch touches a protected head cap vertex")
  patches=components_fast(selected,em);patches.sort(key=min)
  base_mask=np.ones(len(Fopen),bool);base_mask[list(selected)]=False;baseF=Fopen[base_mask]
  base_records,_=pred.float32_surface_records(P32,baseF,source_seam_groups=())
  base_index=pred._prepare_surface_aabb(base_records);external_edges=edge_set(baseF)
  prior=[];nextid=len(baseF)+1;all_new=[];perpatch=[];failed=None
  for pi,rows in enumerate(patches):
   rows=sorted(map(int,rows))
   loops,directed=helpers.boundary_cycles(Fopen,rows)
   if len(loops)!=1:
    failed={"patch_index":pi,"rows":rows,"error":f"patch has {len(loops)} boundary loops"}
    break
   loop=list(map(int,loops[0]))
   try:
    tri,newrecs,plan=remesh.collision_aware_ear_clip(loop,P,P32,external_edges,base_index,pred,lattice,prior,nextid,max_nodes=25000,max_seconds=3.0)
   except Exception as exc:
    failed={"patch_index":pi,"rows":rows,"loop":loop,"error_type":type(exc).__name__,"error":str(exc)}
    break
   newF=np.asarray(tri,np.int64)
   ok,why=boundary_check(directed,newF)
   if not ok:
    failed={"patch_index":pi,"rows":rows,"loop":loop,"error":why}
    break
   n64=face_normals(P,newF);p32=P32.astype(np.float64);n32=face_normals(p32,newF)
   if np.any(np.linalg.norm(n64,axis=1)==0) or np.any(np.linalg.norm(n32,axis=1)==0) or np.any(np.einsum("ij,ij->i",n64,n32)<=0):
    failed={"patch_index":pi,"rows":rows,"loop":loop,"error":"replacement face F64/F32 area/orientation gate failed"}
    break
   all_new.extend(newF.tolist())
   prior.extend((rec,tuple(map(int,face))) for rec,face in zip(newrecs,newF))
   external_edges.update(edge_set(newF))
   nextid+=len(newF)+100
   perpatch.append({"patch_index":pi,"rows":rows,"seed_rows":sorted(set(rows)&seeds),"loop_vertices":loop,"boundary_edges":len(loop),"new_faces":newF.tolist(),"new_face_count":len(newF),"triangulation":plan,"directed_boundary_preserved":True})
   cp=OUT/f"checkpoint-round-{expansion_round:02d}-patch-{pi:02d}.npz"
   np.savez_compressed(cp,points_world_m_f64=P,patch_input_face_rows=np.asarray(rows,np.int64),replacement_faces_f64=newF,boundary_loop=np.asarray(loop,np.int64))
  attempt={"round":expansion_round,"selected_face_count":len(selected),"component_count":len(patches),"successful_patch_count":len(perpatch),"successful_patches":perpatch,"failure":failed}
  attempts.append(attempt)
  if failed is None:
   patch_product={"selected_rows":selected,"patches":perpatch,"new_faces":np.asarray(all_new,np.int64),"round":expansion_round}
   break
  if expansion_round==MAX_EXPANSIONS:
   (OUT/"failure.json").write_text(json.dumps({"status":"BOUNDED_PATCH_SEARCH_FAILED","input_f64_sha256":inputsha,"exact_pair_archive_sha256":pairsha,"attempts":attempts,"physical_admission":False},indent=2)+"\n")
   raise RuntimeError("connected patch search exhausted two local expansions; see failure.json")
  failed_rows=set(map(int,failed["rows"]))
  grown,_=expand_rows(failed_rows,Fopen,em)
  seed_specs.append({"rows":grown,"seed_rows":sorted(failed_rows&seeds),"expanded_rings":expansion_round+1})
  # Recompute connected patch rows from union so overlapping/adjacent regions
  # merge into one patch, while unrelated components remain separate.
  selected=set().union(*(q["rows"] for q in seed_specs))
  merged=components_fast(selected,em)
  seed_specs=[{"rows":set(c),"seed_rows":sorted(set(c)&seeds),"expanded_rings":expansion_round+1} for c in merged]
  seed_specs.sort(key=lambda q:min(q["rows"]))
 if patch_product is None:raise RuntimeError("no successful connected patch attempt")

 selected_open=patch_product["selected_rows"]
 selected_closed=set(int(open_rows[r]) for r in selected_open)
 keep_closed=np.ones(len(F),bool);keep_closed[list(selected_closed)]=False;keep_closed_rows=np.flatnonzero(keep_closed)
 Fpre=np.vstack((F[keep_closed],patch_product["new_faces"]))
 cap_pre=np.concatenate((cap[keep_closed],np.zeros(len(patch_product["new_faces"]),bool)))
 # Compose typed face ancestry as the complete union of each replaced patch's
 # source cache rows. New patch faces receive a dedicated derived I origin.
 old_anc=ancestry_rows(d)
 ancmeta={}
 for row,oid,fid,i0,i1 in zip(np.asarray(d["face_ancestry_input_cache_rows"],np.int64),np.asarray(d["face_ancestry_original_ids"],np.int64),np.asarray(d["face_ancestry_face_ids"],np.int64),np.asarray(d["face_ancestry_identity0"],np.int64),np.asarray(d["face_ancestry_identity1"],np.int64)):
  key=int(row);v=(int(oid),int(fid),int(i0),int(i1))
  if key in ancmeta and ancmeta[key]!=v:raise ValueError(f"conflicting ancestry metadata for cache row {key}")
  ancmeta[key]=v
 patch_anc=[]
 patch_parent_ids=set()
 for patch in patch_product["patches"]:
  closed_rows=[int(open_rows[r]) for r in patch["rows"]]
  union=set().union(*(old_anc[r] for r in closed_rows))
  if not union:raise ValueError(f"patch {patch['patch_index']} has empty parent ancestry")
  patch_parent_ids.update(ancmeta[r][0] for r in union)
  patch_anc.extend([set(union) for _ in patch["new_faces"]])
 if len(patch_anc)!=len(patch_product["new_faces"]):raise ValueError("replacement face ancestry count mismatch")
 outattrs={}
 for key in ("face_cache_row","face_source_layer","face_source_body","face_source_bone","face_run_original_id","face_id","face_source_identity0","face_source_identity1"):
  arr=np.asarray(d[key])
  if key=="face_cache_row":fill=np.full(len(patch_anc),-1,dtype=arr.dtype)
  elif key=="face_source_layer":fill=np.full(len(patch_anc),1,dtype=arr.dtype)
  elif key in ("face_source_body","face_source_bone","face_source_identity1"):fill=np.full(len(patch_anc),-1,dtype=arr.dtype)
  elif key=="face_run_original_id":fill=np.full(len(patch_anc),DERIVED_ID,dtype=arr.dtype)
  else:fill=np.arange(len(patch_anc),dtype=arr.dtype)
  outattrs[key]=np.concatenate((arr[keep_closed],fill))
 # Compact unused points with a reversible row map.
 used=np.unique(Fpre);old_to_compact=np.full(len(P),-1,np.int64);old_to_compact[used]=np.arange(len(used),dtype=np.int64)
 Pout=P[used];Fout=old_to_compact[Fpre]
 if len(Pout)!=len(used) or np.any(Fout<0):raise ValueError("point compaction map invalid")
 if not np.array_equal(Pout[Fout[:len(keep_closed_rows)]],P[F[keep_closed_rows]]):raise ValueError("untouched/cap coordinates or face winding changed")
 # CSR ancestry over retained input faces and new patch faces.
 ancsets=[old_anc[int(r)] for r in keep_closed_rows]+patch_anc
 offsets=[0];flat=[]
 for s in ancsets:
  flat.extend(sorted(s));offsets.append(len(flat))
 offsets=np.asarray(offsets,np.int64);flat=np.asarray(flat,np.int64)
 if any(int(r) not in ancmeta for r in flat):raise ValueError("composed source ancestry has missing metadata")
 am=np.asarray([ancmeta[int(r)] for r in flat],np.int64).reshape((-1,4)) if len(flat) else np.zeros((0,4),np.int64)
 # Build a typed source view for all genuinely derived replacement faces.
 new_closed_rows=np.arange(len(keep_closed_rows),len(Fout),dtype=np.int64)
 dpoints=np.unique(Fout[new_closed_rows]);dremap=np.full(len(Pout),-1,np.int64);dremap[dpoints]=np.arange(len(dpoints))
 sourceview=OUT/"I-inherited-pair-repair-derived-source.npz"
 derived_ids=np.arange(len(new_closed_rows),dtype=np.int64)
 np.savez_compressed(sourceview,points_world_m_f64=Pout[dpoints],faces_f64=dremap[Fout[new_closed_rows]],face_id=derived_ids,face_source_identity0=derived_ids,face_source_identity1=np.full(len(derived_ids),-1,np.int64))
 sourceviewsha=sha(sourceview)
 schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
 schema[str(DERIVED_ID)]={"role":"I","path":str(sourceview),"sha256":sourceviewsha,"identity_keys":["face_source_identity0","face_source_identity1"],"identity_face_id_key":"face_id","face_count":int(len(new_closed_rows)),"derived_operation":"connected_exact_inherited_self_pair_patch_retriangulation","derived_parent_assets":[{"path":str(INPUT_F64),"sha256":inputsha},{"path":str(INPUT_F32),"sha256":typedsha}],"derived_parent_source_ids":sorted(map(int,patch_parent_ids)),"source_geometry_scope":"complete_reconstructed_patch_faces"}
 schema_json=np.asarray([json.dumps(schema,sort_keys=True)])
 outattrs["face_id"][new_closed_rows]=derived_ids;outattrs["face_source_identity0"][new_closed_rows]=derived_ids;outattrs["face_source_identity1"][new_closed_rows]=-1
 modified=np.concatenate((np.asarray(d["conditioning_modified_face_mask"],bool)[keep_closed],np.ones(len(new_closed_rows),bool)))
 packed={k:np.asarray(v).copy() for k,v in d.items()}
 packed.update({"points_world_m_f64":Pout,"faces_f64":Fout,**outattrs,"temporary_cap_descendant_face_mask":cap_pre,"conditioning_modified_face_mask":modified,"face_ancestry_offsets":offsets,"face_ancestry_input_cache_rows":flat,"face_ancestry_original_ids":am[:,0],"face_ancestry_face_ids":am[:,1],"face_ancestry_identity0":am[:,2],"face_ancestry_identity1":am[:,3],"face_source_identity_schema_by_original_id_json":schema_json,"exact_pair_patch_repair_json":np.asarray([json.dumps({"input_pairs":pairs.tolist(),"input_pair_archive_sha256":pairsha,"seed_face_rows_open_domain":sorted(seeds),"selected_face_rows_open_domain":sorted(selected_open),"patches":patch_product["patches"],"expansion_round":patch_product["round"]},sort_keys=True)])})
 f64out=OUT/"candidate-f64.npz";np.savez_compressed(f64out,**packed)
 np.savez_compressed(OUT/"vertex-row-map.npz",input_vertex_to_output=old_to_compact,output_source_vertex_rows=used,kept_input_face_rows=keep_closed_rows,selected_input_closed_face_rows=np.asarray(sorted(selected_closed),np.int64),replacement_output_closed_face_rows=new_closed_rows)
 # Closed edge incidence and Float32 orientation before emission.
 eclosed=np.sort(np.concatenate((Fout[:,[0,1]],Fout[:,[1,2]],Fout[:,[2,0]]),axis=0),axis=1);eu,ec=np.unique(eclosed,axis=0,return_counts=True)
 if np.any(ec!=2):raise ValueError(f"closed edge incidence failed: {dict(collections.Counter(ec.tolist()))}")
 n64=face_normals(Pout,Fout);p32=Pout.astype(np.float32).astype(np.float64);n32=face_normals(p32,Fout)
 closed_bad=np.flatnonzero(np.einsum("ij,ij->i",n64,n32)<=0)
 if len(closed_bad):raise ValueError(f"closed F64/F32 orientation failures remain: {closed_bad[:20].tolist()}")
 open_closed_rows=np.flatnonzero(~cap_pre);Fopenout=Fout[open_closed_rows]
 emitter=loadmod("emit_f32_I_pair_repair_367",EMITTER)
 Y,Femit,emreport=emitter.emit(Pout,Fopenout,np.asarray(d["original_source_points_f32"],np.float32))
 if emreport.get("nudged_vertices"):raise ValueError(f"emitter moved points: {emreport['nudged_vertices'][:10]}")
 if len(Femit)!=len(Fopenout):raise ValueError("emitter changed face count")
 if emreport.get("nonmanifold_edges")!=0 or emreport.get("open_edges")!=45:raise ValueError("emitted open I boundary incidence failed")
 if not np.array_equal(Y[np.asarray(emreport["input_vertex_old_to_emitted"],np.int64)],Pout.astype(np.float32)):raise ValueError("emitted vertices differ from exact Float32 cast")
 emnorm=face_normals(Y,Femit)
 if np.any(np.linalg.norm(emnorm,axis=1)==0):raise ValueError("emitted open I contains zero-area face")
 openref=face_normals(Pout,Fopenout)
 if np.any(np.einsum("ij,ij->i",openref,emnorm)<=0):raise ValueError("emitted open I has face orientation reversals")
 loopstats=boundary_stats(Femit)
 if loopstats["boundary_edges"]!=45 or loopstats["loop_lengths"]!=[21,24] or loopstats["bad_boundary_degree"]:raise ValueError(f"emitted I head boundary changed: {loopstats}")
 if len(np.unique(Y,axis=0))!=len(Y) or len(np.unique(np.sort(Femit,axis=1),axis=0))!=len(Femit):raise ValueError("emitted open I has duplicate coordinates or faces")
 # Save typed output before the changed-face exact query for recoverability.
 open_attrs={k:outattrs[k][open_closed_rows] for k in outattrs}
 typed={"points_world_m":Y,"faces":Femit,"face_source_layer":open_attrs["face_source_layer"],"face_source_body":open_attrs["face_source_body"],"face_source_bone":open_attrs["face_source_bone"],"face_source_face_id":open_attrs["face_id"],"face_source_original_id":open_attrs["face_run_original_id"],"face_source_identity0":open_attrs["face_source_identity0"],"face_source_identity1":open_attrs["face_source_identity1"],"face_source_identity_schema_by_original_id_json":schema_json,"temporary_cap_faces_removed":np.asarray([int(cap_pre.sum())],np.int64),"float32_emission_json":np.asarray([json.dumps({k:v for k,v in emreport.items() if k!="input_vertex_old_to_emitted"})])}
 typedpath=OUT/"candidate-f32-typed.npz";np.savez_compressed(typedpath,**typed)
 open_face_map=np.full(len(Fout),-1,np.int64);open_face_map[open_closed_rows]=np.arange(len(open_closed_rows),dtype=np.int64)
 new_open_rows=open_face_map[new_closed_rows]
 flip=loadmod("exact_float32_I_pair_repair_367",FLIP)
 localowner=flip.Flips(Y.astype(np.float32).astype(np.float64),Femit)
 localcount=localowner.pairs(set(map(int,new_open_rows)))
 if localcount is None or int(localcount)!=0:
  adapter=loadmod("exact_pair_witness_adapter_381_I_pair_repair_367",PAIR_HELPER)
  pairs_new=adapter.pair_witnesses(localowner,set(map(int,new_open_rows)))
  np.savez_compressed(OUT/"new-face-exact-self-witnesses.npz",triangle_pairs=np.asarray(pairs_new if pairs_new is not None else [],np.int64).reshape(-1,2))
  raise ValueError(f"new patch face exact local self pairs: {localcount}")
 # Producer imports a sibling helper; expose only its existing owner directory.
 sys.path.insert(0,str(PRODUCER.parent))
 producer=loadmod("producer260_I_pair_repair_367",PRODUCER)
 producer_report=producer.validate_candidate(typedpath,"I",require_f32=True)
 # Save the row-map and emitter ancestry as compact maps.
 np.savez_compressed(OUT/"emitted-vertex-maps.npz",input_vertex_old_to_emitted=np.asarray(emreport["input_vertex_old_to_emitted"],np.int64),emitted_vertex_input_parent_offsets=np.asarray(emreport["emitted_vertex_input_parent_offsets"],np.int64),emitted_vertex_input_parent_rows=np.asarray(emreport["emitted_vertex_input_parent_rows"],np.int64),emitted_vertex_source_point_offsets=np.asarray(emreport["emitted_vertex_source_point_offsets"],np.int64),emitted_vertex_source_point_rows=np.asarray(emreport["emitted_vertex_source_point_rows"],np.int64),input_open_closed_face_rows=open_closed_rows)
 receipt={"schema":"I-inherited-exact-pair-local-remesh.v1","status":"LOCAL_BATCH_PATCH_AND_TYPED_PROVENANCE_PASS_FULL_SELF_AUDIT_PENDING","input_f64":{"path":str(INPUT_F64),"sha256":inputsha},"input_typed":{"path":str(INPUT_F32),"sha256":typedsha},"input_local_exact_pair_report":str(SELF_REPORT),"input_local_query_scope":"all changed faces versus current full surface; prior 51 pairs removed by batch 287; no full audit claimed","input_exact_pair_archive":{"path":str(PAIR_ARCHIVE),"sha256":pairsha,"pair_count":int(len(pairs))},"input_region_inventory":{"path":str(INVENTORY),"sha256":inventorysha,"pair_clusters":int(inventory["spatial_pair_cluster_count"]),"topological_components":len(rowcomponents)},"authoring_source":{"path":str(shutil_path),"sha256":sha(shutil_path)},"derived_source_view":{"path":str(sourceview),"sha256":sourceviewsha,"operation_id":DERIVED_ID,"face_count":int(len(new_closed_rows)),"parent_source_ids":sorted(map(int,patch_parent_ids))},"candidate_f64":{"path":str(f64out),"sha256":sha(f64out),"vertices":int(len(Pout)),"faces":int(len(Fout)),"cap_faces":int(cap_pre.sum()),"closed_edge_incidence_histogram":{str(int(k)):int(v) for k,v in collections.Counter(ec.tolist()).items()},"closed_f64_to_f32_orientation_failures":0},"candidate_f32_typed":{"path":str(typedpath),"sha256":sha(typedpath),"vertices":int(len(Y)),"faces":int(len(Femit)),"boundary_edges":loopstats["boundary_edges"],"boundary_loop_lengths":loopstats["loop_lengths"],"zero_area_faces":0,"duplicate_coordinate_rows":0,"local_new_face_exact_pairs_vs_whole_surface":int(localcount)},"producer260_validation":{"status":"PASS","sha256":producer_report["sha256"],"faceCount":producer_report["faceCount"],"sourceParentLineagePreservedPerFace":producer_report["sourceParentLineagePreservedPerFace"]},"patch_attempts":attempts,"successful_patches":patch_product["patches"],"selected_input_open_face_rows":sorted(selected_open),"removed_input_open_face_count":int(len(selected_open)),"replacement_face_count":int(len(new_closed_rows)),"source_points_and_cap_faces_unchanged":True,"elapsed_s":time.monotonic()-t0,"physical_admission":False,"limitations":["Full exact self-intersection, all-185 current-frame bone, O/I/D containment and thickness, volume meshing, mass mapping, native dynamics, and physical admission remain outstanding."]}
 (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 print(json.dumps({"status":receipt["status"],"pairs_in":len(pairs),"components":len(patch_product["patches"]),"selected_faces":len(selected_open),"replacement_faces":len(new_closed_rows),"output_f32_faces":len(Femit),"boundary_loops":loopstats["loop_lengths"],"sha256":sha(typedpath),"candidate":str(typedpath),"elapsed_s":receipt["elapsed_s"],"receipt":str(OUT/"receipt.json")},indent=2),flush=True)
if __name__=="__main__":main()

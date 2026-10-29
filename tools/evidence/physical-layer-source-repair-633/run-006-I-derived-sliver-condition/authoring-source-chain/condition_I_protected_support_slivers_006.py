# Adapted from condition_I_f32_seams_271.py; source SHA256: 32d37d48b3564c320f736cbda0663c88f16b003a82a8cec4abfc0b7c536bf349
# Scope: saved run-005 derived support sliver conditioning. No CSG/Minkowski is run.
from pathlib import Path
import sys,json,time,hashlib,heapq,importlib.util,collections
import numpy as np
PBASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=PBASE/"reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm"
B=A/"run-005-protected-cap-minthickness/output";R=PBASE.parent
INPUT=B/"I-pre-emission-f64.npz";OUT=A/"run-006-I-derived-sliver-condition/output";OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(INPUT,allow_pickle=False);d={k:z[k] for k in z.files}
nface=len(d["faces_f64"])
# The 005 output is a fresh geometry owner checkpoint, so make its source-row ancestry explicit one-to-one before conditioning.
d["face_cache_row"]=np.arange(nface,dtype=np.int64)
d["face_ancestry_offsets"]=np.arange(nface+1,dtype=np.int64)
d["face_ancestry_input_cache_rows"]=np.arange(nface,dtype=np.int64)
d["face_ancestry_original_ids"]=np.asarray(d["face_run_original_id"],np.int64).copy()
d["face_ancestry_face_ids"]=np.asarray(d["face_id"],np.int64).copy()
d["face_ancestry_identity0"]=np.asarray(d["face_source_identity0"],np.int64).copy()
d["face_ancestry_identity1"]=np.asarray(d["face_source_identity1"],np.int64).copy()
d["source_points_f32"]=np.asarray(d["original_source_points_f32"],np.float32)
d["source_cache_sha256"]=np.asarray([sha(INPUT)])
P=np.asarray(d["points_world_m_f64"],np.float64);F=np.asarray(d["faces_f64"],np.int64).copy()
schema0=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
rids=np.asarray(d["face_run_original_id"],np.int64)
layer=np.empty(len(F),np.int8);body=np.full(len(F),-1,np.int32);bone=np.full(len(F),-1,np.int32)
for oid in np.unique(rids):
 spec=schema0.get(str(int(oid)))
 if spec is None: raise ValueError(f"missing source schema for original ID {int(oid)}")
 rr=rids==oid;role=spec.get("role")
 if role=="I": layer[rr]=1
 elif role=="BONE":
  layer[rr]=3;body[rr]=int(spec["body"]);bone[rr]=int(spec["bone"])
 else: raise ValueError(f"unexpected I-cache source role {role} for ID {int(oid)}")
d["face_source_layer"]=layer;d["face_source_body"]=body;d["face_source_bone"]=bone
src=d["original_source_points_f32"];cap=np.asarray(d["temporary_cap_descendant_face_mask"],bool)
t0=time.monotonic()
input_edges=np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1)
input_unique_edges,input_edge_counts=np.unique(input_edges,axis=0,return_counts=True)
input_used_vertices=np.unique(F)
input_euler=int(len(input_used_vertices)-len(input_unique_edges)+len(F))
input_closed=bool(np.all(input_edge_counts==2))
if not input_closed:raise ValueError(f"saved 005 I input is not a closed indexed manifold: {dict(zip(*np.unique(input_edge_counts,return_counts=True)))}")
Q=P.astype(np.float32).astype(np.float64);alive=np.ones(len(F),bool);changed=np.zeros(len(F),bool);anc=[{i} for i in range(len(F))]
source_set={tuple(q) for q in src.astype(np.float64)};pinned=np.array([tuple(q) in source_set for q in P]);pinned[np.unique(F[cap])]=True
inc=[set() for _ in range(len(P))]
for fi,tr in enumerate(F):
 for v in tr:inc[int(v)].add(fi)
def zero(rows):
 rows=np.asarray(rows,np.int64);tri=Q[F[rows]]
 return np.all(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])==0,axis=1)
initial=np.flatnonzero(zero(np.arange(len(F))))
queue=[];queued=set()
def push_face(fi):
 if not alive[fi] or not zero([fi])[0]:return
 tr=F[fi]
 for a,b in ((tr[0],tr[1]),(tr[1],tr[2]),(tr[2],tr[0])):
  a,b=sorted((int(a),int(b)));dist=float(np.linalg.norm(P[a]-P[b]))
  if dist<=1e-6 and (a,b) not in queued:heapq.heappush(queue,(dist,a,b));queued.add((a,b))
for fi in initial:push_face(int(fi))
events=[];reject=collections.Counter()
def attempt(a,b):
 edge=inc[a]&inc[b]
 if len(edge)!=2:return None,"edge_not_two_faces"
 if pinned[a]:return None,"source_or_head_vertex_pinned"
 region=sorted(inc[a]|inc[b])
 if np.any(cap[region]):return None,"cap_incident"
 na=set(F[list(inc[a])].reshape(-1).tolist())-{a}
 nb=set(F[list(inc[b])].reshape(-1).tolist())-{b}
 opposite=set(F[list(edge)].reshape(-1).tolist())-{a,b}
 if na&nb!=opposite:return None,"link_condition"
 remaining=[fi for fi in inc[a] if fi not in edge]
 if remaining:
  old=F[remaining].copy();new=old.copy();new[new==a]=b
  if np.any((new[:,0]==new[:,1])|(new[:,1]==new[:,2])|(new[:,2]==new[:,0])):return None,"unexpected_degenerate_face"
  on=np.cross(P[old[:,1]]-P[old[:,0]],P[old[:,2]]-P[old[:,0]])
  nn=np.cross(P[new[:,1]]-P[new[:,0]],P[new[:,2]]-P[new[:,0]])
  if np.any(np.einsum("ij,ij->i",on,nn)<=0):return None,"F64_normal_reversal"
  oq=np.cross(Q[old[:,1]]-Q[old[:,0]],Q[old[:,2]]-Q[old[:,0]])
  nq=np.cross(Q[new[:,1]]-Q[new[:,0]],Q[new[:,2]]-Q[new[:,0]])
  oldgood=np.any(oq!=0,axis=1);newbad=np.all(nq==0,axis=1)
  if np.any(oldgood&newbad):return None,"new_F32_collapse"
  if np.any(oldgood&(np.einsum("ij,ij->i",oq,nq)<=0)):return None,"F32_normal_reversal"
 removed_anc=set().union(*(anc[fi] for fi in edge))
 for fi in sorted(edge):
  for v in F[fi]:inc[int(v)].discard(fi)
  alive[fi]=False
 for fi in remaining:
  inc[a].discard(fi);inc[b].add(fi);F[fi,F[fi]==a]=b;changed[fi]=True;anc[fi]|=removed_anc
 return dict(removed_vertex=a,kept_vertex=b,displacement_m=float(np.linalg.norm(P[a]-P[b])),removed_input_face_rows=sorted(edge),changed_input_face_rows=sorted(remaining)),None
while queue:
 dist,a,b=heapq.heappop(queue);queued.discard((a,b))
 if not inc[a] or not inc[b] or not (inc[a]&inc[b]):continue
 entry=None
 for u,v in ((a,b),(b,a)):
  entry,why=attempt(u,v)
  if entry is not None:break
  reject[why]+=1
 if entry is None:continue
 events.append(entry)
 around=set(inc[entry["kept_vertex"]])
 for fi in list(around):
  for v in F[fi]:around|=inc[int(v)]
 for fi in around:push_face(fi)
 if len(events)%100==0:print(json.dumps(dict(stage="contract",accepted=len(events),queue=len(queue))),flush=True)
rows=np.flatnonzero(alive);ff=F[rows];used,inv=np.unique(ff,return_inverse=True);pp=P[used];ff=inv.reshape(-1,3);caps=cap[rows]
E=np.sort(np.vstack([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1);ue,counts=np.unique(E,axis=0,return_counts=True)
postQ=pp.astype(np.float32).astype(float);norm=np.cross(postQ[ff[:,1]]-postQ[ff[:,0]],postQ[ff[:,2]]-postQ[ff[:,0]]);bad=np.flatnonzero(np.all(norm==0,axis=1))
attrs={k:np.asarray(d[k])[rows].copy() for k in ("face_run_original_id","face_id","face_source_layer","face_source_body","face_source_bone","face_source_identity0","face_source_identity1")}
mask=changed[rows];schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
newid=61000001
while str(newid) in schema:newid+=1
newfids=np.arange(int(mask.sum()),dtype=np.int64)
view=OUT/"I-contracted-seam-source.npz"
np.savez_compressed(view,points_world_m_f64=pp,faces_f64=ff[mask],face_id=newfids,face_source_identity0=newfids,face_source_identity1=np.full(len(newfids),-1,np.int64))
parents=sorted({int(d["face_run_original_id"][r]) for fi in rows[mask] for r in anc[fi]})
schema[str(newid)]=dict(role="I",path=str(view),sha256=sha(view),identity_keys=["face_source_identity0","face_source_identity1"],identity_face_id_key="face_id",face_count=len(newfids),derived_operation="link_condition_precision_seam_contraction",derived_parent_source_ids=parents,derived_parent_assets=[dict(path=str(INPUT),sha256=sha(INPUT))],source_geometry_scope="complete_reconstructed_patch_faces",maximum_vertex_contraction_m=1e-6)
for key,val in (("face_run_original_id",newid),("face_source_layer",1),("face_source_body",-1),("face_source_bone",-1),("face_source_identity1",-1)):attrs[key][mask]=val
attrs["face_id"][mask]=newfids;attrs["face_source_identity0"][mask]=newfids
off=[0];ar=[]
for fi in rows:ar.extend(sorted(anc[int(fi)]));off.append(len(ar))
packed=dict(points_world_m_f64=pp,faces_f64=ff,face_cache_row=np.where(mask,-1,rows),**attrs,
 face_ancestry_offsets=np.asarray(off,np.int64),face_ancestry_input_cache_rows=np.asarray(ar,np.int64),
 face_ancestry_original_ids=d["face_run_original_id"][ar],face_ancestry_face_ids=d["face_id"][ar],face_ancestry_identity0=d["face_source_identity0"][ar],face_ancestry_identity1=d["face_source_identity1"][ar],
 source_points_f32=src,original_source_points_f32=src,temporary_cap_descendant_face_mask=caps,
 output_face_keep_mask_after_exact_cap_removal=~caps,source_cache_sha256=np.asarray([sha(INPUT)]),
 face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True)]))
np.savez_compressed(OUT/"candidate-f64.npz",**packed)
report=dict(input=str(INPUT),input_sha256=sha(INPUT),initial_f32_zero_faces=len(initial),remaining_f32_zero_faces=len(bad),remaining_bad_output_face_rows=bad.tolist(),accepted_contractions=len(events),events=events,reject_reasons=dict(reject),maximum_displacement_m=max([x["displacement_m"] for x in events],default=0),input_closed_euler=input_euler,closed_euler=len(used)-len(ue)+len(ff),closed_edge_incidence=dict(zip(map(str,np.unique(counts)),[int((counts==v).sum()) for v in np.unique(counts)])),source_points_pinned=int(pinned.sum()),new_derived_faces=int(mask.sum()),admission=False,physical_steps=0)
# Local link-safe contractions must preserve this input topology, whatever its Euler characteristic is.
if not len(bad) and np.all(counts==2) and report["closed_euler"]==input_euler:
 sp=importlib.util.spec_from_file_location("emitter270",R/"emit_f32.py");h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
 try:
  # Emit the closed diagnostic candidate with its original temporary cap descendants intact.
  y,f32,em=h.emit(pp,ff,src)
  keep_face_rows=np.arange(len(ff),dtype=np.int64)
  ancestry_counts=np.diff(packed["face_ancestry_offsets"])[keep_face_rows]
  ancestry_offsets_f32=np.r_[0,np.cumsum(ancestry_counts)].astype(np.int64)
  ancestry_slots=np.concatenate([np.arange(packed["face_ancestry_offsets"][i],packed["face_ancestry_offsets"][i+1],dtype=np.int64) for i in keep_face_rows])
  ancestry_cache_rows_f32=packed["face_ancestry_input_cache_rows"][ancestry_slots]
  ancestry_original_ids_f32=packed["face_ancestry_original_ids"][ancestry_slots]
  ancestry_face_ids_f32=packed["face_ancestry_face_ids"][ancestry_slots]
  ancestry_identity0_f32=packed["face_ancestry_identity0"][ancestry_slots]
  ancestry_identity1_f32=packed["face_ancestry_identity1"][ancestry_slots]
  mask_f32=mask
  np.savez_compressed(view,points_world_m_f64=y.astype(np.float64),faces_f64=f32[mask_f32],face_id=newfids,face_source_identity0=newfids,face_source_identity1=np.full(len(newfids),-1,np.int64))
  schema[str(newid)]["sha256"]=sha(view)
  schemajson=np.asarray([json.dumps(schema,sort_keys=True)])
  packed["face_source_identity_schema_by_original_id_json"]=schemajson
  np.savez_compressed(OUT/"candidate-f64.npz",**packed)
  topology_edges=np.sort(np.vstack((f32[:,[0,1]],f32[:,[1,2]],f32[:,[2,0]])),axis=1)
  topology_unique,topology_counts=np.unique(topology_edges,axis=0,return_counts=True)
  emitted_open=int(np.count_nonzero(topology_counts==1));emitted_nonmanifold=int(np.count_nonzero(topology_counts>2))
  if emitted_open!=0 or emitted_nonmanifold:
   raise ValueError(f"I closed emitted boundary mismatch: open={emitted_open} nonmanifold={emitted_nonmanifold}")

  finalattrs={k:val for k,val in attrs.items()};finalattrs["face_source_original_id"]=finalattrs.pop("face_run_original_id");finalattrs["face_source_face_id"]=finalattrs.pop("face_id")
  np.savez_compressed(OUT/"candidate-closed-f32.npz",points_world_m=y,faces=f32,face_cache_row=packed["face_cache_row"],**finalattrs,
   face_ancestry_offsets=ancestry_offsets_f32,face_ancestry_input_cache_rows=ancestry_cache_rows_f32,
   face_ancestry_original_ids=ancestry_original_ids_f32,face_ancestry_face_ids=ancestry_face_ids_f32,
   face_ancestry_identity0=ancestry_identity0_f32,face_ancestry_identity1=ancestry_identity1_f32,
   face_source_identity_schema_by_original_id_json=schemajson,float32_emission_json=np.asarray([json.dumps(em)]),
   temporary_cap_descendant_face_mask=caps,output_face_keep_mask_after_exact_cap_removal=~caps)
  report.update(emission="PASS",candidate_f32_sha256=sha(OUT/"candidate-closed-f32.npz"))
 except ValueError as e:report["emission_error"]=str(e)
report["elapsed_s"]=time.monotonic()-t0
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k not in ("events","remaining_bad_output_face_rows")}),flush=True)


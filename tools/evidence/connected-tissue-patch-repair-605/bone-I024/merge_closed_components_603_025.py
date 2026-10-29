import hashlib,json,os,sys
from pathlib import Path
import numpy as np

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
IN=BASE/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/candidate-run"
ORIG=BASE/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/candidate-f32.npz"
OPS=BASE/"closed-envelope-component-repair-603/closed-component-patch-operations.json"
GATE=BASE/"closed-envelope-component-repair-603/negative-hull-bone-and-I-gates.json"
OUT=BASE/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/closed603-merge-025"
EXPECTED="f464d4f35613b612c055978a01ffff8c596ea248b82a6c4b84a7bcc170b36800"
BASESHA="e8e332fb4ab421db6e44f0f5fe638e378646fd63438a318dcf94ad84b7a8ce96"
OPSSHA="9132750061fb771787ea7d4d3d2f3659b784b65ba56b292a2a914ba071a13b6a"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def csr_rows(off,flat):
 o=np.asarray(off,np.int64);v=np.asarray(flat,np.int64)
 return [v[int(o[i]):int(o[i+1])].tolist() for i in range(len(o)-1)]
def flatten(rows):
 off=[0];flat=[]
 for r in rows: flat.extend(r);off.append(len(flat))
 return np.asarray(off,np.int64),np.asarray(flat,np.int64)
def atomic_npz(path,**kw):
 tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("wb") as f: np.savez_compressed(f,**kw)
 os.replace(tmp,path)
def signed_volume(P,F):
 tri=np.asarray(P,np.float64)[np.asarray(F,np.int64)]
 return float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6.0)
if OUT.exists(): raise SystemExit("output exists")
OUT.mkdir(parents=True)
cand=IN/"candidate-f32-linked-remesh-typed.npz"
raw=IN/"candidate-geometry-final-checkpoint.npz"
if sha(cand)!=EXPECTED: raise ValueError("batch candidate SHA mismatch")
if sha(OPS)!=OPSSHA: raise ValueError("603 operation SHA mismatch")
d={}
with np.load(cand,allow_pickle=False) as z:
 d={k:np.array(z[k],copy=True) for k in z.files}
with np.load(raw,allow_pickle=False) as z:
 RP=np.array(z["points_world_m"],copy=True); RF=np.array(z["faces"],np.int64,copy=True)
 retained=np.array(z["retained_input_face_rows"],np.int64,copy=True)
 removed_batch=np.array(z["removed_input_face_rows"],np.int64,copy=True)
 newbatch=np.array(z["new_faces"],np.int64,copy=True)
with np.load(ORIG,allow_pickle=False) as z:
 baseF=np.array(z["faces"],np.int64,copy=True)
if d["points_world_m"].dtype!=np.float32 or RP.dtype!=np.float32: raise ValueError("not native Float32 point payload")
if not np.array_equal(d["points_world_m"],RP) or not np.array_equal(d["faces"],RF): raise ValueError("typed/raw batch geometry mismatch")
if len(retained)+len(newbatch)!=len(RF) or not np.array_equal(RF[-len(newbatch):],newbatch): raise ValueError("bad raw batch row map")
if len(baseF)<=int(retained.max()): raise ValueError("retained rows outside original face domain")
base_to_out=np.full(len(baseF),-1,np.int64);base_to_out[retained]=np.arange(len(retained),dtype=np.int64)
opobj=json.loads(OPS.read_text())
if opobj.get("candidate_sha256")!=BASESHA or opobj.get("exact_report_sha256")!="e725631f7f337b58a0dbc49d33c4b2f555ce319f0dbed1b0a6c4f58f99b4716d":
 raise ValueError("603 operations are not pinned to candidate007/exact387")
ops=opobj["operations"]
if [int(x["transaction_id"]) for x in ops]!=[270,309]: raise ValueError("closed-operation selection mismatch")
if set(opobj["accepted_transactions"])!={270,309}: raise ValueError("closed operation IDs mismatch")
P32=np.array(d["points_world_m"],copy=True)
F0=np.array(d["faces"],np.int64,copy=True)
allbase_rows=[]
mapped_ops=[]
for op in ops:
 tx=int(op["transaction_id"]);rows=np.asarray(op["removed_face_rows"],np.int64)
 if len(rows)!=8 or len(set(map(int,rows)))!=8: raise ValueError("closed component must replace exactly eight source faces")
 if np.any(rows<0) or np.any(rows>=len(baseF)): raise ValueError("closed source face row outside base domain")
 mapped=base_to_out[rows]
 if np.any(mapped<0): raise ValueError("closed source face rows were not retained through batch")
 if np.intersect1d(rows,np.asarray(removed_batch,np.int64)).size: raise ValueError("closed operation overlaps a remeshed source row")
 nf=np.asarray(op["new_faces"],np.int64)
 if nf.shape!=(8,3) or np.any(nf<0) or np.any(nf>=len(P32)): raise ValueError("closed component replacement shape/index error")
 if op.get("boundary_loops")!=[]: raise ValueError("closed component operation has a boundary")
 old=F0[mapped]
 used=np.unique(old)
 if len(used)!=6 or set(used)!=set(np.unique(nf)): raise ValueError("closed hull must preserve its exact six vertices")
 # Check old and new patches are closed, connected genus-zero components with negative orientation.
 def top(F):
  edges=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
  edges=np.sort(edges,axis=1);ue,c=np.unique(edges,axis=0,return_counts=True)
  chi=len(np.unique(F))-len(ue)+len(F)
  return bool(np.all(c==2)),int(chi),int(len(F))
 oc,ochi,_=top(old);nc,nchi,_=top(nf)
 oldv=signed_volume(P32,old);newv=signed_volume(P32,nf)
 if not (oc and nc and ochi==2 and nchi==2 and oldv<0 and newv<0):
  raise ValueError(f"closed negative component contract failed tx={tx}: old={oc,ochi,oldv} new={nc,nchi,newv}")
 if not np.array_equal(P32,RP): raise ValueError("component coordinates changed")
 mapped_ops.append({"transaction_id":tx,"rows_base":rows,"rows_out":mapped,"new_faces":nf,"vertices":used,
                    "old_signed_volume_m3":oldv,"new_signed_volume_m3":newv,"source_operation":op})
 allbase_rows.extend(map(int,rows))
if len(set(allbase_rows))!=len(allbase_rows): raise ValueError("closed operation source rows overlap")
# Replace only the two selected closed components and checkpoint geometry before metadata.
keep=np.ones(len(F0),bool)
for x in mapped_ops: keep[x["rows_out"]]=False
closed_new=np.concatenate([x["new_faces"] for x in mapped_ops],axis=0)
F1=np.concatenate((F0[keep],closed_new),axis=0)
geom=OUT/"candidate-geometry-final-checkpoint.npz"
atomic_npz(geom,points_world_m=P32,faces=F1,
           batch016_input_candidate_sha256=np.asarray([EXPECTED]),
           source603_operations_sha256=np.asarray([OPSSHA]),
           removed_closed_batch_rows=np.concatenate([x["rows_out"] for x in mapped_ops]),
           added_closed_faces=closed_new)
# Closed component operations cannot alter the outer boundary. Assert edge-boundary identity.
def boundary(F):
 ed=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
 ed=np.sort(ed,axis=1);u,c=np.unique(ed,axis=0,return_counts=True)
 return u[c==1]
if not np.array_equal(boundary(F0),boundary(F1)): raise ValueError("closed component merge changed boundary edge set")
# Metadata update.
n0=len(F0); retainrows=np.flatnonzero(keep).astype(np.int64); nnew=len(closed_new)
used_ids=set(map(int,d["face_source_original_id"]))
if 62050000 in used_ids or 62050001 in used_ids: raise ValueError("closed operation source ID collision")
schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
cand_rows=csr_rows(d["candidate_face_ancestry_offsets"],d["candidate_face_ancestry_rows"])
urows=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_rows"])
uruns=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_run_original_ids"])
ufids=csr_rows(d["union_face_ancestry_offsets"],d["union_face_ancestry_face_ids"])
d289rows=csr_rows(d["d289_parent_row_indptr"],d["d289_parent_rows"])
unionrows=[list(zip(uruns[i],ufids[i],urows[i])) for i in range(n0)]
new_cand=[];new_union=[];new_d289=[];source_views=[];source_entries=[];face_meta=[]
for j,x in enumerate(mapped_ops):
 op=x["source_operation"];sid=62050000+j;rr=list(map(int,x["rows_out"]));tris=x["new_faces"]
 parents_candidate=sorted(set(q for r in rr for q in cand_rows[r]))
 parents_union=sorted(set((int(a),int(b),int(c)) for r in rr for a,b,c in unionrows[r]))
 parents_d289=sorted(set(q for r in rr for q in d289rows[r]))
 parent_source_ids=set(map(int,d["face_source_original_id"][rr]))
 parent_source_ids.update(a for a,_,_ in parents_union)
 layer_vals=set(map(int,d["face_source_layer"][rr]))
 layer=next(iter(layer_vals)) if len(layer_vals)==1 else 2
 body_vals=set(map(int,d["face_source_body"][rr])); body=next(iter(body_vals)) if len(body_vals)==1 else -1
 bone_vals=set(map(int,d["face_source_bone"][rr])); bone=next(iter(bone_vals)) if len(bone_vals)==1 else -1
 fids=np.arange(len(tris),dtype=np.int64);verts=np.unique(tris.reshape(-1));loc=np.full(len(P32),-1,np.int64);loc[verts]=np.arange(len(verts))
 side=OUT/"derived-source-views"/f"D-derived-source-{sid}.npz";side.parent.mkdir(parents=True,exist_ok=True)
 atomic_npz(side,points_world_m=P32[verts],source_faces=loc[tris],
  face_source_original_id=np.full(len(tris),sid,np.int64),face_source_face_id=fids,
  face_source_identity0=fids,face_source_identity1=np.full(len(tris),-1,np.int64),
  parent_candidate_sha256=np.asarray([BASESHA]),parent_candidate_face_offsets=np.r_[0,np.cumsum([len(parents_candidate)]*len(tris))],
  parent_candidate_face_rows=np.tile(np.asarray(parents_candidate,np.int64),len(tris)),
  parent_candidate_face_support_input_rows=np.asarray(sorted(map(int,op["parent_face_rows"])),np.int64),
  parent_union_source_offsets=np.r_[0,np.cumsum([len(parents_union)]*len(tris))],
  parent_union_source_run_original_ids=np.tile(np.asarray([a for a,b,c in parents_union],np.int64),len(tris)),
  parent_union_source_face_ids=np.tile(np.asarray([b for a,b,c in parents_union],np.int64),len(tris)),
  parent_union_source_rows=np.tile(np.asarray([c for a,b,c in parents_union],np.int64),len(tris)),
  parent_d289_offsets=np.r_[0,np.cumsum([len(parents_d289)]*len(tris))],
  parent_d289_rows=np.tile(np.asarray(parents_d289,np.int64),len(tris)),
  input_candidate_sha256=np.asarray([EXPECTED]),operation_manifest_sha256=np.asarray([OPSSHA]),
  operation=np.asarray(["closed negative void component hull using unchanged existing vertices; source faces replaced under pinned 603 operation; negative orientation retained"]))
 side_sha=sha(side)
 source_entries.append({"schema_id":sid,"schema":{"role":"D","layer":layer,"body":body,"bone":bone,
  "identity_face_id_key":"face_source_face_id","identity_keys":["face_source_identity0","face_source_identity1"],
  "derived_operation":"closed_negative_void_component_hull_existing_vertices",
  "derived_parent_assets":[{"asset_role":"candidate007_parent_face_row_domain","path":str(ORIG),"sha256":BASESHA},
   {"asset_role":"bone_barrier_current_geometry","path":str(IN/"candidate-f32-linked-remesh-typed.npz"),"sha256":EXPECTED},
   {"asset_role":"source603_closed_component_operations","path":str(OPS),"sha256":OPSSHA},
   {"asset_role":"source603_negative_component_gates","path":str(GATE),"sha256":sha(GATE)}],
  "derived_parent_source_ids":sorted(parent_source_ids),"face_count":len(tris),
  "source_geometry_scope":"one complete closed negative-oriented void shell; exact six vertices retained; eight parent faces replaced; full source support is in candidate CSR",
  "path":str(side),"sha256":side_sha}})
 source_views.append({"transaction_id":x["transaction_id"],"source_id":sid,"path":str(side),"sha256":side_sha,
  "faces":len(tris),"vertices":len(verts),"old_signed_volume_m3":x["old_signed_volume_m3"],"new_signed_volume_m3":x["new_signed_volume_m3"],
  "parent_candidate_rows":rr,"parent_union_source_count":len(parents_union),"parent_candidate_source_rows":len(parents_candidate)})
 new_cand.extend([parents_candidate]*len(tris));new_union.extend([parents_union]*len(tris));new_d289.extend([parents_d289]*len(tris))
 face_meta.append((sid,layer,body,bone,fids))
out=dict(d);out["faces"]=F1;out["points_world_m"]=P32
def appfield(name,vals):
 a=np.asarray(d[name]);return np.concatenate((a[retainrows],np.asarray(vals,dtype=a.dtype)),axis=0)
out["face_source_original_id"]=appfield("face_source_original_id",np.concatenate([np.full(8,sid,np.int64) for sid,_,_,_,_ in face_meta]))
out["face_source_face_id"]=appfield("face_source_face_id",np.concatenate([fids for _,_,_,_,fids in face_meta]))
out["face_source_identity0"]=appfield("face_source_identity0",np.concatenate([fids for _,_,_,_,fids in face_meta]))
out["face_source_identity1"]=appfield("face_source_identity1",np.full(nnew,-1,np.int64))
for name,idx in (("face_source_layer",1),("face_source_body",2),("face_source_bone",3)):
 out[name]=appfield(name,np.concatenate([np.full(8,x[idx],np.int64) for x in face_meta]))
for name in ("union_input_run_original_id","union_input_face_id","union_input_face_rows"):
 out[name]=appfield(name,np.full(nnew,-1,np.int64))
# Keep exact existing ancestry; each new closed-shell face receives the union support of all eight replaced parent faces.
ca=[cand_rows[int(r)] for r in retainrows]+new_cand
co,cf=flatten(ca);out["candidate_face_ancestry_offsets"]=co;out["candidate_face_ancestry_rows"]=cf
ua=[];uf=[];ui=[];uo=[0]
for row in [unionrows[int(r)] for r in retainrows]+new_union:
 uniq=sorted(set(tuple(map(int,x)) for x in row))
 ui.extend(x[0] for x in uniq);uf.extend(x[1] for x in uniq);ua.extend(x[2] for x in uniq);uo.append(len(ui))
out["union_face_ancestry_offsets"]=np.asarray(uo,np.int64);out["union_face_ancestry_run_original_ids"]=np.asarray(ui,np.int64);out["union_face_ancestry_face_ids"]=np.asarray(uf,np.int64);out["union_face_ancestry_rows"]=np.asarray(ua,np.int64)
do,dr=flatten([d289rows[int(r)] for r in retainrows]+new_d289);out["d289_parent_row_indptr"]=do;out["d289_parent_rows"]=dr
for x in source_entries:
 sid=x["schema_id"]
 if str(sid) in schema: raise ValueError("schema key collision")
 schema[str(sid)]=x["schema"]
out["face_source_identity_schema_by_original_id_json"]=np.asarray([json.dumps(schema,sort_keys=True,separators=(",",":"))])
oprecord=json.loads(str(d["pair_linked_remesh_operation_json"][0]))
oprecord["closed_component_operations_603"]={"manifest_path":str(OPS),"manifest_sha256":OPSSHA,"gate_receipt_path":str(GATE),"gate_receipt_sha256":sha(GATE),
 "operations":[{"transaction_id":x["transaction_id"],"source_id":source_views[i]["source_id"],"removed_input_candidate_rows":list(map(int,x["source_operation"]["removed_face_rows"])),"new_faces":8,"closed_boundary":True,"negative_orientation_preserved":True,"coordinates_changed":0,"old_signed_volume_m3":x["old_signed_volume_m3"],"new_signed_volume_m3":x["new_signed_volume_m3"]} for i,x in enumerate(mapped_ops)],
 "physical_admission":False}
out["pair_linked_remesh_operation_json"]=np.asarray([json.dumps(oprecord,sort_keys=True,separators=(",",":"))])
typed=OUT/"candidate-f32-linked-remesh-typed.npz";atomic_npz(typed,**out)
receipt={"schema":"candidate007-pair-remesh-plus-603-negative-void-components.v1","input_candidate_path":str(IN/"candidate-f32-linked-remesh-typed.npz"),"input_candidate_sha256":EXPECTED,
 "source603_operations_path":str(OPS),"source603_operations_sha256":OPSSHA,"source603_gate_path":str(GATE),"source603_gate_sha256":sha(GATE),
 "raw_geometry_path":str(geom),"raw_geometry_sha256":sha(geom),"typed_candidate_path":str(typed),"typed_candidate_sha256":sha(typed),
 "points":len(P32),"faces_before":len(F0),"faces_after":len(F1),"vertices_unchanged":True,"closed_component_views":source_views,
 "boundary_edge_set_unchanged":True,"ordinary_batch_receipt_sha256":sha(IN/"result.json"),"full_exact_self":"NOT_RUN","all185":"NOT_RUN","D_minus_I":"NOT_RUN","physical_admission":False}
(OUT/"merge-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True),flush=True)

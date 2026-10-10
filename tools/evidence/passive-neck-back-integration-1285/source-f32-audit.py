#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,struct,sys,zipfile
from pathlib import Path
import numpy as np
OUT=Path("/Users/n/numi-human-retained-delivery-20261009/passive-neck-back-coverage-1281/candidate-002")
PAYLOAD=OUT/"bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MANIFEST=OUT/"bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
SOURCES=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources")
sys.path.insert(0,"/Users/n/numi-human-self-separation-partial-resume-1267/src")
from numilab_human.compiled_quotient_embeddedness import classify_quotient
from numilab_human import model as exact_model, cardiac_cavity_intersections as predicate
from numilab_human import compiled_quotient_embeddedness as quotient_owner, whole_body_embeddedness as indexed_owner
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def brief(x):
 t=x["topology"]
 return {"status":x["status"],"self_intersection":x["self_intersection"],
 "exact_intersection_pairs":x["exact_intersection_pairs"],
 "allowed_shared_vertex_or_edge_pairs":x.get("allowed_shared_vertex_or_edge_pairs"),
 "first_intersecting_face_pairs":x.get("first_intersecting_face_pairs"),
 "topology":{k:t.get(k) for k in ("vertex_count","face_count","edge_count","boundary_edge_count",
 "nonmanifold_edge_count","orientation_conflict_count","degenerate_face_count","face_component_count",
 "closed_oriented_manifold_candidate")}}
manifest=json.loads(MANIFEST.read_text());raw=PAYLOAD.read_bytes()
magic,abi,nr,nb,nv,ni,fingerprint,source_sha=struct.unpack_from("<8s6I32s",raw)
assert magic.rstrip(bytes([0]))==b"NHTISS4" and abi==5 and nr==8
assert len(raw)==64+nr*32+nb*36+nv*56+ni*4
records=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(nr,8)
boff=64+nr*32;voff=boff+nb*36;ioff=voff+nv*56
positions=np.ndarray((nv,3),dtype="<f4",buffer=raw,offset=voff,strides=(56,4))
indices=np.frombuffer(raw,dtype="<u4",count=ni,offset=ioff)
binds=np.frombuffer(raw,dtype="<u4",count=nb*9,offset=boff).reshape(nb,9)
name_by_core={int(b["core_body_index"]):b["myosim_body"] for s in manifest["source"]["surfaces"] for b in s["body_bindings"]}
archive=SOURCES/"isa_BP3D_4.0_obj_99.zip"; rows=[]
for sr,rec in zip(manifest["source"]["surfaces"],records,strict=True):
 fb,bc,fv,vc,fi,ic,sid,layer=map(int,rec);assert sid==sr["stable_id"] and layer==1
 gf=indices[fi:fi+ic];assert ic%3==0 and bool(((gf>=fv)&(gf<fv+vc)).all())
 faces=(gf-fv).reshape(-1,3).astype(np.int64);pos=positions[fv:fv+vc];assert np.isfinite(pos).all()
 with zipfile.ZipFile(archive) as zf:
  member=[x for x in zf.namelist() if Path(x).stem==sr["member_id"] and x.lower().endswith(".obj")]
  assert len(member)==1;obj=zf.read(member[0])
 assert hashlib.sha256(obj).hexdigest()==sr["member_sha256"]
 srcv,srcf=exact_model._bodyparts_obj_triangles(obj,member[0])
 cancellation=sr.get("source_topology_cancellation")
 removed=set()
 if cancellation is not None:
  pairs=cancellation["cancelled_opposite_face_pairs"]
  removed={int(i) for pair in pairs for i in pair}
  assert len(removed)==2*len(pairs) and all(0<=i<len(srcf) for i in removed)
  assert cancellation["source_triangle_count"]==len(srcf)
 retained=[f for i,f in enumerate(srcf) if i not in removed]
 assert cancellation is None or len(srcf)-2*len(cancellation["cancelled_opposite_face_pairs"])==len(retained)
 assert len(srcv)==vc and len(retained)==len(faces)
 assert np.array_equal(np.asarray(retained,dtype=np.uint32),faces.astype(np.uint32))
 source=classify_quotient([[float(c) for c in v] for v in srcv],[[int(c) for c in f] for f in srcf])
 retained_source=classify_quotient([[float(c) for c in v] for v in srcv],[[int(c) for c in f] for f in retained])
 packed=classify_quotient(pos.astype(float).tolist(),faces.tolist())
 bnames=[name_by_core[int(x[0])] for x in binds[fb:fb+bc]]
 assert bnames==[x["myosim_body"] for x in sr["body_bindings"]]
 w=np.ndarray((vc,4),dtype="<f4",buffer=raw,offset=voff+fv*56+40,strides=(56,4))
 ids=np.ndarray((vc,4),dtype="<u4",buffer=raw,offset=voff+fv*56+24,strides=(56,4))
 active=[]
 for ii,ww in zip(ids,w,strict=True):
  on=ww>0;assert np.isfinite(ww).all() and bool((ww>=0).all()) and abs(float(ww.sum())-1)<=1e-6
  assert bool((ii[on]<bc).all()) and bool((ii[~on]==0xffffffff).all());active.append(int(on.sum()))
 rows.append({"stable_id":sid,"member_id":sr["member_id"],"source_member_sha256":sr["member_sha256"],
  "source_vertex_count":len(srcv),"compiled_vertex_count":vc,"source_triangle_count":len(srcf),
  "compiled_triangle_count":len(faces),"source_face_rows_accounted_by_declared_cancellation":True,
  "source_topology_cancellation":cancellation,
  "authored_source_exact_quotient":brief(source),"retained_source_exact_quotient":brief(retained_source),
  "compiled_f32_exact_quotient":brief(packed),
  "registered_support_names":bnames,"support_frame_count":len(bnames),
  "active_sparse_influences":{"min":min(active),"max":max(active),"vertices_with_four":sum(x==4 for x in active)},
  "stored_weights_finite_nonnegative_unit_sum":True,"passive_visual_binding":sr["passive_visual_binding"]})
 print(json.dumps({"stable_id":sid,"member":sr["member_id"],"source_pairs":source["exact_intersection_pairs"],
  "retained_source_pairs":retained_source["exact_intersection_pairs"],
  "candidate_pairs":packed["exact_intersection_pairs"],"candidate_status":packed["status"]}),flush=True)
code={"compiled_quotient_embeddedness.py":sha(quotient_owner.__file__),
 "whole_body_embeddedness.py":sha(indexed_owner.__file__),"cardiac_cavity_intersections.py":sha(predicate.__file__)}
assert code["cardiac_cavity_intersections.py"]=="423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd"
result={"schema":"numi.human.passive-neck-back-candidate-exact-self-audit.v1",
"scope":"Eight new passive visual rows only: authored exact-coordinate quotient, existing declared opposite-face cancellation lineage, and serialized NHTISS4 ABI5 Float32 exact-coordinate quotient. No cross-surface, skin, bone, pose/native, attachment-footprint, mechanics, clinical or whole-body admission.",
"inputs":{str(PAYLOAD):{"sha256":sha(PAYLOAD),"bytes":len(raw)},str(MANIFEST):{"sha256":sha(MANIFEST)},
 str(archive):{"sha256":sha(archive)},"surface_map_sha256":sha("/Users/n/numi-human-passive-neck-back-1281/config/bodyparts3d-myosim-surface-map.v1.json"),
 "exact_owner_modules":code,"predicate_import_path":str(Path(predicate.__file__).resolve())},
"coverage":{"rows":len(rows),"source_faces_accounted_all":all(r["source_face_rows_accounted_by_declared_cancellation"] for r in rows),
 "compiled_closed_oriented_all":all(r["compiled_f32_exact_quotient"]["topology"]["closed_oriented_manifold_candidate"] for r in rows),
 "compiled_self_zero_all":all(r["compiled_f32_exact_quotient"]["exact_intersection_pairs"]==0 for r in rows)},
"surfaces":rows,"native_qualified":False,"fullbody_anatomy_admitted":False}
report=OUT/"source-and-f32-exact-audit.json";report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(report),"sha256":sha(report),"coverage":result["coverage"]},indent=2),flush=True)


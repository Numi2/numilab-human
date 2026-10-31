#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/actual-697"
TYPED=H/"d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/typed-bind-002/candidate-D-typed-current.npz"
TYPED_SHA="97ebce376ec202cb63c3339f4bc433c80a724fc6d63362db56d47d3bf0313bbb"
SCHEMA_SRC=H/"quality-warning-connected-union-038-typed-metadata-005-001/D-run-001/candidate-D-connected-union-typed.npz"
SCHEMA_SHA="d0f6cda10552a55be2d6ac5a0740e0142293cdf55177f1ce853a3eb5624374da"
HELPER=H/"current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py"
HELPER_SHA="83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
for p,w in ((TYPED,TYPED_SHA),(SCHEMA_SRC,SCHEMA_SHA),(HELPER,HELPER_SHA)):
 if sha(p)!=w:raise ValueError("pinned input changed: "+str(p))
source=np.load(TYPED,allow_pickle=False);schema_src=np.load(SCHEMA_SRC,allow_pickle=False)
schema_json=np.asarray(schema_src["face_source_identity_schema_by_original_id_json"])
if schema_json.size!=1:raise ValueError("identity schema payload is not scalar")
schema=json.loads(str(schema_json.item()))
if set(map(str,np.unique(source["face_source_original_id"])))-set(schema):raise ValueError("schema source coverage incomplete")
N=len(source["faces"]);off=np.asarray(source["face_ancestry_offsets"],np.int64)
rows=np.asarray(source["face_ancestry_source_face_rows"],np.int64)
runs=np.asarray(source["face_ancestry_source_original_ids"],np.int64)
ids=np.asarray(source["face_ancestry_parent_face_rows"],np.int64)
if off.shape!=(N+1,) or off[0]!=0 or off[-1]!=len(rows) or len(runs)!=len(rows) or len(ids)!=len(rows) or np.any(np.diff(off)<=0):raise ValueError("full ancestry CSR malformed")
primary=off[:-1]
fields=("face_source_layer","face_source_body","face_source_bone","face_source_original_id")
signatures=np.stack([np.asarray(source[k],np.int64) for k in fields],axis=1)
unique,group=np.unique(signatures,axis=0,return_inverse=True)
keys=[dict(zip(fields,map(int,row))) for row in unique]
counts=np.bincount(group,minlength=len(keys)).astype(np.int64)
group_offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
group_rows=np.argsort(group,kind="stable").astype(np.int64)
payload={k:np.asarray(source[k]) for k in source.files}
payload.update({
 "union_input_run_original_id":np.asarray(source["face_source_original_id"],np.int64),
 "union_input_face_id":np.asarray(source["face_source_face_id"],np.int64),
 "union_input_face_rows":rows[primary],
 "union_face_ancestry_offsets":off,
 "union_face_ancestry_rows":rows,
 "union_face_ancestry_run_original_ids":runs,
 "union_face_ancestry_face_ids":ids,
 "candidate_face_ancestry_offsets":off,
 "candidate_face_ancestry_rows":rows,
 "face_source_identity_schema_by_original_id_json":schema_json,
 "lineage_preparation_input_sha256":np.asarray([sha(TYPED)]),
 "lineage_preparation_source_schema_sha256":np.asarray([sha(SCHEMA_SRC)]),
 "face_group_id":group.astype(np.int64),
 "source_group_offsets":group_offsets,
 "source_group_face_rows":group_rows,
 "source_group_keys_json":np.asarray([json.dumps(keys,separators=(",",":"))]),
 "face_parent_source_face_rows":np.arange(N,dtype=np.int64),
 "face_exact_source_row":np.full(N,-1,np.int64),
 "face_ancestry_method":np.asarray(["Adapter-local row grouping by current declared typed primary identity. Full flattened three-lane source-face ancestry is preserved; no exact original face or one-to-one overlap claim."])
})
(RUN/"typed-owner").mkdir(parents=True,exist_ok=True)
adapter=RUN/"typed-owner/candidate-D-typed-697-adapter-002.npz"
np.savez_compressed(adapter,**payload)
check=np.load(adapter,allow_pickle=False)
if not np.array_equal(check["faces"],source["faces"]) or not np.array_equal(check["points_world_m_f32"],source["points_world_m_f32"]):raise ValueError("adapter changed geometry")
for k,v in (("union_face_ancestry_offsets",off),("union_face_ancestry_rows",rows),("union_face_ancestry_run_original_ids",runs),("union_face_ancestry_face_ids",ids),("face_group_id",group),("source_group_offsets",group_offsets),("source_group_face_rows",group_rows)):
 if not np.array_equal(check[k],v):raise ValueError("adapter changed provenance "+k)
spec=importlib.util.spec_from_file_location("handoff697_D411_v2",HELPER);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
owner=mod.load_owner();start=time.monotonic();loaded=mod.load_surface_217(owner,adapter,"D",None)
points=np.asarray(loaded["points"],np.float32);faces=np.asarray(loaded["faces"],np.int64)
if not np.array_equal(points,source["points_world_m_f32"]) or not np.array_equal(faces,source["faces"]):raise ValueError("697 geometry differs from typed source")
if len(loaded["ancestry"])!=N:raise ValueError("697 ancestry row count mismatch")
for i in range(N):
 lo,hi=map(int,off[i:i+2])
 if loaded["ancestry"][i]!=rows[lo:hi].tolist() or loaded["ancestry_original_ids"][i]!=runs[lo:hi].tolist() or loaded["ancestry_parent_rows"][i]!=ids[lo:hi].tolist():raise ValueError("697 full ancestry mismatch at face "+str(i))
report={"schema":"h706.d-shortedge-quotient-475-actual-697-D-load.v1","status":"PASS_ACTUAL_697_217_TYPED_D_LOAD","script_path":str(Path(__file__)),"script_sha256":sha(Path(__file__)),"typed_source_path":str(TYPED),"typed_source_sha256":sha(TYPED),"identity_schema_source_path":str(SCHEMA_SRC),"identity_schema_source_sha256":sha(SCHEMA_SRC),"adapter_path":str(adapter),"adapter_sha256":sha(adapter),"helper_path":str(HELPER),"helper_sha256":sha(HELPER),"owner_build_path":str(mod.BUILD),"owner_build_sha256":sha(mod.BUILD),"owner_build_expected_sha256":mod.BUILD_SHA,"points":len(points),"faces":len(faces),"loaded_ancestry_tuple_count":int(len(rows)),"source_group_count":len(keys),"used_point_count":int(len(np.unique(faces))),"elapsed_s":time.monotonic()-start,"physical_admission":False,"native_steps":0,"tetgen_run":False,"mass_mapping":"NOT_RUN","limitations":["The actual 697/217 D loader accepted the typed D411 candidate and all three full ancestry lanes. This is a loader check only, not volume, mass, TetGen, or native acceptance."]}
out=RUN/"actual-697-D475-load-001.json";out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"receipt":str(out),"sha256":sha(out),"adapter_sha256":sha(adapter),"points":len(points),"faces":len(faces),"ancestry_rows":len(rows),"source_groups":len(keys),"elapsed_s":round(report["elapsed_s"],3)},sort_keys=True))

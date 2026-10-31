#!/usr/bin/env python3
"""Create the existing 697 D-loader envelope from the full run006 typed face unions and invoke it."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-385-run006"
TYPED=RUN/"typed-rebind-003/candidate-D-typed-current.npz"
SCHEMA_SRC=H/"quality-warning-connected-union-038-typed-metadata-005-001/D-run-001/candidate-D-connected-union-typed.npz"
HELPER=H/"current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py"
HELPER_SHA="83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f"
EXPECT={str(TYPED):"9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb",str(SCHEMA_SRC):"d0f6cda10552a55be2d6ac5a0740e0142293cdf55177f1ce853a3eb5624374da",str(HELPER):HELPER_SHA}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
for p,w in EXPECT.items():
 if sha(p)!=w: raise ValueError("pinned input changed: "+p)
source=np.load(TYPED,allow_pickle=False);schema_src=np.load(SCHEMA_SRC,allow_pickle=False)
schema_json=np.asarray(schema_src["face_source_identity_schema_by_original_id_json"])
if schema_json.size!=1: raise ValueError("identity schema payload is not scalar")
schema=json.loads(str(schema_json.item()))
roles=set(map(str,np.unique(source["face_source_original_id"])))
if roles-set(schema): raise ValueError("identity schema does not cover all typed source IDs")
N=len(source["faces"]);off=np.asarray(source["face_ancestry_offsets"],np.int64)
rows=np.asarray(source["face_ancestry_source_face_rows"],np.int64)
runs=np.asarray(source["face_ancestry_source_original_ids"],np.int64)
ids=np.asarray(source["face_ancestry_parent_face_rows"],np.int64)
if off.shape!=(N+1,) or off[0]!=0 or off[-1]!=len(rows) or len(runs)!=len(rows) or len(ids)!=len(rows) or np.any(np.diff(off)<=0): raise ValueError("full face ancestry CSR malformed")
# Keep scalar aliases as the first full-CSR member; the complete CSR remains authoritative.
primary=off[:-1]
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
})
adapter=RUN/"typed-rebind-003/candidate-D-typed-697-adapter-001.npz"
np.savez_compressed(adapter,**payload)
# Read back payload byte-for-byte array-wise before invoking the pinned loader.
check=np.load(adapter,allow_pickle=False)
if not np.array_equal(check["faces"],source["faces"]) or not np.array_equal(check["points_world_m_f32"],source["points_world_m_f32"]): raise ValueError("adapter changed geometry arrays")
for k,v in (("union_face_ancestry_offsets",off),("union_face_ancestry_rows",rows),("union_face_ancestry_run_original_ids",runs),("union_face_ancestry_face_ids",ids)):
 if not np.array_equal(check[k],v): raise ValueError("adapter changed ancestry lane "+k)
spec=importlib.util.spec_from_file_location("handoff697_adapter002",HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);owner=mod.load_owner()
t=time.monotonic();loaded=mod.load_surface_217(owner,adapter,"D",None)
points=np.asarray(loaded["points"],np.float32);faces=np.asarray(loaded["faces"],np.int64)
if not np.array_equal(points,source["points_world_m_f32"]): raise ValueError("697 loader changed D point bits")
if not np.array_equal(faces,source["faces"]): raise ValueError("697 loader changed D faces")
if len(loaded["ancestry"])!=N: raise ValueError("697 loaded ancestry row count mismatch")
for i in range(N):
 lo,hi=map(int,off[i:i+2])
 if loaded["ancestry"][i]!=rows[lo:hi].tolist() or loaded["ancestry_original_ids"][i]!=runs[lo:hi].tolist() or loaded["ancestry_parent_rows"][i]!=ids[lo:hi].tolist(): raise ValueError(f"697 ancestry differs at face {i}")
report={"schema":"h706.d-short-edge-quotient-run006-actual-697-d-load.v1","status":"PASS_ACTUAL_697_217_TYPED_D_LOAD","script_path":str(Path(__file__)),"script_sha256":sha(Path(__file__)),"typed_source_path":str(TYPED),"typed_source_sha256":sha(TYPED),"identity_schema_source_path":str(SCHEMA_SRC),"identity_schema_source_sha256":sha(SCHEMA_SRC),"adapter_path":str(adapter),"adapter_sha256":sha(adapter),"helper_path":str(HELPER),"helper_sha256":sha(HELPER),"owner_build_path":str(mod.BUILD),"owner_build_sha256":sha(mod.BUILD),"owner_build_expected_sha256":mod.BUILD_SHA,"points":int(len(points)),"faces":int(len(faces)),"loaded_ancestry_tuple_count":int(sum(map(len,loaded["ancestry"]))),"used_point_count":int(len(np.unique(faces))),"elapsed_s":time.monotonic()-t,"physical_admission":False,"native_steps":0,"tetgen_run":False,"mass_mapping":"NOT_RUN","limitations":["This invokes the pinned 697/217 D typed loader on the exact run006 D surface. O/I loading and the complete current O/I/D context are separately pinned; this is a metadata/loader check, not volume, mass, TetGen, or native acceptance."]}
out=RUN/"actual-697-D-load-002.json";out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"receipt":str(out),"sha256":sha(out),"adapter":str(adapter),"adapter_sha256":sha(adapter),"points":report["points"],"faces":report["faces"],"ancestry":report["loaded_ancestry_tuple_count"],"elapsed_s":round(report["elapsed_s"],3)},sort_keys=True))

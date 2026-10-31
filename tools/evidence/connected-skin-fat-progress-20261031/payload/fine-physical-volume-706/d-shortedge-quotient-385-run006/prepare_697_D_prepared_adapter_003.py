#!/usr/bin/env python3
"""Build a prepared-group envelope and run the existing 697 D loader."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
H=Path(__file__).resolve().parents[1];RUN=H/"d-shortedge-quotient-385-run006"
TYPED=RUN/"typed-rebind-003/candidate-D-typed-current.npz"
SCHEMA_SRC=H/"quality-warning-connected-union-038-typed-metadata-005-001/D-run-001/candidate-D-connected-union-typed.npz"
HELPER=H/"current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py"
HELPER_SHA="83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f"
WANT={str(TYPED):"9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb",str(SCHEMA_SRC):"d0f6cda10552a55be2d6ac5a0740e0142293cdf55177f1ce853a3eb5624374da",str(HELPER):HELPER_SHA}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
for p,d in WANT.items():
 if sha(p)!=d:raise ValueError("pinned input changed: "+p)
src=np.load(TYPED,allow_pickle=False);schemasrc=np.load(SCHEMA_SRC,allow_pickle=False)
schema_value=np.asarray(schemasrc["face_source_identity_schema_by_original_id_json"])
if schema_value.size!=1:raise ValueError("identity schema is not scalar")
schema=json.loads(str(schema_value.item()));N=len(src["faces"])
source_ids=set(map(str,np.unique(src["face_source_original_id"])))
if source_ids-set(schema):raise ValueError("schema source ID coverage incomplete")
fields=("face_source_layer","face_source_body","face_source_bone","face_source_original_id")
signatures=np.stack([np.asarray(src[k],np.int64) for k in fields],axis=1)
unique,group=np.unique(signatures,axis=0,return_inverse=True)
keys=[dict(zip(fields,map(int,row))) for row in unique]
counts=np.bincount(group,minlength=len(keys)).astype(np.int64)
offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
rows=np.argsort(group,kind="stable").astype(np.int64)
for lane in ("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"):
 if lane not in src.files:raise ValueError("typed source missing full ancestry lane "+lane)
Noff=np.asarray(src["face_ancestry_offsets"],np.int64)
if Noff.shape!=(N+1,) or Noff[0]!=0 or np.any(np.diff(Noff)<=0):raise ValueError("typed source ancestry offsets invalid")
payload={k:np.asarray(src[k]) for k in src.files}
payload.update({"face_source_identity_schema_by_original_id_json":schema_value,"face_group_id":group.astype(np.int64),"source_group_offsets":offsets,"source_group_face_rows":rows,"source_group_keys_json":np.asarray([json.dumps(keys,separators=(",",":"))]),"face_parent_source_face_rows":np.arange(N,dtype=np.int64),"face_exact_source_row":np.full(N,-1,np.int64),"lineage_preparation_input_sha256":np.asarray([sha(TYPED)]),"face_ancestry_method":np.asarray(["Adapter-local row grouping by the current candidate declared typed primary identity. Full flattened three-lane source-face ancestry is preserved unchanged; no exact original face or one-to-one overlap claim is made."])})
adapter=RUN/"typed-rebind-003/candidate-D-typed-697-prepared-003.npz"
np.savez_compressed(adapter,**payload)
# Read back and invoke the actual 697/217 entrypoint.
check=np.load(adapter,allow_pickle=False)
if not np.array_equal(check["faces"],src["faces"]) or not np.array_equal(check["points_world_m_f32"],src["points_world_m_f32"]):raise ValueError("adapter geometry changed")
spec=importlib.util.spec_from_file_location("handoff697_d003",HELPER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);owner=mod.load_owner()
t=time.monotonic();loaded=mod.load_surface_217(owner,adapter,"D",None)
if not np.array_equal(np.asarray(loaded["points"],np.float32),src["points_world_m_f32"]):raise ValueError("697 point bits changed")
if not np.array_equal(np.asarray(loaded["faces"],np.int64),src["faces"]):raise ValueError("697 face rows changed")
rows0=np.asarray(src["face_ancestry_source_face_rows"],np.int64);runs0=np.asarray(src["face_ancestry_source_original_ids"],np.int64);ids0=np.asarray(src["face_ancestry_parent_face_rows"],np.int64)
for i,(lo,hi) in enumerate(zip(Noff[:-1],Noff[1:])):
 if loaded["ancestry"][i]!=rows0[lo:hi].tolist() or loaded["ancestry_original_ids"][i]!=runs0[lo:hi].tolist() or loaded["ancestry_parent_rows"][i]!=ids0[lo:hi].tolist():raise ValueError("697 full ancestry mismatch at face "+str(i))
report={"schema":"h706.d-short-edge-quotient-run006-actual-697-d-load.v1","status":"PASS_ACTUAL_697_217_TYPED_D_LOAD","script_path":str(Path(__file__)),"script_sha256":sha(Path(__file__)),"typed_source_path":str(TYPED),"typed_source_sha256":sha(TYPED),"identity_schema_source_path":str(SCHEMA_SRC),"identity_schema_source_sha256":sha(SCHEMA_SRC),"adapter_path":str(adapter),"adapter_sha256":sha(adapter),"helper_path":str(HELPER),"helper_sha256":sha(HELPER),"owner_build_path":str(mod.BUILD),"owner_build_sha256":sha(mod.BUILD),"owner_build_expected_sha256":mod.BUILD_SHA,"points":int(len(loaded["points"])),"faces":int(len(loaded["faces"])),"loaded_ancestry_tuple_count":int(sum(map(len,loaded["ancestry"]))),"group_count":int(len(keys)),"used_point_count":int(len(np.unique(loaded["faces"]))),"elapsed_s":time.monotonic()-t,"physical_admission":False,"native_steps":0,"tetgen_run":False,"mass_mapping":"NOT_RUN","limitations":["The existing 697/217 loader accepted the D typed adapter. Group indices are adapter-local and full flattened source-face ancestry is retained; this does not assert exact per-face correspondence, volume quality, mass, TetGen admission, or physical/native acceptance."]}
out=RUN/"actual-697-D-load-003.json";out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"receipt":str(out),"sha256":sha(out),"adapter":str(adapter),"adapter_sha256":sha(adapter),"points":report["points"],"faces":report["faces"],"ancestry":report["loaded_ancestry_tuple_count"],"groups":report["group_count"],"elapsed_s":round(report["elapsed_s"],3)},sort_keys=True))

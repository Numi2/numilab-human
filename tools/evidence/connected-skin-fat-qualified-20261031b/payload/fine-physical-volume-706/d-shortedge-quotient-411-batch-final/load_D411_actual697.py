#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path(__file__).resolve().parents[1]
HELPER=H/"current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py"
HELPER_SHA="83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f"
TYPED=H/"d-shortedge-quotient-411-batch-final/typed-rebind-002/typed-owner/candidate-D-typed-current.npz"
TYPED_SHA="58408e6f77e916517b852b365c6245eac2b206a53048b5c95c4f2c29a97f84b5"
OUT=H/"d-shortedge-quotient-411-batch-final/typed-rebind-002/actual-697-D"
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(HELPER)==HELPER_SHA and sha(TYPED)==TYPED_SHA
spec=importlib.util.spec_from_file_location("handoff697_D411",HELPER)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
owner=mod.load_owner();start=time.monotonic()
s=mod.load_surface_217(owner,TYPED,"D",None)
pts=np.asarray(s["points"],np.float32);faces=np.asarray(s["faces"],np.int64)
assert pts.ndim==2 and pts.shape[1]==3 and faces.ndim==2 and faces.shape[1]==3 and np.isfinite(pts).all()
assert len(faces)>0 and faces.min()>=0 and faces.max()<len(pts)
for k in ("ancestry","ancestry_original_ids","ancestry_parent_rows"):
    assert len(s[k])==len(faces)
lens=np.asarray([len(x) for x in s["ancestry"]],np.int64)
assert lens.min()>0
receipt={
 "schema":"h706.d-shortedge-quotient-411-actual-697-D-load.v1",
 "status":"PASS_ACTUAL_697_217_LOAD_D411_TYPED",
 "helper_path":str(HELPER),"helper_sha256":sha(HELPER),
 "owner_build_path":str(mod.BUILD),"owner_build_sha256":sha(mod.BUILD),
 "typed_D_path":str(TYPED),"typed_D_sha256":sha(TYPED),
 "points":len(pts),"faces":len(faces),"loaded_ancestry_tuple_count":int(lens.sum()),
 "ancestry_tuple_minimum_per_face":int(lens.min()),
 "typed_identity_schema_count":len(s["typed_identity_schema"]),
 "exact_source_rows_nonnegative":int(np.count_nonzero(np.asarray(s["exact_source_rows"])>=0)),
 "used_point_count":int(len(np.unique(faces))),"elapsed_s":time.monotonic()-start,
 "physical_admission":False,"tetgen_run":False,"native_runtime_run":False,"mass_mapping":"NOT_RUN"
}
OUT.mkdir(parents=True,exist_ok=False)
path=OUT/"actual-697-D-load-receipt.json";path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":receipt["status"],"receipt":str(path),"sha256":sha(path),"points":receipt["points"],"faces":receipt["faces"],"ancestry_tuples":receipt["loaded_ancestry_tuple_count"]},sort_keys=True))

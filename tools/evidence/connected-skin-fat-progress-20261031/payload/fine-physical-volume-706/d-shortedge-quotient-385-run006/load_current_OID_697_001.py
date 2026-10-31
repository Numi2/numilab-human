#!/usr/bin/env python3
"""Invoke the pinned current 697/217 owner loader on the run006 D typed output and current O/I."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
H=Path(__file__).resolve().parents[1]
HELPER=H/"current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py"
HELPER_SHA="83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f"
roles={
 "O":(H/"joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz","4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20"),
 "I":(H/"paired-I-interface-392-root/qualification393001/typed365/candidate-I-typed-current.npz","31c9fdc18d4c1602a22f4d634843cc3aeb05d64822a9068942e50a999bbef720"),
 "D":(H/"d-shortedge-quotient-385-run006/typed-rebind-003/candidate-D-typed-current.npz","9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb"),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
if sha(HELPER)!=HELPER_SHA: raise ValueError("pinned 697 loader helper changed")
spec=importlib.util.spec_from_file_location("handoff697_run006",HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
owner=mod.load_owner()
result={}
for role,(p,want) in roles.items():
 if sha(p)!=want: raise ValueError(f"pinned typed {role} candidate changed")
 t=time.monotonic();s=mod.load_surface_217(owner,p,role,None)
 pts=np.asarray(s["points"],np.float32);faces=np.asarray(s["faces"],np.int64)
 if pts.ndim!=2 or pts.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(pts).all(): raise ValueError(f"{role} loaded shape/finiteness failure")
 if faces.size==0 or faces.min()<0 or faces.max()>=len(pts): raise ValueError(f"{role} loaded face point range failure")
 for k in ("ancestry","ancestry_original_ids","ancestry_parent_rows"):
  if len(s[k])!=len(faces): raise ValueError(f"{role} loaded {k} row count mismatch")
 lens=[len(x) for x in s["ancestry"]]
 if not lens or min(lens)<1: raise ValueError(f"{role} empty loaded ancestry")
 result[role]={"path":str(p),"sha256":sha(p),"points":len(pts),"faces":len(faces),"loaded_ancestry_tuple_count":sum(lens),"ancestry_kind_counts":{str(int(k)):int(v) for k,v in zip(*np.unique(s["ancestry_kind"],return_counts=True))},"typed_identity_schema_count":len(s["typed_identity_schema"]),"exact_source_rows_nonnegative":int(np.count_nonzero(np.asarray(s["exact_source_rows"])>=0)),"used_point_count":int(len(np.unique(faces))),"elapsed_s":time.monotonic()-t}
report={"schema":"h706.d-short-edge-quotient-run006-current-OID-actual-697-load.v1","status":"PASS_ACTUAL_697_217_LOAD_CURRENT_OID","helper_path":str(HELPER),"helper_sha256":sha(HELPER),"owner_build_path":str(mod.BUILD),"owner_build_sha256":sha(mod.BUILD),"owner_build_expected_sha256":mod.BUILD_SHA,"roles":result,"D_typed_binding_receipt_path":str(H/"d-shortedge-quotient-385-run006/typed-rebind-003/receipt.json"),"D_typed_binding_receipt_sha256":sha(H/"d-shortedge-quotient-385-run006/typed-rebind-003/receipt.json"),"physical_admission":False,"tetgen_run":False,"native_runtime_run":False,"mass_mapping":"NOT_RUN","limitations":["This invokes the existing 697/217 surface loader for current O, I, and D typed candidates only. It does not establish volume quality, mass transport, TetGen admission, or physical/native acceptance."]}
out=H/"d-shortedge-quotient-385-run006/actual-697-load-001.json"
out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"receipt":str(out),"sha256":sha(out),"roles":{r:{"points":v["points"],"faces":v["faces"],"ancestry":v["loaded_ancestry_tuple_count"],"elapsed_s":round(v["elapsed_s"],3)} for r,v in result.items()}},sort_keys=True))

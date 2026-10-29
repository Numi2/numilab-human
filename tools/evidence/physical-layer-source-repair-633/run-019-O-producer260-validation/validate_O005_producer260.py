import hashlib,importlib.util,json,sys,traceback
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
C=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-005-protected-cap-minthickness/output/O-protected-cap-f32.npz")
PROD=P/"bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py"
OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-019-O-producer260-validation/producer260-validation.json")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
OUT.parent.mkdir(parents=True,exist_ok=True)
r={"schema":"numi.O005-producer260-validation.v1","candidate":str(C),"candidate_sha256":sha(C),"producer":str(PROD),"producer_sha256":sha(PROD),"status":"ERROR"}
try:
 sys.path.insert(0,str(PROD.parent))
 spec=importlib.util.spec_from_file_location("producer260_O019",PROD);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 v=m.validate_candidate(C,"O",require_f32=True)
 r.update({"status":"PASS","layer":v["layer"],"role":v["role"],"vertexCount":v["vertexCount"],"faceCount":v["faceCount"],"sourceOriginalFaceIDs":v["sourceOriginalFaceIDs"],"faceAncestry":v["faceAncestry"]})
except BaseException as e:r.update({"error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
OUT.write_text(json.dumps(r,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")
print(json.dumps(r,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
if r["status"]!="PASS":raise SystemExit(1)

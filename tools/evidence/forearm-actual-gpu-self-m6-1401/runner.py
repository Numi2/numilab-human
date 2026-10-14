from pathlib import Path
import sys,json,hashlib,time,signal
import numpy as np
HP=Path("/Users/n/numi-human-final-integration-publish-1397")
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
HAND=Path("/Users/n/numi-human-fds132-m6-1350/attempt-471-native-material-handoff-001/result-001")
HERE=Path(__file__).resolve()
PINS=HERE.with_suffix(".pins.json")
OUT=D/"forearm470-actual-gpu-self-gate-001"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads(PINS.read_text())
for p,h in pins.items():
 if sha(p)!=h: raise ValueError("changed frozen input "+p)
sys.path.insert(0,str(HP/"src"))
from numilab_human import resting_anatomy_verify as v
for m in (v,v.ci):
 if str(Path(m.__file__).resolve()) not in pins: raise ValueError("unfrozen predicate import "+m.__file__)
OUT.mkdir()
report={"schema":"numi.human.forearm.step0.actual-forward-self-gate.v1","status":"running","geometry_basis":"Actual native GPU accepted step-0 capture; no CPU forward approximation.","pins":pins,"stages":[]}
def save():
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def timeout(a,b):raise TimeoutError("20-minute predicate budget")
signal.signal(signal.SIGALRM,timeout)
signal.alarm(1200)
try:
 surfaces,identity=v.load_capture(D/"joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.mrvpack",D/"joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.receipt.json")
 report["capture"]=identity
 save()
 for sid,name in ((132,"row-132-fds-material-handoff.npz"),(134,"row-134-fdp-material-handoff.npz")):
  p,f,n,cg=surfaces[(51005,sid)]
  with np.load(HAND/name,allow_pickle=False) as z:
   sf=z["faces_local"].astype(np.int64); gids=z["gids"]
   cfaces=cg[f].astype(np.int64)
   offset=int(cfaces[0,0]-sf[0,0])
   if not np.array_equal(cfaces,sf+offset):raise ValueError("all ordered facet corners disagree "+str(sid))
   active=cg.astype(np.int64)-offset
   if active.min()<0 or active.max()>=len(gids):raise ValueError("source index outside row")
   if not np.array_equal(np.unique(sf),active):raise ValueError("active source correspondence not bijective")
   sourceg=gids[active]
  np.savez_compressed(OUT/f"row-{sid}-actual-gpu.npz",positions=p,faces_local=f,normals=n,gids=sourceg,source_local_vertex_ids=active)
  print("PREPARE",sid,len(p),len(f),flush=True)
  t=time.monotonic()
  s=v.prepare_surface((51005,sid),p,f,normals=n,source_vertex_ids=sourceg)
  stage={"stage":"preparation","row":sid,"exact_full_ordered_face_mapping":True,"source_capture_offset":offset,"source_allocated_vertices":len(gids),"active_vertices":len(p),"positions_sha256":hashlib.sha256(p.tobytes()).hexdigest(),"diagnostics":s.diagnostics(),"seconds":time.monotonic()-t}
  report["stages"].append(stage);save()
  if not s.closed:report["status"]="failed_topology";save();break
  print("SELF_START",sid,flush=True);t=time.monotonic()
  result=s.self_audit()
  pairs=[]
  for a,b in result.get("triangle_pairs",[]):
   a,b=int(a),int(b)
   pairs.append({"face_indices":[a,b],"source_gids":[sourceg[f[a]].tolist(),sourceg[f[b]].tolist()],"active_vertex_indices":[f[a].tolist(),f[b].tolist()]})
  stage={"stage":"self","row":sid,"count":int(result["count"]),"pairs":pairs,"seconds":time.monotonic()-t}
  report["stages"].append(stage);save()
  print("SELF_DONE",sid,stage["count"],stage["seconds"],flush=True)
  if result["count"]:report["status"]="failed_self";save();break
 else:report["status"]="self_clear_actual_gpu_step0";save()
except Exception as e:
 report["status"]="error";report["error"]=str(e);save();raise
finally:signal.alarm(0)

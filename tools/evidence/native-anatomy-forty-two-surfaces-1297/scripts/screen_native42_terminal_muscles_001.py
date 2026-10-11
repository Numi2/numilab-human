from pathlib import Path
import sys,json,hashlib,time,signal,datetime
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
O=R/"anatomy-completion-1276/native42-terminal-muscle-self-screen-001"
P=R/"anatomy-completion-1276/forty-two-surface-native-composition-001/baseline/native-run/accepted-geometry/step-8000.mrvpack"
RECEIPT=P.with_suffix(".receipt.json")
TM=R/"anatomy-completion-1276/forty-two-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
S=Path("/Users/n/numi-human-self-separation-partial-resume-1267/src")
sys.path.insert(0,str(S))
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(p):sha(p) for p in (P,RECEIPT,TM,Path(__file__),S/"numilab_human/common_atlas_skin_clearance.py",S/"numilab_human/cardiac_cavity_intersections.py",S/"numilab_human/cardiac_cavity_geometry.py")}
assert pins[str(P)]=="e510380863a01a5954c51a35fcf9509526c4fb4350b2695d8c3d1653f7de958f"
assert pins[str(S/"numilab_human/cardiac_cavity_intersections.py")]=="423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd"
assert json.loads(RECEIPT.read_text())["accepted_step"]==8000
assert json.loads(RECEIPT.read_text())["pack_file_sha256"]==pins[str(P)]
rows=json.loads(TM.read_text())["source"]["surfaces"]; byid={int(r["stable_id"]):r for r in rows};assert len(byid)==150
O.mkdir(exist_ok=False)
start=time.monotonic()
def alarm(*_):raise TimeoutError("bounded 900-second CPU screen")
signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
keys={i:(51006 if r["layer"]=="tendon" else 51005,i) for i,r in byid.items()}
assert set(r["layer"] for r in byid.values())=={"muscle","tendon"}
pos,surfaces,inventory=ca._pack_surfaces(P,set(keys.values()))
out={"owner_script_path":'/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-diagnosis-1227/screen_all_captured_muscles_002.py',"owner_script_sha256":'e6a0f7ace0d2d2836d44f924d1caf557e1644c9b14085193dc051c702ae263b6',"scope":"Native42 terminal accepted capture8000 (16seconds) only: exact Float32-coordinate-quotient self and topology screen of all150 passive muscle/tendon rows. No intersurface audit, source-seam identity admission, later-pose clearance or clinical claim.","pins":pins,"surfaces":[],"complete":False}
def save():
 tmp=O/"report.tmp";tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");tmp.replace(O/"report.json")
save()
try:
 for sid,source in sorted(byid.items()):
  at=time.monotonic();sf=surfaces[keys[sid]];f=np.asarray(sf["faces"],int);used=np.unique(f);local=np.searchsorted(used,f);v=np.asarray(pos[used],np.float32)
  u,inv=np.unique(v,axis=0,return_inverse=True);qf=inv[local]
  t=analyze_topology(u.astype(float).tolist(),qf.tolist())
  r={"stable_id":sid,"label":source["label"],"layer":source["layer"],"source_member":source.get("member"),"captured_vertices":len(v),"quotient_vertices":len(u),"faces":len(f),"topology":{k:t[k] for k in ("closed_oriented_manifold_candidate","face_component_count","boundary_edge_count","euler_characteristic","degenerate_face_ids","duplicate_face_ids","vertex_manifold_defect_ids")} }
  try:
   rec=ca._exact_surface_records(u,qf)
   a=ci._audit_pair(rec,rec,same_surface=True)
   r["self_intersection_count"]=a["count"];r["triangle_pairs"]=a["triangle_pairs"];r["aabb_candidate_pairs"]=a["aabb_candidate_pairs"]
  except Exception as e:r["self_audit_error"]=type(e).__name__+": "+str(e)
  r["wall_seconds"]=time.monotonic()-at;out["surfaces"].append(r);out["wall_seconds"]=time.monotonic()-start;save()
  print(json.dumps({k:r.get(k) for k in ("stable_id","label","faces","self_intersection_count","self_audit_error","wall_seconds")}),flush=True)
 out["complete"]=True
except Exception as e:out["failure"]=type(e).__name__+": "+str(e)
finally:
 signal.alarm(0)
 out["wall_seconds"]=time.monotonic()-start;out["inputs_unchanged"]=all(sha(Path(p))==h for p,h in pins.items());save()
 print(json.dumps({"report":str(O/"report.json"),"complete":out["complete"],"count":len(out["surfaces"]),"wall_seconds":out["wall_seconds"],"inputs_unchanged":out["inputs_unchanged"]}),flush=True)
 if not out["complete"]:sys.exit(2)

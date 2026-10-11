from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"chest-shoulder-selected-pose-audit-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports={73:A/"pectoral-73-fixed-source-junction-reference-002/report.json",77:A/"existing-body-77-pinched-reference-001/report.json",78:A/"existing-body-78-pinched-reference-001/report.json",85:A/"existing-body-85-pinched-reference-001/report.json",86:A/"existing-body-86-pinched-reference-001/report.json"}
pins={str(p):sha(p) for p in [Path(__file__),H,T,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__),*reports.values()]}
for rp in reports.values():
 doc=json.loads(rp.read_text());assert doc["complete"] and doc["inputs_unchanged"]
 for p,v in doc["pins"].items():assert sha(p)==v;pins[p]=v
out={"scope":"Bounded Float32 reference-junction sensitivity at8 accepted early poses; old late captures omit required thoracic body25 and cannot establish these chest maps; no native candidate admission.","pins":pins,"trials":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
poses=[]
P=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"thirty-four-surface-native-composition-002/baseline/native-run")]:
 for step in steps:
  f=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(f)]=sha(f);rc=json.loads(f.read_text())
  p={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}
  poses.append((arm,step,p))
trials=[]
for sid,rp in reports.items():
 rr=json.loads(rp.read_text())
 tr=next(tr for tr in rr["trials"] if tr["microns"]==1 and "candidate_path" in tr)
 trials.append({**tr,"stable_id":sid,"maximum_source_displacement_m":tr["max_original_boundary_offset_m"],"bound_m":1e-6,"alpha":None})
for tr in trials:
 row=h.row_data(t,tr["stable_id"])
 cp=Path(tr["candidate_path"]);pins[str(cp)]=sha(cp);assert pins[str(cp)]==tr["candidate_sha256"];z=np.load(cp)
 pure=np.unique(row["faces"]);pure=pure[row["weights"][pure].max(1)>=.999999];missing=[]
 for j in pure:
  same=np.flatnonzero(np.all(z["vertices6"][:,:3]==row["positions"][j],axis=1))
  if not any(np.array_equal(z["weights"][i],row["weights"][j]) and np.array_equal(z["binding_indices"][i],row["local"][j]) for i in same):missing.append(int(j))
 trial={"stable_id":tr["stable_id"],"construction":{k:tr[k] for k in ("alpha","bound_m","maximum_source_displacement_m")},"microns":tr["microns"],"candidate_path":str(cp),"candidate_sha256":sha(cp),"pure_attachment_proxy_count":len(pure),"missing_pure_attachment_proxies":missing,"poses":[]};out["trials"].append(trial)
 for arm,step,p in poses:
  world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,p)
  rec,deg=h.exact_rows(world,z["faces"],ci);a=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None}
  u,iv=np.unique(world,axis=0,return_inverse=True);top=analyze_topology(u.astype(float).tolist(),iv[z["faces"]].tolist())
  trial["poses"].append({"arm":arm,"step":step,"self_count":a["count"],"degenerate":deg,"closed":top["closed_oriented_manifold_candidate"]})
 trial["all8_clear"]=not missing and all(p["closed"] and p["self_count"]==0 and not p["degenerate"] for p in trial["poses"])
 save();print(json.dumps({"sid":tr["stable_id"],"microns":tr["microns"],"all8_clear":trial["all8_clear"],"counts":[p["self_count"] for p in trial["poses"]],"missing_proxy_count":len(missing)}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


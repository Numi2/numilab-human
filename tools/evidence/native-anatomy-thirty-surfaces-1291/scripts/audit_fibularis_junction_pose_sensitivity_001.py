from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"fibularis-junction-pose-sensitivity-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,39)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rp=A/"fibularis-pinched-reference-001/report.json";r=json.loads(rp.read_text());assert r["complete"] and r["inputs_unchanged"]
pins={str(p):sha(p) for p in [Path(__file__),H,T,rp,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__)]}
out={"scope":"Bounded Float32 reference-junction sensitivity at24 accepted early/late poses; no native candidate admission.","pins":pins,"trials":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
poses=[]
P=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"twenty-eight-surface-native-composition-001/baseline/native-run"),("baseline",[0,47519,151999,152607,153215,153823,154431,155000],P/"baseline/native-run"),("intervention",[0,47519,151999,152607,153215,153823,154431,155000],P/"intervention/native-run")]:
 for step in steps:
  f=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(f)]=sha(f);rc=json.loads(f.read_text())
  p={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}
  poses.append((arm,step,p))
for tr in r["trials"]:
 if "candidate_path" not in tr:continue
 cp=Path(tr["candidate_path"]);pins[str(cp)]=sha(cp);assert pins[str(cp)]==tr["candidate_sha256"];z=np.load(cp)
 pure=np.unique(row["faces"]);pure=pure[row["weights"][pure].max(1)>=.999999];missing=[]
 for j in pure:
  same=np.flatnonzero(np.all(z["vertices6"][:,:3]==row["positions"][j],axis=1))
  if not any(np.array_equal(z["weights"][i],row["weights"][j]) and np.array_equal(z["binding_indices"][i],row["local"][j]) for i in same):missing.append(int(j))
 trial={"microns":tr["microns"],"candidate_path":str(cp),"candidate_sha256":sha(cp),"pure_attachment_proxy_count":len(pure),"missing_pure_attachment_proxies":missing,"poses":[]};out["trials"].append(trial)
 for arm,step,p in poses:
  world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,p)
  rec,deg=h.exact_rows(world,z["faces"],ci);a=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None}
  u,iv=np.unique(world,axis=0,return_inverse=True);top=analyze_topology(u.astype(float).tolist(),iv[z["faces"]].tolist())
  trial["poses"].append({"arm":arm,"step":step,"self_count":a["count"],"degenerate":deg,"closed":top["closed_oriented_manifold_candidate"]})
 trial["all24_clear"]=not missing and all(p["closed"] and p["self_count"]==0 and not p["degenerate"] for p in trial["poses"])
 save();print(json.dumps({"microns":tr["microns"],"all24_clear":trial["all24_clear"],"counts":[p["self_count"] for p in trial["poses"]],"missing_proxy_count":len(missing)}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"foot26-final-reference-posed-audit-015";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=A/"forty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
pins={str(p):sha(p) for p in [Path(__file__),H,T,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__)]}
out={"scope":"Source, eight early native poses, sixteen older paired poses and K1 terminal sensitivity pose. Reference inference only; full surrounding-target/native checks pending.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
poses=[]
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-surface-native-composition-001/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
  poses.append((arm,step,{int(r["body_index"]):(np.asarray(r["position_m"],np.float32),np.asarray(r["quaternion_xyzw"],np.float32)) for r in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}))
rp=A/"foot-26-local-range-sensitivity-014/report.json";search=json.loads(rp.read_text());assert search["complete"] and search["inputs_unchanged"];pins[str(rp)]=sha(rp)
selected=search["accepted_unadmitted_steps"][-1];cp=Path(selected["candidate_path"]);assert sha(cp)==selected["candidate_sha256"]
cps={26:cp}
for sid,cp in cps.items():
 row=h.row_data(t,sid);pins[str(cp)]=sha(cp);z=np.load(cp);p=z["vertices6"][:,:3];f=z["faces"]
 pure=np.unique(row["faces"]);pure=pure[row["weights"][pure].max(1)>=.999999];missing=[]
 for j in pure:
  same=np.flatnonzero(np.all(p==row["positions"][j],axis=1))
  if not any(np.array_equal(z["weights"][i],row["weights"][j]) and np.array_equal(z["binding_indices"][i],row["local"][j]) for i in same):missing.append(int(j))
 r={"stable_id":sid,"candidate_path":str(cp),"candidate_sha256":sha(cp),"pure_attachment_proxy_count":len(pure),"missing_pure_attachment_proxies":missing,"poses":[]};out["rows"].append(r)
 for arm,step,pose in [("source",None,None),*poses]:
  world=p if pose is None else h.forward(p,z["binding_indices"],z["weights"],row,t,pose)
  rec,deg=h.exact_rows(world,f,ci);audit=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
  u,iv=np.unique(world,axis=0,return_inverse=True);top=analyze_topology(u.astype(float).tolist(),iv[f].tolist())
  r["poses"].append({"arm":arm,"step":step,"self_count":audit["count"],"pairs":audit["triangle_pairs"],"degenerate":deg,"closed":top["closed_oriented_manifold_candidate"]});save()
 r["all_clear"]=not missing and all(x["self_count"]==0 and not x["degenerate"] and x["closed"] for x in r["poses"]);r["source_and24_K0_poses_clear"]=not missing and all(x["self_count"]==0 and not x["degenerate"] and x["closed"] for x in r["poses"] if x["arm"]!="K1");save();print(json.dumps({"sid":sid,"missing":missing,"all_clear":r["all_clear"],"counts":[x["self_count"] for x in r["poses"]]}),flush=True)
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save()

from pathlib import Path
import numpy as np,json,sys,importlib.util
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");D=A/"sartorius50-native-precision-repair-002";O=D/"pose-target-audit-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"left-scm-masseters-outward-native-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);r=h.row_data(t,50);C=D/"stable-50-unadmitted.npz";z=np.load(C);f=z["faces"];orig=z["face_origins"];pins={str(p):h.sha(p) for p in (Path(__file__),H,T,C,D/"report.json",Path(ci.__file__),Path(ca.__file__),Path(cg.__file__),h.INV)}
P=A.parent/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
contexts=[]
for label,folder,steps in [("current",A/"left-scm-masseters-outward-native-001/baseline/native-run/accepted-geometry",(0,4767,5023,5599,6207,6815,7423,8000)),("K1",A/"mtp-passive-native-sensitivity-one-001/baseline/native-run/accepted-geometry",(8000,)),("late-baseline",P/"baseline/native-run/accepted-geometry",(0,47519,151999,152607,153215,153823,154431,155000)),("late-intervention",P/"intervention/native-run/accepted-geometry",(0,47519,151999,152607,153215,153823,154431,155000)),("current-K2-partial",A/"current-anatomy-mtp-two-native-001/baseline/native-run/accepted-geometry",(0,4767,5023,5599,6207))]:
 for st in steps:contexts.append((label,folder/f"step-{st}.mrvpack"))
keys,*_=ca._load_target_inventory(h.INV)
def exact(p,f):
 u=np.unique(f);rec,dg=h.exact_rows(p[u],np.searchsorted(u,f),ci);assert not dg;return rec
out={"scope":"Current-source left sartorius precision edge-collapse across25 existing plus5 pre-failure K2 poses. Parent-face/target-face identity is compared exactly at identical poses; no existing interfaces are waived. K2 native wholebody run failed and remains unqualified.","pins":pins,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for label,pack in contexts:
 rp=pack.with_suffix(".receipt.json");rc=json.loads(rp.read_text());pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp);assert (rc.get("pack_file_sha256") or rc.get("accepted_pack_file_sha256"))==pins[str(pack)]
 pm={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
 want=set(keys)|({(51005,x) for x in (151,152,159,160)} if label in ("current","current-K2-partial") else set());pos,surfs,_=ca._pack_surfaces(pack,want)
 w=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],r,t,pm);bw=h.forward(r["positions"],r["local"],r["weights"],r,t,pm);rec=exact(w,f);br=exact(bw,r["faces"]);pairs=ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"];top=cg.analyze_topology(w.astype(float).tolist(),f.tolist())
 row={"label":label,"pack":str(pack),"self_pairs":pairs,"closed_oriented":top["closed_oriented_manifold_candidate"],"targets_scanned":0,"target_pairs":[]};out["poses"].append(row);lo=np.minimum(w.min(0),bw.min(0));hi=np.maximum(w.max(0),bw.max(0))
 for key,target in surfs.items():
  if key==(51005,50):continue
  row["targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,1)&np.all(tri.min(1)<=hi,1))
  if not len(sel):continue
  tr=exact(pos,tf[sel]);after=ci._audit_pair(rec,tr,same_surface=False)["triangle_pairs"];before=ci._audit_pair(br,tr,same_surface=False)["triangle_pairs"]
  ap={(int(orig[i]),int(sel[j])) for i,j in after};bp={(int(i),int(sel[j])) for i,j in before}
  if ap or bp:row["target_pairs"].append({"key":list(key),"candidate_parent_pairs":sorted(ap),"baseline_pairs":sorted(bp),"new_pairs":sorted(ap-bp),"removed_pairs":sorted(bp-ap)})
 save();print(json.dumps({"label":label,"step":pack.stem,"self":len(pairs),"closed":row["closed_oriented"],"targets":row["targets_scanned"],"new_pairs":sum(len(x["new_pairs"]) for x in row["target_pairs"])}),flush=True)
out.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),all30_self_closed_clear=all(not p["self_pairs"] and p["closed_oriented"] for p in out["poses"]),no_new_parent_target_pairs=all(not x["new_pairs"] for p in out["poses"] for x in p["target_pairs"]));save()

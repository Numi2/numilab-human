from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"left-chest-shoulder-reference-selection-023";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human.resting_anatomy_interface_patch import signed_volume
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=A/"thirty-eight-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
reports=[A/"pectoral-left-final-poses-022/report.json",A/"subscapularis-left-directed-poses-015/report.json"]
pins={str(p):sha(p) for p in [Path(__file__),H,T,Path(sys.modules[signed_volume.__module__].__file__),*reports]}
for rp in reports:
 d=json.loads(rp.read_text());assert d["complete"] and d["inputs_unchanged"]
 for p,v in d["pins"].items():assert sha(p)==v
 pins.update(d["pins"])
rows=[]
for rp in reports:
 for tr in json.loads(rp.read_text())["trials"]:
  if tr["stable_id"] not in (74,86):continue
  assert tr["all8_clear"] and not tr["missing_pure_attachment_proxies"]
  sid=tr["stable_id"];cp=Path(tr["candidate_path"]);z=np.load(cp);raw=h.row_data(t,sid);p=z["vertices6"][:,:3];f=z["faces"]
  u,iv=np.unique(p,axis=0,return_inverse=True);assert analyze_topology(u.astype(float).tolist(),iv[f].tolist())["closed_oriented_manifold_candidate"]
  area=lambda pp,ff:float(np.linalg.norm(np.cross(pp[ff[:,1]].astype(float)-pp[ff[:,0]],pp[ff[:,2]].astype(float)-pp[ff[:,0]]),axis=1).sum()/2)
  rows.append({"stable_id":sid,"candidate_path":str(cp),"candidate_sha256":sha(cp),"source_and_sampled_pose_self_clear":True,"pure_attachment_proxy_count":tr["pure_attachment_proxy_count"],"changed_or_missing_pure_attachment_proxies":[],"source_area_m2":area(raw["positions"],raw["faces"]),"candidate_area_m2":area(p,f),"source_signed_volume_m3":signed_volume(raw["positions"].astype(float),raw["faces"]),"candidate_signed_volume_m3":signed_volume(p.astype(float),f),"minimum_positive_weight":float(z["weights"][z["weights"]>0].min()),"poses":tr["poses"]})
assert {x["stable_id"] for x in rows}=={74,86}
out={"scope":"Positive-winding reference extraction with micrometre junction separation, bounded generated-boundary repair, and all original source attachment-proxy points and maps retained. Left pectoral temporary source F32 separation is explicitly reversed at original source points before final boundary-only repair. All8 full-body poses selfclear. Physical routes/mass/bindingtables unchanged; external/native qualification pending.","pins":pins,"rows":sorted(rows,key=lambda x:x["stable_id"]),"complete":True,"inputs_unchanged":all(sha(p)==v for p,v in pins.items())}
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps([{k:v for k,v in r.items() if k!="poses"} for r in out["rows"]],indent=2))
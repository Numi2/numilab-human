from pathlib import Path
import json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"foot25-current-reference-selection-013";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
rp=A/"foot25-local-reference-posed-audit-010/report.json";ip=A/"foot25-current-interface-audit-012/report.json"
r=json.loads(rp.read_text());i=json.loads(ip.read_text())
for d in (r,i):
 assert d["complete"] and d["inputs_unchanged"]
 for p,v in d["pins"].items():assert sha(p)==v
assert all(x["self_count"]==x["new_triangle_pair_count"]==0 and x["external_targets_scanned"]==859 for p in i["poses"] for x in p["rows"])
rr=r["rows"][0];assert rr["source_and24_K0_poses_clear"]
T=A/"forty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,25);z=np.load(rr["candidate_path"])
def area(p,f):return float(np.linalg.norm(np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]]),axis=1).sum()/2)
def vol(p,f):return float(np.einsum("ij,ij->i",p[f[:,0]].astype(float),np.cross(p[f[:,1]].astype(float),p[f[:,2]].astype(float))).sum()/6)
pins={**r["pins"],**i["pins"],**{str(p):sha(p) for p in [Path(__file__),H,rp,ip]}}
out={"scope":"Right flexor digitorum longus reference repair; source and24 K0 retained poses self clear. K1 alternate stiffness configuration explicitly fails3 self pairs and is not qualified. Current8-pose surrounding-target comparison passes. Native pending.","pins":pins,"rows":[{**rr,"source_and_sampled_pose_self_clear":rr["source_and24_K0_poses_clear"],"changed_or_missing_pure_attachment_proxies":rr["missing_pure_attachment_proxies"],"source_area_m2":area(row["positions"],row["faces"]),"candidate_area_m2":area(z["vertices6"][:,:3],z["faces"]),"source_signed_volume_m3":vol(row["positions"],row["faces"]),"candidate_signed_volume_m3":vol(z["vertices6"][:,:3],z["faces"])}],"complete":True,"inputs_unchanged":all(sha(p)==v for p,v in pins.items())}
(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out["rows"][0].items() if k!="poses"}))

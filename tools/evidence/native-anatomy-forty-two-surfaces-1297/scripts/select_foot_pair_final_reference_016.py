from pathlib import Path
import json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"foot-pair-final-reference-selection-016";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"forty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
pins={str(p):sha(p) for p in [Path(__file__),H,T]}
out={"scope":"Both long toe-flexors source and25 retained poses including K1 sensitivity clear. Explicit passive reference geometry inference preserving original onehot attachment proxies. External/native checks pending.","pins":pins,"rows":[],"complete":False}
def area(p,f):return float(np.linalg.norm(np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]]),axis=1).sum()/2)
def vol(p,f):return float(np.einsum("ij,ij->i",p[f[:,0]].astype(float),np.cross(p[f[:,1]].astype(float),p[f[:,2]].astype(float))).sum()/6)
for sid in (25,26):
 rp=A/f"foot{sid}-final-reference-posed-audit-015/report.json";r=json.loads(rp.read_text());assert r["complete"] and r["inputs_unchanged"]
 for p,v in r["pins"].items():assert sha(p)==v
 pins.update(r["pins"]);pins[str(rp)]=sha(rp)
 rr=r["rows"][0];assert rr["all_clear"] and len(rr["poses"])==26
 row=h.row_data(t,sid);z=np.load(rr["candidate_path"])
 origin=A/f"foot-{sid}-fixed-source-junction-reference-002/stable-{sid}-junction-1-microns.npz";pins[str(origin)]=sha(origin)
 delta=np.linalg.norm(z["vertices6"][:,:3].astype(float)-np.load(origin)["vertices6"][:,:3],axis=1).max()
 assert delta<.000101
 out["rows"].append({**rr,"source_and_sampled_pose_self_clear":rr["all_clear"],"changed_or_missing_pure_attachment_proxies":rr["missing_pure_attachment_proxies"],"source_area_m2":area(row["positions"],row["faces"]),"candidate_area_m2":area(z["vertices6"][:,:3],z["faces"]),"source_signed_volume_m3":vol(row["positions"],row["faces"]),"candidate_signed_volume_m3":vol(z["vertices6"][:,:3],z["faces"]),"maximum_displacement_from_fan_separated_reference_m":float(delta)})
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()))
(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"complete":True,"candidates":[{k:r[k] for k in ("stable_id","candidate_sha256","maximum_displacement_from_fan_separated_reference_m")} for r in out["rows"]]}))

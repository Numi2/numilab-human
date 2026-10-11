from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"fibularis-reference-selection-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy_interface_patch import signed_volume
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
pins={str(p):sha(p) for p in [Path(__file__),H,T]};rows=[]
for sid,stem in [(39,"fibularis"),(40,"fibularis-left")]:
 rp=A/f"{stem}-junction-pose-sensitivity-001/report.json";d=json.loads(rp.read_text());assert d["complete"] and d["inputs_unchanged"];pins[str(rp)]=sha(rp)
 for path,digest in d["pins"].items():assert sha(path)==digest
 pins.update(d["pins"])
 tr=next(x for x in d["trials"] if x["microns"]==1);assert tr["all24_clear"] and not tr["missing_pure_attachment_proxies"]
 srcp=A/f"{stem}-pinched-reference-001/report.json";src=json.loads(srcp.read_text());assert src["complete"] and src["inputs_unchanged"];pins[str(srcp)]=sha(srcp)
 for path,digest in src["pins"].items():assert sha(path)==digest
 pins.update(src["pins"]);selected=next(x for x in src["trials"] if x["microns"]==1)
 cp=Path(tr["candidate_path"]);assert sha(cp)==selected["candidate_sha256"]==tr["candidate_sha256"]
 z=np.load(cp);raw=h.row_data(t,sid);p=raw["positions"];f=raw["faces"];np0=z["vertices6"][:,:3];nf=z["faces"]
 area=lambda pp,ff:float(np.linalg.norm(np.cross(pp[ff[:,1]].astype(float)-pp[ff[:,0]],pp[ff[:,2]].astype(float)-pp[ff[:,0]]),axis=1).sum()/2)
 top=analyze_topology(np0.astype(float).tolist(),nf.tolist());assert top["closed_oriented_manifold_candidate"]
 rows.append({"stable_id":sid,"candidate_path":str(cp),"candidate_sha256":sha(cp),"source_and_sampled_pose_self_clear":True,"pure_attachment_proxy_count":tr["pure_attachment_proxy_count"],"changed_or_missing_pure_attachment_proxies":[],"selected_junction_retreat_um":1,"maximum_original_boundary_offset_m":selected["max_original_boundary_offset_m"],"source_face_count":len(f),"candidate_face_count":len(nf),"source_area_m2":area(p,f),"candidate_area_m2":area(np0,nf),"source_signed_volume_m3":signed_volume(p.astype(float),f),"candidate_signed_volume_m3":signed_volume(np0.astype(float),nf),"poses":tr["poses"]})
out={"scope":"Explicit positive-winding reference inference followed by local fan separation of two point junctions. Four junction vertices retreat toward their own fan-neighbor centroids by1micrometre; all pure attachment proxies preserved. Exact source and24 retained posed self/topology checks clear. Geometry sensitivity covers0.1/1/10micrometres on both sides. Not anatomical or native admission.","pins":pins,"rows":rows,"complete":True,"inputs_unchanged":all(sha(p)==v for p,v in pins.items())}
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps([{k:v for k,v in r.items() if k not in ("poses",)} for r in rows],indent=2))


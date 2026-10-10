from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"deltoid-head-source-interface-diagnosis-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),T,T.with_suffix(".manifest.json"),H,Path(ci.__file__)]}
t=h.load_tissue(T);manifest=json.loads(T.with_suffix(".manifest.json").read_text())
out={"scope":"Source-coordinate full mutual-pair localization only, not interface admission. Source frames are atlas coordinates; accepted-world checks would be required.","pins":pins,"rows":[]}
for sa,sb in ((75,79),(76,80)):
 ra=h.row_data(t,sa);rb=h.row_data(t,sb)
 aa,da=h.exact_rows(ra["positions"],ra["faces"],ci);bb,db=h.exact_rows(rb["positions"],rb["faces"],ci);assert not da and not db
 v=ci._audit_pair(aa,bb,same_surface=False)
 ps=np.array(v["triangle_pairs"],int).reshape(-1,2)
 r={"stable_ids":[sa,sb],"source_intersection_count":v["count"],"pairs":ps.tolist(),"source_metadata":[next(x for x in manifest["source"]["surfaces"] if x["stable_id"]==sid) for sid in (sa,sb)],"bounds":[]}
 for index,row in enumerate((ra,rb)):
  p=row["positions"];used=np.unique(row["faces"]);q=p[np.unique(row["faces"][ps[:,index]])]
  r["bounds"].append({"stable_id":(sa,sb)[index],"surface_min_m":p[used].min(0).tolist(),"surface_max_m":p[used].max(0).tolist(),"pair_vertices_min_m":q.min(0).tolist(),"pair_vertices_max_m":q.max(0).tolist(),"pair_face_count":len(np.unique(ps[:,index]))})
 out["rows"].append(r);(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:v for k,v in r.items() if k not in ("source_metadata","pairs")}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

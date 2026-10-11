from pathlib import Path
import json,hashlib
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"native45-terminal-muscle-inherited-screen-001";O.mkdir(exist_ok=False)
P=A/"native44-terminal-muscle-inherited-screen-001/report.json";V=A/"forty-five-surface-native-composition-001/native-verification-001/report.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(P.read_text());v=json.loads(V.read_text());assert p["complete"] and v["complete"] and v["inputs_unchanged"] and v["all_prior44_and_skin_geometry_exact"] and v["all_native_pair_checks_pass"]
pack=A/"forty-four-surface-native-composition-002/baseline/native-run/accepted-geometry/step-8000.mrvpack"
assert v["pins"][str(pack)]==sha(pack)
parent_verification=Path(next(k for k in p["pins"] if k.endswith("native-verification-001/report.json")))
assert sha(parent_verification)==p["pins"][str(parent_verification)]
pv=json.loads(parent_verification.read_text());assert pv["pins"][str(pack)]==sha(pack)
pose=next(r for r in v["poses"] if r["accepted_step"]==8000)
assert pose["unchanged_surface_referenced_triangles_exact_parent"]==859
fixed=next(r for r in pose["rows"] if r["stable_id"]==133);assert fixed["self_count"]==0 and fixed["closed_oriented_manifold_candidate"]
rows=[]
for r in p["surfaces"]:
 sid=r["stable_id"]
 if sid==133:rows.append({"stable_id":sid,"label":r["label"],"self_intersection_count":0,"closed":True,"basis":"Direct native45 exact self/topology audit","native_audit":fixed})
 else:rows.append({"stable_id":sid,"label":r["label"],"self_intersection_count":r["self_intersection_count"],"closed":r["closed"],"basis":"Native44 screen inherited through native45 byte-exact referenced triangle coordinates"})
assert len(rows)==150
remaining=[r for r in rows if r["self_intersection_count"] or (not r["closed"] and r["stable_id"] not in (7,8))]
out={"scope":"Terminal16s pose only. Other149 muscle/tendon surfaces inherit exact native44 predicates through native45 triangle-coordinate identity; stable133 directly audited on native45. Not all-pose or whole-body clearance. Achilles7/8 open attachment boundaries remain separately declared; not newly excused.","pins":{str(P):sha(P),str(V):sha(V),str(pack):sha(pack),str(Path(__file__)):sha(Path(__file__))},"surfaces":rows,"self_clear_closed_count":sum(r["closed"] and r["self_intersection_count"]==0 for r in rows),"self_clear_declared_open_achilles_count":sum(r["stable_id"] in (7,8) and r["self_intersection_count"]==0 for r in rows),"remaining_terminal_defects":remaining,"complete":True,"inputs_unchanged":True}
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"closed_self_clear":out["self_clear_closed_count"],"declared_open_self_clear":out["self_clear_declared_open_achilles_count"],"remaining_ids":[r["stable_id"] for r in remaining]}))

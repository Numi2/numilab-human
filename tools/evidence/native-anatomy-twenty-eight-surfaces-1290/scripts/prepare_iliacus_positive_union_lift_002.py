from pathlib import Path
import sys,json,hashlib,importlib.util,inspect,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"iliacus-positive-union-lift-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
h.TISS=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";h.MAN=h.TISS.with_suffix(".manifest.json")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
h.EXPECTED_TISS_SHA=sha(h.TISS);h.EXPECTED_MAN_SHA=sha(h.MAN)
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__)]}
# Reuse the proved exact ancestry/barycentric lift with row identity parameterized.
# No change to the construction, averaging or Float32 admission rules.
src=inspect.getsource(h.make_candidate).replace('require(row["bc"]==2,"gluteus row binding cardinality changed")','require(1<=row["bc"]<=4,"row exceeds four-slot existing binding format")').replace('expected_member={29:"FJ1418",30:"FJ1418M"}[sid]','expected_member=source_manifest["member_id"]')
lp=O/"parameterized_existing_lift.py";lp.write_text(src);pins[str(lp)]=sha(lp)
exec(compile(src,str(lp),"exec"),h.__dict__);h.OUT=O
t=h.load_tissue(h.TISS);manifest=json.loads(h.MAN.read_text())
geometry={"manifest_rows":{int(x["stable_id"]):x for x in manifest["source"]["surfaces"]},"analyze_topology":analyze_topology}
out={"scope":"Explicit strictly-positive winding reference inference and existing canonical ancestry lift on bilateral iliacus; negative exterior folds excluded; no physical route or attachment edit. Reference-derived geometry, not measurement or native admission.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def audit(p,f):
 records,deg=h.exact_rows(p,f,ci)
 return {"count":ci._audit_pair(records,records,same_surface=True)["count"] if not deg else None,"degenerate_faces":deg}
for sid in [37,38]:
 at=time.monotonic();h.UNION_DIR=A/f"existing-body-{sid}-positive-union-002";up=h.UNION_DIR/f"stable-{sid}-exact-union.json";rp=h.UNION_DIR/"result.json"
 pins[str(up)]=sha(up);pins[str(rp)]=sha(rp)
 assert json.loads(rp.read_text())["source_geometry_clear"]
 row=h.row_data(t,sid);assert np.all(row["weights"][row["weights"]>0]>2e-6)
 c=h.make_candidate(sid,t,row,json.loads(up.read_text()),ci,geometry)
 assert np.all(c["weights"][c["weights"]>0]>2e-6)
 # Keep face-source ancestry in the existing optional patch array.
 np.savez(c["candidate_npz"],vertices6=c["vertices6"],binding_indices=c["local"],weights=c["weights"],faces=c["faces"].astype("<u4"),face_origins=np.asarray(c["parents"],int))
 cr={"material_rule":"strictly_positive","stable_id":sid,"candidate_path":str(c["candidate_npz"]),"candidate_sha256":sha(c["candidate_npz"]),"source_union_and_attribute_report":c["report"],"poses":[]}
 cr["source_union_and_attribute_report"]["source_union"]["candidate_npz_sha256"]=sha(c["candidate_npz"])
 cr["source_union_and_attribute_report"]["source_union"]["npz_fields"].append("face_origins")
 used=np.unique(row["faces"]);pure=used[row["weights"][used].max(1)>=.999999]
 missing=[]
 for j in pure:
  matches=np.flatnonzero(np.all(c["positions"]==row["positions"][j],axis=1))
  if not any(np.array_equal(c["weights"][i],row["weights"][j]) and np.array_equal(c["local"][i],row["local"][j]) for i in matches):missing.append(int(j))
 cr["pure_attachment_proxy_count"]=len(pure);cr["changed_or_missing_pure_attachment_proxies"]=missing;assert not missing
 out["rows"].append(cr);save()
 for arm in ["baseline","intervention"]:
  for step in h.STEPS:
   p=h.PAIR/arm/"native-run/accepted-geometry"/f"step-{step}.receipt.json";pins[str(p)]=sha(p)
   pose={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in json.loads(p.read_text())["accepted_registered_body_poses"]}
   world=h.forward(c["positions"],c["local"],c["weights"],row,t,pose)
   cr["poses"].append({"arm":arm,"step":step,**audit(world,c["faces"])})
   save()
 cr["source_and_sampled_pose_self_clear"]=all(x["count"]==0 and not x["degenerate_faces"] for x in cr["poses"])
 cr["wall_seconds"]=time.monotonic()-at;save()
 print(json.dumps({"stable_id":sid,"all16_clear":cr["source_and_sampled_pose_self_clear"],"counts":[x["count"] for x in cr["poses"]],"wall_seconds":cr["wall_seconds"]}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()

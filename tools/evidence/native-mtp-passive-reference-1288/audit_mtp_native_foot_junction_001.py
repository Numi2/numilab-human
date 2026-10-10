from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"mtp-native-foot-junction-audit-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
M=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad/bodyparts3d-myosim-major-bones.manifest.json")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
anchors=json.load(open(M))["source"]["anchors"];pins={str(p):sha(p) for p in [M,H,Path(__file__),Path(ca.__file__),Path(ci.__file__),h.INV]}
out={"scope":"Exact captured bone-surface triangle intersections at the forefoot-to-toe anatomical junction under MTP stiffness sensitivity. Counts are not penetration depths and no anatomical interface is waived.","pins":pins,"runs":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
keys,*_=ca._load_target_inventory(h.INV)
for arm in ("zero","half","one","two"):
 D=A/f"mtp-passive-native-sensitivity-{arm}-001";ep=D/"baseline/execution.json";e=json.load(open(ep));assert e["returncode"]==0 and not e["changed_inputs"];pins[str(ep)]=sha(ep)
 run={"arm":arm,"poses":[]};out["runs"].append(run)
 for step in ((0,8000) if arm=="zero" else (8000,)):
  pack=D/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json");pins[str(pack)]=sha(pack);pins[str(rp)]=sha(rp)
  p,s,_=ca._pack_surfaces(pack,keys);pose={"step":step,"pairs":[],"count":0};run["poses"].append(pose)
  for side in ("r","l"):
   calc=[i+1 for i,a in enumerate(anchors) if a["myosim_body"]=="calcn_"+side]
   toe=[i+1 for i,a in enumerate(anchors) if a["myosim_body"]=="toes_"+side];rec={};bounds={}
   for sid in calc+toe:
    f=np.asarray(s[(51004,sid)]["faces"],int);used=np.unique(f);v=p[used];ff=np.searchsorted(used,f);rr,deg=h.exact_rows(v,ff,ci);assert not deg
    rec[sid]=rr;bounds[sid]=(v.min(0),v.max(0))
   for c in calc:
    for t in toe:
     cl,ch=bounds[c];tl,th=bounds[t]
     if np.any(ch<tl) or np.any(th<cl):continue
     audit=ci._audit_pair(rec[c],rec[t],same_surface=False)
     if audit["count"]:
      pose["pairs"].append({"side":side,"calcn_stable_id":c,"calcn_member_id":anchors[c-1]["member_id"],"toe_stable_id":t,"toe_member_id":anchors[t-1]["member_id"],"count":audit["count"],"face_pairs":audit["triangle_pairs"]});pose["count"]+=audit["count"]
  save();print(json.dumps({"arm":arm,"step":step,"junction_pair_count":pose["count"],"pair_counts":[[x["calcn_member_id"],x["toe_member_id"],x["count"]] for x in pose["pairs"]]}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save();assert out["inputs_unchanged"]


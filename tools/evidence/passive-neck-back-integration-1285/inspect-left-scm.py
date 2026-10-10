from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"left-scm-registration-inspection-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=R/"passive-neck-back-coverage-1281/candidate-002/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
P=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry/step-0.mrvpack"
RP=P.with_suffix(".receipt.json")
AUD=A/"neck-back-retained-pose-audit-002/report.json"
audit=json.loads(AUD.read_text());rr=next(r for r in audit["poses"][0]["rows"] if r["stable_id"]==152)
rc=json.loads(RP.read_text())
poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
t=h.load_tissue(T);row=h.row_data(t,152);w=h.forward(row["positions"],row["local"],row["weights"],row,t,poses)
keys,*_=ca._load_target_inventory(h.INV);p,s,_=ca._pack_surfaces(P,keys)
lo=w.min(0)-.02;hi=w.max(0)+.02
fig,axs=plt.subplots(1,3,figsize=(18,7))
for ax,(a,b) in zip(axs,[(0,1),(0,2),(2,1)]):
 for item in rr["interfaces"]:
  key=(item["semantic"],item["stable_id"]);f=s[key]["faces"]
  if key[0]==51004:
   tri=p[f];chosen=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
   ax.add_collection(PolyCollection(p[f[chosen]][:,:,[a,b]]*1000,facecolors="#d5d7dc",edgecolors="#777777",linewidths=.12,alpha=.45))
  if key[0]==51007:
   ids=np.unique([y for _,y in item["source_target_full_face_pairs"]])
   ax.add_collection(PolyCollection(p[f[ids]][:,:,[a,b]]*1000,facecolors="#29a86a",edgecolors="#29a86a",alpha=.45,linewidths=.2))
 f=row["faces"]
 ax.add_collection(PolyCollection(w[f][:,:,[a,b]]*1000,facecolors="#315dca",edgecolors="#315dca",linewidths=.1,alpha=.12))
 hit=np.unique([x for item in rr["interfaces"] for x,_ in item["source_target_full_face_pairs"]])
 ax.add_collection(PolyCollection(w[f[hit]][:,:,[a,b]]*1000,facecolors="#e13e48",edgecolors="#e13e48",linewidths=.12,alpha=.45))
 ax.set_xlim(lo[a]*1000,hi[a]*1000);ax.set_ylim(lo[b]*1000,hi[b]*1000)
 ax.set_aspect("equal");ax.set_xlabel("world "+"xyz"[a]+" (mm)");ax.set_ylabel("world "+"xyz"[b]+" (mm)")
 ax.grid(alpha=.15)
fig.suptitle("Unadmitted left sternocleidomastoid: retained accepted pose 0\nBlue muscle; red crossing muscle faces; gray bone; green crossing skin faces. Offline geometric projection.")
fig.tight_layout();fig.savefig(O/"left-scm-three-views.png",dpi=150);plt.close(fig)
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
(O/"report.json").write_text(json.dumps({"scope":"Offline diagnostic, not native anatomy admission.","input_sha256":{str(x):sha(x) for x in [Path(__file__),H,T,RP,P]},"audited_row_snapshot":rr,"world_bounds_m":[w.min(0).tolist(),w.max(0).tolist()],"png_sha256":sha(O/"left-scm-three-views.png")},indent=2)+"\n")
print(O/"left-scm-three-views.png")

from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"neck-back-retained-pose-audit-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=R/"passive-neck-back-coverage-1281/candidate-002/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
N=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),Path(ci.__file__),Path(ca.__file__),h.INV]}
assert pins[str(T)]=="08c534229c322391ce5c7fb972810ace13bc07cf9d95ff18825561ac2f792ada"
assert pins[str(T.with_suffix(".manifest.json"))]=="32fcc0ba0d7df7cdc478d15ba7bd8d50753077686ae3ddc90234c736e414a861"
t=h.load_tissue(T)
rows={sid:h.row_data(t,sid) for sid in range(151,159)}
# Native consumes every strictly positive sparse weight. The historical offline
# helper skipped <=2e-6; retain its exact derived source with that one correction.
forward_source=H.read_text()
assert forward_source.count("if w<=f(2e-6): continue")==1
derived=O/"forward_all_positive_weights.py"
derived.write_text(forward_source.replace("if w<=f(2e-6): continue","if w<=f(0): continue"))
sp2=importlib.util.spec_from_file_location("all_positive",derived)
hp=importlib.util.module_from_spec(sp2);sp2.loader.exec_module(hp)
pins[str(derived)]=sha(derived)
keys,*_=ca._load_target_inventory(h.INV)
out={"scope":"Unadmitted passive neck/back source mapped through complete retained accepted poses. Exact F32 self and full-row external intersections; no exemption or native candidate qualification.","pins":pins,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def exact(p,f):
 used=np.unique(f);records,deg=h.exact_rows(p[used],np.searchsorted(used,f),ci)
 assert not deg,deg
 return records
for step in [0,8000]:
 at=time.monotonic()
 pack=N/f"step-{step}.mrvpack";receipt=pack.with_suffix(".receipt.json")
 pins[str(pack)]=sha(pack);pins[str(receipt)]=sha(receipt)
 rc=json.loads(receipt.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
 poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 assert len(poses)==157
 pos,surfaces,_=ca._pack_surfaces(pack,keys)
 surfaces={k:v for k,v in surfaces.items() if k[0]!=51999}
 assert len(surfaces)==860,len(surfaces)
 pr={"accepted_step":step,"body_pose_count":len(poses),"rows":[]};out["poses"].append(pr)
 worlds={sid:hp.forward(row["positions"],row["local"],row["weights"],row,t,poses) for sid,row in rows.items()}
 for sid in [152,151,153,154,155,156,157,158]:
  row=rows[sid];world=worlds[sid];faces=row["faces"]
  full=exact(world,faces)
  rr={"stable_id":sid,"self_intersection_count":ci._audit_pair(full,full,same_surface=True)["count"],
      "closed_oriented":analyze_topology(world.tolist(),faces.tolist())["closed_oriented_manifold_candidate"],
      "external_targets_scanned":0,"external_triangle_pair_count":0,"interfaces":[]}
  pr["rows"].append(rr)
  lo=world.min(0);hi=world.max(0)
  for key,target in surfaces.items():
   rr["external_targets_scanned"]+=1
   tf=np.asarray(target["faces"],int);tri=pos[tf]
   select=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
   if not len(select):continue
   audit=ci._audit_pair(full,exact(pos,tf[select]),same_surface=False)
   if not audit["count"]:continue
   rr["external_triangle_pair_count"]+=audit["count"]
   rr["interfaces"].append({"semantic":key[0],"stable_id":key[1],"body":target["body"],
        "count":audit["count"],"source_target_full_face_pairs":[[x,int(select[y])] for x,y in audit["triangle_pairs"]]})
  # New rows also need mutual checks; these are not in the retained scene.
  rr["new_row_interfaces"]=[]
  for other in range(151,159):
   if other==sid:continue
   audit=ci._audit_pair(full,exact(worlds[other],rows[other]["faces"]),same_surface=False)
   if audit["count"]:rr["new_row_interfaces"].append({"stable_id":other,"count":audit["count"],"full_face_pairs":audit["triangle_pairs"]})
  save();print(json.dumps({k:rr[k] for k in ["stable_id","self_intersection_count","external_targets_scanned","external_triangle_pair_count"]}|{"step":step}),flush=True)
 pr["wall_seconds"]=time.monotonic()-at;save()
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()

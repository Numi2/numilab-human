from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
O=A/"external-oblique-local-interface-audit-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=A/"twenty-two-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
N=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry"
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),h.INV,Path(ci.__file__),Path(ca.__file__)]}
t=h.load_tissue(T);rp=A/"external-oblique-local-reference-003/report.json"
pins[str(rp)]=sha(rp);doc=json.loads(rp.read_text());assert doc["complete"] and doc["inputs_unchanged"]
r=doc["rows"][0];trial=next(x for x in r["trials"] if x["alpha"]==.1)
C=Path(trial["candidate_path"]);pins[str(C)]=sha(C);assert pins[str(C)]==trial["candidate_sha256"]
z=np.load(C);sid=67;row=h.row_data(t,sid);orig=z["face_origins"]
assert np.array_equal(orig,np.arange(len(row["faces"])))
changed=np.flatnonzero(np.any(row["positions"][row["faces"]]!=z["vertices6"][:,:3][z["faces"]],axis=(1,2)))
keys,*_=ca._load_target_inventory(h.INV)
out={"scope":"Unadmitted external-oblique candidate full external-target changed-star check at two retained native poses. Source self and actual-forward endpoint self verified separately; complete native cycle remains required.","pins":pins,"stable_id":sid,"changed_source_face_ids":changed.tolist(),"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
def exact(p,f):
 used=np.unique(f);rec,deg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not deg;return rec
for step in (0,8000):
 at=time.monotonic();pk=N/f"step-{step}.mrvpack";rp=pk.with_suffix(".receipt.json")
 pins[str(pk)]=sha(pk);pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pk)]
 poses={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pk,keys)
 world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,poses)
 sf=np.asarray(surfs[(51005,sid)]["faces"],int);d=sf-row["faces"];assert np.all(d==d.flat[0])
 actual=pos[int(d.flat[0]):int(d.flat[0])+row["vc"]]
 forward=h.forward(row["positions"],row["local"],row["weights"],row,t,poses)
 used=np.unique(row["faces"]);error=float(np.linalg.norm(actual[used].astype(float)-forward[used],axis=1).max())
 bf=row["faces"][changed];cf=z["faces"][changed];points=np.concatenate([actual[np.unique(bf)],world[np.unique(cf)]])
 lo=points.min(0);hi=points.max(0)
 base=exact(actual,bf);cand=exact(world,cf)
 pr={"step":step,"max_referenced_forward_error_m":error,"external_targets_scanned":0,"new_triangle_pair_count":0,"interfaces":[]}
 out["poses"].append(pr)
 for key,target in surfs.items():
  if key==(51005,sid) or key[0]==51999:continue
  pr["external_targets_scanned"]+=1
  tf=np.asarray(target["faces"],int);tri=pos[tf]
  select=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
  if not len(select):continue
  tar=exact(pos,tf[select]);ba=ci._audit_pair(base,tar,same_surface=False);aa=ci._audit_pair(cand,tar,same_surface=False)
  if not ba["count"] and not aa["count"]:continue
  bp={(int(changed[x]),int(select[y])) for x,y in ba["triangle_pairs"]}
  ap={(int(changed[x]),int(select[y])) for x,y in aa["triangle_pairs"]}
  new=sorted(ap-bp);pr["new_triangle_pair_count"]+=len(new)
  pr["interfaces"].append({"semantic":key[0],"stable_id":key[1],"before":len(bp),"after":len(ap),"added_lineage_pairs":new,"removed_lineage_pairs":sorted(bp-ap)})
 assert pr["external_targets_scanned"]==859
 pr["wall_seconds"]=time.monotonic()-at;save();print(json.dumps(pr),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()

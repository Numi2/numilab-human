from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
O=R/"anatomy-completion-1276/acromial-deltoid-local-clearance-allpose-audit-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=R/"anatomy-completion-1276/neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,Path(ci.__file__),Path(ca.__file__)]}
reports=[R/"anatomy-completion-1276"/x/"report.json" for x in ["acromial-deltoid-local-reference-003"]]
candidates={}
t=h.load_tissue(h.TISS)
for rp in reports:
 pins[str(rp)]=sha(rp)
 doc=json.loads(rp.read_text());assert doc["complete"] and doc["inputs_unchanged"]
 for row_report in doc["rows"]:
  sid=row_report["stable_id"]
  r=next(x for x in row_report["trials"] if x["alpha"]==.1)
  p=Path(r["candidate_path"]);pins[str(p)]=sha(p)
  assert pins[str(p)]==r["candidate_sha256"]
  z=np.load(p);row=h.row_data(t,sid)
  origins=z["face_origins"].astype(int)
  unique,counts=np.unique(origins,return_counts=True)
  changed=(set(range(len(row["faces"])))-set(map(int,origins)))|set(map(int,unique[counts!=1]))
  # Oriented cyclic permutation is the same triangle. Check source attributes
  # as well, so changed binding order cannot disappear from this conservative set.
  same=np.zeros(len(origins),bool)
  rf=row["faces"][origins];cf=z["faces"]
  for shift in range(3):
   cr=np.roll(cf,shift,axis=1)
   same |= (np.all(row["positions"][rf]==z["vertices6"][:,:3][cr],axis=(1,2))
            &np.all(row["local"][rf]==z["binding_indices"][cr],axis=(1,2))
            &np.all(row["weights"][rf]==z["weights"][cr],axis=(1,2)))
  changed.update(map(int,origins[~same]))
  beforeids=np.asarray(sorted(changed),int);afterids=np.flatnonzero(np.isin(origins,beforeids))
  candidates[sid]=(row,{k:z[k] for k in z.files},beforeids,afterids)
out={"scope":"24 retained early and late poses compared directly to original source, including union reconstruction plus all local offsets: predicted candidate full self check and changed-star external intersections against all actual native captured targets. Not a native candidate run; existing unrelated interfaces are not admitted.","pins":pins,"poses":[],"complete":False}
def save(): (O/"report.json").write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
keys,*_=ca._load_target_inventory(h.INV);pins[str(h.INV)]=sha(h.INV)
def exact(p,f):
 used=np.unique(f);rec,deg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not deg,deg
 return rec
for arm in ["baseline","intervention","early"]:
 for step in (h.STEPS if arm!="early" else [0,4767,5023,5599,6207,6815,7423,8000]):
  at=time.monotonic()
  pack=(h.PAIR/arm/"native-run/accepted-geometry" if arm!="early" else R/"anatomy-completion-1276/twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry")/f"step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json")
  pins[str(rp)]=sha(rp);pins[str(pack)]=sha(pack)
  rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
  poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}
  positions,surfaces,_=ca._pack_surfaces(pack,keys)
  result={"arm":arm,"step":step,"rows":[]};out["poses"].append(result)
  for sid,(row,z,bids,aids) in candidates.items():
   world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,poses)
   sf=np.asarray(surfaces[(51005,sid)]["faces"],int)
   delta=sf-row["faces"];assert np.all(delta==delta.flat[0])
   actual=positions[int(delta.flat[0]):int(delta.flat[0])+row["vc"]]
   forward=h.forward(row["positions"],row["local"],row["weights"],row,t,poses)
   used=np.unique(row["faces"]);err=np.linalg.norm(forward[used].astype(float)-actual[used].astype(float),axis=1)
   basef=row["faces"][bids];candf=z["faces"][aids]
   allpoints=np.concatenate([actual[np.unique(basef)],world[np.unique(candf)]])
   lo=allpoints.min(0);hi=allpoints.max(0)
   brec=exact(actual,basef);crec=exact(world,candf);full=exact(world,z["faces"])
   rr={"stable_id":sid,"self_count":ci._audit_pair(full,full,same_surface=True)["count"],"max_referenced_forward_error_m":float(err.max()),"external_targets_scanned":0,"changed_star_interfaces":[],"new_triangle_pair_count":0}
   rr["closed_oriented_manifold_candidate"]=analyze_topology(world.tolist(),z["faces"].tolist())["closed_oriented_manifold_candidate"]
   result["rows"].append(rr)
   for key,target in surfaces.items():
    if key==(51005,sid):continue
    rr["external_targets_scanned"]+=1
    tf=np.asarray(target["faces"],int);tri=positions[tf]
    select=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
    if not len(select):continue
    tar=exact(positions,tf[select])
    ba=ci._audit_pair(brec,tar,same_surface=False);aa=ci._audit_pair(crec,tar,same_surface=False)
    if not ba["count"] and not aa["count"]:continue
    bp={(int(bids[x]),int(select[y])) for x,y in ba["triangle_pairs"]}
    ap={(int(z["face_origins"][aids[x]]),int(select[y])) for x,y in aa["triangle_pairs"]}
    new=sorted(ap-bp);removed=sorted(bp-ap)
    rr["new_triangle_pair_count"]+=len(new)
    rr["changed_star_interfaces"].append({"semantic":key[0],"stable_id":key[1],"body":target["body"],"before_pairs":len(bp),"after_pairs":len(ap),"added_lineage_pairs":new,"removed_lineage_pairs":removed})
   save()
  result["wall_seconds"]=time.monotonic()-at;save()
  print(json.dumps({"arm":arm,"step":step,"wall_seconds":result["wall_seconds"],"rows":[{k:r[k] for k in ["stable_id","self_count","new_triangle_pair_count","external_targets_scanned"]} for r in result["rows"]]}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


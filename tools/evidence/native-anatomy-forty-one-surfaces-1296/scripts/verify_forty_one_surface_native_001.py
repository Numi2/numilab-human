from pathlib import Path
import sys,json,hashlib,importlib.util,time,csv
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
C=A/"forty-one-surface-native-composition-001";P=A/"forty-surface-native-composition-001"
N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=P/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),h.INV,Path(ci.__file__),Path(ca.__file__),Path(sys.modules[analyze_topology.__module__].__file__),N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv"]}
pp=P/"native-verification-001/report.json";pins[str(pp)]=sha(pp);prior=json.loads(pp.read_text())
assert prior["all_simultaneously_repaired_pairs_pass"]
assert prior["complete"] and prior["inputs_unchanged"] and prior["all_native_pair_checks_pass"] and prior["all_prior38_and_skin_geometry_exact"] and prior["all_coupled_trace_rows_exact_parent"]
t=h.load_tissue(T);candidates={}
reports=[A/"foot25-current-reference-selection-013/report.json"]
for rp in reports:
 pins[str(rp)]=sha(rp)
 for r in json.loads(rp.read_text())["rows"]:
  if not r["source_and_sampled_pose_self_clear"]:continue
  sid=r["stable_id"];p=Path(r["candidate_path"]);pins[str(p)]=sha(p)
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
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()))
assert len(trace)==len(base)==1000
out={"scope":"Eight actual native captures: full exact self and topology checks for added right flexor digitorum longus, complete859-target changed-star checks against actual prior native coordinates, and byte-exact referenced triangle coordinates for all859 unchanged anatomical surfaces against previously verified40-surface plus skin run. Prior40 and skin qualification inherited only through exact geometry and identical accepted poses. Not whole-body or finalfive-minute qualification.","pins":pins,"trace_rows":len(trace),"trace_columns":len(trace[0]),"all_coupled_trace_rows_exact_parent":trace==base,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def exact(p,f):
 used=np.unique(f);rec,deg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not deg,deg
 return rec
keys,*_=ca._load_target_inventory(h.INV)
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 at=time.monotonic();pack=N/"accepted-geometry"/f"step-{step}.mrvpack";ppack=B/"accepted-geometry"/f"step-{step}.mrvpack"
 receipts=[]
 for pk in (pack,ppack):
  rp=pk.with_suffix(".receipt.json");pins[str(pk)]=sha(pk);pins[str(rp)]=sha(rp)
  rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pk)] and rc["physical_endpoint"]=="accepted";receipts.append(rc)
 rc,prc=receipts;assert rc["accepted_body_poses"]==prc["accepted_body_poses"]
 poses={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,keys);bpos,bsurfs,_=ca._pack_surfaces(ppack,keys)
 assert set(surfs)==set(bsurfs) and len(surfs)==860
 unchanged=0
 for key,s in surfs.items():
  if key in {(51005,25)}:continue
  f=np.asarray(s["faces"],int);bf=np.asarray(bsurfs[key]["faces"],int)
  assert f.shape==bf.shape and np.array_equal(pos[f],bpos[bf]),(step,key)
  unchanged+=1
 assert unchanged==859
 cr={"accepted_step":step,"complete157_body_poses_exact_parent":True,"unchanged_surface_referenced_triangles_exact_parent":unchanged,"rows":[]};out["poses"].append(cr);save()
 for sid,(row,z,bids,aids) in candidates.items():
  sf=np.asarray(surfs[(51005,sid)]["faces"],int);offset=sf-z["faces"];assert np.all(offset==offset.flat[0])
  world=pos[int(offset.flat[0]):int(offset.flat[0])+len(z["vertices6"])]
  prediction=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,poses)
  used=np.unique(z["faces"]);error=float(np.linalg.norm(world[used].astype(float)-prediction[used].astype(float),axis=1).max());assert error<1e-6
  f=z["faces"];u,iv=np.unique(world[used],axis=0,return_inverse=True);qf=iv[np.searchsorted(used,f)]
  topo=analyze_topology(u.astype(float).tolist(),qf.tolist())
  full=exact(u,qf);audit=ci._audit_pair(full,full,same_surface=True)
  rr={"stable_id":sid,"self_count":audit["count"],"self_pairs":audit["triangle_pairs"],"closed_oriented_manifold_candidate":topo["closed_oriented_manifold_candidate"],"max_referenced_forward_error_m":error,"external_targets_scanned":0,"new_triangle_pair_count":0,"interfaces":[]};cr["rows"].append(rr);save()
  oldsf=np.asarray(bsurfs[(51005,sid)]["faces"],int);oldoffset=oldsf-row["faces"];assert np.all(oldoffset==oldoffset.flat[0])
  baseline=bpos[int(oldoffset.flat[0]):int(oldoffset.flat[0])+len(row["positions"])]
  bf=row["faces"][bids];cf=z["faces"][aids]
  points=np.concatenate([baseline[np.unique(bf)],world[np.unique(cf)]]);lo=points.min(0);hi=points.max(0)
  brec=exact(baseline,bf);crec=exact(world,cf)
  for key,target in surfs.items():
   if key==(51005,sid):continue
   rr["external_targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf]
   select=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
   if not len(select):continue
   tar=exact(pos,tf[select]);ba=ci._audit_pair(brec,tar,same_surface=False);aa=ci._audit_pair(crec,tar,same_surface=False)
   if not ba["count"] and not aa["count"]:continue
   bp={(int(bids[x]),int(select[y])) for x,y in ba["triangle_pairs"]}
   ap={(int(z["face_origins"][aids[x]]),int(select[y])) for x,y in aa["triangle_pairs"]};new=sorted(ap-bp)
   rr["new_triangle_pair_count"]+=len(new)
   rr["interfaces"].append({"semantic":key[0],"stable_id":key[1],"before":len(bp),"after":len(ap),"added_lineage_pairs":new,"removed_lineage_pairs":sorted(bp-ap)})
  save();print(json.dumps({"step":step,**rr}),flush=True)
 # Both simultaneously replaced surfaces use their actual old and new geometry.
 # This catches crossings that an old-A/new-B counterfactual alone could miss.
 cr["simultaneously_repaired_pairs"]=[]
 for sid1 in sorted(candidates):
  for sid2 in sorted(candidates):
   if sid2<=sid1:continue
   worlds=[];oldworlds=[];patches=[];oldrows=[]
   for sid in (sid1,sid2):
    row,z,_,_=candidates[sid];newf=np.asarray(surfs[(51005,sid)]["faces"],int);oldf=np.asarray(bsurfs[(51005,sid)]["faces"],int)
    no=newf-z["faces"];bo=oldf-row["faces"];assert np.all(no==no.flat[0]) and np.all(bo==bo.flat[0])
    worlds.append(pos[int(no.flat[0]):int(no.flat[0])+len(z["vertices6"])])
    oldworlds.append(bpos[int(bo.flat[0]):int(bo.flat[0])+len(row["positions"])])
    patches.append(z);oldrows.append(row)
   before=ci._audit_pair(exact(oldworlds[0],oldrows[0]["faces"]),exact(oldworlds[1],oldrows[1]["faces"]),same_surface=False)
   after=ci._audit_pair(exact(worlds[0],patches[0]["faces"]),exact(worlds[1],patches[1]["faces"]),same_surface=False)
   bp={tuple(map(int,p)) for p in before["triangle_pairs"]}
   ap={(int(patches[0]["face_origins"][a]),int(patches[1]["face_origins"][b])) for a,b in after["triangle_pairs"]}
   added=sorted(ap-bp)
   cr["simultaneously_repaired_pairs"].append({"stable_ids":[sid1,sid2],"actual_parent_pairs":len(bp),"actual_candidate_lineage_pairs":len(ap),"added_lineage_pairs":added})
 cr["wall_seconds"]=time.monotonic()-at;save()
out["all_native_pair_checks_pass"]=all(r["self_count"]==r["new_triangle_pair_count"]==0 and r["closed_oriented_manifold_candidate"] and r["external_targets_scanned"]==859 for p in out["poses"] for r in p["rows"])
out["all_prior40_and_skin_geometry_exact"]=all(p["unchanged_surface_referenced_triangles_exact_parent"]==859 for p in out["poses"])
out["all_simultaneously_repaired_pairs_pass"]=all(not r["added_lineage_pairs"] for p in out["poses"] for r in p["simultaneously_repaired_pairs"])
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()
assert out["all_simultaneously_repaired_pairs_pass"] and out["all_native_pair_checks_pass"] and out["all_prior40_and_skin_geometry_exact"] and out["all_coupled_trace_rows_exact_parent"] and out["inputs_unchanged"]
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}),flush=True)

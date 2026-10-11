from pathlib import Path
import sys,json,importlib.util,time,csv,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");C=A/"left-pq-retriangulated-native-mtp-two-002";P=A/"bilateral-edl-ehl-native-mtp-two-001";N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=h.sha;T=C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);rows={130:h.row_data(t,130)}
RP=A/"pq-left-retriangulated-native-target-audit-029/report.json";proof=json.loads(RP.read_text());assert proof["complete"] and proof["inputs_unchanged"] and proof["no_strict_new_parent_pairs"]
expected={(p["step"],130):next(r for r in p["rows"] if r["variant"]=="candidate") for p in proof["poses"]}
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),RP,h.INV,Path(ci.__file__),Path(ca.__file__),N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv",C/"baseline/execution.json",P/"baseline/execution.json"]}
for scene in (C,P):
 execution=json.loads((scene/"baseline/execution.json").read_text());assert execution["returncode"]==0 and not execution["changed_inputs"] and execution["native_argv_matches_prepared_cli_preview"] and execution["native_environment_matches_prepared_cli_preview"]
 declaration=json.loads((scene/"baseline/run-declaration.json").read_text());assert execution["observed_environment"]==declaration["owner_cli_preview"]["recorded_invocation_environment"]
 pins[str(scene/"baseline/run-declaration.json")]=sha(scene/"baseline/run-declaration.json")
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()));assert len(trace)==len(base)==1000
out={"scope":"Bounded integrated16s left pronator quadratus130 reference-boundary diagnostic. Eight actual captures test closed positive components/self intersections and865 neighbors. All865 other surfaces,157 body poses and the coupled trace must remain exact the immediate native parent. Inferred source reconstruction and bounded source correction are explicitly retained; existing external intersections and whole-body qualification remain open.","pins":pins,"trace_rows":len(trace),"trace_columns":len(trace[0]),"coupled_trace_exact_parent":trace==base,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def exact(p,f):
 used=np.unique(f);rec,dg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not dg;return rec
keys,*_=ca._load_target_inventory(h.INV);assert len(keys)==859
allkeys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162)}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 at=time.monotonic();pack=N/"accepted-geometry"/f"step-{step}.mrvpack";ppack=B/"accepted-geometry"/f"step-{step}.mrvpack";receipts=[]
 for pk in (pack,ppack):
  rp=pk.with_suffix(".receipt.json");pins[str(pk)]=sha(pk);pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pk)] and rc["physical_endpoint"]=="accepted";receipts.append(rc)
 rc,prc=receipts;assert len(rc["accepted_body_poses"])==157 and rc["accepted_body_poses"]==prc["accepted_body_poses"];poses={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,allkeys);bpos,bsurfs,_=ca._pack_surfaces(ppack,allkeys);assert set(surfs)==set(bsurfs) and len(surfs)==len(bsurfs)==866
 unchanged=0
 for key,s in bsurfs.items():
  if key==(51005,130):continue
  sf=np.asarray(surfs[key]["faces"],int);bf=np.asarray(s["faces"],int);assert sf.shape==bf.shape and np.array_equal(pos[sf],bpos[bf]),(step,key);unchanged+=1
 assert unchanged==865
 cr={"accepted_step":step,"complete157_body_poses_exact_parent":True,"unchanged_surface_referenced_triangles_exact_parent":unchanged,"rows":[]};out["poses"].append(cr);save()
 for sid,row in rows.items():
  f=row["faces"];sf=np.asarray(surfs[(51005,sid)]["faces"],int);offset=sf-f;assert np.all(offset==offset.flat[0]);world=pos[int(offset.flat[0]):int(offset.flat[0])+row["vc"]]
  prediction=h.forward(row["positions"],row["local"],row["weights"],row,t,poses);error=float(np.linalg.norm(world.astype(float)-prediction.astype(float),axis=1).max());assert error<1e-6
  topo=analyze_topology(world.astype(float).tolist(),f.tolist());rec=exact(world,f);selfcheck=ci._audit_pair(rec,rec,same_surface=True)
  tv=world.astype(float)[f];volume=float(np.sum(np.einsum("ij,ij->i",tv[:,0],np.cross(tv[:,1],tv[:,2])))/6)
  components=[]
  for component in topo["face_components"]:
   faces_component=f[component];v=world.astype(float)[faces_component]-world[np.unique(faces_component)].mean(0,dtype=float)
   components.append(float(np.einsum("ij,ij->i",v[:,0],np.cross(v[:,1],v[:,2])).sum()/6))
  rr={"component_signed_volumes_m3":components,"stable_id":sid,"signed_volume_m3":volume,"self_count":selfcheck["count"],"self_pairs":selfcheck["triangle_pairs"],"topology":topo,"max_forward_error_m":error,"external_targets_scanned":0,"external_pair_count":0,"interfaces":[]};cr["rows"].append(rr)
  lo=world.min(0);hi=world.max(0);nativepairs=set()
  for key,target in surfs.items():
   if key==(51005,sid):continue
   rr["external_targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
   if not len(sel):continue
   tar=exact(pos,tf[sel]);actual=ci._audit_pair(rec,tar,same_surface=False)["triangle_pairs"];rr["external_pair_count"]+=len(actual)
   nativepairs.update((int(key[0]),int(key[1]),int(i),int(sel[j])) for i,j in actual)
   if actual:rr["interfaces"].append({"semantic":key[0],"stable_id":key[1],"native_pairs":[[int(i),int(sel[j])] for i,j in actual]})
  er=expected[(step,sid)];expairs={(int(x["key"][0]),int(x["key"][1]),int(i),int(j)) for x in er["target_pairs"] for i,j in x["pairs"]}
  rr["native_interface_pairs_exact_offline"]=nativepairs==expairs;rr["native_pairs_added_vs_offline"]=sorted(nativepairs-expairs);rr["native_pairs_removed_vs_offline"]=sorted(expairs-nativepairs)
 cr["wall_seconds"]=time.monotonic()-at;save();print(json.dumps({"step":step,"rows":[{k:r[k] for k in ("stable_id","self_count","external_targets_scanned","external_pair_count","max_forward_error_m","native_interface_pairs_exact_offline")} for r in cr["rows"]]}),flush=True)
out["all_native_geometry_checks_pass"]=out["coupled_trace_exact_parent"] and len(out["poses"])==8 and all(r["signed_volume_m3"]>0 and r["self_count"]==0 and r["topology"]["closed_oriented_manifold_candidate"] and r["external_targets_scanned"]==865 and all(v>0 for v in r["component_signed_volumes_m3"]) for p in out["poses"] for r in p["rows"])
out["all_native_interface_pairs_exact_offline"]=all(r["native_interface_pairs_exact_offline"] for p in out["poses"] for r in p["rows"])
out["all_external_interfaces_clear"]=all(r["external_pair_count"]==0 for p in out["poses"] for r in p["rows"])
out.update(complete=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()));save()
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}),flush=True)
assert out["all_native_geometry_checks_pass"] and out["inputs_unchanged"]

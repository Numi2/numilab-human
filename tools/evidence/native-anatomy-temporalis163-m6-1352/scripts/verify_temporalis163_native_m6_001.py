from pathlib import Path
import sys,json,importlib.util,time,csv,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
C=A/"temporalis163-native-m6-001";P=A/"left-fds-native-m6-001";N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-left-pq-publication-1344/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=h.sha;T=C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,163)
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),h.INV,Path(ci.__file__),Path(ca.__file__),N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv",C/"baseline/execution.json",P/"baseline/execution.json"]}
for scene in (C,P):
 execution=json.loads((scene/"baseline/execution.json").read_text());assert execution["returncode"]==0 and not execution["changed_inputs"] and execution["native_argv_matches_prepared_cli_preview"] and execution["native_environment_matches_prepared_cli_preview"]
 declaration=json.loads((scene/"baseline/run-declaration.json").read_text());assert execution["observed_environment"]==declaration["owner_cli_preview"]["recorded_invocation_environment"]
 pins[str(scene/"baseline/run-declaration.json")]=sha(scene/"baseline/run-declaration.json")
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()));assert len(trace)==len(base)==1000
out=dict(scope="Integrated native 16s passive temporalis163 anatomy increment; eight actual accepted captures. All866 prior surfaces,157 body poses and coupled physiology must equal the unchanged parent. Newly appended surface must remain closed, positive, self-clear and external-interface-clear. Not whole-anatomy or five-minute qualification.",pins=pins,coupled_trace_exact_parent=trace==base,trace_rows=len(trace),poses=[],complete=False)
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def exact(p,f):
 used=np.unique(f);rec,dg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not dg,dg;return rec
keys,*_=ca._load_target_inventory(h.INV);allkeys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163)}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 at=time.monotonic();pack=N/"accepted-geometry"/f"step-{step}.mrvpack";ppack=B/"accepted-geometry"/f"step-{step}.mrvpack";receipts=[]
 for pk in (pack,ppack):
  rp=pk.with_suffix(".receipt.json");pins[str(pk)]=sha(pk);pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pk)] and rc["physical_endpoint"]=="accepted";receipts.append(rc)
 rc,prc=receipts;assert len(rc["accepted_body_poses"])==157 and rc["accepted_body_poses"]==prc["accepted_body_poses"]
 poses={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,allkeys);bpos,bsurfs,_=ca._pack_surfaces(ppack,allkeys-{(51005,163)})
 assert len(surfs)==867 and len(bsurfs)==866 and set(surfs)==set(bsurfs)|{(51005,163)}
 for key,s in bsurfs.items():
  sf=np.asarray(surfs[key]["faces"],int);bf=np.asarray(s["faces"],int);assert sf.shape==bf.shape and np.array_equal(pos[sf],bpos[bf]),(step,key)
 f=row["faces"];sf=np.asarray(surfs[(51005,163)]["faces"],int);offset=sf-f;assert np.all(offset==offset.flat[0]);world=pos[int(offset.flat[0]):int(offset.flat[0])+row["vc"]]
 prediction=h.forward(row["positions"],row["local"],row["weights"],row,t,poses);error=float(np.linalg.norm(world.astype(float)-prediction.astype(float),axis=1).max());assert error<1e-6
 topo=analyze_topology(world.astype(float).tolist(),f.tolist());rec=exact(world,f);selfcheck=ci._audit_pair(rec,rec,same_surface=True)
 components=[]
 for component in topo["face_components"]:
  fc=f[component];v=world.astype(float)[fc]-world[np.unique(fc)].mean(0,dtype=float);components.append(float(np.einsum("ij,ij->i",v[:,0],np.cross(v[:,1],v[:,2])).sum()/6))
 result=dict(accepted_step=step,complete157_body_poses_exact_parent=True,unchanged_surface_referenced_triangles_exact_parent=866,component_signed_volumes_m3=components,self_count=selfcheck["count"],self_pairs=selfcheck["triangle_pairs"],topology=topo,max_forward_error_m=error,external_targets_scanned=0,external_pair_count=0,interfaces=[])
 lo=world.min(0);hi=world.max(0)
 for key,target in surfs.items():
  if key==(51005,163):continue
  result["external_targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
  if not len(sel):continue
  pairs=ci._audit_pair(rec,exact(pos,tf[sel]),same_surface=False)["triangle_pairs"];result["external_pair_count"]+=len(pairs)
  if pairs:result["interfaces"].append(dict(semantic=key[0],stable_id=key[1],native_pairs=[[int(i),int(sel[j])] for i,j in pairs]))
 result["wall_seconds"]=time.monotonic()-at;out["poses"].append(result);save();print(json.dumps({k:result[k] for k in ("accepted_step","self_count","external_pair_count","max_forward_error_m")}),flush=True)
out["all_native_geometry_checks_pass"]=out["coupled_trace_exact_parent"] and len(out["poses"])==8 and all(r["self_count"]==0 and r["external_pair_count"]==0 and r["external_targets_scanned"]==866 and r["topology"]["closed_oriented_manifold_candidate"] and len(r["component_signed_volumes_m3"])==1 and r["component_signed_volumes_m3"][0]>0 for r in out["poses"])
out.update(complete=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()));save()
assert out["all_native_geometry_checks_pass"] and out["inputs_unchanged"]
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}),flush=True)


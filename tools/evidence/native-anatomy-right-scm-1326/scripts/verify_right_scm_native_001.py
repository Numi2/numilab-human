from pathlib import Path
import sys,json,hashlib,importlib.util,time,csv,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");C=A/"right-scm-native-composition-001";P=A/"fifty-seven-surface-native-composition-001";N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=h.sha;T=C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,151);Z=C/"right-scm-row-patch.npz";z=np.load(Z);f=z["faces"]
RP=A/"scm-vessel-clearance-001/run-072-scm-ica-offset-source-lift-001/result-001/report.json";proof=json.loads(RP.read_text());expected={x["step"]:x for x in proof["pose_checks"]}
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),Z,RP,h.INV,Path(ci.__file__),Path(ca.__file__),N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv",P/"native-verification-001/report.json",C/"baseline/execution.json"]}
execution=json.loads((C/"baseline/execution.json").read_text());assert execution["returncode"]==0 and not execution["changed_inputs"] and execution["native_argv_matches_prepared_cli_preview"] and execution["native_environment_matches_prepared_cli_preview"]
parent=json.loads((P/"native-verification-001/report.json").read_text());assert parent["complete"] and parent["inputs_unchanged"] and parent["all_native_pair_checks_pass"]
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()));assert len(trace)==len(base)==1000
out={"scope":"Native addition of passive inferred right SCM151 to unchanged native57. Source and eight actual native self/topology checks, full860-target comparison to declared offline candidate at same accepted pose, and ICA/IJV/skin clearance. Existing bone/pectoralis interfaces remain explicitly reported; this is not full-body interface or five-minute qualification.","pins":pins,"trace_rows":len(trace),"trace_columns":len(trace[0]),"coupled_trace_exact_parent":trace==base,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def exact(p,f):
 used=np.unique(f);rec,dg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not dg;return rec
keys,*_=ca._load_target_inventory(h.INV);assert len(keys)==859
keys=set(keys)|{(51005,151)}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 at=time.monotonic();pack=N/"accepted-geometry"/f"step-{step}.mrvpack";ppack=B/"accepted-geometry"/f"step-{step}.mrvpack";receipts=[]
 for pk in (pack,ppack):
  rp=pk.with_suffix(".receipt.json");pins[str(pk)]=sha(pk);pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pk)] and rc["physical_endpoint"]=="accepted";receipts.append(rc)
 rc,prc=receipts;assert rc["accepted_body_poses"]==prc["accepted_body_poses"];poses={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,keys);bpos,bsurfs,_=ca._pack_surfaces(ppack,keys-{(51005,151)})
 assert set(surfs)-set(bsurfs)=={(51005,151)} and len(surfs)==861 and len(bsurfs)==860
 for key,s in bsurfs.items():
  sf=np.asarray(surfs[key]["faces"],int);bf=np.asarray(s["faces"],int);assert sf.shape==bf.shape and np.array_equal(pos[sf],bpos[bf]),(step,key)
 sf=np.asarray(surfs[(51005,151)]["faces"],int);offset=sf-f;assert np.all(offset==offset.flat[0]);world=pos[int(offset.flat[0]):int(offset.flat[0])+len(z["vertices6"])]
 prediction=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,poses);error=float(np.linalg.norm(world.astype(float)-prediction.astype(float),axis=1).max());assert error<1e-6
 topo=analyze_topology(world.astype(float).tolist(),f.tolist());rec=exact(world,f);prec=exact(prediction,f);selfcheck=ci._audit_pair(rec,rec,same_surface=True)
 cr={"accepted_step":step,"complete157_body_poses_exact_parent":True,"unchanged_surface_referenced_triangles_exact_parent":860,"self_count":selfcheck["count"],"self_pairs":selfcheck["triangle_pairs"],"topology":topo,"max_forward_error_m":error,"external_targets_scanned":0,"external_pair_count":0,"native_vs_predicted_pair_differences":[],"interfaces":[]};out["poses"].append(cr);save()
 pts=np.concatenate((world,prediction));lo=pts.min(0);hi=pts.max(0)
 for key,target in surfs.items():
  if key==(51005,151):continue
  cr["external_targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
  if not len(sel):continue
  tar=exact(pos,tf[sel]);actual=ci._audit_pair(rec,tar,same_surface=False)["triangle_pairs"];pred=ci._audit_pair(prec,tar,same_surface=False)["triangle_pairs"];ap={(int(i),int(sel[j])) for i,j in actual};bp={(int(i),int(sel[j])) for i,j in pred};cr["external_pair_count"]+=len(ap)
  if ap or bp:cr["interfaces"].append({"semantic":key[0],"stable_id":key[1],"body_index":int(target["body"]),"native_pairs":sorted(ap),"predicted_pairs":sorted(bp)})
  if ap!=bp:cr["native_vs_predicted_pair_differences"].append({"key":list(key),"native_only":sorted(ap-bp),"predicted_only":sorted(bp-ap)})
 lookup={(x["semantic"],x["stable_id"]):len(x["native_pairs"]) for x in cr["interfaces"]};cr["skin_ica_ijv_clear"]=all(lookup.get(k,0)==0 for k in ((51007,1),(51011,466),(51011,468)))
 cr["external_total_matches_offline_proof"]=cr["external_pair_count"]==expected[step]["candidate_target_total"];cr["wall_seconds"]=time.monotonic()-at;save()
 print(json.dumps({k:cr[k] for k in ("accepted_step","self_count","external_targets_scanned","external_pair_count","skin_ica_ijv_clear","external_total_matches_offline_proof","max_forward_error_m","wall_seconds")}),flush=True)
out["all_native_checks_pass"]=out["coupled_trace_exact_parent"] and len(out["poses"])==8 and all(p["self_count"]==0 and p["topology"]["closed_oriented_manifold_candidate"] and p["external_targets_scanned"]==860 and p["skin_ica_ijv_clear"] and not p["native_vs_predicted_pair_differences"] and p["external_total_matches_offline_proof"] for p in out["poses"])
out.update(complete=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()));save();assert out["all_native_checks_pass"] and out["inputs_unchanged"]
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}),flush=True)

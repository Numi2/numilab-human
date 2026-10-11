from pathlib import Path
import sys,json,csv,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");C=A/"sartorius50-precision-native-001";P=A/"left-scm-masseters-outward-native-001";N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,50);CP=A/"sartorius50-native-precision-repair-002/stable-50-unadmitted.npz";z=np.load(CP);orig=z["face_origins"]
pins={str(p):h.sha(p) for p in (Path(__file__),H,T,CP,Path(ci.__file__),Path(ca.__file__),Path(cg.__file__),h.INV,C/"baseline/execution.json",N/"resting-coupled.csv",B/"resting-coupled.csv")}
ex=json.loads((C/"baseline/execution.json").read_text());assert ex["returncode"]==0 and not ex["changed_inputs"] and ex["native_argv_matches_prepared_cli_preview"] and ex["native_environment_matches_prepared_cli_preview"]
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));bt=list(csv.DictReader((B/"resting-coupled.csv").open()));assert trace==bt and len(trace)==1000
keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,x) for x in (151,152,159,160)}
def exact(p,f):
 u=np.unique(f);rec,dg=h.exact_rows(p[u],np.searchsorted(u,f),ci);assert not dg;return rec
out={"scope":"Native source precision repair of passive left sartorius50. Allother863 surfaces/bodyposes/physiology exact MTP0 parent. Source all3closed components retained;2slivertriangles removed,0.133micrometre endpointmerge. Existing anatomicalinterfaces remainunqualified; no wholebodycompletion/five-minuteclaim.","pins":pins,"coupled_trace_exact_parent":True,"trace_rows":len(trace),"trace_columns":len(trace[0]),"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 pack=N/f"accepted-geometry/step-{step}.mrvpack";pp=B/f"accepted-geometry/step-{step}.mrvpack";rc=json.loads(pack.with_suffix(".receipt.json").read_text());prc=json.loads(pp.with_suffix(".receipt.json").read_text());assert rc["accepted_body_poses"]==prc["accepted_body_poses"]
 for pk,d in ((pack,rc),(pp,prc)):
  pins[str(pk)]=h.sha(pk);pins[str(pk.with_suffix(".receipt.json"))]=h.sha(pk.with_suffix(".receipt.json"));assert d["pack_file_sha256"]==pins[str(pk)]
 pos,surfs,_=ca._pack_surfaces(pack,keys);bp,bs,_=ca._pack_surfaces(pp,keys);assert set(surfs)==set(bs) and len(surfs)==864
 for key,s in surfs.items():
  if key!=(51005,50):assert np.array_equal(pos[np.asarray(s["faces"],int)],bp[np.asarray(bs[key]["faces"],int)])
 pm={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 sf=np.asarray(surfs[(51005,50)]["faces"],int);oldf=np.asarray(bs[(51005,50)]["faces"],int);ids=np.unique(sf);world=pos[ids];f=np.searchsorted(ids,sf);rec=exact(world,f);br=exact(bp,oldf)
 predicted=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,pm);err=float(np.linalg.norm(predicted[z["faces"]].astype(float)-pos[sf],axis=2).max());assert err<6e-7
 # Native emits per-triangle vertices. Quotient exact positions for the same geometric topology used by the intersection owner.
 up,inv=np.unique(world,axis=0,return_inverse=True);qf=inv[f];top=cg.analyze_topology(up.astype(float).tolist(),qf.tolist());pairs=ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"];lo=np.minimum(world.min(0),bp[oldf].min((0,1)));hi=np.maximum(world.max(0),bp[oldf].max((0,1)))
 item={"step":step,"all_other863_triangles_exact_parent":True,"bodyposes_exact_parent":True,"source_forward_native_max_error_m":err,"self_pairs":pairs,"topology":top,"targets_scanned":0,"target_pairs":[]};out["poses"].append(item)
 for key,target in surfs.items():
  if key==(51005,50):continue
  item["targets_scanned"]+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,1)&np.all(tri.min(1)<=hi,1))
  if not len(sel):continue
  tr=exact(pos,tf[sel]);after=ci._audit_pair(rec,tr,same_surface=False)["triangle_pairs"];before=ci._audit_pair(br,tr,same_surface=False)["triangle_pairs"]
  ap={(int(orig[i]),int(sel[j])) for i,j in after};bset={(int(i),int(sel[j])) for i,j in before}
  if ap or bset:item["target_pairs"].append({"key":list(key),"candidate_parent_pairs":sorted(ap),"baseline_pairs":sorted(bset),"new_pairs":sorted(ap-bset),"removed_pairs":sorted(bset-ap)})
 save();print(json.dumps({"step":step,"self":len(pairs),"closed":top["closed_oriented_manifold_candidate"],"components":top["face_component_count"],"targets":item["targets_scanned"],"new_pairs":sum(len(x["new_pairs"]) for x in item["target_pairs"]),"forward_error":err}),flush=True)
out.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),all_native_checks_pass=all(not p["self_pairs"] and p["topology"]["closed_oriented_manifold_candidate"] and p["topology"]["face_component_count"]==3 and p["targets_scanned"]==863 and all(not x["new_pairs"] for x in p["target_pairs"]) for p in out["poses"]));save();assert out["inputs_unchanged"] and out["all_native_checks_pass"]

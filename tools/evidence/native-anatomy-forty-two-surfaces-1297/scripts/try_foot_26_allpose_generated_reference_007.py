from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"foot-26-allpose-generated-reference-007";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"forty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,T,Path(ci.__file__),Path(ca.__file__)]}
report={"scope":"Unadmitted bounded local reference geometry inference, exact all-pose generated-boundary normal/axis proposal; original source points fixed and exact incremental self owner. Routes, weights, one-hot proxy positions unchanged.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def audit(p,f):
 rec,deg=h.exact_rows(p,f,ci);d=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
 return {"count":d["count"],"pairs":d["triangle_pairs"],"degenerate":deg}
def incremental(base,candidate,faces,full_audit):
 audit_record={"count":full_audit["count"],"triangle_pairs":full_audit["pairs"],"degenerate_face_rows":full_audit["degenerate"]}
 try:
  result=ca.audit_incremental_skin_self_intersections(baseline_world_positions=base,candidate_world_positions=candidate,baseline_faces=faces,candidate_faces=faces,baseline_self_audit=audit_record,expected_baseline_world_f32_sha256=hashlib.sha256(base.tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(faces),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(audit_record,len(faces)))
  return {"count":result["count"],"pairs":result["triangle_pairs"],"degenerate":[],"changed_face_count":len(result["changed_skin_face_rows"])}
 except Exception as e:return {"count":None,"error":repr(e)}
poses={};pose_labels={}
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-surface-native-composition-001/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
  idx=len(poses);pose_labels[idx]={"arm":arm,"step":step}
  poses[idx]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}
rp=A/"foot-source-reference-posed-audit-004/report.json";pins[str(rp)]=sha(rp);cache=json.loads(rp.read_text());assert cache["complete"] and cache["inputs_unchanged"]
for p,v in cache["pins"].items():assert sha(p)==v
report["pose_labels"]=pose_labels

t=h.load_tissue(T)

sid=26;cp=A/"foot-26-fixed-source-junction-reference-002/stable-26-junction-1-microns.npz";pins[str(cp)]=sha(cp);z=np.load(cp);row=h.row_data(t,sid);p=z["vertices6"][:,:3].copy();f=z["faces"];local=z["binding_indices"];w=z["weights"]
cached=next(r for r in cache["rows"] if r["stable_id"]==sid);assert cached["candidate_sha256"]==sha(cp)
def restore(r):return {"count":r["self_count"],"pairs":r["pairs"],"degenerate":r["degenerate"]}
source=restore(cached["poses"][0]);baseline_worlds={step:h.forward(p,local,w,row,t,pose) for step,pose in poses.items()};baselines={idx:restore(next(r for r in cached["poses"] if r["arm"]==label["arm"] and r["step"]==label["step"])) for idx,label in pose_labels.items()}

assert source["count"]==0 and all(not b["degenerate"] for b in baselines.values())
pure=np.unique(row["faces"]);pure=pure[row["weights"][pure].max(1)>=.999999]
frozen={row["positions"][j].tobytes() for j in pure}
locked={j for j in range(len(p)) if p[j].tobytes() in frozen}
norm=h.vertex_normals(p,f).astype(float);oldn=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]])
oldlength=np.linalg.norm(oldn,axis=1);q=p.copy();report.update(stable_id=sid,base_candidate_path=str(cp),source=source,baseline_poses=baselines,locked_original_source_point_count=len(locked),trials=[],accepted_unadmitted_steps=[],selected=None);save()
def all_poses(x):
 return {step:incremental(baseline_worlds[step],h.forward(x,local,w,row,t,pose),f,baselines[step]) for step,pose in poses.items()}
current=all_poses(q)
for iteration in range(16):
 counts=[r["count"] for r in current.values()];assert all(c is not None for c in counts)
 if not sum(counts):break
 target=max(current,key=lambda step:current[step]["count"]);pairs=current[target]["pairs"];seeds=np.unique(f[np.unique(np.asarray(pairs,int))]);best=None
 for microns in (.5,1,5,10,25,50):
  for j in seeds:
   if int(j) in locked:continue
   for di,direction in enumerate([norm[j],-norm[j],*np.eye(3),*(-np.eye(3))]):
    x=q.copy();x[j]=(q[j].astype(float)+direction*microns*1e-6).astype(np.float32)
    bound=float(np.linalg.norm(x.astype(float)-p,axis=1).max())
    if bound>.000101:continue
    nn=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]);ll=np.linalg.norm(nn,axis=1);cos=np.einsum("ij,ij->i",nn,oldn)/np.maximum(ll*oldlength,1e-30)
    if cos.min()<=0 or (ll/oldlength).min()<.2:continue
    src=incremental(p,x,f,source)
    if src["count"]!=0:continue
    target_audit=incremental(baseline_worlds[target],h.forward(x,local,w,row,t,poses[target]),f,baselines[target])
    if target_audit["count"] is None or target_audit["count"]>=current[target]["count"]:continue
    checks=all_poses(x);after=[r["count"] for r in checks.values()]
    if any(v is None for v in after):continue
    record={"iteration":iteration,"vertex":int(j),"direction_index":di,"microns":microns,"max_delta_m":bound,"before_counts":counts,"after_counts":after,"min_area_ratio":float((ll/oldlength).min()),"min_normal_cosine":float(cos.min())};report["trials"].append(record)
    score=(max(after),sum(after),bound)
    if score[:2]<(max(counts),sum(counts)) and (best is None or score<best[0]):best=(score,x,checks,record)
    if not sum(after):break
   if best is not None and not best[0][1]:break
  save()
  if best is not None:break
 if best is None:break
 _,q,current,record=best;report["accepted_unadmitted_steps"].append(record)
 dest=O/f"stable-26-unadmitted-iteration-{iteration}.npz"
 np.savez(dest,vertices6=np.column_stack([q,h.vertex_normals(q,f)]).astype("<f4"),faces=f,weights=w,binding_indices=local,face_origins=z["face_origins"]);record.update(candidate_path=str(dest),candidate_sha256=sha(dest));save();print(json.dumps(record),flush=True)
 if all(x["count"]==0 for x in current.values()):
  report["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"maximum_source_displacement_m":float(np.linalg.norm(q.astype(float)-p,axis=1).max()),"poses":[{"step":step,**a} for step,a in current.items()]};break
report.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save();print(json.dumps({"selected":report["selected"],"remaining_counts":[a["count"] for a in current.values()]}))

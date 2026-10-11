from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"stable-133-reference-continuation-006";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"forty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
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
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-four-surface-native-composition-002/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
  idx=len(poses);pose_labels[idx]={"arm":arm,"step":step}
  poses[idx]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc["accepted_registered_body_poses"])}
report["pose_labels"]=pose_labels

t=h.load_tissue(T)

sid=133;cp=A/"remaining-limb-local-reference-005/stable-133-iteration-0-unadmitted-partial.npz";pins[str(cp)]=sha(cp);z=np.load(cp);row=h.row_data(t,sid);p=z["vertices6"][:,:3].copy();f=z["faces"];local=z["binding_indices"];w=z["weights"]
source=audit(p,f);baseline_worlds={idx:h.forward(p,local,w,row,t,pose) for idx,pose in poses.items()};baselines={idx:audit(world,f) for idx,world in baseline_worlds.items()}

assert source["count"]==0 and all(not b["degenerate"] for b in baselines.values())
pure=np.unique(row["faces"]);pure=pure[row["weights"][pure].max(1)>=.999999]
frozen={row["positions"][j].tobytes() for j in pure}
locked={j for j in range(len(p)) if p[j].tobytes() in frozen}
norm=h.vertex_normals(p,f).astype(float);oldn=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]])
oldlength=np.linalg.norm(oldn,axis=1);q=p.copy();report.update(stable_id=sid,base_candidate_path=str(cp),source=source,baseline_poses=baselines,locked_original_source_point_count=len(locked),trials=[],accepted_unadmitted_steps=[],selected=None);save()
def all_poses(x):
 return {step:incremental(baseline_worlds[step],h.forward(x,local,w,row,t,pose),f,baselines[step]) for step,pose in poses.items()}

origin_path=A/"remaining-limb-positive-union-lift-001/stable-133-reference-union-row-patch.npz";pins[str(origin_path)]=sha(origin_path)
origin=np.load(origin_path)["vertices6"][:,:3]
oldn=np.cross(origin[f[:,1]].astype(float)-origin[f[:,0]],origin[f[:,2]].astype(float)-origin[f[:,0]]);oldlength=np.linalg.norm(oldn,axis=1)
def posed(x,idx):
 changed=np.flatnonzero(np.any(x!=p,axis=1));world=baseline_worlds[idx].copy()
 if len(changed):world[changed]=h.forward(x[changed],local[changed],w[changed],row,t,poses[idx])
 return world
def all_poses(x):
 return {idx:incremental(baseline_worlds[idx],posed(x,idx),f,baselines[idx]) for idx in poses}
def jacobian(j,idx):
 J=np.zeros((3,3))
 for slot in range(4):
  if w[j,slot]<=0:continue
  b=t["bindings"][row["fb"]+int(local[j,slot])];v=b["value"];bq=poses[idx][int(b["core"])][1]
  for k in range(3):J[:,k]+=float(w[j,slot])*h.rotate32(bq,h.rotate32(v[3:7],np.eye(3,dtype=np.float32)[k]))*float(v[7])
 return J

def area(x):return float(np.linalg.norm(np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]),axis=1).sum()/2)
def volume(x):return float(np.einsum("ij,ij->i",x[f[:,0]].astype(float),np.cross(x[f[:,1]].astype(float),x[f[:,2]].astype(float))).sum()/6)
basearea=area(origin);basevol=volume(origin)
report["reference_bounds"]={"maximum_position_correction_m":.0005,"maximum_area_relative_change":.001,"maximum_signed_volume_relative_change":.001,"status":"Explicit engineering sensitivity limits for passive reference-geometry inference, not measured source accuracy. Original attachment proxies/maps remain fixed; source/topology/allpose exact checks mandatory."}
current=all_poses(q)
started=time.monotonic();deadline=started+300
for iteration in range(8):
 counts=[r["count"] for r in current.values()];assert all(v is not None for v in counts)
 if not sum(counts):break
 target=max(current,key=lambda idx:current[idx]["count"]);seeds=np.unique(f[np.unique(current[target]["pairs"])]);best=None
 for microns in (.5,5,25,50,100,250,400):
  if time.monotonic()>deadline:break
  for j in seeds:
   if time.monotonic()>deadline:break
   if int(j) in locked:continue
   for di,direction in enumerate([norm[j],-norm[j],*np.eye(3),*(-np.eye(3))]):
    x=q.copy();x[j]=(q[j].astype(float)+direction*microns*1e-6).astype(np.float32)
    bound=float(np.linalg.norm(x.astype(float)-origin,axis=1).max())
    if bound>.0005:continue
    nn=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]])
    ll=np.linalg.norm(nn,axis=1);cos=np.einsum("ij,ij->i",nn,oldn)/np.maximum(ll*oldlength,1e-30)
    if cos.min()<=0 or (ll/oldlength).min()<.2:continue
    ar=area(x)/basearea-1;vr=volume(x)/basevol-1
    if abs(ar)>.001 or abs(vr)>.001:continue
    if incremental(p,x,f,source)["count"]!=0:continue
    ta=incremental(baseline_worlds[target],posed(x,target),f,baselines[target])
    if ta["count"] is None or ta["count"]>=current[target]["count"]:continue
    checks=all_poses(x);after=[r["count"] for r in checks.values()]
    if any(v is None for v in after):continue
    record={"iteration":iteration,"vertex":int(j),"direction_index":di,"microns":microns,"max_delta_m":bound,"before_counts":counts,"after_counts":after,"min_area_ratio":float((ll/oldlength).min()),"min_normal_cosine":float(cos.min()),"area_relative_change":ar,"signed_volume_relative_change":vr};report["trials"].append(record)
    score=(max(after),sum(after),bound)
    if all(a<=b for a,b in zip(after,counts)) and score[:2]<(max(counts),sum(counts)) and (best is None or score<best[0]):best=(score,x,checks,record)
    if best and not best[0][1]:break
   if best and not best[0][1]:break
  save()
  if best:break
 if best is None:break
 _,q,current,record=best;report["accepted_unadmitted_steps"].append(record)
 dest=O/f"stable-{sid}-unadmitted-iteration-{iteration}.npz"
 np.savez(dest,vertices6=np.column_stack([q,h.vertex_normals(q,f)]).astype("<f4"),faces=f,weights=w,binding_indices=local,face_origins=z["face_origins"])
 record.update(candidate_path=str(dest),candidate_sha256=sha(dest));save();print(json.dumps(record),flush=True)
 if all(x["count"]==0 for x in current.values()):
  report["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"maximum_source_displacement_m":float(np.linalg.norm(q.astype(float)-origin,axis=1).max())};break
report.update(complete=True,budget_seconds=300,budget_expired=time.monotonic()>deadline,inputs_unchanged=all(sha(p)==v for p,v in pins.items()),remaining_counts=[a["count"] for a in current.values()]);save();print(json.dumps({"selected":report["selected"],"remaining_counts":report["remaining_counts"]}),flush=True)

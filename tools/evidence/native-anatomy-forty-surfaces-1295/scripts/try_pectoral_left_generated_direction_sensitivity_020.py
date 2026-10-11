from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"pectoral-left-generated-direction-sensitivity-020";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,T,Path(ci.__file__),Path(ca.__file__)]}
report={"scope":"Unadmitted bounded local reference geometry inference, fixed generated-vertex direction magnitude sensitivity and exact incremental self owner. Routes, weights, one-hot proxy positions unchanged.","pins":pins,"rows":[],"complete":False}
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
poses={}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 rp=A/"thirty-eight-surface-native-composition-001/baseline/native-run/accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
 poses[step]={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
t=h.load_tissue(T)
for sid in (74,):
 cp=A/"pectoral-left-generated-boundary-repair-018/stable-74-local-reference.npz"
 pins[str(cp)]=sha(cp);z=np.load(cp);row=h.row_data(t,sid);p=z["vertices6"][:,:3].copy();f=z["faces"];local=z["binding_indices"];w=z["weights"]
 source=audit(p,f);baseline_worlds={step:h.forward(p,local,w,row,t,pose) for step,pose in poses.items()};baselines={step:audit(world,f) for step,world in baseline_worlds.items()}
 assert source["count"]==0 and all(not b["degenerate"] for b in baselines.values())
 cr={"stable_id":sid,"base_candidate_path":str(cp),"source":source,"baseline_poses":baselines,"trials":[]};report["rows"].append(cr);save()
 original_y=float(p[949,1])+1.043081283569336e-7
 for microns in (.5,1,2,5,10,25,50,100):
  q=p.copy();q[949,1]=np.float32(original_y-microns*1e-6)
  trial={"microns":microns,"maximum_additional_displacement_m":float(np.linalg.norm(q.astype(float)-p,axis=1).max()),"source":incremental(p,q,f,source),"poses":[]};cr["trials"].append(trial)
  if trial["source"]["count"]==0:
   for step,pose in poses.items():trial["poses"].append({"step":step,**incremental(baseline_worlds[step],h.forward(q,local,w,row,t,pose),f,baselines[step])})
   if all(x["count"]==0 for x in trial["poses"]):
    u,iv=np.unique(q,axis=0,return_inverse=True);top=analyze_topology(u.astype(float).tolist(),iv[f].tolist());assert top["closed_oriented_manifold_candidate"]
    dest=O/f"stable-74-generated-{microns}-microns.npz";np.savez(dest,vertices6=np.column_stack([q,h.vertex_normals(q,f)]).astype("<f4"),faces=f,weights=w,binding_indices=local,face_origins=z["face_origins"]);trial.update(candidate_path=str(dest),candidate_sha256=sha(dest))
  save();print(json.dumps({"microns":microns,"source":trial["source"]["count"],"poses":[x["count"] for x in trial["poses"]],"candidate":trial.get("candidate_path")}),flush=True)
report["complete"]=True;report["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


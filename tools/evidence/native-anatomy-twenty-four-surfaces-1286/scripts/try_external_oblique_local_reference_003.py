from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";O=A/"external-oblique-local-reference-003";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,Path(ci.__file__)]}
report={"scope":"Unadmitted bounded reference geometry proposal. Routes and weights unchanged. One-hot visual weight is not assumed to identify a measured attachment. Any candidate still requires anatomical anchor and complete interface qualification.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def audit(p,f):
 rec,deg=h.exact_rows(p,f,ci)
 d=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
 return {"count":d["count"],"pairs":d["triangle_pairs"],"degenerate":deg}
poses={}
for step in (0,8000):
 rp=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry"/f"step-{step}.receipt.json"
 pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
 poses[step]={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
t=h.load_tissue(h.TISS)
prior_path=A/"external-oblique-local-reference-002/report.json"
prior=json.loads(prior_path.read_text());pins[str(prior_path)]=sha(prior_path)
for path,digest in prior["pins"].items():assert sha(path)==digest
prior_row=next(r for r in prior["rows"] if r["stable_id"]==67)
def incremental(base,candidate,faces,full_audit):
 audit_record={"count":full_audit["count"],"triangle_pairs":full_audit["pairs"],"degenerate_face_rows":full_audit["degenerate"]}
 result=ca.audit_incremental_skin_self_intersections(
  baseline_world_positions=base,candidate_world_positions=candidate,
  baseline_faces=faces,candidate_faces=faces,baseline_self_audit=audit_record,
  expected_baseline_world_f32_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
  expected_face_index_sha256=ca._face_index_sha256(faces),
  expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(audit_record,len(faces)))
 return {"count":result["count"],"pairs":result["triangle_pairs"],"degenerate":[],
         "changed_face_count":len(result["changed_skin_face_rows"]),
         "fresh_exact_pair_count":result["fresh_changed_face_exact_pair_count"],
         "reused_exact_pair_count":result["reused_baseline_exact_pair_count"]}
pins[str(Path(ca.__file__))]=sha(Path(ca.__file__))
for sid in (67,):
 row=h.row_data(t,sid);p0=row["positions"]
 u,first,iv=np.unique(p0,axis=0,return_index=True,return_inverse=True)
 used=np.unique(iv[row["faces"]]);ix=np.full(len(u),-1);ix[used]=np.arange(len(used))
 f=ix[iv[row["faces"]]];p=u[used];local=row["local"][first[used]];w=row["weights"][first[used]]
 dense=np.zeros((len(p0),row["bc"]),float)
 for v in range(len(p0)):
  for li,weight in zip(row["local"][v],row["weights"][v]):
   if weight>0:dense[v,li]+=float(weight)
 assert np.array_equal(dense,dense[first[iv]])
 source=prior_row["source"];baselines={int(k):v for k,v in prior_row["baseline_poses"].items()}
 baseline_worlds={step:h.forward(p,local,w,row,t,pose) for step,pose in poses.items()}
 assert all(not b["degenerate"] for b in [source,*baselines.values()])
 pairs=source["pairs"]+sum((b["pairs"] for b in baselines.values()),[])
 seeds=np.unique(f[np.unique(np.asarray(pairs,int))])
 neighbors=[set() for _ in p]
 for tri in f:
  for v in tri:neighbors[v].update(int(x) for x in tri if x!=v)
 delta=np.zeros_like(p,dtype=float)
 for v in seeds:delta[v]=p[sorted(neighbors[v])].astype(float).mean(0)-p[v]
 cr={"stable_id":sid,"source":source,"baseline_poses":baselines,"seed_count":len(seeds),"one_hot_seed_count":int(np.count_nonzero(w[seeds].max(1)>=.999999)),"source_quotient_closed":analyze_topology(p.tolist(),f.tolist())["closed_oriented_manifold_candidate"],"trials":[]}
 report["rows"].append(cr);save()
 for alpha in (.03,.1,.3,.6,1.0):
  disp=delta*alpha;length=np.linalg.norm(disp,axis=1);disp*=np.minimum(1,.00025/np.maximum(length,1e-30))[:,None]
  q=(p.astype(float)+disp).astype("<f4")
  trial={"alpha":alpha,"maximum_source_displacement_m":float(np.linalg.norm(q.astype(float)-p,axis=1).max()),"source":incremental(p,q,f,source),"poses":[]};cr["trials"].append(trial)
  if trial["source"]["count"]==0:
   for step,pose in poses.items():trial["poses"].append({"step":step,**incremental(baseline_worlds[step],h.forward(q,local,w,row,t,pose),f,baselines[step])})
   if all(x["count"]==0 for x in trial["poses"]):
    dest=O/f"stable-{sid}-alpha-{alpha}.npz"
    np.savez(dest,vertices6=np.column_stack((q,h.vertex_normals(q,f))).astype("<f4"),faces=f,weights=w,binding_indices=local,face_origins=np.arange(len(f)))
    trial["candidate_path"]=str(dest);trial["candidate_sha256"]=sha(dest)
  save();print(json.dumps({"sid":sid,"alpha":alpha,"source_count":trial["source"]["count"],"pose_counts":[x["count"] for x in trial["poses"]],"max_m":trial["maximum_source_displacement_m"]}),flush=True)
report["complete"]=True;report["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()

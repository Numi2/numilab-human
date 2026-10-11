from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-right-local-binding-sensitivity-029";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=A/"current-anatomy-mtp-two-native-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CP=A/"edl-right-positive-union-lift-001/stable-21-reference-union-row-patch.npz"
pins={str(p):sha(p) for p in [Path(__file__),H,T,CP,Path(ci.__file__),Path(ca.__file__)]}
poses={}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 rp=A/"current-anatomy-mtp-two-native-002/baseline/native-run/accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text())
 poses[step]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
t=h.load_tissue(T);row=h.row_data(t,21);z=np.load(CP)
p=z["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"].copy();w=z["weights"].copy()
fixed=w.max(1)>=.999999
def audit(world):
 rec,deg=h.exact_rows(world,f,ci)
 a=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
 return {"count":a["count"],"triangle_pairs":a["triangle_pairs"],"degenerate_face_rows":deg}
source=audit(p);assert source["count"]==0
baseworld={step:h.forward(p,local,w,row,t,pose) for step,pose in poses.items()}
baseaudit={step:audit(world) for step,world in baseworld.items()}
report={"scope":"Unadmitted local existing-binding sensitivity: source geometry, transforms, pure one-hot proxies and physical routes fixed. Mixed passive display bindings may redistribute among their existing three bodies. Maximum 2 mm displacement across eight actual K2 poses is a declared inference sensitivity bound, not measured anatomy. Full neighboring tissue audits and later poses remain mandatory.","pins":pins,"source":source,"baseline":baseaudit,"trials":[],"accepted":[],"selected":None,"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def inc(world,step):
 result=ca.audit_incremental_skin_self_intersections(baseline_world_positions=baseworld[step],candidate_world_positions=world,baseline_faces=f,candidate_faces=f,baseline_self_audit=baseaudit[step],expected_baseline_world_f32_sha256=hashlib.sha256(baseworld[step].tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(f),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(baseaudit[step],len(f)))
 result["degenerate_face_rows"]=[]
 return result
worlds={s:v.copy() for s,v in baseworld.items()};current=baseaudit.copy()
deadline=time.monotonic()+300
for it in range(30):
 counts=[x["count"] for x in current.values()]
 if not sum(counts):break
 target=max(current,key=lambda s:current[s]["count"])
 seeds=np.unique(f[np.unique(current[target]["triangle_pairs"])])
 best=None
 for j in seeds:
  if time.monotonic()>deadline:break
  if fixed[j]:continue
  for to in range(row["bc"]):
   for frac in (.02,.05,.1,.25,.5,1.):
    dw=np.zeros(3,float)
    for lane in range(4):
     if w[j,lane]>0:dw[int(local[j,lane])]+=float(w[j,lane])
    dw*=1-frac;dw[to]+=frac
    nl=np.array([[0,1,2,4294967295]],np.uint32);nw=np.array([[*dw,0]],np.float32);nw[0,:3]/=nw[0,:3].sum()
    changed={s:h.forward(p[j:j+1],nl,nw,row,t,pose)[0] for s,pose in poses.items()}
    bound=max(float(np.linalg.norm(changed[s].astype(float)-baseworld[s][j])) for s in poses)
    if bound>.002:continue
    trialworld=worlds[target].copy();trialworld[j]=changed[target]
    ta=inc(trialworld,target)
    if ta["degenerate_face_rows"] or ta["count"]>=current[target]["count"]:continue
    checks={};candidateworld={}
    for s in poses:
     v=worlds[s].copy();v[j]=changed[s];candidateworld[s]=v
     checks[s]=ta if s==target else inc(v,s)
    after=[a["count"] for a in checks.values()]
    if any(a["degenerate_face_rows"] for a in checks.values()) or any(a>b for a,b in zip(after,counts)):continue
    score=(max(after),sum(after),bound)
    rec={"iteration":it,"vertex":int(j),"destination_local_binding":to,"fraction":frac,"before":counts,"after":after,"max_changed_vertex_world_displacement_m":bound}
    report["trials"].append(rec)
    if score[:2]<(max(counts),sum(counts)) and (best is None or score<best[0]):best=(score,int(j),nl[0],nw[0],candidateworld,checks,rec)
  save()
 if best is None:break
 _,j,local[j],w[j],worlds,current,rec=best
 report["accepted"].append(rec);dest=O/f"stable-21-unadmitted-iteration-{it}.npz"
 np.savez(dest,vertices6=z["vertices6"],faces=f,binding_indices=local,weights=w,face_origins=z["face_origins"])
 rec.update(candidate_path=str(dest),candidate_sha256=sha(dest));save();print(json.dumps(rec),flush=True)
 if time.monotonic()>deadline:break
if all(a["count"]==0 for a in current.values()):report["selected"]=str(dest)
report.update(complete=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()),remaining_counts=[a["count"] for a in current.values()],budget_expired=time.monotonic()>deadline)
save();print(json.dumps({k:report[k] for k in ["complete","remaining_counts","selected","budget_expired"]}),flush=True)

from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys,time
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}";O=D/"source-after-posed-correction-008";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
from numilab_human.resting_anatomy_interface_patch import normals
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";J=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-012/solve_terminal_self_pairs_012.py"
def module(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
h=module(H,"h");jo=module(J,"jo");sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C=D/f"posed-local-separation-005/stable-{sid}-unadmitted-local-candidate.npz";z=np.load(C);p=z["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"];w=z["weights"]
T=A/"fifty-five-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,sid)
ar=json.loads((A/"remaining-original-boundary-batch-001/stable-121/current50-target-audit-001/report.json").read_text());receipts=[Path(x) for x in ar["pins"] if x.endswith(".receipt.json") and "/accepted-geometry/" in x];assert len(receipts)==25
pins={str(x):sha(x) for x in [Path(__file__),C,T,H,J,Path(ca.__file__),Path(ci.__file__),*receipts]}
def audit(x):
 rec,dg=h.exact_rows(x,f,ci);return ci._audit_pair(rec,rec,same_surface=True)["triangle_pairs"],dg
contexts=[]
for rp in receipts:
 d=json.loads(rp.read_text());pm={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in d.get("accepted_body_poses",d.get("accepted_registered_body_poses"))}
 world=h.forward(p,local,w,row,t,pm);pairs,dg=audit(world);assert not dg
 contexts.append(dict(receipt=str(rp),step=d["accepted_step"],pm=pm,world=world,pairs=pairs))
reference_path=D/f"source-local-translation-002/stable-{sid}-geometric-reference-row-patch.npz";reference=np.load(reference_path)["vertices6"][:,:3];pins[str(reference_path)]=sha(reference_path)
clear=[c for c in contexts if not c["pairs"]];assert clear
samples=[];selected=[];seedset=set()
for c in contexts:
 if not c["pairs"]:continue
 if "/fifty-surface-" not in c["receipt"] and c["step"] not in [0,47519,155000,8000]:continue
 pairs=[(a,b) for a,b in c["pairs"] if not set(f[a])&set(f[b])]
 if not pairs:continue
 ids=np.unique(f[np.unique(pairs)]);ref=dict(receipt=str(reference_path),world=h.forward(reference,local,w,row,t,c["pm"]))
 assert not audit(ref["world"])[0]
 jac=jo.pose_jacobians(p,local,w,row,t,c["pm"])
 samples.append(dict(vertex_ids=ids,baseline_world_positions=np.ascontiguousarray(ref["world"][ids],dtype="<f4"),rejected_world_positions=np.ascontiguousarray(c["world"][ids],dtype="<f4"),jacobians=np.ascontiguousarray(jac[ids],float),self_pair_rows=np.asarray(pairs,int)))
 selected.append(dict(receipt=c["receipt"],reference_receipt=ref["receipt"],pairs=pairs));seedset.update(map(int,ids))
assert 0<len(samples)<=17
pure=np.flatnonzero(w.max(1)>=.999999);seeds=np.asarray(sorted(seedset-set(pure)),int);movable=set(seeds)
for face in f:
 if set(face)&set(seeds):movable.update(map(int,face))
movable=np.asarray(sorted(movable-set(pure)),int)
out={"scope":"Existing bounded local self-separation owner, using exact-clear retained poses and unchanged canonical binding maps. All onehot attachment proxy coordinates fixed. Source and all25 retained contexts rechecked; external/native admission remains required.","pins":pins,"pose_samples":selected,"movable_count":len(movable),"source_candidate":str(C),"candidate":None,"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)+"\n")
save()
try:
 result=ca._propose_local_self_separation_increment(source_positions_m=p,faces=f,movable_vertex_ids=movable,self_patch_seed_vertex_ids=seeds,required_target_seed_vertex_ids=np.asarray([],int),active_target_face_vertex_ids=np.asarray([],int),fixed_support_vertex_ids=pure,preserved_anchor_vertex_ids=pure,pose_samples=samples)
 x=result["proposal_source_positions_m_f32"];out["helper"]={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
 assert np.array_equal(x[pure],p[pure]);out["maximum_source_increment_m"]=float(np.linalg.norm(x.astype(float)-p,axis=1).max());out["cumulative_displacement_m"]=float(np.linalg.norm(x.astype(float)-reference,axis=1).max());assert out["cumulative_displacement_m"]<=.001
 out["checks"]=[]
 for c in [dict(receipt="source",pm=None),*contexts]:
  world=x if c["pm"] is None else h.forward(x,local,w,row,t,c["pm"]);pairs,dg=audit(world);out["checks"].append(dict(receipt=c["receipt"],pairs=pairs,degenerate_faces=dg))
 data={k:z[k] for k in z.files};data["vertices6"]=np.column_stack((x,normals(x.astype(float),f))).astype(np.float32);dst=O/f"stable-{sid}-unadmitted-local-candidate.npz";np.savez(dst,**data)
 out["candidate"]={"path":str(dst),"sha256":sha(dst),"source_all25_self_clear":all(not c["pairs"] and not c["degenerate_faces"] for c in out["checks"])}
except Exception as e:out["error"]=type(e).__name__+": "+str(e)
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save();print(json.dumps({k:out.get(k) for k in ["candidate","error","maximum_source_increment_m","movable_count"]}),flush=True)

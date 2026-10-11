from pathlib import Path
from fractions import Fraction
import sys,json,importlib.util
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-left-posed-union-source-lift-diagnostic-051";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
def mod(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";h=mod(H,"h")
L=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py";lift=mod(L,"lift")
J=L.parent.parent/"attempt-012/solve_terminal_self_pairs_012.py";jo=mod(J,"jo")
N=A/"ehl-restored-pair-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";C=A/"edl-left-protected-interface-binding-034/stable-22-unadmitted-iteration-1.npz";U=A/"edl-left-posed-positive-union-049/stable-22-posed-exact-union.json";UR=U.parent/"report.json"
pins={str(x):h.sha(x) for x in [Path(__file__),H,L,J,T,C,U,UR,Path(ci.__file__),Path(cg.__file__)]}
check=json.loads(UR.read_text());assert check["complete"] and check["inputs_unchanged"] and check["posed_float32_self"]["count"]==0
z=dict(np.load(C));p=z["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"];w=z["weights"];t=h.load_tissue(T);row=h.row_data(t,22)
poses={}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 rp=N/f"baseline/native-run/accepted-geometry/step-{step}.receipt.json";pins[str(rp)]=h.sha(rp);d=json.loads(rp.read_text())
 poses[step]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in d["accepted_body_poses"]}
world=h.forward(p,local,w,row,t,poses[8000]);exact=[tuple(Fraction.from_float(float(v)) for v in x) for x in world]
doc=json.loads(U.read_text());mesh=doc["construction"]["output_mesh"]
op=[tuple(lift.rational_pair(v) for v in x) for x in mesh["vertices_m"]];of=np.asarray(mesh["triangles"],np.uint32);parents=np.array([r["source_face"] for r in doc["ancestry_by_output_face"]],int)
lookup={}
for j,key in enumerate(exact):lookup.setdefault(key,[]).append(j)
incident=[[] for _ in op]
for i,tri in enumerate(of):
 for j in tri:incident[j].append(i)
outp=np.zeros((len(op),3),np.float32);outw=np.zeros((len(op),4),np.float32);outl=np.full((len(op),4),4294967295,np.uint32)
proof=[];new=[];originalmap={}
def source_map(j):
 return {int(local[j,k]):Fraction.from_float(float(w[j,k])) for k in range(4) if w[j,k]>0}
for j,point in enumerate(op):
 if point in lookup:
  old=lookup[point][0];assert all(np.array_equal(p[old],p[k]) and np.array_equal(local[old],local[k]) and np.array_equal(w[old],w[k]) for k in lookup[point])
  outp[j]=p[old];outl[j]=local[old];outw[j]=w[old];originalmap[old]=j;continue
 maps={};sourcepoints=[]
 for fi in incident[j]:
  sf=int(parents[fi]);ids=f[sf];lam=lift.barycentric_exact(point,[exact[k] for k in ids])
  weights={}
  for a,k in zip(lam,ids):
   for li,ww in source_map(k).items():weights[li]=weights.get(li,Fraction(0))+a*ww
  maps[lift.map_sig(weights)]=weights
  sourcepoints.append(sum((float(a)*p[k].astype(float) for a,k in zip(lam,ids)),np.zeros(3)))
 vals=list(maps.values());mean={k:sum((m.get(k,Fraction(0)) for m in vals),Fraction(0))/len(vals) for k in range(row["bc"])}
 assert all(x>=0 for x in mean.values()) and abs(float(sum(mean.values())-1))<1e-5
 outl[j,:row["bc"]]=np.arange(row["bc"]);outw[j,:row["bc"]]=[np.float32(float(mean[k])) for k in range(row["bc"])]
 target=np.asarray([lift.round_f32_exact(x) for x in point],np.float32)
 guess=np.mean(sourcepoints,axis=0).astype(np.float32);jac=jo.pose_jacobians(guess[None,:],outl[j:j+1],outw[j:j+1],row,t,poses[8000])[0]
 best=None
 for iteration in range(5):
  fw=h.forward(guess[None,:],outl[j:j+1],outw[j:j+1],row,t,poses[8000])[0]
  err=float(np.linalg.norm(fw.astype(float)-target));trial=(err,guess.copy())
  if best is None or err<best[0]:best=trial
  guess=(guess.astype(float)+np.linalg.solve(jac,target.astype(float)-fw)).astype(np.float32)
 outp[j]=best[1];new.append(j)
 proof.append({"vertex":j,"parent_faces":sorted(set(int(parents[i]) for i in incident[j])),"distinct_parent_map_count":len(vals),"world_inverse_error_m":best[0],"max_source_ancestry_point_distance_m":float(max(np.linalg.norm(outp[j].astype(float)-sp) for sp in sourcepoints))})
assert all(i in originalmap for i in range(len(p))),"posed union removed original vertices"
pure=np.flatnonzero(w.max(1)>=.999999)
assert all(np.array_equal(outp[originalmap[i]],p[i]) and np.array_equal(outl[originalmap[i]],local[i]) and np.array_equal(outw[originalmap[i]],w[i]) for i in pure)
report={"scope":"Unadmitted source lift of existing exact positive-winding posed boundary. Original source vertices and their maps are byte-preserved; new union vertices use existing exact ancestry interpolation/mean map and inverse existing affine bindings. Inferred geometry, not measurement. Source/all8poses and all-neighbor checks required before native admission.","pins":pins,"source_coordinate_count_retained":len(originalmap),"new_vertex_count":len(new),"new_vertex_proofs":proof,"pure_proxy_count":len(pure),"source_ancestry_distance_cap_m":.0005,"complete":False,"candidate_admitted":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
save()
report["passes_050_generated_point_ancestry_distance_cap"]=max(x["max_source_ancestry_point_distance_m"] for x in proof)<=.0005
report["qualification_limit"]="Diagnostic only: generated source points exceed the earlier0.5mm ancestry-point guard. Original source vertices have zero displacement. No bound waiver or native admission is implied."
assert max(x["world_inverse_error_m"] for x in proof)<=.0000006
report["audits"]=[]
for name,v in [("source",outp),*[(str(s),h.forward(outp,outl,outw,row,t,pm)) for s,pm in poses.items()]]:
 rr,d=h.exact_rows(v,of,ci);aa=ci._audit_pair(rr,rr,same_surface=True) if not d else {"count":None,"triangle_pairs":[]}
 top=cg.analyze_topology(v.astype(float).tolist(),of.tolist())
 report["audits"].append({"pose":name,"self":aa["count"],"pairs":aa["triangle_pairs"],"degenerate":d,"closed":top["closed_oriented_manifold_candidate"],"component_volumes":lift.signed_volumes(top,v,of)})
 print(json.dumps(report["audits"][-1]),flush=True);save()
orig=z["face_origins"][parents]
dest=O/"stable-22-unadmitted-posed-union-lift.npz";np.savez(dest,vertices6=np.column_stack([outp,h.vertex_normals(outp,of)]).astype(np.float32),faces=of,binding_indices=outl,weights=outw,face_origins=orig)
report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),candidate_path=str(dest),candidate_sha256=h.sha(dest),all_source_and_pose_self_clear=all(x["self"]==0 and not x["degenerate"] and x["closed"] for x in report["audits"]));save()
print(json.dumps({k:report[k] for k in ["complete","all_source_and_pose_self_clear","candidate_path"]}),flush=True)

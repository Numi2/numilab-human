from pathlib import Path
import json,sys,importlib.util,itertools,time
from fractions import Fraction
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-left-posed-union-ancestry-map-sensitivity-053";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg,common_atlas_skin_clearance as ca
def mod(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";h=mod(H,"h")
L=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py";lift=mod(L,"lift")
J=L.parent.parent/"attempt-012/solve_terminal_self_pairs_012.py";jo=mod(J,"jo")
N=A/"right-edl-ehl-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
C=A/"edl-left-protected-interface-binding-034/stable-22-unadmitted-iteration-1.npz";B=A/"edl-left-positive-union-lift-003/stable-22-reference-union-row-patch.npz"
U=A/"edl-left-posed-positive-union-049/stable-22-posed-exact-union.json";R=A/"edl-left-posed-union-current-target-audit-052/report.json"
pins={str(x):h.sha(x) for x in [Path(__file__),H,L,J,T,C,B,U,R,Path(ci.__file__),Path(cg.__file__),Path(ca.__file__)]}
z=dict(np.load(C));b=dict(np.load(B));p=z["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"];w=z["weights"];t=h.load_tissue(T);row=h.row_data(t,22)
steps=[0,4767,5023,5599,6207,6815,7423,8000];poses={}
for step in steps:
 rp=N/f"baseline/native-run/accepted-geometry/step-{step}.receipt.json";pins[str(rp)]=h.sha(rp);d=json.loads(rp.read_text())
 poses[step]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in d["accepted_body_poses"]}
world=h.forward(p,local,w,row,t,poses[8000]);exact=[tuple(Fraction.from_float(float(v)) for v in x) for x in world]
doc=json.loads(U.read_text());mesh=doc["construction"]["output_mesh"];op=[tuple(lift.rational_pair(v) for v in x) for x in mesh["vertices_m"]];of=np.asarray(mesh["triangles"],np.uint32);parents=np.asarray([r["source_face"] for r in doc["ancestry_by_output_face"]],int)
lookup={}
for j,key in enumerate(exact):lookup.setdefault(key,[]).append(j)
incident=[[] for _ in op]
for i,tri in enumerate(of):
 for j in tri:incident[j].append(i)
basep=np.zeros((len(op),3),np.float32);basew=np.zeros((len(op),4),np.float32);basel=np.full((len(op),4),4294967295,np.uint32);new=[];proof=[]
def smap(j):return {int(local[j,k]):Fraction.from_float(float(w[j,k])) for k in range(4) if w[j,k]>0}
for j,point in enumerate(op):
 if point in lookup:
  old=lookup[point][0];basep[j]=p[old];basel[j]=local[old];basew[j]=w[old];continue
 groups={}
 for fi in incident[j]:
  sf=int(parents[fi]);ids=f[sf];lam=lift.barycentric_exact(point,[exact[k] for k in ids]);weights={}
  for a,k in zip(lam,ids):
   for li,ww in smap(k).items():weights[li]=weights.get(li,Fraction(0))+a*ww
  sig=lift.map_sig(weights);sp=sum((float(a)*p[k].astype(float) for a,k in zip(lam,ids)),np.zeros(3))
  groups.setdefault(sig,{"map":weights,"points":[],"faces":[]});groups[sig]["points"].append(sp);groups[sig]["faces"].append(sf)
 gs=[groups[k] for k in sorted(groups)];assert len(gs)==2
 new.append((j,gs));proof.append({"vertex":j,"groups":[{"faces":sorted(set(g["faces"])),"map":{str(k):float(v) for k,v in g["map"].items()},"source_mean":np.mean(g["points"],0).tolist()} for g in gs]})
assert len(new)==3
pack=N/"baseline/native-run/accepted-geometry/step-0.mrvpack";pins[str(pack)]=h.sha(pack)
pos,surfs,_=ca._pack_surfaces(pack,{(51004,21)});tf=np.asarray(surfs[(51004,21)]["faces"],int);ids=np.unique(tf);tr,deg=h.exact_rows(pos[ids],np.searchsorted(ids,tf),ci);assert not deg
rr,deg=h.exact_rows(h.forward(b["vertices6"][:,:3],b["binding_indices"],b["weights"],row,t,poses[0]),b["faces"],ci);assert not deg
oldpairs=set(tuple(x) for x in ci._audit_pair(rr,tr,same_surface=False)["triangle_pairs"])
def audit(p):
 rr,d=h.exact_rows(p,of,ci)
 return {"degenerate":d,"pairs":ci._audit_pair(rr,rr,same_surface=True)["triangle_pairs"] if not d else []}
report={"scope":"Diagnostic sensitivity over exact ancestry-derived display maps for only three generated posed-union vertices. Original source vertices and maps remain fixed. No native admission; previous050 ancestry-distance failure retained. Alpha blends two actual incident ancestry maps, followed by inversion of the existing affine owner at step8000. Full neighbor checks remain required.","pins":pins,"new_vertex_ancestry":proof,"trials":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
trials=list(dict.fromkeys([tuple([a]*3) for a in [0,.1,.25,.5,.75,.9,1]]+list(itertools.product([0,.5,1],repeat=3))))
for alphas in trials:
 outp=basep.copy();outw=basew.copy();outl=basel.copy();vp=[]
 for (j,gs),alpha in zip(new,alphas):
  mean={k:(1-alpha)*float(gs[0]["map"].get(k,0))+alpha*float(gs[1]["map"].get(k,0)) for k in range(row["bc"])}
  outl[j,:row["bc"]]=np.arange(row["bc"]);outw[j,:row["bc"]]=list(mean.values())
  target=np.asarray([lift.round_f32_exact(x) for x in op[j]],np.float32)
  sp0=np.mean(gs[0]["points"],0);sp1=np.mean(gs[1]["points"],0)
  guess=((1-alpha)*sp0+alpha*sp1).astype(np.float32);jac=jo.pose_jacobians(guess[None,:],outl[j:j+1],outw[j:j+1],row,t,poses[8000])[0];best=None
  for k in range(5):
   fw=h.forward(guess[None,:],outl[j:j+1],outw[j:j+1],row,t,poses[8000])[0];err=float(np.linalg.norm(fw.astype(float)-target))
   if best is None or err<best[0]:best=(err,guess.copy())
   guess=(guess.astype(float)+np.linalg.solve(jac,target.astype(float)-fw)).astype(np.float32)
  outp[j]=best[1];vp.append({"vertex":j,"inverse_world_error_m":best[0],"distance_to_source_segment_m":float(np.linalg.norm(best[1].astype(float)-((1-alpha)*sp0+alpha*sp1))),"max_source_ancestry_distance_m":float(max(np.linalg.norm(outp[j].astype(float)-sp) for g in gs for sp in g["points"]))})
 p0=h.forward(outp,outl,outw,row,t,poses[0]);rr,deg=h.exact_rows(p0,of,ci);assert not deg
 pairs=ci._audit_pair(rr,tr,same_surface=False)["triangle_pairs"];after={(int(parents[i]),int(j)) for i,j in pairs};added=sorted(after-oldpairs)
 ent={"alphas":alphas,"vertex_proofs":vp,"initial_bone21_strict_added":added,"audits":[]};report["trials"].append(ent)
 if not added:
  for name,pp in [("source",outp),*[(str(s),h.forward(outp,outl,outw,row,t,poses[s])) for s in steps]]:
   aa=audit(pp);ent["audits"].append({"pose":name,**aa})
   if aa["pairs"] or aa["degenerate"]:break
  if len(ent["audits"])==9 and all(not a["pairs"] and not a["degenerate"] for a in ent["audits"]):
   dest=O/("stable-22-unadmitted-alpha-"+"-".join(str(a) for a in alphas)+".npz");np.savez(dest,vertices6=np.column_stack([outp,h.vertex_normals(outp,of)]).astype(np.float32),faces=of,binding_indices=outl,weights=outw,face_origins=z["face_origins"][parents]);ent.update(candidate=str(dest),sha256=h.sha(dest))
 print(json.dumps({"alphas":alphas,"added":len(added),"self":[[a["pose"],len(a["pairs"])] for a in ent["audits"]],"candidate":ent.get("candidate")}),flush=True);save()
report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()));save()

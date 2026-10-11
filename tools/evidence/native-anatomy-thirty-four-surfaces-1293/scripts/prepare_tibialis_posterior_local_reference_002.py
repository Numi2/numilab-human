from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"tibialis-posterior-local-reference-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy import _split_disconnected_vertex_fans
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__),Path(sys.modules[_split_disconnected_vertex_fans.__module__].__file__)]}
out={"scope":"Explicit bounded local reference inference after exact positive-winding construction. Geometry only; source and native gates separate.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for sid in (59,60):
 cp=A/f"existing-body-{sid}-pinched-reference-001/stable-{sid}-reference-union-row-patch.npz";wp=A/f"existing-body-{sid}-rejected-boundary-004/rejected-boundary-witness.json";pins[str(cp)]=sha(cp);pins[str(wp)]=sha(wp)
 z=np.load(cp);p=z["vertices6"][:,:3];f=z["faces"];w=json.loads(wp.read_text())
 top=analyze_topology(p.astype(float).tolist(),f.tolist());v,ff,edits=_split_disconnected_vertex_fans(p.tolist(),f.tolist(),top["vertex_manifold_defect_ids"])
 pp=np.asarray(v,np.float32);ff=np.asarray(ff,int);assert np.array_equal(pp[ff],p[f]);original=pp.copy()
 mapping=list(range(len(p)));moved=[]
 for e in edits:
  moved.append(e["source_vertex_id"])
  for j in e["copied_vertex_ids"]:assert j==len(mapping);mapping.append(e["source_vertex_id"]);moved.append(j)
 local=z["binding_indices"][mapping];weights=z["weights"][mapping]
 for j in moved:
  assert weights[j].max()<.999999
  nbr=np.unique(ff[np.any(ff==j,axis=1)]);nbr=nbr[nbr!=j];d=original[nbr].astype(float).mean(0)-original[j];pp[j]=(original[j].astype(float)+1e-6*d/np.linalg.norm(d)).astype(np.float32)
 def full():
  u,iv=np.unique(pp,axis=0,return_inverse=True);tt=analyze_topology(u.astype(float).tolist(),iv[ff].tolist())
  rec,deg=h.exact_rows(pp,ff,ci)
  aa=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
  return tt,deg,aa
 def bad(pair):
  fi,fj=ff[pair[0]],ff[pair[1]]
  ti=tuple(tuple(Fraction.from_float(float(q)) for q in pp[int(z)]) for z in fi)
  tj=tuple(tuple(Fraction.from_float(float(q)) for q in pp[int(z)]) for z in fj)
  hits=set(ci.triangle_intersection_points(ti,tj));shared={tuple(Fraction.from_float(float(q)) for q in pp[z]) for z in set(fi)&set(fj)}
  return bool(hits-shared)
 tt,deg,aa=full();assert not deg and tt["closed_oriented_manifold_candidate"]
 rr={"stable_id":sid,"initial_self_pairs":aa["triangle_pairs"],"edits":[],"selected":None};out["rows"].append(rr);save()
 for step in range(8):
  pairs=[tuple(x) for x in aa["triangle_pairs"]]
  if not pairs:break
  candidates=[];verts=np.unique(ff[np.unique(pairs)])
  for j in verts:
   if weights[j].max()>=.999999:continue
   old=pp[j].copy()
   affected=[pair for pair in pairs if j in ff[list(pair)]]
   for axis in range(3):
    for sign in (-1,1):
     for microns in (.1,1,10,50):
      pp[j,axis]=np.float32(float(old[axis])+sign*microns*1e-6)
      if np.linalg.norm(pp[j].astype(float)-original[j])>1e-4:continue
      remain=sum(bad(pair) for pair in affected)
      if remain<len(affected):candidates.append((len(pairs)-len(affected)+remain,microns,int(j),axis,sign,float(pp[j,axis])))
      pp[j]=old
   pp[j]=old
  accepted=False
  for score,microns,j,axis,sign,value in sorted(candidates):
   old=pp[j].copy();pp[j,axis]=np.float32(value)
   nt,nd,na=full()
   if nt["closed_oriented_manifold_candidate"] and not nd and na["count"]<aa["count"]:
    rr["edits"].append({"vertex":j,"axis":axis,"sign":sign,"microns":microns,"old":old.tolist(),"new":pp[j].tolist(),"self_before":aa["count"],"self_after":na["count"]});aa=na;accepted=True;save();print(json.dumps({"sid":sid,**rr["edits"][-1]}),flush=True);break
   pp[j]=old
  if not accepted:break
 tt,deg,aa=full();rr.update(final_self_count=aa["count"],final_pairs=aa["triangle_pairs"],closed=tt["closed_oriented_manifold_candidate"],degenerate=deg,max_boundary_delta_m=float(np.linalg.norm(pp.astype(float)-original.astype(float),axis=1).max()))
 if rr["closed"] and not deg and aa["count"]==0:
  dest=O/f"stable-{sid}-local-reference.npz";np.savez(dest,vertices6=np.concatenate([pp,h.vertex_normals(pp,ff)],axis=1).astype("<f4"),faces=ff.astype("<u4"),binding_indices=local,weights=weights,face_origins=np.asarray([int(x["source_face"]) for x in w["output_face_ancestry"]],int))
  rr["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"microns":rr["max_boundary_delta_m"]*1e6}
 save();print(json.dumps({k:v for k,v in rr.items() if k not in ("edits","initial_self_pairs")}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pectoral-left-generated-boundary-repair-018";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy import _split_disconnected_vertex_fans
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__),Path(sys.modules[_split_disconnected_vertex_fans.__module__].__file__)]}
out={"scope":"Explicit bounded adjustment of generated boundary vertices after restoring original pectoral source points. All original one-hot source positions are fixed; generated points with one-hot maps may move. No physical routes or binding maps change; source and posed/native gates remain mandatory.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for sid in (74,):
 cp=A/"pectoral-left-junction-reference-016/stable-74-junction-1-microns.npz";rp=A/"pectoral-left-source-point-restoration-017/report.json";T=A/"thirty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
 for inp in (cp,rp,T):pins[str(inp)]=sha(inp)
 z=np.load(cp);pp=z["vertices6"][:,:3].copy();ff=z["faces"].copy();local=z["binding_indices"].copy();weights=z["weights"].copy()
 restored=json.loads(rp.read_text())
 for e in restored["edits"]:
  j=e["candidate_vertex"];pp[j]=e["restored_position"];weights[j]=e["source_weights"];local[j]=e["source_local"]
 original=pp.copy();raw=h.row_data(h.load_tissue(T),74)
 pure=np.unique(raw["faces"]);pure=pure[raw["weights"][pure].max(1)>=.999999]
 frozen={raw["positions"][j].tobytes() for j in pure}
 locked={j for j in range(len(pp)) if pp[j].tobytes() in frozen}
 # Original source-point proxies stay fixed; only generated boundary points can move.
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
   if int(j) in locked:continue
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
  dest=O/f"stable-{sid}-local-reference.npz";np.savez(dest,vertices6=np.concatenate([pp,h.vertex_normals(pp,ff)],axis=1).astype("<f4"),faces=ff.astype("<u4"),binding_indices=local,weights=weights,face_origins=z["face_origins"])
  rr["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"microns":rr["max_boundary_delta_m"]*1e6}
 save();print(json.dumps({k:v for k,v in rr.items() if k not in ("edits","initial_self_pairs")}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


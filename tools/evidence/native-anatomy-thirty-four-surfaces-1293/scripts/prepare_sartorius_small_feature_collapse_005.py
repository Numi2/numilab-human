from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
from scipy.spatial import cKDTree
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"sartorius-small-feature-collapse-005";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy import _split_disconnected_vertex_fans
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,Path(ci.__file__),Path(sys.modules[analyze_topology.__module__].__file__),Path(sys.modules[_split_disconnected_vertex_fans.__module__].__file__)]}
out={"scope":"Explicit reference removal of unrepresentably small local features; construction changes, not validator tolerance. Not admitted.","pins":pins,"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for sid in (49,50):
 cp=A/f"existing-body-{sid}-pinched-reference-001/stable-{sid}-reference-union-row-patch.npz";wp=A/f"existing-body-{sid}-rejected-boundary-004/rejected-boundary-witness.json";pins[str(cp)]=sha(cp);pins[str(wp)]=sha(wp)
 z=np.load(cp);p=z["vertices6"][:,:3];f=z["faces"];w=json.loads(wp.read_text())
 top=analyze_topology(p.astype(float).tolist(),f.tolist());v,ff,edits=_split_disconnected_vertex_fans(p.tolist(),f.tolist(),top["vertex_manifold_defect_ids"])
 pp=np.asarray(v,np.float32);ff=np.asarray(ff,int);assert np.array_equal(pp[ff],p[f])
 mapping=list(range(len(p)));moved=[]
 for e in edits:
  moved.append(e["source_vertex_id"])
  for j in e["copied_vertex_ids"]:assert j==len(mapping);mapping.append(e["source_vertex_id"]);moved.append(j)
 local=z["binding_indices"][mapping];weights=z["weights"][mapping];norm=h.vertex_normals(pp,ff).astype(float)
 origins=np.asarray([int(x["source_face"]) for x in w["output_face_ancestry"]],int)
 rr={"stable_id":sid,"trials":[]};out["rows"].append(rr)
 for radius in (1e-7,1e-6,1e-5):
  pairs=[(i,j) for i,j in cKDTree(pp).query_pairs(radius) if i not in moved and j not in moved]
  parent=np.arange(len(pp))
  def root(i):
   while parent[i]!=i:i=parent[i]
   return i
  for i,j in sorted(pairs):
   a,b=root(i),root(j)
   if a!=b:parent[max(a,b)]=min(a,b)
  m=np.array([root(i) for i in range(len(pp))]);cf=m[ff];keep=np.array([len(set(t))==3 for t in cf]);dropped=np.flatnonzero(~keep)
  nf=cf[keep];orig=origins[keep]
  for mode in ("outward","centroid","inward"):
   for microns in (1,10,100):
    trial=pp.copy()
    for j in moved:
     assert weights[j].max()<.999999
     if mode=="centroid":
      nbr=np.unique(ff[np.any(ff==j,axis=1)]);nbr=nbr[nbr!=j];d=pp[nbr].astype(float).mean(0)-pp[j];d/=np.linalg.norm(d)
     else:d=norm[j]*(1 if mode=="outward" else -1)
     trial[j]=(pp[j].astype(float)+microns*1e-6*d).astype(np.float32)
    used=np.unique(nf);remap=np.full(len(pp),-1);remap[used]=np.arange(len(used));qq=trial[used];qf=remap[nf]
    u,iv=np.unique(qq,axis=0,return_inverse=True);tt=analyze_topology(u.astype(float).tolist(),iv[qf].tolist());count=None;deg=[]
    if tt["closed_oriented_manifold_candidate"]:
     rec,deg=h.exact_rows(qq,qf,ci)
     if not deg:count=ci._audit_pair(rec,rec,same_surface=True)["count"]
    tr={"radius_m":radius,"mode":mode,"microns":microns,"merged":pairs,"removed_face_ids":dropped.tolist(),"closed":tt["closed_oriented_manifold_candidate"],"self_count":count,"degenerate":deg}
    if tr["closed"] and count==0 and not deg:
     dest=O/f"stable-{sid}-{mode}-{microns}um-radius-{radius}.npz"
     np.savez(dest,vertices6=np.concatenate([qq,h.vertex_normals(qq,qf)],axis=1).astype("<f4"),faces=qf.astype("<u4"),binding_indices=local[used],weights=weights[used],face_origins=orig)
     tr.update(candidate_path=str(dest),candidate_sha256=sha(dest))
    rr["trials"].append(tr);save();print(json.dumps({"sid":sid,**{k:v for k,v in tr.items() if k not in ("merged","removed_face_ids","degenerate")}}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()


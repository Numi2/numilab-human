from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/remaining-original-boundary-batch-001/stable-137/post-correction-boundary-006');O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_partition as part,cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy_interface_patch import signed_volume
H=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/remaining-original-boundary-batch-001/stable-137/posed-local-separation-005/stable-137-unadmitted-local-candidate.npz');z=np.load(p)
pins={str(x):sha(x) for x in [p,H,Path(__file__),Path(part.__file__),Path(ci.__file__)]}
v,iv=np.unique(z["vertices6"][:,:3],axis=0,return_inverse=True);f=iv[z["faces"]];exact=[tuple(Fraction.from_float(float(c)) for c in p) for p in v]
mesh=None
try:
 union=part.construct_positive_winding_self_union({"vertices":exact,"triangles":f.tolist(),"source_sha256":sha(p)})
 mesh=union["output_mesh"];ancestry=union["output_face_ancestry"];error=None
except Exception as e:
 error=str(e);tb=e.__traceback__
 while tb:
  if tb.tb_frame.f_code.co_name=="_construct_winding_self_union":
   loc=tb.tb_frame.f_locals;mesh=loc.get("output_mesh");ancestry=loc.get("output_ancestry")
  tb=tb.tb_next
assert mesh and ancestry
(O/"rejected-partition.json").write_text(json.dumps(part.encode_rational({"original_error":error,"mesh":mesh,"face_ancestry":ancestry}),sort_keys=True)+"\n")
v=np.asarray([[float(c) for c in p] for p in mesh["vertices_m"]]);f=np.asarray(mesh["triangles"],int)
splits=[]
for vid in range(len(v)):
 incident=np.flatnonzero(np.any(f==vid,axis=1));adj={}
 for face in f[incident]:
  a,b=[int(x) for x in face if x!=vid];adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
 assert all(len(s)==2 for s in adj.values()),"not purely point-link pathology"
 pending=set(adj);components=[]
 while pending:
  stack=[min(pending)];comp=set()
  while stack:
   x=stack.pop()
   if x in comp:continue
   comp.add(x);stack.extend(adj[x]-comp)
  pending-=comp;components.append(comp)
 if len(components)>1:
  rr=[]
  for comp in components:
   ids=[int(i) for i in incident if set(int(x) for x in f[i] if x!=vid)<=comp]
   direction=v[sorted(comp)].mean(0)-v[vid];direction/=np.linalg.norm(direction)
   rr.append({"faces":ids,"direction":direction.tolist(),"link_vertices":sorted(comp)})
  matches=np.flatnonzero(np.all(z["vertices6"][:,:3].astype(float)==v[vid],axis=1))
  assert len(matches) and not np.any(z["weights"][matches].max(1)>=.999999),"only mapped non-proxy source pinches"
  splits.append({"vertex":vid,"parent_vertices":matches.tolist(),"components":rr})
out={"scope":"Diagnostic geometric regularization of a rejected partition's isolated point pinches. Splits receive distinct displaced coordinates directed into each incident branch, followed by independent exact Float32 self/topology checks. This is inferred reference geometry, not an accepted unchanged positive union or a topology-only ID split. No native/pose/attribute admission.", "pins":pins,"rejected_error":error,"splits":splits,"trials":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for epsilon in ([0.0] if not splits else [1e-6,5e-6,1e-5,2.5e-5,5e-5]):
 nv=v.tolist();nf=f.copy();mapping=list(range(len(v)))
 for sp in splits:
  for k,comp in enumerate(sp["components"]):
   idx=sp["vertex"] if k==0 else len(nv)
   if k: nv.append(v[sp["vertex"]].tolist());mapping.append(sp["vertex"])
   nv[idx]=(v[sp["vertex"]]+epsilon*np.asarray(comp["direction"])).tolist()
   for faceid in comp["faces"]:nf[faceid,nf[faceid]==sp["vertex"]]=idx
 nv=np.asarray(nv,np.float32);top=analyze_topology(nv.astype(float).tolist(),nf.tolist())
 rec,deg=h.exact_rows(nv,nf,ci);audit=ci._audit_pair(rec,rec,same_surface=True)
 row={"epsilon_m":epsilon,"vertex_count":len(nv),"topology":top,"self_pair_count":audit["count"],"pairs":audit["triangle_pairs"],"degenerate_faces":deg,"signed_volume_delta_from_rejected_partition_m3":signed_volume(nv.astype(float),nf)-signed_volume(v,f)}
 out["trials"].append(row);save()
 print(json.dumps({k:row[k] for k in ["epsilon_m","vertex_count","self_pair_count","degenerate_faces","signed_volume_delta_from_rejected_partition_m3"]}),flush=True)
 if audit["count"]==0 and not deg and top["closed_oriented_manifold_candidate"]:
  dst=O/f"source-geometric-candidate-{round(epsilon*1e6):03d}um.npz";np.savez(dst,positions=nv,faces=nf,partition_vertex_ancestry=np.asarray(mapping));row["candidate_path"]=str(dst);row["candidate_sha256"]=sha(dst)
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save()

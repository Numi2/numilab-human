from pathlib import Path
import sys,json,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-left-generated-source-characterization-055";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-complete-reference-inventory-1335/src")
from numilab_human import myosim_bone_proximity as mbp,cardiac_cavity_geometry as cg,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
B=A/"edl-left-protected-interface-binding-034/stable-22-unadmitted-iteration-1.npz";C=A/"edl-left-posed-union-ancestry-map-sensitivity-053/stable-22-unadmitted-alpha-0.75-0.75-0.75.npz";U=A/"edl-left-posed-positive-union-049/stable-22-posed-exact-union.json"
pins={str(x):h.sha(x) for x in [Path(__file__),H,B,C,U,Path(mbp.__file__),Path(cg.__file__),Path(ci.__file__)]}
b=dict(np.load(B));c=dict(np.load(C));p=b["vertices6"][:,:3];f=b["faces"];q=c["vertices6"][:,:3];g=c["faces"];parents=np.asarray([r["source_face"] for r in json.loads(U.read_text())["ancestry_by_output_face"]],int)
lookup={tuple(v):i for i,v in enumerate(p)};om={};new=[]
for j,v in enumerate(q):
 if tuple(v) in lookup:
  i=lookup[tuple(v)];om[i]=j
  assert np.array_equal(b["binding_indices"][i],c["binding_indices"][j]) and np.array_equal(b["weights"][i],c["weights"][j])
 else:new.append(j)
assert len(om)==len(p) and len(new)==3
tris=p[f].astype(float)
def nearest(point,subset):
 best=None
 for i in subset:
  qp,bc=mbp._closest_point_on_triangle(point,tris[i],np);d=float(np.linalg.norm(point-qp))
  if best is None or d<best[0]:best=(d,int(i),qp.tolist(),bc)
 return {"distance_m":best[0],"source_face":best[1],"closest_point":best[2],"barycentric":best[3]}
proof=[]
for j in new:
 incident=np.flatnonzero(np.any(g==j,axis=1));ancestor=sorted(set(parents[incident].tolist()))
 proof.append({"vertex":j,"source_position":q[j].tolist(),"incident_parent_faces":ancestor,"nearest_entire_original_source_surface":nearest(q[j].astype(float),range(len(f))),"nearest_incident_ancestry_surface":nearest(q[j].astype(float),ancestor)})
changed=[]
for i,tri in enumerate(g):
 if not np.array_equal(q[tri],p[f[parents[i]]]):changed.append(i)
sample=[]
for i in changed:
 tr=q[g[i]].astype(float);sf=int(parents[i])
 for kind,pt in [("centroid",tr.mean(0)),*[(f"edge{k}",(tr[k]+tr[(k+1)%3])/2) for k in range(3)]]:
  sample.append({"child_face":int(i),"kind":kind,"parent_face":sf,"distance_to_parent_triangle_m":nearest(pt,[sf])["distance_m"]})
top=cg.analyze_topology(q.astype(float).tolist(),g.tolist());components=[]
for k,comp in enumerate(top["face_components"]):
 ids=np.asarray(comp,int);tr=q[g[ids]].astype(float);tr-=tr.reshape(-1,3).mean(0)
 components.append({"id":k,"faces":len(ids),"signed_volume_m3":float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6)})
report={"scope":"Characterize generated reference boundary points, not measured new anatomy. All2289 existing source coordinates/maps are byte-preserved relative034; earlier050 max-distance-to-each-ancestry-point cap remains a failed check and is not relabeled. Triangle-distance and sampled child deformation are distinct additional metrics, not a full Hausdorff bound.","pins":pins,"original_vertices_preserved":len(om),"generated_vertex_count":len(new),"generated_points":proof,"changed_child_face_count":len(changed),"sampled_child_to_own_parent_triangle_max_m":max(x["distance_to_parent_triangle_m"] for x in sample),"sampled_child_points":sample,"source_closed":top["closed_oriented_manifold_candidate"],"component_volumes":components,"complete":True,"inputs_unchanged":all(h.sha(p)==v for p,v in pins.items())}
(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ["pins","sampled_child_points"]},indent=2))

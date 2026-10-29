from pathlib import Path
import numpy as np,csv,json,time,hashlib,sys
from scipy.spatial import cKDTree
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
R=P.parent
sys.path.insert(0,str(R/"authoring-deps"))
import manifold3d as mf
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
out=P/"source-cell-overlap-511/donor4-control";out.mkdir(parents=True,exist_ok=False)
def read(s):
 with (CAP/("compiled-parent-cell-mass.csv"+s)).open() as f:return list(csv.DictReader(f))
n=read(".nodes.csv");tr=read(".tets.csv");lab=read(".labels.csv")
pp=np.array([[float(r["rest_"+k+"_m"]) for k in "xyz"] for r in n]);pt=np.array([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tr])
selected=np.array([i for i,r in enumerate(lab) if [int(r[k]) for k in ["material_index","geometric_donor_body","anatomical_region_id"]]==[15,4,4]])
with np.load(P/"connected-measured-contact-472/run-005-capacity-gradient-ulp/candidate.npz") as z:x=z["points_reference_world_m_f64"].astype(np.float32).astype(float);ct=z["volume_tetrahedra"]
faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.uint32)
def solid(p):
 assert np.linalg.det(np.array([p[1]-p[0],p[2]-p[0],p[3]-p[0]]))>0
 m=mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(p),tri_verts=faces))
 assert m.status()==mf.Error.NoError,m.status()
 return m
ps={int(i):solid(pp[pt[i]]) for i in selected};cs={int(i):solid(x[ct[i]]) for i in selected}
rows=[]
for pi in selected:
 for ci in selected:
  isec=ps[int(pi)]^cs[int(ci)]
  v=float(isec.volume())
  rows.append({"parent_row":int(pi),"child_row":int(ci),"centroid_distance_m":float(np.linalg.norm(pp[pt[pi]].mean(0)-x[ct[ci]].mean(0))),"intersection_volume_m3":v,"within_436_centroid_radius":bool(np.linalg.norm(pp[pt[pi]].mean(0)-x[ct[ci]].mean(0))<=.002797),"strict_positive_overlap":v>1e-14})
r={"scope":"source-only geometric support diagnosis; actual positive overlap is not a complete remap, source closure, density, or physics qualification","group":[15,4,4],"rows":rows,"source_parent_cell_rows":selected.tolist(),"positive_intersection_edges_excluded_by_436_centroid_bound":[z for z in rows if z["strict_positive_overlap"] and not z["within_436_centroid_radius"]],"geometry_modified":False,"physical_admission":False}
(out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))


"""Existing-wire boundary refinement control; no anatomical repair claimed."""
from pathlib import Path
import csv,json,numpy as np,hashlib,argparse,importlib.util,time
P=Path(__file__).resolve().parent
B=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
D=P/"refined-native-producer-433/interface-refinement-control-442-retry-001";D.mkdir(exist_ok=False)
with np.load(B/"candidate.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
T=a["volume_tetrahedra"];p=a["points_reference_world_m_f64"].astype(np.float32).astype(float)
face=a["coarse_boundary_faces_I"][3933].astype(int);adj=np.flatnonzero(np.isin(T,face).sum(1)==3)
assert len(adj)==2 and set(a["material_index"][adj])=={15,16}
nid=len(p);pt=int(adj[a["material_index"][adj]==15][0]);beta=np.isin(T[pt],face).astype(float)/3
tt=[];cp=[]
def volume(x,t):
 y=x[np.array(t)];return np.dot(y[1]-y[0],np.cross(y[2]-y[0],y[3]-y[0]))/6
newp=np.vstack([p,p[face].mean(0).astype(np.float32).astype(float)])
for row,t in enumerate(T):
 if row not in adj:tt.append(t.tolist());cp.append(row);continue
 apex=int(t[~np.isin(t,face)][0])
 for k in range(3):
  c=[apex,int(face[k]),int(face[(k+1)%3]),nid]
  if volume(newp,c)<0:c[1],c[2]=c[2],c[1]
  assert volume(newp,c)>0
  tt.append(c);cp.append(row)
cp=np.asarray(cp,int);a["volume_tetrahedra"]=np.array(tt,int);a["child_parent_tet_row"]=cp
for key in ["material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]:a[key]=a[key][cp]
for key in ["points_reference_world_m_f64","points_current_world_m_f32","points_donor_local_m_f64","full86_weights","target_current_world_m"]:
 v=a[key];x=v[face].mean(0)
 if key in ["points_reference_world_m_f64","points_current_world_m_f32"]:x=x.astype(np.float32).astype(v.dtype)
 a[key]=np.vstack([v,x])
a["parent_volume_node_row"]=np.append(a["parent_volume_node_row"],-1)
np.savez_compressed(D/"candidate.npz",**a)
with (D/"parent-reduced-boundary.node").open("w") as f:
 f.write(f"{nid+1} 3 0 0\n")
 for i,x in enumerate(a["points_reference_world_m_f64"]):f.write(str(i+1)+" "+" ".join(format(float(v),".17g") for v in x)+"\n")
with (B/"coarse-node-to-fine-surface.csv").open(newline="") as f:r=csv.DictReader(f);header=r.fieldnames;rows=list(r)
new=dict(rows[int(face[0])]);new.update(volume_node_row=str(nid),role="I",role_row="2411",source_vertex_row="-1",source_face_row="-1",embedding_kind="derived_original_interface_centroid_protocol_control")
rows.append(new)
with (D/"coarse-node-to-fine-surface.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=header,lineterminator="\n");w.writeheader();w.writerows(rows)
with (D/"refined-node-parent-ancestry.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["volume_node_row","parent_tet_row",*["parent_bary"+str(k) for k in range(4)]]);w.writerow([nid,pt,*beta])
spec=importlib.util.spec_from_file_location("producer",P/"export_refined_coarse_registered_volume_433.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=argparse.Namespace(candidate_npz=D/"candidate.npz",candidate_node=D/"parent-reduced-boundary.node",candidate_node_map=D/"coarse-node-to-fine-surface.csv",inserted_node_ancestry=D/"refined-node-parent-ancestry.csv",capture=P/"native-physical-parent-mass-330/native-source-identity-control-003",out=D/"sidecars",maximum_mass_transport_distance_m=.02)
r=m.export(args)
with (args.capture/"parent-layer-rows.csv").open(newline="") as f:parent=list(csv.DictReader(f))
with Path(r["contact_thickness"]["I"]["path"]).open(newline="") as f:tr=list(csv.DictReader(f))
assert len(tr)==2412
for key in ["skin_thickness_m","fat_thickness_m"]:
 expected=float(beta@np.array([float(parent[n][key]) for n in T[pt]]))
 assert abs(float(tr[-1][key])-expected)<=4*np.spacing(expected)
report=dict(scope="I-boundary scalar/CSV protocol control only; source coarse geometry unresolved",nodes=len(newp),cells=len(tt),shared_I_nodes=2412,new_boundary_thickness_interpolation="PASS",source_face=face.tolist(),parent_cells=adj.tolist(),physical_admission=False,candidate_sha256=hashlib.sha256((D/"candidate.npz").read_bytes()).hexdigest())
(D/"result.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))

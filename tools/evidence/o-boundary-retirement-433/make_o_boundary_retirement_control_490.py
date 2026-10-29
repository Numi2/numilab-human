import csv,hashlib,importlib.util,json,math,pathlib,struct
import numpy as np
P=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
R=P/"refined-native-producer-433"/"o-boundary-retirement-490-retry-002"
BASE=P/"reduced-physical-boundary-400"/"attempt-002-parent-ancestry-walk"
CAP=P/"native-physical-parent-mass-330"/"native-source-identity-control-003"
OUT=R/"positive-retirement-control"
OUT.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location("producer433",P/"export_refined_coarse_registered_volume_433.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
 with path.open(newline="") as f:return list(csv.DictReader(f))
def write_csv(path,header,rows):
 with path.open("w",newline="") as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(header);w.writerows(rows)
def signed_volume(points,tet):
 a,b,c,d=(points[int(i)] for i in tet)
 return float(np.dot(b-a,np.cross(c-a,d-a))/6.0)
with np.load(BASE/"candidate.npz",allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
points=base["points_reference_world_m_f64"].astype(np.float32).astype(np.float64)
tets=base["volume_tetrahedra"]
pn=base["parent_volume_node_row"]
node=30
assert int(pn[node])==node
maps=read_csv(BASE/"coarse-node-to-fine-surface.csv")
maprow=next(r for r in maps if int(r["volume_node_row"])==node)
assert maprow["role"]=="O" and int(maprow["role_row"])==node
output_role_by_volume={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in maps}
face_uses={}
for cell,tet in enumerate(tets):
 for corner in m.CORNER_FACES:
  tri=tuple(int(tet[k]) for k in corner);face_uses.setdefault(tuple(sorted(tri)),[]).append((cell,tri))
star=[]
for uses in face_uses.values():
 if len(uses)==1:
  cell,tri=uses[0]
  if int(base["material_index"][cell])==15 and all(output_role_by_volume.get(n,(None,None))[0]=="O" for n in tri) and node in tri:
   star.append((cell,tri))
assert len(star)==7, f"expected local O star with 7 faces, got {len(star)}"
unit_normals=[]
for cell,tri in star:
 q=points[list(tri)];n=np.cross(q[1]-q[0],q[2]-q[0]);length=float(np.linalg.norm(n))
 assert length>0
 unit_normals.append(n/length)
direction=np.sum(unit_normals,axis=0);direction/=np.linalg.norm(direction)
min_outward_dot=float((np.asarray(unit_normals)@direction).min())
assert min_outward_dot>0.25, f"selected node is not a locally outward-convex star: {min_outward_dot}"
epsilon=20e-6
delta=-epsilon*direction
moved=np.asarray(points[node]+delta,dtype=np.float32).astype(np.float64)
actual_delta=moved-points[node]
new_node=len(points)
synthetic_points=np.vstack([points.copy(),points[node]])
synthetic_points[node]=moved
wedge_tets=[];wedge_parent=[]
for cell,tri in star:
 k=tri.index(node)
 oriented=(tri[k],tri[(k+1)%3],tri[(k+2)%3])
 tet=(node,new_node,oriented[1],oriented[2])
 volume=signed_volume(synthetic_points,tet)
 assert volume>0, f"outward wedge is nonpositive: {volume}"
 wedge_tets.append(tet);wedge_parent.append(int(base["child_parent_tet_row"][cell]))
new_tets=np.vstack([tets,np.asarray(wedge_tets,dtype=np.int64)])
new_cp=np.concatenate([base["child_parent_tet_row"],np.asarray(wedge_parent,dtype=np.int64)])
assert min(signed_volume(synthetic_points,t) for t in new_tets)>0
# Confirm the old source faces are now internal and the new faces close the same star.
new_face_uses={}
for cell,tet in enumerate(new_tets):
 for corner in m.CORNER_FACES:
  tri=tuple(int(tet[k]) for k in corner);new_face_uses.setdefault(tuple(sorted(tri)),[]).append((cell,tri))
assert all(len(new_face_uses[tuple(sorted((tri[0],tri[1],tri[2])))])==2 for _,tri in star)
assert all(node not in tri for key,uses in new_face_uses.items() if len(uses)==1 for _,tri in uses if int(new_cp[uses[0][0]])>=0)
# Update candidate arrays and preserve every original source row.
a={k:v.copy() for k,v in base.items()}
a["points_reference_world_m_f64"]=synthetic_points
current=base["points_current_world_m_f32"].astype(np.float64)
current[node]=np.asarray(current[node]+actual_delta,dtype=np.float32).astype(np.float64)
a["points_current_world_m_f32"]=np.vstack([current.astype(np.float32),base["points_current_world_m_f32"][node]])
local=base["points_donor_local_m_f64"].copy()
frame_owner=int(maprow["frame_owner_body"])
poses=json.loads((CAP/"registered-skin-poses-100.json").read_text())
frames,_=m.load_frame_table(poses)
frame=frames[frame_owner]
local[node]=m.matrix_vector3(frame["current_invR"],m.sub(tuple(moved),frame["origin"]))
a["points_donor_local_m_f64"]=np.vstack([local,base["points_donor_local_m_f64"][node]])
target=base["target_current_world_m"].copy()
target[node]=target[node]+actual_delta
a["target_current_world_m"]=np.vstack([target,base["target_current_world_m"][node]])
a["volume_tetrahedra"]=new_tets
a["child_parent_tet_row"]=new_cp
for key in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"):
 a[key]=np.concatenate([base[key],base[key][np.asarray([cell for cell,_ in star],dtype=np.int64)]])
a["parent_volume_node_row"]=np.append(base["parent_volume_node_row"],-1)
a["full86_weights"]=np.vstack([base["full86_weights"],base["full86_weights"][node]])
# Role row 30 now maps to the added boundary node; original node 30 remains volume-only.
for row in maps:
 if int(row["volume_node_row"])==node and row["role"]=="O":
  row["volume_node_row"]=str(new_node)
  for axis,label in enumerate(("x","y","z")):
   row["candidate_current_"+label+"_m"]=format(float(a["points_current_world_m_f32"][new_node,axis]),".17g")
   row["candidate_reference_"+label+"_m"]=format(float(a["points_reference_world_m_f64"][new_node,axis]),".17g")
   row["candidate_donor_local_"+label+"_m"]=format(float(a["points_donor_local_m_f64"][new_node,axis]),".17g")
  row["embedding_kind"]="synthetic_outward_O_star_retirement_control"
# Regenerate O faces in role-row coordinates from actual child tetra facets.
volume_to_role={int(r["volume_node_row"]):int(r["role_row"]) for r in maps if r["role"]=="O"}
out_faces=[]
for key,uses in new_face_uses.items():
 if len(uses)!=1:continue
 cell,tri=uses[0]
 if int(a["material_index"][cell])!=15 or any(n not in volume_to_role for n in tri):continue
 role_tri=tuple(volume_to_role[n] for n in tri)
 out_faces.append((tuple(sorted(role_tri)),role_tri,int(new_cp[cell])))
out_faces.sort(key=lambda x:x[0])
assert len(out_faces)==len(base["coarse_boundary_faces_O"])==4777
a["coarse_boundary_faces_O"]=np.asarray([tri for _,tri,_ in out_faces],dtype=np.int64)
a["coarse_boundary_other_parent_tets_O"]=np.asarray([pt for _,_,pt in out_faces],dtype=np.int64)
np.savez_compressed(OUT/"candidate.npz",**a)
with (OUT/"candidate.node").open("w") as f:
 f.write(f"{len(synthetic_points)} 3 0 0\n")
 for i,q in enumerate(synthetic_points):f.write(str(i+1)+" "+" ".join(format(float(x),".17g") for x in q)+"\n")
fields=list(maps[0])
write_csv(OUT/"candidate-node-map.csv",fields,[[r[k] for k in fields] for r in maps])
parent_tets=[[int(r["parent_node"+str(k)]) for k in range(4)] for r in read_csv(CAP/"compiled-parent-cell-mass.csv.tets.csv")]
source_labels=read_csv(CAP/"compiled-parent-cell-mass.csv.labels.csv")
source_layers=read_csv(CAP/"parent-layer-rows.csv")
source_role_by_node={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in source_layers}
source_faces=m.collect_parent_o_faces(parent_tets,source_labels,source_role_by_node)
source_face_by_nodes={tuple(r["node_rows"]):r for r in source_faces}
face_records=[]
for cell,tri in star:
 old_nodes=tuple(sorted(int(pn[n]) for n in tri))
 source=source_face_by_nodes[old_nodes]
 target_nodes=tuple(sorted(new_node if n==node else n for n in tri))
 face_records.append({
  "parent_o_face_row":int(source["parent_o_face_row"]),
  "source_stable_source_face_id":int(source["stable_source_face_id"]),
  "source_material_index":int(source["material_index"]),
  "source_geometric_donor_body":int(source["geometric_donor_body"]),
  "source_anatomical_region_id":int(source["anatomical_region_id"]),
  "replacement_child_node_rows":[list(target_nodes)],
 })
assert len(face_records)==7 and len({r["parent_o_face_row"] for r in face_records})==7
# The new node is a one-hot child of the original source node in an incident parent cell.
source_parent_tet=int(wedge_parent[0]);source_tet=parent_tets[source_parent_tet]
assert node in source_tet
beta=[1.0 if int(x)==node else 0.0 for x in source_tet]
write_csv(OUT/"inserted-node-ancestry.csv",["volume_node_row","parent_tet_row","parent_bary0","parent_bary1","parent_bary2","parent_bary3"],[[new_node,source_parent_tet,*beta]])
candidate_path=OUT/"candidate.npz";node_path=OUT/"candidate.node";node_map_path=OUT/"candidate-node-map.csv";ancestry_path=OUT/"inserted-node-ancestry.csv"
pins={
 "parent_packet_sha256":sha(CAP/"compiled-parent-cell-mass.csv"),
 "parent_nodes_sha256":sha(CAP/"compiled-parent-cell-mass.csv.nodes.csv"),
 "parent_tets_sha256":sha(CAP/"compiled-parent-cell-mass.csv.tets.csv"),
 "parent_labels_sha256":sha(CAP/"compiled-parent-cell-mass.csv.labels.csv"),
 "parent_weights_sha256":sha(CAP/"compiled-parent-cell-mass.csv.weights.csv"),
 "parent_layer_rows_sha256":sha(CAP/"parent-layer-rows.csv"),
 "native_poses_sha256":sha(CAP/"registered-skin-poses-100.json"),
 "candidate_npz_sha256":sha(candidate_path),
 "candidate_node_sha256":sha(node_path),
 "candidate_node_map_sha256":sha(node_map_path),
 "inserted_node_ancestry_sha256":sha(ancestry_path),
}
retirement={
 "schema":"numi-human.explicit-o-boundary-retirement.v1",
 "pins":pins,
 "retired_parent_o_nodes":[{"volume_node_row":node,"parent_o_role_row":node}],
 "replaced_parent_o_faces":face_records,
 "derivation":"synthetic outward wedge control; old source O node retained in volume interior, one-hot source ancestry retained; not anatomical repair",
}
retirement_path=OUT/"o-boundary-retirement.json"
retirement_path.write_text(json.dumps(retirement,indent=2,sort_keys=True)+"\n")
with np.load(candidate_path,allow_pickle=False) as z:
 assert z["volume_tetrahedra"].shape==(28639,4) and z["points_reference_world_m_f64"].shape==(7234,3)
plan={
 "schema":"numi-human.synthetic-o-boundary-retirement-control.v1",
 "scope":"full exporter/source-sidecar protocol control only; no anatomical geometry or physical admission",
 "base_candidate":str(BASE/"candidate.npz"),"base_candidate_sha256":sha(BASE/"candidate.npz"),
 "parent_capture":str(CAP),"retired_parent_volume_node_row":node,"retired_parent_o_role_row":node,
 "new_boundary_node_row":new_node,"replaced_parent_o_face_rows":[r["parent_o_face_row"] for r in face_records],
 "replacement_face_count":len(face_records),"star_face_count":len(star),"outward_direction":direction.tolist(),
 "minimum_face_normal_dot":min_outward_dot,"requested_offset_m":epsilon,"actual_old_node_offset_m":float(np.linalg.norm(actual_delta)),
 "min_positive_child_tet_volume_m3":min(signed_volume(synthetic_points,t) for t in new_tets),
 "candidate_sha256":sha(candidate_path),"candidate_node_sha256":sha(node_path),"candidate_node_map_sha256":sha(node_map_path),
 "retirement_map_sha256":sha(retirement_path),"physical_admission":False,"native_steps":0,
}
(OUT/"fixture.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
print(json.dumps(plan,sort_keys=True))

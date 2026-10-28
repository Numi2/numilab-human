#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).with_name("register_combined_volume_nodes_217.py")
spec=importlib.util.spec_from_file_location("register_combined_volume_nodes_217_test_module",SCRIPT)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

def writecsv(path,header,rows):
 with path.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=header,lineterminator="\n");w.writeheader();w.writerows(rows)

def pfloat(x): return format(float(x),".17g")

def fixture(root,persist):
 reg=root/"registered";reg.mkdir()
 parentbase=root/"compiled-parent-cell-mass.csv"
 # Two same-shape positive reference/current parent cells with separate material identities.
 verts=np.array([[-10,-10,-10],[30,-10,-10],[-10,30,-10],[-10,-10,30]],float)
 parentpoints=np.vstack((verts,verts,verts+np.asarray([100.0,0.0,0.0])))
 nodes=[]
 for i,p in enumerate(parentpoints):
  donor=0 if i<8 else 1
  nodes.append({"parent_node_index":i,"compiled_global_node":i,
   **{f"current_{a}_m":pfloat(p[k]) for k,a in enumerate("xyz")},
   **{f"rest_{a}_m":pfloat(p[k]) for k,a in enumerate("xyz")},
   "compiled_lumped_mass_kg":"0.25","node_frame_owner_body":str(donor),
   "donor_local_x_m":pfloat(p[0]),"donor_local_y_m":pfloat(p[1]),"donor_local_z_m":pfloat(p[2]),
   "mass_bits":"1048576000"})
 writecsv(Path(str(parentbase)+".nodes.csv"),mod.PARENT_NODE_HEADER,nodes)
 source_header=["object_local_node","material_index","donor_body","mass_kg","local_x_m","local_y_m","local_z_m"]
 source_rows=[]
 for material,offset,donor in ((15,0,0),(16,4,0),(16,8,1)):
  for node in range(offset,offset+4):
   point=parentpoints[node]
   source_rows.append({"object_local_node":node,"material_index":material,"donor_body":donor,"mass_kg":"0.25",
    "local_x_m":pfloat(point[0]),"local_y_m":pfloat(point[1]),"local_z_m":pfloat(point[2])})
 writecsv(Path(str(parentbase)+".source.csv"),source_header,source_rows)
 mass=[{"source_tet_row":0,"material_index":15,"mass_kg":"1"},
       {"source_tet_row":1,"material_index":16,"mass_kg":"1"},
       {"source_tet_row":2,"material_index":16,"mass_kg":"1"}]
 writecsv(parentbase,["source_tet_row","material_index","mass_kg"],mass)
 cellvol=64000.0/6.0
 tetrows=[];labrows=[]
 for row,(mat,offset,face,donor) in enumerate(((15,0,98586,0),(16,4,98587,0),(16,8,98589,1))):
  tetrows.append({"parent_tet_row":row,"material_index":mat,"donor_body":donor,
   "parent_node0":offset,"parent_node1":offset+1,"parent_node2":offset+2,"parent_node3":offset+3,
   "parent_reference_volume_m3":pfloat(cellvol),"parent_mass_kg":"1"})
  labrows.append({"parent_tet_row":row,"material_index":mat,"geometric_donor_body":donor,
   "anatomical_region_id":42,"stable_source_face_id":face})
 writecsv(Path(str(parentbase)+".tets.csv"),
  ["parent_tet_row","material_index","donor_body","parent_node0","parent_node1","parent_node2",
   "parent_node3","parent_reference_volume_m3","parent_mass_kg"],tetrows)
 writecsv(Path(str(parentbase)+".labels.csv"),
  ["parent_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"],labrows)

 points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[.1,.1,.1],[.2,.2,.2]],dtype=np.float32)
 nodepath=root/"combined.node";elepath=root/"combined.ele";labelpath=root/"child.labels.csv"
 with nodepath.open("w") as f:
  f.write(f"{len(points)} 3 0 0\n")
  for i,p in enumerate(points): f.write(f"{i+1} {pfloat(p[0])} {pfloat(p[1])} {pfloat(p[2])}\n")
 tets=[(0,1,2,5),(1,3,2,6),(2,4,3,6),(1,3,6,5)]
 mats=[15,16,16,16]
 donors=[0,0,0,1]
 with elepath.open("w") as f:
  f.write(f"{len(tets)} 4 1\n")
  for i,(tet,mat) in enumerate(zip(tets,mats)):
   f.write(f"{91-i*7} "+" ".join(str(x+1) for x in tet)+f" {mat}\n")
 labels=[]
 for i,(mat,donor) in enumerate(zip(mats,donors)):
  labels.append({"child_tet_row":i,"material_index":mat,"geometric_donor_body":donor,
   "anatomical_region_id":42,"stable_source_face_id":98586+i})
 writecsv(labelpath,["child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"],labels)

 # 157-body native pose table; only body 0 is one of the registered skin binding donors.
 allbodies=[{"body_index":i,"current":{"position_world_m":[0,0,0],"orientation_world_xyzw":[0,0,0,1]},
             "rest":{"position_world_m":[0,0,0],"orientation_world_xyzw":[0,0,0,1]}} for i in range(157)]
 posepath=root/"poses.json";posepath.write_text(json.dumps({"all_bodies":allbodies}),encoding="utf-8")
 frame_header=mod.FRAME_HEADER
 frame={k:"0" for k in frame_header};frame.update({"binding":"43","donor":"0"})
 for i in range(3):
  frame[f"current_R{i}{i}"]="1";frame[f"rest_R{i}{i}"]="1"
 writecsv(reg/"donor-frames.csv",frame_header,[frame])

 # O: direct source vertex; I: three source-face embeddings; D: direct source vertex.
 surf_headers={
  "O":["vertex","donor","wx","wy","wz","lx","ly","lz"]+[f"m{i}{j}" for i in range(3) for j in range(3)]+
     ["source_domain","source_mapping_kind","source_vertex_row","source_face_row","source_bary0","source_bary1","source_bary2",
      "source_residual_x_m","source_residual_y_m","source_residual_z_m","frame_owner_method"],
  "I":["vertex","donor","wx","wy","wz","lx","ly","lz"]+[f"m{i}{j}" for i in range(3) for j in range(3)]+
     ["source_domain","d37_parent_node_row","d37_parent_surface_face_row","beta0","beta1","beta2",
      "source_residual_x_m","source_residual_y_m","source_residual_z_m","frame_owner_method"],
  "D":["vertex","donor","wx","wy","wz","lx","ly","lz"]+[f"m{i}{j}" for i in range(3) for j in range(3)]+
     ["source_domain","d37_parent_node_row","d37_parent_surface_face_row","beta0","beta1","beta2",
      "source_residual_x_m","source_residual_y_m","source_residual_z_m","frame_owner_method"]}
 data={"O":[(0,0)],"I":[(0,1),(1,2),(2,3)],"D":[(0,4)]}
 surf={}
 for layer,items in data.items():
  rows=[]
  for srow,node in items:
   point=points[node].astype(float);r={"vertex":srow,"donor":0}
   for k,a in enumerate("xyz"): r["w"+a]=pfloat(point[k]);r["l"+a]=pfloat(point[k])
   for i in range(3):
    for j in range(3): r[f"m{i}{j}"]="1" if i==j else "0"
   if layer=="O":
    r.update({"source_domain":"fixture-detail","source_mapping_kind":"legacy-source-kind",
     "source_vertex_row":str(100+node),"source_face_row":"-1","source_bary0":"0","source_bary1":"0","source_bary2":"0",
     "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0","frame_owner_method":"fixture-frame"})
   elif layer=="I":
    bary=np.eye(3)[srow]
    r.update({"source_domain":"D37_coarse_parent","d37_parent_node_row":"-1","d37_parent_surface_face_row":"77",
     "beta0":pfloat(bary[0]),"beta1":pfloat(bary[1]),"beta2":pfloat(bary[2]),
     "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0","frame_owner_method":"fixture-frame"})
   else:
    r.update({"source_domain":"D37_coarse_parent","d37_parent_node_row":"40","d37_parent_surface_face_row":"-1",
     "beta0":"0","beta1":"0","beta2":"0","source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0",
     "frame_owner_method":"fixture-frame"})
   rows.append(r)
  writecsv(reg/f"registered-{layer}.csv",surf_headers[layer],rows);surf[layer]=rows

 def beta_point(point):
  return np.asarray([1-sum((point+10)/40),*(point+10)/40],dtype=np.float64)
 registrations=[]
 choices={"O":[(0,0)],"I":[(0,0),(1,0),(2,1)],"D":[(0,1)]}
 for layer,items in choices.items():
  for sr,tetrow in items:
   node=data[layer][sr][1];point=points[node].astype(float)
   beta=beta_point(point)
   source=surf[layer][sr]
   r={"layer":layer,"surface_vertex_row":sr,
      "parent_registration_kind":"source_parent_tet_reference_registration",
      "parent_tet_row":tetrow,**{f"parent_bary{k}":pfloat(beta[k]) for k in range(4)},
      "parent_residual_x_m":"0","parent_residual_y_m":"0","parent_residual_z_m":"0",
      "parent_registration_distance_m":"0","parent_registration_tie_count":"0",
      "frame_owner_method":source["frame_owner_method"],"source_domain":source["source_domain"],
      "source_vertex_row":source.get("source_vertex_row",source.get("d37_parent_node_row")),
      "source_face_row":source.get("source_face_row",source.get("d37_parent_surface_face_row")),
      "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0"}
   if layer=="O":
    for k in range(3): r[f"source_bary{k}"]=source[f"source_bary{k}"]
   else:
    for k in range(3): r[f"source_bary{k}"]=source[f"beta{k}"]
   registrations.append(r)
 writecsv(reg/"node-parent-registration.csv",mod.REG_HEADER,registrations)

 # 8-column existing map: source TetGen IDs are deliberately nonmonotone and source-row order is shuffled.
 maprows=[]
 def mrow(joint,partition,srcid,layer="",surface=-1,irow=-1,peer="",peerid=""):
  return {"joint_node_id":joint,"source_partition":partition,"source_tetgen_node_id":srcid,
   "registered_I_source_vertex_row":irow,"merge_peer_partition":peer,"merge_peer_tetgen_node_id":peerid,
   "registered_surface_layer":layer,"registered_surface_vertex_row":surface}
 maprows += [mrow(1,"skin",177,"O",0)]
 for sr,joint,skinid,fatid in ((0,2,181,271),(1,3,175,269),(2,4,179,265)):
  maprows.append(mrow(joint,"skin",skinid,"I",sr,sr,"fat",fatid))
 maprows.append(mrow(6,"skin",188))
 maprows.append(mrow(2,"fat",271,"I",0,0,"skin",181))
 maprows.append(mrow(3,"fat",269,"I",1,1,"skin",175))
 maprows.append(mrow(4,"fat",265,"I",2,2,"skin",179))
 maprows.append(mrow(5,"fat",211,"D",0))
 maprows.append(mrow(7,"fat",215))
 maprows=maprows[::-1]
 node_map=root/"skin-fat-combined-node-map.csv.gz"
 with gzip.open(node_map,"wt",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=mod.BASE_HEADER,lineterminator="\n");w.writeheader();w.writerows(maprows)
 return {"node_map":node_map,"node":nodepath,"ele":elepath,"labels":labelpath,"parent":parentbase,
  "registered":reg,"poses":posepath,"surface_nodes":surf,"registrations":registrations,
  "persist":persist}

def test_native_matrix_pose_path(root):
 allbodies=[]
 identity=[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
 near=float(np.float32(1.0000002))
 current=[near,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
 origin=np.asarray([0.1,-0.2,0.3],dtype=np.float32).astype(np.float64).tolist()
 for i in range(157):
  matrix=current if i==0 else identity
  allbodies.append({"body_index":i,
   "current":{"position_world_m":origin,"orientation_world_xyzw":[0.0,0.0,0.0,1.0],"local_to_world_row_major_f32":matrix},
   "rest":{"position_world_m":origin,"orientation_world_xyzw":[0.0,0.0,0.0,1.0],"local_to_world_row_major_f32":identity}})
 poses_path=root/"native-matrix-poses.json"
 frames_path=root/"native-matrix-donor-frames.csv"
 poses_path.write_text(json.dumps({"all_bodies":allbodies}),encoding="utf-8")
 frame={k:"0" for k in mod.FRAME_HEADER};frame.update({"binding":"7","donor":"0"})
 for state,matrix in (("current",current),("rest",identity)):
  for axis,value in zip("xyz",origin):frame[f"{state}_{axis}"]=pfloat(value)
  for i in range(3):
   for j in range(3):frame[f"{state}_R{i}{j}"]=pfloat(matrix[3*i+j])
 writecsv(frames_path,mod.FRAME_HEADER,[frame])
 registered,bindings=mod.read_poses(poses_path,frames_path)
 pose=registered[0];R=np.asarray(current,dtype=np.float32).astype(np.float64).reshape(3,3)
 assert bindings=={0:7} and pose["nativeMatrixFrame"]
 assert np.array_equal(pose["current_R"],R)
 assert np.array_equal(pose["current_world_to_local_R"],np.linalg.inv(R))
 assert not np.array_equal(pose["current_world_to_local_R"],R.T)
 assert pose["nativeMatrixSO3Bound"]==8.0*float(np.finfo(np.float32).eps)
 point=np.asarray([0.4,0.6,0.8]);local=pose["current_world_to_local_R"]@(point-pose["current_origin"])
 assert np.linalg.norm(pose["current_origin"]+R@local-point)<1e-14
 # Keep the legacy quaternion route and its transpose behavior intact.
 legacy=[dict(row,current=dict(row["current"]),rest=dict(row["rest"])) for row in allbodies]
 for row in legacy:
  for state in ("current","rest"):row[state].pop("local_to_world_row_major_f32")
 legacy_path=root/"legacy-poses.json";legacy_path.write_text(json.dumps({"all_bodies":legacy}),encoding="utf-8")
 legacy_frame=dict(frame);legacy_frame["current_R00"]="1"
 legacy_frames_path=root/"legacy-donor-frames.csv";writecsv(legacy_frames_path,mod.FRAME_HEADER,[legacy_frame])
 legacy_poses,_=mod.read_poses(legacy_path,legacy_frames_path)
 assert not legacy_poses[0]["nativeMatrixFrame"]
 assert np.array_equal(legacy_poses[0]["current_world_to_local_R"],legacy_poses[0]["current_R"].T)
 bad=[dict(row,current=dict(row["current"]),rest=dict(row["rest"])) for row in allbodies]
 bad[0]["rest"].pop("local_to_world_row_major_f32")
 bad_path=root/"partial-matrix-poses.json";bad_path.write_text(json.dumps({"all_bodies":bad}),encoding="utf-8")
 try:mod.read_poses(bad_path,frames_path)
 except ValueError as exc:assert "incomplete native" in str(exc)
 else:raise AssertionError("partial native all_bodies matrix table accepted")
 bad=[dict(row,current=dict(row["current"]),rest=dict(row["rest"])) for row in allbodies]
 bad[0]["current"]["local_to_world_row_major_f32"][0]+=1e-10
 bad_path=root/"non-f32-matrix-poses.json";bad_path.write_text(json.dumps({"all_bodies":bad}),encoding="utf-8")
 try:mod.read_poses(bad_path,frames_path)
 except ValueError as exc:assert "not exact Float32" in str(exc)
 else:raise AssertionError("non-Float32 native matrix accepted")
 bad=[dict(row,current=dict(row["current"]),rest=dict(row["rest"])) for row in allbodies]
 bad[0]["current"]["local_to_world_row_major_f32"][0]=float(np.float32(1.000002))
 bad_path=root/"outside-so3-matrix-poses.json";bad_path.write_text(json.dumps({"all_bodies":bad}),encoding="utf-8")
 try:mod.read_poses(bad_path,frames_path)
 except ValueError as exc:assert "SO(3)" in str(exc)
 else:raise AssertionError("native matrix outside trusted SO(3) envelope accepted")

def main():
 persist=Path(sys.argv[1]);persist.mkdir(parents=True,exist_ok=True)
 inputs=persist/"inputs"
 if inputs.exists(): shutil.rmtree(inputs)
 inputs.mkdir(parents=True)
 test_native_matrix_pose_path(inputs)
 x=fixture(inputs,persist)
 args=type("Args",(),{})()
 args.node_map=x["node_map"];args.node=x["node"];args.ele=x["ele"];args.labels=x["labels"]
 args.parent_base=x["parent"];args.registered_dir=x["registered"];args.poses=x["poses"]
 args.maximum_parent_distance_m=.02;args.ambiguity_tolerance_m=1e-10
 args.output=persist/"fixture-registered-node-map.csv";args.receipt=persist/"fixture-receipt.json"
 receipt=mod.emit(args)
 rows=mod.csvread(args.output,mod.HEADER)
 assert len(rows)==10 and receipt["volume_nodes"]==7 and receipt["shared_interface_nodes"]==3
 bynode={}
 for r in rows:bynode.setdefault(r["joint_node_id"],[]).append(r)
 assert bynode["2"][0]["role"]=="I" and bynode["2"][0]["incident_material_mask"]==str((1<<15)|(1<<16))
 assert bynode["6"][0]["source_mapping_kind"]=="none"
 assert bynode["6"][0]["parent_registration_kind"]=="inferred_current_parent_tet_same_material_donor_region"
 assert bynode["6"][0]["donor_binding"]=="43"
 assert bynode["7"][0]["geometric_donor_body"]=="0"
 assert float(bynode["7"][0]["parent_registration_distance_m"])<=.02
 assert bynode["2"][0]["source_mapping_kind"]=="face"
 assert bynode["1"][0]["source_vertex_row"]=="100"
 assert bynode["5"][0]["source_vertex_row"]=="40"
 assert bynode["2"][0]["source_face_row"]=="77"
 parent,owner=mod.read_parent_current(x["parent"],Path(__file__).parent)
 poses,_=mod.read_poses(x["poses"],x["registered"]/"donor-frames.csv")
 child=mod.read_child(x["node"],x["ele"],x["labels"])
 surfaces=mod.read_surfaces(x["registered"])
 regs=mod.read_parent_registrations(x["registered"],surfaces,parent,.02)
 lookup=mod.read_combined_node_map(x["node_map"])
 bad_parent=dict(parent);bad_parent["donors"]=parent["donors"].copy();bad_parent["donors"][0]=1
 try: mod.build_rows(lookup,child,surfaces,regs,bad_parent,owner,poses,.02,1e-10)
 except ValueError as e: assert "parent semantic tuple" in str(e) or "incident child-cell tuple" in str(e)
 else: raise AssertionError("accepted boundary parent with a conflicting geometric donor")
 badlabels=list(child["labels"]);badlabels[0]=(badlabels[0][0],1,badlabels[0][2],badlabels[0][3])
 badchild=dict(child);badchild["labels"]=badlabels;badchild["incident"]=[[] for _ in child["points"]]
 for tet,label in zip(badchild["tets"],badlabels):
  for node in tet:badchild["incident"][node].append(label)
 try: mod.build_rows(lookup,badchild,surfaces,regs,parent,owner,poses,.02,1e-10)
 except ValueError as e: assert "parent" in str(e) or "semantic" in str(e)
 else: raise AssertionError("accepted child cells whose donor has no matching parent cells")
 for layer,wrong_partition,message in (("O","fat","skin 15/skin"),("D","skin","fat 16/fat")):
  wrong_map=[dict(r) for r in lookup]
  target=next(r for r in wrong_map if r["registered_surface_layer"]==layer)
  target["source_partition"]=wrong_partition
  try: mod.build_rows(wrong_map,child,surfaces,regs,parent,owner,poses,.02,1e-10)
  except ValueError as e: assert message in str(e)
  else: raise AssertionError(f"accepted {layer} node in wrong singleton partition")
 print("PASS exact native matrix/inverse and legacy frame paths; combined registered node-map producer: 7 nodes/4 cells, 2 interior nodes, far alternate semantic parent filtered while valid incident parent selected, exact shared I maps, shuffled source IDs/map rows, source vertex/face ancestry, wrong O/D singleton partitions and wrong donor/parent rejected; physical admission false")
if __name__=="__main__":main()

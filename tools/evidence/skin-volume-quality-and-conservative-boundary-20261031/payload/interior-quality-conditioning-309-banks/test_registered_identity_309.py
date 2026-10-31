#!/usr/bin/env python3
from pathlib import Path
import csv, importlib.util, json, tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("condition_registered_interior_309",ROOT/"condition_registered_interior_309.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
FIELDS=["source_partition","source_tetgen_node_id","role","source_domain","source_mapping_kind","source_vertex_row","source_face_row","registered_surface_layer","registered_surface_vertex_row","current_x_m","current_y_m","current_z_m","parent_registration_kind","parent_tet_row","anatomical_region_id"]
points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.2,.2,.2]],np.float32)
rows=[
 {"source_partition":"skin","source_tetgen_node_id":1,"role":"O","source_domain":"derived_registered_skin_authoring","source_mapping_kind":"vertex","source_vertex_row":0,"source_face_row":-1,"registered_surface_layer":"O","registered_surface_vertex_row":0,"parent_registration_kind":"surface","parent_tet_row":-1,"anatomical_region_id":1},
 {"source_partition":"skin","source_tetgen_node_id":2,"role":"I","source_domain":"derived_registered_skin_authoring","source_mapping_kind":"vertex","source_vertex_row":0,"source_face_row":-1,"registered_surface_layer":"I","registered_surface_vertex_row":0,"parent_registration_kind":"surface","parent_tet_row":-1,"anatomical_region_id":1},
 {"source_partition":"skin","source_tetgen_node_id":3,"role":"O","source_domain":"derived_registered_skin_authoring","source_mapping_kind":"vertex","source_vertex_row":1,"source_face_row":-1,"registered_surface_layer":"O","registered_surface_vertex_row":1,"parent_registration_kind":"surface","parent_tet_row":-1,"anatomical_region_id":1},
 {"source_partition":"skin","source_tetgen_node_id":4,"role":"I","source_domain":"derived_registered_skin_authoring","source_mapping_kind":"vertex","source_vertex_row":1,"source_face_row":-1,"registered_surface_layer":"I","registered_surface_vertex_row":1,"parent_registration_kind":"surface","parent_tet_row":-1,"anatomical_region_id":1},
 {"source_partition":"skin","source_tetgen_node_id":5,"role":"interior","source_domain":"derived_volume_interior_no_surface_ancestry","source_mapping_kind":"none","source_vertex_row":-1,"source_face_row":-1,"registered_surface_layer":"","registered_surface_vertex_row":-1,"parent_registration_kind":"inferred_current_parent_tet_same_material_donor_region","parent_tet_row":0,"anatomical_region_id":1}
]
for i,row in enumerate(rows):
 for k,v in zip(["current_x_m","current_y_m","current_z_m"],points[i]):row[k]=str(float(v))
def write_map(rs,p):
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rs)
def must_reject(rs,path):
 write_map(rs,path)
 try:m.registered_source_rows_from_map(path,points)
 except ValueError:return
 raise AssertionError("invalid registration map was accepted")
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);good=tmp/"good.csv";write_map(rows,good)
 mask,counts=m.registered_source_rows_from_map(good,points)
 assert mask.tolist()==[True,True,True,True,False]
 faces=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]
 t=np.asarray([[4,*f] for f in faces],np.int64)
 det,_=m.m.measurements(points,t)
 for i in range(len(t)):
  if det[i]<0:t[i,[2,3]]=t[i,[3,2]]
 assert np.all(m.m.measurements(points,t)[0]>0)
 _,_,boundary,fixed,eligible,_=m.validate_input(points,t,np.ones(4,np.int32),points,mask)
 assert fixed.tolist()==[True,True,True,True,False]
 assert eligible.tolist()==[4]
 bad=[dict(x) for x in rows];bad[4]["role"]="unknown";must_reject(bad,tmp/"role.csv")
 bad=[dict(x) for x in rows];bad[4]["source_vertex_row"]=2;must_reject(bad,tmp/"ancestry.csv")
 must_reject(rows[:-1],tmp/"missing.csv")
 bad=[dict(x) for x in rows];bad[3]["current_x_m"]="9";must_reject(bad,tmp/"coords.csv")
 bad=[dict(x) for x in rows];bad[2]["source_vertex_row"]=0;must_reject(bad,tmp/"duplicate.csv")
report={"schema":"h706.registered-anchor-interior-conditioning-controls.v1","status":"PASS_REGISTERED_IDENTITY_ELIGIBILITY_CONTROLS","controls":["direct_OI_vertex_identity_fixed","no_surface_ancestry_interior_eligible","nonboundary_registered_identity_fixed","unknown_role_rejected","false_surface_ancestry_rejected","missing_source_row_rejected","coordinate_mismatch_rejected","duplicate_registered_identity_rejected"],"fixture":{"registered_mask":mask.tolist(),"eligible_rows":eligible.tolist(),"boundary_rows":np.unique(boundary).tolist(),"positive_cells":len(t)},"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
out=ROOT/"control-010";out.mkdir(exist_ok=False);(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

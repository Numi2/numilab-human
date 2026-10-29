#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
R=P/'paired-fine-headwall-rim-635/run-013'; OUT=R/'output'
O=P/'surface-transfer-checkpoint-392/payload/O/candidate.npz'
I=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
REG=P/'full-current-surface-registration-399/sidecars/registered-O.csv'
REGREC=P/'full-current-surface-registration-399/sidecars/receipt.json'
ASSET=P/'conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets'
CLOSED=OUT/'fine-skin-closed-boundary.npz'; SOURCE_MAP=OUT/'fine-rim-source-map.npz'; RIM_ORIENT=OUT/'fine-skin-rim-oriented-faces.npz'
DEST=OUT/'closed-skin-source-map.npz'; REGISTRY=OUT/'source-registry.json'; RECEIPT=OUT/'source-map-receipt.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
for p in (DEST,REGISTRY,RECEIPT):
 if p.exists():raise SystemExit(f'refusing overwrite of existing {p}')
report=json.loads((OUT/'report.json').read_text())
if report['status']!='PASS_FINE_PAIRED_SKIN_RIM_GEOMETRY_DIAGNOSTIC' or report['physical_admission'] or report['runtime_created'] or report['accepted_physical_steps']:
 raise ValueError('source geometry report is not the expected preparation-only pass')
if sha(O)!=report['fine_surfaces']['O']['sha256'] or sha(I)!=report['fine_surfaces']['I']['sha256']:
 raise ValueError('fine source candidates do not match geometry report')
regrec=json.loads(REGREC.read_text())
if regrec['candidateSurfaces']['O']['sha256']!=sha(O) or regrec['outputSHA256']['registered-O.csv']!=sha(REG):
 raise ValueError('399 O registration receipt does not authenticate the O source inputs')
with np.load(O,allow_pickle=False) as z:op=np.asarray(z['points_world_m']);of=np.asarray(z['faces'],np.int64)
with np.load(I,allow_pickle=False) as z:ip=np.asarray(z['points_world_m']);inf=np.asarray(z['faces'],np.int64)
with np.load(CLOSED,allow_pickle=False) as z:
 points=np.asarray(z['points_world_m_f64']);faces=np.asarray(z['faces'],np.uint32)
 rim_oriented=np.asarray(z['oriented_rim_faces'],np.int64)
with np.load(SOURCE_MAP,allow_pickle=False) as z:sm={k:np.asarray(z[k]) for k in z.files}
with np.load(RIM_ORIENT,allow_pickle=False) as z:
 rim_registered_rows=np.asarray(z['source_headwall_face_rows'],np.int64)
 rim_mode=str(z['rim_orientation_applied'].item())
 oriented_rim_from_audit=np.asarray(z['oriented_rim_faces'],np.int64)
if not np.array_equal(rim_oriented,oriented_rim_from_audit):raise ValueError('closed boundary rim differs from the oriented rim receipt')
reversed_rim=rim_mode.startswith('reversed_')
source_points=np.concatenate((op.astype(np.float32),ip.astype(np.float32)),axis=0)
nO=len(op);nI=len(ip)
source_rim_faces=np.asarray(sm['fine_skin_rim_faces'],np.int64)
rim_rows=np.asarray(sm['skin_headwall_source_face_rows'],np.int64)
if len(rim_rows)!=90 or len(set(map(int,rim_rows)))!=90 or len(source_rim_faces)!=90:raise ValueError('skin rim source row domain changed')
source_faces=np.vstack((of,inf[:,[0,2,1]]+nO,source_rim_faces))
if len(source_faces)!=len(faces):raise ValueError('closed skin source face count does not reconstruct from O/I/rim')
used=np.unique(source_faces)
inverse=np.full(len(source_points),-1,dtype=np.int64);inverse[used]=np.arange(len(used))
if not np.array_equal(source_points[used].astype(np.float64),points):raise ValueError('closed skin points are not exact source-row compaction')
if not np.array_equal(inverse[source_faces],faces):raise ValueError('closed skin faces are not exact O/I/rim source-face compaction')
if len(used)!=len(points) or np.any(inverse[source_faces]<0):raise ValueError('invalid compacted vertex row map')
vertex_role=np.where(used<nO,1,2).astype(np.uint8)
vertex_source_row=np.where(used<nO,used,used-nO).astype(np.int64)
face_role=np.concatenate((np.full(len(of),1,np.uint8),np.full(len(inf),2,np.uint8),np.full(90,3,np.uint8)))
face_source_row=np.concatenate((np.arange(len(of),dtype=np.int64),np.arange(len(inf),dtype=np.int64),rim_rows))
face_winding=np.concatenate((np.zeros(len(of),bool),np.ones(len(inf),bool),np.full(90,reversed_rim,bool)))
# Rebuild the raw registered 545 rim triangles using joint-node ancestry and
# the authenticated O dominant-parent/I one-hot correspondence.
facet_path=ASSET/'volume/facet-ancestry.csv';joint_path=ASSET/'volume/combined-node-map.csv'
facet_rows=[x for x in read_rows(facet_path) if x['source_partition']=='skin' and x['face_source_role']=='donor_skin_OI_headwall']
if len(facet_rows)!=90 or [int(x['face_source_row']) for x in facet_rows]!=list(map(int,rim_rows)):
 raise ValueError('545 registered skin rim face source row sequence changed')
coarse={}
for x in read_rows(joint_path):
 if x['registered_surface_layer'] in ('O','I'):
  v=(x['registered_surface_layer'],int(x['registered_surface_vertex_row']),int(x['source_parent_node_row']))
  joint=int(x['joint_node_id'])
  if joint in coarse and coarse[joint]!=v:raise ValueError('conflicting 545 joint node mapping')
  coarse[joint]=v
O_parent_to_fine={int(p):int(v) for p,v in zip(sm['O_parent330_dominant_representative_rows'],sm['O_fine_vertex_rows'])}
I_parent_to_fine={int(p):int(v) for p,v in zip(sm['I_parent330_node_rows'],sm['I_fine_vertex_rows'])}
raw_rim_by_source={}
for row in facet_rows:
 tri=[]
 for k in range(3):
  role,role_row,parent=int(0),0,0
  role,role_row,parent=coarse[int(row[f'joint_node{k}'])]
  if role=='O':
   if parent not in O_parent_to_fine:raise ValueError(f'O rim parent {parent} absent from derived representative map')
   tri.append(O_parent_to_fine[parent])
  else:
   if parent not in I_parent_to_fine:raise ValueError(f'I rim parent {parent} absent from one-hot map')
   tri.append(nO+I_parent_to_fine[parent])
 raw_rim_by_source[int(row['face_source_row'])]=np.asarray(tri,np.int64)
raw_rim=np.asarray([raw_rim_by_source[int(x)] for x in rim_rows],np.int64)
registered_orientation_expected=raw_rim[:,[0,2,1]] if reversed_rim else raw_rim
if not np.array_equal(registered_orientation_expected,rim_oriented):raise ValueError('oriented rim differs from exact 545 joint-node reconstruction')
# Typed source-face decoder: layer 1=O fine face, 2=I fine face, 3=545 rim face.
def reconstruct_faces(layers,rows,winding):
 out=np.empty((len(layers),3),np.int64)
 for j,(layer,row,flip) in enumerate(zip(layers,rows,winding)):
  row=int(row);layer=int(layer)
  if layer==1:tri=np.asarray(of[row],np.int64)
  elif layer==2:tri=np.asarray(inf[row],np.int64)+nO
  elif layer==3:tri=raw_rim_by_source[row].copy()
  else:raise ValueError(f'unknown source face layer {layer}')
  if bool(flip):tri=tri[[0,2,1]]
  out[j]=inverse[tri]
 return out.astype(np.uint32)
rebuilt_faces=reconstruct_faces(face_role,face_source_row,face_winding)
if not np.array_equal(rebuilt_faces,faces):raise ValueError('typed source face mapping does not exactly reconstruct compacted output')
def reconstruct_vertices(layers,rows):
 out=np.empty_like(points)
 for j,(layer,row) in enumerate(zip(layers,rows)):
  if int(layer)==1:out[j]=op[int(row)].astype(np.float64)
  elif int(layer)==2:out[j]=ip[int(row)].astype(np.float64)
  else:raise ValueError(f'unknown source vertex layer {layer}')
 return out
rebuilt_points=reconstruct_vertices(vertex_role,vertex_source_row)
if not np.array_equal(rebuilt_points,points):raise ValueError('typed source vertex mapping does not exactly reconstruct output coordinates')
# Negative controls: wrong point ID, wrong source face ID, wrong winding, and a
# one-F32-ULP coordinate change must each be observable by exact reconstruction.
wrong_vertex_rows=vertex_source_row.copy(); witness_v=next(i for i,x in enumerate(vertex_role) if int(x)==1)
orig=int(wrong_vertex_rows[witness_v]);replacement=(orig+1)%nO
while np.array_equal(op[orig].astype(np.float64),op[replacement].astype(np.float64)):
 replacement=(replacement+1)%nO
wrong_vertex_rows[witness_v]=replacement
wrong_vertex_rejected=not np.array_equal(reconstruct_vertices(vertex_role,wrong_vertex_rows),points)
if not wrong_vertex_rejected:raise ValueError('negative control did not reject a wrong O point ID')
wrong_face_rows=face_source_row.copy();witness_f=int(np.flatnonzero(face_role==1)[0]);orig_face=int(wrong_face_rows[witness_f]);replacement_face=(orig_face+1)%len(of)
while np.array_equal(of[orig_face],of[replacement_face]):replacement_face=(replacement_face+1)%len(of)
wrong_face_rows[witness_f]=replacement_face
wrong_face_rejected=not np.array_equal(reconstruct_faces(face_role,wrong_face_rows,face_winding),faces)
if not wrong_face_rejected:raise ValueError('negative control did not reject a wrong O face ID')
wrong_winding=face_winding.copy();wrong_winding[witness_f]=not wrong_winding[witness_f]
wrong_winding_rejected=not np.array_equal(reconstruct_faces(face_role,face_source_row,wrong_winding),faces)
if not wrong_winding_rejected:raise ValueError('negative control did not reject changed face winding')
wrong_points=rebuilt_points.copy();wrong_points[0,0]=np.nextafter(wrong_points[0,0],np.inf)
wrong_coordinate_rejected=not np.array_equal(wrong_points,points)
if not wrong_coordinate_rejected:raise ValueError('negative control did not reject a changed coordinate')
# Preserve the full O boundary registration, accurately labeled as a derived
# dominant-corner representative, alongside all four tet weights and IDs.
extra={}
for key,val in sm.items():
 if key.startswith('O_') or key.startswith('I_') or key.startswith('skin_headwall_'):
  extra['boundary_'+key]=val
np.savez_compressed(DEST,output_vertex_input_rows=used.astype(np.int64),output_vertex_source_layer=vertex_role,output_vertex_source_row=vertex_source_row,face_source_layer=face_role,face_source_face_row=face_source_row,face_source_winding_reversed=face_winding,headwall_original_source_face_rows=rim_rows,**extra)
registry={
 '1':{'kind':'fine_registered_outer_skin_O','path':str(O),'sha256':sha(O),'registered_parent_tet_sidecar':str(REG),'registered_parent_tet_sidecar_sha256':sha(REG),'registration_receipt':str(REGREC),'registration_receipt_sha256':sha(REGREC),'point_count':len(op),'face_count':len(of),'output_vertex_point_identity':'exact candidate row; O head-ring source330 representative derives from the retained full parent-tet+barycentric registration and exact edge graph, not one-hot identity'},
 '2':{'kind':'fine_registered_inner_surface_I','path':str(I),'sha256':sha(I),'point_count':len(ip),'face_count':len(inf),'output_face_orientation':'reversed for closed skin boundary'},
 '3':{'kind':'registered_545_skin_OI_headwall','facets_csv':str(facet_path),'facets_csv_sha256':sha(facet_path),'face_domain':'face_source_row from skin donor_skin_OI_headwall facets','face_count':len(rim_rows),'orientation_applied':rim_mode,'joint_node_map_sha256':sha(joint_path)}
}
REGISTRY.write_text(json.dumps(registry,indent=2,sort_keys=True)+'\n')
receipt={'schema':'numi.closed-skin-exact-source-map.v1','status':'PASS_EXACT_COMPACTED_VERTEX_FACE_SOURCE_MAP','closed_skin_report':str(OUT/'report.json'),'closed_skin_report_sha256':sha(OUT/'report.json'),'closed_skin_npz_sha256':sha(CLOSED),'fine_rim_registration_npz_sha256':sha(SOURCE_MAP),'rim_orientation_npz_sha256':sha(RIM_ORIENT),'source_map_sha256':sha(DEST),'source_registry_sha256':sha(REGISTRY),'point_count':len(points),'face_count':len(faces),'input_O_point_count':nO,'input_I_point_count':nI,'source_O_face_count':len(of),'source_I_face_count':len(inf),'authenticated_headwall_face_count':len(rim_rows),'unused_input_points_excluded':len(source_points)-len(used),'exact_point_reconstruction':True,'exact_face_order_and_winding_reconstruction':True,'O_boundary_parent_tet_and_four_barycentric_weights_retained':True,'O_dominant_corner_is_derived_representative_not_one_hot_identity':True,'negative_controls':{'wrong_O_point_id_rejected':wrong_vertex_rejected,'wrong_O_face_id_rejected':wrong_face_rejected,'changed_face_winding_rejected':wrong_winding_rejected,'one_F32_ulp_coordinate_change_rejected':wrong_coordinate_rejected,'witness_output_vertex_row':witness_v,'correct_O_source_row':orig,'wrong_O_source_row':replacement,'witness_output_face_row':witness_f,'correct_O_face_row':orig_face,'wrong_O_face_row':replacement_face},'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'note':'Compaction/source mapping is an exact representation of the checked closed boundary. It does not admit a tetrahedral volume, native scene, mechanics, or physical motion.'}
RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))

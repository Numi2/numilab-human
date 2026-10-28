#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT=Path(__file__).with_name('export_authoring_registered_surface_sidecars_230.py')
spec=importlib.util.spec_from_file_location('authoring230',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(condition,message):
 if not condition:raise AssertionError(message)

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);src=root/'original-O.npz';out=root/'candidate-O.npz'
  np.savez(src,points_world_m=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float64),faces=np.array([[0,1,2]],np.int64),
   face_source_identity0=np.array([11],np.int64),face_source_identity1=np.array([22],np.int64))
  schema={'1':{'role':'O','path':str(src),'sha256':sha(src),'identity_keys':['face_source_identity0','face_source_identity1']}}
  reference=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float64)
  affine=np.array([[1.2,0.1,0.0],[0.0,0.9,0.1],[0.05,0.0,1.1]],np.float64)
  translation=np.array([2.0,-1.0,0.5]);current=reference@affine.T+translation
  p0=np.array([0.2,0.2,0.3,0.3])@current
  p1=np.array([0.1,0.4,0.2,0.3])@current
  tri=current[[1,2,3]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
  p2=tri.mean(axis=0)+0.01*normal
  points=np.asarray([p0,p1,p2],np.float32)
  np.savez(out,points_world_m=points,faces=np.array([[0,1,2]],np.int64),
   face_source_layer=np.array([0],np.int8),face_source_body=np.array([-1],np.int32),face_source_bone=np.array([-1],np.int32),
   face_source_face_id=np.array([0],np.int64),face_source_original_id=np.array([1],np.int64),
   face_source_identity0=np.array([11],np.int64),face_source_identity1=np.array([22],np.int64),
   face_source_identity_schema_by_original_id_json=np.array([json.dumps(schema)]))
  surface=m.validate_candidate(out,'O',True)
  derived=root/'derived-O-source.npz'
  np.savez(derived,face_id=np.array([7,8]),face_source_identity0=np.array([70,80]),
   face_source_identity1=np.array([-1,-1]),face_run_original_id=np.array([99,100]))
  derived_meta={'role':'O','derived_operation':'local_connected_patch_retriangulation',
   'path':str(derived),'sha256':sha(derived),'identity_keys':['face_source_identity0','face_source_identity1'],
   'identity_face_id_key':'face_id','identity_original_id_key':'face_run_original_id','identity_original_id':99,
   'derived_parent_source_ids':[1],'derived_parent_assets':[{'path':str(src),'sha256':sha(src)}],
   'source_geometry_scope':'referenced_descendant_faces_only'}
  def derived_candidate(target,meta=derived_meta,identity=70):
   np.savez(target,points_world_m=points,faces=np.array([[0,1,2],[0,2,1]],np.int64),
    face_source_layer=np.array([0,0],np.int8),face_source_body=np.array([-1,-1],np.int32),
    face_source_bone=np.array([-1,-1],np.int32),face_source_face_id=np.array([0,7],np.int64),
    face_source_original_id=np.array([1,99],np.int64),face_source_identity0=np.array([11,identity],np.int64),
    face_source_identity1=np.array([22,-1],np.int64),
    face_source_identity_schema_by_original_id_json=np.array([json.dumps({**schema,'99':meta})]))
  good=root/'derived-candidate.npz';derived_candidate(good)
  check(m.validate_candidate(good,'O')['faceAncestry']['99']['role']=='O','derived source stays distinct from BONE')
  bad_cases=[('wrong_source_hash',{**derived_meta,'sha256':'0'*64},70),
   ('wrong_parent_hash',{**derived_meta,'derived_parent_assets':[{'path':str(src),'sha256':'0'*64}]},70),
   ('missing_parent_assets',{**derived_meta,'derived_parent_assets':[]},70),
   ('wrong_original_filter',{**derived_meta,'identity_original_id':100},70),
   ('missing_derived_identity',derived_meta,80),
   ('wrong_role',{**derived_meta,'role':'I'},70)]
  for name,meta,identity in bad_cases:
   target=root/(name+'.npz');derived_candidate(target,meta,identity)
   rejected=False
   try:m.validate_candidate(target,'O')
   except ValueError:rejected=True
   check(rejected,'invalid derived ancestry rejected: '+name)
  parent={'points':current,'tets':np.array([[0,1,2,3]],np.int64),'materials':np.array([15],np.uint32),
   'donors':np.array([7],np.uint32),'regions':np.array([42],np.uint32),'faces':np.array([98586],np.uint32)}
  index=m.native.build_index(parent);mi=m.build_material_indices(parent)
  frames={}
  for body in [7]+[100+i for i in range(86)]:
   R=Rotation.from_rotvec([0.002*body,-0.003,0.001]).as_matrix();origin=np.array([body*1.0e-4,2.0,-0.5])
   frames[body]={'currentR':R,'currentOrigin':origin,'restR':np.eye(3),'restOrigin':np.zeros(3)}
  parent_weights=np.zeros((4,86),dtype=np.float64)
  for lane in range(4):parent_weights[lane,lane]=1.0
  weight_path=root/'parent-full86.csv'
  m.write_csv(weight_path,m.WEIGHT_HEADER,[{'vertex':i,**{f'w{j}':m.f64(parent_weights[i,j]) for j in range(86)}} for i in range(4)])
  loaded_weights,weight_meta=m.load_parent_full86_weights(weight_path,parent,{'nodesPath':'fixture-parent.nodes.csv','nodeSHA256':'fixture-parent-nodes-sha'})
  binding_bodies=[100+i for i in range(86)]
  result=m.register_layer(surface,parent,index,mi,frames,loaded_weights,binding_bodies,0.02,1.0e-8)
  check(result['sourceDomain']['vertexCount']==3 and result['sourceDomain']['faceCount']==1,'authoring registry counts')
  check([r['source_vertex_row'] for r in result['nodes']]==[0,1,2],'direct authoring row identities')
  check(all(r['source_domain']==m.DOMAIN and r['source_face_row']==-1 for r in result['nodes']),'no copied NHSKIN identities')
  check(all(r['parent_tet_row']==0 and r['parent_material_index']==15 and r['parent_geometric_donor_body']==7 for r in result['nodes']),'same-material parent labels')
  check(abs(float(result['nodes'][2]['parent_registration_distance_m'])-0.01)<2e-7,'current affine parent residual')
  check(result['faces'][0]['face_source_identity0']==11 and result['faces'][0]['face_source_identity1']==22,'face ancestry preserved')
  check(weight_meta['rowDomain']=='parent_node_index' and weight_meta['bindingCount']==86,'parent weight domain separately pinned')
  check(len(result['weightRows'])==3 and len(result['weightOwnerRows'])==3,'one full86 weight row per candidate node')
  for row,weight_row,owner_row,point in zip(result['nodes'],result['weightRows'],result['weightOwnerRows'],points.astype(np.float64)):
   local=np.asarray([float(row[k]) for k in ('lx','ly','lz')])
   M=np.array([[float(row[f'm{i}{j}']) for j in range(3)] for i in range(3)])
   frame=frames[int(row['donor'])];frame_origin=frame['currentOrigin']
   check(np.linalg.norm(M@(point-frame_origin)-local)<1e-12,'dominant full86 frame-local reconstruction')
   bary=np.array([float(row[f'parent_bary{i}']) for i in range(4)])
   resid=np.array([float(row[f'parent_residual_{a}_m']) for a in 'xyz'])
   check(np.linalg.norm(bary@current+resid-point)<2e-7,'current affine parent residual reconstruction')
   interpolated=np.asarray([float(weight_row[f'w{i}']) for i in range(86)])
   expected=np.zeros(86)
   for k,node in enumerate(parent['tets'][0]):expected+=bary[k]*loaded_weights[int(node)]
   check(np.linalg.norm(interpolated-expected)<1e-12,'parent-tet barycentric full86 interpolation')
   check(abs(interpolated.sum()-1.0)<1e-12,'interpolated full86 normalization')
   check(int(owner_row['dominant_binding_body'])==binding_bodies[int(owner_row['dominant_binding_index'])],'binding lane owner stays explicit')
   check(int(row['donor'])==int(owner_row['dominant_binding_body']),'registered donor/local frame uses interpolated dominant binding')
   check(int(owner_row['parent_geometric_donor_body'])==7,'geometric parent donor remains separately recorded')
  check(result['metrics']['full86DominantBindingDiffersFromParentGeometricDonorRows']>0,'binding owner is not equated with geometric donor')
  bad_weights=root/'bad-parent-full86.csv'
  bad_rows=[{'vertex':i,**{f'w{j}':m.f64(parent_weights[i,j]) for j in range(86)}} for i in range(4)]
  bad_rows[-1]['vertex']='2'
  m.write_csv(bad_weights,m.WEIGHT_HEADER,bad_rows)
  rejected=False
  try:m.load_parent_full86_weights(bad_weights,parent,{'nodesPath':'fixture','nodeSHA256':'fixture'})
  except ValueError:rejected=True
  check(rejected,'duplicate parent-node full86 identity rejected')
  bad=root/'bad-candidate.npz'
  np.savez(bad,points_world_m=points,faces=np.array([[0,1,2]],np.int64),
   face_source_layer=np.array([0],np.int8),face_source_body=np.array([-1],np.int32),face_source_bone=np.array([-1],np.int32),
   face_source_face_id=np.array([0],np.int64),face_source_original_id=np.array([1],np.int64),
   face_source_identity0=np.array([99],np.int64),face_source_identity1=np.array([22],np.int64),
   face_source_identity_schema_by_original_id_json=np.array([json.dumps(schema)]))
  rejected=False
  try:m.validate_candidate(bad,'O',True)
  except ValueError:rejected=True
  check(rejected,'unmatched original face ancestry rejection')
  print('authoring_direct_node_identity=PASS; original_face_ancestry_pin=PASS; current_affine_parent_registration=PASS; dominant_full86_frame_reconstruction=PASS; parent_tet_full86_interpolation=PASS; separate_weight_and_geometric_owners=PASS; bad_weight_row_rejected=PASS; invalid_ancestry_rejected=PASS')
if __name__=='__main__':main()

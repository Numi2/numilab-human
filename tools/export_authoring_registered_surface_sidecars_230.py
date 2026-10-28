#!/usr/bin/env python3
"""Build registered O/I/D sidecars whose node identity is the exact Boolean candidate NPZ row.

This producer is isolated from the published D37 sidecar exporter. It keeps face-level
source identity from the candidate NPZ and never copies D37/NHSKIN node identity onto
new Boolean vertices. Geometry admission remains a separate gate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,sys,tempfile
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import export_child_semantic_labels_217 as native

DOMAIN='derived_registered_skin_authoring'
ROLE={'O':('outerSurface',1,0),'I':('innerSurface',2,1),'D':('deepSurface',3,2)}
MATERIAL={'O':15,'I':15,'D':16}
INFERRED='inferred_current_parent_tet_same_material_donor_region'
NODE_HEADER=['vertex','donor','wx','wy','wz','nx','ny','nz','lx','ly','lz',
 *[f'm{i}{j}' for i in range(3) for j in range(3)],
 'source_domain','source_mapping_kind','source_vertex_row','source_face_row',
 'source_bary0','source_bary1','source_bary2','source_residual_x_m','source_residual_y_m','source_residual_z_m',
 'source_registration_method','frame_owner_method','parent_registration_kind','parent_tet_row',
 'parent_bary0','parent_bary1','parent_bary2','parent_bary3',
 'parent_residual_x_m','parent_residual_y_m','parent_residual_z_m','parent_registration_distance_m',
 'parent_registration_tie_count','parent_tied_tet_rows_json','parent_material_index',
 'parent_geometric_donor_body','parent_anatomical_region_id','parent_stable_source_face_id']
FACE_HEADER=['face','v0','v1','v2','face_source_layer','face_source_body','face_source_bone',
 'face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1']
PARENT_HEADER=['layer','surface_vertex_row','parent_registration_kind','parent_tet_row',
 'parent_bary0','parent_bary1','parent_bary2','parent_bary3',
 'parent_residual_x_m','parent_residual_y_m','parent_residual_z_m',
 'parent_registration_distance_m','parent_registration_tie_count','parent_tied_tet_rows_json',
 'material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id',
 'frame_owner_method','source_domain','source_mapping_kind','source_vertex_row','source_face_row',
 'source_bary0','source_bary1','source_bary2','source_residual_x_m','source_residual_y_m','source_residual_z_m']
WEIGHT_HEADER=['vertex']+[f'w{i}' for i in range(86)]
WEIGHT_OWNER_HEADER=['vertex','dominant_binding_index','dominant_binding_body','parent_geometric_donor_body','parent_tet_row','parent_bary0','parent_bary1','parent_bary2','parent_bary3','parent_anatomical_region_id']

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def f64(x:float)->str:return format(float(x),'.17g')
def read_csv(path:Path)->list[dict[str,str]]:
 with path.open('r',encoding='utf-8',newline='') as f:
  r=csv.DictReader(f);rows=list(r)
  if not r.fieldnames or any(None in x for x in rows):raise ValueError(f'malformed CSV {path}')
 return rows
def write_csv(path:Path,header:list[str],rows:list[dict[str,Any]])->None:
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=header,extrasaction='raise',lineterminator='\n');w.writeheader();w.writerows(rows)

def validate_candidate(path:Path,layer:str,require_f32:bool=True)->dict[str,Any]:
 z=np.load(path,allow_pickle=False)
 required=['points_world_m','faces','face_source_layer','face_source_body','face_source_bone',
           'face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1',
           'face_source_identity_schema_by_original_id_json']
 missing=[k for k in required if k not in z.files]
 if missing:raise ValueError(f'{layer} candidate lacks typed source arrays {missing}: {path}')
 x=np.asarray(z['points_world_m']);f=np.asarray(z['faces'],dtype=np.int64)
 if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or len(x)<3:raise ValueError(f'{layer} candidate points invalid')
 if require_f32 and x.dtype!=np.float32:raise ValueError(f'{layer} candidate is not finalized Float32 points: {x.dtype}')
 if f.ndim!=2 or f.shape[1]!=3 or len(f)<1 or np.any(f<0) or np.any(f>=len(x)) or np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2])):
  raise ValueError(f'{layer} candidate face indices invalid')
 if len(np.unique(f))==0 or len(np.unique(f))<len(x):raise ValueError(f'{layer} candidate has unreferenced node rows')
 arrays={k:np.asarray(z[k]) for k in required if k.startswith('face_source_') and k!='face_source_identity_schema_by_original_id_json'}
 if any(a.shape!=(len(f),) for a in arrays.values()):raise ValueError(f'{layer} face source metadata row count differs from face table')
 try: schema=json.loads(str(z['face_source_identity_schema_by_original_id_json'][0]))
 except Exception as e:raise ValueError(f'{layer} source identity schema is malformed') from e
 if not isinstance(schema,dict) or not schema:raise ValueError(f'{layer} source identity schema is empty')
 orig=np.asarray(z['face_source_original_id'],dtype=np.int64)
 id0=np.asarray(z['face_source_identity0'],dtype=np.int64)
 id1=np.asarray(z['face_source_identity1'],dtype=np.int64)
 lay=np.asarray(z['face_source_layer'],dtype=np.int64)
 expected_original={'O':1,'I':3,'D':5}[layer]
 allowed_bone_role=3
 if not set(map(str,np.unique(orig))).issubset(schema):raise ValueError(f'{layer} output has face original IDs absent from its schema')
 if expected_original not in set(map(int,np.unique(orig))):raise ValueError(f'{layer} output omitted its declared original layer source')
 errors=[]; ancestry={}
 for sid in sorted(set(map(int,orig))):
  mask=orig==sid;info=schema[str(sid)]
  if sid==expected_original:
   if int(np.unique(lay[mask])[0])!=ROLE[layer][2] or len(np.unique(lay[mask]))!=1:errors.append(f'{sid}: wrong original source layer role')
   source_path=Path(info.get('path',''))
   if not source_path.is_file() or sha(source_path)!=info.get('sha256'):errors.append(f'{sid}: original source file/hash unavailable or mismatched')
   else:
    src=np.load(source_path,allow_pickle=False);k0,k1=info.get('identity_keys',['',''])
    if k0 not in src.files or k1 not in src.files:errors.append(f'{sid}: original source identity arrays absent')
    else:
     pairs=set(zip(np.asarray(src[k0],dtype=np.int64).tolist(),np.asarray(src[k1],dtype=np.int64).tolist()))
     outpairs=set(zip(id0[mask].tolist(),id1[mask].tolist()))
     missing_pairs=outpairs-pairs
     if missing_pairs:errors.append(f'{sid}: {len(missing_pairs)} output face identity pairs absent from pinned source')
   ancestry[str(sid)]={'sourcePath':str(info.get('path','')),'sourceSHA256':info.get('sha256'),
      'outputFaces':int(mask.sum()),'identityKeys':info.get('identity_keys')}
  else:
   if info.get('role')!='BONE' or not isinstance(info.get('face_count'),int) or info['face_count']<=0:
    errors.append(f'{sid}: bone source schema lacks declared face_count')
   elif np.any(id0[mask]<0) or np.any(id0[mask]>=int(info['face_count'])) or np.any(id1[mask]!=-1):
    errors.append(f'{sid}: bone envelope identity is outside declared face rows')
   if len(np.unique(lay[mask]))!=1 or int(np.unique(lay[mask])[0])!=allowed_bone_role:
    errors.append(f'{sid}: bone source has wrong layer role')
   ancestry[str(sid)]={'role':'BONE','outputFaces':int(mask.sum()),'envelopeFaceCount':info.get('face_count'),
      'envelopeKernelMeshSHA256':info.get('envelope_kernel_mesh_sha256'),
      'parentBankSourceID':info.get('parent_bank_source_id'),'parentBankWorldGeometrySHA256':info.get('parent_bank_world_geometry_sha256'),
      'envelopeOperation':info.get('envelope_operation'),'envelopeRadiusM':info.get('envelope_radius_m'),
      'identityKeys':info.get('identity_keys')}
 if errors:raise ValueError(f'{layer} candidate source ancestry invalid: '+ '; '.join(errors[:8]))
 used=np.unique(f)
 # The candidate table itself is a distinct typed source; every point maps one-to-one by row.
 role,role_index,layer_code=ROLE[layer]
 return {'path':str(path),'sha256':sha(path),'layer':layer,'role':role,'roleIndex':role_index,
         'domain':DOMAIN,'vertexCount':len(x),'faceCount':len(f),'dtype':str(x.dtype),
         'points':x.astype(np.float64),'faces':f,'sourceSchema':schema,'faceAncestry':ancestry,
         'faceArrays':arrays,'sourceOriginalFaceIDs':sorted(map(int,np.unique(orig))),
         'sourceRowsAllPointRowsDirect':True,'sourceParentLineagePreservedPerFace':True}

def surface_normals(points:np.ndarray,faces:np.ndarray)->np.ndarray:
 tri=points[faces].astype(np.float64);fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 lengths=np.linalg.norm(fn,axis=1)
 if not np.isfinite(fn).all() or np.any(lengths<=0):raise ValueError('candidate has degenerate or nonfinite Float32 face')
 out=np.zeros_like(points,dtype=np.float64)
 for c in range(3):np.add.at(out,faces[:,c],fn)
 ln=np.linalg.norm(out,axis=1)
 if not np.isfinite(out).all() or np.any(ln<=0):raise ValueError('candidate has unreferenced or canceling vertex normals')
 return out/ln[:,None]

def load_frames(path:Path)->tuple[dict[int,dict[str,np.ndarray]],dict[str,Any],list[int]]:
 d=json.loads(path.read_text());rows=d.get('bodies')
 if not isinstance(rows,list) or len(rows)!=int(d.get('binding_count',-1)):raise ValueError('pose snapshot must contain every explicitly registered body frame')
 out={};binding_by_body={};binding_bodies=[]
 for binding,row in enumerate(rows):
  body=int(row['body_index'])
  if body in out:raise ValueError(f'duplicate pose body {body}')
  binding_bodies.append(body)
  entry={}
  for state in ('current','rest'):
   origin=np.asarray(row[state]['position_world_m'],dtype=np.float64)
   R=Rotation.from_quat(np.asarray(row[state]['orientation_world_xyzw'],dtype=np.float64)).as_matrix()
   if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(R).all() or np.linalg.det(R)<0.999999:
    raise ValueError(f'invalid {state} pose for body {body}')
   entry[state+'Origin']=origin;entry[state+'R']=R
  entry['registeredBindingRow']=binding
  out[body]=entry;binding_by_body[body]=binding
 return out,{'sha256':sha(path),'path':str(path),'registeredPoseBodyCount':len(out),
             'snapshotBindingCount':int(d['binding_count']),'poseBodyRows':sorted(out),'bindingByBody':binding_by_body},binding_bodies

def build_material_indices(parent:dict[str,Any])->dict[int,dict[str,Any]]:
 material_indices={}
 for m in (15,16):
  rows=np.flatnonzero(parent['materials']==m)
  if len(rows)==0:continue
  tets=parent['tets'][rows];tetpoints=parent['points'][tets];cent=tetpoints.mean(axis=1)
  radii=np.linalg.norm(tetpoints-cent[:,None,:],axis=2).max(axis=1)
  material_indices[m]={'globalRows':rows,'centroids':cent,'tree':cKDTree(cent),'maxRadius':float(radii.max())+native.QUERY_PAD_M}
 return material_indices

def load_parent_current(base:Path)->tuple[dict[str,Any],dict[str,Any],dict[int,dict[str,Any]],dict[str,Any]]:
 parent=native.read_parent(base)
 npth=Path(str(base)+'.nodes.csv'); rows=read_csv(npth)
 if len(rows)!=len(parent['points']):raise ValueError('parent current node table row count mismatch')
 current=np.empty_like(parent['points'],dtype=np.float64)
 for i,r in enumerate(rows):
  if int(r['parent_node_index'])!=i:raise ValueError(f'parent current node row mismatch at {i}')
  current[i]=[float(r[k]) for k in ('current_x_m','current_y_m','current_z_m')]
 if not np.isfinite(current).all():raise ValueError('parent current-world nodes are nonfinite')
 current_parent=dict(parent);current_parent['points']=current
 six=np.asarray([native.tet_volume(current[t]) for t in current_parent['tets']])
 if not np.isfinite(six).all() or np.any(six<=0):raise ValueError('parent current-world tetrahedra are inverted or collapsed')
 index=native.build_index(current_parent)
 material_indices=build_material_indices(current_parent)
 if any(m not in material_indices for m in (15,16)):raise ValueError('actual parent lacks a required skin/fat material domain')
 return current_parent,index,material_indices,{'nodesPath':str(npth),'nodeSHA256':sha(npth),'massPath':str(base),'massSHA256':sha(base),
  'sourcePath':str(Path(str(base)+'.source.csv')),'sourceSHA256':sha(Path(str(base)+'.source.csv')),
  'tetsPath':str(Path(str(base)+'.tets.csv')),'tetsSHA256':sha(Path(str(base)+'.tets.csv')),
  'labelsPath':str(Path(str(base)+'.labels.csv')),'labelsSHA256':sha(Path(str(base)+'.labels.csv')),
  'nodeCount':len(current),'tetCount':len(six),'currentPositiveTets':int(len(six))}

def load_parent_full86_weights(path:Path,parent:dict[str,Any],parentmeta:dict[str,Any])->tuple[np.ndarray,dict[str,Any]]:
 rows=read_csv(path);n=len(parent['points']);expected=['vertex']+[f'w{i}' for i in range(86)]
 if not rows:raise ValueError('parent full86 weight table is empty')
 if list(rows[0].keys())!=expected:raise ValueError('parent full86 weight table must use existing vertex,w0..w85 CSV columns')
 if len(rows)!=n:raise ValueError(f'parent full86 table row count {len(rows)} differs from captured parent nodes {n}')
 weights=np.empty((n,86),dtype=np.float64);seen=np.zeros(n,dtype=np.uint8)
 for row in rows:
  vertex=int(row['vertex'])
  if vertex<0 or vertex>=n or seen[vertex]:raise ValueError(f'parent full86 row has duplicate/out-of-range vertex {vertex}')
  values=np.asarray([float(row[f'w{i}']) for i in range(86)],dtype=np.float64)
  if not np.isfinite(values).all() or np.any(values<0.0):raise ValueError(f'parent full86 row {vertex} has invalid weights')
  total=float(values.sum())
  if abs(total-1.0)>1.0e-4:raise ValueError(f'parent full86 row {vertex} does not normalize: {total:.17g}')
  weights[vertex]=values;seen[vertex]=1
 if not np.all(seen):raise ValueError('parent full86 table omits one or more exact parent_node_index rows')
 return weights,{'path':str(path),'sha256':sha(path),'header':expected,'nodeCount':n,'bindingCount':86,
  'rowDomain':'parent_node_index','parentNodesPath':parentmeta['nodesPath'],'parentNodesSHA256':parentmeta['nodeSHA256'],
  'sourceMeaning':'actual WholeBodySkinSourceConsumerRows fullBodyWeights86 captured on the exact parent node domain'}

def closest_same_material(parent:dict[str,Any],index:dict[str,Any],mat_index:dict[str,Any],material:int,
                          point:np.ndarray,max_distance:float,tie_tol:float)->dict[str,Any]:
 mi=mat_index[material];dist0,local0=mi['tree'].query(point,k=1)
 if not math.isfinite(float(dist0)):raise ValueError('empty same-material parent query')
 seed_row=int(mi['globalRows'][int(local0)])
 _,_,seed_dist=native.closest_point_on_parent_tet(parent,index,seed_row,point)
 radius=float(seed_dist)+mi['maxRadius']+tie_tol+native.QUERY_PAD_M
 local_candidates=mi['tree'].query_ball_point(point,radius)
 measured=[]
 for local in local_candidates:
  row=int(mi['globalRows'][int(local)])
  proj,bary,d=native.closest_point_on_parent_tet(parent,index,row,point)
  measured.append((float(d),row,np.asarray(proj),np.asarray(bary)))
 if not measured:raise ValueError('same-material parent query returned no candidates')
 best=min(x[0] for x in measured)
 ties=sorted((x for x in measured if abs(x[0]-best)<=tie_tol),key=lambda x:(x[0],x[1]))
 if max(float(np.linalg.norm(x[2]-ties[0][2])) for x in ties)>max(tie_tol,1.0e-9):
  raise ValueError('nearest parent tie has distinct projected current-world points')
 chosen=ties[0];distance=chosen[0]
 if distance>max_distance:raise ValueError(f'nearest material-{material} parent residual {distance:.9g} exceeds {max_distance:.9g} m')
 row=chosen[1]
 return {'distance':distance,'tet':row,'projected':chosen[2],'bary':chosen[3],
         'residual':np.asarray(point,dtype=np.float64)-chosen[2],
         'ties':[x[1] for x in ties],'tieCount':len(ties)-1,
         'material':int(parent['materials'][row]),'donor':int(parent['donors'][row]),
         'region':int(parent['regions'][row]),'stableFace':int(parent['faces'][row])}

def fmt_vec(v):return [f64(x) for x in np.asarray(v,dtype=np.float64)]
def register_layer(surface:dict[str,Any],parent:dict[str,Any],index:dict[str,Any],material_index:dict[int,dict[str,Any]],
                   frames:dict[int,dict[str,np.ndarray]],parent_weights86:np.ndarray,binding_bodies:list[int],
                   maximum_distance:float,tie_tol:float)->dict[str,Any]:
 layer=surface['layer'];points=surface['points'];faces=surface['faces'];normals=surface_normals(points,faces)
 if parent_weights86.shape!=(len(parent['points']),86) or len(binding_bodies)!=86 or len(set(binding_bodies))!=86:
  raise ValueError('parent full86 table or binding-body lane map does not match exact 86-body domain')
 rows=[];registrations=[];weight_rows=[];weight_owner_rows=[];distances=[];tie_total=0;frame_owners=set();geometric_donors=set();weight_donor_mismatch=0;max_weight_sum_error=0.0
 for vertex,point in enumerate(points):
  m=MATERIAL[layer]
  reg=closest_same_material(parent,index,material_index,m,point,maximum_distance,tie_tol)
  geometric_donor=reg['donor']
  parent_tet_nodes=parent['tets'][reg['tet']]
  weights=np.asarray(reg['bary'],dtype=np.float64)@parent_weights86[parent_tet_nodes]
  if not np.isfinite(weights).all() or np.any(weights < -1.0e-8):raise ValueError(f'{layer} row {vertex} interpolated full86 weights are invalid')
  weight_sum=float(weights.sum());max_weight_sum_error=max(max_weight_sum_error,abs(weight_sum-1.0))
  if abs(weight_sum-1.0)>1.0e-4:raise ValueError(f'{layer} row {vertex} interpolated full86 weights do not normalize: {weight_sum:.17g}')
  dominant_binding=int(np.argmax(weights));frame_owner=int(binding_bodies[dominant_binding])
  if frame_owner not in frames:raise ValueError(f'{layer} row {vertex} dominant full86 binding body {frame_owner} lacks an actual pose frame')
  if frame_owner!=geometric_donor:weight_donor_mismatch+=1
  frame=frames[frame_owner];R=frame['currentR'];origin=frame['currentOrigin'];local=R.T@(point-origin)
  # Direct authoring-node identity is not D37/NHSKIN ancestry. The output NPZ SHA
  # pins this node table; all original/bone source identity stays face-level.
  row={'vertex':vertex,'donor':frame_owner,'wx':f64(point[0]),'wy':f64(point[1]),'wz':f64(point[2]),
       'nx':f64(normals[vertex,0]),'ny':f64(normals[vertex,1]),'nz':f64(normals[vertex,2]),
       'lx':f64(local[0]),'ly':f64(local[1]),'lz':f64(local[2]),
       'source_domain':DOMAIN,'source_mapping_kind':'vertex','source_vertex_row':vertex,'source_face_row':-1,
       'source_bary0':'0','source_bary1':'0','source_bary2':'0',
       'source_residual_x_m':'0','source_residual_y_m':'0','source_residual_z_m':'0',
       'source_registration_method':'candidate_surface_npz_direct_vertex_identity',
       'frame_owner_method':'parent_tet_full86_dominant_binding_current_pose',
       'parent_registration_kind':INFERRED,'parent_tet_row':reg['tet'],
       **{f'parent_bary{i}':f64(reg['bary'][i]) for i in range(4)},
       'parent_residual_x_m':f64(reg['residual'][0]),'parent_residual_y_m':f64(reg['residual'][1]),
       'parent_residual_z_m':f64(reg['residual'][2]),'parent_registration_distance_m':f64(reg['distance']),
       'parent_registration_tie_count':reg['tieCount'],'parent_tied_tet_rows_json':json.dumps(reg['ties'],separators=(',',':')),
       'parent_material_index':reg['material'],'parent_geometric_donor_body':geometric_donor,
       'parent_anatomical_region_id':reg['region'],'parent_stable_source_face_id':reg['stableFace']}
  for i in range(3):
   for j in range(3):row[f'm{i}{j}']=f64(R.T[i,j])
  rows.append(row);distances.append(reg['distance']);tie_total+=reg['tieCount'];frame_owners.add(frame_owner);geometric_donors.add(geometric_donor)
  weight_rows.append({'vertex':vertex,**{f'w{i}':f64(weights[i]) for i in range(86)}})
  weight_owner_rows.append({'vertex':vertex,'dominant_binding_index':dominant_binding,'dominant_binding_body':frame_owner,
   'parent_geometric_donor_body':geometric_donor,'parent_tet_row':reg['tet'],
   **{f'parent_bary{i}':f64(reg['bary'][i]) for i in range(4)},'parent_anatomical_region_id':reg['region']})
  registrations.append({'layer':layer,'surface_vertex_row':vertex,'parent_registration_kind':INFERRED,
   'parent_tet_row':reg['tet'],**{f'parent_bary{i}':f64(reg['bary'][i]) for i in range(4)},
   'parent_residual_x_m':f64(reg['residual'][0]),'parent_residual_y_m':f64(reg['residual'][1]),
   'parent_residual_z_m':f64(reg['residual'][2]),'parent_registration_distance_m':f64(reg['distance']),
   'parent_registration_tie_count':reg['tieCount'],'parent_tied_tet_rows_json':json.dumps(reg['ties'],separators=(',',':')),
   'material_index':reg['material'],'geometric_donor_body':reg['donor'],'anatomical_region_id':reg['region'],
   'stable_source_face_id':reg['stableFace'],'frame_owner_method':'parent_tet_full86_dominant_binding_current_pose',
   'source_domain':DOMAIN,'source_mapping_kind':'vertex','source_vertex_row':vertex,'source_face_row':-1,
   'source_bary0':'0','source_bary1':'0','source_bary2':'0',
   'source_residual_x_m':'0','source_residual_y_m':'0','source_residual_z_m':'0'})
 face_rows=[];a=surface['faceArrays'];F=faces
 for frow,tri in enumerate(F):
  face_rows.append({'face':frow,'v0':int(tri[0]),'v1':int(tri[1]),'v2':int(tri[2]),
   **{k:int(v[frow]) for k,v in a.items()}})
 role,role_index,_=ROLE[layer]
 regdomain={'domain':DOMAIN,'role':role,'roleIndex':role_index,'sourceSHA256':surface['sha256'],
            'vertexCount':int(len(points)),'faceCount':int(len(faces))}
 return {'nodes':rows,'faces':face_rows,'registrations':registrations,'weightRows':weight_rows,
  'weightOwnerRows':weight_owner_rows,'sourceDomain':regdomain,
  'metrics':{'nodes':len(rows),'faces':len(face_rows),'maximumParentRegistrationDistanceM':max(distances),
   'p95ParentRegistrationDistanceM':float(np.quantile(distances,.95)),'parentRegistrationTieCountTotal':tie_total,
   'selectedFrameOwnerBodies':sorted(frame_owners),'selectedGeometricDonorBodies':sorted(geometric_donors),
   'materialIndex':MATERIAL[layer],
   'maximumInterpolatedFull86NormalizationError':max_weight_sum_error,
   'full86DominantBindingDiffersFromParentGeometricDonorRows':weight_donor_mismatch,
   'allSourceNodeIDsDirectCandidateRowIdentity':True}}

def frame_csv_rows(frames:list[dict[str,Any]])->list[dict[str,Any]]:
 out=[]
 for binding,row in enumerate(frames):
  donor=int(row['body_index']);x={'binding':binding,'donor':donor}
  for state in ('current','rest'):
   origin=np.asarray(row[state]['position_world_m'],dtype=np.float64)
   R=Rotation.from_quat(np.asarray(row[state]['orientation_world_xyzw'],dtype=np.float64)).as_matrix()
   for a,v in zip('xyz',origin):x[f'{state}_{a}']=f64(v)
   for i in range(3):
    for j in range(3):x[f'{state}_R{i}{j}']=f64(R[i,j])
  out.append(x)
 return out

def export(args:argparse.Namespace)->dict[str,Any]:
 if not all((args.outer,args.inner,args.deep,args.parent,args.poses,args.parent_full86_weights,args.out)):
  raise ValueError('complete registered sidecar export requires explicit O/I/D NPZ, parent, poses, parent full86 weights, and output paths')
 paths={'O':Path(args.outer),'I':Path(args.inner),'D':Path(args.deep)}
 if any(not p.is_file() for p in paths.values()):raise ValueError('all three explicit candidate surface paths must exist')
 if args.out.exists():raise ValueError(f'refuse existing output directory {args.out}')
 surfaces={l:validate_candidate(p,l,require_f32=True) for l,p in paths.items()}
 frames,posemeta,binding_bodies=load_frames(args.poses)
 if len(binding_bodies)!=86:raise ValueError('actual pose snapshot does not define exactly 86 binding lanes')
 parent,index,material_indices,parentmeta=load_parent_current(args.parent)
 parent_weights86,parent_weight_meta=load_parent_full86_weights(args.parent_full86_weights,parent,parentmeta)
 staged=args.out.with_name(args.out.name+'.staging')
 if staged.exists():raise ValueError(f'refuse existing staging directory {staged}')
 staged.mkdir(parents=True)
 try:
  outputs={};regrows=[];metrics={}
  for layer in ('O','I','D'):
   result=register_layer(surfaces[layer],parent,index,
       material_indices,frames,parent_weights86,binding_bodies,
       args.maximum_parent_distance_m,args.tie_tolerance_m)
   outputs[layer]=result;metrics[layer]=result['metrics']
   write_csv(staged/f'registered-{layer}.csv',NODE_HEADER,result['nodes'])
   write_csv(staged/f'registered-{layer}-faces.csv',FACE_HEADER,result['faces'])
   write_csv(staged/f'parent-tet-full-weights-{layer}.csv',WEIGHT_HEADER,result['weightRows'])
   write_csv(staged/f'parent-tet-full-weight-owners-{layer}.csv',WEIGHT_OWNER_HEADER,result['weightOwnerRows'])
   regrows.extend(result['registrations'])
  write_csv(staged/'node-parent-registration.csv',PARENT_HEADER,regrows)
  frame_header=['binding','donor',*[f'{s}_{a}' for s in ('current','rest') for a in 'xyz'],
      *[f'{s}_R{i}{j}' for s in ('current','rest') for i in range(3) for j in range(3)]]
  write_csv(staged/'donor-frames.csv',frame_header,frame_csv_rows(json.loads(args.poses.read_text())['bodies']))
  registry=[outputs[l]['sourceDomain'] for l in ('O','I','D')]
  (staged/'source-domain-registry.json').write_text(json.dumps(registry,indent=2,sort_keys=True)+'\n')
  source_schemas={l:surfaces[l]['sourceSchema'] for l in ('O','I','D')}
  output_hashes={p.name:sha(p) for p in staged.iterdir() if p.is_file()}
  receipt={'schema':'registered-authoring-surface-sidecars-receipt.v1','status':'reference_registration_ready_geometry_not_admitted',
   'physical_admission':False,'geometry_admission':False,'native_steps':0,
   'method':'node source identity is direct row identity into each exact Boolean candidate NPZ; nearest current parent registration is bounded to the layer material set; full86 weights are interpolated by parent tet barycentrics; donor/local frames use the dominant interpolated binding pose, separately from parent geometric donor and cell labels.',
   'source_ancestry_claimed_from_D37_for_candidate_nodes':False,
   'face_level_source_ancestry_preserved':True,'face_ancestry_source_arrays':'face_source_original_id,face_source_identity0,face_source_identity1 plus pinned face_source_identity_schema_by_original_id_json in each candidate NPZ',
   'candidateSurfaces':{l:{'path':surfaces[l]['path'],'sha256':surfaces[l]['sha256'],'points':surfaces[l]['vertexCount'],'faces':surfaces[l]['faceCount'],
      'sourceOriginalFaceIDs':surfaces[l]['sourceOriginalFaceIDs'],'sourceFaceAncestry':surfaces[l]['faceAncestry']} for l in ('O','I','D')},
   'sourceDomainRegistry':registry,'parentCurrent':parentmeta,'poseSnapshot':posemeta,
   'parentFull86Weights':parent_weight_meta,
   'full86WeightMethod':'For each candidate node, interpolate the exact captured parent node full86 vectors with the selected current-world parent tet barycentrics. The dominant binding pose owns RegisteredSurface donor/local frame fields; parent geometric donor/material/region remain separate registration labels and source-namespace identity remains separate.',
   'bindingBodyByLane':binding_bodies,
   'limits':{'maximumParentRegistrationDistanceM':args.maximum_parent_distance_m,'tieToleranceM':args.tie_tolerance_m,
      'materialBySurface':MATERIAL,'selectedTie':'minimum distance then lowest parent_tet_row among geometrically indistinguishable projections',
      'outsideOrMissingPose':'reject'},
   'results':metrics,'outputSHA256':output_hashes}
  (staged/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
  args.out.parent.mkdir(parents=True,exist_ok=True);os.replace(staged,args.out)
  receipt['receiptSHA256']=sha(args.out/'receipt.json')
  print(json.dumps({'output':str(args.out),'receiptSHA256':receipt['receiptSHA256'],'results':metrics,'sourceDomainRegistry':registry,'physical_admission':False},indent=2))
  return receipt
 except Exception:
  shutil.rmtree(staged,ignore_errors=True);raise

def sample_check(paths:dict[str,Path],parent_path:Path,pose_path:Path,samples:int,max_distance:float,tie_tol:float,
                 parent_full86_path:Path|None=None)->dict[str,Any]:
 surfaces={l:validate_candidate(p,l,require_f32=True) for l,p in paths.items() if p is not None}
 frames,posemeta,binding_bodies=load_frames(pose_path);parent,index,material_indices,parentmeta=load_parent_current(parent_path)
 parent_weights86=weight_meta=None
 if parent_full86_path is not None:parent_weights86,weight_meta=load_parent_full86_weights(parent_full86_path,parent,parentmeta)
 out={}
 for layer,s in surfaces.items():
  n=len(s['points']);ids=sorted(set([0,max(0,n//2),n-1]))[:samples]
  records=[]
  for i in ids:
   reg=closest_same_material(parent,index,material_indices,MATERIAL[layer],s['points'][i],max_distance,tie_tol)
   frame_owner=reg['donor'];weight_owner=None;dominant_binding=None
   if parent_weights86 is not None:
    tet_nodes=parent['tets'][reg['tet']]
    weights=np.asarray(reg['bary'],dtype=np.float64)@parent_weights86[tet_nodes]
    if not np.isfinite(weights).all() or np.any(weights < -1.0e-8) or abs(float(weights.sum())-1.0)>1.0e-4:
     raise ValueError(f'sampled {layer} row {i} interpolated full86 weights are invalid')
    dominant_binding=int(np.argmax(weights));weight_owner=int(binding_bodies[dominant_binding]);frame_owner=weight_owner
   if frame_owner not in frames:raise ValueError(f'sampled {layer} row {i} frame owner {frame_owner} absent from actual pose snapshot')
   fr=frames[frame_owner];local=fr['currentR'].T@(s['points'][i]-fr['currentOrigin'])
   local_recon=np.linalg.norm(fr['currentR']@local+fr['currentOrigin']-s['points'][i])
   point_recon=np.linalg.norm(reg['projected']+reg['residual']-s['points'][i])
   records.append({'vertex':i,'parentTet':reg['tet'],'material':reg['material'],'geometricDonor':reg['donor'],
      'frameOwner':frame_owner,'dominantBindingIndex':dominant_binding,'dominantBindingBody':weight_owner,
      'frameOwnerEqualsParentGeometricDonor':frame_owner==reg['donor'],
      'region':reg['region'],'distanceM':reg['distance'],'tieCount':reg['tieCount'],
      'currentParentReconstructionM':float(point_recon),'donorLocalReconstructionM':float(local_recon)})
  out[layer]={'candidateSHA256':s['sha256'],'nodeCount':n,'sampled':records}
 return {'status':'sample_only_not_a_complete_packet','physical_admission':False,'parent':parentmeta,'poseSnapshot':posemeta,
  'parentFull86Weights':weight_meta,'layers':out}

def main()->None:
 ap=argparse.ArgumentParser()
 ap.add_argument('--outer',type=Path);ap.add_argument('--inner',type=Path);ap.add_argument('--deep',type=Path)
 ap.add_argument('--parent',type=Path,help='base path without .nodes.csv/.tets.csv suffix')
 ap.add_argument('--poses',type=Path)
 ap.add_argument('--parent-full86-weights',type=Path,help='captured parent_node_index keyed vertex,w0..w85 table')
 ap.add_argument('--out',type=Path)
 ap.add_argument('--maximum-parent-distance-m',type=float,default=0.02)
 ap.add_argument('--tie-tolerance-m',type=float,default=1.0e-8)
 ap.add_argument('--preflight-only',action='store_true')
 ap.add_argument('--sample-check-count',type=int,default=0)
 args=ap.parse_args()
 if not math.isfinite(args.maximum_parent_distance_m) or args.maximum_parent_distance_m<=0 or not math.isfinite(args.tie_tolerance_m) or args.tie_tolerance_m<0:raise ValueError('invalid registration bound/tie tolerance')
 if args.preflight_only:
  paths={l:p for l,p in [('O',args.outer),('I',args.inner),('D',args.deep)] if p is not None}
  if not paths:raise ValueError('preflight-only requires at least one explicit candidate surface')
  surfaces={l:validate_candidate(p,l,require_f32=True) for l,p in paths.items()}
  out={'status':'source-schema-and-ancestry-preflight-only','physical_admission':False,
       'layers':{l:{'path':s['path'],'sha256':s['sha256'],'points':s['vertexCount'],'faces':s['faceCount'],
        'role':s['role'],'roleIndex':s['roleIndex'],'domain':s['domain'],'sourceOriginalFaceIDs':s['sourceOriginalFaceIDs'],
        'faceSourceAncestry':s['faceAncestry']} for l,s in surfaces.items()},
       'completeOID':set(paths)=={'O','I','D'}}
  if args.sample_check_count:
   if not(args.parent and args.poses):raise ValueError('sample-check-count requires actual --parent and --poses')
   out['parentRegistrationSample']=sample_check(paths,args.parent,args.poses,args.sample_check_count,
       args.maximum_parent_distance_m,args.tie_tolerance_m,args.parent_full86_weights)
  print(json.dumps(out,indent=2,sort_keys=True));return
 if args.sample_check_count:raise ValueError('--sample-check-count is preflight-only')
 export(args)
if __name__=='__main__':main()

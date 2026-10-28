#!/usr/bin/env python3
"""Build registered O/I/D sidecars whose node identity is the exact Boolean candidate NPZ row.

This producer is isolated from the published D37 sidecar exporter. It keeps face-level
source identity from the candidate NPZ and never copies D37/NHSKIN node identity onto
new Boolean vertices. Geometry admission remains a separate gate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,sys,tempfile,time
from fractions import Fraction
from decimal import Decimal, localcontext
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
FRAME_HEADER=['binding','donor',
 *[f'current_{a}' for a in 'xyz'],
 *[f'current_R{i}{j}' for i in range(3) for j in range(3)],
 *[f'rest_{a}' for a in 'xyz'],
 *[f'rest_R{i}{j}' for i in range(3) for j in range(3)]]

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
  elif info.get('derived_operation') and info.get('role')==layer:
   # A same-layer repair is a typed derived source, never a bone or NHSKIN ID.
   if len(np.unique(lay[mask]))!=1 or int(lay[mask][0])!=ROLE[layer][2]:
    errors.append(f'{sid}: derived source has wrong layer role')
   source_path=Path(info.get('path',''))
   parents=info.get('derived_parent_assets')
   if not isinstance(parents,list) or not parents:
    errors.append(f'{sid}: derived source has no pinned parent asset')
   else:
    for asset in parents:
     if not isinstance(asset,dict) or not Path(asset.get('path','')).is_file() or sha(Path(asset['path']))!=asset.get('sha256'):
      errors.append(f'{sid}: derived parent file/hash unavailable or mismatched')
   if not isinstance(info.get('derived_parent_source_ids'),list) or not info['derived_parent_source_ids']:
    errors.append(f'{sid}: derived source lacks explicit parent source identities')
   if not source_path.is_file() or sha(source_path)!=info.get('sha256'):
    errors.append(f'{sid}: derived source file/hash unavailable or mismatched')
   else:
    src=np.load(source_path,allow_pickle=False)
    keys=info.get('identity_keys')
    face_key=info.get('identity_face_id_key')
    if not isinstance(keys,list) or len(keys)!=2 or any(k not in src.files for k in keys) or face_key not in src.files:
     errors.append(f'{sid}: derived source identity/face arrays absent')
    else:
     a,b,c=(np.asarray(src[k],dtype=np.int64) for k in [face_key,*keys])
     if a.ndim!=1 or b.shape!=a.shape or c.shape!=a.shape:
      errors.append(f'{sid}: derived source identity array domains differ')
     else:
      selected=np.ones(len(a),dtype=bool)
      filter_key=info.get('identity_original_id_key')
      if filter_key is not None:
       if filter_key not in src.files or np.asarray(src[filter_key]).shape!=a.shape or info.get('identity_original_id')!=sid:
        errors.append(f'{sid}: derived source original-ID filter is invalid');selected[:]=False
       else:selected=np.asarray(src[filter_key],dtype=np.int64)==sid
      if not np.any(selected):errors.append(f'{sid}: derived source filter selects no source faces')
      triples=set(zip(a[selected].tolist(),b[selected].tolist(),c[selected].tolist()))
      outtriples=set(zip(np.asarray(z['face_source_face_id'],dtype=np.int64)[mask].tolist(),id0[mask].tolist(),id1[mask].tolist()))
      if outtriples-triples:errors.append(f'{sid}: output derived face identities absent from pinned source view')
   ancestry[str(sid)]={'role':layer,'operation':info.get('derived_operation'),'sourcePath':str(source_path),
    'sourceSHA256':info.get('sha256'),'outputFaces':int(mask.sum()),'identityKeys':info.get('identity_keys'),
    'identityFaceIDKey':info.get('identity_face_id_key'),'identityOriginalIDFilter':info.get('identity_original_id'),
    'parentAssets':parents,'parentSourceIDs':info.get('derived_parent_source_ids'),
    'sourceGeometryScope':info.get('source_geometry_scope')}
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
 if not isinstance(rows,list) or len(rows)!=int(d.get('binding_count',-1)):
  raise ValueError('pose snapshot must contain every explicitly registered body frame')
 all_rows=d.get('all_bodies');use_native_matrices=False
 if all_rows is not None:
  if not isinstance(all_rows,list):raise ValueError('all_bodies pose snapshot must be a body-indexed list')
  for i,native_row in enumerate(all_rows):
   if not isinstance(native_row,dict) or int(native_row.get('body_index',-1))!=i:
    raise ValueError(f'all_bodies body-index identity mismatch at row {i}')
  matrix_flags=[]
  for row in rows:
   body=int(row['body_index'])
   if body<0 or body>=len(all_rows):raise ValueError(f'registered body {body} is outside all_bodies')
   for state in ('current','rest'):
    matrix_flags.append('local_to_world_row_major_f32' in all_rows[body].get(state,{}))
  if any(matrix_flags) and not all(matrix_flags):
   raise ValueError('all_bodies pose snapshot has incomplete native current/rest matrices for registered bodies')
  use_native_matrices=bool(matrix_flags and all(matrix_flags))
 out={};binding_by_body={};binding_bodies=[];orthogonality_error_max=0.0;determinant_error_max=0.0
 f32_eps=float(np.finfo(np.float32).eps);rotation_bound=8.0*f32_eps
 for binding,row in enumerate(rows):
  body=int(row['body_index'])
  if body in out:raise ValueError(f'duplicate pose body {body}')
  binding_bodies.append(body);entry={}
  native_row=all_rows[body] if use_native_matrices else None
  if use_native_matrices and int(native_row.get('body_index',-1))!=body:
   raise ValueError(f'all_bodies identity differs for registered body {body}')
  for state in ('current','rest'):
   binding_origin=np.asarray(row[state]['position_world_m'],dtype=np.float64)
   binding_quat=np.asarray(row[state]['orientation_world_xyzw'],dtype=np.float64)
   if binding_origin.shape!=(3,) or not np.isfinite(binding_origin).all() or binding_quat.shape!=(4,) or not np.isfinite(binding_quat).all():
    raise ValueError(f'invalid {state} pose for body {body}')
   if use_native_matrices:
    source=native_row[state]
    native_origin_raw=np.asarray(source['position_world_m'],dtype=np.float64)
    native_quat_raw=np.asarray(source['orientation_world_xyzw'],dtype=np.float64)
    if native_origin_raw.shape!=(3,) or not np.isfinite(native_origin_raw).all() or native_quat_raw.shape!=(4,) or not np.isfinite(native_quat_raw).all():
     raise ValueError(f'invalid native {state} origin/quaternion for body {body}')
    if not np.array_equal(binding_origin.astype(np.float32),native_origin_raw.astype(np.float32)) or not np.array_equal(binding_quat.astype(np.float32),native_quat_raw.astype(np.float32)):
     raise ValueError(f'all_bodies {state} pose differs from registered binding body {body}')
    origin=native_origin_raw.astype(np.float32).astype(np.float64)
    matrix_raw=np.asarray(source['local_to_world_row_major_f32'],dtype=np.float64)
    if matrix_raw.shape!=(9,) or not np.isfinite(matrix_raw).all():
     raise ValueError(f'invalid native {state} local_to_world matrix for body {body}')
    R=matrix_raw.astype(np.float32).astype(np.float64).reshape(3,3)
    if not np.array_equal(R.reshape(-1),matrix_raw):
     raise ValueError(f'native {state} local_to_world matrix is not exact Float32 data for body {body}')
    orth_error=float(np.max(np.abs(R.T@R-np.eye(3))))
    det=float(np.linalg.det(R));det_error=abs(det-1.0)
    orthogonality_error_max=max(orthogonality_error_max,orth_error);determinant_error_max=max(determinant_error_max,det_error)
    if orth_error>rotation_bound or det<=0.0 or det_error>rotation_bound:
     raise ValueError(f'native {state} local_to_world matrix is outside the bounded Float32 SO(3) envelope for body {body}: orth={orth_error:.9g}, det={det:.9g}')
    world_to_local=np.linalg.inv(R)
    if not np.isfinite(world_to_local).all():raise ValueError(f'native {state} world_to_local inverse is nonfinite for body {body}')
   else:
    origin=binding_origin;R=Rotation.from_quat(binding_quat).as_matrix()
    if not np.isfinite(R).all() or np.linalg.det(R)<0.999999:
     raise ValueError(f'invalid legacy quaternion rotation for body {body}')
    # Compatibility route preserves the previously emitted transpose exactly.
    world_to_local=R.T
   entry[state+'Origin']=origin;entry[state+'R']=R;entry[state+'WorldToLocalR']=world_to_local
  entry['registeredBindingRow']=binding;entry['nativeMatrixFrame']=use_native_matrices
  out[body]=entry;binding_by_body[body]=binding
 metadata={'sha256':sha(path),'path':str(path),'registeredPoseBodyCount':len(out),
  'snapshotBindingCount':int(d['binding_count']),'poseBodyRows':sorted(out),'bindingByBody':binding_by_body}
 if use_native_matrices:
  metadata.update({'frameSourceMethod':'native_all_bodies_f32_local_to_world_matrix_and_inverse',
   'nativeMatrixFrameBodyCount':len(out),'nativeMatrixOrthogonalityErrorMax':orthogonality_error_max,
   'nativeMatrixDeterminantErrorMax':determinant_error_max,'nativeMatrixSO3Bound':rotation_bound})
 # The legacy record intentionally keeps its previous metadata shape. It does
 # not assert that quaternion-reconstructed frames are native-admitted frames.
 return out,metadata,binding_bodies

def build_material_indices(parent:dict[str,Any])->dict[int,dict[str,Any]]:
 material_indices={}
 for m in (15,16):
  rows=np.flatnonzero(parent['materials']==m)
  if len(rows)==0:continue
  tetpoints=parent['points'][parent['tets'][rows]];cent=tetpoints.mean(axis=1)
  radii=np.linalg.norm(tetpoints-cent[:,None,:],axis=2).max(axis=1)
  if not np.isfinite(radii).all() or np.any(radii<=0):
   raise ValueError('same-material parent radii must be finite and positive')
  # Every tet lies in its own centroid sphere. Size bins avoid using the
  # largest whole-body tet radius for every small hand/skin cell query.
  exponents=np.frexp(radii)[1];bins=[]
  for exponent in np.unique(exponents):
   local=np.flatnonzero(exponents==exponent)
   bins.append({'localRows':local,'tree':cKDTree(cent[local]),
                'maxRadius':float(radii[local].max())+native.QUERY_PAD_M})
  material_indices[m]={'globalRows':rows,'centroids':cent,'tree':cKDTree(cent),
    'maxRadius':float(radii.max())+native.QUERY_PAD_M,'sphereBins':bins,
    'boundsMin':tetpoints.min(axis=1),'boundsMax':tetpoints.max(axis=1)}
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

class ExactProjectionTieError(ValueError):
 def __init__(self,message:str,diagnostic:dict[str,Any]):
  super().__init__(message);self.diagnostic=diagnostic

def _fraction(x:float)->Fraction:
 return Fraction.from_float(float(x))

def _vsub(a,b):return tuple(a[i]-b[i] for i in range(3))
def _vadd_scaled(a,v,s):return tuple(a[i]+s*v[i] for i in range(3))
def _dot_exact(a,b):return sum((a[i]*b[i] for i in range(3)),Fraction(0))
def _dist2_exact(a,b):return _dot_exact(_vsub(a,b),_vsub(a,b))
def _det3_exact(a,b,c):
 return (a[0]*(b[1]*c[2]-b[2]*c[1])-
         b[0]*(a[1]*c[2]-a[2]*c[1])+
         c[0]*(a[1]*b[2]-a[2]*b[1]))
def _ratio_text(x:Fraction)->str:return f'{x.numerator}/{x.denominator}'
def _sqrt_fraction_decimal_text(x:Fraction,precision:int=80)->str:
 with localcontext() as ctx:
  ctx.prec=precision
  return str((Decimal(x.numerator)/Decimal(x.denominator)).sqrt())

def _sqrt_fraction_gap_decimal_text(high:Fraction,low:Fraction,precision:int=80)->str:
 with localcontext() as ctx:
  ctx.prec=precision
  high_distance=(Decimal(high.numerator)/Decimal(high.denominator)).sqrt()
  low_distance=(Decimal(low.numerator)/Decimal(low.denominator)).sqrt()
  return str(high_distance-low_distance)

def _closest_segment_exact(p,a,b):
 ab=_vsub(b,a);den=_dot_exact(ab,ab)
 if den==0:return a
 t=_dot_exact(_vsub(p,a),ab)/den
 t=max(Fraction(0),min(Fraction(1),t))
 return _vadd_scaled(a,ab,t)

def _closest_triangle_exact(p,a,b,c):
 u=_vsub(b,a);v=_vsub(c,a);w=_vsub(p,a)
 d00=_dot_exact(u,u);d01=_dot_exact(u,v);d11=_dot_exact(v,v)
 d20=_dot_exact(w,u);d21=_dot_exact(w,v);den=d00*d11-d01*d01
 if den<=0:raise ValueError('exact nearest projection encountered a degenerate parent tetrahedron face')
 beta=(d11*d20-d01*d21)/den;gamma=(d00*d21-d01*d20)/den;alpha=1-beta-gamma
 if alpha>=0 and beta>=0 and gamma>=0:
  return _vadd_scaled(_vadd_scaled(a,u,beta),v,gamma)
 # Outside the triangle, the exact closest point lies on one of its three edges.
 edge_points=(_closest_segment_exact(p,a,b),_closest_segment_exact(p,b,c),_closest_segment_exact(p,c,a))
 return min(edge_points,key=lambda q:(_dist2_exact(p,q),q))

def _tetra_barycentric_exact(vertices,p):
 a,b,c,d=vertices;u=_vsub(b,a);v=_vsub(c,a);w=_vsub(d,a);rhs=_vsub(p,a)
 det=_det3_exact(u,v,w)
 if det==0:raise ValueError('exact nearest projection encountered a degenerate parent tetrahedron')
 b1=_det3_exact(rhs,v,w)/det;b2=_det3_exact(u,rhs,w)/det;b3=_det3_exact(u,v,rhs)/det
 return (1-b1-b2-b3,b1,b2,b3)

def _closest_tetra_exact(vertices,p):
 bary=_tetra_barycentric_exact(vertices,p)
 if min(bary)>=0:q=p
 else:
  candidates=[]
  for omit in range(4):
   face=tuple(vertices[i] for i in range(4) if i!=omit)
   candidates.append(_closest_triangle_exact(p,*face))
  q=min(candidates,key=lambda x:(_dist2_exact(p,x),x))
  bary=_tetra_barycentric_exact(vertices,q)
 return q,bary,_dist2_exact(p,q)

def _exact_aabb_distance_squared(point,lo,hi):
 total=Fraction(0)
 for k in range(3):
  q=_fraction(point[k]);a=_fraction(lo[k]);b=_fraction(hi[k])
  delta=a-q if q<a else q-b if q>b else Fraction(0)
  total+=delta*delta
 return total

def _exact_nearest_same_material(parent,mi,material,point,seed_row):
 """Certify the nearest projection by exact dyadic arithmetic over all material tets.

 The seed tet supplies an exact upper bound. Every tet whose exact point-to-AABB
 lower bound can meet that upper bound is evaluated exactly; all other tets are
 proven unable to win. The scan covers the complete same-material row set, not a
 floating nearest-K or rounded-distance shortlist.
 """
 p=tuple(_fraction(x) for x in point);rows=np.asarray(mi['globalRows'],dtype=np.int64)
 row_to_local=np.flatnonzero(rows==int(seed_row))
 if len(row_to_local)!=1:raise ValueError('exact near-tie seed is outside its declared material row set')
 def exact_tet(row):
  vertices=tuple(tuple(_fraction(x) for x in parent['points'][int(node)]) for node in parent['tets'][int(row)])
  q,bary,d2=_closest_tetra_exact(vertices,p)
  return {'row':int(row),'q':q,'bary':bary,'d2':d2}
 seed=exact_tet(int(seed_row));seed_d2=seed['d2'];evaluated=[seed];aabb_count=0
 for local,raw_row in enumerate(rows):
  row=int(raw_row)
  if row==int(seed_row):continue
  if _exact_aabb_distance_squared(point,mi['boundsMin'][local],mi['boundsMax'][local])<=seed_d2:
   aabb_count+=1;evaluated.append(exact_tet(row))
 best_d2=min(x['d2'] for x in evaluated)
 minimizers=[x for x in evaluated if x['d2']==best_d2]
 projections={x['q'] for x in minimizers}
 ranked=sorted(evaluated,key=lambda x:(x['d2'],x['row']))[:8]
 projection_best={}
 for item in evaluated:
  incumbent=projection_best.get(item['q'])
  if incumbent is None or (item['d2'],item['row'])<(incumbent['d2'],incumbent['row']):projection_best[item['q']]=item
 ranked_projections=sorted(projection_best.values(),key=lambda x:(x['d2'],x['row']))[:8]
 diagnostic={'method':'exact_binary64_rational_simplex_projection_with_full_material_AABB_certificate',
  'status':'rejected_exact_equal_distance_distinct_projections' if len(projections)>1 else 'accepted_exact_nearest_projection',
  'material':int(material),'pointWorldM':[float(x) for x in point],
  'sameMaterialParentTetCount':int(len(rows)),'exactAABBCandidateCount':int(aabb_count+1),
  'seedParentTetRow':int(seed_row),'seedDistanceSquaredM2Exact':_ratio_text(seed_d2),
  'nearestDistanceSquaredM2Exact':_ratio_text(best_d2),
  'exactMinimizerParentTetRows':sorted(x['row'] for x in minimizers),
  'exactNearestDistinctProjectionCount':int(len(projections)),
  'rankedExactCandidates':[{'parentTetRow':x['row'],'distanceSquaredM2Exact':_ratio_text(x['d2']),
      'projectionWorldMExact':[_ratio_text(v) for v in x['q']],
      'barycentricExact':[_ratio_text(v) for v in x['bary']]} for x in ranked],
  'rankedDistinctProjections':[{'representativeParentTetRow':x['row'],'distanceSquaredM2Exact':_ratio_text(x['d2']),
      'projectionWorldMExact':[_ratio_text(v) for v in x['q']],
      'distanceMDecimal80':_sqrt_fraction_decimal_text(x['d2'])} for x in ranked_projections],
  'nextDistinctProjectionDistanceSquaredGapM2Exact':(_ratio_text(ranked_projections[1]['d2']-best_d2) if len(ranked_projections)>1 else None),
  'nextDistinctProjectionDistanceGapMDecimal80':(_sqrt_fraction_gap_decimal_text(
    ranked_projections[1]['d2'],best_d2) if len(ranked_projections)>1 else None),
  'candidateCompletenessProof':'Every same-material parent tetrahedron was scanned; exact point-to-AABB lower bounds greater than the seed tetrahedron exact distance were excluded.'}
 if len(projections)>1:
  raise ExactProjectionTieError('nearest parent tie has distinct projected current-world points (exact equal-distance projections)',diagnostic)
 chosen=min(minimizers,key=lambda x:x['row']);ties=sorted(x['row'] for x in minimizers)
 diagnostic['selectedParentTetRow']=chosen['row'];diagnostic['sameProjectionExactTieCount']=len(ties)-1
 diagnostic['selectedProjectionWorldMExact']=[_ratio_text(v) for v in chosen['q']]
 diagnostic['selectedBarycentricExact']=[_ratio_text(v) for v in chosen['bary']]
 return {'row':chosen['row'],'q':chosen['q'],'bary':chosen['bary'],'d2':best_d2,'ties':ties,'diagnostic':diagnostic}

def closest_same_material(parent:dict[str,Any],index:dict[str,Any],mat_index:dict[str,Any],material:int,
                          point:np.ndarray,max_distance:float,tie_tol:float)->dict[str,Any]:
 mi=mat_index[material];dist0,local0=mi['tree'].query(point,k=1)
 if not math.isfinite(float(dist0)):raise ValueError('empty same-material parent query')
 seed_row=int(mi['globalRows'][int(local0)])
 seed_proj,seed_bary,seed_dist=native.closest_point_on_parent_tet(parent,index,seed_row,point)
 # Triangle inequality certifies this complete candidate set, including all
 # current tie projections. No fixed nearest-K truncation is used.
 local_candidates=[]
 for group in mi['sphereBins']:
  radius=float(seed_dist)+group['maxRadius']+tie_tol+native.QUERY_PAD_M
  hit=group['tree'].query_ball_point(point,radius)
  local_candidates.extend(group['localRows'][hit].tolist())
 local_candidates=np.asarray(local_candidates,dtype=np.int64)
 if not len(local_candidates):raise ValueError('same-material parent query returned no candidates')
 # A tet is contained in its AABB. Its point-to-box distance is therefore a
 # lower bound on point-to-tet distance; padding protects boundary roundoff.
 delta=np.maximum(np.maximum(mi['boundsMin'][local_candidates]-point,
                            point-mi['boundsMax'][local_candidates]),0.0)
 lower=np.linalg.norm(delta,axis=1)
 order=np.argsort(lower,kind='stable')
 measured=[(float(seed_dist),seed_row,np.asarray(seed_proj),np.asarray(seed_bary))]
 best=float(seed_dist)
 for j in order:
  if float(lower[j])>best+tie_tol+native.QUERY_PAD_M:break
  row=int(mi['globalRows'][int(local_candidates[j])])
  if row==seed_row:continue
  proj,bary,d=native.closest_point_on_parent_tet(parent,index,row,point)
  measured.append((float(d),row,np.asarray(proj),np.asarray(bary)))
  best=min(best,float(d))
 ties=sorted((x for x in measured if abs(x[0]-best)<=tie_tol),key=lambda x:(x[0],x[1]))
 exact_fallback=None
 if max(float(np.linalg.norm(x[2]-ties[0][2])) for x in ties)>max(tie_tol,1.0e-9):
  exact_started=time.perf_counter()
  exact=_exact_nearest_same_material(parent,mi,material,point,seed_row)
  exact['diagnostic']['exactFallbackElapsedSeconds']=time.perf_counter()-exact_started
  chosen_row=exact['row'];q=np.asarray([float(x) for x in exact['q']],dtype=np.float64)
  bary=np.asarray([float(x) for x in exact['bary']],dtype=np.float64)
  distance=math.sqrt(float(exact['d2']));ties=exact['ties'];exact_fallback=exact['diagnostic']
 else:
  chosen=ties[0];chosen_row=chosen[1];q=chosen[2];bary=chosen[3];distance=chosen[0]
 if distance>max_distance:raise ValueError(f'nearest material-{material} parent residual {distance:.9g} exceeds {max_distance:.9g} m')
 row=chosen_row
 result={'distance':distance,'tet':row,'projected':q,'bary':bary,
         'residual':np.asarray(point,dtype=np.float64)-q,
         'ties':ties if exact_fallback is not None else [x[1] for x in ties],
         'tieCount':len(ties)-1,
         'material':int(parent['materials'][row]),'donor':int(parent['donors'][row]),
         'region':int(parent['regions'][row]),'stableFace':int(parent['faces'][row])}
 if exact_fallback is not None:result['exactTieFallback']=exact_fallback
 return result

def fmt_vec(v):return [f64(x) for x in np.asarray(v,dtype=np.float64)]
def register_layer(surface:dict[str,Any],parent:dict[str,Any],index:dict[str,Any],material_index:dict[int,dict[str,Any]],
                   frames:dict[int,dict[str,np.ndarray]],parent_weights86:np.ndarray,binding_bodies:list[int],
                   maximum_distance:float,tie_tol:float)->dict[str,Any]:
 layer=surface['layer'];points=surface['points'];faces=surface['faces'];normals=surface_normals(points,faces)
 if parent_weights86.shape!=(len(parent['points']),86) or len(binding_bodies)!=86 or len(set(binding_bodies))!=86:
  raise ValueError('parent full86 table or binding-body lane map does not match exact 86-body domain')
 rows=[];registrations=[];weight_rows=[];weight_owner_rows=[];distances=[];tie_total=0;frame_owners=set();geometric_donors=set();weight_donor_mismatch=0;max_weight_sum_error=0.0;exact_fallbacks=[]
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
  frame=frames[frame_owner];R=frame['currentR'];origin=frame['currentOrigin'];world_to_local=frame.get('currentWorldToLocalR',R.T);local=world_to_local@(point-origin)
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
   for j in range(3):row[f'm{i}{j}']=f64(world_to_local[i,j])
  rows.append(row);distances.append(reg['distance']);tie_total+=reg['tieCount'];frame_owners.add(frame_owner);geometric_donors.add(geometric_donor)
  weight_rows.append({'vertex':vertex,**{f'w{i}':f64(weights[i]) for i in range(86)}})
  weight_owner_rows.append({'vertex':vertex,'dominant_binding_index':dominant_binding,'dominant_binding_body':frame_owner,
   'parent_geometric_donor_body':geometric_donor,'parent_tet_row':reg['tet'],
   **{f'parent_bary{i}':f64(reg['bary'][i]) for i in range(4)},'parent_anatomical_region_id':reg['region']})
  if 'exactTieFallback' in reg:
   fallback=dict(reg['exactTieFallback']);fallback['layer']=layer;fallback['surfaceVertexRow']=vertex
   exact_fallbacks.append(fallback)
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
 result={'nodes':rows,'faces':face_rows,'registrations':registrations,'weightRows':weight_rows,
  'weightOwnerRows':weight_owner_rows,'sourceDomain':regdomain,
  'metrics':{'nodes':len(rows),'faces':len(face_rows),'maximumParentRegistrationDistanceM':max(distances),
   'p95ParentRegistrationDistanceM':float(np.quantile(distances,.95)),'parentRegistrationTieCountTotal':tie_total,
   'selectedFrameOwnerBodies':sorted(frame_owners),'selectedGeometricDonorBodies':sorted(geometric_donors),
   'materialIndex':MATERIAL[layer],
   'maximumInterpolatedFull86NormalizationError':max_weight_sum_error,
   'full86DominantBindingDiffersFromParentGeometricDonorRows':weight_donor_mismatch,
   'allSourceNodeIDsDirectCandidateRowIdentity':True}}
 if exact_fallbacks:
  result['exactNearTieFallbacks']=exact_fallbacks
  result['metrics']['exactNearTieFallbackCount']=len(exact_fallbacks)
 return result

def frame_csv_rows(frames:dict[int,dict[str,np.ndarray]],binding_bodies:list[int])->list[dict[str,Any]]:
 out=[]
 for binding,donor in enumerate(binding_bodies):
  donor=int(donor);frame=frames[donor];x={'binding':binding,'donor':donor}
  for state in ('current','rest'):
   origin=frame[state+'Origin'];R=frame[state+'R']
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
  outputs={};regrows=[];metrics={};exact_fallback_reports={}
  for layer in ('O','I','D'):
   result=register_layer(surfaces[layer],parent,index,
       material_indices,frames,parent_weights86,binding_bodies,
       args.maximum_parent_distance_m,args.tie_tolerance_m)
   outputs[layer]=result;metrics[layer]=result['metrics']
   if result.get('exactNearTieFallbacks'):
    report=staged/f'exact-parent-near-tie-fallbacks-{layer}.json'
    report.write_text(json.dumps({'schema':'exact-parent-near-tie-fallbacks.v1','layer':layer,
      'candidatePath':surfaces[layer]['path'],'candidateSHA256':surfaces[layer]['sha256'],
      'parentCurrentSHA256':parentmeta['nodeSHA256'],'poseSnapshotSHA256':posemeta['sha256'],
      'fallbacks':result['exactNearTieFallbacks'],'physical_admission':False,'native_steps':0},indent=2,sort_keys=True)+'\n')
    exact_fallback_reports[layer]={'path':report.name,'sha256':sha(report),'count':len(result['exactNearTieFallbacks'])}
   write_csv(staged/f'registered-{layer}.csv',NODE_HEADER,result['nodes'])
   write_csv(staged/f'registered-{layer}-faces.csv',FACE_HEADER,result['faces'])
   write_csv(staged/f'parent-tet-full-weights-{layer}.csv',WEIGHT_HEADER,result['weightRows'])
   write_csv(staged/f'parent-tet-full-weight-owners-{layer}.csv',WEIGHT_OWNER_HEADER,result['weightOwnerRows'])
   regrows.extend(result['registrations'])
  write_csv(staged/'node-parent-registration.csv',PARENT_HEADER,regrows)
  write_csv(staged/'donor-frames.csv',FRAME_HEADER,frame_csv_rows(frames,binding_bodies))
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
  if exact_fallback_reports:
   receipt['exactParentNearTieFallbackReports']=exact_fallback_reports
   receipt['exactParentNearTieFallbackPolicy']='Fallback runs only after the existing floating tie guard rejects distinct projections. Exact binary64 rationals select the global nearest projection using a complete same-material exact-AABB candidate certificate. Exact equal-distance distinct projections still reject.'
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

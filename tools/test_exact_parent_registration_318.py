#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json, hashlib, tempfile, time
from fractions import Fraction
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('authoring230_exact318',HERE/'export_authoring_registered_surface_sidecars_230.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def check(ok,message):
 if not ok:raise AssertionError(message)

def make_parent(points,tets,materials=None):
 points=np.asarray(points,dtype=np.float64);tets=np.asarray(tets,dtype=np.int64)
 n=len(tets);materials=np.asarray(materials if materials is not None else [15]*n,dtype=np.int64)
 return {'points':points,'tets':tets,'materials':materials,'donors':np.arange(n,dtype=np.int64)+40,
         'regions':np.arange(n,dtype=np.int64)+70,'faces':np.arange(n,dtype=np.int64)+1000}

def exact_result(parent,point,seed=0,material=15):
 mi=m.build_material_indices(parent)[material]
 return m._exact_nearest_same_material(parent,mi,material,np.asarray(point,dtype=np.float64),seed)

def test_exact_tetra_feature_regions():
 vertices=tuple(tuple(m._fraction(x) for x in row) for row in
  ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
 cases=[('interior',(0.2,0.2,0.2),(0.2,0.2,0.2),4),
        ('face',(0.25,0.25,-0.1),(0.25,0.25,0.),3),
        ('edge',(0.5,-0.1,-0.1),(0.5,0.,0.),2),
        ('vertex',(-0.1,-0.1,-0.1),(0.,0.,0.),1)]
 for name,p,expected,positive_count in cases:
  q,bary,d2=m._closest_tetra_exact(vertices,tuple(m._fraction(x) for x in p))
  check(q==tuple(m._fraction(x) for x in expected),name+' exact projection')
  check(sum(x>0 for x in bary)==positive_count,name+' exact active simplex dimension')
  if name=='interior':check(d2==0,'interior point has exact zero distance')
  else:check(d2>0,name+' outside point has positive exact distance')

def test_shared_edge():
 # Two positive tets share only edge AB. A query below both z>=0 wedges
 # projects exactly onto the shared edge, so equal-distance ties have one point.
 A=(0.,0.,0.);B=(1.,0.,0.);C1=(0.,1.,0.);D1=(0.,0.,1.)
 C2=(0.,-1.,0.);D2=(1.,0.,1.)
 p=make_parent([A,B,C1,D1,C2,D2],[[0,1,2,3],[0,4,1,5]])
 r=exact_result(p,(0.5,0.0,-0.1),0)
 check(r['row']==0 and r['ties']==[0,1],'shared-edge projection chooses lowest stable row')
 check(r['diagnostic']['sameProjectionExactTieCount']==1,'shared-edge exact tie count')
 check(r['q']==(m._fraction(.5),m._fraction(0),m._fraction(0)),'shared-edge exact projection')

def test_coincident_tetrahedra():
 tet=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]
 p=make_parent(tet+tet,[[0,1,2,3],[4,5,6,7]])
 r=exact_result(p,(0.2,0.2,-0.1),1)
 check(r['row']==0 and r['ties']==[0,1],'coincident tetrahedra choose lowest stable row')
 check(r['diagnostic']['sameProjectionExactTieCount']==1,'coincident tetra exact tie count')

def test_distinct_equal_distance_rejected():
 pts=[(-1.,0.,0.),(-2.,0.,0.),(-1.,1.,0.),(-1.,0.,-1.),
      (1.,0.,0.),(2.,0.,0.),(1.,0.,-1.),(1.,1.,0.)]
 p=make_parent(pts,[[0,1,2,3],[4,5,6,7]])
 index=m.native.build_index(p);material=m.build_material_indices(p)
 rejected=False
 try:m.closest_same_material(p,index,material,15,np.zeros(3),2.0,1.0e-8)
 except m.ExactProjectionTieError as e:
  rejected=True
  check(e.diagnostic['status']=='rejected_exact_equal_distance_distinct_projections','distinct equal-distance witness status')
  check(e.diagnostic['exactNearestDistinctProjectionCount']==2,'distinct equal-distance projections remain separate')
 check(rejected,'truly distinct exact-equal nearest surfaces reject through the public routine')

def test_native_matrix_frames():
 f32=lambda x:float(np.float32(x))
 native_current=[f32(1.0000001),0.,0.,0.,1.,0.,0.,0.,1.]
 ident=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
 pos=[0.100000001,0.2,0.3];quat=[0.,0.,0.,1.]
 body={'body_index':2,'current':{'position_world_m':pos,'orientation_world_xyzw':quat},
       'rest':{'position_world_m':[0.4,0.5,0.6],'orientation_world_xyzw':quat}}
 all_rows=[{'body_index':i} for i in range(3)]
 all_rows[2]={'body_index':2,'current':{'position_world_m':pos,'orientation_world_xyzw':quat,
  'local_to_world_row_major_f32':native_current},'rest':{'position_world_m':[0.4,0.5,0.6],
  'orientation_world_xyzw':quat,'local_to_world_row_major_f32':ident}}
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'poses.json';path.write_text(json.dumps({'binding_count':1,'bodies':[body],'all_bodies':all_rows}))
  frames,meta,bindings=m.load_frames(path)
  check(meta['nativeMatrixFrameBodyCount']==1 and bindings==[2],'native binding to all_bodies identity')
  frame=frames[2];R=np.asarray(native_current,dtype=np.float32).astype(np.float64).reshape(3,3)
  check(np.array_equal(frame['currentR'],R),'captured native F32 matrix remains unchanged')
  check(np.array_equal(frame['currentWorldToLocalR'],np.linalg.inv(R)),'native inverse uses full inverse')
  check(meta['nativeMatrixSO3Bound']==8.0*float(np.finfo(np.float32).eps),'native matrix SO3 bound is the trusted 8-epsilon owner limit')
  check(not np.array_equal(frame['currentWorldToLocalR'],R.T),'native near-SO3 inverse is not approximated by transpose')
  check(np.array_equal(frame['currentOrigin'],np.asarray(pos,dtype=np.float32).astype(np.float64)),'native origin is cast to its emitted F32 value')
  frame_row=m.frame_csv_rows(frames,bindings)[0]
  expected_frame_header=['binding','donor',*[f'current_{a}' for a in 'xyz'],
   *[f'current_R{i}{j}' for i in range(3) for j in range(3)],
   *[f'rest_{a}' for a in 'xyz'],
   *[f'rest_R{i}{j}' for i in range(3) for j in range(3)]]
  check(m.FRAME_HEADER==expected_frame_header,
   'donor frame CSV follows native current-position/current-matrix/rest-position/rest-matrix column order')
  check(float(frame_row['current_R00'])==R[0,0],'frame CSV preserves the captured matrix')
  bad=json.loads(path.read_text());bad['all_bodies'][2]['body_index']=1;path.write_text(json.dumps(bad))
  try:m.load_frames(path);raise AssertionError('all_bodies identity mismatch accepted')
  except ValueError as e:check('identity mismatch' in str(e),'wrong all_bodies index rejected')
  bad=json.loads(path.read_text());bad['all_bodies'][2]['body_index']=2;bad['bodies'][0]['current']['orientation_world_xyzw']=[0.,0.,0.1,0.995];path.write_text(json.dumps(bad))
  try:m.load_frames(path);raise AssertionError('binding quaternion mismatch accepted')
  except ValueError as e:check('differs from registered binding' in str(e),'binding quaternion mismatch rejected')
  bad=json.loads(path.read_text());bad['bodies'][0]['current']['orientation_world_xyzw']=[0.,0.,0.,1.];bad['all_bodies'][2]['current']['local_to_world_row_major_f32'][0]=1.00000001;path.write_text(json.dumps(bad))
  try:m.load_frames(path);raise AssertionError('non-F32 matrix accepted')
  except ValueError as e:check('not exact Float32' in str(e),'non-F32 matrix rejected')
  bad=json.loads(path.read_text());bad['all_bodies'][2]['current']['local_to_world_row_major_f32'][0]=float(np.float32(1.0001));path.write_text(json.dumps(bad))
  try:m.load_frames(path);raise AssertionError('outside-SO3 matrix accepted')
  except ValueError as e:check('SO(3)' in str(e),'outside SO3 bound rejected')
  bad=json.loads(path.read_text());bad['all_bodies'][2]['current']['local_to_world_row_major_f32'][0]=native_current[0];del bad['all_bodies'][2]['rest']['local_to_world_row_major_f32'];path.write_text(json.dumps(bad))
  try:m.load_frames(path);raise AssertionError('partial native matrices accepted')
  except ValueError as e:check('incomplete native' in str(e),'partial matrices rejected')
  legacy={'binding_count':1,'bodies':[body]};path.write_text(json.dumps(legacy))
  frames,meta,_=m.load_frames(path)
  check('nativeMatrixFrameBodyCount' not in meta and 'frameSourceMethod' not in meta,'legacy compatibility route makes no native-frame admission claim')
  check(np.array_equal(frames[2]['currentWorldToLocalR'],frames[2]['currentR'].T),'legacy transpose compatibility preserved')

def test_real_near_tie_witnesses():
 P=HERE.parent
 A=P/'native-parent-layer-capture-256/run-001'
 candidate=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/O-exact-witness-repair-286/candidate.npz'
 if not candidate.is_file() or not (A/'compiled-parent-cell-mass.csv').is_file():
  return {'status':'actual_witness_assets_unavailable','tested':0}
 z=np.load(candidate,allow_pickle=False);points=np.asarray(z['points_world_m'])
 parent,index,mi,parentmeta=m.load_parent_current(A/'compiled-parent-cell-mass.csv')
 for material,material_index in mi.items():
  rows=np.flatnonzero(parent['materials']==material)
  tetpoints=parent['points'][parent['tets'][rows]]
  check(np.array_equal(material_index['boundsMin'],tetpoints.min(axis=1)),
        f'material-{material} AABB minima are exact bounds of pinned parent vertices')
  check(np.array_equal(material_index['boundsMax'],tetpoints.max(axis=1)),
        f'material-{material} AABB maxima are exact bounds of pinned parent vertices')
 report={'status':'focused_exact_rational_near_tie_witnesses','candidatePath':str(candidate),
  'candidateSHA256':m.sha(candidate),'parent':parentmeta,'fallbacks':[],'physical_admission':False,'native_steps':0}
 for vertex in (120,756):
  point=np.asarray(points[vertex],dtype=np.float64)
  started=time.perf_counter();result=m.closest_same_material(parent,index,mi,15,point,.02,1.0e-8);elapsed=time.perf_counter()-started
  check('exactTieFallback' in result,f'actual O vertex {vertex} did not use bounded exact fallback')
  d=result['exactTieFallback']
  check(d['exactFallbackElapsedSeconds']>=0.0,f'actual O vertex {vertex} exact fallback timing recorded')
  check(d['status']=='accepted_exact_nearest_projection',f'actual O vertex {vertex} exact nearest did not resolve uniquely')
  check(d['exactNearestDistinctProjectionCount']==1,f'actual O vertex {vertex} exact nearest projection not unique')
  ranked=d['rankedDistinctProjections']
  high=Fraction(*map(int,ranked[1]['distanceSquaredM2Exact'].split('/')))
  low=Fraction(*map(int,ranked[0]['distanceSquaredM2Exact'].split('/')))
  check(d['nextDistinctProjectionDistanceGapMDecimal80']==m._sqrt_fraction_gap_decimal_text(high,low),
        f'actual O vertex {vertex} distance gap is computed at the reported 80-digit precision')
  check(result['tieCount']==0 or len(result['ties'])>1,'exact tie rows and count disagree')
  report['fallbacks'].append({'vertex':vertex,'selectedParentTetRow':result['tet'],
   'distanceM':result['distance'],'distanceSquaredM2Exact':d['nearestDistanceSquaredM2Exact'],
   'exactMinimumRows':d['exactMinimizerParentTetRows'],'sameProjectionExactTieCount':d['sameProjectionExactTieCount'],
   'exactAABBCandidateCount':d['exactAABBCandidateCount'],
   'rankedDistinctProjections':d['rankedDistinctProjections'],
   'nextDistinctProjectionDistanceSquaredGapM2Exact':d['nextDistinctProjectionDistanceSquaredGapM2Exact'],
   'pointWorldM':point.tolist(),'exactFallbackElapsedSeconds':elapsed})
 base=HERE/'exact-tie-witness-318';run=1
 while (base/f'run-{run:03d}').exists():run+=1
 out=base/f'run-{run:03d}';out.mkdir(parents=True,exist_ok=False)
 (out/'actual-witness-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 report['reportPath']=str(out/'actual-witness-report.json')
 report['reportSHA256']=hashlib.sha256((out/'actual-witness-report.json').read_bytes()).hexdigest()
 print(json.dumps({'actual_witness_report':report['reportPath'],'sha256':report['reportSHA256'],
  'selected':[(x['vertex'],x['selectedParentTetRow'],x['distanceM'],x['nextDistinctProjectionDistanceSquaredGapM2Exact']) for x in report['fallbacks']]},indent=2),flush=True)
 return report

def test_actual_native_matrix_capture():
 path=HERE.parent/'P/physical-volume-producer-311/native-pose-matrix-capture-001/registered-skin-poses-100.json'
 if not path.is_file():return {'status':'native_matrix_capture_unavailable'}
 frames,meta,bindings=m.load_frames(path)
 check(len(bindings)==86 and meta['nativeMatrixFrameBodyCount']==86,'real all-body native matrices are selected')
 check(meta['nativeMatrixOrthogonalityErrorMax']<=meta['nativeMatrixSO3Bound'],'captured native matrix SO3 bound')
 check(meta['nativeMatrixDeterminantErrorMax']<=meta['nativeMatrixSO3Bound'],'captured native matrix determinant bound')
 return {'status':'native_matrix_capture_pass','bodyCount':len(bindings),'poseSHA256':meta['sha256'],
  'frameSourceMethod':meta['frameSourceMethod'],'orthogonalityErrorMax':meta['nativeMatrixOrthogonalityErrorMax'],
  'determinantErrorMax':meta['nativeMatrixDeterminantErrorMax'],'bound':meta['nativeMatrixSO3Bound']}

def main():
 test_exact_tetra_feature_regions();test_shared_edge();test_coincident_tetrahedra();test_distinct_equal_distance_rejected();test_native_matrix_frames()
 actual=test_real_near_tie_witnesses();matrix=test_actual_native_matrix_capture()
 print(json.dumps({'exact_tetra_interior_face_edge_vertex':'PASS','shared_edge_tie':'PASS','coincident_tet_tie':'PASS','distinct_equal_distance_reject':'PASS',
  'native_matrix_positive_and_negative_cases':'PASS','actual_witnesses':actual,'native_capture':matrix},indent=2),flush=True)
if __name__=='__main__':main()

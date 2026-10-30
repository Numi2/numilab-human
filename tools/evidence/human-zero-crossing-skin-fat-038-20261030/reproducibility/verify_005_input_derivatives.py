import hashlib,json
from pathlib import Path
import numpy as np
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=B/'quality-warning-connected-union-038'
OUT=B/'quality-warning-connected-union-038-typed-metadata-005-001/input-derivative-verification.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
checks=[]
for role,source,restored,receipt,receipt_sha in [
 ('I',R/'run-005/I-quality-warning-union-proposal.npz',R/'run005-source-precision-restoration-001/I-source-precision-restored.npz',R/'run005-source-precision-restoration-001/receipt.json',''),
 ('D',R/'run-005/D-quality-warning-union-proposal.npz',R/'run005-source-dtype-restoration-002/D-source-dtype-restored.npz',R/'run005-source-dtype-restoration-002/receipt.json','')]:
 s=load(source);d=load(restored);r=json.loads(receipt.read_text())
 assert r['source']['sha256']==sha(source) and r['output']['sha256']==sha(restored)
 assert set(s)==set(d)
 for k in s:
  if k=='points_world_m':continue
  assert s[k].dtype==d[k].dtype and s[k].shape==d[k].shape and s[k].tobytes()==d[k].tobytes(),(role,k)
 if role=='I':
  assert s['points_world_m'].dtype==d['points_world_m'].dtype==np.float64
  assert s['points_world_m'].tobytes()==d['points_world_m'].tobytes()
  assert r['status']=='RESTORED_EXACT_RETAINED_SOURCE_F64_WITH_IDENTICAL_NATIVE_F32'
 else:
  assert s['points_world_m'].dtype==np.float64 and d['points_world_m'].dtype==np.float32
  assert s['points_world_m'].astype(np.float32).tobytes()==d['points_world_m'].tobytes()
  assert r['status']=='RESTORED_EXACT_RETAINED_SOURCE_DTYPE_WITH_IDENTICAL_COORDINATES'
  assert r['numeric_coordinate_values_unchanged'] and r['native_f32_coordinates_bitwise_unchanged']
  assert r['faces_maps_other_arrays_unchanged']
 checks.append({'role':role,'source_path':str(source),'source_sha256':sha(source),'source_points_dtype':s['points_world_m'].dtype.name,'restored_path':str(restored),'restored_sha256':sha(restored),'restored_points_dtype':d['points_world_m'].dtype.name,'restoration_receipt_path':str(receipt),'restoration_receipt_sha256':sha(receipt),'array_members_equal_except_declared_points_dtype':True,'f32_surface_view_bitwise_equal':s['points_world_m_f32'].tobytes()==d['points_world_m_f32'].tobytes()})
 if role=='D':assert checks[-1]['f32_surface_view_bitwise_equal']
OUT.write_text(json.dumps({'schema':'quality-warning-connected-union-005-input-derivative-verification.v1','status':'PASS_I_SOURCE_F64_AND_D_SOURCE_F32_DERIVATIVE_EQUIVALENCE','roles':checks,'scope':'Read-only array equivalence and source-precision restoration verification; no geometry predicate or admission claim.'},sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS_INPUT_DERIVATIVE_VERIFICATION','path':str(OUT),'sha256':sha(OUT),'roles':checks},sort_keys=True))

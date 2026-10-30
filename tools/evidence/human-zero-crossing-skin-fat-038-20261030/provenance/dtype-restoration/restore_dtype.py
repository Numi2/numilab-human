from pathlib import Path
import numpy as np,json,hashlib,shutil
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
out=H/'quality-warning-connected-union-038/run005-source-dtype-restoration-002';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src=H/'quality-warning-connected-union-038/run-005/D-quality-warning-union-proposal.npz';original=H/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz'
assert sha(original)=='15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f'
with np.load(src,allow_pickle=False)as z:d={k:z[k]for k in z.files}
with np.load(original,allow_pickle=False)as z:p=z['points_world_m']
restored=p[d['point_output_to_input_rows']].copy()
assert restored.dtype==np.float32 and d['points_world_m'].dtype==np.float64
assert np.array_equal(restored.astype(np.float64),d['points_world_m'])
assert np.array_equal(restored.view(np.uint32),d['points_world_m_f32'].view(np.uint32))
d['points_world_m']=restored;dst=out/'D-source-dtype-restored.npz';np.savez_compressed(dst,**d)
r={'status':'RESTORED_EXACT_RETAINED_SOURCE_DTYPE_WITH_IDENTICAL_COORDINATES','source':{'path':str(src),'sha256':sha(src)},'exact_original_source':{'path':str(original),'sha256':sha(original)},'output':{'path':str(dst),'sha256':sha(dst)},'changed_array_only':'points_world_m','source_dtype':'float32','proposal_dtype':'float64','native_f32_coordinates_bitwise_unchanged':True,'numeric_coordinate_values_unchanged':True,'faces_maps_other_arrays_unchanged':True,'physical_admission':False,'runtime':0,'accepted_steps':0}
(out/'receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
shutil.copy2('/tmp/numi-restore-I005-precision.py',H/'quality-warning-connected-union-038/run005-source-precision-restoration-001/restore_precision.py')
print(json.dumps(r))


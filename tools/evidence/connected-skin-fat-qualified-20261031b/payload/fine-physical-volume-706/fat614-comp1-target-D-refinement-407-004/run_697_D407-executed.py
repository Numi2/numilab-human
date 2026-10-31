#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TYPED=H/'fat614-comp1-target-D-refinement-407-004/candidate-D-comp1-current-target-refinement-typed.npz'
HELPER=H/'current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py'
OUT=H/'fat614-comp1-target-D-refinement-407-004/actual-697-adapter-001'
CANDIDATE_SHA='bf0cef3ab74566428cd46f38716233745b0e2dfb21d7d50d8cfe7b2790b82c38'
HELPER_SHA='83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
if sha(TYPED)!=CANDIDATE_SHA: raise ValueError('typed D candidate pin mismatch')
if sha(HELPER)!=HELPER_SHA: raise ValueError('697 helper pin mismatch')
OUT.mkdir(parents=False,exist_ok=False)
with np.load(TYPED,allow_pickle=False) as z:
 src={k:np.array(z[k],copy=True) for k in z.files}
N=len(src['faces'])
schema_value=np.asarray(src['face_source_identity_schema_by_original_id_json'])
if schema_value.size!=1: raise ValueError('identity schema is not scalar')
schema=json.loads(str(schema_value.item()))
source_ids=set(map(str,np.unique(src['face_source_original_id'])))
missing=source_ids-set(schema)
if missing: raise ValueError(f'candidate identity registry lacks IDs: {sorted(missing)[:10]}')
fields=('face_source_layer','face_source_body','face_source_bone','face_source_original_id')
signatures=np.stack([np.asarray(src[k],np.int64) for k in fields],axis=1)
unique,group=np.unique(signatures,axis=0,return_inverse=True)
keys=[dict(zip(fields,map(int,row))) for row in unique]
counts=np.bincount(group,minlength=len(keys)).astype(np.int64)
offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
rows=np.argsort(group,kind='stable').astype(np.int64)
ancestry_offsets=np.asarray(src['face_ancestry_offsets'],np.int64)
if ancestry_offsets.shape!=(N+1,) or ancestry_offsets[0]!=0 or np.any(np.diff(ancestry_offsets)<=0):
 raise ValueError('full ancestry offsets invalid or empty')
ancestry_lanes=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
for lane in ancestry_lanes:
 if lane not in src: raise ValueError('typed candidate missing ancestry '+lane)
payload=dict(src)
payload.update({
 'face_source_identity_schema_by_original_id_json':schema_value,
 'face_group_id':group.astype(np.int64),
 'source_group_offsets':offsets,
 'source_group_face_rows':rows,
 'source_group_keys_json':np.asarray([json.dumps(keys,separators=(',',':'))]),
 'face_parent_source_face_rows':np.arange(N,dtype=np.int64),
 'face_exact_source_row':np.full(N,-1,np.int64),
 'lineage_preparation_input_sha256':np.asarray([CANDIDATE_SHA]),
 'face_ancestry_method':np.asarray(['Adapter-local row grouping by current declared typed primary identity. Full flattened source-face ancestry is preserved; no exact per-face identity claim is made.']),
})
adapter=OUT/'candidate-D-comp1-typed-697-prepared.npz'
np.savez_compressed(adapter,**payload)
with np.load(adapter,allow_pickle=False) as z:
 if not np.array_equal(z['faces'],src['faces']) or not np.array_equal(z['points_world_m_f32'],src['points_world_m_f32']):
  raise ValueError('adapter geometry changed')
 if not np.array_equal(z['face_source_identity_schema_by_original_id_json'],schema_value):
  raise ValueError('adapter schema changed')
spec=importlib.util.spec_from_file_location('handoff697_D407',HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);owner=mod.load_owner()
t=time.monotonic();loaded=mod.load_surface_217(owner,adapter,'D',None)
if not np.array_equal(np.asarray(loaded['points'],np.float32),src['points_world_m_f32']): raise ValueError('697 point bits changed')
if not np.array_equal(np.asarray(loaded['faces'],np.int64),src['faces']): raise ValueError('697 faces changed')
for i,(lo,hi) in enumerate(zip(ancestry_offsets[:-1],ancestry_offsets[1:])):
 for lane,gotkey in zip(ancestry_lanes,('ancestry','ancestry_original_ids','ancestry_parent_rows')):
  if loaded[gotkey][i]!=src[lane][lo:hi].tolist(): raise ValueError(f'697 ancestry mismatch face {i}, lane {lane}')
report={
 'schema':'h706.comp1-target-D-refinement-actual-697-load.v1',
 'status':'PASS_ACTUAL_697_217_TYPED_D_LOAD',
 'candidate_path':str(TYPED),'candidate_sha256':sha(TYPED),
 'adapter_path':str(adapter),'adapter_sha256':sha(adapter),
 'identity_registry_source':'candidate embedded complete registry with explicit DERIVED_PATCH addition',
 'helper_path':str(HELPER),'helper_sha256':sha(HELPER),
 'owner_build_path':str(mod.BUILD),'owner_build_sha256':sha(mod.BUILD),'owner_build_expected_sha256':mod.BUILD_SHA,
 'points':int(len(loaded['points'])),'faces':int(len(loaded['faces'])),
 'ancestry_tuple_count':int(sum(map(len,loaded['ancestry']))),'group_count':int(len(keys)),
 'used_point_count':int(len(np.unique(loaded['faces']))),'elapsed_s':time.monotonic()-t,
 'physical_admission':False,'native_steps':0,'tetgen_run':False,'mass_mapping':'NOT_RUN',
 'limitations':['Existing 697/217 typed D loader acceptance only. This is not volume transfer, mass mapping, native execution or physical qualification.']
}
receipt=OUT/'actual-697-D-load-receipt.json'
receipt.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'receipt':str(receipt),'sha256':sha(receipt),'adapter':str(adapter),'adapter_sha256':sha(adapter),'points':report['points'],'faces':report['faces'],'ancestry_tuples':report['ancestry_tuple_count'],'groups':report['group_count'],'elapsed_s':round(report['elapsed_s'],3)},sort_keys=True))

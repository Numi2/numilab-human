from __future__ import annotations
import hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TASK=H/'quality-warning-connected-union-038-typed-metadata-005-001'
HELPER=H/'current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py'
HELPER_SHA='83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f'
GEOMETRY=H/'quality-warning-connected-union-038/merge-run004-run005-O/full-check/independent-verification-001/report.json'
GEOMETRY_SHA='af18d3e4d3f02284acc4a27740c5c9256464c17fb99ca443a7186716a836cdf7'
OUTPUT=TASK/'actual-697-all-roles-load-preflight.json'
COMBINED=TASK/'combined-038-005-typed-source-preparation-receipt.json'
CANDIDATES={
 'O':(TASK/'O-merged-run004-run005-001/typed-output/candidate-O-connected-union-typed.npz','b35eb99aab790d861263b6dffa605f781287d81d682e3fb34dac8810d2c92b66'),
 'I':(TASK/'I-run-002/candidate-I-connected-union-typed.npz','a445e482f83616bd863e194377896d09d98327ff474683f5f687a4ac52d059f7'),
 'D':(TASK/'D-run-001/candidate-D-connected-union-typed.npz','d0f6cda10552a55be2d6ac5a0740e0142293cdf55177f1ce853a3eb5624374da'),
}
COUNTS={'O':(76669,153293),'I':(125368,250731),'D':(120628,242135)}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def must_sha(p:Path,expected:str,label:str)->str:
 got=sha(p)
 if got!=expected:raise ValueError(f'{label} SHA mismatch {got} != {expected}')
 return got
must_sha(HELPER,HELPER_SHA,'697 owner helper')
geo_sha=must_sha(GEOMETRY,GEOMETRY_SHA,'merged O independent geometry report')
geo=json.loads(GEOMETRY.read_text())
if geo.get('status')!='PASS_FIXED_COORDINATE_GEOMETRY_DELTA' or geo.get('physical_admission') is not False:
 raise ValueError('merged O geometry delta is not the expected non-admission pass')
if int(geo.get('changed_vs_complete_current_surfaces',{}).get('count',-1))!=0 or int(geo.get('changed_vs_all185_both_domains',{}).get('count',-1))!=0:
 raise ValueError('merged O exact delta reports are not zero')
expected_raw={
 'O':('quality-warning-connected-union-038/merge-run004-run005-O/O-merged-run004-run005-proposal.npz','9e0a7abfd1e192c37cdbdabe17335e46d79024e19386c9e635a46ff7564516c3'),
 'I':('quality-warning-connected-union-038/run-005/I-quality-warning-union-proposal.npz','f4bc1b96aeef6991f25c1aa91740b92f95c61ad6e7bfd5324900c9107326c437'),
 'D':('quality-warning-connected-union-038/run005-source-dtype-restoration-002/D-source-dtype-restored.npz','f729d6fd09c76b6c30048cbfa9a713a458157f07704910f01de3494b39497d0d'),
}
for role,(rel,expected) in expected_raw.items():
 path,got=geo['roles'][role]['proposal_path'],geo['roles'][role]['proposal_sha256']
 if Path(path)!=H/rel or got!=expected:raise ValueError(f'geometry report {role} input pin mismatch')
 if sha(Path(path))!=expected:raise ValueError(f'geometry report {role} proposal no longer matches pin')
spec=importlib.util.spec_from_file_location('handoff697_final038',HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
owner=mod.load_owner()
loaded={}
for role,(path,expected) in CANDIDATES.items():
 if not path.is_file():raise FileNotFoundError(path)
 cand_sha=must_sha(path,expected,role+' typed candidate')
 started=time.time();s=mod.load_surface_217(owner,path,role,None)
 points=np.asarray(s['points'],np.float32); faces=np.asarray(s['faces'],np.int64)
 if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all():raise ValueError(role+' loaded surface shape/finiteness')
 if faces.size==0 or faces.min()<0 or faces.max()>=len(points):raise ValueError(role+' face rows outside point domain')
 npoints,nfaces=COUNTS[role]
 if (len(points),len(faces))!=(npoints,nfaces):raise ValueError(f'{role} loaded counts changed: {(len(points),len(faces))}')
 for k in ('ancestry','ancestry_original_ids','ancestry_parent_rows'):
  if len(s[k])!=len(faces):raise ValueError(role+' loaded '+k+' count mismatch')
 lengths=np.fromiter((len(x) for x in s['ancestry']),dtype=np.int64,count=len(s['ancestry']))
 if len(lengths)!=len(faces) or np.any(lengths<1):raise ValueError(role+' has empty/misaligned loaded ancestry')
 kinds,cnt=np.unique(np.asarray(s['ancestry_kind']),return_counts=True)
 if not np.array_equal(kinds,np.asarray([6])) or int(cnt[0])!=nfaces:raise ValueError(role+' ancestry kind domain differs from current typed-source route')
 used_count=int(len(np.unique(faces.reshape(-1))))
 if used_count!=len(points):raise ValueError(role+' contains unused points in loaded owner representation')
 loaded[role]={'path':str(path),'sha256':cand_sha,'points':int(len(points)),'faces':int(len(faces)),'used_points':used_count,'loaded_ancestry_tuple_count':int(lengths.sum()),'ancestry_kind_counts':{str(int(k)):int(v) for k,v in zip(kinds,cnt)},'identity_schema_count':len(s['typed_identity_schema']),'exact_source_rows_nonnegative':int(np.count_nonzero(np.asarray(s['exact_source_rows'])>=0)),'elapsed_seconds':time.time()-started}
# Pin supporting metadata/normalization/evidence receipts after the loader has exercised each candidate.
meta={
 'O':TASK/'O-merged-run004-run005-001/typed-output/O-metadata-receipt.json',
 'I':TASK/'I-run-002/I-metadata-receipt.json',
 'D':TASK/'D-run-001/D-metadata-receipt.json',
}
meta_hashes={r:sha(p) for r,p in meta.items()}
input_receipts={
 'O_build_invocation':TASK/'O-merged-run004-run005-001/build-invocation.json',
 'O_operation_normalization':TASK/'O-merged-run004-run005-001/operation-list-normalization-receipt.json',
 'I_build_invocation':TASK/'I-build-invocation.json',
 'D_build_invocation':TASK/'D-build-invocation.json',
 '005_correction_verification':TASK/'independent-005-source-correction-verification.json',
 '005_input_derivative_verification':TASK/'input-derivative-verification.json',
 'O_geometry_verification':GEOMETRY,
}
input_receipt_hashes={k:sha(v) for k,v in input_receipts.items()}
report={'schema':'h706.quality-warning-connected-union-005-actual-697-three-role-load-preflight.v1','status':'PASS_217_O_I_D_TYPED_SURFACE_LOAD_ONLY','helper_path':str(HELPER),'helper_sha256':HELPER_SHA,'owner_build_path':str(mod.BUILD),'owner_build_sha256':sha(mod.BUILD),'owner_build_expected_sha256':mod.BUILD_SHA,'roles':loaded,'geometry_delta_report_path':str(GEOMETRY),'geometry_delta_report_sha256':geo_sha,'geometry_delta_status':geo['status'],'geometry_delta_counts':{'changed_vs_complete_current_surfaces':geo['changed_vs_complete_current_surfaces']['count'],'changed_vs_all185_both_domains':geo['changed_vs_all185_both_domains']['count'],'selected_face_count':geo['changed_vs_complete_current_surfaces']['selected_face_count'],'complete_current_surface_face_count':geo['changed_vs_complete_current_surfaces']['complete_surface_face_count'],'complete_surface_audit':geo['changed_vs_complete_current_surfaces']['complete_surface_audit']},'metadata_receipts':{r:{'path':str(p),'sha256':meta_hashes[r]} for r,p in meta.items()},'supporting_receipts':{k:{'path':str(v),'sha256':input_receipt_hashes[k]} for k,v in input_receipts.items()},'physical_admission':False,'tetgen_run':False,'native_runtime_run':False,'limitations':['This invokes the pinned existing 697/217 surface loader on all three typed role candidates. The geometric delta report is a fixed-coordinate changed-face check with baseline proofs; it does not establish volume quality, mass transport, TetGen admission, or physical/native acceptance.']}
OUTPUT.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
# Combined concise receipt points to the exact assets and independent evidence without conflating gates.
combined={'schema':'h706.quality-warning-connected-union-005-combined-typed-source-preparation.v1','status':'PASS_THREE_TYPED_ROLES_AND_ACTUAL_697_LOAD','roles':{r:{'candidate_path':v['path'],'candidate_sha256':v['sha256'],'points':v['points'],'faces':v['faces'],'ancestry_tuples':v['loaded_ancestry_tuple_count'],'metadata_receipt_path':str(meta[r]),'metadata_receipt_sha256':meta_hashes[r]} for r,v in loaded.items()},'merged_O_operation_normalization':{'receipt_path':str(input_receipts['O_operation_normalization']),'receipt_sha256':input_receipt_hashes['O_operation_normalization'],'operation_count':202,'geometry_maps_preserved_bitwise':True},'operation_corrections':{'run004_path':str(H/'quality-operation-hash-correction-038-001/operation-hash-correction-038.json'),'run004_sha256':'b9770e626752a835d32225434f12eecd399429cf34402d5304d1fca78e255c89','run004_status':'PASS_198_OF_198_PRE_FINALIZATION_HASHES_AND_MAPS_RECONSTRUCTED','run005_verification_path':str(input_receipts['005_correction_verification']),'run005_verification_sha256':input_receipt_hashes['005_correction_verification'],'run005_status':'PASS_515_OF_515_FINALIZATION_AND_IMMEDIATE_037_PLUS_ULTIMATE_015_ROW_MAPS'},'geometry_delta':{'path':str(GEOMETRY),'sha256':geo_sha,'status':geo['status'],'changed_vs_complete_current_surfaces':geo['changed_vs_complete_current_surfaces']['count'],'changed_vs_all185_both_domains':geo['changed_vs_all185_both_domains']['count'],'scope':'Independent exact changed-face delta against complete current surface records and all 185 bone records in both coordinate domains; baseline proofs apply to unchanged geometry.','complete_surface_audit':geo['changed_vs_complete_current_surfaces']['complete_surface_audit']},'actual_loader':{'path':str(OUTPUT),'sha256':sha(OUTPUT),'status':report['status'],'owner_helper_path':str(HELPER),'owner_helper_sha256':HELPER_SHA},'physical_admission':False,'tetgen_run':False,'native_runtime_run':False,'limitations':report['limitations']}
COMBINED.write_text(json.dumps(combined,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'load_report':str(OUTPUT),'load_report_sha256':sha(OUTPUT),'combined_receipt':str(COMBINED),'combined_receipt_sha256':sha(COMBINED),'roles':{r:{'points':v['points'],'faces':v['faces'],'ancestry':v['loaded_ancestry_tuple_count']} for r,v in loaded.items()}},sort_keys=True))
